import os
import copy
import re
import io
import logging
import json
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from .base_io_dataset import BaseIODataset
from .heatmap_utils import generate_heatmap, generate_future_waypoints, get_yaw_angle
from .det_utils import generate_det_data
from morai_data import load_split_manifest, split_entries
from morai_traffic_labels import load_traffic_label_sidecar, checked_traffic_target
try:
    from skimage.measure import block_reduce
except ImportError:
    def block_reduce(array, block_size, func=np.mean):
        """Dependency-free fallback for the 2-D mean reduction used below."""
        if func is not np.mean or array.ndim != 2:
            raise ImportError("install scikit-image for this block_reduce configuration")
        by, bx = block_size
        height = array.shape[0] // by * by
        width = array.shape[1] // bx * bx
        return array[:height, :width].reshape(
            height // by, by, width // bx, bx
        ).mean(axis=(1, 3))

_logger = logging.getLogger(__name__)


def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [ np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0,0,1,0],
        [0,0,0,1]
    ]

def lidar_to_raw_features(lidar, rotation_deg=-90, rng=None):
    def preprocess(lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)

        idx = np.argwhere(idx)

        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        else:
            return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

    lidar_xyzr = preprocess(lidar)

    idxs = np.arange(len(lidar_xyzr))
    (rng if rng is not None else np.random).shuffle(idxs)
    lidar_xyzr = lidar_xyzr[idxs]

    lidar = np.zeros((40000, 4), dtype=np.float32)
    num_points = min(40000, len(lidar_xyzr))
    lidar[:num_points,:4] = lidar_xyzr
    lidar[np.isinf(lidar)] = 0
    lidar[np.isnan(lidar)] = 0
    lidar[lidar>100] = 0
    lidar[lidar<-100] = 0
    lidar = rotate_lidar(lidar, rotation_deg).astype(np.float32)
    return lidar, num_points

def check_data(data, info):
    for key in data:
        if isinstance(data[key], np.ndarray):
            if np.isnan(data[key]).any():
                print(key)
                print(info)
                data[key][np.isnan(data[key])] = 0
            if np.isinf(data[key]).any():
                print(key)
                print(info)
                data[key][np.isinf(data[key])] = 0
        elif isinstance(data[key], torch.Tensor):
            if torch.isnan(data[key]).any():
                print(key)
                print(info)
                data[key][torch.isnan(data[key])] = 0
            if torch.isinf(data[key]).any():
                print(key)
                print(info)
                data[key][torch.isinf(data[key])] = 0
    return data

def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 4
        hist_max_per_pixel = 5
        x_meters_max = 35
        y_meters_max = 40
        y_meters_min = 30
        xbins = np.linspace(
            -1 * x_meters_max,
            1 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )
        ybins = np.linspace(-y_meters_max, y_meters_min, (y_meters_max + y_meters_min)* pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    total_features = below_features + above_features
    features = np.stack([below_features, above_features, total_features], axis=-1)
    features = np.rot90(features, 3)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]
    return out

class CarlaMVDetDataset(BaseIODataset):
    def __init__(
        self,
        root,
        towns,
        weathers,
        head="det",
        input_rgb_size=224,
        input_lidar_size=224,
        rgb_transform=None,
        depth_transform=None,
        seg_transform=None,
        lidar_transform=None,
        multi_view_transform=None,
        with_waypoints=False,
        with_seg=False,
        with_depth=False,
        with_lidar=False,
        multi_view=False,
        **kwargs,
    ):
        super().__init__()

        self.head = head
        self.input_lidar_size = input_lidar_size
        self.input_rgb_size = input_rgb_size
        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform
        self.lidar_transform = lidar_transform
        self.multi_view_transform = multi_view_transform

        self.with_waypoints = with_waypoints

        self.with_seg = with_seg
        self.with_depth = with_depth
        self.with_lidar = with_lidar
        self.multi_view = multi_view

        self.route_frames = self._get_frames_paths(root, weathers, towns)

    def _get_frames_paths(self, root, weathers, towns):
        route_frames = []
        route_dir_nums = 0
        index_path = os.path.join(root, 'dataset_index_test.txt')
        if not os.path.isfile(index_path):
            index_path = os.path.join(root, 'dataset_index.txt')
        dataset_indexs = self._load_text(index_path).split('\n')
        pattern = re.compile(r'town(\d\d).*w(\d+)')
        for line in dataset_indexs:
            if len(line.split()) != 2:
                continue
            path, frames = line.split()
            if path.startswith('data/'):
                path = path[len('data/'):]
            path = os.path.join(root, 'data', path)
            frames = int(frames)
            res = pattern.findall(path)
            if len(res) == 1:
                town = int(res[0][0])
                weather = int(res[0][1])
                if weather not in weathers or town not in towns:
                    continue
            elif not os.path.isfile(os.path.join(path, 'route.json')):
                continue
            route_dir_nums += 1
            for i in range(0, frames):
                if os.path.isfile(os.path.join(path, 'route.json')):
                    measurement_path = os.path.join(
                        path, 'measurements_full', '%04d.json' % i
                    )
                    if os.path.isfile(measurement_path):
                        with open(measurement_path) as measurement_file:
                            navigation = json.load(measurement_file).get('navigation', {})
                        if navigation.get('route_valid') is False:
                            continue
                route_frames.append((path, i, route_dir_nums))
        _logger.info("Sub route dir nums: %d" % len(route_frames))
        return route_frames

    def __len__(self):
        return int(len(self.route_frames) / 5)

    def __getitem__(self, idx):
        idx = idx * 5 + np.random.randint(0, 5)
        route_dir, frame_id, route_dir_id = self.route_frames[idx]
        return self._extract_data_item(route_dir, frame_id, route_dir_id)


    def _extract_data_item(self, route_dir, frame_id, route_dir_id, with_actor_infos=False, cached_measurements=None):
        data = {}
        # You can use tools/data/batch_merge_data.py to generate FULL image (including front, left, right) for reducing io cost
        morai_route = os.path.isfile(os.path.join(route_dir, "route.json"))
        rgb_full_path = os.path.join(route_dir, "rgb_full", "%04d.jpg" % frame_id)
        if os.path.isfile(rgb_full_path):
            rgb_full_image = self._load_image(rgb_full_path)
            rgb_image = rgb_full_image.crop((0, 0, 800, 600))
            rgb_left_image = rgb_full_image.crop((0, 600, 800, 1200))
            rgb_right_image = rgb_full_image.crop((0, 1200, 800, 1800))
            rgb_rear_image = rgb_full_image.crop((0, 1800, 800, 2400))
        else:
            rgb_image = self._load_image(os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id))
            rgb_left_image = self._load_image(os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id))
            rgb_right_image = self._load_image(os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id))
            rgb_rear_image = self._load_image(os.path.join(route_dir, "rgb_rear", "%04d.jpg" % frame_id))

        '''
        rgb_image = self._load_image(
            os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id)
        )
        rgb_left_image = self._load_image(
            os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id)
        )
        rgb_right_image = self._load_image(
            os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id)
        )
        '''

        if self.with_waypoints:
            waypoints_data = self._load_npy(os.path.join(route_dir, "waypoints.npy"))[
                frame_id
            ]
            data["waypoints"] = torch.from_numpy(waypoints_data)

        if self.with_seg:
            seg_image = self._load_image(
                os.path.join(route_dir, "seg_front", "%04d.jpg" % frame_id)
            )
            if self.multi_view:
                seg_left_image = self._load_image(
                    os.path.join(route_dir, "seg_left", "%04d.jpg" % frame_id)
                )
                seg_right_image = self._load_image(
                    os.path.join(route_dir, "seg_right", "%04d.jpg" % frame_id)
                )
        if self.with_depth:
            depth_image = self._load_image(
                os.path.join(route_dir, "depth_front", "%04d.jpg" % frame_id)
            )
            if self.multi_view:
                depth_left_image = self._load_image(
                    os.path.join(route_dir, "depth_left", "%04d.jpg" % frame_id)
                )
                depth_right_image = self._load_image(
                    os.path.join(route_dir, "depth_right", "%04d.jpg" % frame_id)
                )

        if cached_measurements:
            measurements = cached_measurements
        else:
            measurements_all = self._load_json(
                os.path.join(route_dir, "measurements_all.json")
            )
            measurements = self._load_json(
                os.path.join(route_dir, "measurements_full", "%04d.json" % frame_id)
            )
        actors_data = measurements["actors_data"]
        stop_sign = int(measurements["stop_sign"])
        '''
        # You can use tools/data/batch_merge_data.py to generate FULL measurements for reducing io cost
        measurements = self._load_json(
            os.path.join(route_dir, "measurements", "%04d.json" % frame_id)
        )
        actors_data = self._load_json(
            os.path.join(route_dir, "actors_data", "%04d.json" % frame_id)
        )
        affordances = self._load_npy(os.path.join(route_dir, 'affordances/%04d.npy' % frame_id))
        stop_sign = int(affordances.item()['stop_sign'])
        '''

        if measurements["is_junction"] is True:
            is_junction = 1
        else:
            is_junction = 0

        if len(measurements['is_red_light_present']) > 0:
            traffic_light_state = 0
        else:
            traffic_light_state = 1

        if self.with_lidar:
            lidar_unprocessed_front = self._load_npy(
                os.path.join(route_dir, "lidar", "%04d.npy" % frame_id)
            )[..., :4]
            lidar_odd_path = os.path.join(
                route_dir, "lidar_odd", "%04d.npy" % max(frame_id - 1, 0)
            )
            if os.path.isfile(lidar_odd_path):
                lidar_unprocessed_back = self._load_npy(lidar_odd_path)[..., :4]
                lidar_unprocessed = np.concatenate([lidar_unprocessed_front, lidar_unprocessed_back])
            else:
                lidar_unprocessed = lidar_unprocessed_front
            lidar_processed, num_points = lidar_to_raw_features(
                lidar_unprocessed, rotation_deg=0 if morai_route else -90
            )
            data['lidar'] = lidar_processed
            data['num_points'] = num_points

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = measurements["command"] - 1
        if cmd < 0:
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(measurements["speed"])
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float()

        data["measurements"] = mes
        data['velocity'] = torch.from_numpy(np.array([measurements['speed']])).float()
        data['command'] = torch.from_numpy(np.array(cmd))

        if np.isnan(measurements["theta"]):
            measurements["theta"] = 0
        ego_theta = measurements["theta"]
        x_command = measurements["x_command"]
        y_command = measurements["y_command"]
        ego_x = measurements["gps_x"]
        ego_y = measurements["gps_y"]
        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
            ]
        )
        local_command_point = np.array([x_command - ego_x, y_command - ego_y])
        local_command_point = R.T.dot(local_command_point)
        if any(np.isnan(local_command_point)):
            local_command_point[np.isnan(local_command_point)] = np.mean(
                local_command_point
            )
        local_command_point = torch.from_numpy(local_command_point).float()
        data["target_point"] = local_command_point

        command_waypoints = []
        for i in range(5):
            fid = min(len(measurements_all)-1, frame_id+5*(i+1))
            waypoint = [measurements_all[fid]['gps_x'], measurements_all[fid]['gps_y']]
            new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
            command_waypoints.append(new_loc.reshape(1, 2))
        command_waypoints = np.concatenate(command_waypoints)
        command_waypoints = torch.from_numpy(command_waypoints).float()

        if self.rgb_transform is not None:
            rgb_main_image = self.rgb_transform(rgb_image)
        data["rgb_front"] = rgb_main_image

        if self.rgb_center_transform is not None:
            rgb_center_image = self.rgb_center_transform(rgb_image)
        data["rgb_center"] = rgb_center_image

        if self.with_seg:
            if self.seg_transform is not None:
                seg_image = self.seg_transform(seg_image)
            data["seg"] = seg_image
            if self.multi_view:
                if self.multi_view_transform is not None:
                    seg_left_image = self.seg_transform(seg_left_image)
                    seg_right_image = self.seg_transform(seg_right_image)
                data["seg_left"] = seg_left_image
                data["seg_right"] = seg_right_image

        if self.with_depth:
            if self.depth_transform is not None:
                depth_image = self.depth_transform(depth_image)
            data["depth"] = depth_image
            if self.multi_view:
                if self.multi_view_transform is not None:
                    depth_left_image = self.multi_view_transform(depth_left_image)
                    depth_right_image = self.multi_view_transform(depth_right_image)
                data["depth_left"] = depth_left_image
                data["depth_right"] = depth_right_image


        if self.multi_view:
            if self.multi_view_transform is not None:
                rgb_left_image = self.multi_view_transform(rgb_left_image)
                rgb_right_image = self.multi_view_transform(rgb_right_image)
                rgb_rear_image = self.multi_view_transform(rgb_rear_image)
            data["rgb_left"] = rgb_left_image
            data["rgb_right"] = rgb_right_image
            data["rgb_rear"] = rgb_rear_image

        heatmap = generate_heatmap(
            copy.deepcopy(measurements), copy.deepcopy(actors_data)
        )
        if with_actor_infos:
            det_data, actor_infos = (
                generate_det_data(heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data), return_object_ids=True))
            det_data = det_data.reshape(2500, -1).astype(np.float32)
        else:
            det_data = (
            generate_det_data(
                heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data)).reshape(2500, -1).astype(np.float32)
            )

        heatmap_mask = np.zeros((50, 50), dtype=bool).reshape(-1)
        reduced_heatmap = block_reduce(heatmap[:250, 25:275], block_size=(5, 5), func=np.mean)
        flattened_reduced_heatmap = reduced_heatmap.reshape(-1)
        heatmap_mask[np.argsort(flattened_reduced_heatmap)[-10:]] = True
        heatmap_mask = heatmap_mask.reshape(50, 50)
        data['heatmap_mask'] = transforms.ToTensor()(heatmap_mask)

        img_traffic = heatmap[:250, 25:275, None]
        img_traffic = transforms.ToTensor()(img_traffic)

        img_traj = generate_future_waypoints(measurements)
        img_traj = img_traj[:250, 25:275, None]
        img_traj = transforms.ToTensor()(img_traj)

        reduced_heatmap = reduced_heatmap[:, :, None]
        reduced_heatmap = transforms.ToTensor()(reduced_heatmap) / 255.

        data = check_data(data, info=route_dir+str(frame_id))

        if with_actor_infos:
            return data, (
                img_traffic,
                command_waypoints,
                is_junction,
                traffic_light_state,
                det_data,
                img_traj,
                stop_sign,
                reduced_heatmap,
            ), actor_infos


        return data, (
            img_traffic,
            command_waypoints,
            is_junction,
            traffic_light_state,
            det_data,
            img_traj,
            stop_sign,
            reduced_heatmap,
        )


class MoraiExpertDataset(BaseIODataset):
    """Strict, single-frame MORAI adapter for LMDrive encoder pretraining.

    Global route points are used only to form ``target_point``.  The waypoint
    supervision is reconstructed from future synchronized Ego positions at
    +0.5 ... +2.5 seconds (10 Hz dataset).  No missing file is replaced with a
    neighbouring frame.
    """

    def __init__(
        self,
        root,
        split_file,
        split="train",
        input_rgb_size=224,
        input_lidar_size=224,
        rgb_transform=None,
        depth_transform=None,
        seg_transform=None,
        lidar_transform=None,
        multi_view_transform=None,
        with_lidar=True,
        multi_view=True,
        lidar_axis_policy="morai_flu_to_lmdrive_fru",
        sample_stride=1,
        traffic_label_sidecar=None,
        **kwargs,
    ):
        super().__init__()
        if not with_lidar or not multi_view:
            raise ValueError("MORAI LMDrive pretraining requires LiDAR and all four camera views")
        if lidar_axis_policy not in ("morai_flu_to_lmdrive_fru", "native_passthrough"):
            raise ValueError("Unsupported LiDAR axis policy: {}".format(lidar_axis_policy))
        if int(sample_stride) < 1:
            raise ValueError("sample_stride must be positive")

        self.dataset_root = Path(root).expanduser().resolve()
        self.split_file = Path(split_file).expanduser().resolve()
        self.split = "validation" if split in ("val", "validation") else split
        self.input_lidar_size = input_lidar_size
        self.input_rgb_size = input_rgb_size
        self.rgb_transform = rgb_transform
        self.rgb_center_transform = None
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform
        self.lidar_transform = lidar_transform
        self.multi_view_transform = multi_view_transform
        self.with_lidar = True
        self.multi_view = True
        self.lidar_axis_policy = lidar_axis_policy
        self.future_offsets = (5, 10, 15, 20, 25)
        self._measurements_cache = {}

        manifest = load_split_manifest(self.split_file)
        entries = list(split_entries(manifest, self.split))
        self.route_frames = entries[:: int(sample_stride)]
        if not self.route_frames:
            raise ValueError("MORAI {} split is empty".format(self.split))
        self.traffic_label_provenance = None
        self._traffic_label_records = None
        if traffic_label_sidecar is not None:
            self._traffic_label_records, self.traffic_label_provenance = load_traffic_label_sidecar(
                traffic_label_sidecar, self.dataset_root, self.split_file
            )

    def __len__(self):
        return len(self.route_frames)

    @staticmethod
    def _read_image(path):
        with Image.open(str(path)) as image:
            image.load()
            return image.convert("RGB")

    @staticmethod
    def _read_json(path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _route_path(self, relative):
        route = (self.dataset_root / "data" / relative).resolve()
        data_root = (self.dataset_root / "data").resolve()
        if data_root not in route.parents:
            raise ValueError("Route escapes dataset root: {}".format(relative))
        if any(part.startswith("_invalid") for part in route.relative_to(data_root).parts):
            raise ValueError("Invalid/quarantined route cannot be trained: {}".format(route))
        return route

    def _measurements_all(self, route):
        key = str(route)
        if key not in self._measurements_cache:
            values = self._read_json(route / "measurements_all.json")
            if not isinstance(values, list):
                raise ValueError("measurements_all.json must contain a list: {}".format(route))
            self._measurements_cache[key] = values
        return self._measurements_cache[key]

    @staticmethod
    def _local_rotation(theta):
        # Preserve the public LMDrive waypoint-head convention exactly.
        return np.array(
            [
                [np.cos(np.pi / 2 + theta), -np.sin(np.pi / 2 + theta)],
                [np.sin(np.pi / 2 + theta), np.cos(np.pi / 2 + theta)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _require_finite(name, value):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError("Non-finite {}".format(name))

    def __getitem__(self, index):
        relative, frame_id = self.route_frames[index]
        route = self._route_path(relative)
        stem = "{:04d}".format(frame_id)

        images = {
            name: self._read_image(route / ("rgb_" + name) / (stem + ".jpg"))
            for name in ("front", "left", "right", "rear")
        }
        measurements_all = self._measurements_all(route)
        measurement = self._read_json(route / "measurements_full" / (stem + ".json"))
        if int(measurement.get("frame_id", frame_id)) != frame_id:
            raise ValueError("Measurement frame_id mismatch at {}".format(stem))
        if measurement.get("navigation", {}).get("route_valid") is not True:
            raise ValueError("Invalid navigation sample reached split: {}".format(stem))

        native_frame = int(measurement["native_sync_frame"])
        if int(measurement.get("grpc_label_tick_raw", native_frame)) != native_frame:
            raise ValueError("gRPC/native tick mismatch at {}".format(stem))

        theta = float(measurement["theta"])
        ego_x = float(measurement["gps_x"])
        ego_y = float(measurement["gps_y"])
        rotation = self._local_rotation(theta)
        target_world = np.array(
            [float(measurement["x_command"]) - ego_x, float(measurement["y_command"]) - ego_y],
            dtype=np.float64,
        )
        target_point = rotation.T.dot(target_world).astype(np.float32)
        self._require_finite("target_point", target_point)

        future_waypoints = []
        for offset in self.future_offsets:
            future_id = frame_id + offset
            if future_id >= len(measurements_all):
                raise IndexError("Incomplete future horizon at frame {}".format(frame_id))
            future = measurements_all[future_id]
            if int(future.get("frame_id", future_id)) != future_id:
                raise ValueError("Future frame_id mismatch: {}".format(future_id))
            future_native = int(future["native_sync_frame"])
            if future_native - native_frame != offset * 5:
                raise ValueError(
                    "Non-contiguous native ticks at frame {} -> {}".format(frame_id, future_id)
                )
            delta = np.array(
                [float(future["gps_x"]) - ego_x, float(future["gps_y"]) - ego_y],
                dtype=np.float64,
            )
            future_waypoints.append(rotation.T.dot(delta))
        future_waypoints = np.asarray(future_waypoints, dtype=np.float32)
        self._require_finite("future_waypoints", future_waypoints)

        lidar = np.load(str(route / "lidar" / (stem + ".npy")), allow_pickle=False)
        if lidar.ndim != 2 or lidar.shape[1] < 4:
            raise ValueError("LiDAR must be [N,4+] at {}".format(stem))
        lidar = np.asarray(lidar[:, :4], dtype=np.float32).copy()
        self._require_finite("lidar", lidar)
        if self.lidar_axis_policy == "morai_flu_to_lmdrive_fru":
            lidar[:, 1] *= -1.0
        rng = None if self.split == "train" else np.random.RandomState(frame_id)
        lidar, num_points = lidar_to_raw_features(lidar, rotation_deg=0, rng=rng)

        if self.rgb_transform is None or self.rgb_center_transform is None:
            raise RuntimeError("Use create_carla_loader so RGB transforms are configured")
        if self.multi_view_transform is None:
            raise RuntimeError("MORAI multi-view transform is not configured")

        input_data = {
            "rgb_front": self.rgb_transform(images["front"]),
            "rgb_center": self.rgb_center_transform(images["front"]),
            "rgb_left": self.multi_view_transform(images["left"]),
            "rgb_right": self.multi_view_transform(images["right"]),
            "rgb_rear": self.multi_view_transform(images["rear"]),
            "lidar": lidar,
            "num_points": int(num_points),
            "velocity": torch.tensor([float(measurement["speed"])], dtype=torch.float32),
            "target_point": torch.from_numpy(target_point),
        }

        labels_aligned = (
            measurement.get("simulator_labels_source") == "grpc"
            and int(measurement.get("grpc_label_tick_raw", -1)) == native_frame
        )
        traffic_light_state = 0 if measurement.get("is_red_light_present") else 1
        if self._traffic_label_records is not None:
            traffic_light_state, labels_aligned = checked_traffic_target(
                self._traffic_label_records, relative, frame_id, measurement
            )
        targets = {
            "waypoints": torch.from_numpy(future_waypoints),
            "traffic_light": torch.tensor(traffic_light_state, dtype=torch.long),
            "traffic_available": torch.tensor(labels_aligned, dtype=torch.bool),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
            "native_sync_frame": torch.tensor(native_frame, dtype=torch.long),
        }
        return input_data, targets
