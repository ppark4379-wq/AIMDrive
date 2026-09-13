"""CARLA frames with perception targets for stage 1 (vision encoder pre-training).

Dataset root layout (LMDrive release)
  <root>/dataset_index.txt     "<route folder>/ <frame count>" per line
  <root>/data/routes_townXX_<type>_wYY_<time>/
      rgb_full/0000.jpg         front, left, right and rear 800x600 views stacked vertically
                                (or rgb_front/, rgb_left/, rgb_right/, rgb_rear/ folders)
      lidar/0000.npy, lidar_odd/0000.npy
      measurements_full/0000.json, measurements_all.json

Each item is (inputs, targets)
  inputs   rgb_front, rgb_left, rgb_right, rgb_rear, rgb_center, lidar, num_points,
           velocity, target_point, measurements, command, heatmap_mask
  targets  0 BEV occupancy image      [1, 250, 250]
           1 future waypoints         [5, 2]
           2 is_junction              int
           3 traffic light state      int, 0 = a red light affects the ego vehicle
           4 detection map            [2500, 8]
           5 future route image       [1, 250, 250]
           6 stop sign                int
           7 reduced occupancy        [1, 50, 50]
"""
import copy
import fnmatch
import logging
import os
import re

import numpy as np
import torch
from skimage.measure import block_reduce
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from utils.distributed import OrderedDistributedSampler

from .lidar import lidar_to_raw_features, replace_non_finite
from .perception_labels import generate_det_data, generate_future_waypoints, generate_heatmap
from .sensor_io import SensorFileDataset
from .transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_camera_transform

logger = logging.getLogger(__name__)

ROUTE_PATTERN = re.compile(r"town(\d\d).*w(\d+)")


def matches_route(route_dir, patterns):
    """True when the route folder name matches one of the glob patterns."""
    name = os.path.basename(os.path.normpath(route_dir))
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


class CarlaPerceptionDataset(SensorFileDataset):
    """Route selection: towns and weathers read the CARLA folder naming, routes matches
    folder names with glob patterns.  Leave all three unset to take every route, which is
    what recordings without town/weather in the name (MORAI for example) need."""

    def __init__(self, root, towns=None, weathers=None, routes=None, index_file="dataset_index.txt",
                 with_lidar=True, multi_view=True, lidar_rotation_deg=-90):
        super().__init__()
        self.with_lidar = with_lidar
        self.multi_view = multi_view
        self.lidar_rotation_deg = lidar_rotation_deg
        # Assigned by create_perception_loader.
        self.rgb_transform = None
        self.rgb_center_transform = None
        self.multi_view_transform = None
        self.route_frames = self._get_frames_paths(root, index_file, weathers, towns, routes)

    def _get_frames_paths(self, root, index_file, weathers, towns, routes=None):
        route_frames = []
        route_dir_nums = 0
        for line in self._load_text(os.path.join(root, index_file)).split("\n"):
            if len(line.split()) != 2:
                continue
            path, frames = line.split()
            # Index entries are relative either to the root ("data/routes_town01_...")
            # or to its data/ folder ("routes_town01_..."); both are accepted.
            path = os.path.join(root, path)
            if not os.path.isdir(path):
                path = os.path.join(root, "data", os.path.relpath(path, root))
            if routes is not None and not matches_route(path, routes):
                continue
            if towns is not None or weathers is not None:
                match = ROUTE_PATTERN.findall(path)
                if len(match) != 1:
                    continue  # no town/weather in the name, so it cannot pass the filter
                town, weather = int(match[0][0]), int(match[0][1])
                if (towns is not None and town not in towns) or (weathers is not None and weather not in weathers):
                    continue
            route_dir_nums += 1
            for i in range(int(frames)):
                route_frames.append((path, i, route_dir_nums))
        logger.info("Sub route dir nums: %d", len(route_frames))
        return route_frames

    def __len__(self):
        return int(len(self.route_frames) / 5)

    def __getitem__(self, idx):
        # One item per 5 frames; a random frame of each group is used every epoch.
        idx = idx * 5 + np.random.randint(0, 5)
        route_dir, frame_id, _ = self.route_frames[idx]
        return self._extract_data_item(route_dir, frame_id)

    def _extract_data_item(self, route_dir, frame_id):
        data = {}
        if os.path.isdir(os.path.join(route_dir, "rgb_full")):
            rgb_full_image = self._load_image(os.path.join(route_dir, "rgb_full", "%04d.jpg" % frame_id))
            rgb_image = rgb_full_image.crop((0, 0, 800, 600))
            rgb_left_image = rgb_full_image.crop((0, 600, 800, 1200))
            rgb_right_image = rgb_full_image.crop((0, 1200, 800, 1800))
            rgb_rear_image = rgb_full_image.crop((0, 1800, 800, 2400))
        else:  # data not merged with batch_merge_data.py
            rgb_image = self._load_image(os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id))
            rgb_left_image = self._load_image(os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id))
            rgb_right_image = self._load_image(os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id))
            rgb_rear_image = self._load_image(os.path.join(route_dir, "rgb_rear", "%04d.jpg" % frame_id))

        measurements_all = self._load_json(os.path.join(route_dir, "measurements_all.json"))
        measurements = self._load_json(os.path.join(route_dir, "measurements_full", "%04d.json" % frame_id))
        actors_data = measurements["actors_data"]
        stop_sign = int(measurements["stop_sign"])
        is_junction = 1 if measurements["is_junction"] is True else 0
        traffic_light_state = 0 if len(measurements["is_red_light_present"]) > 0 else 1

        if self.with_lidar:
            points = self._load_npy(os.path.join(route_dir, "lidar", "%04d.npy" % frame_id))[..., :4]
            # CARLA writes half a sweep per frame and the other half into lidar_odd;
            # recordings with a full sweep per frame have no lidar_odd folder.
            odd_path = os.path.join(route_dir, "lidar_odd", "%04d.npy" % max(frame_id - 1, 0))
            if os.path.isfile(odd_path):
                points = np.concatenate([points, self._load_npy(odd_path)[..., :4]])
            lidar, num_points = lidar_to_raw_features(
                points, rotation_deg=self.lidar_rotation_deg, clip_abs=100)
            data["lidar"] = lidar
            data["num_points"] = num_points

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = measurements["command"] - 1
        if cmd < 0:
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(measurements["speed"])
        data["measurements"] = torch.from_numpy(np.array(cmd_one_hot)).float()
        data["velocity"] = torch.from_numpy(np.array([measurements["speed"]])).float()
        data["command"] = torch.from_numpy(np.array(cmd))

        if np.isnan(measurements["theta"]):
            measurements["theta"] = 0
        ego_theta = measurements["theta"]
        ego_x = measurements["gps_x"]
        ego_y = measurements["gps_y"]
        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
        ])
        local_command_point = np.array([measurements["x_command"] - ego_x, measurements["y_command"] - ego_y])
        local_command_point = R.T.dot(local_command_point)
        if any(np.isnan(local_command_point)):
            local_command_point[np.isnan(local_command_point)] = np.mean(local_command_point)
        data["target_point"] = torch.from_numpy(local_command_point).float()

        command_waypoints = []
        for i in range(5):
            fid = min(len(measurements_all) - 1, frame_id + 5 * (i + 1))
            waypoint = [measurements_all[fid]["gps_x"], measurements_all[fid]["gps_y"]]
            new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
            command_waypoints.append(new_loc.reshape(1, 2))
        command_waypoints = torch.from_numpy(np.concatenate(command_waypoints)).float()

        data["rgb_front"] = self.rgb_transform(rgb_image)
        data["rgb_center"] = self.rgb_center_transform(rgb_image)
        if self.multi_view:
            data["rgb_left"] = self.multi_view_transform(rgb_left_image)
            data["rgb_right"] = self.multi_view_transform(rgb_right_image)
            data["rgb_rear"] = self.multi_view_transform(rgb_rear_image)

        heatmap = generate_heatmap(copy.deepcopy(measurements), copy.deepcopy(actors_data))
        det_data = generate_det_data(heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data))
        det_data = det_data.reshape(2500, -1).astype(np.float32)

        heatmap_mask = np.zeros((50, 50), dtype=bool).reshape(-1)
        reduced_heatmap = block_reduce(heatmap[:250, 25:275], block_size=(5, 5), func=np.mean)
        heatmap_mask[np.argsort(reduced_heatmap.reshape(-1))[-10:]] = True
        data["heatmap_mask"] = transforms.ToTensor()(heatmap_mask.reshape(50, 50))

        img_traffic = transforms.ToTensor()(heatmap[:250, 25:275, None])
        img_traj = transforms.ToTensor()(generate_future_waypoints(measurements)[:250, 25:275, None])
        reduced_heatmap = transforms.ToTensor()(reduced_heatmap[:, :, None]) / 255.0

        data = replace_non_finite(data, info=route_dir + str(frame_id))
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


def build_perception_dataset(data_cfg, split):
    split_cfg = data_cfg[split]
    return CarlaPerceptionDataset(
        root=data_cfg["root"],
        towns=split_cfg.get("towns"),
        weathers=split_cfg.get("weathers"),
        routes=split_cfg.get("routes"),
        index_file=data_cfg.get("index_file", "dataset_index.txt"),
        lidar_rotation_deg=data_cfg.get("lidar_rotation_deg", -90),
    )


def create_perception_loader(dataset, batch_size, is_training, input_size=(3, 224, 224),
                             multi_view_input_size=(3, 128, 128), scale=(1.0, 1.0), color_jitter=0.0,
                             num_workers=4, distributed=False, pin_memory=False,
                             mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    dataset.rgb_transform = create_camera_transform(
        input_size, is_training=is_training, scale=scale, color_jitter=color_jitter, mean=mean, std=std)
    dataset.rgb_center_transform = create_camera_transform(
        128, is_training=is_training, scale=None, color_jitter=color_jitter, need_scale=False, mean=mean, std=std)
    dataset.multi_view_transform = create_camera_transform(
        multi_view_input_size or input_size, is_training=is_training, scale=scale,
        color_jitter=color_jitter, mean=mean, std=std)

    sampler = None
    if distributed:
        # Validation pads with a few repeated samples so every rank sees the same count.
        sampler = DistributedSampler(dataset) if is_training else OrderedDistributedSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=default_collate,
        pin_memory=pin_memory,
        drop_last=is_training,
        persistent_workers=num_workers > 0,
    )
