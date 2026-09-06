"""MORAI dataset with separate route indexing, raw I/O and label calculation.

Uses per-view images and one 360-degree LiDAR scan without CARLA rotation.
The model input keys and eight-element label tuple follow Memfuser's contract.
Waypoint targets use original frame IDs in measurements_all.json; filtering
invalid navigation frames must not compress that timeline.
"""

import os
import copy
import logging

import numpy as np
import torch
from torchvision import transforms
from skimage.measure import block_reduce

from .base_io_dataset import BaseIODataset
from .heatmap_utils import generate_heatmap, generate_future_waypoints
from .det_utils import generate_det_data
from .morai_utils import lidar_to_raw_features
from .morai_routes import get_route_frames

_logger = logging.getLogger(__name__)


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


class MoraiMVDetDataset(BaseIODataset):
    def __init__(
        self,
        root,
        routes=None,
        head="det",
        input_rgb_size=224,
        rgb_transform=None,
        lidar_transform=None,
        multi_view_transform=None,
        with_waypoints=False,
        with_seg=False,
        with_depth=False,
        waypoint_interval=5,
        index_filename="dataset_index.txt",
        with_lidar=True,
        multi_view=True,
        **kwargs,
    ):
        super().__init__()
        if with_waypoints or with_seg or with_depth:
            raise ValueError("MORAI raw data does not provide waypoints.npy, segmentation or depth")
        if not isinstance(waypoint_interval, int) or waypoint_interval < 1:
            raise ValueError("waypoint_interval must be a positive frame count")
        self.waypoint_interval = waypoint_interval
        self.head = head
        self.input_rgb_size = input_rgb_size

        # morai_loader.create_morai_loader가 이 속성들에 직접 transform을 주입함
        self.rgb_transform = rgb_transform
        self.rgb_center_transform = None
        self.lidar_transform = lidar_transform
        self.multi_view_transform = multi_view_transform

        self.with_waypoints = with_waypoints
        self.with_lidar = with_lidar
        self.multi_view = multi_view
        self.index_filename = index_filename

        self.route_frames = self._get_frames_paths(root, routes)

    # ------------------------------------------------------------------ #
    # 1. 인덱싱
    # ------------------------------------------------------------------ #
    def _get_frames_paths(self, root, routes):
        """
        town/weather 정규식 없이 dataset_index.txt의 "경로 프레임수" 줄을 그대로 사용.
        routes(라우트 이름 리스트)가 주어지면 그 라우트만 필터링 (train/val 분리용).
        """
        route_frames = get_route_frames(root, routes, index_filename=self.index_filename)
        _logger.info("Loaded %d valid frames from %d routes", len(route_frames),
                     len({path for path, _ in route_frames}))
        return route_frames

    def __len__(self):
        return len(self.route_frames)

    # ------------------------------------------------------------------ #
    # 2. I/O -- 디스크에서 열기만 함, 가공 없음
    # ------------------------------------------------------------------ #
    def _load_raw_frame(self, route_dir, frame_id):
        raw = {
            "rgb_front": self._load_image(os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id)),
            "measurements_all": self._load_json(os.path.join(route_dir, "measurements_all.json")),
            "measurements": self._load_json(
                os.path.join(route_dir, "measurements_full", "%04d.json" % frame_id)
            ),
        }
        if self.multi_view:
            for view in ("left", "right", "rear"):
                raw["rgb_" + view] = self._load_image(
                    os.path.join(route_dir, "rgb_" + view, "%04d.jpg" % frame_id)
                )
        if self.with_lidar:
            raw["lidar"] = self._load_npy(os.path.join(route_dir, "lidar", "%04d.npy" % frame_id))[..., :4]
        return raw

    # ------------------------------------------------------------------ #
    # 3. 라벨계산 -- measurements/actors_data로부터 정답 계산.
    #    generate_heatmap/generate_det_data는 CARLA 원본 함수 그대로 재사용.
    # ------------------------------------------------------------------ #
    def _compute_labels(self, raw, frame_id):
        measurements = raw["measurements"]
        measurements_all = raw["measurements_all"]
        actors_data = measurements["actors_data"]

        stop_sign = int(measurements["stop_sign"])
        is_junction = 1 if measurements["is_junction"] else 0
        traffic_light_state = 0 if len(measurements["is_red_light_present"]) > 0 else 1

        # command / speed
        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = measurements["command"] - 1
        if cmd < 0:
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(measurements["speed"])

        # target point (gps_x/gps_y 기준. heatmap이 쓰는 x/y와는 다른 좌표라 주의)
        ego_theta = measurements["theta"] if not np.isnan(measurements["theta"]) else 0
        ego_x, ego_y = measurements["gps_x"], measurements["gps_y"]
        x_command, y_command = measurements["x_command"], measurements["y_command"]
        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
            ]
        )
        local_command_point = R.T.dot(np.array([x_command - ego_x, y_command - ego_y]))
        if np.any(np.isnan(local_command_point)):
            local_command_point[np.isnan(local_command_point)] = np.mean(local_command_point)

        # LMDrive 원본과 동일한 별도 GRU waypoint target이다. 수집 frame의
        # 실제 미래 Ego 위치를 사용하며, planned dense route를 담는
        # measurements["future_waypoints"]와 혼동하지 않는다.
        command_waypoints = []
        for i in range(5):
            fid = min(len(measurements_all) - 1, frame_id + self.waypoint_interval * (i + 1))
            waypoint = [measurements_all[fid]["gps_x"], measurements_all[fid]["gps_y"]]
            new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
            command_waypoints.append(new_loc.reshape(1, 2))
        command_waypoints = torch.from_numpy(np.concatenate(command_waypoints)).float()

        # Match waypoint/image BEV axes: lateral +x, forward -y.
        bev_measurements = copy.deepcopy(measurements)
        bev_measurements["theta"] = ego_theta + np.pi / 2
        heatmap = generate_heatmap(bev_measurements, copy.deepcopy(actors_data))
        det_data = (
            generate_det_data(heatmap, bev_measurements, copy.deepcopy(actors_data))
            .reshape(2500, -1)
            .astype(np.float32)
        )

        heatmap_mask = np.zeros((50, 50), dtype=bool).reshape(-1)
        reduced_heatmap = block_reduce(heatmap[:250, 25:275], block_size=(5, 5), func=np.mean)
        heatmap_mask[np.argsort(reduced_heatmap.reshape(-1))[-10:]] = True
        heatmap_mask = heatmap_mask.reshape(50, 50)

        img_traffic = heatmap[:250, 25:275, None]
        img_traj = generate_future_waypoints(measurements)[:250, 25:275, None]

        return {
            "cmd_one_hot": cmd_one_hot,
            "speed": measurements["speed"],
            "cmd": cmd,
            "target_point": local_command_point,
            "command_waypoints": command_waypoints,
            "is_junction": is_junction,
            "traffic_light_state": traffic_light_state,
            "stop_sign": stop_sign,
            "det_data": det_data,
            "img_traffic": img_traffic,
            "img_traj": img_traj,
            "heatmap_mask": heatmap_mask,
            "reduced_heatmap": reduced_heatmap,
        }

    # ------------------------------------------------------------------ #
    # 접합부: I/O -> transform 적용 -> 라벨계산 결과를 최종 형태로 포장
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        route_dir, frame_id = self.route_frames[idx]
        return self._extract_data_item(route_dir, frame_id)

    def _extract_data_item(self, route_dir, frame_id):
        raw = self._load_raw_frame(route_dir, frame_id)
        labels = self._compute_labels(raw, frame_id)

        data = {}

        if self.with_lidar:
            lidar_processed, num_points = lidar_to_raw_features(raw["lidar"])
            data["lidar"] = lidar_processed
            data["num_points"] = num_points

        data["measurements"] = torch.from_numpy(np.array(labels["cmd_one_hot"])).float()
        data["velocity"] = torch.from_numpy(np.array([labels["speed"]])).float()
        data["command"] = torch.from_numpy(np.array(labels["cmd"]))
        data["target_point"] = torch.from_numpy(labels["target_point"]).float()

        if self.rgb_transform is not None:
            data["rgb_front"] = self.rgb_transform(raw["rgb_front"])
        if self.rgb_center_transform is not None:
            data["rgb_center"] = self.rgb_center_transform(raw["rgb_front"])
        if self.multi_view and self.multi_view_transform is not None:
            data["rgb_left"] = self.multi_view_transform(raw["rgb_left"])
            data["rgb_right"] = self.multi_view_transform(raw["rgb_right"])
            data["rgb_rear"] = self.multi_view_transform(raw["rgb_rear"])

        data["heatmap_mask"] = transforms.ToTensor()(labels["heatmap_mask"])
        img_traffic = transforms.ToTensor()(labels["img_traffic"])
        img_traj = transforms.ToTensor()(labels["img_traj"])
        reduced_heatmap = transforms.ToTensor()(labels["reduced_heatmap"][:, :, None]) / 255.0

        data = check_data(data, info=route_dir + str(frame_id))

        return data, (
            img_traffic,
            labels["command_waypoints"],
            labels["is_junction"],
            labels["traffic_light_state"],
            labels["det_data"],
            img_traj,
            labels["stop_sign"],
            reduced_heatmap,
        )
