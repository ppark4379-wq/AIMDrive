"""CARLA clips paired with language instructions for stage 2 (instruction fine-tuning).

Dataset root layout (LMDrive release)
  <root>/navigation_instruction_list.txt  one JSON object per line: route_path, town_id,
      weather_id, start_frame, end_frame, instruction, instruction_id, instruction_args, route_frames
  <root>/notice_instruction_list.json     {route_path: [{"frame_id": ..., "instruction_id": ...}, ...]}
  <root>/<route_path>/rgb_full, lidar, lidar_odd, measurements_all.json

An item holds up to frames_per_clip frames taken every sample_interval frames,
zero-padded to frames_per_clip, the instruction text, and for every frame the
next 5 waypoints expressed in that frame's own ego coordinates.
"""
import fnmatch
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate

from .lidar import lidar_to_raw_features, replace_non_finite
from .sensor_io import SensorFileDataset
from .transforms import create_camera_transform

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION_TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruction_templates.json")


class CarlaInstructionDataset(SensorFileDataset):
    def __init__(
        self,
        root,
        towns=None,
        weathers=None,
        routes=None,
        frames_per_clip=32,
        sample_interval=2,
        start_frame_augment=False,
        use_notice=False,
        instruction_templates=None,
        fill_instruction_placeholders=False,
        input_rgb_size=224,
        input_multi_view_size=128,
        is_training=False,
        scale=None,
        lidar_rotation_deg=-90,
    ):
        super().__init__()
        self.token_max_length = frames_per_clip
        self.lidar_rotation_deg = lidar_rotation_deg
        self.rgb_transform = create_camera_transform(input_rgb_size, is_training=is_training, scale=scale)
        self.rgb_center_transform = create_camera_transform(
            128, is_training=is_training, scale=None, need_scale=False)
        self.multi_view_transform = create_camera_transform(
            input_multi_view_size, is_training=is_training, scale=scale)

        self.scenario_infos = self._get_scenario_paths(root, weathers, towns, routes)
        logger.info("Scenario nums: %d", len(self.scenario_infos))
        with open(instruction_templates or DEFAULT_INSTRUCTION_TEMPLATES) as stream:
            self.instruction_dict = json.load(stream)
        self.sample_interval = sample_interval
        self.enable_start_frame_augment = start_frame_augment
        self.enable_notice = use_notice
        self.fill_instruction_placeholders = fill_instruction_placeholders
        if self.enable_notice:
            raw_notice_data = self._load_json(os.path.join(root, "notice_instruction_list.json"))
            self.notice_data = {os.path.join(root, key): value for key, value in raw_notice_data.items()}

    @staticmethod
    def collater(samples):
        return default_collate(samples)

    def _get_scenario_paths(self, root, weathers, towns, routes=None):
        scenario_infos = []
        for line in self._load_text(os.path.join(root, "navigation_instruction_list.txt")).split("\n"):
            if len(line) < 10:
                continue
            info = json.loads(line.strip())
            if towns is not None and info["town_id"] not in towns:
                continue
            if weathers is not None and info["weather_id"] not in weathers:
                continue
            name = os.path.basename(os.path.normpath(info["route_path"]))
            if routes is not None and not any(fnmatch.fnmatch(name, pattern) for pattern in routes):
                continue
            info["route_path"] = os.path.join(root, info["route_path"])
            scenario_infos.append(info)
        return scenario_infos

    def __len__(self):
        return len(self.scenario_infos)

    def pad_and_stack(self, data):
        if isinstance(data[0], np.ndarray):
            for _ in range(self.token_max_length - len(data)):
                data.append(np.zeros_like(data[0]))
            return np.stack(data, 0)
        if torch.is_tensor(data[0]):
            for _ in range(self.token_max_length - len(data)):
                data.append(torch.zeros_like(data[0]))
            return torch.stack(data, 0)
        for _ in range(self.token_max_length - len(data)):
            data.append(0)
        return np.array(data).reshape(-1)

    @staticmethod
    def _fill_placeholders(text, args, info):
        try:
            if "[x]" in text:
                text = text.replace("[x]", str(args[0]))
            if "left/right" in text:
                text = text.replace("left/right", str(args[1]))
            if "[y]" in text:
                text = text.replace("[y]", str(args[2]))
        except IndexError:
            logger.error("instruction %s lacks arguments for: %s", info.get("instruction_id"), text)
        return text

    def __getitem__(self, idx):
        info = self.scenario_infos[idx]
        route_path = info["route_path"]
        route_frames = int(info["route_frames"])
        end_frame = info["end_frame"]
        if "Turn" in info["instruction"]:
            # LMDrive wrote this back into info, which made it grow on every
            # epoch when num_workers=0; the first-epoch value is used instead.
            end_frame = min(route_frames - 1, end_frame + 12)

        sample_interval = self.sample_interval
        start_frame_id = info["start_frame"] + random.randint(0, sample_interval)
        if self.enable_start_frame_augment and len(info["instruction_args"]) == 0:
            # instructions without a distance argument may start a little earlier
            if "Other" not in info["instruction"] or info["instruction"] != "Follow-01":
                augment_range = min(16, max(0, self.token_max_length * self.sample_interval
                                            - (end_frame - info["start_frame"])))
                start_frame_id = max(0, start_frame_id - random.randint(0, augment_range))
        end_frame_id = min(end_frame + 1, start_frame_id + self.token_max_length * sample_interval
                           - random.randint(0, sample_interval - 1))

        if self.enable_notice:
            notice_frame_id = -1
            notice_text = ""
            if route_path in self.notice_data:
                notice_list = [x for x in self.notice_data[route_path]
                               if start_frame_id < x["frame_id"] < end_frame_id - 16]
                if not (len(notice_list) < 1 or random.random() < 0.75):
                    notice = random.choice(notice_list)
                    # absolute frame id -> index of the clip frame after which the notice appears
                    notice_frame_id = (notice["frame_id"] - start_frame_id) // sample_interval + 1
                    notice_text = np.random.choice(self.instruction_dict[str(notice["instruction_id"])])

        measurements = self._load_json(os.path.join(route_path, "measurements_all.json"))
        ego_theta = measurements[start_frame_id]["theta"]
        if np.isnan(ego_theta):
            ego_theta = 0
        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
        ])
        origin_x = measurements[start_frame_id]["gps_x"]
        origin_y = measurements[start_frame_id]["gps_y"]

        ego_throttles, ego_steers, ego_brakes, ego_velocitys = [], [], [], []
        ego_xs, ego_ys, local_positions, local_future_waypoints = [], [], [], []
        text_before_img, text_after_img, target_points = [], [], []

        for frame_id in range(start_frame_id, end_frame_id, sample_interval):
            ego_x = measurements[frame_id]["gps_x"]
            ego_y = measurements[frame_id]["gps_y"]
            velocity = measurements[frame_id]["speed"]
            local_position = R.T.dot(np.array([ego_x - origin_x, ego_y - origin_y]))
            text_before_img.append("<frame %.1f,%.1f>" % (local_position[0], local_position[1]))
            text_after_img.append("</frame>")

            ego_xs.append(ego_x)
            ego_ys.append(ego_y)
            ego_throttles.append(measurements[frame_id]["throttle"])
            ego_steers.append(measurements[frame_id]["steer"])
            ego_brakes.append(int(measurements[frame_id]["brake"]))
            ego_velocitys.append(velocity)
            local_positions.append(local_position.reshape(-1))

            local_ego_theta = measurements[frame_id]["theta"]
            if np.isnan(local_ego_theta):
                local_ego_theta = 0
            local_R = np.array([
                [np.cos(np.pi / 2 + local_ego_theta), -np.sin(np.pi / 2 + local_ego_theta)],
                [np.sin(np.pi / 2 + local_ego_theta), np.cos(np.pi / 2 + local_ego_theta)],
            ])
            local_command_point = np.array([measurements[frame_id]["x_command"] - ego_x,
                                            measurements[frame_id]["y_command"] - ego_y])
            local_command_point = local_R.T.dot(local_command_point)
            if any(np.isnan(local_command_point)):
                local_command_point[np.isnan(local_command_point)] = np.mean(local_command_point)
            target_points.append(local_command_point.reshape(-1))

            future = []
            for future_frame_delta in range(1, 6):
                future_frame_id = min(frame_id + future_frame_delta * 5, route_frames - 1)
                future_waypoint = np.array([measurements[future_frame_id]["gps_x"] - ego_x,
                                            measurements[future_frame_id]["gps_y"] - ego_y])
                future.append(local_R.T.dot(future_waypoint).reshape(1, 2))
            local_future_waypoints.append(np.concatenate(future, axis=0).reshape(-1))

        valid_frames = len(ego_xs)
        ego_throttles = self.pad_and_stack(ego_throttles)
        ego_steers = self.pad_and_stack(ego_steers)
        ego_brakes = self.pad_and_stack(ego_brakes)
        ego_velocitys = self.pad_and_stack(ego_velocitys)
        self.pad_and_stack(ego_xs)
        self.pad_and_stack(ego_ys)
        target_points = self.pad_and_stack(target_points)
        local_positions = self.pad_and_stack(local_positions)
        local_future_waypoints = self.pad_and_stack(local_future_waypoints)

        lidar_data, lidar_num_points = [], []
        rgb_front, rgb_center, rgb_left, rgb_right, rgb_rear = [], [], [], [], []
        for frame_id in range(start_frame_id, end_frame_id, sample_interval):
            sensor_data = self._extract_data_item(route_path, frame_id)
            lidar_data.append(sensor_data["lidar"])
            lidar_num_points.append(sensor_data["num_points"])
            rgb_front.append(sensor_data["rgb"])
            rgb_center.append(sensor_data["rgb_center"])
            rgb_left.append(sensor_data["rgb_left"])
            rgb_right.append(sensor_data["rgb_right"])
            rgb_rear.append(sensor_data["rgb_rear"])

        processed_data = {
            "lidar": self.pad_and_stack(lidar_data),
            "num_points": self.pad_and_stack(lidar_num_points),
            "rgb_front": self.pad_and_stack(rgb_front),
            "rgb_left": self.pad_and_stack(rgb_left),
            "rgb_right": self.pad_and_stack(rgb_right),
            "rgb_rear": self.pad_and_stack(rgb_rear),
            "rgb_center": self.pad_and_stack(rgb_center),
        }

        instruction_text = np.random.choice(self.instruction_dict[str(info["instruction_id"])])
        if self.fill_instruction_placeholders:
            instruction_text = self._fill_placeholders(instruction_text, info["instruction_args"], info)
        # With fill_instruction_placeholders false the text keeps "[x]" as LMDrive did:
        # its str.replace results were discarded, so the released models never saw distances.

        processed_data["target_point"] = torch.from_numpy(target_points).float()
        processed_data["valid_frames"] = valid_frames
        processed_data["text_input"] = instruction_text
        processed_data["text_before_img"] = "|".join(text_before_img)
        processed_data["text_after_img"] = "|".join(text_after_img)
        processed_data["ego_throttles"] = ego_throttles
        processed_data["ego_steers"] = ego_steers
        processed_data["ego_brakes"] = ego_brakes
        processed_data["velocity"] = torch.from_numpy(np.array(ego_velocitys)).float()
        processed_data["local_positions"] = local_positions
        processed_data["local_future_waypoints"] = local_future_waypoints
        if self.enable_notice:
            processed_data["notice_frame_id"] = notice_frame_id
            processed_data["notice_text"] = notice_text
        return processed_data

    def _extract_data_item(self, route_path, frame_id):
        data = {}
        if os.path.isdir(os.path.join(route_path, "rgb_full")):
            rgb_full_image = self._load_image(os.path.join(route_path, "rgb_full", "%04d.jpg" % frame_id))
            rgb_image = rgb_full_image.crop((0, 0, 800, 600))
            rgb_left_image = rgb_full_image.crop((0, 600, 800, 1200))
            rgb_right_image = rgb_full_image.crop((0, 1200, 800, 1800))
            rgb_rear_image = rgb_full_image.crop((0, 1800, 800, 2400))
        else:  # camera views not merged into rgb_full
            rgb_image = self._load_image(os.path.join(route_path, "rgb_front", "%04d.jpg" % frame_id))
            rgb_left_image = self._load_image(os.path.join(route_path, "rgb_left", "%04d.jpg" % frame_id))
            rgb_right_image = self._load_image(os.path.join(route_path, "rgb_right", "%04d.jpg" % frame_id))
            rgb_rear_image = self._load_image(os.path.join(route_path, "rgb_rear", "%04d.jpg" % frame_id))

        points = self._load_npy(os.path.join(route_path, "lidar", "%04d.npy" % frame_id))[..., :4]
        odd_path = os.path.join(route_path, "lidar_odd", "%04d.npy" % max(frame_id - 1, 0))
        if os.path.isfile(odd_path):
            points = np.concatenate([points, self._load_npy(odd_path)[..., :4]])
        data["lidar"], data["num_points"] = lidar_to_raw_features(
            points, rotation_deg=self.lidar_rotation_deg)

        data["rgb"] = self.rgb_transform(rgb_image)
        data["rgb_center"] = self.rgb_center_transform(rgb_image)
        data["rgb_left"] = self.multi_view_transform(rgb_left_image)
        data["rgb_right"] = self.multi_view_transform(rgb_right_image)
        data["rgb_rear"] = self.multi_view_transform(rgb_rear_image)
        return replace_non_finite(data, info=route_path + str(frame_id))


def build_instruction_dataset(data_cfg, split):
    split_cfg = data_cfg[split]
    return CarlaInstructionDataset(
        root=data_cfg["root"],
        towns=split_cfg.get("towns"),
        weathers=split_cfg.get("weathers"),
        routes=split_cfg.get("routes"),
        frames_per_clip=int(data_cfg.get("frames_per_clip", 40)),
        sample_interval=int(data_cfg.get("sample_interval", 2)),
        start_frame_augment=bool(data_cfg.get("start_frame_augment", False)),
        use_notice=bool(data_cfg.get("use_notice", False)),
        instruction_templates=data_cfg.get("instruction_templates"),
        fill_instruction_placeholders=bool(data_cfg.get("fill_instruction_placeholders", False)),
        lidar_rotation_deg=data_cfg.get("lidar_rotation_deg", -90),
    )


def create_instruction_loader(dataset, batch_size, is_training, num_workers=4, distributed=False):
    sampler = DistributedSampler(dataset, shuffle=is_training) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None and is_training,
        collate_fn=dataset.collater,
        drop_last=is_training,
    )
