"""Stage 1 dataset: one synchronized MORAI frame -> vision-encoder inputs and targets.

The frames come from a split manifest written by ``tools/create_split.py``
(see ``datasets/split_manifest.py``), so the train/validation boundary is
chronological and reproducible.  Waypoint supervision is rebuilt from the
future synchronized ego positions at +0.5 ... +2.5 s (10 Hz recordings); no
missing file is ever silently replaced with a neighbouring frame.

Traffic-light labels come from the measurement stream, or from the audited
sidecar built by ``tools/build_traffic_labels.py`` when one is given.

Formerly ``vision_encoder/timm/data/carla_dataset.py::MoraiExpertDataset``.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .lidar import lidar_to_raw_features
from .split_manifest import load_split_manifest, split_entries
from .traffic_labels import checked_traffic_target, load_traffic_label_sidecar


class MoraiPerceptionDataset(Dataset):
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
        lidar, num_points = lidar_to_raw_features(lidar, rotation_deg=0, clip_abs=100, rng=rng)

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
