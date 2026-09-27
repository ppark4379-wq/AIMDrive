"""Stage-1 inputs straight from a MORAI native sensor capture.

The scenario-campaign recorder writes two things side by side: ROS chunks for
the control/label streams, and MORAI's own per-sensor folders for the heavy
payloads.  ``capture.record_ros_cameras`` is false, so the cameras exist only
in the native folders:

    <capture>/CAMERA_1/<stamp>_Intensity.png     front   1280x720 RGBA
    <capture>/CAMERA_2/<stamp>_Intensity.png     left     640x480
    <capture>/CAMERA_3/<stamp>_Intensity.png     right    640x480
    <capture>/CAMERA_4/<stamp>_Intensity.png     rear     640x480
    <capture>/LIDAR_5/<stamp>_Intensity.bin      [N, 4] float32
    <capture>/GPS_6/<stamp>.txt                  lat/lon/alt + UTM offsets
    <capture>/IMU_7/<stamp>.txt                  orientation quaternion

Every sensor shares one timestamp per frame, which is what the recorder counts
as a "complete filename bundle", so frames are formed by filename rather than
by re-synchronising streams.

This class exists because that layout is not the CARLA-style
``data/<route>/{rgb_front,lidar,measurements_full}`` tree
:mod:`datasets.morai_perception_dataset` reads, and post-processing a capture
into that tree is a separate offline stage that may not have run yet.  It
yields exactly the same input dict, so the encoder and the visualisers do not
care which one produced a frame.

Derived quantities, and how each was checked against a real capture
------------------------------------------------------------------
ego x, y   lat/lon -> UTM zone 52N, minus the EastOffset/NorthOffset recorded
           in the same GPS file.  Measured against the campaign's own route
           CSV, the resulting track sits 0.11-0.28 m from the route.
theta      yaw of the IMU quaternion.  On a turning frame this read 149.0 deg
           against 148.1 deg for the direction of travel between consecutive
           positions, so it is the same heading convention the CARLA-style
           ``measurements_full`` stores as ``theta``.
speed      distance between consecutive ego positions over the 0.1 s frame
           period; smooth 8.16 -> 8.35 m/s across the probed window.
target_point
           the route point ~25.4 m ahead, the constant lookahead the
           CARLA-style recordings use (measured 25.39-25.49 m over three
           sampled frames), expressed in the ego frame by the same rotation
           :mod:`datasets.morai_perception_dataset` applies.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .lidar import lidar_to_raw_features

CAMERA_FOLDERS = {"rgb_front": "CAMERA_1", "rgb_left": "CAMERA_2",
                  "rgb_right": "CAMERA_3", "rgb_rear": "CAMERA_4"}
LIDAR_FOLDER = "LIDAR_5"
GPS_FOLDER = "GPS_6"
IMU_FOLDER = "IMU_7"
FRAME_PERIOD_S = 0.1
# The CARLA-style recordings place x_command/y_command this far along the route.
TARGET_LOOKAHEAD_M = 25.4
# Korea UTM; the GPS files carry the matching East/North offsets per frame.
UTM_EPSG = "EPSG:32652"


def _read_key_values(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
        if " : " in line:
            key, value = line.split(" : ", 1)
            values[key.strip()] = value.strip()
    return values


def _yaw_from_quaternion(imu):
    x = float(imu["Orientation X"])
    y = float(imu["Orientation Y"])
    z = float(imu["Orientation Z"])
    w = float(imu["Orientation W"])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _local_rotation(theta):
    """The waypoint-head convention of datasets.morai_perception_dataset."""
    return np.array([
        [np.cos(np.pi / 2 + theta), -np.sin(np.pi / 2 + theta)],
        [np.sin(np.pi / 2 + theta), np.cos(np.pi / 2 + theta)],
    ], dtype=np.float64)


class MoraiNativeCaptureDataset(Dataset):
    """Frames of a native capture, in the encoder's input contract."""

    def __init__(self, root, route_csv=None, lidar_axis_policy="morai_flu_to_lmdrive_fru",
                 sample_stride=1, max_frames=0):
        self.root = Path(root).expanduser().resolve()
        if lidar_axis_policy not in ("morai_flu_to_lmdrive_fru", "native_passthrough"):
            raise ValueError("Unsupported LiDAR axis policy: {}".format(lidar_axis_policy))
        self.lidar_axis_policy = lidar_axis_policy

        missing = [name for name in list(CAMERA_FOLDERS.values()) + [LIDAR_FOLDER, GPS_FOLDER, IMU_FOLDER]
                   if not (self.root / name).is_dir()]
        if missing:
            raise FileNotFoundError("Native capture {} is missing {}".format(
                self.root, ", ".join(missing)))

        stamps = sorted(path.name[: -len("_Intensity.png")]
                        for path in (self.root / CAMERA_FOLDERS["rgb_front"]).glob("*_Intensity.png"))
        complete = [stamp for stamp in stamps if self._is_complete(stamp)]
        if not complete:
            raise ValueError("No frame of {} has every sensor present".format(self.root))
        self.stamps = complete[:: int(sample_stride)]
        if max_frames:
            self.stamps = self.stamps[:max_frames]
        self.all_stamps = complete
        self._index_of = {stamp: position for position, stamp in enumerate(complete)}

        self.route = None
        if route_csv:
            table = np.loadtxt(str(route_csv), delimiter=",", skiprows=1)
            self.route = np.asarray(table[:, 1:3], dtype=np.float64)
            steps = np.hypot(*np.diff(self.route, axis=0).T)
            self.route_arc = np.concatenate([[0.0], np.cumsum(steps)])

        # route_frames mirrors MoraiPerceptionDataset so callers can index alike.
        self.route_frames = [(self.root.name, position)
                             for position, _ in enumerate(self.stamps)]
        self.traffic_label_provenance = None
        self.rgb_transform = None
        self.rgb_center_transform = None
        self.multi_view_transform = None

    def _is_complete(self, stamp):
        for folder in CAMERA_FOLDERS.values():
            if not (self.root / folder / (stamp + "_Intensity.png")).is_file():
                return False
        return ((self.root / LIDAR_FOLDER / (stamp + "_Intensity.bin")).is_file()
                and (self.root / GPS_FOLDER / (stamp + ".txt")).is_file()
                and (self.root / IMU_FOLDER / (stamp + ".txt")).is_file())

    def __len__(self):
        return len(self.stamps)

    def _pose(self, stamp):
        import pyproj

        gps = _read_key_values(self.root / GPS_FOLDER / (stamp + ".txt"))
        imu = _read_key_values(self.root / IMU_FOLDER / (stamp + ".txt"))
        if not hasattr(self, "_utm"):
            self._utm = pyproj.Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
        east, north = self._utm.transform(float(gps["Longitude"]), float(gps["Latitude"]))
        x = east - float(gps["EastOffset"])
        y = north - float(gps["NorthOffset"])
        return x, y, _yaw_from_quaternion(imu)

    def _speed(self, stamp):
        """Distance to the previous complete frame over the frame period."""
        position = self._index_of[stamp]
        if position == 0:
            position = 1
        previous = self.all_stamps[position - 1]
        x0, y0, _ = self._pose(previous)
        x1, y1, _ = self._pose(self.all_stamps[position])
        return float(math.hypot(x1 - x0, y1 - y0) / FRAME_PERIOD_S)

    def _target_point(self, x, y, theta):
        """The route point TARGET_LOOKAHEAD_M ahead, in the ego frame."""
        if self.route is None:
            return np.zeros(2, dtype=np.float32)
        nearest = int(np.argmin(np.hypot(self.route[:, 0] - x, self.route[:, 1] - y)))
        wanted = self.route_arc[nearest] + TARGET_LOOKAHEAD_M
        ahead = int(np.searchsorted(self.route_arc, wanted))
        ahead = min(ahead, len(self.route) - 1)
        delta = np.array([self.route[ahead, 0] - x, self.route[ahead, 1] - y], dtype=np.float64)
        return _local_rotation(theta).T.dot(delta).astype(np.float32)

    @staticmethod
    def _read_image(path):
        with Image.open(str(path)) as image:
            image.load()
            return image.convert("RGB")

    def __getitem__(self, index):
        stamp = self.stamps[index]
        if self.rgb_transform is None or self.rgb_center_transform is None:
            raise RuntimeError("Attach the camera transforms first, e.g. with "
                               "data.src.loaders.loader.attach_camera_transforms")
        if self.multi_view_transform is None:
            raise RuntimeError("Multi-view transform is not configured")

        images = {key: self._read_image(self.root / folder / (stamp + "_Intensity.png"))
                  for key, folder in CAMERA_FOLDERS.items()}

        lidar = np.fromfile(str(self.root / LIDAR_FOLDER / (stamp + "_Intensity.bin")),
                            dtype=np.float32)
        if lidar.size % 4:
            raise ValueError("LiDAR bin is not a multiple of 4 floats: {}".format(stamp))
        lidar = lidar.reshape(-1, 4).copy()
        if not np.isfinite(lidar).all():
            raise ValueError("Non-finite LiDAR at {}".format(stamp))
        if self.lidar_axis_policy == "morai_flu_to_lmdrive_fru":
            lidar[:, 1] *= -1.0
        # Deterministic sampling: an inference frame should render the same twice.
        lidar, num_points = lidar_to_raw_features(
            lidar, rotation_deg=0, clip_abs=100, rng=np.random.RandomState(index))

        x, y, theta = self._pose(stamp)
        inputs = {
            "rgb_front": self.rgb_transform(images["rgb_front"]),
            "rgb_center": self.rgb_center_transform(images["rgb_front"]),
            "rgb_left": self.multi_view_transform(images["rgb_left"]),
            "rgb_right": self.multi_view_transform(images["rgb_right"]),
            "rgb_rear": self.multi_view_transform(images["rgb_rear"]),
            "lidar": lidar,
            "num_points": int(num_points),
            "velocity": torch.tensor([self._speed(stamp)], dtype=torch.float32),
            "target_point": torch.from_numpy(self._target_point(x, y, theta)),
        }
        targets = {
            "frame_id": torch.tensor(index, dtype=torch.long),
            "stamp": stamp,
            "ego_x": torch.tensor(x, dtype=torch.float64),
            "ego_y": torch.tensor(y, dtype=torch.float64),
            "theta": torch.tensor(theta, dtype=torch.float64),
        }
        return inputs, targets

    def describe(self):
        return {
            "format": "morai_native_capture",
            "root": str(self.root),
            "frames_complete": len(self.all_stamps),
            "frames_selected": len(self.stamps),
            "lidar_axis_policy": self.lidar_axis_policy,
            "route_points": 0 if self.route is None else len(self.route),
            "target_lookahead_m": TARGET_LOOKAHEAD_M if self.route is not None else None,
        }
