"""Turn the live MORAI UDP streams into one stage-1 encoder input frame.

The encoder wants the nine tensors of its input contract; this assembles them
from the newest sample of each stream, using the same preprocessing the offline
datasets use so a live frame and a recorded one are interchangeable:

    rgb_front    [1,3,224,224]   camera front, datasets.transforms
    rgb_center   [1,3,128,128]   un-resized centre crop of the front image
    rgb_left/right/rear [1,3,128,128]
    lidar        [1,40000,4]     datasets.lidar.lidar_to_raw_features
    num_points   [1]
    velocity     [1,1]           Competition Vehicle Status vel_x
    target_point [1,2]           route point ~25.4 m ahead, in the ego frame

Competition Vehicle Status does not carry pos_x/pos_y (rule 2-8 withholds them,
and a live packet confirms they arrive as 0.0), so ego position comes from the
GPS fix and heading from the status yaw, exactly as
:mod:`datasets.morai_native_capture` does for a recorded capture.

Nothing here transmits: every socket is receive-only.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.lidar import lidar_to_raw_features
from datasets.transforms import create_camera_transform
from inference.morai_udp import (
    CameraReceiver,
    GpsReceiver,
    ImuReceiver,
    LidarReceiver,
    VehicleStatusReceiver,
)

# Matches datasets/morai_native_capture.py, checked against the recorded captures.
TARGET_LOOKAHEAD_M = 25.4
UTM_EPSG = "EPSG:32652"
CAMERA_KEYS = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")
# The BEV figure's window, from utils.vision_encoder_visualization.BEV_EXTENT.
BEV_FORWARD_M = (-20.0, 30.0)
BEV_RIGHT_M = (-25.0, 25.0)
# Enough to read the shape of a wall or a car, few enough to scatter cheaply.
BEV_CLOUD_POINTS = 6000


def _bev_cloud(points, rng=np.random.RandomState(0)):
    """Points inside the BEV figure's window, as (forward, right) metres.

    Only for the overlay on the detection figure, never for the encoder: it is
    thinned to keep the scatter cheap, so it is a picture of the sweep and not
    the sweep itself.
    """
    forward, right = points[:, 0], points[:, 1]
    inside = ((forward > BEV_FORWARD_M[0]) & (forward < BEV_FORWARD_M[1]) &
              (right > BEV_RIGHT_M[0]) & (right < BEV_RIGHT_M[1]))
    cloud = np.stack([forward[inside], right[inside]], axis=1)
    if len(cloud) > BEV_CLOUD_POINTS:
        cloud = cloud[rng.choice(len(cloud), BEV_CLOUD_POINTS, replace=False)]
    return cloud.astype(np.float32)


def _local_rotation(theta):
    return np.array([
        [np.cos(np.pi / 2 + theta), -np.sin(np.pi / 2 + theta)],
        [np.sin(np.pi / 2 + theta), np.cos(np.pi / 2 + theta)],
    ], dtype=np.float64)


class LiveFrameSource:
    """Receivers for every stream, plus the assembly into encoder inputs."""

    def __init__(self, ports, route_csv=None, lidar_elevations=None,
                 lidar_axis_policy="morai_flu_to_lmdrive_fru", bind="0.0.0.0"):
        self.ports = dict(ports)
        self.lidar_axis_policy = lidar_axis_policy
        self.cameras = {
            key: CameraReceiver(self.ports[key], bind=bind, name=key)
            for key in CAMERA_KEYS if key in self.ports
        }
        self.lidar = LidarReceiver(self.ports["lidar"], bind=bind,
                                   elevations=lidar_elevations, name="lidar")
        self.status = VehicleStatusReceiver(self.ports["vehicle_status"], bind=bind,
                                            name="vehicle_status")
        self.gps = GpsReceiver(self.ports["gps"], bind=bind, name="gps") if "gps" in self.ports else None
        self.imu = ImuReceiver(self.ports["imu"], bind=bind, name="imu") if "imu" in self.ports else None

        # Evaluation transforms: no augmentation, matching the offline loader.
        self.front_transform = create_camera_transform((3, 224, 224), is_training=False)
        self.centre_transform = create_camera_transform(128, is_training=False, need_scale=False)
        self.side_transform = create_camera_transform((3, 128, 128), is_training=False)

        self.route = None
        if route_csv:
            table = np.loadtxt(str(route_csv), delimiter=",", skiprows=1)
            self.route = np.asarray(table[:, 1:3], dtype=np.float64)
            steps = np.hypot(*np.diff(self.route, axis=0).T)
            self.route_arc = np.concatenate([[0.0], np.cumsum(steps)])
        self._utm = None
        self.frames = 0

    # ------------------------------------------------------------------ status

    def stream_report(self):
        """Per-stream packet/sample counts and staleness, for the console."""
        rows = []
        for name, receiver in self._receivers():
            _, age = receiver.latest()
            rows.append((name, receiver.port, receiver.packets, receiver.samples, age))
        return rows

    def _receivers(self):
        for key, receiver in self.cameras.items():
            yield key, receiver
        yield "lidar", self.lidar
        yield "vehicle_status", self.status
        if self.gps:
            yield "gps", self.gps
        if self.imu:
            yield "imu", self.imu

    def wait_ready(self, timeout=20.0, required=None):
        """Block until every required stream has produced one sample."""
        required = required or [name for name, _ in self._receivers()]
        deadline = time.time() + timeout
        while time.time() < deadline:
            missing = [name for name, receiver in self._receivers()
                       if name in required and receiver.latest()[0] is None]
            if not missing:
                return True, []
            time.sleep(0.1)
        return False, [name for name, receiver in self._receivers()
                       if name in required and receiver.latest()[0] is None]

    # ------------------------------------------------------------------- pose

    def _ego_xy(self, fix):
        import pyproj

        if self._utm is None:
            self._utm = pyproj.Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
        east, north = self._utm.transform(fix["longitude"], fix["latitude"])
        return east, north

    def _target_point(self, east, north, theta, offsets):
        """Route point TARGET_LOOKAHEAD_M ahead, expressed in the ego frame."""
        if self.route is None:
            return np.zeros(2, dtype=np.float32)
        x, y = east - offsets[0], north - offsets[1]
        nearest = int(np.argmin(np.hypot(self.route[:, 0] - x, self.route[:, 1] - y)))
        ahead = int(np.searchsorted(self.route_arc, self.route_arc[nearest] + TARGET_LOOKAHEAD_M))
        ahead = min(ahead, len(self.route) - 1)
        delta = np.array([self.route[ahead, 0] - x, self.route[ahead, 1] - y], dtype=np.float64)
        return _local_rotation(theta).T.dot(delta).astype(np.float32)

    # ------------------------------------------------------------------ frame

    def build(self, device, utm_offsets=(0.0, 0.0), max_age=1.0):
        """Newest sample of every stream -> (inputs, meta), or (None, meta) if stale."""
        import cv2

        meta = {"ages": {}, "missing": []}
        images = {}
        for key, receiver in self.cameras.items():
            payload, age = receiver.latest()
            meta["ages"][key] = age
            if payload is None or age > max_age:
                meta["missing"].append(key)
                continue
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                meta["missing"].append(key + "(undecodable)")
                continue
            images[key] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        sweep, lidar_age = self.lidar.latest()
        meta["ages"]["lidar"] = lidar_age
        if sweep is None or lidar_age > max_age:
            meta["missing"].append("lidar")

        status, status_age = self.status.latest()
        meta["ages"]["vehicle_status"] = status_age
        if status is None or status_age > max_age:
            meta["missing"].append("vehicle_status")

        if meta["missing"]:
            return None, meta

        from PIL import Image

        pil = {key: Image.fromarray(value) for key, value in images.items()}
        points = np.asarray(sweep, dtype=np.float32)   # already an array from the receiver
        if self.lidar_axis_policy == "morai_flu_to_lmdrive_fru":
            points = points.copy()
            points[:, 1] *= -1.0
        meta["lidar_xy"] = _bev_cloud(points)
        lidar, num_points = lidar_to_raw_features(points, rotation_deg=0, clip_abs=100,
                                                  rng=np.random.RandomState(0))

        theta = status["yaw"]
        meta["speed"] = status["speed"]
        meta["yaw_degrees"] = status["yaw_degrees"]
        target_point = np.zeros(2, dtype=np.float32)
        fix, gps_age = (self.gps.latest() if self.gps else (None, float("inf")))
        meta["ages"]["gps"] = gps_age
        if fix is not None and gps_age <= max_age and self.route is not None:
            east, north = self._ego_xy(fix)
            target_point = self._target_point(east, north, theta, utm_offsets)
            meta["ego_local"] = (east - utm_offsets[0], north - utm_offsets[1])
        meta["target_point"] = target_point.tolist()
        meta["lidar_points"] = int(num_points)

        inputs = {
            "rgb_front": self.front_transform(pil["rgb_front"]).unsqueeze(0),
            "rgb_center": self.centre_transform(pil["rgb_front"]).unsqueeze(0),
            "rgb_left": self.side_transform(pil["rgb_left"]).unsqueeze(0),
            "rgb_right": self.side_transform(pil["rgb_right"]).unsqueeze(0),
            "rgb_rear": self.side_transform(pil["rgb_rear"]).unsqueeze(0),
            "lidar": torch.from_numpy(lidar).unsqueeze(0),
            "num_points": torch.tensor([int(num_points)]),
            "velocity": torch.tensor([[float(status["speed"])]], dtype=torch.float32),
            "target_point": torch.from_numpy(target_point).unsqueeze(0),
        }
        self.frames += 1
        meta["front_image"] = images["rgb_front"]
        return {key: value.to(device) for key, value in inputs.items()}, meta

    def close(self):
        for _, receiver in self._receivers():
            receiver.close()
