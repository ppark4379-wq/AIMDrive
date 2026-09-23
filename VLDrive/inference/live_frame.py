"""Turn the live MORAI UDP streams into one stage-1 encoder input frame.

SensorRecord checkpoints use the training MoraiLMDriveVisionAdapter directly.
Raw UDP LiDAR origin and the velocity clock must be declared explicitly; latest
packets are not a proof of sensor synchrony. Missing route/GPS never fabricates
a zero navigation target. Legacy constant-zero model inputs remain supported
only when the supplied versioned checkpoint contract declares them.

Nothing here transmits: every socket is receive-only.

For native Stage-2 caches, pass encoder_recipe explicitly. That opt-in delegates
all image/LiDAR transforms to MoraiLMDriveVisionAdapter and uses the checkpoint's
constant-zero velocity recipe. The legacy stage-1 display path stays unchanged.
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

from datasets.morai_encoder_contract import LiveInputAdapter, world_target_to_local
from inference.morai_route import official_route_tracker
from inference.morai_udp import (
    CameraReceiver,
    GpsReceiver,
    ImuReceiver,
    LidarReceiver,
    VehicleStatusReceiver,
)

# SensorRecord manifest target distance.
TARGET_LOOKAHEAD_M = 25.0
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


class LiveFrameSource:
    """Receivers for every stream, plus the assembly into encoder inputs."""

    def __init__(self, ports, route_csv=None, lidar_elevations=None,
                 lidar_axis_policy="morai_flu_to_lmdrive_fru", bind="0.0.0.0",
                 encoder_recipe=None, open_receivers=True,
                 encoder_contract=None, lidar_frame=None, lidar_to_base=None,
                 velocity_source=None, allow_velocity_mismatch=False,
                 route_map_root=None, route_definition=None):
        self.ports = dict(ports)
        self.lidar_axis_policy = lidar_axis_policy
        self.adapter = None
        self.input_adapter = None
        self.contract = None
        self.route_tracker = None
        self.velocity_source = velocity_source
        self.velocity_mismatch = False
        if encoder_recipe is not None:
            if encoder_contract is not None:
                raise ValueError('Choose encoder_recipe or encoder_contract, not both')
            from datasets.morai_transfer_adapter import MoraiLMDriveVisionAdapter, MoraiVisionAdapterConfig
            if (encoder_recipe.get('schema') != 'morai_encoder_recipe_v1'
                    or encoder_recipe.get('feature_shape') != [106, 256]
                    or encoder_recipe.get('token_order') != 'bev100_light1_waypoint5'
                    or encoder_recipe.get('velocity_input') != 'constant_zero'
                    or encoder_recipe.get('target_point_input') != 'constant_zero'
                    or encoder_recipe.get('pillar_decoration') != 'legacy_lmdrive'
                    or encoder_recipe['adapter']['use_previous_lidar_sweep']
                    or encoder_recipe['adapter']['lidar_axis_policy'] != lidar_axis_policy):
                raise ValueError('Unsupported live encoder recipe')
            self.adapter = MoraiLMDriveVisionAdapter(MoraiVisionAdapterConfig(**encoder_recipe['adapter']))
        else:
            self.input_adapter = LiveInputAdapter(encoder_contract or {}, lidar_frame=lidar_frame,
                                                  lidar_to_base=lidar_to_base)
            self.contract = self.input_adapter.contract
            if lidar_axis_policy != self.input_adapter.adapter.config.lidar_axis_policy:
                raise ValueError('Configured LiDAR axis policy differs from the checkpoint contract')
            self.velocity_mismatch = self.contract['velocity_input'] != 'constant_zero'
            if self.velocity_mismatch and not (velocity_source == 'vehicle_status' and allow_velocity_mismatch):
                raise ValueError('This model requires source-time GPS speed. UDP clock parity is unverified; '
                                 'use runtime.velocity_source=vehicle_status with '
                                 'runtime.allow_velocity_mismatch=true only for an explicit diagnostic')
            if self.contract['target_point_input'] != 'constant_zero':
                if not route_csv:
                    raise ValueError('This checkpoint requires an official navigation route')
                self.route_tracker = official_route_tracker(self.contract, route_csv,
                    map_root=route_map_root, route_definition=route_definition)

        self.cameras = {
            key: CameraReceiver(self.ports[key], bind=bind, name=key)
            for key in CAMERA_KEYS if key in self.ports and open_receivers
        }
        self.lidar = LidarReceiver(self.ports["lidar"], bind=bind,
                                   elevations=lidar_elevations, name="lidar") if open_receivers else None
        self.status = VehicleStatusReceiver(self.ports["vehicle_status"], bind=bind,
                                            name="vehicle_status") if open_receivers else None
        self.gps = GpsReceiver(self.ports["gps"], bind=bind, name="gps") if open_receivers and "gps" in self.ports else None
        self.imu = ImuReceiver(self.ports["imu"], bind=bind, name="imu") if open_receivers and "imu" in self.ports else None

        self.route = self.route_tracker.points if self.route_tracker is not None else None
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
        if self.lidar is not None:
            yield "lidar", self.lidar
        if self.status is not None:
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
            raise ValueError("No navigation route")
        x, y = east - offsets[0], north - offsets[1]
        progress = self.route_tracker.update(x, y, theta)
        return world_target_to_local((progress.target_x, progress.target_y), (x, y), theta,
                                     axes=self.contract["target_point_axes"])

    # ------------------------------------------------------------------ frame

    def prepare_encoder_frame(self, images, sweep, *, device='cpu', rng_seed=0):
        """Native cache recipe shared by live UDP and recorded regression frames.

        Images are RGB uint8; sweep is unmodified sensor-local FLU XYZI. The
        existing adapter owns cropping, extrinsics, axis and intensity changes.
        This checkpoint was trained with zero velocity and zero route input.
        """
        if self.adapter is None:
            raise ValueError('An explicit cache encoder_recipe is required')
        if set(images) != set(CAMERA_KEYS):
            raise ValueError('Four camera images are required')
        inputs = self.adapter.camera_tensors(**{
            key[4:] + '_rgb': value for key, value in images.items()})
        points = np.asarray(sweep, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError('Expected finite sensor-local XYZI')
        lidar, count = self.adapter.lidar_to_raw_features(points, rng_seed=rng_seed)
        if count == 0:
            raise ValueError('Empty LiDAR after preprocessing')
        inputs.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(count),
                      velocity=torch.zeros(1))
        return {key: value.unsqueeze(0).to(device) for key, value in inputs.items()}

    def build(self, device, utm_offsets=(0.0, 0.0), max_age=1.0, rng_seed=0):

        """Newest sample of every stream -> (inputs, meta), or (None, meta) if stale."""
        import cv2

        meta = {"ages": {}, "missing": [], "sample_ids": {}, "received_at": {}}
        def read(name, receiver):
            if hasattr(receiver, 'snapshot'):
                payload, age, sequence, stamp = receiver.snapshot()
            else:
                payload, age = receiver.latest()
                sequence, stamp = 0, None
            meta['ages'][name] = age
            meta['sample_ids'][name] = sequence
            meta['received_at'][name] = stamp
            return payload, age
        images = {}
        camera_bytes = {}
        meta['missing'].extend(key for key in CAMERA_KEYS if key not in self.cameras)
        for key, receiver in self.cameras.items():
            payload, age = read(key, receiver)
            meta["ages"][key] = age
            if payload is None or age > max_age:
                meta["missing"].append(key)
                continue
            if getattr(self, 'adapter', None) is not None:
                camera_bytes[key] = payload
                import io
                from PIL import Image
                try:
                    with Image.open(io.BytesIO(payload)) as im:
                        images[key] = np.asarray(im.convert('RGB')).copy()
                except (OSError, ValueError):
                    meta['missing'].append(key + '(undecodable)')
                continue
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                meta["missing"].append(key + "(undecodable)")
                continue
            images[key] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        sweep, lidar_age = read('lidar', self.lidar)
        meta["ages"]["lidar"] = lidar_age
        if sweep is None or lidar_age > max_age:
            meta["missing"].append("lidar")

        status, status_age = read('vehicle_status', self.status)
        meta["ages"]["vehicle_status"] = status_age
        if status is None or status_age > max_age:
            meta["missing"].append("vehicle_status")

        if meta["missing"]:
            return None, meta

        if getattr(self, 'adapter', None) is not None:
            inputs = self.prepare_encoder_frame(images, sweep, device=device,
                                                rng_seed=meta['sample_ids']['lidar'] - 1)
            meta.update(speed=float(status['speed']), yaw_degrees=float(status['yaw_degrees']),
                        lidar_points=int(inputs['num_points'].item()), sim_time=status.get('sim_time'))
            meta['front_jpeg'] = camera_bytes['rgb_front']
            meta['status_snapshot'] = (status, status_age, meta['sample_ids']['vehicle_status'],
                                       meta['received_at']['vehicle_status'])
            meta['gps_snapshot'] = self.gps.snapshot() if self.gps else (None, float('inf'), 0, None)
            self.frames += 1
            return inputs, meta


        theta = status["yaw"]
        meta["speed"] = status["speed"]
        meta["velocity_contract_mismatch"] = self.velocity_mismatch
        meta["velocity_source"] = "constant_zero" if not self.velocity_mismatch else self.velocity_source
        meta["waypoint_output_axes"] = self.contract["waypoint_output_axes"]
        meta["detection_output_axes"] = self.contract["detection_output_axes"]
        meta["yaw_degrees"] = status["yaw_degrees"]
        target_point = np.zeros(2, dtype=np.float32)
        fix, gps_age = (self.gps.latest() if self.gps else (None, float("inf")))
        meta["ages"]["gps"] = gps_age
        if fix is not None and gps_age <= max_age and self.route is not None:
            east, north = self._ego_xy(fix)
            target_point = self._target_point(east, north, theta, utm_offsets)
            meta["ego_local"] = (east - utm_offsets[0], north - utm_offsets[1])
        elif self.contract["target_point_input"] != "constant_zero":
            meta["missing"].append("gps/route (navigation target unavailable)")
            return None, meta
        meta["target_point"] = target_point.tolist()

        inputs = self.input_adapter.assemble(images, sweep, velocity=float(status["speed"]),
                                             target_point=target_point, rng_seed=rng_seed)
        meta["target_point"] = inputs["target_point"].tolist()
        meta["model_velocity"] = float(inputs["velocity"][0])
        meta["lidar_points"] = int(inputs["num_points"])
        meta["lidar_xy"] = _bev_cloud(inputs["lidar"][:meta["lidar_points"]].numpy())
        inputs = {key: value.unsqueeze(0) for key, value in inputs.items()}
        self.frames += 1
        meta["front_image"] = images["rgb_front"]
        return {key: value.to(device) for key, value in inputs.items()}, meta

    def close(self):
        for _, receiver in self._receivers():
            receiver.close()
