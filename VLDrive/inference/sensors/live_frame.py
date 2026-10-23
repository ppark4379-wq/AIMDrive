"""Turn the live MORAI UDP streams into one stage-1 encoder input frame.

SensorRecord checkpoints use the training MoraiLMDriveVisionAdapter directly.
Raw UDP LiDAR origin and the velocity clock must be declared explicitly; latest
packets are not a proof of sensor synchrony. Missing route/GPS never fabricates
a zero navigation target. Legacy constant-zero model inputs remain supported
only when the supplied versioned checkpoint contract declares them.

Nothing here transmits: every socket is receive-only.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.src.loaders.morai_encoder_contract import LiveInputAdapter, world_target_to_local
from data.src.loaders.morai_map_geometry import grid_yaw
from inference.navigation.morai_route import official_route_tracker
from inference.sensors.morai_udp import (
    CameraReceiver,
    GpsReceiver,
    ImuReceiver,
    LidarReceiver,
    VehicleStatusReceiver,
    resolve_lidar_hour_stamp,
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
                 encoder_contract=None, lidar_frame=None, lidar_to_base=None,
                 velocity_source=None, allow_velocity_mismatch=False,
                 route_map_root=None, route_definition=None, orientation_frame=None,
                 perception_only=False, status_receiver=None, pose_source='gps', diagnostics=False,
                 camera_receiver_mode='thread', gps_receiver=None,
                 lidar_frame_policy='morai_frame_timestamp', localization=None):
        if camera_receiver_mode not in ('thread', 'process'):
            raise ValueError('Camera receiver mode must be thread or process')
        self.ports = dict(ports)
        self.perception_only = bool(perception_only)
        self.localization = localization
        if localization is not None and not (perception_only and pose_source == 'gps' and status_receiver is localization):
            raise ValueError('External localization requires perception-only GPS mode with the same status source')
        self.audit_enabled = diagnostics
        self.decode_failures = {key: 0 for key in CAMERA_KEYS}
        if pose_source not in ('gps', 'ego_topic'):
            raise ValueError('Explicit gps or ego_topic pose source required')
        if pose_source == 'ego_topic' and not (
                perception_only and velocity_source == 'ego_topic' and status_receiver is not None):
            raise ValueError('ego_topic pose requires perception-only mode and a ROS Ego receiver')
        self.pose_source = pose_source
        self.input_adapter = LiveInputAdapter(encoder_contract or {}, lidar_frame=lidar_frame,
                                              lidar_to_base=lidar_to_base)
        self.contract = self.input_adapter.contract
        if lidar_axis_policy != self.input_adapter.adapter.config.lidar_axis_policy:
            raise ValueError("Configured LiDAR axis policy differs from the checkpoint contract")
        self.velocity_source = velocity_source
        expected_velocity = self.contract["velocity_input"]
        self.velocity_mismatch = expected_velocity == "causal_gps_source_time"
        if self.velocity_mismatch and not (velocity_source == "vehicle_status" and allow_velocity_mismatch):
            raise ValueError("This checkpoint requires source-time GPS speed; vehicle_status requires an explicit diagnostic override")
        if expected_velocity != "constant_zero" and velocity_source not in ('vehicle_status', 'ego_topic'):
            raise ValueError("Live speed requires an explicit vehicle_status or ego_topic source")
        if velocity_source == 'ego_topic' and status_receiver is None:
            raise ValueError('ego_topic requires an explicit ROS Ego receiver')
        # Ego-topic and competition status both supply planar speed in m/s. This
        # establishes the scalar recipe, not a source-time 50ms clock match.
        self.orientation_frame = orientation_frame
        if self.contract.get("map_basis") is not None and orientation_frame not in ("true_enu", "projected_grid"):
            raise ValueError("Declare runtime.orientation_frame for the checkpoint map basis")
        self.route_tracker = None
        if not self.perception_only and self.contract['target_point_input'] != 'constant_zero':
            if not route_csv:
                raise ValueError("This checkpoint requires an official navigation route")
            self.route_tracker = official_route_tracker(self.contract, route_csv,
                map_root=route_map_root, route_definition=route_definition)
        self.camera_group = None
        if camera_receiver_mode == 'process':
            from inference.sensors.camera_process import CameraProcessGroup
            self.camera_group = CameraProcessGroup(
                {key: self.ports[key] for key in CAMERA_KEYS if key in self.ports},
                bind=bind, diagnostics=diagnostics)
            self.cameras = self.camera_group.cameras
        else:
            self.cameras = {
                key: CameraReceiver(self.ports[key], bind=bind, name=key, diagnostics=diagnostics)
                for key in CAMERA_KEYS if key in self.ports
            }
        self.lidar = self.status = self.gps = self.imu = None
        try:
            self.lidar = LidarReceiver(self.ports["lidar"], bind=bind,
                                       elevations=lidar_elevations, name="lidar", diagnostics=diagnostics,
                                       frame_policy=lidar_frame_policy)
            self.status = (status_receiver if status_receiver is not None else
                           VehicleStatusReceiver(self.ports["vehicle_status"], bind=bind,
                                                 name="vehicle_status"))
            self.gps = (gps_receiver if gps_receiver is not None else
                        GpsReceiver(self.ports["gps"], bind=bind, name="gps") if "gps" in self.ports else None)
            self.imu = ImuReceiver(self.ports["imu"], bind=bind, name="imu") if "imu" in self.ports else None
        except BaseException:
            self.close()
            raise

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

    def diagnostics(self):
        return {name: dict(receiver.diagnostics(),
                           decode_failure_attempts=self.decode_failures.get(name, 0))
                for name, receiver in self._receivers() if hasattr(receiver, 'diagnostics')}

    def _latest_for_build(self, name, receiver, meta):
        if hasattr(receiver, 'latest_snapshot'):
            sample, age, identity = receiver.latest_snapshot()
            meta['selected_samples'][name] = identity
            return sample, age
        return receiver.latest()

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
        basis = self.contract.get("map_basis")
        if basis is None:
            progress = self.route_tracker.update(x, y, theta)
            return world_target_to_local((progress.target_x, progress.target_y), (x, y), theta,
                                         axes=self.contract["target_point_axes"])
        ego_xyz = (x, y, 0.)
        orientation = getattr(self, "orientation_frame", None)
        corrected_yaw = grid_yaw(theta, ego_xyz, basis, source_frame=orientation)
        progress = self.route_tracker.update(x, y, corrected_yaw)
        return self.input_adapter.world_target((progress.target_x, progress.target_y), ego_xyz,
                                               theta, orientation_frame=orientation)

    # ------------------------------------------------------------------ frame

    def build(self, device, utm_offsets=(0.0, 0.0), max_age=1.0, rng_seed=0,
              after_lidar_sample_id=None):
        """Newest sample of every stream -> (inputs, meta), or (None, meta) if stale."""
        import cv2

        begin_ns = time.monotonic_ns()
        meta = {"ages": {}, "missing": [], "selected_samples": {}, "stage_ms": {}}
        if after_lidar_sample_id is not None:
            # Pin the exact sweep before expensive decoding/preprocessing.
            # Its identity cannot change when the receiver finishes another sweep.
            sweep, lidar_age = self._latest_for_build('lidar', self.lidar, meta)
            meta['ages']['lidar'] = lidar_age
            if sweep is None or lidar_age > max_age:
                meta['missing'].append('lidar')
                return None, meta
            identity = meta['selected_samples'].get('lidar')
            if identity is None:
                raise ValueError('LiDAR gating requires an atomic sample identity')
            if identity['sample_id'] <= after_lidar_sample_id:
                meta['waiting_for_new_lidar'] = True
                return None, meta
        images = {}
        for key, receiver in self.cameras.items():
            payload, age = self._latest_for_build(key, receiver, meta)
            meta["ages"][key] = age
            if payload is None or age > max_age:
                meta["missing"].append(key)
                continue
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                self.decode_failures[key] += 1
                meta["missing"].append(key + "(undecodable)")
                continue
            images[key] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        meta['stage_ms']['camera_decode'] = (time.monotonic_ns() - begin_ns) / 1e6

        if after_lidar_sample_id is None:
            sweep, lidar_age = self._latest_for_build('lidar', self.lidar, meta)
        meta["ages"]["lidar"] = lidar_age
        if sweep is None or lidar_age > max_age:
            meta["missing"].append("lidar")

        status, status_age = self._latest_for_build('vehicle_status', self.status, meta)
        meta["ages"]["vehicle_status"] = status_age
        if status is None or status_age > max_age:
            meta["missing"].append("vehicle_status")

        lidar_identity = meta['selected_samples'].get('lidar', {})
        if 'source_stamp_us_within_hour' in lidar_identity:
            # The packet clock is modulo one hour. Resolve it only against a
            # real source header; receipt time is not a substitute for capture.
            reference = meta['selected_samples'].get('vehicle_status', {}).get('source_stamp_ns', 0)
            if not reference:
                reference = meta['selected_samples'].get('rgb_front', {}).get('source_stamp_ns', 0)
            try:
                lidar_identity['source_stamp_ns'] = resolve_lidar_hour_stamp(
                    lidar_identity['source_stamp_us_within_hour'], reference)
                if hasattr(self.status, 'at_source_stamp'):
                    status, status_age, identity = self.status.at_source_stamp(lidar_identity['source_stamp_ns'])
                    if status is None or status_age > max_age:
                        meta['missing'].append('vehicle_status_source_alignment')
                    else:
                        meta['selected_samples']['vehicle_status'] = identity
                        meta['ages']['vehicle_status'] = status_age
                        meta['ego_alignment'] = 'causal_lidar_source_50ms'
            except ValueError:
                meta['missing'].append('lidar_source_clock')

        if meta["missing"]:
            return None, meta

        source_stamps = [s['source_stamp_ns'] for s in meta['selected_samples'].values()
                         if s.get('source_stamp_ns', 0) > 0]
        meta['source_spread_ms'] = ((max(source_stamps)-min(source_stamps))/1e6
                                    if source_stamps else None)

        theta = status["yaw"]
        meta["speed"] = status["speed"]
        meta['ego_control'] = {k: status[k] for k in ('accel', 'brake', 'steer', 'gear') if k in status}
        meta["velocity_contract_mismatch"] = self.velocity_mismatch
        meta["velocity_source"] = "constant_zero" if self.contract["velocity_input"] == "constant_zero" else self.velocity_source
        meta["waypoint_output_axes"] = self.contract["waypoint_output_axes"]
        meta["detection_output_axes"] = self.contract["detection_output_axes"]
        meta["yaw_degrees"] = status["yaw_degrees"]
        target_point = np.zeros(2, dtype=np.float32)
        fix, gps_age = (self._latest_for_build('gps', self.gps, meta) if self.gps else (None, float("inf")))
        meta["ages"]["gps"] = gps_age
        perception_only = getattr(self, 'perception_only', False)
        if perception_only:
            if self.pose_source == 'ego_topic':
                # Use the same map position as the original Expert, including
                # GPS-denied tunnels. Never fall back to a different pose source.
                position = np.asarray(status.get('ego_local'), dtype=float)
                if position.shape != (2,) or not np.isfinite(position).all():
                    meta['missing'].append('ego_topic (world pose unavailable)')
                    return None, meta
                meta['ego_local'] = tuple(position)
            else:
                if fix is None or gps_age > max_age:
                    meta['missing'].append('gps (world pose unavailable)')
                    return None, meta
                if getattr(self, 'localization', None) is not None:
                    pose_packet = self.localization.pose_packet(meta['selected_samples'])
                    if not pose_packet['ready']:
                        meta['missing'].append(pose_packet['reason'])
                        return None, meta
                    meta['localization_pose'] = pose_packet
                    pose = pose_packet['ego_pose']
                    meta['ego_local'] = (pose['x'],pose['y'])
                    meta['pose_frame'] = 'enu'
                else:
                    east, north = self._ego_xy(fix)
                    meta['ego_local'] = (east - utm_offsets[0], north - utm_offsets[1])
            meta['pose_source'] = self.pose_source
        elif fix is not None and gps_age <= max_age and self.route is not None:
            east, north = self._ego_xy(fix)
            target_point = self._target_point(east, north, theta, utm_offsets)
            meta["ego_local"] = (east - utm_offsets[0], north - utm_offsets[1])
        elif self.contract["target_point_input"] != "constant_zero":
            meta["missing"].append("gps/route (navigation target unavailable)")
            return None, meta
        meta["target_point"] = target_point.tolist()

        assemble_ns = time.monotonic_ns()
        inputs = self.input_adapter.assemble(images, sweep, velocity=float(status["speed"]),
                                             target_point=target_point, rng_seed=rng_seed)
        meta['stage_ms']['adapter_assemble'] = (time.monotonic_ns() - assemble_ns) / 1e6
        if perception_only:
            # The shared training adapter allocates a target tensor. Remove it
            # before inference: no fabricated navigation target reaches a model.
            inputs.pop('target_point')
            meta['target_point'] = None
            meta['perception_only'] = True
        else:
            meta["target_point"] = inputs["target_point"].tolist()
        meta["model_velocity"] = float(inputs["velocity"][0])
        meta["lidar_points"] = int(inputs["num_points"])
        meta["lidar_xy"] = _bev_cloud(inputs["lidar"][:meta["lidar_points"]].numpy())
        inputs = {key: value.unsqueeze(0) for key, value in inputs.items()}
        self.frames += 1
        meta["front_image"] = images["rgb_front"]
        transfer_ns = time.monotonic_ns()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        meta['stage_ms']['device_transfer_call'] = (time.monotonic_ns() - transfer_ns) / 1e6
        meta['stage_ms']['build_total'] = (time.monotonic_ns() - begin_ns) / 1e6
        return inputs, meta

    def close(self):
        try:
            for _, receiver in self._receivers():
                if receiver is not None:
                    receiver.close()
        finally:
            if getattr(self, 'camera_group', None) is not None:
                self.camera_group.close()
