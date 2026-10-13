"""Stamped ROS camera/LiDAR/Ego inputs for obstacle-only VE inference.

This module never subscribes to simulator object or signal GT. A missing,
stale or unpaired observation is not converted into an empty road. Source
timestamp proximity is checked explicitly; physical simultaneity is unproven.
"""
from collections import deque
import math
import threading
import time

import numpy as np


CAMERAS = {'Camera-1': 'rgb_front', 'Camera-2': 'rgb_left',
           'Camera-3': 'rgb_right', 'Camera-4': 'rgb_rear'}
POINT_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
               5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def source_stamp_ns(message):
    stamp = message.header.stamp
    value = int(stamp.to_nsec()) if hasattr(stamp, 'to_nsec') else (
        int(stamp.secs) * 10**9 + int(stamp.nsecs))
    if value <= 0:
        raise ValueError('A nonzero sensor source timestamp is required')
    return value


def pointcloud_xyzi(message):
    """Honor PointCloud2 field offsets, byte order, padding and row stride."""
    width, height, step, row_step = map(int, (
        message.width, message.height, message.point_step, message.row_step))
    if width <= 0 or height <= 0 or step <= 0 or row_step < width * step:
        raise ValueError('Invalid PointCloud2 dimensions/stride')
    fields = {field.name: field for field in message.fields}
    if not all(name in fields for name in ('x', 'y', 'z', 'intensity')):
        raise ValueError('PointCloud2 requires explicit x/y/z/intensity fields')
    names, formats, offsets = [], [], []
    prefix = '>' if message.is_bigendian else '<'
    for name in ('x', 'y', 'z', 'intensity'):
        field = fields[name]
        if int(field.count) != 1 or int(field.datatype) not in POINT_TYPES:
            raise ValueError('Unsupported PointCloud2 field: ' + name)
        kind = prefix + POINT_TYPES[int(field.datatype)]
        offset = int(field.offset)
        if offset < 0 or offset + np.dtype(kind).itemsize > step:
            raise ValueError('PointCloud2 field escapes its point stride')
        names.append(name); formats.append(kind); offsets.append(offset)
    payload = bytes(message.data)
    if len(payload) != row_step * height:
        raise ValueError('Truncated or overlong PointCloud2 payload')
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=step))
    records = np.ndarray((height, width), dtype=dtype, buffer=payload,
                         strides=(row_step, step))
    points = np.stack([records[name].ravel() for name in names], axis=1).astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError('PointCloud2 has no finite XYZI samples')
    if points[:, 3].min() < 0 or points[:, 3].max() > 255:
        raise ValueError('ROS LiDAR intensity is outside the declared reflectance byte range')
    return points


class StampedPerceptionBuffer:
    """Pure pairing policy, independent of ROS, CUDA or simulator operations."""
    def __init__(self, max_age_s=.35, slop_ns=50_000_000):
        if not math.isfinite(max_age_s) or max_age_s <= 0 or not 0 < slop_ns <= 50_000_000:
            raise ValueError('Use a positive receipt age and at most 50ms sensor skew')
        self.max_age_ns = int(max_age_s * 10**9)
        self.slop_ns = int(slop_ns)
        self.rows = {key: deque(maxlen=12 if key != 'ego' else 128)
                     for key in tuple(CAMERAS.values()) + ('lidar', 'ego')}
        self.lock = threading.Lock()
        self.last_emitted_stamp = None

    def push(self, role, stamp_ns, value, received_ns=None):
        if role not in self.rows or not isinstance(stamp_ns, int) or stamp_ns <= 0:
            raise ValueError('Unknown role or missing sensor source timestamp')
        received_ns = time.monotonic_ns() if received_ns is None else int(received_ns)
        with self.lock:
            history = self.rows[role]
            if history and stamp_ns < history[-1]['stamp_ns']:
                # A simulation reset invalidates every old cross-stream pair.
                for samples in self.rows.values():
                    samples.clear()
                self.last_emitted_stamp = None
            self.rows[role].append(dict(stamp_ns=stamp_ns, received_ns=received_ns, value=value))

    def paired(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self.lock:
            missing = [role for role, rows in self.rows.items() if not rows]
            if missing:
                return None, dict(ready=False, invalid_reason='missing:' + ','.join(missing))
            failure = None
            # Camera callbacks arrive separately. Reuse the most recent complete
            # pair while a newer frame is incomplete, within the same age limit.
            # This avoids interpreting transport order as a perception failure.
            for front in reversed(self.rows['rgb_front']):
                if self.last_emitted_stamp is not None and front['stamp_ns'] < self.last_emitted_stamp:
                    continue
                selected, meta = self._pair_anchor(front, now_ns)
                if not meta['ready']:
                    if failure is None:
                        failure = meta
                    continue
                anchor = front['stamp_ns']
                if self.last_emitted_stamp == anchor:
                    meta['waiting_for_new_frame'] = True
                    return None, meta
                self.last_emitted_stamp = anchor
                return selected, meta
            return None, failure or dict(ready=False, invalid_reason='no_fresh_complete_pair')

    def _pair_anchor(self, front, now_ns):
        anchor = front['stamp_ns']
        selected = {'rgb_front': front}
        for role in tuple(CAMERAS.values())[1:] + ('lidar',):
            selected[role] = min(self.rows[role], key=lambda row: abs(row['stamp_ns'] - anchor))
            if abs(selected[role]['stamp_ns'] - anchor) > self.slop_ns:
                return None, dict(ready=False, invalid_reason='unpaired_source_time:' + role,
                                  stamp_ns=anchor)
        past = [row for row in self.rows['ego'] if row['stamp_ns'] <= anchor]
        if not past:
            return None, dict(ready=False, invalid_reason='ego_no_preceding_sample', stamp_ns=anchor)
        ego = max(past, key=lambda row: row['stamp_ns'])
        if anchor - ego['stamp_ns'] > 50_000_000:
            return None, dict(ready=False, invalid_reason='ego_older_than_50ms', stamp_ns=anchor)
        same_time = [row['value'] for row in past if row['stamp_ns'] == ego['stamp_ns']]
        if any(value != ego['value'] for value in same_time):
            return None, dict(ready=False, invalid_reason='ego_conflicting_duplicate', stamp_ns=anchor)
        selected['ego'] = ego
        ages = {role: (now_ns - row['received_ns']) / 1e9 for role, row in selected.items()}
        if any(age < 0 or age * 1e9 > self.max_age_ns for age in ages.values()):
            return None, dict(ready=False, invalid_reason='stale_sensor_receipt',
                              stamp_ns=anchor, ages_s=ages)
        stamps = {role: row['stamp_ns'] for role, row in selected.items()}
        meta = dict(ready=True, invalid_reason=None, stamp_ns=anchor,
                    ages_s=ages, source_stamps_ns=stamps,
                    sensor_received_monotonic_ns=min(row['received_ns'] for row in selected.values()),
                    matched_sensor_skew_ns=max(abs(stamp - anchor) for stamp in stamps.values()),
                    physical_simultaneity_verified=False)
        return selected, meta


def forward_perception(model, inputs):
    """Same trained feature/decoder/heads; the waypoint branch is unused."""
    import torch
    if model.return_feature:
        raise ValueError('Perception needs the trained prediction heads enabled')
    velocity = inputs['velocity'].view(1, -1, 1)
    features, lidar_token = model.forward_features(
        inputs['rgb_front'], inputs['rgb_left'], inputs['rgb_right'], inputs['rgb_rear'],
        inputs['rgb_center'], inputs['lidar'], inputs['num_points'])
    batch = inputs['rgb_front'].shape[0]
    target = model.position_encoding(torch.ones((batch, 1, 50, 50), device=inputs['rgb_front'].device))
    target = torch.cat([target.flatten(2), model.query_pos_embed.repeat(batch, 1, 1)], 2).permute(2, 0, 1)
    memory = model.encoder(features, mask=model.attn_mask)
    query_embed = model.query_embed.repeat(1, batch, 1) + model.velocity_fc(velocity).repeat(6, 1, 1)
    query = torch.cat([lidar_token, query_embed], 0)
    hidden = model.decoder(query, memory, query_pos=target)[0].permute(1, 0, 2)
    traffic = model.traffic_pred_head(torch.cat([
        hidden[:, :2500], velocity.view(-1, 1, 1).repeat(1, 2500, 32)], dim=2))
    light = model.traffic_light_pred_head(hidden[:, 2500])
    return traffic, light


class RosPerceptionSource:
    def __init__(self, contract, lidar_frame='sensor_flu', lidar_to_base=None,
                 max_age_s=.35, slop_ns=50_000_000,
                 expected_lidar_frame_id='Lidar3D-5'):
        from data.src.loaders.morai_encoder_contract import LiveInputAdapter
        if contract.get('velocity_input') != 'causal_ego_topic_speed_mps_50ms':
            raise ValueError('ROS perception requires the causal Ego-topic checkpoint contract')
        if not contract.get('map_basis'):
            raise ValueError('ROS perception requires a declared projected-map basis')
        self.input_adapter = LiveInputAdapter(contract, lidar_frame=lidar_frame,
                                              lidar_to_base=lidar_to_base)
        self.contract = self.input_adapter.contract
        if not isinstance(expected_lidar_frame_id, str) or not expected_lidar_frame_id:
            raise ValueError('Declare the ROS LiDAR frame ID')
        self.expected_lidar_frame_id = expected_lidar_frame_id
        self.buffer = StampedPerceptionBuffer(max_age_s, slop_ns)
        self.subscribers = []
        self.last_error = None

    def _camera(self, message):
        try:
            role = CAMERAS.get(message.header.frame_id)
            if role is None:
                return
            self.buffer.push(role, source_stamp_ns(message), bytes(message.data))
            self.last_error = None
        except Exception as error:
            self.last_error = str(error)

    def _lidar(self, message):
        try:
            self.buffer.push('lidar', source_stamp_ns(message), message)
        except Exception as error:
            self.last_error = str(error)

    def _ego(self, message):
        try:
            value = dict(x=float(message.position.x), y=float(message.position.y),
                         z=float(message.position.z), yaw_raw=math.radians(float(message.heading)),
                         speed=math.hypot(message.velocity.x, message.velocity.y))
            if not all(math.isfinite(v) for v in value.values()):
                raise ValueError('Nonfinite Ego pose/speed')
            self.buffer.push('ego', source_stamp_ns(message), value)
        except Exception as error:
            self.last_error = str(error)

    def start(self):
        import rospy
        from sensor_msgs.msg import CompressedImage, PointCloud2
        from morai_msgs.msg import EgoVehicleStatus
        if self.subscribers:
            raise ValueError('ROS perception source already started')
        self.subscribers = [
            rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self._camera,
                             queue_size=4, buff_size=32 * 1024 * 1024, tcp_nodelay=True),
            rospy.Subscriber('/lidar3D', PointCloud2, self._lidar,
                             queue_size=2, buff_size=32 * 1024 * 1024, tcp_nodelay=True),
            rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self._ego,
                             queue_size=128, tcp_nodelay=True)]

    def latest(self, device):
        import cv2
        import torch
        from data.src.loaders.morai_map_geometry import grid_yaw
        selected, meta = self.buffer.paired()
        if selected is None:
            return None, meta
        try:
            images = {}
            for role in CAMERAS.values():
                image = cv2.imdecode(np.frombuffer(selected[role]['value'], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Undecodable camera: ' + role)
                images[role] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cloud = selected['lidar']['value']
            if cloud.header.frame_id != self.expected_lidar_frame_id:
                raise ValueError('ROS LiDAR frame ID differs from the calibrated sensor')
            points = pointcloud_xyzi(cloud)
            ego = selected['ego']['value']
            yaw = grid_yaw(ego['yaw_raw'], (ego['x'], ego['y'], ego['z']),
                           self.contract['map_basis'], source_frame='true_enu')
            adapter = self.input_adapter.adapter
            inputs = adapter.camera_tensors(**{name + '_rgb': images['rgb_' + name]
                for name in ('front', 'left', 'right', 'rear')})
            lidar, count = adapter.lidar_to_raw_features(points, rng_seed=meta['stamp_ns'] % (2**32))
            if not count:
                raise ValueError('Empty LiDAR after checkpoint preprocessing')
            inputs.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(count),
                          velocity=torch.tensor([ego['speed']], dtype=torch.float32))
            meta.update(ego_pose=dict(x=ego['x'], y=ego['y'], yaw=float(yaw)),
                        speed_mps=ego['speed'], front_image=images['rgb_front'],
                        lidar_frame_id=selected['lidar']['value'].header.frame_id,
                        waypoint_branch_used=False)
            return {key: value.unsqueeze(0).to(device) for key, value in inputs.items()}, meta
        except Exception as error:
            meta.update(ready=False, invalid_reason=str(error))
            return None, meta

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.subscribers = []
