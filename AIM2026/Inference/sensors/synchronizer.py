"""Causal joins of permitted UDP observations, bounded by freshness/skew."""
from dataclasses import dataclass
import math
import time


CAMERAS = ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear')


@dataclass(frozen=True)
class SensorFrame:
    source_time: float
    received_at: float
    sequence: int
    images: dict
    lidar: object
    speed_mps: float
    pose: dict


def causal_sample(records, anchor, tolerance):
    """Prefer simulator time when both stamps exist; otherwise receipt time.

    NMEA GPS and the assembled Velodyne sweep have no common simulator stamp.
    Their receipt-time bound is an approximation, not a claim of hardware sync.
    """
    candidates = []
    for record in records:
        if anchor.source_time is not None and record.source_time is not None:
            delta = anchor.source_time - record.source_time
        else:
            delta = anchor.received_at - record.received_at
        if 0 <= delta <= tolerance + 1e-7:
            candidates.append((delta, record))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


class SensorSynchronizer:
    def __init__(self, udp, projector, *, max_age_s=.3, max_skew_s=.15):
        self.udp, self.projector = udp, projector
        self.max_age, self.max_skew = max_age_s, max_skew_s
        self.last_sequence = 0
        self.reason = 'waiting_for_sensors'

    def current_state(self):
        snapshot = self.udp.snapshot()
        names = ('gps', 'imu', 'vehicle_status')
        if any(name not in snapshot or snapshot[name][0] is None or snapshot[name][1] > self.max_age
               for name in names):
            return None, 'stale_pose_or_status'
        stamps = [snapshot[name][3] for name in names]
        if max(stamps) - min(stamps) > self.max_skew:
            return None, 'pose_status_skew'
        status = snapshot['vehicle_status'][0]
        if status.get('competition_variant') is not True:
            return None, 'competition_status_required'
        pose = self.projector.project(snapshot['gps'][0], snapshot['imu'][0])
        return dict(pose=pose, speed_mps=status['speed'], source_time=status['sim_time']), None

    def next_frame(self):
        recent = self.udp.recent()
        if not recent.get('rgb_front'):
            self.reason = 'waiting_for_front_camera'
            return None
        anchor = recent['rgb_front'][-1]
        if anchor.sequence <= self.last_sequence:
            return None
        now = time.monotonic()
        if now - anchor.received_at > self.max_age:
            self.reason = 'stale_front_camera'
            return None
        if anchor.source_time is None or not math.isfinite(anchor.source_time):
            self.reason = 'front_camera_timestamp_required'
            return None
        values = {}
        for name in (*CAMERAS, 'lidar', 'gps', 'imu', 'vehicle_status'):
            # Match the frozen VE's causal speed contract (50 ms).
            tolerance = .05 if name == 'vehicle_status' else self.max_skew
            record = causal_sample(recent.get(name, ()), anchor, tolerance)
            if record is None or now - record.received_at > self.max_age:
                self.reason = 'waiting_for_causal_' + name
                return None
            values[name] = record.value
        if not values['vehicle_status'].get('competition_variant'):
            self.reason = 'competition_status_required'
            return None
        speed = values['vehicle_status']['speed']
        if not math.isfinite(speed) or speed < 0:
            self.reason = 'non_forward_speed'
            return None
        pose = self.projector.project(values['gps'], values['imu'])
        self.last_sequence = anchor.sequence
        self.reason = None
        return SensorFrame(anchor.source_time, anchor.received_at, anchor.sequence,
                           {name: values[name] for name in CAMERAS}, values['lidar'], speed, pose)
