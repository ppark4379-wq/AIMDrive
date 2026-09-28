"""Sensor-time regression of BEV positions; no implied ego compensation."""
from collections import deque
import numpy as np


class BEVTrackHistory:
    def __init__(self, window_s=.6, max_gap_s=.5, min_samples=3, max_speed_mps=80.):
        if window_s <= 0 or max_gap_s <= 0 or min_samples < 2 or max_speed_mps <= 0:
            raise ValueError('Invalid BEV motion limits')
        self.window_s, self.max_gap_s = window_s, max_gap_s
        self.min_samples, self.max_speed_mps = min_samples, max_speed_mps
        self.history = {}
        self.last_timestamp = None
        self.epoch = None

    def update(self, rows, timestamp_ns, epoch):
        if timestamp_ns is None:
            return
        if self.epoch != epoch or (self.last_timestamp is not None and timestamp_ns < self.last_timestamp):
            self.history.clear()
        if timestamp_ns == self.last_timestamp and self.epoch == epoch:
            return
        self.epoch, self.last_timestamp = epoch, timestamp_ns
        for track_id, history in list(self.history.items()):
            if (timestamp_ns-history[-1][0])/1e9 > self.max_gap_s:
                del self.history[track_id]
        for row in rows:
            position = row.get('position')
            motion = {'status': 'unavailable', 'velocity_xy_mps': None,
                      'speed_mps': None, 'absolute_velocity_xy_mps': None,
                      'ego_compensated': False, 'method': 'sensor_time_linear_regression',
                      'velocity_reference': 'derivative_in_moving_base_link'}
            row['motion'] = motion
            if position is None:
                continue
            track_id = row['track_id']
            if position['ambiguous'] or position['bev_obb']['degenerate']:
                self.history.pop(track_id, None)
                motion['status'] = 'ambiguous_or_degenerate'
                continue
            xy = np.asarray(position['xyz_m'][:2])
            history = self.history.setdefault(track_id, deque(maxlen=120))
            if history:
                dt = (timestamp_ns-history[-1][0])/1e9
                if dt <= 0:
                    continue
                if np.linalg.norm(xy-history[-1][1])/dt > self.max_speed_mps:
                    history.clear()
                    motion['status'] = 'position_jump_reset'
            history.append((timestamp_ns, xy))
            while history and (timestamp_ns-history[0][0])/1e9 > self.window_s:
                history.popleft()
            motion['sample_count'] = len(history)
            motion['coordinate_frame'] = position['coordinate_frame']
            if len(history) < self.min_samples:
                if motion['status'] != 'position_jump_reset':
                    motion['status'] = 'warming_up'
                continue
            times = np.array([(t-timestamp_ns)/1e9 for t, _ in history])
            times -= times.mean()
            values = np.array([p for _, p in history])
            velocity = (times[:, None]*(values-values.mean(axis=0))).sum(axis=0)/(times@times)
            motion.update(status='relative_only', velocity_xy_mps=velocity.tolist(),
                          speed_mps=float(np.linalg.norm(velocity)),
                          interval_s=(history[-1][0]-history[0][0])/1e9)
