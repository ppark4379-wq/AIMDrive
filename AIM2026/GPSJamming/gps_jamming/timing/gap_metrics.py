"""Measure selection outages separately from display presentation intervals."""
from collections import Counter


class GapMetrics:
    def __init__(self):
        self.epoch = None
        self.previous_frame = None
        self.success = None
        self.failures = 0
        self.skipped = 0
        self.reasons = Counter()
        self.last_recovery = None

    def observe(self, frame_id, epoch, timestamp_ns, now_ns, accepted, reason):
        if epoch != self.epoch:
            self.__init__()
            self.epoch = epoch
        if self.previous_frame is not None and frame_id <= self.previous_frame:
            raise ValueError('Each LiDAR selection attempt must be counted once in order')
        skipped = max(0, frame_id - self.previous_frame - 1) if self.previous_frame is not None else 0
        self.previous_frame = frame_id
        self.skipped += skipped
        recovery = None
        if accepted:
            if self.success is not None:
                old_frame, old_stamp, old_wall = self.success
                recovery = {
                    'previous_success_frame_id': old_frame,
                    'current_success_frame_id': frame_id,
                    'failed_attempts_between': self.failures,
                    'skipped_lidar_frames_between': self.skipped,
                    'intervening_lidar_frames': frame_id - old_frame - 1,
                    'success_interval_wall_ms': (now_ns-old_wall)/1e6,
                    'success_interval_sensor_ms': ((timestamp_ns-old_stamp)/1e6
                                                   if timestamp_ns is not None and old_stamp is not None else None),
                    'failure_reasons': dict(self.reasons),
                }
                self.last_recovery = recovery
            self.success = (frame_id, timestamp_ns, now_ns)
            self.failures = self.skipped = 0
            self.reasons.clear()
        else:
            self.failures += 1
            self.reasons[reason] += 1
        return {'consecutive_failed_attempts': self.failures,
                'skipped_lidar_frames_this_attempt': skipped,
                'elapsed_since_success_ms': (now_ns-self.success[2])/1e6 if self.success else None,
                'recovery': recovery}


class DisplayIntervals:
    """Intervals between different submitted display states, not monitor scanout."""
    def __init__(self):
        self.current = None

    def present(self, pair_key, mode, now_ns):
        if self.current is not None and self.current[:2] == (pair_key, mode):
            return None
        event = self.finish(now_ns)
        self.current = (pair_key, mode, now_ns)
        return event

    def finish(self, now_ns):
        if self.current is None:
            return None
        key, mode, start = self.current
        self.current = None
        return {'event': 'display_interval', 'pair_key': key, 'mode': mode,
                'start_monotonic_ns': start, 'end_monotonic_ns': now_ns,
                'duration_ms': (now_ns-start)/1e6}
