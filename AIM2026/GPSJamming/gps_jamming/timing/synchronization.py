"""Buffered frame selection within timestamp tolerance."""
from collections import deque
from dataclasses import replace

HOUR_NS = 3_600_000_000_000


def restore_hour_timestamp(raw_us, reference_ns):
    """MORAI observed model: microseconds since hour start, nearest reference hour."""
    if not isinstance(raw_us, int) or not 0 <= raw_us < 3_600_000_000:
        raise ValueError('LiDAR timestamp outside microseconds-within-hour range')
    hour = reference_ns // HOUR_NS * HOUR_NS
    return min((hour + offset + raw_us * 1000 for offset in (-HOUR_NS, 0, HOUR_NS)),
               key=lambda value: abs(value - reference_ns))


class FrameSynchronizer:
    """LiDAR-driven nearest selection from bounded, already tracked camera frames.

    Receipt time only rejects stale data; it never supplies a clock anchor.
    Camera timestamp epoch must be verified by the caller's sensor configuration.
    """
    def __init__(self, max_delta_ms=50, max_age_ms=800, camera_buffer_size=30):
        if max_delta_ms <= 0 or max_age_ms <= 0 or camera_buffer_size < 1:
            raise ValueError('Synchronization limits must be positive')
        self.tolerance = int(max_delta_ms * 1e6)
        self.max_age = int(max_age_ms * 1e6)
        self.cameras = deque(maxlen=camera_buffer_size)
        self.last_lidar_ns = None
        self.epoch = 0

    def add_camera(self, sample):
        if self.cameras and sample['timestamp_ns'] <= self.cameras[-1]['timestamp_ns']:
            self.cameras.clear()
            self.last_lidar_ns = None
            self.epoch += 1
        self.cameras.append(sample)

    def select(self, frame, now_ns):
        if not self.cameras:
            return None, None, {'status': 'waiting_camera', 'delta_ns': None}
        reference = self.cameras[-1]
        if (not 0 <= now_ns - reference['receive_ns'] <= self.max_age
                or not 0 <= now_ns - frame.receive_end_monotonic_time <= self.max_age):
            return None, None, {'status': 'stale_sensor', 'delta_ns': None}
        try:
            stamp = restore_hour_timestamp(frame.raw_timestamp, reference['timestamp_ns'])
        except ValueError:
            return None, None, {'status': 'invalid_lidar_timestamp', 'delta_ns': None}
        if self.last_lidar_ns is not None and stamp <= self.last_lidar_ns:
            self.cameras.clear()
            self.last_lidar_ns = None
            self.epoch += 1
            return None, None, {'status': 'lidar_clock_discontinuity', 'delta_ns': None}
        self.last_lidar_ns = stamp
        eligible = [c for c in self.cameras if 0 <= now_ns - c['receive_ns'] <= self.max_age]
        chosen = min(eligible, key=lambda c: abs(c['timestamp_ns'] - stamp))
        delta = stamp - chosen['timestamp_ns']
        status = ('incomplete_lidar' if not frame.complete else
                  'timestamp_out_of_tolerance' if abs(delta) > self.tolerance else 'matched')
        info = {'status': status, 'delta_ns': delta, 'epoch': self.epoch,
                'clock_model': 'microseconds_within_hour', 'max_delta_ns': self.tolerance}
        mapped = replace(frame, timestamp=stamp, timestamp_status='camera_hour_reconstructed',
                         clock_epoch=self.epoch)
        return chosen, mapped, info
