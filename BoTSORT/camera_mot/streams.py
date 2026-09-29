"""Bounded camera buffers and fair scheduling, independent of ROS and torch."""
from collections import Counter
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    source_frame_id: str
    stamp: tuple
    received_monotonic_ns: int
    received_frame_id: int
    stream_epoch: int
    jpeg: bytes
    transport: str = 'unknown'
    timestamp_domain: str = 'morai_sensor_header_raw'

    @property
    def stamp_ns(self):
        return self.stamp[0] * 1_000_000_000 + self.stamp[1]


class CameraBuffers:
    """One latest frame per camera; reset is explicit, old stamps are rejected."""

    def __init__(self, cameras, transport='unknown', timestamp_domain='morai_sensor_header_raw'):
        if not cameras:
            raise ValueError("At least one camera is required")
        self.cameras = dict(cameras)
        self.transport = transport
        self.timestamp_domain = timestamp_domain
        self.by_frame_id = {v['frame_id']: k for k, v in cameras.items()}
        if len(self.by_frame_id) != len(cameras):
            raise ValueError("Camera frame IDs must be unique")
        self.order = list(cameras)
        self.condition = threading.Condition()
        self.latest = {}
        self.last_stamp = {}
        self.last_received = {}
        self.epochs = {k: 0 for k in cameras}
        self.counters = {k: Counter() for k in cameras}
        self.ignored_frames = 0
        self.cursor = 0
        self.closed = False

    def put(self, source_frame_id, sec, nsec, jpeg, received_ns=None):
        camera = self.by_frame_id.get(source_frame_id)
        if camera is None:
            with self.condition:
                self.ignored_frames += 1
            return False
        if not isinstance(sec, int) or not isinstance(nsec, int) or sec < 0 or not 0 <= nsec < 1_000_000_000:
            with self.condition:
                self.counters[camera]['invalid_frames'] += 1
            return False
        if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
            with self.condition:
                self.counters[camera]['invalid_frames'] += 1
            return False
        stamp = (sec, nsec)
        now = time.monotonic_ns() if received_ns is None else received_ns
        with self.condition:
            if self.closed:
                return False
            stats = self.counters[camera]
            if camera in self.last_stamp and stamp <= self.last_stamp[camera]:
                stats['stale_frames'] += 1
                if stamp < self.last_stamp[camera]:
                    stats['backward_timestamps'] += 1
                return False
            self.last_stamp[camera] = stamp
            self.last_received[camera] = now
            stats['received_frames'] += 1
            if camera in self.latest:
                stats['overwritten_frames'] += 1
            self.latest[camera] = CameraFrame(camera, source_frame_id, stamp, now,
                stats['received_frames'], self.epochs[camera], bytes(jpeg),
                self.transport, self.timestamp_domain)
            self.condition.notify_all()
        return True

    def reset(self, camera):
        """Call after a known simulator reset; never infer a reset from one late packet."""
        with self.condition:
            self.epochs[camera] += 1
            if self.latest.pop(camera, None) is not None:
                self.counters[camera]['discarded_on_reset'] += 1
            self.last_stamp.pop(camera, None)
            self.last_received.pop(camera, None)
            self.counters[camera]['resets'] += 1
            self.condition.notify_all()

    def get(self, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.latest or self.closed, timeout)
            for offset in range(len(self.order)):
                index = (self.cursor + offset) % len(self.order)
                camera = self.order[index]
                if camera in self.latest:
                    self.cursor = (index + 1) % len(self.order)
                    return self.latest.pop(camera)
            return None

    def stats(self, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self.condition:
            return {camera: {**dict(self.counters[camera]),
                    'stream_epoch': self.epochs[camera],
                    'pending_frames': int(camera in self.latest),
                    'last_received_age_ms': ((now - self.last_received[camera]) / 1e6
                        if camera in self.last_received else None)} for camera in self.order}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
