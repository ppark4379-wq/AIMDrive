"""Views of AIM2026's shared receivers; never open or own sensor sockets.

The numerical estimator still sees the original allowed-input contract:
signed km/h, IMU quaternion/gyro, and receipt-causal NMEA fixes. Source stamps
use the original float-to-nanosecond conversion for migration equivalence.
"""
from collections import deque
import threading
import time


def receipt_ns(record):
    stamp = getattr(record, 'received_ns', None)
    return stamp if stamp is not None else round(record.received_at * 1e9)


def identity(record):
    source = record.value.get('sim_time') if isinstance(record.value, dict) else None
    return dict(sample_id=record.sequence, received_ns=receipt_ns(record),
                source_stamp_ns=round(source * 1e9) if source is not None else 0)


class HistoryView:
    """Snapshot history with the same shape/limit as the improved DR receiver."""

    def __init__(self, receiver, fields):
        self.receiver, self.fields = receiver, fields
        self.history, self.history_lock = deque(maxlen=200), threading.Lock()

    def refresh(self):
        records = self.receiver.recent()
        rows = [(dict((key, record.value[key]) for key in self.fields if key in record.value),
                 identity(record)) for record in records if record.value is not None]
        with self.history_lock:
            self.history.clear()
            self.history.extend(rows)
        return records


class StatusView:
    """Explicit allow-list: packet GT position and packet attitude never enter DR."""

    def __init__(self, speed_receiver, imu_receiver):
        self.speed = HistoryView(speed_receiver, ('signed_speed', 'sim_time'))
        self.imu = HistoryView(imu_receiver, ('quaternion', 'yaw', 'angular_velocity', 'sim_time'))

    def refresh(self):
        return self.speed.refresh(), self.imu.refresh()


class GpsView:
    """Non-owning, validated GPS view with the legacy snapshot interface."""

    def __init__(self, receiver):
        self.receiver = receiver
        self._records = None

    def refresh(self):
        self._records = self.receiver.recent()
        return self._records

    @property
    def port(self):
        return self.receiver.port

    @property
    def packets(self):
        return self.receiver.packets

    @property
    def samples(self):
        return self.receiver.samples

    def latest_snapshot(self):
        records = self.receiver.recent() if self._records is None else self._records
        if not records:
            return None, float('inf'), {}
        record = records[-1]
        value = {key: record.value.get(key) for key in ('latitude', 'longitude', 'altitude')}
        meta = identity(record)
        return value, (time.monotonic_ns() - meta['received_ns']) / 1e9, meta

    def latest(self):
        value, age, _ = self.latest_snapshot()
        return value, age

    def diagnostics(self):
        return self.receiver.diagnostics()

    def close(self):
        """The UDPManager, not this view, closes the shared receiver."""
