"""Allowed UDP ego inputs: GPS position, IMU heading, signed status speed.

No status position, status yaw, simulator actor, or object GT is exposed.
NMEA fixes have receipt-time freshness, not a proven source-time alignment.
"""
from collections import deque
import math
import threading
import time

from inference.sensors.morai_udp import GpsReceiver, ImuReceiver, VehicleStatusReceiver


class ValidGpsReceiver(GpsReceiver):
    def _consume(self, payload):
        valid = []
        for line in payload.decode('ascii', 'ignore').split('$'):
            line = line.strip()
            if not line:
                continue
            if '*' in line:
                body, checksum = line.split('*', 1)
                computed = 0
                for char in body:
                    computed ^= ord(char)
                try:
                    if computed != int(checksum[:2], 16):
                        continue
                except ValueError:
                    continue
            else:
                body = line
            parts = body.split(',')
            if (parts[0].endswith('RMC') and len(parts) > 6 and parts[2] == 'A'):
                valid.append('$' + body)
            elif parts[0].endswith('GGA') and len(parts) > 9:
                try:
                    if int(parts[6]) > 0:
                        valid.append('$' + body)
                except ValueError:
                    pass
        value = super()._consume('\n'.join(valid).encode('ascii')) if valid else None
        if value is not None:
            lat, lon = value['latitude'], value['longitude']
            if (not all(math.isfinite(v) for v in (lat, lon)) or not -90 <= lat <= 90
                    or not -180 <= lon <= 180 or (lat == 0 and lon == 0)):
                return None
        return value


class HistoryMixin:
    def __init__(self, *args, **kwargs):
        self.history, self.history_lock = deque(maxlen=200), threading.Lock()
        super().__init__(*args, **kwargs)

    def _consume(self, payload):
        value = super()._consume(payload)
        if value is None:
            return None
        timestamp = value.get('sim_time')
        stamp = round(timestamp * 1e9) if timestamp is not None else 0
        if stamp < 0:
            return None
        now = time.monotonic_ns()
        with self.history_lock:
            if stamp and self.history:
                previous = self.history[-1][1]['source_stamp_ns']
                if stamp == previous:
                    return None
                if stamp < previous:
                    self.history.clear()
            identity = dict(source_stamp_ns=stamp, received_ns=now, sample_id=self.samples+1)
            self.history.append((value, identity))
        self._completed_source_stamp_ns = stamp
        return value

    def sample(self, source_ns=None):
        with self.history_lock:
            candidates = list(self.history)
        if source_ns is not None:
            candidates = [(v, i) for v, i in candidates if
                          0 < i['source_stamp_ns'] <= source_ns
                          and source_ns-i['source_stamp_ns'] <= 50_000_000]
        if not candidates:
            return None, {}
        return candidates[-1]


class SpeedReceiver(HistoryMixin, VehicleStatusReceiver):
    def _consume(self, payload):
        if len(payload) not in (181, 229, 245) or not payload.startswith(b'#MoraiInfo$') or payload[-2:] != b'\r\n':
            return None
        return super()._consume(payload)


class HeadingReceiver(HistoryMixin, ImuReceiver):
    pass


class GpsImuStatusReceiver:
    """LiveFrameSource status interface, deliberately excluding status GT pose."""
    def __init__(self, ports, bind='0.0.0.0', max_age=.5):
        self.port, self.max_age = ports['vehicle_status'], max_age
        self.speed = SpeedReceiver(self.port, bind=bind, name='status_speed')
        try:
            self.imu = HeadingReceiver(ports['imu'], bind=bind, name='imu_heading')
        except BaseException:
            self.speed.close()
            raise

    @property
    def packets(self):
        return self.speed.packets

    @property
    def samples(self):
        return self.speed.samples

    def _snapshot(self, source_ns=None):
        status, identity = self.speed.sample(source_ns)
        if status is None:
            return None, float('inf'), {}
        imu, imu_id = self.imu.sample(source_ns)
        # The 107-byte IMU has no source timestamp. Only receipt freshness can
        # be checked; do not pretend it established the trained 50ms clock match.
        if imu is None:
            candidate, candidate_id = self.imu.sample()
            if candidate is not None and candidate_id['source_stamp_ns'] == 0:
                imu, imu_id = candidate, candidate_id
        if imu is None:
            return None, float('inf'), {}
        received_ns = min(identity['received_ns'], imu_id['received_ns'])
        age = (time.monotonic_ns()-received_ns) / 1e9
        speed, yaw = status['signed_speed']/3.6, imu['yaw']
        if (age < 0 or age > self.max_age or not all(math.isfinite(v) for v in (speed, yaw))
                or abs(speed) > 100):
            return None, age, {}
        # Model input is planar speed magnitude, matching the trained recipe.
        value = dict(speed=abs(speed), signed_speed_mps=speed, yaw=yaw,
                     yaw_degrees=math.degrees(yaw),
                     accel=status['accel'], brake=status['brake'], steer=status['steer'],
                     gear=status['gear'], competition_variant=status['competition_variant'])
        identity = dict(identity, received_ns=received_ns, imu_received_ns=imu_id['received_ns'],
                        imu_source_stamp_ns=imu_id['source_stamp_ns'],
                        imu_alignment='causal_50ms' if source_ns is not None and imu_id['source_stamp_ns'] else 'receipt_freshness_only')
        return value, age, identity

    def latest(self):
        value, age, _ = self._snapshot()
        return value, age

    def latest_snapshot(self):
        return self._snapshot()

    def at_source_stamp(self, source_ns, max_delta_ns=50_000_000):
        return self._snapshot(source_ns)

    def diagnostics(self):
        return dict(status=self.speed.diagnostics(), imu=self.imu.diagnostics(),
                    pose_source='gps', heading_source='imu', speed_source='signed_status_kmh_div_3.6')

    def close(self):
        self.speed.close()
        self.imu.close()
