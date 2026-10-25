"""Socket-free test sources shared by localization regression tests."""
import threading
import time

BASIS = {'schema': 'morai_projected_grid_basis_v1', 'position_frame': 'projected_grid', 'source_orientation_frame': 'true_enu', 'orientation_frame': 'projected_grid', 'crs': '+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs', 'origin_xyz': [302595.0, 4124145.0, 0.0]}

class FakeGps:

    def __init__(self):
        self.fix, self.received_ns, self.packets, self.samples = (None, 0, 0, 0)

    def set(self, fix, received_ns):
        self.fix, self.received_ns = (fix, received_ns)

    def latest_snapshot(self):
        if self.fix is None:
            return (None, float('inf'), dict(sample_id=0, received_ns=0, source_stamp_ns=0))
        age = (time.monotonic_ns() - self.received_ns) / 1000000000.0
        return (dict(self.fix), age, dict(sample_id=1, received_ns=self.received_ns, source_stamp_ns=0))

    def diagnostics(self):
        return {}

    def close(self):
        pass

class History:

    def __init__(self):
        self.history, self.history_lock = ([], threading.Lock())

class FakeStatus:

    def __init__(self):
        self.speed, self.imu = (History(), History())

    def add(self, received_ns, speed_mps, yaw, source_ns=0):
        self.speed.history.append((dict(signed_speed=speed_mps * 3.6), dict(received_ns=received_ns, source_stamp_ns=source_ns)))
        self.imu.history.append((dict(yaw=yaw), dict(received_ns=received_ns - 1000, source_stamp_ns=0)))
