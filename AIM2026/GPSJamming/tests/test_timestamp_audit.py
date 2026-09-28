import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'scripts/compare_timestamps.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_header_timestamp_and_unrelated_packets(self):
        packet = b'MOR' + struct.pack('<IIII', 123, 456, 0, 2) + b'xxEI'
        self.assertEqual(audit.timestamp('camera', packet), 123000000456)
        self.assertIsNone(audit.timestamp('camera', b'BOX' + packet[3:]))
        lidar = bytearray(1206)
        lidar[:2] = b'\xff\xee'
        lidar[1204:] = b'\x37\x22'
        struct.pack_into('<I', lidar, 1200, 12345)
        self.assertEqual(audit.timestamp('lidar', lidar), 12345)

    def test_microseconds_hour_hypothesis_and_peer_isolation(self):
        rows = []
        for i in range(3):
            camera = 3600*10**9 + i*100_000_000
            rows.extend([
                dict(sensor='camera', peer=['127.0.0.1', 5000], raw_timestamp=camera, receive_monotonic_ns=i*100_000_000),
                dict(sensor='lidar', peer=['127.0.0.1', 2369], raw_timestamp=i*100_000+2000, receive_monotonic_ns=i*100_000_000+1000)])
        report = audit.summarize(rows)
        self.assertFalse(report['synchronized'])
        candidates = report['comparisons'][0]['candidates']
        match = next(x for x in candidates if x['lidar_unit']=='microseconds' and x['relation']=='within_hour')
        self.assertEqual(match['lidar_minus_camera_ms']['median'], 2)
        self.assertEqual(audit.summarize([])['status'], 'insufficient_sensor_data')


if __name__ == '__main__':
    unittest.main()
