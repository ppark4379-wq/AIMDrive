from pathlib import Path
import sys
import unittest
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.timing.synchronization import FrameSynchronizer, restore_hour_timestamp, HOUR_NS
from gps_jamming.types import LidarFrame


def frame(raw, complete=True):
    return LidarFrame(1, raw, None, 0, 'base_link', complete, np.empty((0, 4)), 100_000_000)


def camera(stamp, receive=100_000_000):
    return {'timestamp_ns': stamp, 'receive_ns': receive}


class SynchronizationTests(unittest.TestCase):
    def test_previous_and_following_hour(self):
        self.assertEqual(restore_hour_timestamp(3_599_990_000, HOUR_NS + 10_000_000), HOUR_NS-10_000_000)
        self.assertEqual(restore_hour_timestamp(10_000, HOUR_NS-10_000_000), HOUR_NS+10_000_000)
        for raw in [-1, 3_600_000_000]:
            with self.assertRaises(ValueError):
                restore_hour_timestamp(raw, HOUR_NS)

    def test_nearest_and_inclusive_tolerance(self):
        sync = FrameSynchronizer()
        sync.add_camera(camera(HOUR_NS+90_000_000))
        sync.add_camera(camera(HOUR_NS+140_000_000))
        chosen, mapped, info = sync.select(frame(100_000), 200_000_000)
        self.assertEqual(chosen['timestamp_ns'], HOUR_NS+90_000_000)
        self.assertEqual(info['delta_ns'], 10_000_000)
        self.assertEqual(mapped.timestamp, HOUR_NS+100_000_000)
        self.assertEqual(info['status'], 'matched')
        self.assertEqual(sync.select(frame(190_000), 200_000_000)[2]['status'], 'matched')
        self.assertEqual(sync.select(frame(190_001), 200_000_000)[2]['status'], 'timestamp_out_of_tolerance')

    def test_stale_missing_incomplete_and_backward(self):
        sync = FrameSynchronizer()
        self.assertEqual(sync.select(frame(100), 0)[2]['status'], 'waiting_camera')
        sync.add_camera(camera(HOUR_NS))
        self.assertEqual(sync.select(frame(100), 2_000_000_000)[2]['status'], 'stale_sensor')
        self.assertEqual(sync.select(frame(100, False), 200_000_000)[2]['status'], 'incomplete_lidar')
        self.assertEqual(sync.select(frame(50), 200_000_000)[2]['status'], 'lidar_clock_discontinuity')
        self.assertFalse(sync.cameras)

    def test_rollover_is_not_clock_reset_and_camera_reset_clears(self):
        sync = FrameSynchronizer()
        sync.add_camera(camera(HOUR_NS-10_000_000))
        self.assertEqual(sync.select(frame(3_599_990_000), 200_000_000)[2]['status'], 'matched')
        sync.add_camera(camera(HOUR_NS+10_000_000))
        self.assertEqual(sync.select(frame(10_000), 200_000_000)[2]['status'], 'matched')
        self.assertEqual(sync.epoch, 0)
        sync.add_camera(camera(100))
        self.assertEqual(len(sync.cameras), 1)
        self.assertEqual(sync.epoch, 1)


if __name__ == '__main__':
    unittest.main()
