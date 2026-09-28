from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.timing.gap_metrics import GapMetrics, DisplayIntervals


class GapTests(unittest.TestCase):
    def test_recovery_separates_failures_skips_and_clock_intervals(self):
        m = GapMetrics()
        m.observe(10, 0, 1000000000, 0, True, 'matched')
        a = m.observe(11, 0, 1100000000, 100000000, False, 'timestamp_out_of_tolerance')
        self.assertEqual(a['consecutive_failed_attempts'], 1)
        m.observe(13, 0, 1300000000, 400000000, False, 'incomplete_lidar')
        r = m.observe(14, 0, 1400000000, 500000000, True, 'matched')['recovery']
        self.assertEqual(r['failed_attempts_between'], 2)
        self.assertEqual(r['skipped_lidar_frames_between'], 1)
        self.assertEqual(r['intervening_lidar_frames'], 3)
        self.assertEqual(r['success_interval_wall_ms'], 500)
        self.assertEqual(r['success_interval_sensor_ms'], 400)

    def test_startup_reset_and_repeated_attempt(self):
        m = GapMetrics()
        self.assertIsNone(m.observe(0, 0, None, 0, False, 'waiting_camera')['elapsed_since_success_ms'])
        with self.assertRaises(ValueError):
            m.observe(0, 0, None, 10, False, 'waiting_camera')
        self.assertIsNone(m.observe(1, 0, 123, 20, True, 'matched')['recovery'])
        self.assertIsNone(m.observe(0, 1, 10, 30, True, 'matched')['recovery'])

    def test_display_intervals_include_blocking_processing(self):
        d = DisplayIntervals()
        self.assertIsNone(d.present([1, 2], 'matched', 0))
        self.assertIsNone(d.present([1, 2], 'matched', 100000000))
        self.assertEqual(d.present([1, 2], 'hold', 150000000)['duration_ms'], 150)
        event = d.present([3, 4], 'matched', 700000000)
        self.assertEqual(event['mode'], 'hold')
        self.assertEqual(event['duration_ms'], 550)
        self.assertEqual(d.finish(800000000)['duration_ms'], 100)
        self.assertIsNone(d.finish(900000000))


if __name__ == '__main__':
    unittest.main()
