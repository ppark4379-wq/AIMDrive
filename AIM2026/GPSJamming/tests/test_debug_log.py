from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.timing.gap_metrics import GapMetrics
from gps_jamming.timing.debug_log import selection_lines, display_line


class DebugLogTests(unittest.TestCase):
    def test_unavailable_clock_and_recovery(self):
        m = GapMetrics()
        gap = m.observe(1, 0, None, 0, False, 'waiting_camera')
        row = {'gap_metrics': gap, 'association_status': 'waiting_camera',
               'camera_timestamp': None, 'lidar': {'frame_id': 1}, 'display': {'mode': 'unmatched'}}
        self.assertIn('fail_streak=1', selection_lines(row)[0])
        self.assertIn('delta_ms=N/A', selection_lines(row)[0])
        m.observe(2, 0, 100000000, 100000000, True, 'matched')
        m.observe(3, 0, 200000000, 200000000, False, 'timestamp_out_of_tolerance')
        row['gap_metrics'] = m.observe(5, 0, 400000000, 500000000, True, 'matched')
        lines = selection_lines(row)
        self.assertIn('wall_ms=400.00 sensor_ms=300.00 failed=1 skipped=1', lines[-1])
        self.assertIn('duration_ms=123.40', display_line({'pair_key': [1, 2], 'mode': 'hold', 'duration_ms': 123.4}))


if __name__ == '__main__':
    unittest.main()
