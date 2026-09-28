from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.visualization.preview_hold import PreviewHold


class PreviewHoldTests(unittest.TestCase):
    def test_same_measurement_does_not_extend_hold(self):
        hold = PreviewHold(350)
        canvas = np.zeros((2, 2, 3), dtype=np.uint8)
        hold.update(canvas, (1, 2), 0, 1_000_000_000, {'delta_ms': 10})
        canvas[:] = 255
        hold.update(canvas, (1, 2), 300_000_000, 1_000_000_000, {})
        saved, meta = hold.get(300_000_000)
        self.assertEqual(int(saved.max()), 0)
        self.assertEqual(meta['held_ms'], 300)
        self.assertIsNone(hold.get(351_000_000))

    def test_stale_and_reset_clear(self):
        hold = PreviewHold()
        hold.update(np.ones((2, 2)), 1, 0, 100_000_000, {})
        self.assertIsNone(hold.get(100_000_001))
        hold.update(np.ones((2, 2)), 2, 200_000_000, 1_000_000_000, {})
        self.assertIsNotNone(hold.get(210_000_000))
        hold.clear()
        self.assertIsNone(hold.get(210_000_000))


if __name__ == '__main__':
    unittest.main()
