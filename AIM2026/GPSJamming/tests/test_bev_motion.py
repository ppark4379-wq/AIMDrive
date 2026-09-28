from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.geometry.bev_box import fit_bev_box
from gps_jamming.motion.bev_tracks import BEVTrackHistory


class BEVTests(unittest.TestCase):
    def test_rotated_box_center_not_density_mean(self):
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = np.array([[-2,-1], [2,-1], [2,1], [-2,1], [-2,-1], [-2,-1]]) @ rotation.T + [10,3]
        box = fit_bev_box(np.column_stack((xy, np.zeros(len(xy)))))
        np.testing.assert_allclose(box['center_xy_m'], [10,3], atol=1e-5)
        self.assertAlmostEqual(box['length_m'], 4, places=5)
        self.assertAlmostEqual(box['width_m'], 2, places=5)
        self.assertFalse(np.allclose(xy.mean(axis=0), box['center_xy_m']))

    def row(self, x, ambiguous=False):
        return {'track_id': 1, 'position': {'xyz_m': [x,0,0], 'ambiguous': ambiguous,
                'bev_obb': {'degenerate': False}, 'coordinate_frame': 'base_link'}}

    def test_velocity_sensor_time_gaps_ambiguity_epoch(self):
        tracker = BEVTrackHistory()
        for t in [0, 100_000_000, 250_000_000]:
            rows = [self.row(10+2*t/1e9)]
            tracker.update(rows, t, 0)
        np.testing.assert_allclose(rows[0]['motion']['velocity_xy_mps'], [2,0])
        self.assertFalse(rows[0]['motion']['ego_compensated'])
        rows = [self.row(12)]
        tracker.update(rows, 1_000_000_000, 0)
        self.assertEqual(rows[0]['motion']['status'], 'warming_up')
        rows = [self.row(12, True)]
        tracker.update(rows, 1_100_000_000, 0)
        self.assertEqual(rows[0]['motion']['status'], 'ambiguous_or_degenerate')
        self.assertFalse(tracker.history)
        rows = [self.row(12)]
        tracker.update(rows, 0, 1)
        self.assertEqual(rows[0]['motion']['sample_count'], 1)
