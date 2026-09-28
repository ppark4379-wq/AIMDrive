from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.types import CameraTrack, AssociatedCluster, AssociationResult, ProjectedCluster
from gps_jamming.association.positions import estimate_positions
from gps_jamming.association.observations import track_observations, position_labels


class PositionTests(unittest.TestCase):
    def data(self):
        track = CameraTrack(2, [0, 0, 10, 10], 2, .9)
        points = np.array([[10, -2, 1, 0], [12, -2, 2, 0], [12, 0, 3, 0], [10, 0, 1, 0]])
        match = AssociatedCluster(2, 2, .9, 3, points, points[:, :3].mean(0), 'base_link', 3, 4, .75)
        projected = ProjectedCluster(3, np.array([[1, 1], [2, 2], [3, 3], [20, 20]]), np.arange(4), np.array([10, 11, 12, 100]))
        return [track], AssociationResult(5, 123, [match], [projected], [], [])

    def test_obb_center_keeps_original_centroid(self):
        tracks, result = self.data()
        rows = track_observations(tracks, result)
        p = rows[0]['position']
        self.assertEqual(p['xyz_m'], [11, -1, 2])
        self.assertEqual(p['support_points'], 3)
        self.assertAlmostEqual(p['planar_distance_m'], np.sqrt(122))
        self.assertEqual(p['timestamp_ns'], 123)
        self.assertFalse(p['ambiguous'])
        self.assertNotEqual(rows[0]['clusters'][0]['centroid_xyz_m'], p['xyz_m'])
        labels = position_labels(rows[0])
        self.assertEqual(labels[:2], ['ID 2 | matched', 'Cluster 3 | XY range 11.05m'])
        self.assertEqual(labels[2], 'Rel POS X +11.00 Y -1.00 m')
        self.assertEqual(labels[3], 'Rel VEL -- (unavailable)')
        rows[0]['motion'] = {'status': 'relative_only', 'velocity_xy_mps': [-2.5, .3]}
        self.assertEqual(position_labels(rows[0])[3], 'Rel VEL X -2.50 Y +0.30 m/s')

    def test_nearest_depth_not_average_between_objects(self):
        tracks, result = self.data()
        points = np.array([[30, 0, 0, 0], [31, 0, 0, 0], [32, 0, 0, 0]])
        result.matches.insert(0, AssociatedCluster(2, 2, .9, 7, points, points[:, :3].mean(0), 'base_link', 3, 3, 1.))
        result.projections.append(ProjectedCluster(7, np.ones((3, 2)), np.arange(3), np.array([30, 31, 32])))
        p = estimate_positions(tracks, result)[2]
        self.assertEqual(p['cluster_id'], 3)
        self.assertEqual(p['xyz_m'], [11, -1, 2])
        self.assertTrue(p['ambiguous'])
        self.assertEqual(p['candidate_count'], 2)

    def test_missing_or_insufficient_support_has_no_position(self):
        tracks, result = self.data()
        self.assertEqual(estimate_positions(tracks, None), {})
        result.projections[0].pixels[:] = 100
        self.assertEqual(estimate_positions(tracks, result), {})


if __name__ == '__main__':
    unittest.main()
