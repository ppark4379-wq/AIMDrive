"""Wall geometry and publication prerequisites independent of camera/MOT."""
import unittest
import numpy as np

from main import load_config, DEFAULT_CONFIG
from src.Ransac import detect_walls
from src.Boundary import WallWidthFallback
from src.Process import LidarPipeline
from src.LidarDecoder import LidarFrame


class WallDetectionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)
        x, z = np.meshgrid(np.linspace(0, 20, 81), np.linspace(.4, 2.5, 12))
        self.clouds = {side: np.column_stack((x.ravel(), np.full(x.size, sign*5.),
                          z.ravel(), np.ones(x.size)))
                       for side, sign in (('left', 1), ('right', -1))}

    def detect(self, points):
        return detect_walls(points, [0, 0, 1, 0], **self.config['walls'])

    def test_straight_vehicle_facing_clearance_and_observed_extent(self):
        result, _ = self.detect(np.vstack(list(self.clouds.values())))
        clearance = self.config['walls']['clearance_m']
        for side, sign in (('left', 1), ('right', -1)):
            self.assertTrue(result[side])
            for segment in result[side]:
                points = np.asarray(segment['points'])
                np.testing.assert_allclose(points[:, 1], sign*(5-clearance), atol=1e-8)
                np.testing.assert_allclose(points[:, 2], 0, atol=1e-8)
                self.assertGreaterEqual(points[:, 0].min(), 0)
                self.assertLessEqual(points[:, 0].max(), 20)

    def test_no_ground_or_no_points_does_not_invent_boundaries(self):
        for points, plane in ((np.empty((0, 4)), [0, 0, 1, 0]),
                              (self.clouds['left'], None)):
            result, _ = detect_walls(points, plane, **self.config['walls'])
            self.assertFalse(result['left'])
            self.assertFalse(result['right'])

    def test_pipeline_partitions_ground_and_preserves_metadata(self):
        frame = LidarFrame(1, 0, None, 1, 'base_link', True,
                           np.vstack(list(self.clouds.values())), 2)
        pipe = LidarPipeline(self.config['lidar'], capture_debug=True)
        result = pipe.process(frame)
        self.assertEqual(result.frame_id, frame.frame_id)
        self.assertEqual(result.receive_end_monotonic_time, 2)
        self.assertEqual(len(pipe.last_debug['voxel']),
                         len(pipe.last_debug['ground']) + len(pipe.last_debug['nonground']))

    def test_width_fallback_keeps_inferred_provenance(self):
        fallback = WallWidthFallback(**self.config['wall_width_fallback'])
        for i in range(4):
            result, _ = self.detect(np.vstack(list(self.clouds.values())))
            result.update(frame_id='base_link', clock_epoch=0, complete=True,
                          valid=dict(left=True, right=True))
            fallback.update(result, 1+i*.1)
        result, _ = self.detect(self.clouds['left'])
        result.update(frame_id='base_link', clock_epoch=0, complete=True,
                      valid=dict(left=True, right=False))
        display = fallback.update(result, 1.4)
        self.assertTrue(display['right'])
        self.assertTrue(all(s['inferred'] for s in display['right']))
        self.assertFalse(display['valid']['right'])
        self.assertEqual(result['right'], [])
