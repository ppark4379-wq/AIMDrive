"""Reused visualization arrays must match the original numerical outputs."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ROIDownsampling import ROI
from src.Ransac import remove_ground
from main import DEFAULT_CONFIG, load_config
from src.Process import LidarPipeline
from src.LidarDecoder import LidarFrame


class IntermediateTests(unittest.TestCase):
    def test_roi_before_voxel_is_identical_including_empty_and_disabled_voxel(self):
        raw = np.array([[0, 0, 0, 10], [.01, 0, 0, 20], [99, 0, 0, 1],
                        [np.nan, 0, 0, 1]], dtype=np.float32)
        for points in [raw, np.empty((0, 4), dtype=np.float32)]:
            for size in [0, .15]:
                cached = {}
                result = ROI(points, [-1]*3, [1]*3, size, intermediates=cached)
                np.testing.assert_array_equal(result, ROI(points, [-1]*3, [1]*3, size))
                np.testing.assert_array_equal(cached['roi'], ROI(points, [-1]*3, [1]*3, 0))

    def test_ground_mask_for_success_insufficient_and_degenerate_plane(self):
        x, y = np.meshgrid(np.arange(8), np.arange(8))
        road = np.column_stack((x.ravel(), y.ravel(), np.full(64, -1.48), np.ones(64)))
        scene = np.vstack((road, [[1, 1, 0, 20]]))
        for points, status in [(scene, 'plane_found'), (scene[:2], 'insufficient_seeds'),
                               (np.tile(road[0], (64, 1)), 'no_supported_plane'),
                               (np.empty((0, 4)), 'insufficient_seeds')]:
            cached = {}
            remaining, info = remove_ground(points, intermediates=cached)
            reference, reference_info = remove_ground(points)
            np.testing.assert_array_equal(remaining, reference)
            self.assertEqual(info, reference_info)
            self.assertEqual(info['ground_status'], status)
            mask = cached['ground_mask']
            self.assertEqual(int(mask.sum()), info['ground_points'])
            np.testing.assert_array_equal(points[~mask], remaining)

    def test_pipeline_uses_roi_once_and_reuses_intermediates(self):
        config = load_config(DEFAULT_CONFIG)['lidar']
        frame = LidarFrame(1, 1, None, 1, config['assembly']['coordinate_frame'], True,
                           np.array([[0, 0, 0, 42]], dtype=np.float32), 1)
        pipe = LidarPipeline(config, capture_debug=True)
        with patch('src.Process.preprocess', wraps=ROI) as roi, \
             patch('src.Process.remove_ground', wraps=remove_ground) as ground:
            pipe.process(frame)
        self.assertEqual(roi.call_count, 1)
        self.assertEqual(ground.call_count, 1)
        cached = roi.call_args.kwargs['intermediates']
        self.assertIs(pipe.last_debug['roi'], cached['roi'])
        self.assertIs(ground.call_args.kwargs['intermediates'], cached)
        np.testing.assert_array_equal(pipe.last_debug['ground'],
                                      pipe.last_debug['voxel'][cached['ground_mask']])



if __name__ == '__main__':
    unittest.main()
