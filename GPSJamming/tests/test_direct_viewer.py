import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.visualization.direct_viewer import viewer_payload, stage_arrays, centroid_crosses
from gps_jamming.lidar.runtime import DEFAULT_CONFIG, load_config
from gps_jamming.pipeline import LidarPipeline
from gps_jamming.types import LidarFrame


class DirectViewerTests(unittest.TestCase):
    def test_ground_preview_partitions_points_without_clustering(self):
        config = load_config(DEFAULT_CONFIG)["lidar"]
        config['preprocessing'].update(roi_min=[-10, -10, -3], roi_max=[20, 10, 5], voxel_size=0)
        config['ground'].update(expected_z=-.03, min_inliers=40)
        x, y = np.meshgrid(np.arange(8), np.arange(8))
        ground = np.column_stack((x.ravel(), y.ravel(), np.full(64, -1.48), np.ones(64)))
        points = np.vstack((ground, [[1, 1, 0, 50], [2, 2, 0, 100]])).astype(np.float32)
        frame = LidarFrame(9, 123, None, 10, config['assembly']['coordinate_frame'], True, points, 20)
        pipe = LidarPipeline(config, capture_debug=True, ground_only=True)
        with patch('gps_jamming.pipeline.cluster_indices', side_effect=AssertionError('clustering ran')):
            result = pipe.process(frame)
        self.assertEqual(len(result.points), 2)
        payload = viewer_payload(pipe.last_debug)
        self.assertEqual(payload['ground_status'], 'plane_found')
        self.assertEqual(set(payload['points']), {'voxel', 'ground', 'nonground'})
        self.assertEqual(len(stage_arrays(payload, 'ground', (-3, 5))[0]), 64)
        self.assertEqual(len(payload['points']['voxel']), 66)
        # Unsupported plane: keep all input points in the remaining view.
        config['ground']['min_inliers'] = 100
        pipe.process(frame)
        payload = viewer_payload(pipe.last_debug)
        self.assertEqual(payload['ground_status'], 'insufficient_seeds')
        self.assertEqual(len(payload['points']['ground']), 0)
        self.assertEqual(len(payload['points']['nonground']), 66)

    def test_voxel_preview_centroids_and_roi_comparison_without_downstream(self):
        config = load_config(DEFAULT_CONFIG)["lidar"]
        config['preprocessing'].update(roi_min=[-2, -2, -2], roi_max=[2, 2, 2], voxel_size=1)
        points = np.array([[0, 0, 0, 40], [.02, 0, 0, 100], [10, 0, 0, 1]], dtype=np.float32)
        frame = LidarFrame(9, 123, None, 10, config['assembly']['coordinate_frame'], True, points, 20)
        pipe = LidarPipeline(config, capture_debug=True, voxel_only=True)
        with patch('gps_jamming.pipeline.remove_ground', side_effect=AssertionError('RANSAC ran')), \
             patch('gps_jamming.pipeline.cluster_indices', side_effect=AssertionError('clustering ran')):
            result = pipe.process(frame)
        np.testing.assert_allclose(result.points, [[1.01, 0, 1.45, 70]])
        self.assertEqual(result.coordinate_frame, 'base_link')
        self.assertEqual(result.diagnostics['roi_points'], 2)
        self.assertEqual(result.diagnostics['voxel_points'], 1)
        payload = viewer_payload(pipe.last_debug)
        self.assertEqual(set(payload['points']), {'roi', 'voxel'})
        self.assertEqual(len(stage_arrays(payload, 'roi', (-2, 2))[0]), 2)
        self.assertEqual(len(stage_arrays(payload, 'voxel', (-2, 2))[0]), 1)

    def test_roi_only_skips_downstream_stages_and_preserves_close_points(self):
        config = load_config(DEFAULT_CONFIG)["lidar"]
        config['preprocessing'].update(roi_min=[-2, -2, -2], roi_max=[2, 2, 2], voxel_size=1)
        points = np.array([[0, 0, 0, 42], [.01, 0, 0, 99], [10, 0, 0, 1]], dtype=np.float32)
        frame = LidarFrame(9, 123, None, 10, config['assembly']['coordinate_frame'], True, points, 20)
        pipe = LidarPipeline(config, capture_debug=True, roi_only=True)
        with patch('gps_jamming.pipeline.remove_ground', side_effect=AssertionError('RANSAC ran')), \
             patch('gps_jamming.pipeline.cluster_indices', side_effect=AssertionError('clustering ran')):
            result = pipe.process(frame)
        self.assertEqual(result.coordinate_frame, 'base_link')
        self.assertEqual(len(result.points), 2)
        np.testing.assert_allclose(result.points[:, :3], points[:2, :3] + [1, 0, 1.45])
        np.testing.assert_array_equal(result.points[:, 3], [42, 99])
        payload = viewer_payload(pipe.last_debug)
        self.assertEqual(set(payload['points']), {'roi'})
        xyz, colors = stage_arrays(payload, 'roi', (-2, 2))
        self.assertEqual(xyz.shape, (2, 3))
        self.assertEqual(colors.shape, xyz.shape)
        self.assertEqual(result.clusters, [])

    def test_full_points_and_cluster_colors(self):
        points = np.array([[1, 2, 3, 100], [2, 2, 3, 50], [10, 2, 3, 10]], dtype=np.float32)
        config = load_config(DEFAULT_CONFIG)["lidar"]
        config["clustering"].update(min_points=1)
        pipe = LidarPipeline(config, capture_debug=True)
        pipe.process(LidarFrame(9, 123, None, 10, config["assembly"]["coordinate_frame"], True, points, 20))
        payload = viewer_payload(pipe.last_debug)
        xyz, colors = stage_arrays(payload, "clusters", (-3, 5))
        self.assertEqual(len(xyz), len(pipe.last_debug["nonground"]))
        self.assertEqual(colors.shape, xyz.shape)
        np.testing.assert_allclose(payload["points"]["raw"], points[:, :3] + [1, 0, 1.45])
        self.assertEqual(payload["raw_timestamp"], 123)
        self.assertTrue(np.all(payload["labels"] > 0))
        self.assertEqual(len(np.unique(colors, axis=0)), len(pipe.last_debug["clusters"]))

    def test_empty_cloud_and_centroid_geometry(self):
        payload = {"points": {"nonground": np.empty((0, 3))}, "labels": np.empty(0, dtype=int)}
        xyz, colors = stage_arrays(payload, "clusters", (-3, 5))
        self.assertEqual(xyz.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))
        vertices, lines = centroid_crosses(np.array([[1, 2, 3]]))
        self.assertEqual(vertices.shape, (6, 3))
        self.assertEqual(lines.shape, (3, 2))
        np.testing.assert_allclose(vertices.mean(axis=0), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
