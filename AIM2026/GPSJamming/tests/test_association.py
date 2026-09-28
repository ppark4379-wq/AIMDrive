"""Analytic camera projection and cluster/track association contracts."""
from pathlib import Path
import sys
import unittest
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.geometry.calibration import CameraCalibration
from gps_jamming.geometry.projection import project_cluster
from gps_jamming.lidar.features import build_clusters
from gps_jamming.pipeline import AssociationPipeline
from gps_jamming.types import CameraTrack, LidarFrame


def calibration():
    # base forward/left/up -> optical right/down/forward, plus camera translation.
    transform = np.array([[0, -1, 0, 1], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    return CameraCalibration([[100, 0, 50], [0, 100, 50], [0, 0, 1]],
                             transform, (100, 100), 'base_link')


def clusters(xyz):
    points = np.column_stack((xyz, np.arange(len(xyz))))
    return build_clusters(points, [np.arange(len(points))])


def frame(cs):
    return LidarFrame(7, 10, None, 20, 'base_link', True,
                      np.empty((0, 4)), 30, clusters=cs)


class AssociationTests(unittest.TestCase):
    def test_projection_transform_and_visibility(self):
        c = clusters([[10, 1, 0], [10, 0, 1], [-10, 0, 0], [0, 0, 0],
                      [10, -4, 0], [10, 6, 0]])[0]
        c.points = np.vstack((c.points, [np.nan, 0, 0, 1]))
        result = project_cluster(c, calibration())
        np.testing.assert_allclose(result.pixels, [[50, 50], [60, 40], [0, 50]])
        np.testing.assert_array_equal(result.point_indices, [0, 1, 5])
        np.testing.assert_allclose(result.depths, [10, 10, 10])

    def test_distortion(self):
        cal = calibration()
        cal.distortion = np.array([1, 0, 0, 0, 0])
        result = project_cluster(clusters([[10, 0, 0]])[0], cal)
        np.testing.assert_allclose(result.pixels, [[60.1, 50]])

    def test_best_overlap_and_preserve_cluster(self):
        cs = clusters([[10, 1, 0], [10, 0, 0], [10, -1, 0], [-10, 0, 0]])
        tracks = [CameraTrack(8, [49, 49, 61, 51], 2, .99),
                  CameraTrack(9, [49, 49, 71, 51], 2, .7)]
        result = AssociationPipeline(calibration()).process(frame(cs), tracks, image_size=(100, 100))
        self.assertEqual(len(result.matches), 1)
        match = result.matches[0]
        self.assertEqual((match.track_id, match.inside_points, match.visible_points), (9, 3, 3))
        self.assertEqual(match.overlap_ratio, 1)
        self.assertIs(match.cluster_points, cs[0].points)
        np.testing.assert_array_equal(match.centroid, cs[0].centroid)
        self.assertEqual(result.unmatched_track_ids, [8])
        self.assertEqual(match.coordinate_frame, 'base_link')

    def test_thresholds_empty_and_many_to_one(self):
        cs = clusters([[10, 1, 0], [10, 0, 0], [10, -1, 0], [10, -2, 0]])
        tracks = [CameraTrack(1, [49, 49, 61, 51], 2, .8)]
        pipeline = AssociationPipeline(calibration(), {'min_inside_points': 2, 'min_overlap_ratio': .5})
        cs.append(replace(cs[0], cluster_id=1))
        result = pipeline.process(frame(cs), tracks, image_size=(100, 100))
        self.assertEqual([m.track_id for m in result.matches], [1, 1])
        self.assertEqual(result.matches[0].overlap_ratio, .5)
        pipeline.config['min_overlap_ratio'] = .51
        self.assertEqual(pipeline.process(frame(cs), tracks, image_size=(100, 100)).unmatched_cluster_ids, [0, 1])
        pipeline.config = {'min_inside_points': 3}
        self.assertEqual(pipeline.process(frame(cs), tracks, image_size=(100, 100)).matches, [])
        self.assertEqual(pipeline.process(frame(cs), [], image_size=(100, 100)).unmatched_cluster_ids, [0, 1])
        self.assertEqual(pipeline.process(frame([]), tracks, image_size=(100, 100)).unmatched_track_ids, [1])
        behind = clusters([[-10, 0, 0]])
        self.assertEqual(pipeline.process(frame(behind), tracks, image_size=(100, 100)).matches, [])

    def test_tie_break_is_independent_of_track_order(self):
        cs = clusters([[10, 1, 0], [10, 0, 0], [10, -1, 0]])
        tracks = [CameraTrack(2, [40, 40, 80, 60], 2, .9), CameraTrack(1, [40, 40, 80, 60], 2, .9)]
        pipeline = AssociationPipeline(calibration())
        for order in [tracks, tracks[::-1]]:
            self.assertEqual(pipeline.process(frame(cs), order, image_size=(100, 100)).matches[0].track_id, 1)

    def test_invalid_contracts(self):
        pipeline = AssociationPipeline(calibration())
        with self.assertRaisesRegex(ValueError, 'expects'):
            pipeline.process(replace(frame([]), coordinate_frame='lidar'), [], image_size=(100, 100))
        with self.assertRaisesRegex(ValueError, 'resolution'):
            pipeline.process(frame([]), [], image_size=(200, 100))
        with self.assertRaisesRegex(ValueError, 'Missing verified'):
            AssociationPipeline({'intrinsic': None})
        with self.assertRaises(ValueError):
            replace(calibration(), extrinsic=np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            CameraTrack(1, [10, 10, 0, 0], 2, .9)
        track = CameraTrack(1, [0, 0, 10, 10], 2, .9)
        with self.assertRaisesRegex(ValueError, 'unique'):
            pipeline.process(frame([]), [track, track], image_size=(100, 100))


if __name__ == '__main__':
    unittest.main()
