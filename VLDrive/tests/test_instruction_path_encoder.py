"""Geometry tests also runnable without torch: python -m unittest discover ..."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from instruction.global_path.preprocess.process import process_path
from instruction.global_path.encoder import PathEncoder, PathFeatureCache
from instruction.navigation.selector import select_instruction
from data.src.processing.build_path_features import build_path_features


class PathGeometryTests(unittest.TestCase):
    def test_straight_signs_distance_and_invalid(self):
        encoder = PathEncoder(process_path(dict(points=[[0, 0], [150, 0]])))
        p = encoder.project(dict(x=20, y=2, yaw_rad=.1))
        self.assertAlmostEqual(p['progress_m'], 20)
        self.assertAlmostEqual(p['lat_err'], 2)
        f = encoder.encode(p)
        np.testing.assert_allclose(f['ego_relation'], [.4, math.sin(.1), math.cos(.1)])
        np.testing.assert_allclose(f['path_context'], [0] * 5)
        self.assertTrue(f['path_valid'])
        self.assertEqual(select_instruction(encoder, [p])['kind'], 'straight')
        self.assertIsNone(encoder.project(dict(x=20, y=50, yaw_rad=0)))
        self.assertFalse(encoder.encode(None)['path_valid'])

    def test_event_distances_are_frame_specific_and_signed(self):
        encoder = PathEncoder(process_path(dict(points=[[0, 0], [150, 0]])))
        event = dict(start_m=50., end_m=70., curv_event=-.05)
        a = encoder.encode(encoder.project(dict(x=20, y=0, yaw_rad=0)), event)
        b = encoder.encode(encoder.project(dict(x=55, y=0, yaw_rad=0)), event)
        self.assertAlmostEqual(a['path_context'][0] * 60, 30)
        self.assertAlmostEqual(b['path_context'][0] * 60, -5)
        self.assertEqual(b['path_context'][-1], 1)

    def test_rigid_transform_does_not_change_features(self):
        points = np.array([[0., 0.], [100., 0.]])
        yaw = .7
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        shift = np.array([123., -52.])
        a = PathEncoder(process_path(dict(points=points.tolist())))
        b = PathEncoder(process_path(dict(points=(points @ rotation.T + shift).tolist())))
        xy = np.array([20., 1.]) @ rotation.T + shift
        fa = a.encode(a.project(dict(x=20., y=1., yaw_rad=.2)))
        fb = b.encode(b.project(dict(x=xy[0], y=xy[1], yaw_rad=yaw + .2)))
        np.testing.assert_allclose(fa['ego_relation'], fb['ego_relation'], atol=1e-8)
        np.testing.assert_allclose(fa['path_context'], fb['path_context'], atol=1e-8)

    def test_left_right_bends_and_duplicate_vertices(self):
        angle = np.linspace(0, math.pi / 2, 101)
        left = np.column_stack((20 * np.sin(angle), 20 * (1 - np.cos(angle))))
        for sign, expected in [(1, 'left'), (-1, 'right')]:
            points = left * [1, sign]
            route = process_path(dict(points=np.vstack((points[0], points)).tolist()))
            self.assertTrue(route['events'])
            self.assertEqual(route['events'][0]['direction'], expected)
            self.assertGreater(sign * route['events'][0]['curv_event'], 0)

    def test_closed_progress_crosses_seam(self):
        angle = np.linspace(0, 2 * math.pi, 241)
        points = np.column_stack((20 * np.cos(angle), 20 * np.sin(angle)))
        points[-1] = points[0]
        route = process_path(dict(points=points.tolist(), closed=True))
        encoder = PathEncoder(route)
        before = encoder.project(dict(x=20 * math.cos(-.04), y=20 * math.sin(-.04), yaw_rad=math.pi / 2 - .04))
        after = encoder.project(dict(x=20 * math.cos(.04), y=20 * math.sin(.04), yaw_rad=math.pi / 2 + .04), before['progress_m'])
        self.assertGreater(after['progress_m'], before['progress_m'])
        event = encoder.next_event(after['progress_m'])
        self.assertIsNotNone(event)
        self.assertGreaterEqual(event['end_m'], after['progress_m'])

    def test_selection_does_not_read_future_ego_positions(self):
        encoder = PathEncoder(process_path(dict(points=[[0, 0], [100, 0], [100, 100]])))
        first = encoder.project(dict(x=70, y=0, yaw_rad=0))
        early = encoder.project(dict(x=80, y=0, yaw_rad=0))
        late = encoder.project(dict(x=100, y=50, yaw_rad=math.pi / 2))
        self.assertEqual(select_instruction(encoder, [first, early]), select_instruction(encoder, [first, late]))
        self.assertEqual(select_instruction(encoder, [first])['kind'], 'follow')

    def test_invalid_geometry_and_options(self):
        for points in ([[0, 0], [0, 0]], [[0, 0], [float('nan'), 1]]):
            with self.assertRaises(ValueError):
                process_path(dict(points=points))
        with self.assertRaises(ValueError):
            process_path(dict(points=[[0, 0], [10, 0]]), spacing_m=0)

    def test_cache_timestamp_join_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = dict(episode_id='drive', route=dict(points=[[0, 0], [100, 0]]),
                           frames=[dict(index=i, sensor_source_ns=i * 100_000_000, x=i, y=0, yaw_rad=0,
                                        future_trajectory='MUST_NOT_ENTER_FEATURES') for i in range(5)])
            content = json.dumps(episode).encode()
            (root / 'episode.json').write_bytes(content)
            manifest = dict(schema='lmdrive-navigation-labels-v1', episodes=[dict(episode_id='drive',
                episode='episode.json', episode_sha256=hashlib.sha256(content).hexdigest())])
            (root / 'manifest.json').write_text(json.dumps(manifest))
            target = root / 'features.json'
            build_path_features(root, target)
            digest = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
            cache = PathFeatureCache(target, digest)
            self.assertTrue(cache.eligible('drive', [0, 100_000_000, 200_000_000]))
            self.assertFalse(cache.eligible('drive', [0, 200_000_000]))
            text, features = cache.sample('drive', [0, 200_000_000, 400_000_000])
            self.assertEqual(text['kind'], 'straight')
            self.assertEqual(len(features['ego_relation']), 3)
            self.assertNotIn('MUST_NOT_ENTER_FEATURES', target.read_text())
            with self.assertRaises(ValueError):
                PathFeatureCache(target, 'wrong source')
            with self.assertRaises(FileExistsError):
                build_path_features(root, target)


if __name__ == '__main__':
    unittest.main()
