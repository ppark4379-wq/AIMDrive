"""Audit records must preserve ambiguous, missing and unsynchronized measurements."""
from pathlib import Path
import sys
import unittest
from unittest.mock import mock_open, patch
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.types import CameraTrack, AssociatedCluster, AssociationResult, LidarFrame
from gps_jamming.association.observations import (
    track_observations, position_labels, frame_observation, ObservationLog)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.tracks = [CameraTrack(2, [10, 10, 90, 90], 2, .9),
                       CameraTrack(3, [100, 10, 190, 90], 0, .8)]
        self.match = AssociatedCluster(2, 2, .9, 7, np.ones((10, 4)),
                                       np.array([10., -1., .5]), 'base_link', 8, 10, .8)
        self.result = AssociationResult(1, None, [self.match], [], [], [3])

    def test_positions_counts_and_unmatched(self):
        rows = track_observations(self.tracks, self.result)
        self.assertEqual(rows[0]['status'], 'matched')
        self.assertEqual(rows[0]['clusters'][0]['centroid_xyz_m'], [10., -1., .5])
        self.assertEqual(rows[0]['clusters'][0]['point_count'], 10)
        self.assertEqual(position_labels(rows[0])[1], 'Cluster 7 | XY range --')
        self.assertEqual(rows[1]['status'], 'unmatched')
        self.assertEqual(rows[1]['clusters'], [])

    def test_multiple_candidates_are_not_silently_merged(self):
        self.result.matches.append(self.match)
        row = track_observations(self.tracks, self.result)[0]
        self.assertEqual(row['status'], 'multiple_clusters')
        self.assertEqual(len(row['clusters']), 2)
        self.assertEqual(len(position_labels(row)), 4)

    def test_clock_domains_and_missing_data_remain_explicit(self):
        frame = LidarFrame(1, 123, None, 456, 'base_link', True, np.empty((0, 4)), 789)
        row = frame_observation(5, (12, 34), 1.5, frame, self.tracks,
                                self.result, 'preview_unsynchronized', {})
        self.assertFalse(row['synchronized'])
        self.assertIsNone(row['acquisition_delta_ns'])
        self.assertIsNone(row['lidar']['timestamp_unix_ns'])
        self.assertEqual(row['lidar']['raw_timestamp'], 123)
        self.assertEqual(row['camera_receive_monotonic_ns'], 1500000000)
        json.dumps(row, allow_nan=False)
        missing = frame_observation(6, (12, 35), 1.6, None, self.tracks, None, 'no_lidar', {})
        self.assertIsNone(missing['lidar'])
        self.assertEqual(missing['tracks'][0]['status'], 'unavailable')
        self.assertEqual(missing['tracks'][0]['clusters'], [])

    def test_jsonl_append_and_exclusive_file_creation(self):
        handle = mock_open()
        with patch.object(Path, 'mkdir'), patch.object(Path, 'open', handle):
            log = ObservationLog('unused')
            log.write({'camera_frame_id': 1})
            log.write({'camera_frame_id': 2})
            log.close()
        handle.assert_called_once_with('x', encoding='utf-8', buffering=1)
        lines = [call.args[0] for call in handle().write.call_args_list]
        self.assertEqual([json.loads(line)['camera_frame_id'] for line in lines], [1, 2])
        self.assertTrue(all(line.endswith('\n') for line in lines))
        handle().close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
