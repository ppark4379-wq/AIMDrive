from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from camera_mot.pipeline import CameraPipeline
from camera_mot.streams import CameraFrame
from camera_mot.replay import ReplaySource
from tools.morai_multi_camera_mot import make_parser


class FakeDetector:
    names = ['person', 'bicycle', 'car']

    def __init__(self):
        self.rows = np.array([[10, 10, 30, 30, .9, 2]], dtype=np.float32)

    def detect(self, image):
        return self.rows.copy()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.args = make_parser().parse_args(['--weights', 'unused', '--run-dir', 'unused', '--cmc-method', 'none'])
        self.detector = FakeDetector()
        self.cameras = {key: {'frame_id': key, 'image_size': [100, 80], 'expected_hz': 20}
                        for key in ('front', 'left', 'right')}
        self.pipeline = CameraPipeline(self.detector, self.cameras, self.args, 'test-session')
        ok, encoded = cv2.imencode('.jpg', np.zeros((80, 100, 3), np.uint8))
        self.assertTrue(ok)
        self.frame = CameraFrame('front', 'front', (100, 123456789), 1, 1, 0, encoded.tobytes(), 'udp')

    def test_initializing_or_resetting_other_tracker_never_reuses_front_id(self):
        first, _ = self.pipeline.process(self.frame)
        self.assertEqual(first['tracks'][0]['track_id'], 1)
        self.pipeline.process(replace(self.frame, camera_id='left'))
        self.pipeline.process(replace(self.frame, camera_id='left', stream_epoch=1))
        self.detector.rows = np.array([[10, 10, 30, 30, .9, 2], [60, 40, 90, 70, .9, 0]], np.float32)
        self.pipeline.process(replace(self.frame, stamp=(101, 0)))
        result, _ = self.pipeline.process(replace(self.frame, stamp=(102, 0)))
        self.assertEqual({t['track_id'] for t in result['tracks']}, {1, 2})
        self.assertEqual(result['session_id'], 'test-session')
        self.assertEqual(result['transport'], 'udp')
        self.assertEqual(result['timestamp'], {'sec': 102, 'nanosec': 0})

    def test_empty_detection_advances_tracker_and_short_gap_recovers_same_id(self):
        self.pipeline.process(self.frame)
        self.detector.rows = np.empty((0, 6), dtype=np.float32)
        result, _ = self.pipeline.process(replace(self.frame, stamp=(100, 173456789)))
        self.assertEqual(result['tracks'], [])
        self.assertEqual(self.pipeline.states['front']['tracker'].frame_id, 2)
        self.detector.rows = np.array([[10, 10, 30, 30, .9, 2]], dtype=np.float32)
        result, _ = self.pipeline.process(replace(self.frame, stamp=(100, 223456789)))
        self.assertEqual(result['tracks'][0]['track_id'], 1)
        self.assertAlmostEqual(result['source_dt_s'], .05)

    def test_tentative_duplicate_is_not_published_but_raw_detection_is_retained(self):
        first, _ = self.pipeline.process(self.frame)
        original_id = first['tracks'][0]['track_id']
        # Two overlapping candidates for one object survive detector NMS once.
        self.detector.rows = np.array([[10, 10, 30, 30, .9, 2],
                                      [18, 10, 38, 30, .8, 2]], np.float32)
        result, _ = self.pipeline.process(replace(self.frame, stamp=(100, 173456789)))
        self.assertEqual([t['track_id'] for t in result['tracks']], [original_id])
        self.assertEqual(len(result['detections']), 2)
        self.assertEqual(len(self.pipeline.states['front']['tracker'].tracked_stracks), 2)
        self.detector.rows = self.detector.rows[:1]
        result, _ = self.pipeline.process(replace(self.frame, stamp=(100, 223456789)))
        self.assertEqual([t['track_id'] for t in result['tracks']], [original_id])

    def test_real_new_object_is_published_after_confirmation_with_same_internal_id(self):
        self.pipeline.process(self.frame)
        self.detector.rows = np.array([[10, 10, 30, 30, .9, 2],
                                      [60, 40, 90, 70, .9, 0]], np.float32)
        second, _ = self.pipeline.process(replace(self.frame, stamp=(100, 173456789)))
        self.assertEqual([t['track_id'] for t in second['tracks']], [1])
        third, _ = self.pipeline.process(replace(self.frame, stamp=(100, 223456789)))
        self.assertEqual({t['track_id'] for t in third['tracks']}, {1, 2})
        self.assertTrue(all(t['confirmed'] for t in third['tracks']))

    def test_diagnostic_option_preserves_tentative_output(self):
        self.args.include_tentative_tracks = True
        self.pipeline.process(self.frame)
        self.detector.rows = np.array([[10, 10, 30, 30, .9, 2],
                                      [60, 40, 90, 70, .9, 0]], np.float32)
        result, _ = self.pipeline.process(replace(self.frame, stamp=(100, 173456789)))
        self.assertEqual([(t['track_id'], t['confirmed']) for t in result['tracks']],
                         [(1, True), (2, False)])

    def test_timestamp_monotonicity_and_explicit_epoch(self):
        self.pipeline.process(self.frame)
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            self.pipeline.process(self.frame)
        result, _ = self.pipeline.process(replace(self.frame, stamp=(1, 0), stream_epoch=1))
        self.assertEqual((result['stream_epoch'], result['frame_id']), (1, 1))

    def test_one_class_outlier_keeps_identity_but_repeated_evidence_can_reclassify(self):
        for i in range(3):
            result, _ = self.pipeline.process(replace(self.frame, stamp=(100+i, 0)))
        original_id = result['tracks'][0]['track_id']
        self.detector.rows[0, 5] = 0  # Same box, one erroneous person classification.
        result, _ = self.pipeline.process(replace(self.frame, stamp=(103, 0)))
        self.assertEqual(result['tracks'][0]['class_name'], 'car')
        self.assertEqual(result['tracks'][0]['track_id'], original_id)
        for i in range(104, 108):
            result, _ = self.pipeline.process(replace(self.frame, stamp=(i, 0)))
        self.assertEqual(result['tracks'][0]['class_name'], 'person')
        self.assertEqual(result['tracks'][0]['track_id'], original_id)

    def test_ego_mask_only_changes_inference_input_and_preserves_coordinates(self):
        self.cameras['front']['ego_mask_polygons'] = [[[0, 60], [99, 60], [99, 79], [0, 79]]]
        seen = []
        self.detector.detect = lambda image: (seen.append(image.copy()) or self.detector.rows.copy())
        result, original = self.pipeline.process(self.frame)
        self.assertTrue(np.all(seen[0][65, 50] == 114))
        self.assertTrue(np.all(original[65, 50] == 0))
        self.assertTrue(np.all(seen[0][20, 50] == original[20, 50]))
        self.assertEqual(result['tracks'][0]['bbox_xyxy'], [10, 10, 30, 30])
        self.assertEqual(result['ego_mask_polygon_count'], 1)

    def test_reject_invalid_jpeg_resolution_and_nonfinite_detection(self):
        with self.assertRaisesRegex(ValueError, 'JPEG'):
            self.pipeline.process(replace(self.frame, jpeg=b'bad'))
        self.cameras['front']['image_size'] = [10, 10]
        with self.assertRaisesRegex(ValueError, 'image size'):
            self.pipeline.process(self.frame)
        self.cameras['front']['image_size'] = [100, 80]
        self.detector.rows[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.pipeline.process(self.frame)

    def test_saved_capture_replays_exact_camera_timestamp_and_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'image.jpg').write_bytes(self.frame.jpeg)
            row = dict(camera_id='front', source_frame_id='Camera-1', stamp=[100, 123456789],
                       received_monotonic_ns=1, received_frame_id=7, stream_epoch=3,
                       image_path='image.jpg', transport='udp', timestamp_domain='morai_udp_header_raw')
            (root / 'frames.jsonl').write_text(json.dumps(row) + '\n')
            replay = ReplaySource(root, self.cameras)
            try:
                frame = replay.get()
                self.assertEqual(frame.stamp, (100, 123456789))
                self.assertEqual(frame.stream_epoch, 3)
                self.assertEqual(frame.received_frame_id, 7)
                self.assertEqual(frame.jpeg, self.frame.jpeg)
                self.assertIsNone(replay.get())
                self.assertTrue(replay.closed)
            finally:
                replay.close()


if __name__ == '__main__':
    unittest.main()
