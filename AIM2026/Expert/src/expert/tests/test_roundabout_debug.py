"""Diagnostics must explain decisions without changing them."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.lattice import parameters as p
from expert.src.planning.roundabout import diagnostics as debug
from expert.src.planning.roundabout.planner import RoundaboutEntryPlanner
from expert.src.vehicle_state import VehicleState


class MemoryWriter:
    failed = False

    def __init__(self):
        self.records = []

    def submit(self, record):
        self.records.append(json.loads(json.dumps(debug._json_safe(record), allow_nan=False)))
        return True


class RoundaboutDebugTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(p, 'ROUNDABOUT_DEBUG_ENABLED', True).start()
        patch.object(p, 'ROUNDABOUT_ENTRY_HOLD_POINT', (0., 0.)).start()
        patch.object(p, 'ROUNDABOUT_ENTRY_MERGE_POINT', (10., 0.)).start()
        self.sink = MemoryWriter()
        patch.object(debug, '_writer', self.sink).start()
        patch.object(debug, '_input_context', {}).start()
        self.planner = RoundaboutEntryPlanner()
        self.path = [Point(float(x), 0.) for x in range(-20, 201)]
        self.stdout = io.StringIO()
        contextlib_redirect = contextlib.redirect_stdout(self.stdout)
        contextlib_redirect.__enter__()
        self.addCleanup(contextlib_redirect.__exit__, None, None, None)

    @staticmethod
    def obj(x, y, speed=6., heading=0., identity=1, kind=1, static=False):
        return ObjectInfo(x, y, speed, kind, unique_id=identity, heading_deg=heading,
                          size_x=4.5, size_y=2., is_static=static)

    def run_plan(self, x=-5., objects=(), planner=None):
        return (planner or self.planner).plan(
            VehicleState(x, 0., 0., 5.), self.path,
            [{'object_info': obj} for obj in objects], {'in_approach': True}, 7.)

    def test_filter_evidence_and_input_metadata(self):
        debug.input_context(ego_seq=42, source='ros', object_delta_s=None, object_count=5)
        objects = [self.obj(20., 0., identity=1), self.obj(200., 0., identity=2),
                   self.obj(65., 0., identity=3), self.obj(3., 0., identity=4, kind=0),
                   self.obj(4., 0., identity=5, static=True)]
        self.run_plan(objects=objects)
        record = self.sink.records[-1]
        self.assertEqual(42, record['input']['ego_seq'])
        self.assertEqual(['base_range', 'outside_max_distance', 'rear_direction_not_confirmed',
                          'static_or_non_vehicle', 'static_or_non_vehicle'],
                         [obj['filter'] for obj in record['objects']])
        self.assertEqual(['1'], record['tracked_ids_added'])
        self.assertEqual(p.ROUNDABOUT_REAR_TRACK_DISTANCE, record['parameters']['ROUNDABOUT_REAR_TRACK_DISTANCE'])
        self.assertTrue(record['checks'])
        self.assertTrue(record['checks'][0]['ego_samples'])
        self.assertTrue(record['checks'][0]['closest_reference_distance'])
        self.assertEqual([0., 2., 4., 6., 8.], [x['t_s'] for x in record['object_predictions'][0]['samples']])

    def test_periodic_sampling_and_boundary_events(self):
        with patch.object(debug.time, 'monotonic', side_effect=[10., 10.05, 10.06, 10.07, 10.30]):
            self.run_plan(x=-.48)
            self.run_plan(x=-.47)
            self.assertEqual(1, len(self.sink.records))
            self.run_plan(x=.48)
            record = self.sink.records[-1]
            self.assertTrue(record['event'])
            self.assertFalse(record['before']['entering'])
            self.assertTrue(record['decision']['entering'])
            self.run_plan(x=.49)
            self.assertEqual(2, len(self.sink.records))
            self.run_plan(x=.50)
            self.assertEqual(3, len(self.sink.records))
            self.assertFalse(self.sink.records[-1]['event'])

    def test_object_disappearance_is_logged_immediately(self):
        with patch.object(debug.time, 'monotonic', side_effect=[10., 10.01]):
            self.run_plan(objects=[self.obj(-10., 10., identity=50)])
            self.run_plan()
        record = self.sink.records[-1]
        self.assertTrue(record['event'])
        self.assertEqual(['50'], record['input_ids_removed'])
        self.assertEqual(['50'], record['tracked_ids_removed'])

    def test_enabled_capture_preserves_entire_planning_output(self):
        baseline = RoundaboutEntryPlanner()
        scenes = [(-10., [self.obj(25., 0.)]), (-.48, [self.obj(4., -3., 4., 90.)]),
                  (.48, [self.obj(-10., 0., 12.)]), (11., []), (12., [])]
        for x, objects in scenes:
            with self.subTest(x=x):
                with patch.object(p, 'ROUNDABOUT_DEBUG_ENABLED', False):
                    expected = self.run_plan(x, objects, baseline)
                actual = self.run_plan(x, objects)
                def snapshot(result):
                    return json.dumps(debug._json_safe({
                        key: value for key, value in vars(result).items()
                        if key not in ('path', 'predicted_object_paths', 'roundabout_debug_frame')
                    }), sort_keys=True, allow_nan=False)
                self.assertEqual(snapshot(expected), snapshot(actual))
                for expected_path, actual_path in zip(expected.predicted_object_paths, actual.predicted_object_paths):
                    self.assertEqual([(pt.x, pt.y) for pt in expected_path], [(pt.x, pt.y) for pt in actual_path])

    def test_control_records_link_to_plan_and_capture_failure_does_not_stop_planning(self):
        result = self.run_plan()
        result.debug_details['roundabout_entry']['speed_limits'] = {'path_speed': 7.}
        debug.record_control(result, 7., .3, 1./30., -.2)
        self.assertEqual(self.sink.records[-2]['frame'], self.sink.records[-1]['frame'])
        self.assertEqual('control', self.sink.records[-1]['kind'])
        self.assertEqual(.3, self.sink.records[-1]['pedal_command'])
        with patch.object(self.planner.diagnostics, 'finish', side_effect=OSError('disk unavailable')):
            result = self.run_plan()
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertTrue(self.planner.diagnostics.disabled)

    def test_background_writer_flushes_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roundabout.jsonl'
            writer = debug._JsonWriter(str(path))
            self.assertTrue(writer.submit(dict(kind='test', value=math.inf)))
            writer.close()
            self.assertFalse(writer.failed)
            self.assertEqual([dict(kind='test', value=None, dropped_records=0)],
                             [json.loads(line) for line in path.read_text().splitlines()])


if __name__ == '__main__':
    unittest.main()
