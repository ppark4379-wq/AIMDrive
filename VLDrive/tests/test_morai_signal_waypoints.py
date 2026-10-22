"""CPU checks for signal-context admission and timed future holding diagnostics."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from utils.evaluation.morai_signal_waypoint_metrics import (
    SignalWaypointMetricsAccumulator, annotate_entry, load_arrow_transition,
)


def transition(lower=12_000_000_000, upper=12_060_000_000):
    return dict(light_id='C1256W000052', from_color=16, to_color=33,
                lower_source_ns=lower, upper_source_ns=upper)


def entry(frame=0, stamp=8_000_000_000, raw=1, speed=0., waypoints=None):
    permission = raw == 33
    return dict(frame_id=frame, sensor_time_ns=stamp,
        ego_pose=dict(speed=speed), waypoints=waypoints or [[0., 0.]] * 5,
        traffic_light={1: 0, 16: 1, 33: -1}[raw],
        route_permission=dict(valid=True, light_id='C1256W000052',
                              signal_movement='left', raw_signal_bits=raw, target=permission),
        traffic_signal=dict(light_id='C1256W000052', signal_movement='left',
                            raw_signal_bits=raw, permission_valid=True, signal_permission=permission))


def batch(entries, predictions=None, color=None):
    gt = torch.tensor([row['waypoints'] for row in entries], dtype=torch.float32)
    targets = dict(waypoints=gt, waypoint_valid=torch.ones(len(entries), dtype=torch.bool),
        frame_id=torch.tensor([row['frame_id'] for row in entries]),
        sensor_time_ns=torch.tensor([row['sensor_time_ns'] for row in entries]),
        traffic_light=torch.tensor([row['traffic_light'] for row in entries]))
    outputs = (torch.zeros(len(entries), 1),
               gt.clone() if predictions is None else torch.tensor(predictions, dtype=torch.float32),
               torch.tensor(color or [[0., 1.]] * len(entries)))
    inputs = dict(velocity=torch.tensor([[row['ego_pose']['speed']] for row in entries]))
    return outputs, targets, inputs


def sealed_audit(episode, events):
    folder = episode / 'traffic_audit_v1'
    folder.mkdir()
    report = dict(schema='morai_sensor_record_direct_traffic_audit_v1',
                  structural_valid=True, training_visible=False, raw_manifest_sha256='a' * 64)
    (folder / 'report.json').write_text(json.dumps(report))
    (folder / 'transitions.json').write_text(json.dumps(events))
    files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
             for name in ('report.json', 'transitions.json')}
    (folder / 'COMPLETE.json').write_text(json.dumps(dict(files=files, raw_manifest_sha256='a' * 64)))
    return folder


def event(signal='C1256W000052', source=16, target=33, interval=None):
    return dict(id=signal, from_color=source, to_color=target,
                uncertainty_source_ns=interval or [12_000_000_000, 12_060_000_000],
                sensor_anchor_indices=[40])


class ArrowEvidenceTest(unittest.TestCase):
    def test_exact_transition_is_verified_with_other_signal_events_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp)
            sealed_audit(episode, [event(signal='other'), event(source=1, target=16), event()])
            loaded = load_arrow_transition(episode)
            self.assertEqual(loaded['lower_source_ns'], 12_000_000_000)
            self.assertEqual(loaded['upper_source_ns'], 12_060_000_000)
            self.assertEqual(loaded['sensor_anchor_indices'], [40])
            self.assertEqual(len(loaded['seal_sha256']), 64)
            json.dumps(loaded, allow_nan=False)

    def test_wrong_signal_wrong_direction_or_multiple_events_are_rejected(self):
        cases = ([event(signal='C1256W000123')], [event(source=33, target=16)],
                 [event(), event(interval=[20_000_000_000, 20_100_000_000])])
        for events in cases:
            with self.subTest(events=events), tempfile.TemporaryDirectory() as tmp:
                sealed_audit(Path(tmp), events)
                with self.assertRaisesRegex(ValueError, 'exactly one'):
                    load_arrow_transition(Path(tmp))

    def test_ambiguous_or_malformed_intervals_are_rejected(self):
        for interval in ([12, 12], [13, 12], [12, 13, 14], [12., 13]):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as tmp:
                sealed_audit(Path(tmp), [event(interval=interval)])
                with self.assertRaises(ValueError):
                    load_arrow_transition(Path(tmp))

    def test_modified_transition_bytes_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = sealed_audit(Path(tmp), [event()])
            (folder / 'transitions.json').write_text(json.dumps([event(interval=[1, 2])]))
            with self.assertRaisesRegex(ValueError, 'Changed sealed traffic evidence'):
                load_arrow_transition(Path(tmp))


class SignalWaypointDiagnosticTest(unittest.TestCase):
    def test_red_approach_and_launch_are_distinguished_from_holding(self):
        rows = [entry(frame=1, speed=2.), entry(frame=2, speed=.1),
                entry(frame=3, raw=33, speed=.2), entry(frame=4, raw=33, speed=.5)]
        groups = [annotate_entry(row, transition())['group'] for row in rows]
        self.assertEqual(groups, ['target_red_moving', 'target_red_stopped',
                                  'left_arrow_startup', 'left_arrow_moving'])

    def test_invalid_permission_and_wrong_movement_are_other_context(self):
        for mutation in ('invalid', 'wrong_signal', 'wrong_movement'):
            row = entry()
            if mutation == 'invalid':
                row['route_permission']['valid'] = False
            elif mutation == 'wrong_signal':
                row['route_permission']['light_id'] = 'C1256W000123'
            else:
                row['route_permission']['signal_movement'] = 'right_unprotected'
            with self.subTest(mutation=mutation):
                annotated = annotate_entry(row, transition())
                self.assertEqual(annotated['group'], 'other_context')
                self.assertFalse(annotated['bound_left_permission_valid'])

    def test_inconsistent_color_binding_is_rejected(self):
        row = entry(raw=16)
        row['traffic_signal']['raw_signal_bits'] = 33
        with self.assertRaisesRegex(ValueError, 'GT disagree'):
            annotate_entry(row, transition())

    def test_green_future_departure_is_not_a_prerelease_holding_error(self):
        # The last two points fall at/after possible release. Their nonzero
        # forecast errors remain visible in ordinary ADE, but not holding L2.
        row = entry(stamp=10_000_000_000, raw=16,
                    waypoints=[[0., 0.], [0., 0.], [0., 0.], [0., -1.], [0., -2.]])
        prediction = [[0., 0.], [0., 0.], [0., 0.], [0., -2.], [0., -4.]]
        accumulator = SignalWaypointMetricsAccumulator([row], transition())
        accumulator.update(*batch([row], [prediction]))
        report = accumulator.compute()
        self.assertEqual(accumulator.rows[0]['group'], 'green_horizon_overlaps_release')
        self.assertEqual(accumulator.rows[0]['waypoint_before_possible_release'],
                         [True, True, True, False, False])
        self.assertAlmostEqual(accumulator.rows[0]['ade_m'], .6)
        self.assertEqual(report['green_before_release']['masked_point_count'], 3)
        self.assertEqual(report['green_before_release']['masked_point_l2_sum_m'], 0.)
        self.assertEqual(report['stable_holds']['frames'], 0)

    def test_batch_sensitivity_counts_first_point_separately_and_masks_arrow_color(self):
        red, green, startup, moving = (entry(frame=1), entry(frame=2, raw=16),
            entry(frame=3, raw=33, speed=.2, waypoints=[[0., -1.]] * 5),
            entry(frame=4, raw=1, speed=.1, waypoints=[[0., -2.]] * 5))
        predictions = [[[0., 0.]] + [[.2, 0.]] * 4,
                       [[.6, 0.]] * 5, startup['waypoints'], moving['waypoints']]
        rows = [red, green, startup, moving]
        accumulator = SignalWaypointMetricsAccumulator(rows, transition())
        accumulator.update(*batch(rows, predictions))
        report = accumulator.compute()
        self.assertEqual(report['stable_holds']['frames'], 2)
        sensitivity = report['stable_holds']['displacement_sensitivity']
        self.assertEqual(sensitivity['0.1']['any_waypoint_exceeds_count'], 2)
        self.assertEqual(sensitivity['0.1']['first_waypoint_exceeds_count'], 1)
        self.assertEqual(sensitivity['0.5']['any_waypoint_exceeds_count'], 1)
        self.assertEqual(sensitivity['1.0']['all_waypoints_within_count'], 2)
        self.assertEqual(report['groups']['left_arrow_startup']['color_observed_frames'], 0)
        self.assertEqual(report['green_before_release']['masked_point_count'], 5)
        self.assertEqual(len(accumulator.rows[0]['predicted_color_probabilities']), 2)
        self.assertEqual(accumulator.rows[2]['input_velocity_mps'], float(torch.tensor(.2)))
        json.dumps(dict(report=report, rows=accumulator.rows), allow_nan=False)

    def test_exact_batch_order_and_gt_must_match_admitted_entries(self):
        rows = [entry(frame=2), entry(frame=1)]
        for corruption in ('frame', 'time', 'waypoint'):
            accumulator = SignalWaypointMetricsAccumulator(rows, transition())
            outputs, targets, inputs = batch(rows)
            if corruption == 'frame':
                targets['frame_id'][1] = 7
            elif corruption == 'time':
                targets['sensor_time_ns'][1] += 1
            else:
                targets['waypoints'][1, 4, 0] = 1.
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, 'differs from batch'):
                accumulator.update(outputs, targets, inputs)
            self.assertEqual(accumulator.rows, [])

    def test_incomplete_evaluation_and_nonfinite_outputs_are_rejected(self):
        rows = [entry(frame=1), entry(frame=2)]
        accumulator = SignalWaypointMetricsAccumulator(rows, transition())
        accumulator.update(*batch(rows[:1]))
        with self.assertRaisesRegex(ValueError, 'length differs'):
            accumulator.compute()
        outputs, targets, inputs = batch(rows[1:])
        outputs[1][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            accumulator.update(outputs, targets, inputs)


if __name__ == '__main__':
    unittest.main()
