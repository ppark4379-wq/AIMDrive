import json
import unittest

import numpy as np

from training.vision_encoder.utils.morai_route_permission_metrics import (
    RoutePermissionAccumulator, selection_decision,
)


def metadata(bits, movement='left', distance=10., scenario='traffic_light'):
    return dict(movement=movement, raw_signal_bits=bits, light_id='C1256W000052',
                stopline_distance_m=distance, scenario_id=scenario)


def evaluate(targets, predictions, rows=None):
    scores = np.full((len(targets), 2), -2.)
    for index, prediction in enumerate(predictions):
        scores[index, prediction] = 2.
    rows = rows or [metadata(33 if target == 1 else 16) for target in targets]
    result = RoutePermissionAccumulator()
    result.update(scores, np.array(targets), rows)
    return result.compute()


class RoutePermissionMetricsTests(unittest.TestCase):
    def test_perfect_deny_allow_and_phase_specific_metrics(self):
        report = evaluate([0, 0, 1, 1], [0, 0, 1, 1],
                          [metadata(1), metadata(16), metadata(33), metadata(33)])
        self.assertEqual(report['confusion_matrix'], [[2, 0], [0, 2]])
        self.assertEqual(report['balanced_accuracy'], 1.)
        self.assertEqual(report['classes']['deny']['precision'], 1.)
        self.assertEqual(report['classes']['allow']['recall'], 1.)
        self.assertEqual(report['critical_subgroups']['left_straight_green_16']['unsafe_allow_rate'], 0.)
        self.assertEqual(report['critical_subgroups']['left_arrow_33']['false_deny_rate'], 0.)
        self.assertTrue(selection_decision(report)['eligible'])
        self.assertFalse(selection_decision(report)['deployment_ready'])
        json.dumps(report, allow_nan=False)

    def test_always_deny_is_not_success_despite_zero_unsafe_allow(self):
        report = evaluate([0] * 99 + [1], [0] * 100)
        self.assertEqual(report['accuracy'], .99)
        self.assertEqual(report['unsafe_allow_rate'], 0.)
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertEqual(report['false_deny_rate'], 1.)
        self.assertEqual(report['constant_prediction'], 'always_deny')
        self.assertFalse(selection_decision(report)['eligible'])

    def test_rare_forbidden_false_go_is_exposed_despite_high_accuracy(self):
        report = evaluate([0, 0] + [1] * 98, [0, 1] + [1] * 98,
                          [metadata(1), metadata(16)] + [metadata(33)] * 98)
        self.assertEqual(report['accuracy'], .99)
        self.assertEqual(report['unsafe_allow'], 1)
        self.assertEqual(report['unsafe_allow_rate'], .5)
        self.assertEqual(report['critical_subgroups']['left_straight_green_16']['unsafe_allow_rate'], 1.)
        self.assertEqual(report['balanced_accuracy'], .75)

    def test_mask_minus_one_does_not_count_as_deny_or_as_error(self):
        acc = RoutePermissionAccumulator()
        acc.update(np.array([[5., 0.], [0., 5.], [np.nan, np.nan]]),
                   np.array([0, 1, -1]), [metadata(16), metadata(33), metadata(16)])
        report = acc.compute()
        self.assertEqual(report['frames'], 3)
        self.assertEqual(report['valid'], 2)
        self.assertEqual(report['excluded'], 1)
        self.assertEqual(report['accuracy'], 1.)
        self.assertEqual(report['critical_subgroups']['left_straight_green_16']['excluded'], 1)

    def test_tied_logits_predict_deny(self):
        acc = RoutePermissionAccumulator()
        acc.update(np.array([[0., 0.], [1., 1.]]), np.array([0, 1]), [metadata(16), metadata(33)])
        self.assertEqual(acc.compute()['confusion_matrix'], [[1, 0], [1, 0]])

    def test_no_critical_condition_is_unmeasured_and_ineligible(self):
        report = evaluate([0, 1], [0, 1], [metadata(1), metadata(33)])
        absent = report['critical_subgroups']['left_straight_green_16']
        self.assertEqual(absent['measurement_status'], 'unmeasured')
        self.assertIsNone(absent['unsafe_allow_rate'])
        self.assertFalse(selection_decision(report)['eligible'])
        self.assertIsNone(selection_decision(report)['score'])
        report = evaluate([0, 0], [0, 1])
        self.assertIsNone(report['balanced_accuracy'])
        self.assertFalse(selection_decision(report)['eligible'])

    def test_phase_label_conflict_is_explicit(self):
        report = evaluate([0, 1], [0, 1], [metadata(33), metadata(16)])
        decision = selection_decision(report)
        self.assertEqual(report['accuracy'], 1.)
        self.assertFalse(decision['eligible'])
        self.assertEqual(sum(row['reason'] == 'label_phase_conflict' for row in decision['violations']), 2)

    def test_group_partitions_sum_to_global_counts(self):
        rows = [metadata(1, 'left', 2., 'a'), metadata(16, 'left', 18., 'a'),
                metadata(33, 'left', 32., 'b'), metadata(16, 'straight', None, 'b')]
        report = evaluate([0, 0, 1, -1], [0, 1, 1, 0], rows)
        for key in ('by_movement', 'by_raw_signal_bits', 'by_light_id',
                    'by_stopline_distance', 'by_scenario_id'):
            groups = report[key].values()
            self.assertEqual(sum(group['frames'] for group in groups), report['frames'])
            self.assertEqual(sum(group['valid'] for group in groups), report['valid'])
            self.assertEqual(sum(group['excluded'] for group in groups), report['excluded'])
            matrices = np.sum([group['confusion_matrix'] for group in groups], axis=0)
            np.testing.assert_array_equal(matrices, report['confusion_matrix'])

    def test_multiple_batches_weight_by_frames_not_by_batch_accuracy(self):
        targets = np.array([0] + [1] * 9)
        scores = np.array([[0., 1.]] + [[0., 2.]] * 9)
        rows = [metadata(16)] + [metadata(33)] * 9
        together, split = RoutePermissionAccumulator(), RoutePermissionAccumulator()
        together.update(scores, targets, rows)
        split.update(scores[:1], targets[:1], rows[:1])
        split.update(scores[1:], targets[1:], rows[1:])
        self.assertEqual(together.compute(), split.compute())
        self.assertEqual(split.compute()['accuracy'], .9)
        self.assertEqual(split.compute()['balanced_accuracy'], .5)

    def test_missing_distance_or_phase_overlap_never_certifies_visual_use(self):
        report = evaluate([0, 1], [0, 1], [metadata(16, distance=40.), metadata(33, distance=2.)])
        self.assertEqual(report['distance_confound']['measurement_status'], 'no_measured_approach_overlap')
        self.assertEqual(report['distance_confound']['shared_approach_distance_bins'], [])
        same = evaluate([0, 1], [0, 1], [metadata(16, distance=2.), metadata(33, distance=3.)])
        self.assertEqual(same['distance_confound']['measurement_status'], 'coarse_overlap_observed')
        self.assertFalse(same['distance_confound']['confounding_ruled_out'])

    def test_baseline_comparison_requires_identical_inputs(self):
        baseline = evaluate([0, 1], [0, 1])
        candidate = evaluate([0, 1], [1, 0])
        self.assertEqual(selection_decision(candidate, baseline)['score'][0], 0.)
        moved = evaluate([0, 1], [0, 1], [metadata(16, distance=20.), metadata(33)])
        with self.assertRaisesRegex(ValueError, 'identical input'):
            selection_decision(moved, baseline)

    def test_input_validation_is_atomic_and_rejects_invalid_supervision(self):
        acc = RoutePermissionAccumulator()
        before = acc.compute()
        with self.assertRaises(ValueError):
            acc.update(np.array([[0., 1.], [1., 0.]]), np.array([0, 1]),
                       [metadata(16), metadata(33, distance=float('nan'))])
        self.assertEqual(acc.compute(), before)
        with self.assertRaises(ValueError):
            acc.update(np.array([[np.nan, 1.]]), np.array([0]), [metadata(16)])
        with self.assertRaises(ValueError):
            acc.update(np.array([[0., 1.]]), np.array([2]), [metadata(16)])
        with self.assertRaisesRegex(ValueError, 'aligned metadata'):
            acc.update(np.array([[0., 1.]]), np.array([0]), [dict(metadata(16), target=1, valid=True)])
        self.assertEqual(acc.compute(), before)


if __name__ == '__main__':
    unittest.main()
