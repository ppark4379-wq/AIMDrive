import copy
import json
import math
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.vision_encoder.utils.morai_perception_validation import (
    DEFAULT_LIMITS, MatchedSpeedAccumulator, scenario_groups, selection_decision,
)
from training.vision_encoder.utils.morai_transfer_metrics import evaluate
from utils.evaluation.morai_object_metrics import ObjectMetricsAccumulator


def example(speed=0., predicted_speed=0.):
    prediction = np.zeros((2500, 8), dtype=np.float32)
    prediction[:, 0] = -20.
    # Ego origin, world (forward=10,right=0), 4m x 2m, forward heading.
    prediction[20 * 50 + 25] = [8., -.5 / 3.5, .5 / 3.5, 0., -1., 4. / 7., .5, predicted_speed / 8.]
    entry = dict(frame_id=7, sensor_time_ns=123, object_valid=True,
                 ego_pose=dict(x=0., y=0., yaw=0.), objects=[dict(
                     instance_key='0|OBJECT_TYPE_VEHICLE|1|0', model_name='car',
                     geometry_valid=True, x=10., y=0., heading_deg=0.,
                     length=4., width=2., height=1.5, vel_x=speed, vel_y=0.)])
    return prediction, entry


def metrics(prediction, entry):
    boxes, speeds = ObjectMetricsAccumulator(), MatchedSpeedAccumulator()
    row = boxes.update(prediction, entry)
    speeds.update(prediction, entry, row)
    return boxes.compute(), speeds.compute()


def report():
    prediction, entry = example(speed=2., predicted_speed=2.1)
    boxes, speed = metrics(prediction, entry)
    row = dict(frames=100, detection_observed_frames=100,
               detection_cell_average_precision=.7, detection_cell_recall=.7,
               box_center_l2_m=.2, box_length_mae_m=.3, box_width_mae_m=.2,
               detection_loss_policy='lmdrive_lav_v1',
               traffic_light_observed_frames=50, traffic_light_class_counts={'red': 25, 'green': 25},
               traffic_light_accuracy=.98, traffic_light_class_recall={'red': .96, 'green': 1.},
               object_detection=boxes, object_speed_decoded=speed, ade_m=.1, fde_m=.2)
    return dict(per_scenario={'highway': row}, object_score_scenarios=['highway'])


class MatchedSpeedTests(unittest.TestCase):
    def test_raw_scalar_world_speed_and_vehicle_motion_buckets(self):
        p, e = example(speed=3., predicted_speed=6.)
        # Ground truth is the norm of its WORLD XY vector, not ego-relative.
        e['objects'][0]['vel_y'] = 4.
        _, speed = metrics(p, e)
        group = speed['thresholds']['0.5']['vehicle_moving']
        self.assertEqual(group['gt_boxes'], 1)
        self.assertEqual(group['mae_mps'], 1.)
        self.assertEqual(group['p95_mps'], 1.)
        self.assertEqual(group['valid_speed_coverage'], 1.)
        json.dumps(speed, allow_nan=False)

    def test_finite_negative_speed_is_counted_and_included_in_raw_mae(self):
        p, e = example(speed=0., predicted_speed=-2.)
        _, speed = metrics(p, e)
        group = speed['thresholds']['0.3']['vehicle_stationary']
        self.assertEqual(group['mae_mps'], 2.)
        self.assertEqual(group['negative_prediction_speed'], 1)
        self.assertEqual(group['finite_matched_speed'], 1)
        self.assertEqual(group['valid_matched_speed'], 0)
        self.assertEqual(group['finite_speed_coverage'], 1.)
        self.assertEqual(group['valid_speed_coverage'], 0.)

    def test_missed_box_is_not_reported_as_zero_speed_error(self):
        p, e = example(speed=0., predicted_speed=20.)
        p[:, 0] = -20.
        _, speed = metrics(p, e)
        group = speed['thresholds']['0.5']['stationary']
        self.assertEqual(group['unmatched_gt_boxes'], 1)
        self.assertEqual(group['matched_recall'], 0.)
        self.assertEqual(group['valid_speed_coverage'], 0.)
        self.assertIsNone(group['mae_mps'])

    def test_nonfinite_speed_rejected_by_decoder_still_has_a_counter(self):
        p, e = example()
        p[20 * 50 + 25, 7] = np.nan
        _, speed = metrics(p, e)
        self.assertEqual(speed['prediction_cells_before_geometry_rejection_and_nms']['nonfinite_speed'], 1)
        self.assertEqual(speed['thresholds']['0.5']['stationary']['unmatched_gt_boxes'], 1)
        self.assertIsNone(speed['thresholds']['0.5']['stationary']['mae_mps'])

    def test_unsupervised_frame_and_invalid_gt_do_not_become_zero(self):
        p, e = example()
        e['objects'][0]['vel_x'] = float('nan')
        boxes, speed = metrics(p, e)
        group = speed['thresholds']['0.5']['all']
        self.assertEqual(group['invalid_gt_speed'], 1)
        self.assertIsNone(group['mae_mps'])
        acc = MatchedSpeedAccumulator()
        acc.update(None, {}, {'supervised': False})
        self.assertEqual(acc.compute()['supervised_frames'], 0)
        json.dumps(speed, allow_nan=False)

    def test_evaluation_forwards_once_and_excludes_waypoint_from_loss(self):
        prediction, entry = example(speed=2., predicted_speed=3.)
        det = torch.tensor(prediction)
        target = dict(detection=det, object_valid=torch.tensor(True), waypoint_valid=torch.tensor(True),
                      waypoints=torch.full((5, 2), 100.), traffic_light=torch.tensor(-1),
                      frame_id=torch.tensor(entry['frame_id']), sensor_time_ns=torch.tensor(entry['sensor_time_ns']))
        class Model:
            calls = 0
            def eval(self):
                pass
            def __call__(self, inputs):
                self.calls += 1
                return det.unsqueeze(0), torch.zeros(1, 5, 2), torch.zeros(1, 2)
        model = Model()
        result = evaluate(model, DataLoader([({}, target)], batch_size=1), 'cpu',
                          detection_loss_policy='lmdrive_lav_v1', object_entries=[entry],
                          waypoint_loss_weight=0., perception_metrics=True)
        self.assertEqual(model.calls, 1)
        self.assertEqual(result['task_losses']['waypoint_l1'], 0.)
        self.assertAlmostEqual(result['loss'], .5 * result['task_losses']['detection']
                               + .05 * result['task_losses']['object_speed'])
        self.assertEqual(result['object_speed_decoded']['thresholds']['0.5']['vehicle_moving']['mae_mps'], 1.)


class PerceptionSelectionTests(unittest.TestCase):
    def test_waypoint_regression_never_blocks_perception(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        candidate['per_scenario']['highway'].update(ade_m=100., fde_m=float('nan'))
        self.assertTrue(selection_decision(candidate, baseline)['eligible'])

    def test_speed_regression_blocks_despite_same_bbox(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        candidate['per_scenario']['highway']['object_speed_decoded']['thresholds']['0.5']['vehicle_moving']['mae_mps'] = .5
        decision = selection_decision(candidate, baseline)
        self.assertFalse(decision['eligible'])
        self.assertTrue(any('vehicle_moving' in row['metric'] for row in decision['violations']))

    def test_equal_bbox_and_improved_speed_ranks_above_baseline(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        group = candidate['per_scenario']['highway']['object_speed_decoded']['thresholds']['0.3']['all']
        group['mae_mps'] = .01
        before = selection_decision(baseline, baseline)
        after = selection_decision(candidate, baseline)
        self.assertTrue(after['eligible'])
        self.assertEqual(after['score'][:2], before['score'][:2])
        self.assertGreater(after['score'], before['score'])
        self.assertEqual(after['speed_tiebreak_scenarios'], ['highway'])
        self.assertAlmostEqual(after['score'][2], -.01)

    def test_zero_baseline_speed_error_is_supported_not_falsey(self):
        baseline = report()
        baseline['per_scenario']['highway']['object_speed_decoded']['thresholds']['0.3']['all']['mae_mps'] = 0.
        candidate = copy.deepcopy(baseline)
        candidate['per_scenario']['highway']['object_speed_decoded']['thresholds']['0.3']['all']['mae_mps'] = .1
        before = selection_decision(baseline, baseline)
        after = selection_decision(candidate, baseline)
        self.assertEqual(before['speed_tiebreak_scenarios'], ['highway'])
        self.assertEqual(before['score'][2], 0.)
        self.assertLess(after['score'], before['score'])

    def test_missing_baseline_supported_speed_blocks_selection(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        candidate['per_scenario']['highway']['object_speed_decoded']['thresholds']['0.3']['all']['mae_mps'] = None
        result = selection_decision(candidate, baseline)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['speed_tiebreak_scenarios'], ['highway'])
        self.assertIsNone(result['score'][2])
        json.dumps(result, allow_nan=False)

    def test_missing_matched_speed_cannot_appear_to_improve_mae(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        group = candidate['per_scenario']['highway']['object_speed_decoded']['thresholds']['0.5']['all']
        group.update(valid_speed_coverage=.5, matched_recall=.5, mae_mps=0., p95_mps=0.)
        self.assertFalse(selection_decision(candidate, baseline)['eligible'])

    def test_light_minority_recall_is_guarded_independently_of_accuracy(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        candidate['per_scenario']['highway']['traffic_light_class_recall']['red'] = .5
        self.assertFalse(selection_decision(candidate, baseline)['eligible'])

    def test_speed_or_box_policy_and_support_must_match(self):
        for field in ('policy', 'support', 'light'):
            baseline = report()
            candidate = copy.deepcopy(baseline)
            row = candidate['per_scenario']['highway']
            if field == 'policy':
                row['object_speed_decoded']['policy'] = {}
            elif field == 'support':
                row['object_speed_decoded']['thresholds']['0.5']['all']['gt_boxes'] = 2
            else:
                row['traffic_light_class_counts']['red'] = 26
            with self.assertRaises(ValueError):
                selection_decision(candidate, baseline)

    def test_invalid_limits_are_rejected(self):
        baseline = report()
        with self.assertRaises(ValueError):
            selection_decision(baseline, baseline, dict(DEFAULT_LIMITS, speed_absolute_increase_mps=float('nan')))

    def test_optional_new_scenarios_and_unknown_scenario(self):
        dataset = type('DS', (), {'row_manifest_paths': ['a', 'b', 'c']})()
        mapping = dict(a='full_loop_tunnel_static', b='full_loop_tunnel_normal', c='full_loop_traffic_light')
        groups = scenario_groups(dataset, mapping)
        self.assertEqual(groups, {'tunnel_static': [0], 'tunnel_normal': [1], 'traffic_light': [2]})
        mapping['a'] = 'new_unknown'
        with self.assertRaises(ValueError):
            scenario_groups(dataset, mapping)

    def test_no_gt_background_group_is_eligible_and_false_boxes_guarded(self):
        baseline = report()
        row = copy.deepcopy(baseline['per_scenario']['highway'])
        prediction, entry = example()
        prediction[:, 0] = -20.
        entry['objects'] = []
        row['object_detection'], row['object_speed_decoded'] = metrics(prediction, entry)
        row.update(detection_cell_average_precision=None, detection_cell_recall=None,
                   box_center_l2_m=None, box_length_mae_m=None, box_width_mae_m=None)
        baseline['per_scenario']['tunnel_normal'] = row
        candidate = copy.deepcopy(baseline)
        self.assertTrue(selection_decision(candidate, baseline)['eligible'])
        candidate['per_scenario']['tunnel_normal']['object_detection']['false_positive_boxes_per_empty_frame'] = .1
        self.assertFalse(selection_decision(candidate, baseline)['eligible'])


if __name__ == '__main__':
    unittest.main()
