import json
import math
import unittest

import numpy as np

from data.src.loaders.morai_transfer_targets import render_object_target
from utils.evaluation.morai_object_metrics import (
    ObjectMetricsAccumulator, decode_object_boxes, evaluate_object_frame, oriented_iou, _rectangle,
)


def empty_prediction():
    output = np.zeros((2500, 8), dtype=np.float64)
    output[:, 0] = -20.
    return output


def add_prediction(output, cell=(20, 25), center=(0., 10.), angle=0., length=4., width=2., logit=5.):
    row, col = cell
    output[row * 50 + col] = [logit,
        (center[0] - (col - 24.5)) / 3.5, (center[1] - (29.5 - row)) / 3.5,
        -math.sin(angle), -math.cos(angle), length / 7., width / 4., 0.]
    return output


def entry_with_object(center=(0., 10.), angle=0., length=4., width=2., ego_yaw=0.):
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    right, forward = center
    obj = dict(instance_key='a', model_name='test_car', geometry_valid=True,
               x=100. + forward * c + right * s, y=200. + forward * s - right * c,
               z=0., heading_deg=math.degrees(ego_yaw + angle), length=length, width=width,
               height=2., vel_x=0., vel_y=0.)
    return dict(object_valid=True, ego_pose=dict(x=100., y=200., yaw=ego_yaw), objects=[obj])


class MoraiObjectMetricsTest(unittest.TestCase):
    def test_near_coincident_real_frame_does_not_lose_half_intersection(self):
        # Woman8 val frame 51: OpenCV 5 float32 clipping returned a triangle
        # and IoU=1/3 although all four corners differ by <3e-7 metres.
        truth = _rectangle([3.424744605212365, 28.486468406850324],
                           [-0.9858397647291774, -0.16769006613010823],
                           1.0462336561586723, 0.6060894475492828)
        prediction = _rectangle([3.42474465072155, 28.48646840720903],
                                [-0.9858397648861317, -0.16769006520738372],
                                1.0462336838245392, 0.6060894727706909)
        self.assertGreater(oriented_iou(prediction, truth), .999999)
        self.assertAlmostEqual(oriented_iou(prediction, truth), oriented_iou(truth, prediction), places=12)
        self.assertAlmostEqual(oriented_iou(truth, truth), 1., places=12)

    def test_tiny_rotated_near_duplicates_are_suppressed(self):
        # Dimensions match default-scale Bollards. Offset/corner precision must
        # not create duplicate boxes or fail IoU=.5 at any heading.
        for angle in np.linspace(-math.pi, math.pi, 41):
            prediction = add_prediction(empty_prediction(), center=(3.1, 28.4), angle=angle,
                                        length=.41124, width=.41110, logit=5.)
            add_prediction(prediction, cell=(21, 25), center=(3.10000003, 28.40000002),
                           angle=angle + 1e-8, length=.41124001, width=.41110001, logit=4.)
            boxes, counts = decode_object_boxes(prediction)
            self.assertEqual(len(boxes), 1)
            self.assertEqual(counts['nms_suppressed'], 1)

    def test_float64_intersection_analytic_cases(self):
        a = _rectangle([0., 0.], [0., 1.], 4., 2.)
        rotated = _rectangle([0., 0.], [1., 0.], 4., 2.)
        self.assertAlmostEqual(oriented_iou(a, rotated), 1 / 3., places=12)
        inside = _rectangle([0., 0.], [0., 1.], 2., 1.)
        self.assertAlmostEqual(oriented_iou(a, inside), .25, places=12)
        touch = _rectangle([2., 0.], [0., 1.], 4., 2.)
        self.assertEqual(oriented_iou(a, touch), 0.)
        distant = _rectangle([10., 10.], [0., 1.], 4., 2.)
        self.assertEqual(oriented_iou(a, distant), 0.)

    def test_perfect_prediction_axes_and_world_yaw(self):
        center, angle = (3., 12.), math.radians(40.)
        entry = entry_with_object(center, angle, ego_yaw=math.radians(65.))
        prediction = add_prediction(empty_prediction(), center=center, angle=angle)
        row = evaluate_object_frame(prediction, entry)
        self.assertEqual(row['matching']['0.5']['tp'], 1)
        self.assertEqual(row['matching']['0.5']['fp'], 0)
        gt = row['gt_diagnostics'][0]
        np.testing.assert_allclose(gt['center_right_forward_m'], center, atol=1e-10)
        self.assertEqual(gt['range_bin'], '10_20m')
        self.assertGreater(gt['matches']['0.5']['iou'], .99999)
        self.assertLess(gt['matches']['0.5']['center_l2_m'], 1e-10)
        self.assertAlmostEqual(gt['matches']['0.5']['length_abs_error_m'], 0.)
        self.assertAlmostEqual(gt['matches']['0.5']['width_abs_error_m'], 0.)

    def test_renderer_targets_decode_to_perfect_object(self):
        entry = entry_with_object((2., 9.), angle=.7, ego_yaw=-.2)
        target, valid, _ = render_object_target({'ego_raw': entry['ego_pose']}, entry['objects'])
        self.assertTrue(valid)
        prediction = target.astype(np.float64)
        p = np.clip(prediction[:, 0], 1e-6, 1 - 1e-6)
        prediction[:, 0] = np.log(p / (1 - p))
        result = evaluate_object_frame(prediction, entry)
        self.assertEqual(result['prediction_boxes'], 1)
        self.assertEqual(result['matching']['0.5']['tp'], 1)
        self.assertGreater(result['prediction_decode']['nms_suppressed'], 0)

    def test_duplicate_nms_keeps_highest_confidence(self):
        prediction = add_prediction(empty_prediction(), cell=(20, 25), logit=3.)
        add_prediction(prediction, cell=(21, 25), logit=5.)
        boxes, counts = decode_object_boxes(prediction)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['cell'], [21, 25])
        self.assertEqual(counts['nms_suppressed'], 1)

    def test_one_prediction_cannot_match_two_ground_truth_objects(self):
        entry = entry_with_object()
        entry['objects'].append(dict(entry['objects'][0], instance_key='b'))
        result = evaluate_object_frame(add_prediction(empty_prediction()), entry)
        self.assertEqual(result['matching']['0.5']['tp'], 1)
        self.assertEqual(result['matching']['0.5']['fn'], 1)
        self.assertEqual(sum(x['matches']['0.5'] is not None for x in result['gt_diagnostics']), 1)

    def test_two_predictions_cannot_claim_same_ground_truth(self):
        entry = entry_with_object(width=4.)
        prediction = add_prediction(empty_prediction(), center=(-1.2, 10.), logit=5.)
        add_prediction(prediction, cell=(20, 26), center=(1.2, 10.), logit=4.)
        result = evaluate_object_frame(prediction, entry)
        self.assertEqual(result['prediction_boxes'], 2)
        self.assertEqual(result['matching']['0.3']['tp'], 1)
        self.assertEqual(result['matching']['0.3']['fp'], 1)
        self.assertEqual(result['matching']['0.5']['tp'], 0)

    def test_size_error_changes_iou_and_threshold_result(self):
        result = evaluate_object_frame(add_prediction(empty_prediction(), length=12.), entry_with_object())
        self.assertEqual(result['matching']['0.3']['tp'], 1)
        self.assertEqual(result['matching']['0.5']['tp'], 0)
        self.assertAlmostEqual(result['gt_diagnostics'][0]['matches']['0.3']['iou'], 1 / 3)
        self.assertAlmostEqual(result['gt_diagnostics'][0]['matches']['0.3']['length_abs_error_m'], 8.)

    def test_empty_and_unsupervised_frames_have_different_denominators(self):
        acc = ObjectMetricsAccumulator()
        empty = entry_with_object()
        empty['objects'] = []
        prediction = add_prediction(empty_prediction())
        acc.update(prediction, empty)
        acc.update(None, {'object_valid': False})  # Must skip before inspecting prediction.
        result = acc.compute()
        self.assertEqual(result['frames'], 2)
        self.assertEqual(result['supervised_frames'], 1)
        self.assertEqual(result['skipped_frames'], 1)
        self.assertEqual(result['empty_frames'], 1)
        self.assertEqual(result['empty_frame_false_boxes'], 1)
        self.assertEqual(result['false_positive_boxes_per_empty_frame'], 1.)
        self.assertEqual(result['thresholds']['0.5']['false_positive'], 1)
        self.assertIsNone(result['thresholds']['0.5']['recall'])
        json.dumps(result, allow_nan=False)

    def test_partial_box_with_center_outside_bev_is_included_unclipped(self):
        entry = entry_with_object((0., 31.))
        prediction = add_prediction(empty_prediction(), cell=(0, 25), center=(0., 31.))
        result = evaluate_object_frame(prediction, entry)
        self.assertEqual(result['gt_partial_boxes'], 1)
        self.assertEqual(result['prediction_partial_boxes'], 1)
        self.assertEqual(result['matching']['0.5']['tp'], 1)
        self.assertEqual(result['gt_diagnostics'][0]['range_bin'], '30_40m')
        self.assertEqual(evaluate_object_frame(empty_prediction(), entry_with_object((0., 32.)))['gt_boxes'], 0)

    def test_invalid_predictions_are_rejected_and_valid_gt_is_strict(self):
        prediction = add_prediction(empty_prediction(), length=-4.)
        add_prediction(prediction, cell=(20, 26))
        prediction[20 * 50 + 26, 3:5] = 0.
        add_prediction(prediction, cell=(20, 27))
        prediction[20 * 50 + 27, 2] = np.nan
        prediction[0, 0] = np.inf
        boxes, counts = decode_object_boxes(prediction)
        self.assertEqual(boxes, [])
        self.assertEqual(counts['invalid_geometry'], 3)
        self.assertEqual(counts['nonfinite_logits'], 1)
        entry = entry_with_object()
        entry['objects'][0]['geometry_valid'] = False
        with self.assertRaises(ValueError):
            evaluate_object_frame(empty_prediction(), entry)
        with self.assertRaises(ValueError):
            decode_object_boxes(np.zeros((50, 50, 8)))

    def test_logit_zero_is_probability_half(self):
        row = evaluate_object_frame(add_prediction(empty_prediction(), logit=0.), entry_with_object())
        self.assertEqual(row['prediction_boxes'], 1)
        row = evaluate_object_frame(add_prediction(empty_prediction(), logit=-1e-6), entry_with_object())
        self.assertEqual(row['prediction_boxes'], 0)

    def test_accumulator_per_model_range_and_geometry_denominators(self):
        acc = ObjectMetricsAccumulator()
        acc.update(add_prediction(empty_prediction(), length=6.), entry_with_object())
        acc.update(empty_prediction(), entry_with_object())
        result = acc.compute()
        self.assertEqual(result['by_gt_model']['test_car']['gt_boxes'], 2)
        self.assertEqual(result['by_gt_range']['10_20m']['gt_boxes'], 2)
        group = result['by_gt_model']['test_car']['matching']['0.5']
        self.assertEqual(group['recall'], .5)
        self.assertAlmostEqual(group['matched_geometry_mean']['length_abs_error_m'], 2.)
        self.assertEqual(result['thresholds']['0.5']['true_positive'], 1)
        self.assertEqual(result['thresholds']['0.5']['false_negative'], 1)
        self.assertIsNone(result['false_positive_boxes_per_empty_frame'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
