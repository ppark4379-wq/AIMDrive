import json
import math
import unittest
from pathlib import Path

import numpy as np

from utils.evaluation.morai_object_metrics import (
    _rectangle, decode_object_boxes, oriented_iou,
)
from utils.evaluation.morai_stable_object_decode import decode_object_boxes_raw_logit


def empty_prediction():
    value = np.zeros((2500, 8), dtype=np.float64)
    value[:, 0] = -20.
    return value


def add_prediction(value, cell=1025, center=(0., 10.), angle=0., length=4., width=2., logit=5.):
    row, col = divmod(cell, 50)
    value[cell] = [logit, (center[0] - (col - 24.5)) / 3.5,
                   (center[1] - (29.5 - row)) / 3.5,
                   -math.sin(angle), -math.cos(angle), length / 7., width / 4., 0.]
    return value


class MoraiStableObjectDecodeTest(unittest.TestCase):
    def assert_box_equal(self, a, b):
        self.assertEqual(set(a), set(b))
        for key in a:
            if isinstance(a[key], np.ndarray):
                np.testing.assert_array_equal(a[key], b[key])
            else:
                self.assertEqual(a[key], b[key])

    def test_saturation_uses_logit_instead_of_lower_cell(self):
        raw = add_prediction(empty_prediction(), cell=521, logit=38.)
        add_prediction(raw, cell=571, logit=65.)
        historical, old_counts = decode_object_boxes(raw)
        fixed, new_counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual(historical[0]['score'], fixed[0]['score'])
        self.assertEqual(fixed[0]['score'], 1.)
        self.assertEqual(historical[0]['cell_index'], 521)
        self.assertEqual(fixed[0]['cell_index'], 571)
        self.assertEqual(old_counts, new_counts)

    def test_genuinely_equal_logits_keep_lower_cell(self):
        for logit in (3., 65.):
            with self.subTest(logit=logit):
                raw = add_prediction(empty_prediction(), cell=571, logit=logit)
                add_prediction(raw, cell=521, logit=logit)
                boxes, counts = decode_object_boxes_raw_logit(raw)
                self.assertEqual([b['cell_index'] for b in boxes], [521])
                self.assertEqual(counts['nms_suppressed'], 1)

    def test_normal_logit_decoder_and_schema_parity(self):
        raw = add_prediction(empty_prediction(), cell=100, center=(0., 10.), logit=.3)
        add_prediction(raw, cell=200, center=(0., 10.), logit=2.)
        add_prediction(raw, cell=150, center=(7., 3.), angle=.8, length=.5, width=.3, logit=5.)
        expected, old_counts = decode_object_boxes(raw)
        actual, new_counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual(old_counts, new_counts)
        self.assertEqual(len(expected), len(actual))
        for a, b in zip(actual, expected):
            self.assert_box_equal(a, b)

    def test_return_order_uses_logit_even_for_nonoverlapping_score_ties(self):
        raw = add_prediction(empty_prediction(), cell=521, center=(-5., 10.), logit=38.)
        add_prediction(raw, cell=571, center=(5., 10.), logit=65.)
        boxes, _ = decode_object_boxes_raw_logit(raw)
        self.assertEqual([b['cell_index'] for b in boxes], [571, 521])
        self.assertEqual([b['score'] for b in boxes], [1., 1.])

    def test_wrong_shapes_raise(self):
        for shape in ((50, 50, 8), (2500, 7), (0,), (1, 2500, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                decode_object_boxes_raw_logit(np.zeros(shape))

    def test_nonfinite_logits_and_all_eight_geometry_fields(self):
        raw = empty_prediction()
        raw[0, 0] = np.inf
        raw[1, 0] = np.nan
        raw[2, 0] = -np.inf
        add_prediction(raw, cell=100)
        raw[100, 7] = np.nan
        add_prediction(raw, cell=101)
        raw[101, 2] = np.inf
        boxes, counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual(boxes, [])
        self.assertEqual(counts['nonfinite_logits'], 3)
        self.assertEqual(counts['candidate_cells'], 2)
        self.assertEqual(counts['invalid_geometry'], 2)
        self.assertEqual(counts, decode_object_boxes(raw)[1])

    def test_invalid_sizes_and_zero_heading(self):
        raw = add_prediction(empty_prediction(), cell=100, length=-1.)
        add_prediction(raw, cell=101, width=0.)
        add_prediction(raw, cell=102)
        raw[102, 3:5] = 0.
        boxes, counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual(boxes, [])
        self.assertEqual(counts['invalid_geometry'], 3)
        self.assertEqual(counts, decode_object_boxes(raw)[1])

    def test_outside_tangent_and_partial_bev_boxes(self):
        raw = add_prediction(empty_prediction(), cell=100, center=(100., 10.))
        add_prediction(raw, cell=101, center=(0., 32.))  # Tangent at forward=30.
        add_prediction(raw, cell=102, center=(0., 31.))
        boxes, counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual(counts['outside_bev'], 2)
        self.assertEqual([b['cell_index'] for b in boxes], [102])
        self.assertTrue(boxes[0]['partial'])
        self.assertEqual(boxes[0]['length'], 4.)
        self.assert_box_equal(boxes[0], decode_object_boxes(raw)[0][0])

    def test_tiny_rotated_near_duplicates_preserve_geometry(self):
        for angle in np.linspace(-math.pi, math.pi, 41):
            with self.subTest(angle=angle):
                raw = add_prediction(empty_prediction(), cell=100, center=(3.1, 28.4), angle=angle,
                                     length=.41124, width=.41110, logit=5.)
                add_prediction(raw, cell=101, center=(3.10000003, 28.40000002), angle=angle+1e-8,
                               length=.41124001, width=.41110001, logit=4.)
                boxes, counts = decode_object_boxes_raw_logit(raw)
                self.assertEqual(len(boxes), 1)
                self.assertEqual(counts['nms_suppressed'], 1)
                self.assert_box_equal(boxes[0], decode_object_boxes(raw)[0][0])

    def test_large_finite_positive_logits_and_threshold_boundary(self):
        raw = add_prediction(empty_prediction(), cell=100, center=(-8., 10.), logit=0.)
        add_prediction(raw, cell=101, center=(0., 10.), logit=np.finfo(np.float64).max)
        add_prediction(raw, cell=102, center=(8., 10.), logit=np.nextafter(0., -1.))
        boxes, counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual([b['cell_index'] for b in boxes], [101, 100])
        self.assertEqual([b['score'] for b in boxes], [1., .5])
        self.assertEqual(counts['candidate_cells'], 2)

    def test_nms_threshold_is_strictly_greater_than_point_three(self):
        raw = add_prediction(empty_prediction(), cell=1450, center=(0., 0.), length=13., logit=5.)
        add_prediction(raw, cell=1451, center=(0., 7.), length=13., logit=4.)
        boxes, counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(oriented_iou(boxes[0], boxes[1]), .3)
        self.assertEqual(counts['nms_suppressed'], 0)
        add_prediction(raw, cell=1451, center=(0., 7.-1e-6), length=13., logit=4.)
        boxes, counts = decode_object_boxes_raw_logit(raw)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(counts['nms_suppressed'], 1)

    def test_no_input_mutation(self):
        raw = add_prediction(empty_prediction(), logit=65.)
        original = raw.copy()
        decode_object_boxes_raw_logit(raw)
        np.testing.assert_array_equal(raw, original)

    def test_tensor_like_input_uses_detach_cpu_numpy(self):
        raw = add_prediction(empty_prediction())
        class TensorLike:
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return raw
        boxes, counts = decode_object_boxes_raw_logit(TensorLike())
        expected, expected_counts = decode_object_boxes(raw)
        self.assertEqual(counts, expected_counts)
        self.assert_box_equal(boxes[0], expected[0])

    def test_actual_saved_steel_barricade_saturation_if_cache_available(self):
        run = Path('/home/dydwns/workspace/MoraiDrive/local_integration/morai_visual_pretraining/validation/'
                   'bbox_logsize_comparison_20261007/run_20261007_142250_3404778')
        predictions = run / 'logsize_aux_predictions.npz'
        cases_path = run / 'logsize_aux_cases.jsonl'
        if not predictions.is_file() or not cases_path.is_file():
            self.skipTest('Optional saved diagnostic cache is unavailable')
        with np.load(predictions, allow_pickle=False) as cache:
            frame = next(i for i, identity in enumerate(cache['identities'])
                         if str(identity).endswith('lap_072_val_manifest.json#413'))
            identity = str(cache['identities'][frame])
            raw = cache['detection'][frame].copy()
        records = (json.loads(line) for line in cases_path.read_text().splitlines())
        record = next(row for row in records if row['identity'] == identity and row['object_index'] == 0)
        gt = record['gt']
        truth = _rectangle(gt['center_right_forward_m'], gt['heading_right_forward'], gt['length_m'], gt['width_m'])
        original, _ = decode_object_boxes(raw)
        fixed, _ = decode_object_boxes_raw_logit(raw)
        self.assertNotIn(571, [b['cell_index'] for b in original])
        self.assertIn(571, [b['cell_index'] for b in fixed])
        self.assertLess(max(oriented_iou(b, truth) for b in original), .5)
        self.assertGreater(max(oriented_iou(b, truth) for b in fixed), .6)


if __name__ == '__main__':
    unittest.main()
