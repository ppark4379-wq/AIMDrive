"""The truck object head and fragment merge in AIM2026 behave like the VLDrive candidate path."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from AIM2026.Inference.vision import checkpoint, postprocess

try:
    import torch
except ImportError:
    torch = None


def _prediction(cells):
    values = np.full((2500, 8), -20., dtype=np.float64)
    values[:, 1:] = 0.
    for index, (logit, off_x, off_y, length, width) in cells.items():
        values[index] = [logit, off_x, off_y, 0., -1., length / 7., width / 4., .1]
    return values


class FragmentMergeTest(unittest.TestCase):
    def test_two_overlapping_pieces_become_one_long_box(self):
        prediction = _prediction({1225: (3., 0., 0., 12., 2.5), 1226: (2., 0., 0., 6., 2.5)})
        separate, _ = postprocess.decode_boxes(prediction, nms_iou=0.9)
        merged, counts = postprocess.decode_boxes(prediction, nms_iou=0.9, merge=True)
        self.assertEqual(len(separate), 2)
        self.assertEqual(len(merged), 1)
        self.assertEqual(counts['fragment_merges'], 1)
        self.assertLessEqual(merged[0]['length'], postprocess.MERGE_MAX_LENGTH_M + 1e-6)

    def test_merge_is_off_by_default(self):
        prediction = _prediction({1225: (3., 0., 0., 12., 2.5), 1226: (2., 0., 0., 6., 2.5)})
        self.assertEqual(len(postprocess.decode_boxes(prediction, nms_iou=0.9)[0]), 2)

    def test_separate_objects_stay_separate(self):
        prediction = _prediction({1225: (3., 0., 0., 4., 2.), 1235: (3., 0., 0., 4., 2.)})
        self.assertEqual(len(postprocess.decode_boxes(prediction, merge=True)[0]), 2)


class ObjectHeadRegistryTest(unittest.TestCase):
    def test_head_is_bound_to_p1(self):
        with self.assertRaises(ValueError):
            checkpoint.resolve_object_head('truck', 'E3')
        with self.assertRaises(ValueError):
            checkpoint.resolve_object_head('nope', 'P1')

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_sha_mismatch_fails_closed(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            checkpoint.load_object_head(torch.nn.Linear(1, 1), ROOT / 'AIM2026/Weights/VE/tmp.py', 'bad', 'cpu')


@unittest.skipUnless(torch is not None and checkpoint.resolve('P1')[0].is_file() and checkpoint.resolve_object_head('truck', 'P1')[0].is_file(),
                     'weights not present')
class HeadEquivalenceTest(unittest.TestCase):
    def test_channels_0_6_from_truck_head_and_speed_from_p1(self):
        path, sha, _ = checkpoint.resolve('P1')
        model, _, _ = checkpoint.load_encoder(path, sha, 'cpu')
        reference = model.traffic_pred_head
        x = torch.randn(1, 64, reference[0].in_features if hasattr(reference, '__getitem__') else 64)
        before = reference(x)
        head_path, head_sha = checkpoint.resolve_object_head('truck', 'P1')
        checkpoint.load_object_head(model, head_path, head_sha, 'cpu')
        after = model.traffic_pred_head(x)
        self.assertTrue(torch.equal(after[..., 7], before[..., 7]))
        self.assertFalse(torch.equal(after[..., :7], before[..., :7]))


if __name__ == '__main__':
    unittest.main()
