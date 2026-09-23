import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tools.morai_detectors import ultralytics_detections


class DetectorAdapterTests(unittest.TestCase):
    def test_coordinates_score_and_class_stay_aligned(self):
        class Boxes:
            xyxy = torch.tensor([[12.5, 30, 300, 420], [1, 2, 3, 4]])
            conf = torch.tensor([0.8, 0.3])
            cls = torch.tensor([5, 0])
            def __len__(self):
                return 2
        rows = ultralytics_detections(SimpleNamespace(boxes=Boxes()))
        np.testing.assert_allclose(rows, [[12.5,30,300,420,0.8,5],[1,2,3,4,0.3,0]])
        self.assertEqual(rows.dtype, np.float32)

    def test_empty_results_have_six_columns(self):
        for boxes in (None, []):
            rows = ultralytics_detections(SimpleNamespace(boxes=boxes))
            self.assertEqual(rows.shape, (0, 6))


if __name__ == "__main__":
    unittest.main()
