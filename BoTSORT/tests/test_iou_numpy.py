import unittest
import numpy as np
from tracker.iou_numpy import bbox_overlaps


class PortableIoUTests(unittest.TestCase):
    def test_inclusive_edges_and_identity(self):
        boxes = np.array([[0, 0, 9, 9], [9, 0, 18, 9], [10, 0, 19, 9]], dtype=float)
        result = bbox_overlaps(boxes, boxes[:1])
        np.testing.assert_allclose(result[:, 0], [1, 10 / 190, 0])

    def test_fractional_and_one_pixel(self):
        result = bbox_overlaps([[0.5, 0.5, 1.5, 1.5], [1, 1, 1, 1]], [[1, 1, 1, 1]])
        np.testing.assert_allclose(result[:, 0], [0.25, 1])

    def test_empty(self):
        self.assertEqual(bbox_overlaps([], [[0, 0, 1, 1]]).shape, (0, 1))
        self.assertEqual(bbox_overlaps([[0, 0, 1, 1]], []).shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
