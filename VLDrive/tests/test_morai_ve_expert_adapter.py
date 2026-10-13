import math
import unittest

import numpy as np

from utils.evaluation.morai_ve_expert_adapter import (
    decode_world_predictions, right_forward_to_world,
    verify_adapter_roundtrip, world_ground_truth,
)


def entry(yaw=0.):
    return dict(object_valid=True, ego_pose=dict(x=10., y=20., yaw=yaw),
                objects=[dict(x=22., y=17., heading_deg=35., length=2., width=.4,
                              height=1., vel_x=0., vel_y=0., geometry_valid=True,
                              model_name='small_obstacle', instance_key='gt-only-id')])


class MoraiVEExpertAdapterTest(unittest.TestCase):
    def test_right_is_negative_map_y_at_zero_yaw(self):
        position, heading = right_forward_to_world([3., 12.], [0., 1.],
                                                   dict(x=10., y=20., yaw=0.))
        self.assertEqual(position, [22., 17.])
        self.assertAlmostEqual(heading, 0.)

    def test_northbound_forward_and_right_axes(self):
        position, heading = right_forward_to_world([3., 12.], [1., 0.],
                                                   dict(x=10., y=20., yaw=math.pi / 2))
        np.testing.assert_allclose(position, [13., 32.], atol=1e-10)
        self.assertAlmostEqual(heading, 0.)

    def test_sealed_gt_geometry_survives_rotated_pose(self):
        for yaw in (0., .45, math.pi / 2, -math.pi / 2):
            self.assertEqual(verify_adapter_roundtrip(entry(yaw))['status'], 'PASS')

    def test_decode_keeps_small_physical_dimensions_and_no_gt_identity(self):
        raw = np.zeros((2500, 8), np.float32)
        raw[:, 0] = -100.
        i = 17 * 50 + 27
        raw[i] = [5., .5 / 3.5, 0., 0., -1., .6 / 7., .3 / 4., 2. / 8.]
        result = decode_world_predictions(raw, dict(x=10., y=20., yaw=0.))
        self.assertEqual(len(result['boxes']), 1)
        box = result['boxes'][0]
        np.testing.assert_allclose(box['center_world'], [22.5, 17.], atol=1e-7)
        self.assertAlmostEqual(box['length_m'], .6, places=6)
        self.assertAlmostEqual(box['width_m'], .3, places=6)
        self.assertAlmostEqual(box['speed_mps'], 2.)
        self.assertEqual(box['kind'], 'unknown')
        self.assertIsNone(box['track_id'])
        self.assertIsNone(box['is_static'])
        self.assertNotIn('model_name', box)

    def test_negative_model_speed_is_unknown_without_dropping_bbox(self):
        raw = np.zeros((2500, 8), np.float32)
        raw[:, 0] = -100.
        raw[17 * 50 + 24] = [5., 0., 0., 0., -1., 2. / 7., 1. / 4., -1.]
        result = decode_world_predictions(raw, dict(x=0., y=0., yaw=0.))
        self.assertEqual(result['rejections']['invalid_speed_values'], 1)
        self.assertEqual(len(result['boxes']), 1)
        self.assertIsNone(result['boxes'][0]['speed_mps'])

    def test_gt_roi_excludes_unobserved_far_obstacle(self):
        row = entry()
        far = dict(row['objects'][0], x=110.)
        row['objects'].append(far)
        result = world_ground_truth(row)
        self.assertEqual(len(result['boxes']), 1)
        self.assertEqual(result['outside_bev'], 1)

    def test_invalid_supervision_or_pose_is_rejected(self):
        with self.assertRaises(ValueError):
            world_ground_truth(dict(entry(), object_valid=False))
        with self.assertRaises(ValueError):
            right_forward_to_world([0., 1.], [0., 1.], dict(x=float('nan'), y=0., yaw=0.))


if __name__ == '__main__':
    unittest.main()
