"""Physical geometry and ownership invariants for opt-in small-object support."""
import hashlib
import math
import unittest

import numpy as np

from data.src.loaders.morai_transfer_targets import (
    OBJECT_TARGET_POLICIES,
    SMALL_OBJECT_SUPPORT_MIN_SIZE_M,
    render_object_target,
)
from utils.evaluation.morai_object_metrics import evaluate_object_frame, _inside_bev, _rectangle


def make_scene(specs, ego_yaw=0.):
    """Specs are (right, forward, length, width, heading_relative_to_ego)."""
    origin = dict(x=100., y=200., yaw=ego_yaw)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    objects = []
    for index, (right, forward, length, width, heading) in enumerate(specs):
        objects.append(dict(
            instance_key=str(index), model_name='test_object', geometry_valid=True,
            x=origin['x'] + forward * c + right * s,
            y=origin['y'] + forward * s - right * c, z=0.,
            heading_deg=math.degrees(ego_yaw + heading),
            length=length, width=width, height=1., vel_x=3., vel_y=4.,
        ))
    return dict(ego_raw=origin), objects


def render(frame, objects, policy='small_object_support_v1'):
    return render_object_target(frame, objects, target_policy=policy, return_instance_ids=True)


class MoraiTargetSupportTest(unittest.TestCase):
    def test_default_target_is_bitwise_compatible_with_before_change(self):
        frame = {'ego_raw': dict(x=100., y=200., yaw=.37)}
        objects = [
            dict(x=111.25, y=204.6, heading_deg=52., length=4.1, width=1.8,
                 height=1.7, vel_x=3., vel_y=4., geometry_valid=True),
            dict(x=112., y=205., heading_deg=-13., length=.41, width=.41,
                 height=1., vel_x=0., vel_y=0., geometry_valid=True),
            dict(x=130., y=210., heading_deg=80., length=5., width=1.4,
                 height=2., vel_x=-1., vel_y=2., geometry_valid=True),
        ]
        result = render_object_target(frame, objects)
        self.assertEqual(len(result), 3)
        target, valid, reason = result
        self.assertTrue(valid)
        self.assertIsNone(reason)
        # Recorded from the original renderer before this policy was added.
        self.assertEqual(hashlib.sha256(target.tobytes()).hexdigest(),
                         '6f7fc337e860f990bdec0b73364602369b8b0fa39a1fe855af61b8a6e1162c47')
        with_ids, _, _, ids = render(frame, objects, 'footprint_v1')
        np.testing.assert_array_equal(target, with_ids)
        self.assertEqual(ids.dtype, np.int64)
        self.assertEqual(ids.shape, (2500,))
        np.testing.assert_array_equal(ids >= 0, target[:, 0] > 0)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown object target policy'):
            render_object_target({'ego_raw': {}}, [], target_policy='typo')

    def test_empty_and_invalid_observations_have_no_owner(self):
        frame, objects = make_scene([(.5, 10.5, .41, .41, 0.)])
        for policy in OBJECT_TARGET_POLICIES:
            for invalid, valid, reason in (
                ([], True, None),
                ([dict(objects[0], geometry_valid=False)], False, 'unverified_object_model'),
                ([dict(objects[0], width=0.)], False, 'invalid_object_geometry'),
                ([dict(objects[0], length=float('nan'))], False, 'invalid_object_geometry'),
            ):
                target, observed, cause, ids = render(frame, invalid, policy)
                self.assertEqual(observed, valid)
                self.assertEqual(cause, reason)
                self.assertEqual(np.count_nonzero(target), 0)
                np.testing.assert_array_equal(ids, np.full(2500, -1, dtype=np.int64))
            # A later invalid object invalidates the whole observation.
            target, observed, _, ids = render(frame, objects + [dict(objects[0], width=-1)], policy)
            self.assertFalse(observed)
            self.assertEqual(np.count_nonzero(target), 0)
            self.assertTrue(np.all(ids == -1))

    def test_tiny_object_gains_support_without_growing_its_box(self):
        frame, objects = make_scene([(.5, 10.5, .41, .42, 0.)])
        original, _, _, _ = render(frame, objects, 'footprint_v1')
        enlarged, valid, _, ids = render(frame, objects)
        self.assertTrue(valid)
        self.assertGreater(np.count_nonzero(enlarged[:, 0]), np.count_nonzero(original[:, 0]))
        self.assertLessEqual(np.count_nonzero(enlarged[:, 0]), 9)
        self.assertEqual(SMALL_OBJECT_SUPPORT_MIN_SIZE_M, 1.5)
        active = ids == 0
        np.testing.assert_allclose(enlarged[active, 5] * 7., .41, atol=1e-7)
        np.testing.assert_allclose(enlarged[active, 6] * 4., .42, atol=1e-7)
        np.testing.assert_allclose(enlarged[active, 7] * 8., 5., atol=1e-7)
        # The original physical cells retain identical complete labels.
        np.testing.assert_array_equal(enlarged[original[:, 0] > 0], original[original[:, 0] > 0])

    def test_vehicle_sized_support_is_unchanged(self):
        frame, objects = make_scene([(2., 12., 4.5, 1.8, .72), (-8., 7., 5., 2., -.3)])
        original, _, _, original_ids = render(frame, objects, 'footprint_v1')
        candidate, _, _, candidate_ids = render(frame, objects)
        np.testing.assert_array_equal(original, candidate)
        np.testing.assert_array_equal(original_ids, candidate_ids)

    def test_rotated_support_preserves_world_transformed_centres_sizes_and_heading(self):
        for angle in np.linspace(-math.pi, math.pi, 17):
            for length, width in ((.41, .42), (4., .3)):
                with self.subTest(angle=angle, size=(length, width)):
                    right, forward = 2.3, 12.7
                    frame, objects = make_scene([(right, forward, length, width, angle)], ego_yaw=.83)
                    target, valid, _, ids = render(frame, objects)
                    self.assertTrue(valid)
                    active = np.flatnonzero(ids == 0)
                    self.assertGreater(len(active), 0)
                    rows, cols = np.divmod(active, 50)
                    np.testing.assert_allclose(cols - 24.5 + target[active, 1] * 3.5, right, atol=1e-6)
                    np.testing.assert_allclose(29.5 - rows + target[active, 2] * 3.5, forward, atol=1e-6)
                    np.testing.assert_allclose(target[active, 3], -math.sin(angle), atol=1e-6)
                    np.testing.assert_allclose(target[active, 4], -math.cos(angle), atol=1e-6)
                    np.testing.assert_allclose(target[active, 5] * 7., length, atol=1e-6)
                    np.testing.assert_allclose(target[active, 6] * 4., width, atol=1e-6)

    def test_expanded_targets_decode_to_the_original_physical_boxes(self):
        for spec in ((2., 9., .41, .41, .7), (0., 10., .6, 3.1, -1.1)):
            frame, objects = make_scene([spec], ego_yaw=-.2)
            target, _, _, _ = render(frame, objects)
            prediction = target.astype(np.float64)
            p = np.clip(prediction[:, 0], 1e-6, 1. - 1e-6)
            prediction[:, 0] = np.log(p / (1. - p))
            entry = dict(object_valid=True, ego_pose=frame['ego_raw'], objects=objects)
            result = evaluate_object_frame(prediction, entry)
            self.assertEqual(result['prediction_boxes'], 1)
            self.assertEqual(result['matching']['0.5']['tp'], 1)
            geometry = result['gt_diagnostics'][0]['matches']['0.5']
            self.assertGreater(geometry['iou'], .99999)

    def test_actual_footprint_ownership_wins_over_a_closer_expanded_object(self):
        frame, objects = make_scene([(.5, 10.5, 5., .3, 0.), (.5, 13.5, .3, .3, 0.)])
        for ordered in (objects, list(reversed(objects))):
            original, _, _, old_ids = render(frame, ordered, 'footprint_v1')
            candidate, _, _, new_ids = render(frame, ordered)
            actual = old_ids >= 0
            np.testing.assert_array_equal(new_ids[actual], old_ids[actual])
            np.testing.assert_array_equal(candidate[actual], original[actual])
            # This physical cell is closer to the tiny object, but belongs to
            # the long physical box because only its actual footprint touches.
            cell = 17 * 50 + 25  # right=.5, forward=12.5
            self.assertAlmostEqual(float(candidate[cell, 5] * 7.), 5., places=6)

    def test_equal_distance_support_tie_is_deterministic(self):
        frame, objects = make_scene([(.5, 10.5, .3, .3, 0.)])
        objects.append(dict(objects[0], instance_key='second'))
        target, _, _, ids = render(frame, objects)
        self.assertTrue(np.all(ids[target[:, 0] > 0] == 0))
        repeated, _, _, repeated_ids = render(frame, objects)
        np.testing.assert_array_equal(target, repeated)
        np.testing.assert_array_equal(ids, repeated_ids)

    def test_bev_boundaries_clip_support_without_changing_geometry_or_wrapping(self):
        for spec in ((24.8, 12., .4, .4, .3), (-24.8, 12., .4, .4, .3),
                     (.5, 29.8, .4, .4, .3), (.5, -19.8, .4, .4, .3)):
            frame, objects = make_scene([spec])
            target, valid, _, ids = render(frame, objects)
            self.assertTrue(valid)
            active = np.flatnonzero(ids == 0)
            self.assertGreater(len(active), 0)
            self.assertLessEqual(len(active), 9)
            rows, cols = np.divmod(active, 50)
            np.testing.assert_allclose(cols - 24.5 + target[active, 1] * 3.5, spec[0], atol=1e-6)
            np.testing.assert_allclose(29.5 - rows + target[active, 2] * 3.5, spec[1], atol=1e-6)
            self.assertTrue(np.all(np.abs((cols - 24.5) - spec[0]) < 2.))
            self.assertTrue(np.all(np.abs((29.5 - rows) - spec[1]) < 2.))
        frame, objects = make_scene([(100., 100., .4, .4, 0.)])
        target, valid, _, ids = render(frame, objects)
        self.assertTrue(valid)
        self.assertEqual(np.count_nonzero(target), 0)
        self.assertTrue(np.all(ids == -1))

    def test_four_near_outside_and_tangent_boxes_do_not_gain_support(self):
        # All are within the added 0.75m half-support distance of the ROI.
        # Some would rasterize onto a boundary pixel despite zero physical area.
        cases = [(25.3, 10.), (-25.3, 10.), (0., 30.3), (0., -20.3),
                 (25.2, 10.), (-25.2, 10.), (0., 30.2), (0., -20.2)]
        for right, forward in cases:
            with self.subTest(right=right, forward=forward):
                frame = dict(ego_raw=dict(x=0., y=0., yaw=0.))
                obj = dict(x=forward, y=-right, heading_deg=0., length=.4,
                           width=.4, height=1., vel_x=0., vel_y=0., geometry_valid=True)
                physical = _rectangle([right, forward], [0., 1.], .4, .4)
                self.assertFalse(_inside_bev(physical))
                target, valid, _, ids = render(frame, [obj])
                self.assertTrue(valid)
                self.assertFalse(np.any(target))
                self.assertTrue(np.all(ids == -1))

    def test_four_subpixel_physical_intersections_still_gain_support(self):
        # One micrometre of true overlap must not be discarded because the
        # physical polygon rounds to an out-of-image pixel column or row.
        epsilon = 1e-6
        cases = [(25.2-epsilon, 10.), (-25.2+epsilon, 10.),
                 (0., 30.2-epsilon), (0., -20.2+epsilon)]
        for right, forward in cases:
            with self.subTest(right=right, forward=forward):
                frame = dict(ego_raw=dict(x=0., y=0., yaw=0.))
                obj = dict(x=forward, y=-right, heading_deg=0., length=.4,
                           width=.4, height=1., vel_x=0., vel_y=0., geometry_valid=True)
                physical = _rectangle([right, forward], [0., 1.], .4, .4)
                self.assertTrue(_inside_bev(physical))
                target, valid, _, ids = render(frame, [obj])
                self.assertTrue(valid)
                self.assertGreater(np.count_nonzero(ids == 0), 0)
                active = ids == 0
                np.testing.assert_allclose(target[active, 5] * 7., .4, atol=1e-7)
                np.testing.assert_allclose(target[active, 6] * 4., .4, atol=1e-7)

    def test_rotated_outside_corner_cannot_gain_support_from_its_aabb(self):
        right, forward, angle = 25.5, 30.5, math.pi/4
        frame = dict(ego_raw=dict(x=0., y=0., yaw=0.))
        obj = dict(x=forward, y=-right, heading_deg=math.degrees(angle),
                   length=1.6, width=.1, height=1., vel_x=0., vel_y=0., geometry_valid=True)
        physical = _rectangle([right, forward], [-math.sin(angle), math.cos(angle)], 1.6, .1)
        # Axis-aligned bounds overlap the BEV, but the true rotated box does not.
        self.assertLess(physical['bounds'][0], 25.)
        self.assertLess(physical['bounds'][2], 30.)
        self.assertFalse(_inside_bev(physical))
        target, valid, _, ids = render(frame, [obj])
        self.assertTrue(valid)
        self.assertFalse(np.any(target))
        self.assertTrue(np.all(ids == -1))

    def test_legacy_self_entry_filter_still_returns_consistent_instance_ids(self):
        frame, objects = make_scene([(0., 0., 4., 2., 0.), (.5, 10.5, .4, .4, 0.)])
        target, valid, _, ids = render_object_target(
            frame, objects, grpc_bounds=False, target_policy='small_object_support_v1',
            return_instance_ids=True)
        self.assertTrue(valid)
        self.assertTrue(np.all(ids[target[:, 0] > 0] == 0))
        np.testing.assert_allclose(target[ids == 0, 5] * 7., .4, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
