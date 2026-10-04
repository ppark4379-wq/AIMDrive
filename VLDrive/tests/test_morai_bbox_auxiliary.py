"""Mathematical invariants of the opt-in relative center/size objective."""
import math
import unittest

import torch

from training.vision_encoder.losses.morai_bbox_auxiliary import relative_bbox_loss


def scene(batch=1):
    target = torch.zeros(batch, 2500, 8, dtype=torch.float64)
    owners = torch.full((batch, 2500), -1, dtype=torch.long)
    return target, owners


def box(target, owners, frame, cell, owner=0, length=2., width=1., angle=0., probability=1.):
    target[frame, cell] = torch.tensor([
        probability, .1, -.2, math.cos(angle), -math.sin(angle),
        length / 7., width / 4., .3], dtype=target.dtype)
    owners[frame, cell] = owner


class RelativeBoxLossTest(unittest.TestCase):
    def test_identical_geometry_is_zero_and_all_prediction_gradients_are_zero(self):
        target, owners = scene()
        box(target, owners, 0, 8, angle=.65)
        prediction = target.clone().requires_grad_()
        loss, terms = relative_bbox_loss(prediction, target, owners)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(set(terms), {"bbox_relative_center", "bbox_relative_size"})
        loss.backward()
        self.assertEqual(prediction.grad.abs().sum().item(), 0.)

    def test_same_metric_error_penalizes_small_objects_more(self):
        losses = []
        for length, width in ((.5, .5), (4., 2.)):
            target, owners = scene()
            box(target, owners, 0, 1, length=length, width=width)
            prediction = target.clone()
            prediction[0, 1, 1:3] += .2 / 3.5
            prediction[0, 1, 5] += .2 / 7.
            prediction[0, 1, 6] += .2 / 4.
            losses.append(relative_bbox_loss(prediction, target, owners)[0].item())
        self.assertGreater(losses[0], losses[1])

    def test_known_loss_and_dimension_floor(self):
        target, owners = scene()
        box(target, owners, 0, 4, length=.1, width=.2)
        prediction = target.clone()
        prediction[0, 4, 1] += .1 / 3.5
        prediction[0, 4, 5] += .1 / 7.
        loss, terms = relative_bbox_loss(prediction, target, owners)
        # Relative errors [0.2,0] yield SmoothL1 mean (.1+0)/2 = .05.
        self.assertAlmostEqual(terms["bbox_relative_center"].item(), .05)
        self.assertAlmostEqual(terms["bbox_relative_size"].item(), .05)
        self.assertAlmostEqual(loss.item(), .05)

    def test_rotation_invariance_with_right_negative_forward_heading_channels(self):
        losses = []
        for angle in (0., .37, math.pi / 2., -2.1):
            target, owners = scene()
            box(target, owners, 0, 4, length=3., width=.7, angle=angle)
            prediction = target.clone()
            longitudinal, lateral = .36, -.12
            delta = torch.tensor([
                longitudinal * math.cos(angle) - lateral * math.sin(angle),
                longitudinal * math.sin(angle) + lateral * math.cos(angle)], dtype=target.dtype)
            prediction[0, 4, 1:3] += delta / 3.5
            losses.append(relative_bbox_loss(prediction, target, owners)[0].item())
        for value in losses[1:]:
            self.assertAlmostEqual(value, losses[0], places=12)

    def test_duplicate_occupied_cells_do_not_reweight_an_instance(self):
        target, owners = scene()
        box(target, owners, 0, 4, owner=10)
        box(target, owners, 0, 5, owner=20)
        prediction = target.clone()
        prediction[0, 4, 1] += .4 / 3.5
        base = relative_bbox_loss(prediction, target, owners)[0]
        for cell in range(100, 110):
            target[0, cell] = target[0, 4]
            prediction[0, cell] = prediction[0, 4]
            owners[0, cell] = 10
        duplicated = relative_bbox_loss(prediction, target, owners)[0]
        self.assertAlmostEqual(base.item(), duplicated.item(), places=12)

    def test_probability_weights_within_instances_but_instances_are_equal(self):
        target, owners = scene()
        box(target, owners, 0, 4, owner=9, probability=1.)
        box(target, owners, 0, 5, owner=9, probability=.25)
        box(target, owners, 0, 6, owner=999999999, probability=.1)
        prediction = target.clone()
        prediction[0, 4, 1] += .4 / 3.5
        # For the erroneous cell: relative center .2 => cell mean .05.
        # Object 9: .05/1.25=.04; object 999999999:0; frame center mean=.02.
        loss, terms = relative_bbox_loss(prediction, target, owners)
        self.assertAlmostEqual(terms["bbox_relative_center"].item(), .02)
        self.assertAlmostEqual(loss.item(), .01)

    def test_frames_are_equal_and_empty_frames_do_not_dilute(self):
        target, owners = scene(batch=3)
        box(target, owners, 0, 4, owner=0)
        box(target, owners, 1, 4, owner=0)
        box(target, owners, 1, 5, owner=1)
        prediction = target.clone()
        prediction[0, 4, 1] += .4 / 3.5
        loss, _ = relative_bbox_loss(prediction, target, owners)
        self.assertAlmostEqual(loss.item(), .0125)

    def test_gradients_reach_only_center_and_sizes_of_assigned_cells(self):
        target, owners = scene()
        box(target, owners, 0, 4, angle=.37)
        prediction = target.clone()
        prediction[0, 4, 1:3] += torch.tensor([.2, .1], dtype=prediction.dtype)
        prediction[0, 4, 5:7] = torch.tensor([-.3, -.2], dtype=prediction.dtype)
        prediction.requires_grad_()
        loss, _ = relative_bbox_loss(prediction, target, owners)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[0, 4, [1, 2, 5, 6]].abs() > 0).all())
        self.assertEqual(prediction.grad[0, 4, [0, 3, 4, 7]].abs().sum().item(), 0.)
        self.assertEqual((prediction.grad.abs().sum(-1) > 0).sum().item(), 1)

    def test_empty_and_fully_masked_are_finite_differentiable_zero_with_ignored_nan(self):
        for batch, mask in ((0, None), (2, None), (2, torch.tensor([False, False]))):
            target, owners = scene(batch)
            prediction = torch.full_like(target, float('nan'), requires_grad=True)
            if mask is not None:
                box(target, owners, 0, 1)
                target[0, 1, 5] = float('nan')
            loss, terms = relative_bbox_loss(prediction, target, owners, mask)
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            self.assertEqual(prediction.grad.abs().sum().item(), 0.)
            self.assertTrue(all(value.item() == 0. for value in terms.values()))

    def test_cell_and_frame_validity_masks_ignore_invalid_geometry(self):
        target, owners = scene(batch=2)
        box(target, owners, 0, 4)
        box(target, owners, 1, 5)
        prediction = target.clone()
        target[1, 5, 5] = float('nan')
        frame_mask = torch.tensor([True, False])
        cell_mask = frame_mask[:, None].expand(2, 2500)
        for mask in (frame_mask, cell_mask):
            self.assertEqual(relative_bbox_loss(prediction, target, owners, mask)[0].item(), 0.)

    def test_invalid_ids_shapes_and_masks_fail_closed(self):
        target, owners = scene()
        invalid_calls = [
            lambda: relative_bbox_loss(target[:, :3], target[:, :3], owners[:, :3]),
            lambda: relative_bbox_loss(target, target[:, :3], owners),
            lambda: relative_bbox_loss(target, target, owners[:, :3]),
            lambda: relative_bbox_loss(target, target, owners.float()),
            lambda: relative_bbox_loss(target, target, owners - 1),
            lambda: relative_bbox_loss(target, target, owners, torch.ones(1)),
            lambda: relative_bbox_loss(target, target, owners, torch.ones(2, dtype=torch.bool)),
        ]
        for call in invalid_calls:
            with self.assertRaises((ValueError, TypeError)):
                call()

    def test_invalid_assigned_target_is_rejected(self):
        for channel, value in ((0, 0.), (0, 1.01), (1, float('nan')),
                               (5, 0.), (6, -.1), (3, 0.)):
            target, owners = scene()
            box(target, owners, 0, 4)
            prediction = target.clone()
            target[0, 4, channel] = value
            with self.assertRaises(ValueError):
                relative_bbox_loss(prediction, target, owners)

    def test_float16_computation_is_promoted(self):
        target, owners = scene()
        box(target, owners, 0, 4)
        target = target.half()
        prediction = target.clone()
        prediction[0, 4, 5] += .1
        prediction.requires_grad_()
        loss, _ = relative_bbox_loss(prediction, target, owners)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == '__main__':
    unittest.main()
