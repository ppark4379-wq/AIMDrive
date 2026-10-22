"""CPU-only physical-corner objective invariants and finite differences."""
import math
import unittest

import torch

from training.vision_encoder.losses.morai_bbox_corner import corner_bbox_loss


def scene(batch=1, dtype=torch.float64):
    return (torch.zeros(batch, 2500, 8, dtype=dtype),
            torch.full((batch, 2500), -1, dtype=torch.long))


def box(target, ids, frame=0, cell=20, owner=0, length=2., width=.6,
        angle=.37, probability=1., offset=(.1, -.2)):
    target[frame, cell] = target.new_tensor([
        probability, *offset, math.cos(angle), -math.sin(angle),
        length/7., width/4., .3])
    ids[frame, cell] = owner


class CornerBoxLossTests(unittest.TestCase):
    def test_perfect_and_equivalent_180_or_swapped_90_boxes_are_zero(self):
        target, ids = scene()
        box(target, ids, angle=.37, length=3., width=.4)
        for angle, length, width in ((.37, 3., .4), (.37+math.pi, 3., .4),
                                     (.37+math.pi/2., .4, 3.), (.37-math.pi/2., .4, 3.)):
            prediction = target.clone()
            prediction[0, 20, 3:7] = target.new_tensor([
                math.cos(angle), -math.sin(angle), length/7., width/4.])
            prediction.requires_grad_()
            loss, terms = corner_bbox_loss(prediction, target, ids)
            self.assertAlmostEqual(loss.item(), 0., places=25)
            self.assertEqual(terms['bbox_corner_size_validity'].item(), 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_same_heading_direction_is_invariant_to_norm_above_floor(self):
        target, ids = scene()
        box(target, ids)
        for multiplier in (.2, 2., 12.):
            prediction = target.clone()
            prediction[0, 20, 3:5] *= multiplier
            self.assertAlmostEqual(corner_bbox_loss(prediction, target, ids)[0].item(), 0., places=25)

    def test_center_error_in_metres_and_shared_grid_origin_cancel(self):
        values = []
        for cell, offset in ((0, (0., 0.)), (2222, (8., -4.))):
            target, ids = scene()
            box(target, ids, cell=cell, angle=0., length=2., width=1., offset=offset)
            prediction = target.clone()
            prediction[0, cell, 1] += .1/3.5
            values.append(corner_bbox_loss(prediction, target, ids)[0].item())
        # GT-axis error [.05,0] => SmoothL1 .00625, coordinate mean .003125.
        for value in values:
            self.assertAlmostEqual(value, .003125, places=12)

    def test_small_object_relative_error_and_dimension_floor(self):
        values = []
        for size in (.1, .2, 2.):
            target, ids = scene()
            box(target, ids, angle=0., length=size, width=size)
            prediction = target.clone()
            prediction[0, 20, 1] += .1/3.5
            values.append(corner_bbox_loss(prediction, target, ids)[0].item())
        self.assertAlmostEqual(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_nonsquare_orientation_error_is_positive_and_rotation_invariant(self):
        values = []
        for rotation in (0., .74, -1.2):
            target, ids = scene()
            box(target, ids, angle=rotation, length=3., width=.5)
            prediction = target.clone()
            prediction[0, 20, 3:5] = target.new_tensor([
                math.cos(rotation+.42), -math.sin(rotation+.42)])
            values.append(corner_bbox_loss(prediction, target, ids)[0].item())
        self.assertGreater(values[0], 0.)
        for value in values[1:]:
            self.assertAlmostEqual(value, values[0], places=12)

    def test_negative_signed_vertices_need_explicit_size_validity_penalty(self):
        target, ids = scene()
        box(target, ids, angle=.6, length=.6, width=.4)
        prediction = target.clone()
        prediction[0, 20, 5:7] *= -1
        prediction.requires_grad_()
        loss, terms = corner_bbox_loss(prediction, target, ids)
        self.assertAlmostEqual(terms['bbox_corner_geometry'].item(), 0., places=25)
        self.assertGreater(terms['bbox_corner_size_validity'].item(), 0.)
        self.assertEqual(terms['bbox_corner_invalid_size_cells'].item(), 1)
        loss.backward()
        self.assertLess(prediction.grad[0, 20, 5].item(), 0.)

    def test_zero_and_negative_sizes_receive_restoring_gradients(self):
        for sign in (-1., 0.):
            target, ids = scene()
            box(target, ids, length=.6, width=.4, angle=.37)
            prediction = target.clone()
            prediction[0, 20, 5:7] *= sign
            prediction.requires_grad_()
            loss, _ = corner_bbox_loss(prediction, target, ids)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertTrue((prediction.grad[0, 20, 5:7] < 0).all())

    def test_invalid_size_restoring_gradient_for_random_rotated_nonsquare_boxes(self):
        generator = torch.Generator().manual_seed(810)
        target, ids = scene()
        prediction = target.clone()
        count = 128
        for cell in range(count):
            dimensions = torch.exp(torch.rand(2, generator=generator)*6.-3.)
            angle = float(torch.rand((), generator=generator)*2.*math.pi)
            box(target, ids, cell=cell, owner=cell, length=float(dimensions[0]),
                width=float(dimensions[1]), angle=angle,
                probability=float(torch.rand((), generator=generator)*.9+.1))
            prediction[0,cell] = target[0,cell]
            rotated = angle + float(torch.rand((), generator=generator)*2.*math.pi)
            prediction[0,cell,3:5] = target.new_tensor([math.cos(rotated),-math.sin(rotated)])
            prediction[0,cell,1:3] += (torch.rand(2, generator=generator)-.5)*2.
            predicted_sizes = -torch.exp(torch.rand(2, generator=generator)*6.-3.)
            if cell % 3 == 0:
                predicted_sizes[cell % 2] = 0.
            if cell % 5 == 0:
                predicted_sizes[cell % 2] = 1.
            prediction[0,cell,5:7] = predicted_sizes/target.new_tensor([7.,4.])
        prediction.requires_grad_()
        loss, _ = corner_bbox_loss(prediction, target, ids)
        loss.backward()
        invalid = prediction.detach()[0,:count,5:7] <= 0
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[0,:count,5:7][invalid] < 0).all())

    def test_zero_heading_has_finite_loss_and_gradients(self):
        target, ids = scene()
        box(target, ids)
        prediction = target.clone()
        prediction[0, 20, 3:5] = 0
        prediction.requires_grad_()
        loss, terms = corner_bbox_loss(prediction, target, ids)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad[0, 20, 3:5].abs().sum().item(), 0.)
        self.assertEqual(terms['bbox_corner_small_heading_cells'].item(), 1)

    def test_no_auxiliary_gradient_on_confidence_speed_or_unassigned_cells(self):
        target, ids = scene()
        box(target, ids)
        prediction = target.clone()
        prediction[0, 20, 1:3] += target.new_tensor([.04, .02])
        prediction[0, 20, 3:5] = target.new_tensor([.7, -.6])
        prediction[0, 20, 5:7] += .1
        prediction.requires_grad_()
        loss, _ = corner_bbox_loss(prediction, target, ids)
        loss.backward()
        self.assertTrue((prediction.grad[0, 20, 1:7].abs() > 0).all())
        self.assertEqual(prediction.grad[..., [0, 7]].abs().sum().item(), 0.)
        self.assertEqual((prediction.grad.abs().sum(-1) > 0).sum().item(), 1)

    def test_duplicate_cells_do_not_increase_object_weight(self):
        target, ids = scene()
        box(target, ids, cell=20, owner=0)
        box(target, ids, cell=40, owner=10)
        prediction = target.clone()
        prediction[0, 20, 1] += .1
        expected = corner_bbox_loss(prediction, target, ids)[0].item()
        for cell in range(100, 110):
            target[0, cell] = target[0, 20]
            prediction[0, cell] = prediction[0, 20]
            ids[0, cell] = 0
        self.assertAlmostEqual(corner_bbox_loss(prediction, target, ids)[0].item(), expected, places=12)

    def test_probability_then_object_then_positive_frame_weighting(self):
        target, ids = scene(batch=3)
        box(target, ids, frame=0, cell=20, angle=0., length=2., width=1., owner=0)
        box(target, ids, frame=0, cell=21, angle=0., length=2., width=1., owner=0, probability=.25)
        box(target, ids, frame=0, cell=22, owner=999999999, probability=.1)
        box(target, ids, frame=1, cell=20, owner=0)
        prediction = target.clone()
        prediction[0, 20, 1] += .1/3.5
        # Erroneous cell .003125; /1.25 within object; /2 objects; /2 frames.
        expected = .003125/1.25/2/2
        self.assertAlmostEqual(corner_bbox_loss(prediction, target, ids)[0].item(), expected, places=12)

    def test_empty_and_masked_ignored_nan_are_differentiable_finite_zero(self):
        for batch, mask in ((0,None),(2,None),(2,torch.tensor([False,False]))):
            target, ids = scene(batch)
            if mask is not None:
                box(target, ids);target[0,20,5]=float('nan')
            prediction=torch.full_like(target,float('nan'),requires_grad=True)
            loss,terms=corner_bbox_loss(prediction,target,ids,mask)
            self.assertEqual(loss.item(),0.)
            loss.backward()
            self.assertEqual(prediction.grad.abs().sum().item(),0.)
            self.assertTrue(all(value.item()==0 for value in terms.values()))

    def test_frame_and_cell_masks_and_ignored_confidence_speed_nan(self):
        target, ids = scene(batch=2)
        box(target, ids,frame=0);box(target,ids,frame=1)
        prediction=target.clone();target[1,20,5]=float('nan')
        prediction[0,20,[0,7]]=float('nan');target[0,20,7]=float('nan')
        for mask in (torch.tensor([True,False]),torch.tensor([True,False])[:,None].expand(2,2500)):
            self.assertAlmostEqual(corner_bbox_loss(prediction,target,ids,mask)[0].item(),0.,places=25)

    def test_invalid_structure_ids_and_mask_are_rejected(self):
        target, ids = scene()
        calls=[lambda:corner_bbox_loss(target[:,:2],target[:,:2],ids[:,:2]),
               lambda:corner_bbox_loss(target,target[:,:2],ids),
               lambda:corner_bbox_loss(target,target,ids[:,:2]),
               lambda:corner_bbox_loss(target,target,ids.float()),
               lambda:corner_bbox_loss(target,target,ids-1),
               lambda:corner_bbox_loss(target,target,ids,torch.ones(1)),
               lambda:corner_bbox_loss(target,target,ids,torch.ones(2,dtype=torch.bool))]
        for call in calls:
            with self.assertRaises((TypeError,ValueError)):
                call()

    def test_invalid_active_geometry_is_rejected(self):
        for channel,value in ((0,0.),(0,1.01),(2,float('nan')),(5,0.),(6,-.1)):
            target,ids=scene();box(target,ids)
            prediction=target.clone();target[0,20,channel]=value
            with self.assertRaises(ValueError):corner_bbox_loss(prediction,target,ids)
        target,ids=scene();box(target,ids)
        prediction=target.clone();target[0,20,3:5]=0
        with self.assertRaises(ValueError):corner_bbox_loss(prediction,target,ids)
        box(target,ids);prediction[0,20,3]=float('nan')
        with self.assertRaises(ValueError):corner_bbox_loss(prediction,target,ids)

    def test_low_precision_is_promoted_including_empty(self):
        for active in (False,True):
            target,ids=scene(dtype=torch.float16)
            if active:box(target,ids)
            prediction=target.clone();prediction[0,20,5]+=.1;prediction.requires_grad_()
            loss,_=corner_bbox_loss(prediction,target,ids)
            self.assertEqual(loss.dtype,torch.float32)
            loss.backward();self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_finite_difference_matches_geometry_gradient_off_correspondence_ties(self):
        target,ids=scene();box(target,ids,angle=.23,length=2.,width=.7)
        prediction=target.clone()
        prediction[0,20,1:7] += target.new_tensor([.018,-.013,.035,.011,.017,-.013])
        prediction.requires_grad_()
        loss,_=corner_bbox_loss(prediction,target,ids)
        analytical=torch.autograd.grad(loss,prediction)[0]
        epsilon=1e-6
        for channel in range(1,7):
            plus=prediction.detach().clone();minus=prediction.detach().clone()
            plus[0,20,channel]+=epsilon;minus[0,20,channel]-=epsilon
            numerical=(corner_bbox_loss(plus,target,ids)[0]-corner_bbox_loss(minus,target,ids)[0])/(2*epsilon)
            self.assertAlmostEqual(numerical.item(),analytical[0,20,channel].item(),places=6)


if __name__=='__main__':
    unittest.main()
