"""CPU-only mathematical invariants for the optional native log-size loss."""
import math
import unittest

import torch

from training.vision_encoder.losses.morai_bbox_corner import corner_bbox_loss
from training.vision_encoder.losses.morai_bbox_logsize import logsize_bbox_loss


def scene(batch=1, dtype=torch.float64):
    return (torch.zeros(batch, 2500, 8, dtype=dtype),
            torch.full((batch, 2500), -1, dtype=torch.long))


def box(target, ids, frame=0, cell=20, owner=0, length=2., width=.6,
        angle=.37, probability=1.):
    target[frame, cell] = target.new_tensor([
        probability, .1, -.2, math.cos(angle), -math.sin(angle),
        length / 7., width / 4., .3])
    ids[frame, cell] = owner


class LogSizeBoxLossTests(unittest.TestCase):
    def test_perfect_sizes_zero_even_with_bad_pose(self):
        target, ids = scene()
        box(target, ids)
        prediction = target.clone()
        prediction[0,20,1:5] = target.new_tensor([100., -100., 0., 0.])
        prediction.requires_grad_()
        loss, terms = logsize_bbox_loss(prediction, target, ids)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(terms['bbox_logsize_validity'].item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad.abs().sum().item(), 0.)

    def test_reciprocal_multiplicative_errors_are_symmetric(self):
        values = []
        for ratio in (.5, 2.):
            target, ids = scene()
            box(target, ids, length=1., width=1.)
            prediction = target.clone()
            prediction[0,20,5:7] *= ratio
            values.append(logsize_bbox_loss(prediction, target, ids)[0].item())
        self.assertAlmostEqual(values[0], values[1], places=14)
        self.assertAlmostEqual(values[0], math.log(2.)-.1, places=14)

    def test_center_and_heading_cannot_change_size_loss_or_gradient(self):
        target, ids = scene()
        box(target, ids, length=4., width=1., angle=0.)
        values, gradients = [], []
        for center, angle in ((0.,0.),(200.,math.pi/6),(-100.,math.pi/2)):
            prediction = target.clone()
            prediction[0,20,1:3] = target.new_tensor([center,-center])
            prediction[0,20,3:5] = target.new_tensor([math.cos(angle),-math.sin(angle)])
            prediction[0,20,5] = 3.6/7.
            prediction.requires_grad_()
            loss, _ = logsize_bbox_loss(prediction, target, ids)
            gradients.append(torch.autograd.grad(loss,prediction)[0][0,20,5:7])
            values.append(loss.item())
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[0], values[2])
        self.assertTrue(torch.equal(gradients[0],gradients[1]))
        self.assertTrue(torch.equal(gradients[0],gradients[2]))
        self.assertLess(gradients[0][0].item(), 0.)

    def test_corner_compensation_counterexample_gets_correct_size_gradient(self):
        target, ids = scene()
        box(target, ids, length=4., width=1., angle=0.)
        prediction = target.clone()
        prediction[0,20,3:5] = target.new_tensor([math.cos(math.pi/6),-math.sin(math.pi/6)])
        prediction[0,20,5] = 3.6/7.
        prediction.requires_grad_()
        coupled, _ = corner_bbox_loss(prediction,target,ids)
        independent, _ = logsize_bbox_loss(prediction,target,ids)
        corner_grad = torch.autograd.grad(coupled,prediction,retain_graph=True)[0]
        size_grad = torch.autograd.grad(independent,prediction)[0]
        self.assertGreater(corner_grad[0,20,5].item(),0.)
        self.assertLess(size_grad[0,20,5].item(),0.)
        self.assertEqual(size_grad[0,20,6].item(),0.)

    def test_random_positive_sizes_always_receive_correct_sign(self):
        generator = torch.Generator().manual_seed(20261007)
        target, ids = scene()
        for cell in range(128):
            sizes = torch.exp(torch.rand(2,generator=generator)*4.-2.)
            box(target,ids,cell=cell,owner=cell,length=float(sizes[0]),width=float(sizes[1]),
                probability=float(torch.rand((),generator=generator)*.9+.1))
        prediction = target.clone()
        ratios = torch.exp(torch.rand(128,2,generator=generator,dtype=torch.float64)*2.-1.)
        prediction[0,:128,5:7] *= ratios
        prediction[0,:128,1:5] = torch.randn(128,4,generator=generator,dtype=torch.float64)*100.
        prediction.requires_grad_()
        loss,_ = logsize_bbox_loss(prediction,target,ids)
        loss.backward()
        error = prediction.detach()[0,:128,5:7]-target[0,:128,5:7]
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[0,:128,5:7]*error > 0).all())

    def test_zero_negative_and_subfloor_sizes_have_restoring_gradient(self):
        for size in (0.,-.3,.01):
            target,ids = scene()
            box(target,ids,length=.6,width=.4)
            prediction = target.clone()
            prediction[0,20,5:7] = target.new_tensor([size/7.,size/4.])
            prediction.requires_grad_()
            loss,terms = logsize_bbox_loss(prediction,target,ids)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertTrue((prediction.grad[0,20,5:7] < 0).all())
            self.assertGreater(terms['bbox_logsize_validity'].item(),0.)

    def test_physical_scales_seven_and_four_are_decoded(self):
        target,ids = scene()
        box(target,ids,length=7.,width=4.)
        prediction = target.clone()
        prediction[0,20,5:7] = target.new_tensor([2.,2.])
        self.assertAlmostEqual(logsize_bbox_loss(prediction,target,ids)[0].item(),math.log(2.)-.1,places=14)
        # Swapping the raw dimensions does not authorize swapping the GT axes.
        box(target,ids,length=4.,width=1.)
        prediction = target.clone()
        prediction[0,20,5:7] = target.new_tensor([1./7.,4./4.])
        prediction[0,20,3:5] = target.new_tensor([0.,-1.])
        self.assertGreater(logsize_bbox_loss(prediction,target,ids)[0].item(),0.)

    def test_absolute_size_floor_uses_decoded_metres_for_each_channel(self):
        target,ids = scene()
        box(target,ids,length=.6,width=.4)
        prediction = target.clone()
        prediction[0,20,5:7] = target.new_tensor([.01/7.,.01/4.])
        _,terms = logsize_bbox_loss(prediction,target,ids)
        # Both physical sizes are 1cm: 2*mean((.02-.01)/.4) = .05.
        # The logarithmic ratio itself cancels the 7/4 scales, so the absolute
        # validity term is the meaningful test of metric conversion.
        self.assertAlmostEqual(terms['bbox_logsize_validity'].item(),.05,places=14)
        self.assertEqual(terms['bbox_logsize_subfloor_size_cells'].item(),1)

    def test_joint_physical_scale_invariance_above_floor(self):
        values = []
        for scale in (.1,1.,10.):
            target,ids = scene()
            box(target,ids,length=2.*scale,width=.6*scale)
            prediction = target.clone()
            prediction[0,20,5:7] *= target.new_tensor([1.2,.8])
            values.append(logsize_bbox_loss(prediction,target,ids)[0].item())
        self.assertAlmostEqual(values[0],values[1],places=14)
        self.assertAlmostEqual(values[0],values[2],places=14)

    def test_size_gradient_contains_inverse_predicted_dimension(self):
        gradients = []
        for ratio in (.5,2.):
            target,ids = scene()
            box(target,ids,length=1.,width=1.)
            prediction = target.clone()
            prediction[0,20,5] *= ratio
            prediction.requires_grad_()
            loss,_ = logsize_bbox_loss(prediction,target,ids)
            gradients.append(torch.autograd.grad(loss,prediction)[0][0,20,5].item()/7.)
        self.assertAlmostEqual(abs(gradients[0])/gradients[1],4.,places=14)

    def test_duplicate_cells_preserve_equal_object_weight(self):
        target,ids = scene()
        box(target,ids,cell=20,owner=0)
        box(target,ids,cell=40,owner=10)
        prediction = target.clone()
        prediction[0,20,5:7] *= 2.
        expected = logsize_bbox_loss(prediction,target,ids)[0].item()
        for cell in range(100,110):
            target[0,cell] = target[0,20]
            prediction[0,cell] = prediction[0,20]
            ids[0,cell] = 0
        self.assertAlmostEqual(logsize_bbox_loss(prediction,target,ids)[0].item(),expected,places=14)

    def test_probability_object_and_positive_frame_balance(self):
        target,ids = scene(batch=3)
        box(target,ids,frame=0,cell=20,owner=0)
        box(target,ids,frame=0,cell=21,owner=0,probability=.25)
        box(target,ids,frame=0,cell=22,owner=9999999)
        box(target,ids,frame=1,cell=20,owner=0)
        prediction = target.clone()
        prediction[0,20,5:7] *= 2.
        expected = (math.log(2.)-.1)/1.25/2./2.
        self.assertAlmostEqual(logsize_bbox_loss(prediction,target,ids)[0].item(),expected,places=14)

    def test_empty_masked_nan_returns_differentiable_zero(self):
        for batch,mask in ((0,None),(2,None),(2,torch.tensor([False,False]))):
            target,ids = scene(batch=batch)
            if mask is not None:
                box(target,ids)
                target[0,20,5] = float('nan')
            prediction = torch.full_like(target,float('nan'),requires_grad=True)
            loss,terms = logsize_bbox_loss(prediction,target,ids,mask)
            self.assertEqual(loss.item(),0.)
            self.assertTrue(all(v.item() == 0 for v in terms.values()))
            loss.backward()
            self.assertEqual(prediction.grad.abs().sum().item(),0.)

    def test_pose_confidence_speed_and_unassigned_cells_have_no_gradient(self):
        target,ids = scene()
        box(target,ids)
        prediction = target.clone()
        prediction[0,20,5:7] *= target.new_tensor([1.2,.8])
        prediction[0,20,[0,1,2,3,4,7]] = float('nan')
        target[0,20,[1,2,3,4,7]] = float('nan')
        prediction.requires_grad_()
        loss,_ = logsize_bbox_loss(prediction,target,ids)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[:,:,[0,1,2,3,4,7]].abs().sum().item(),0.)
        self.assertEqual((prediction.grad.abs().sum(-1)>0).sum().item(),1)

    def test_frame_and_cell_masks_ignore_invalid_sizes(self):
        target,ids = scene(batch=2)
        box(target,ids,frame=0)
        box(target,ids,frame=1)
        prediction = target.clone()
        target[1,20,5] = float('nan')
        for mask in (torch.tensor([True,False]),torch.tensor([True,False])[:,None].expand(2,2500)):
            self.assertEqual(logsize_bbox_loss(prediction,target,ids,mask)[0].item(),0.)

    def test_invalid_shape_ids_and_mask_are_rejected(self):
        target,ids = scene()
        calls = [lambda:logsize_bbox_loss(target[:,:2],target[:,:2],ids[:,:2]),
            lambda:logsize_bbox_loss(target,target[:,:2],ids),
            lambda:logsize_bbox_loss(target,target,ids.float()),
            lambda:logsize_bbox_loss(target,target,ids-1),
            lambda:logsize_bbox_loss(target,target,ids,torch.ones(1)),
            lambda:logsize_bbox_loss(target,target,ids,torch.ones(2,dtype=torch.bool))]
        for call in calls:
            with self.assertRaises((TypeError,ValueError)):
                call()

    def test_invalid_active_probability_and_sizes_are_rejected(self):
        for channel,value in ((0,0.),(0,1.01),(0,float('nan')),(5,0.),(6,-.1),(5,float('nan'))):
            target,ids = scene()
            box(target,ids)
            prediction = target.clone()
            target[0,20,channel] = value
            with self.assertRaises(ValueError):
                logsize_bbox_loss(prediction,target,ids)
        target,ids = scene()
        box(target,ids)
        prediction = target.clone()
        prediction[0,20,6] = float('inf')
        with self.assertRaises(ValueError):
            logsize_bbox_loss(prediction,target,ids)

    def test_low_precision_including_empty_is_promoted(self):
        for active in (False,True):
            target,ids = scene(dtype=torch.float16)
            if active:
                box(target,ids)
            prediction = target.clone()
            prediction[0,20,5] += .1
            prediction.requires_grad_()
            loss,_ = logsize_bbox_loss(prediction,target,ids)
            self.assertEqual(loss.dtype,torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_finite_difference_agrees_for_both_native_size_channels(self):
        target,ids = scene()
        box(target,ids,length=2.,width=.7)
        prediction = target.clone()
        prediction[0,20,5:7] *= target.new_tensor([1.3,.85])
        prediction.requires_grad_()
        loss,_ = logsize_bbox_loss(prediction,target,ids)
        analytical = torch.autograd.grad(loss,prediction)[0]
        epsilon = 1e-6
        for channel in (5,6):
            plus,minus = prediction.detach().clone(),prediction.detach().clone()
            plus[0,20,channel] += epsilon
            minus[0,20,channel] -= epsilon
            numeric = (logsize_bbox_loss(plus,target,ids)[0]-logsize_bbox_loss(minus,target,ids)[0])/(2*epsilon)
            self.assertAlmostEqual(numeric.item(),analytical[0,20,channel].item(),places=7)


if __name__ == '__main__':
    unittest.main()
