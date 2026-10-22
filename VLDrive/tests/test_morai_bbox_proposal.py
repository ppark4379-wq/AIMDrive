"""CPU checks for detached best-owned-query corner aggregation."""
import math
import unittest

import torch

from training.vision_encoder.losses.morai_bbox_corner import corner_bbox_loss
from training.vision_encoder.losses.morai_bbox_proposal import best_owned_corner_bbox_loss


def scene(batch=1, dtype=torch.float64):
    return (torch.zeros(batch, 2500, 8, dtype=dtype),
            torch.full((batch, 2500), -1, dtype=torch.long))


def box(target, ids, frame=0, cell=20, owner=0, probability=1.,
        length=2., width=.6, angle=.37):
    target[frame, cell] = target.new_tensor([
        probability, .1, -.2, math.cos(angle), -math.sin(angle),
        length/7., width/4., .3])
    ids[frame, cell] = owner


def selected_mask(ids, *positions):
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for frame, cell in positions:
        mask[frame, cell] = True
    return mask


class BestOwnedCornerBoxLossTests(unittest.TestCase):
    def assert_selected_formula_and_gradient(self, prediction, target, ids, mask, valid_mask=None):
        actual_prediction = prediction.clone().requires_grad_()
        expected_prediction = prediction.clone().requires_grad_()
        actual, terms = best_owned_corner_bbox_loss(actual_prediction, target, ids, valid_mask)
        expected, old_terms = corner_bbox_loss(expected_prediction, target, ids, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(terms['bbox_proposal_geometry'], old_terms['bbox_corner_geometry'], rtol=0, atol=0)
        torch.testing.assert_close(terms['bbox_proposal_size_validity'], old_terms['bbox_corner_size_validity'], rtol=0, atol=0)
        actual.backward(); expected.backward()
        torch.testing.assert_close(actual_prediction.grad, expected_prediction.grad, rtol=0, atol=0)
        return actual_prediction.grad, terms

    def test_selected_formula_and_gradient_match_old_explicit_mask(self):
        target, ids = scene()
        box(target, ids, cell=20); box(target, ids, cell=40)
        prediction = target.clone()
        prediction[0, 20, 1:7] += target.new_tensor([.018, -.013, .035, .011, .017, -.013])
        prediction[0, 40, 1:3] += 1.
        gradient, terms = self.assert_selected_formula_and_gradient(
            prediction, target, ids, selected_mask(ids, (0,20)))
        self.assertTrue((gradient[0,20,1:7].abs()>0).all())
        self.assertEqual(gradient[0,40].abs().sum().item(), 0.)
        self.assertEqual(gradient[..., [0,7]].abs().sum().item(), 0.)
        self.assertEqual(terms['bbox_proposal_assigned_cells'].item(), 2)
        self.assertEqual(terms['bbox_proposal_selected_cells'].item(), 1)
        self.assertEqual(terms['bbox_proposal_assigned_objects'].item(), 1)

    def test_probability_does_not_bias_minimum_cost(self):
        target, ids = scene()
        box(target, ids, cell=20, probability=1.)
        box(target, ids, cell=40, probability=.001)
        prediction = target.clone()
        prediction[0,20,1] += .02
        prediction[0,40,1] += .08  # A probability*cost selector would prefer 40.
        self.assert_selected_formula_and_gradient(prediction, target, ids, selected_mask(ids,(0,20)))

    def test_low_probability_owned_query_can_win_without_half_threshold(self):
        target, ids = scene()
        box(target, ids, cell=20, probability=1.)
        box(target, ids, cell=40, probability=.001)
        prediction = target.clone()
        prediction[0,20,1] += .08
        prediction[0,40,1] += .02
        self.assert_selected_formula_and_gradient(prediction, target, ids, selected_mask(ids,(0,40)))

    def test_cost_tie_uses_lowest_global_cell_and_no_gradient_leak(self):
        target, ids = scene()
        box(target, ids, cell=40, probability=.3)
        box(target, ids, cell=20, probability=1.)
        prediction = target.clone()
        prediction[0,[20,40],1] += .02
        gradient, _ = self.assert_selected_formula_and_gradient(
            prediction, target, ids, selected_mask(ids,(0,20)))
        self.assertGreater(gradient[0,20].abs().sum().item(),0.)
        self.assertEqual(gradient[0,40].abs().sum().item(),0.)

    def test_selector_includes_signed_size_validity_cost(self):
        target, ids = scene()
        box(target, ids, cell=20, length=.6, width=.4)
        box(target, ids, cell=40, length=.6, width=.4)
        prediction = target.clone()
        prediction[0,20,5:7] *= -1  # Equivalent signed vertices, invalid sizes.
        prediction[0,40,1] += .02
        self.assert_selected_formula_and_gradient(prediction, target, ids, selected_mask(ids,(0,40)))

    def test_single_zero_or_negative_size_winner_retains_restoring_gradient(self):
        for sign in (-1.,0.):
            with self.subTest(sign=sign):
                target,ids=scene();box(target,ids,length=.6,width=.4)
                prediction=target.clone();prediction[0,20,5:7]*=sign
                gradient,terms=self.assert_selected_formula_and_gradient(
                    prediction,target,ids,selected_mask(ids,(0,20)))
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient[0,20,5:7]<0).all())
                self.assertEqual(terms['bbox_proposal_selected_invalid_size_cells'].item(),1)

    def test_zero_heading_winner_is_finite_and_connected(self):
        target,ids=scene();box(target,ids)
        prediction=target.clone();prediction[0,20,3:5]=0
        gradient,terms=self.assert_selected_formula_and_gradient(
            prediction,target,ids,selected_mask(ids,(0,20)))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[0,20,3:5].abs().sum().item(),0.)
        self.assertEqual(terms['bbox_proposal_selected_small_heading_cells'].item(),1)

    def test_frame_local_ids_and_equal_objects_then_positive_frames(self):
        target,ids=scene(batch=3)
        box(target,ids,frame=0,cell=20,owner=0,probability=.01)
        box(target,ids,frame=0,cell=30,owner=999999999,probability=1.)
        box(target,ids,frame=1,cell=20,owner=0,probability=.2)
        box(target,ids,frame=1,cell=40,owner=0,probability=1.)
        prediction=target.clone()
        prediction[0,20,1]+=.02;prediction[0,30,1]+=.04
        prediction[1,20,1]+=.06;prediction[1,40,1]+=.08
        mask=selected_mask(ids,(0,20),(0,30),(1,20))
        _,terms=self.assert_selected_formula_and_gradient(prediction,target,ids,mask)
        costs=[]
        for position in ((0,20),(0,30),(1,20)):
            costs.append(corner_bbox_loss(prediction,target,ids,selected_mask(ids,position))[0])
        actual,_=best_owned_corner_bbox_loss(prediction,target,ids)
        torch.testing.assert_close(actual,((costs[0]+costs[1])/2+costs[2])/2)
        self.assertEqual(terms['bbox_proposal_assigned_objects'].item(),3)
        self.assertEqual(terms['bbox_proposal_selected_objects'].item(),3)
        self.assertEqual(terms['bbox_proposal_positive_frames'].item(),2)

    def test_duplicate_owned_queries_do_not_increase_instance_weight(self):
        target,ids=scene();box(target,ids,cell=20);box(target,ids,cell=40,owner=1)
        prediction=target.clone();prediction[0,20,1]+=.02
        before,_=best_owned_corner_bbox_loss(prediction,target,ids)
        for cell in range(100,110):
            target[0,cell]=target[0,20];prediction[0,cell]=prediction[0,20];ids[0,cell]=0
        after,terms=best_owned_corner_bbox_loss(prediction,target,ids)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertEqual(terms['bbox_proposal_selected_cells'].item(),2)
        self.assertEqual(terms['bbox_proposal_assigned_cells'].item(),12)

    def test_winner_flips_when_cost_order_changes(self):
        target,ids=scene();box(target,ids,cell=20);box(target,ids,cell=40)
        prediction=target.clone();prediction[0,20,1]+=.02;prediction[0,40,1]+=.04
        self.assert_selected_formula_and_gradient(prediction,target,ids,selected_mask(ids,(0,20)))
        prediction[0,20,1]=target[0,20,1]+.08
        self.assert_selected_formula_and_gradient(prediction,target,ids,selected_mask(ids,(0,40)))

    def test_finite_difference_off_selection_and_correspondence_ties(self):
        target,ids=scene();box(target,ids,cell=20);box(target,ids,cell=40)
        prediction=target.clone()
        prediction[0,20,1:7]+=target.new_tensor([.018,-.013,.035,.011,.017,-.013])
        prediction[0,40,1:3]+=1.
        prediction.requires_grad_()
        loss,_=best_owned_corner_bbox_loss(prediction,target,ids)
        analytical=torch.autograd.grad(loss,prediction)[0]
        epsilon=1e-6
        for channel in range(1,7):
            plus=prediction.detach().clone();minus=prediction.detach().clone()
            plus[0,20,channel]+=epsilon;minus[0,20,channel]-=epsilon
            numerical=(best_owned_corner_bbox_loss(plus,target,ids)[0]-best_owned_corner_bbox_loss(minus,target,ids)[0])/(2*epsilon)
            self.assertAlmostEqual(numerical.item(),analytical[0,20,channel].item(),places=6)

    def test_joint_geometry_optimizer_positive_control(self):
        target,ids=scene();box(target,ids,cell=20);box(target,ids,cell=40)
        prediction=target.clone()
        prediction[0,20,1:3]+=target.new_tensor([.04,.02])
        prediction[0,20,3:5]=target.new_tensor([math.cos(.57),-math.sin(.57)])
        prediction[0,20,5:7]+=target.new_tensor([.04,-.02])
        prediction[0,40,1:3]+=1.
        prediction.requires_grad_()
        def errors():
            p=prediction.detach()[0,20];t=target[0,20]
            center=torch.linalg.vector_norm((p[1:3]-t[1:3])*3.5).item()
            size=((p[5:7]-t[5:7])*p.new_tensor([7.,4.])).abs().sum().item()
            heading=math.acos(float(torch.clamp(torch.abs(torch.dot(p[3:5],t[3:5])/torch.linalg.vector_norm(p[3:5])),max=1.)))
            return center,size,heading
        initial=errors();initial_loss=best_owned_corner_bbox_loss(prediction,target,ids)[0].item()
        optimizer=torch.optim.Adam([prediction],lr=.01)
        for _ in range(100):
            optimizer.zero_grad()
            loss,_=best_owned_corner_bbox_loss(prediction,target,ids)
            loss.backward();optimizer.step()
        final=errors();final_loss=best_owned_corner_bbox_loss(prediction,target,ids)[0].item()
        for before,after in zip(initial,final):self.assertLess(after,before*.1)
        self.assertLess(final_loss,initial_loss*.01)

    def test_empty_and_masked_nan_are_finite_differentiable_zero(self):
        for batch,mask in ((0,None),(2,None),(2,torch.tensor([False,False]))):
            target,ids=scene(batch)
            if mask is not None:box(target,ids);target[0,20,5]=float('nan')
            prediction=torch.full_like(target,float('nan'),requires_grad=True)
            loss,terms=best_owned_corner_bbox_loss(prediction,target,ids,mask)
            self.assertEqual(loss.item(),0.)
            loss.backward();self.assertEqual(prediction.grad.abs().sum().item(),0.)
            self.assertTrue(all(value.item()==0 for value in terms.values()))

    def test_frame_and_cell_masks_ignore_nan_and_confidence_speed(self):
        target,ids=scene(batch=2);box(target,ids,frame=0);box(target,ids,frame=1)
        prediction=target.clone();target[1,20,5]=float('nan')
        prediction[0,20,[0,7]]=float('nan');target[0,20,7]=float('nan')
        for mask in (torch.tensor([True,False]),torch.tensor([True,False])[:,None].expand(2,2500)):
            loss,terms=best_owned_corner_bbox_loss(prediction,target,ids,mask)
            self.assertEqual(loss.item(),0.)
            self.assertEqual(terms['bbox_proposal_assigned_cells'].item(),1)

    def test_active_invalid_nonwinner_cannot_hide_behind_selection(self):
        for channel,value in ((0,0.),(0,1.01),(2,float('nan')),(5,0.),(6,-.1)):
            with self.subTest(channel=channel,value=value):
                target,ids=scene();box(target,ids,cell=20);box(target,ids,cell=40)
                prediction=target.clone();prediction[0,40,1]+=10.
                target[0,40,channel]=value
                with self.assertRaises(ValueError):best_owned_corner_bbox_loss(prediction,target,ids)
        target,ids=scene();box(target,ids,cell=20);box(target,ids,cell=40)
        prediction=target.clone();prediction[0,40,3]=float('nan')
        with self.assertRaises(ValueError):best_owned_corner_bbox_loss(prediction,target,ids)
        target[0,40,3:5]=0;prediction=target.clone()
        with self.assertRaises(ValueError):best_owned_corner_bbox_loss(prediction,target,ids)

    def test_invalid_structure_dtype_ids_and_mask_raise(self):
        target,ids=scene()
        calls=[lambda:best_owned_corner_bbox_loss(target[:,:2],target[:,:2],ids[:,:2]),
               lambda:best_owned_corner_bbox_loss(target,target[:,:2],ids),
               lambda:best_owned_corner_bbox_loss(target,target,ids[:,:2]),
               lambda:best_owned_corner_bbox_loss(target.long(),target,ids),
               lambda:best_owned_corner_bbox_loss(target,target.long(),ids),
               lambda:best_owned_corner_bbox_loss(target,target,ids.float()),
               lambda:best_owned_corner_bbox_loss(target,target,ids-1),
               lambda:best_owned_corner_bbox_loss(target,target,ids,torch.ones(1)),
               lambda:best_owned_corner_bbox_loss(target,target,ids,torch.ones(2,dtype=torch.bool))]
        for call in calls:
            with self.assertRaises((TypeError,ValueError)):call()

    def test_mismatched_devices_raise_without_gpu(self):
        target,ids=scene()
        for other in ('target','ids','mask'):
            with self.subTest(other=other),self.assertRaises(ValueError):
                best_owned_corner_bbox_loss(target,target.to('meta') if other=='target' else target,
                    ids.to('meta') if other=='ids' else ids,
                    torch.ones(1,dtype=torch.bool,device='meta') if other=='mask' else None)

    def test_low_precision_and_mixed_dtype_promote(self):
        for dtype in (torch.float16,torch.bfloat16):
            for active in (False,True):
                with self.subTest(dtype=dtype,active=active):
                    target,ids=scene(dtype=dtype)
                    if active:box(target,ids)
                    prediction=target.clone();prediction[0,20,5]+=.1;prediction.requires_grad_()
                    loss,_=best_owned_corner_bbox_loss(prediction,target,ids)
                    self.assertEqual(loss.dtype,torch.float32)
                    loss.backward();self.assertTrue(torch.isfinite(prediction.grad).all())
        target,ids=scene();box(target,ids)
        loss,_=best_owned_corner_bbox_loss(target.float(),target,ids)
        self.assertEqual(loss.dtype,torch.float64)

    def test_inputs_are_not_mutated(self):
        target,ids=scene();box(target,ids,cell=20);box(target,ids,cell=40)
        prediction=target.clone();prediction[0,20,1]+=.02;prediction[0,40,1]+=.04
        mask=torch.ones(1,dtype=torch.bool)
        originals=[value.clone() for value in (prediction,target,ids,mask)]
        best_owned_corner_bbox_loss(prediction,target,ids,mask)
        for value,original in zip((prediction,target,ids,mask),originals):
            torch.testing.assert_close(value,original,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
