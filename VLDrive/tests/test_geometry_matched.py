import math
import unittest
import numpy as np
import torch

from training.vision_encoder.losses.morai_geometry_matched import (
    GeometryMLPDelta, encode_query_target, matched_geometry_loss,
)
from utils.evaluation.morai_object_metrics import _rectangle


class MatchedGeometryTests(unittest.TestCase):
    def test_target_roundtrip_for_different_queries(self):
        box = _rectangle([2.3, 17.4], [.4, .8], 3.2, .7)
        for cell in (0, 349, 2499):
            q = encode_query_target(box, cell).numpy()
            row, col = divmod(cell, 50)
            decoded = _rectangle([col - 24.5 + q[1]*3.5, 29.5 - row + q[2]*3.5],
                                 [q[3], -q[4]], q[5]*7, q[6]*4)
            np.testing.assert_allclose(decoded['polygon'], box['polygon'], atol=3e-6)

    def test_joint_geometry_can_optimize_and_frozen_channels_have_no_gradient(self):
        target = torch.zeros(1, 2500, 8)
        target[0, 100] = encode_query_target(_rectangle([1., 12.], [.3, .9], 2., .6), 100)
        ids = torch.full((1, 2500), -1, dtype=torch.long); ids[0, 100] = 3
        pred = target.clone(); pred[0,100,1:3] += .08; pred[0,100,5:7] += .03
        pred.requires_grad_(); baseline = pred.detach().clone()
        opt = torch.optim.Adam([pred], lr=.003)
        initial = float(matched_geometry_loss(pred, target, ids, baseline)[0])
        for _ in range(80):
            opt.zero_grad(); loss, _ = matched_geometry_loss(pred, target, ids, baseline)
            loss.backward(); opt.step()
        self.assertLess(float(loss), initial * .15)
        self.assertEqual(float(pred.grad[..., [0,7]].abs().sum()), 0.)

    def test_mlp_cannot_change_other_outputs_or_backpropagate_into_encoder(self):
        x = torch.randn(1, 20, 64, requires_grad=True)
        base = torch.randn(1, 20, 8, requires_grad=True)
        model = GeometryMLPDelta(); self.assertTrue(torch.equal(model(x, base), base))
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(3):
            opt.zero_grad(); model(x, base)[...,1:7].square().mean().backward(); opt.step()
        self.assertTrue(torch.equal(model(x, base)[...,[0,7]], base[...,[0,7]]))
        self.assertIsNone(x.grad); self.assertIsNone(base.grad)

    def test_empty_supervision_and_rotational_equivalence(self):
        target = torch.zeros(1,2500,8)
        target[0,100] = encode_query_target(_rectangle([0., 12.], [1.,0.], 2.,1.),100)
        ids = torch.full((1,2500),-1,dtype=torch.long)
        pred = target.clone().requires_grad_()
        loss,_ = matched_geometry_loss(pred,target,ids,pred.detach());self.assertEqual(float(loss),0.)
        ids[0,100]=0
        pred = target.clone();pred[0,100,3:5] *= -1;pred.requires_grad_()
        _,terms = matched_geometry_loss(pred,target,ids,pred.detach())
        self.assertEqual(float(terms['corner_m']),0.)
        self.assertEqual(float(terms['center_m']),0.)


if __name__ == '__main__':
    unittest.main()
