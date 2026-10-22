import unittest
import torch
from training.vision_encoder.losses.morai_bev_geometry import BevGeometryDelta


class BevGeometryTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(261016)
        x=torch.randn(2,7,288,requires_grad=True)
        base=torch.randn(2,7,8,requires_grad=True)
        with torch.no_grad():base[...,5:7].abs_().add_(.1)
        return x,base

    def test_identity_and_no_backbone_or_original_head_gradient(self):
        model=BevGeometryDelta();x,base=self.inputs()
        out=model(x,base)
        self.assertTrue(torch.equal(out,base))
        out[...,1:7].sum().backward()
        self.assertIsNone(x.grad);self.assertIsNone(base.grad)
        self.assertGreater(float(model.delta[-1].weight.grad.abs().sum()),0)

    def test_geometry_bounds_norm_and_non_geometry_exact_after_updates(self):
        model=BevGeometryDelta();x,base=self.inputs()
        with torch.no_grad():model.delta[-1].bias.fill_(20)
        out=model(x,base)
        self.assertTrue(torch.equal(out[...,[0,7]],base[...,[0,7]]))
        self.assertTrue(torch.all((out[...,1:3]-base[...,1:3]).abs()*3.5<=1.500001))
        self.assertTrue(torch.all(out[...,5:7]>0))
        self.assertTrue(torch.all(out[...,5:7]/base[...,5:7]<=2.000001))
        self.assertTrue(torch.allclose(out[...,3:5].norm(dim=-1),base[...,3:5].norm(dim=-1),atol=1e-6))
        with torch.no_grad():model.delta[-1].bias.fill_(-20)
        out=model(x,base)
        self.assertTrue(torch.all(out[...,5:7]/base[...,5:7]>=.499999))

    def test_shape_mismatch_rejected(self):
        x,base=self.inputs()
        with self.assertRaises(ValueError):BevGeometryDelta()(x[...,:64],base)


if __name__=='__main__':unittest.main()
