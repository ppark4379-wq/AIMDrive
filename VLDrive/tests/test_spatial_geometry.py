import unittest
import torch
from training.vision_encoder.losses.morai_spatial_geometry import pool_proposal_bev,SpatialBevGeometryDelta

class SpatialGeometryTests(unittest.TestCase):
    def scene(self):
        rows,cols=torch.meshgrid(torch.arange(50),torch.arange(50),indexing='ij')
        bev=torch.zeros(2500,256);bev[:,0]=cols.flatten();bev[:,1]=rows.flatten()
        base=torch.tensor([[1.,0.,0.,0.,-1.,2/7,2/4,3.]])
        return bev,base,torch.tensor([25*50+25])
    def test_metric_axis_and_rotation(self):
        bev,b,i=self.scene();roi,mask=pool_proposal_bev(bev,b,i,margin_m=0.)
        self.assertTrue(mask.all())
        torch.testing.assert_close(roi[0,12,:2],torch.tensor([25.,25.]))
        torch.testing.assert_close(roi[0,22,:2],torch.tensor([25.,24.]))
        b[:,3:5]=torch.tensor([1.,0.]);roi,_=pool_proposal_bev(bev,b,i,margin_m=0.)
        torch.testing.assert_close(roi[0,22,:2],torch.tensor([26.,25.]))
    def test_edge_mask_and_no_input_gradients(self):
        bev,b,i=self.scene();bev.requires_grad_();b.requires_grad_()
        roi,mask=pool_proposal_bev(bev,b,torch.tensor([0]))
        self.assertFalse(mask.all());self.assertTrue(mask.any());self.assertFalse(roi.requires_grad)
        self.assertTrue((roi[~mask]==0).all())
    def test_identity_frozen_channels_and_trainable_branch(self):
        f=torch.randn(3,1,288+25*256+25,requires_grad=True)
        b=torch.randn(3,1,8,requires_grad=True)
        for mode in ['spatial','repeat_center']:
            m=SpatialBevGeometryDelta(mode);o=m(f,b);self.assertTrue(torch.equal(o,b))
            o[...,1:7].sum().backward();self.assertIsNone(f.grad);self.assertIsNone(b.grad)
            self.assertGreater(float(m.delta[-1].weight.grad.abs().sum()),0)
            with torch.no_grad():m.delta[-1].bias.fill_(.2)
            self.assertTrue(torch.equal(m(f,b)[...,[0,7]],b[...,[0,7]]))
    def test_invalid_query_rejected(self):
        bev,b,i=self.scene()
        with self.assertRaises(ValueError):pool_proposal_bev(bev,b,torch.tensor([2500]))
if __name__=='__main__':unittest.main()
