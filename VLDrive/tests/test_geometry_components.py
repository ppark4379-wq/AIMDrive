import unittest
import torch
from training.vision_encoder.losses.morai_geometry_components import component_geometry_loss
from training.vision_encoder.geometry_balanced_schedule import make_schedule


class ComponentLossTest(unittest.TestCase):
    def example(self):
        t=torch.tensor([[[1.,0.,0.,0.,-1.,4/7,2/4,0.]]])
        return t,torch.tensor([[0]])

    def test_equivalent_box_has_zero_loss(self):
        t,ids=self.example();p=t.clone();p[...,3]=1.;p[...,4]=0.;p[...,5]=2/7;p[...,6]=4/4
        loss,terms=component_geometry_loss(p,t,ids,p,anchor_weight=0.)
        self.assertAlmostEqual(float(loss),0.,places=6)

    def test_independent_terms_and_no_score_speed_gradient(self):
        t,ids=self.example();p=t.clone();p[...,1]=.2;p[...,5]=6/7;p.requires_grad_()
        loss,terms=component_geometry_loss(p,t,ids,t,anchor_weight=0.)
        self.assertGreater(float(terms['center_m']),0.)
        self.assertGreater(float(terms['size_corner_m']),0.)
        self.assertAlmostEqual(float(terms['heading_corner_m']),0.)
        loss.backward();self.assertTrue(torch.equal(p.grad[...,[0,7]],torch.zeros_like(p.grad[...,[0,7]])))

    def test_invalid_size_gradient_restores_positive_size(self):
        t,ids=self.example();p=t.clone();p[...,5]=-.1;p.requires_grad_()
        loss,_=component_geometry_loss(p,t,ids,p.detach(),anchor_weight=0.)
        loss.backward();self.assertLess(float(p.grad[...,5]),0.)

    def test_empty_selection_is_finite_and_differentiable(self):
        t,ids=self.example();ids.fill_(-1);p=t.clone().requires_grad_()
        loss,_=component_geometry_loss(p,t,ids,p.detach());loss.backward()
        self.assertEqual(float(loss),0.)

    def test_schedule_covers_types_and_is_reproducible(self):
        instances=[(0,q,'a') for q in range(7)]+[(1,0,'b')]
        replay={c:{'episode':[i+2]} for i,c in enumerate(['dynamic','highway','roundabout','tunnel_static'])}
        a=make_schedule(instances,replay,20,17)
        self.assertEqual(a,make_schedule(instances,replay,20,17))
        seen={tuple(x) for b in a for x in b[:4]}
        self.assertEqual(seen,{(f,q) for f,q,m in instances})
        self.assertTrue(all(len(b)==8 and all(q is None for f,q in b[4:]) for b in a))


if __name__=='__main__':unittest.main()
