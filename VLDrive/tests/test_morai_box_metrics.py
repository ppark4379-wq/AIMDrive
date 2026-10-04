import unittest
import torch
from utils.evaluation.morai_box_metrics import box_regression_sums

class BoxMetricsTest(unittest.TestCase):
    def test_scales_and_gt_mask_ignore_prediction_confidence(self):
        target=torch.zeros(1,2,8);target[0,0,0]=1
        prediction=torch.zeros_like(target);prediction[0,0,0]=-10
        prediction[0,0,1:3]=torch.tensor([3/3.5,4/3.5])
        prediction[0,0,5]=2/7;prediction[0,0,6]=.5/4
        prediction[0,1,:]=100 # invalid/negative cell must not affect geometry
        n,c,l,w=box_regression_sums(prediction,target)
        self.assertEqual(n,1);self.assertAlmostEqual(c,5,places=5)
        self.assertAlmostEqual(l,2,places=5);self.assertAlmostEqual(w,.5,places=5)
    def test_empty_is_not_a_perfect_geometry_score(self):
        self.assertEqual(box_regression_sums(torch.zeros(0,2500,8),torch.zeros(0,2500,8)),(0,0.,0.,0.))
if __name__=='__main__':unittest.main()
