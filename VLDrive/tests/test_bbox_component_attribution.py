import unittest
import numpy as np
from utils.evaluation.morai_object_metrics import _rectangle
from tools.evaluation.audit_ve_bbox_components import decompose_footprint, summary


class BboxAttributionTest(unittest.TestCase):
    def test_equivalent_axis_encoding_has_no_size_or_heading_error(self):
        truth=_rectangle([0,10],[0,1],4,2)
        pred=_rectangle([0,10],[1,0],2,4)
        r=decompose_footprint(pred,truth)
        self.assertTrue(r['axis_representation_swapped'])
        self.assertAlmostEqual(r['corner_m'],0.)
        self.assertAlmostEqual(r['size_only_corner_m'],0.)
        self.assertAlmostEqual(r['heading_only_corner_m'],0.)

    def test_translation_only_attribution(self):
        g=_rectangle([0,10],[0,1],4,2);p=_rectangle([.3,10.4],[0,1],4,2)
        r=decompose_footprint(p,g)
        self.assertAlmostEqual(r['corner_shapley_m']['center'],.5)
        self.assertAlmostEqual(r['corner_shapley_m']['size'],0.)
        self.assertAlmostEqual(r['corner_shapley_m']['heading'],0.)

    def test_size_only_attribution(self):
        g=_rectangle([0,10],[0,1],4,2);p=_rectangle([0,10],[0,1],6,4)
        r=decompose_footprint(p,g)
        self.assertAlmostEqual(r['corner_shapley_m']['size'],2**.5)
        self.assertAlmostEqual(r['corner_shapley_m']['center'],0.)

    def test_components_sum_to_actual_corner_error(self):
        g=_rectangle([1,10],[0,1],4,2)
        p=_rectangle([1.2,10.4],[np.sin(.4),np.cos(.4)],3,2.5)
        r=decompose_footprint(p,g)
        self.assertAlmostEqual(sum(r['corner_shapley_m'].values()),r['corner_m'])

    def test_summary_does_not_treat_axis_flag_as_numeric_error(self):
        g=_rectangle([0,10],[0,1],4,2)
        r=summary([dict(native='episode',metrics=decompose_footprint(g,g))])
        self.assertNotIn('axis_representation_swapped',r['metrics'])
        self.assertEqual(r['metrics']['corner_m']['mean'],0.)


if __name__=='__main__':unittest.main()
