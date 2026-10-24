import unittest
from MoraiDrive.scenario_runner.legacy_expert.ve_gt_geometry_audit import interpolate, metric
from MoraiDrive.scenario_runner.tests.test_ve_gt_comparison import box, prediction


class GeometryAuditTests(unittest.TestCase):
    def test_source_time_interpolation_wrap_and_no_extrapolation(self):
        history = ([100_000_000,200_000_000], [dict(position=[0.,0.], heading_deg=179.),
                                             dict(position=[1.,0.], heading_deg=-179.)])
        r = interpolate(history,150_000_000,['position','heading_deg'])
        self.assertEqual(r['position'], [.5,0.])
        self.assertEqual(r['heading_deg'],180.)
        self.assertIsNone(interpolate(history,50_000_000,['position']))
        self.assertIsNone(interpolate(history,250_000_000,['position']))

    def test_gap_and_teleport_not_filled(self):
        history = ([100_000_000,400_000_000], [dict(position=[0.,0.]),dict(position=[1.,0.])])
        self.assertIsNone(interpolate(history,200_000_000,['position']))
        history = ([100_000_000,200_000_000], [dict(position=[0.,0.]),dict(position=[20.,0.])])
        self.assertIsNone(interpolate(history,150_000_000,['position']))

    def test_independent_bbox_origin_correction_keeps_prediction(self):
        gt, ve = box(10.), prediction(11.3)
        before = metric(gt,ve)
        corrected = dict(gt,center_world=[gt['center_world'][0]+1.3,0.])
        after = metric(corrected,ve)
        self.assertAlmostEqual(before['center_error_m'],1.3)
        self.assertEqual(after['center_error_m'],0.)
        self.assertAlmostEqual(after['iou'],1.)
        self.assertEqual(ve['center_world'],[11.3,0.])
        self.assertEqual(gt['center_world'],[10.,0.])
