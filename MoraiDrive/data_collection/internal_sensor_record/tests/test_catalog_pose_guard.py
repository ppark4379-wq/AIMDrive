from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from audit_sensor_record_object_geometry import check_object_geometry as v1
from audit_sensor_record_object_geometry_v2 import check_object_geometry as v2


class CatalogPoseGuardTest(unittest.TestCase):
    def setUp(self):
        self.item=dict(kind='obstacle',model_name='Box',size_wlh_m=[2.,4.,3.])
        self.case=dict(kind='static',visible_return_contradiction=False,
                       views=[dict(dimensions_lwh_m=[4.,2.,3.]) for _ in range(4)])

    def check(self,checker,case):
        return checker(self.item,{'Box':case},'test-hash',.03,.02)

    def test_pose_failure_blocks_even_identical_dimensions(self):
        for checker in (v1,v2):
            for evidence in (None,{},dict(consistent=False)):
                case=dict(self.case,actual_pose_audit=evidence)
                result=self.check(checker,case)
                self.assertFalse(result['api_box_geometry_candidate'])
                self.assertIn('catalog_actual_pose_inconsistent',result['reasons'])

    def test_old_catalog_not_misreported_as_pose_checked(self):
        for checker in (v1,v2):
            result=self.check(checker,self.case)
            self.assertTrue(result['api_box_geometry_candidate'])
            self.assertFalse(result['catalog_actual_pose_checked'])

    def test_nonfinite_or_nonpositive_reference_never_passes(self):
        for checker in (v1,v2):
            for value in (float('nan'),float('inf'),0.,-1.,None,'4',True):
                case=deepcopy(self.case);case['views'][0]['dimensions_lwh_m'][0]=value
                self.assertFalse(self.check(checker,case)['api_box_geometry_candidate'])

    def test_good_pose_only_authorizes_api_candidate(self):
        for checker in (v1,v2):
            result=self.check(checker,dict(self.case,actual_pose_audit=dict(consistent=True)))
            self.assertTrue(result['api_box_geometry_candidate'])
            self.assertTrue(result['catalog_actual_pose_checked'])


if __name__=='__main__':unittest.main()
