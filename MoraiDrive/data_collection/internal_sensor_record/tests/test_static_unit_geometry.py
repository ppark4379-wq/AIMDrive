import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from morai_pretraining.static_unit_geometry import apply_static_unit_profile
from morai_pretraining.fused_geometry import check_fused_object_geometry, model_key
from morai_pretraining.sensor_record_manifest import objects_for_entry

class StaticUnitGeometryTest(unittest.TestCase):
    def setUp(self):
        self.item={'id':'1','instance_key':'actor1','kind':'obstacle','model_name':'Scooter_02',
                   'position':[10.,20.,3.],'heading_deg':90.,'size_wlh_m':[.8,1.7,1.2], 'velocity_kmh':[0.,0.,0.]}
        self.profile={'approved':True,'validation_scope':'static_unit_visible_surface_multiview_3d',
                      'evidence_sha256':'a'*64,'camera_review_pass':True,'lidar_pass':True,
                      'center_offset_flu_m':[.55,0.,.6],'raw_offset_flu_m':[0.,0.,0.],
                      'size_lwh_m':[1.7,.8,1.2],'raw_size_lwh_m':[1.7,.8,1.2]}
        self.entry={'approved':True,'geometry_3d':self.profile}
        self.pivot={'valid':True,'offset_xyz_m':[0.,0.,0.],'offset_range_xyz_m':[0.,0.,0.],
                    'max_actor_roll_pitch_deg':0.,'max_box_pivot_yaw_difference_deg':0.}
    def apply(self,scale=(1.,1.,1.)):
        return apply_static_unit_profile(self.item,self.entry,scale,self.pivot)
    def test_rotated_offset_and_height_reach_manifest_without_changing_raw(self):
        original=copy.deepcopy(self.item)
        result,reason=self.apply();self.assertIsNone(reason)
        np.testing.assert_allclose(result['training_center_world_xyz'],[10,20.55,3.6])
        result.update(instance_key='actor1',api_box_geometry_candidate=True)
        target=objects_for_entry({'object_gt':{'objects':[self.item]}},{'objects':[result]})[0]
        np.testing.assert_allclose([target['x'],target['y'],target['z'],target['width'],target['length'],target['height']], [10,20.55,3.6,.8,1.7,1.2])
        self.assertEqual(self.item,original)
    def test_validation_and_contract_fail_closed(self):
        for key,value in [('approved',False),('camera_review_pass',False),('lidar_pass',False),('evidence_sha256',''),('size_lwh_m',[1.,float('nan'),1.]),('raw_offset_flu_m',[.3,0,0])]:
            old=self.profile[key];self.profile[key]=value
            self.assertIsNone(self.apply()[0],key);self.profile[key]=old
        for scale in [None,[1,2,1],[1,1,float('nan')]]:self.assertIsNone(self.apply(scale)[0])
        for key,value in [('max_actor_roll_pitch_deg',1.),('offset_range_xyz_m',[0,0,.1]),('valid',False)]:
            old=self.pivot[key];self.pivot[key]=value
            self.assertIsNone(self.apply()[0],key);self.pivot[key]=old
    def test_failed_profile_does_not_fall_back_to_previously_valid_v4(self):
        registry={'models':{model_key(self.item,[1,1,1]):self.entry}}
        self.profile['approved']=False
        result=check_fused_object_geometry(self.item,{'reasons':[],'api_box_geometry_candidate':True,'training_center_world_xy':[99,99]},registry,None,[1,1,1],pivot_evidence=self.pivot)
        self.assertFalse(result['api_box_geometry_candidate']);self.assertNotIn('training_center_world_xy',result)
    def test_nonzero_raw_offset_is_removed_before_model_offset(self):
        self.item['position']=[10,20.2,3.1];self.pivot['offset_xyz_m']=[.2,0,.1];self.profile['raw_offset_flu_m']=[.2,0,.1]
        result,_=self.apply();np.testing.assert_allclose(result['training_center_world_xyz'],[10,20.55,3.6])

if __name__=='__main__':unittest.main()
