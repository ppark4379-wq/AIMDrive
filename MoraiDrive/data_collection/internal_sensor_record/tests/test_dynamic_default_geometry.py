import copy
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from morai_pretraining.dynamic_default_geometry import apply_dynamic_default_profile, rotation_matrix
from morai_pretraining.sensor_record_manifest import objects_for_entry
from async_capture.native_gt_timeline import interpolate_object_value
from morai_pretraining.fused_geometry import check_fused_object_geometry, model_key

class DynamicDefaultGeometryTest(unittest.TestCase):
    def setUp(self):
        self.item={'kind':'pedestrian','position':[10.,20.,1.], 'heading_deg':90.,'size_wlh_m':[1.8,1.8,1.8], 'velocity_kmh':[9.,0.,0.]}
        self.profile={'kind':'pedestrian','reviewed_motion_speeds_kmh':[9.,10.8],'approved':True,'validation_scope':'dynamic_default_held_motion_surface_multiview_3d','default_scale_contract':'morai_native_actor_spawn_no_scale','motion_pose_review_pass':True,'evidence_sha256':'a'*64,'camera_review_pass':True,'lidar_pass':True,'center_offset_flu_m':[.1,0.,.9],'raw_offset_flu_m':[0.,0.,.8],'size_lwh_m':[1.,.6,1.8],'raw_size_lwh_m':[1.8,1.8,1.8]}
        self.entry={'approved':True,'geometry_3d':self.profile}
        self.pivot={'pedestrian_velocity_parameters_kmh':[9.],'valid':True,'offset_xyz_m':[0.,0.,.8],'offset_range_xyz_m':[0.,0.,0.],'max_actor_roll_pitch_deg':0.,'max_box_pivot_yaw_difference_deg':0.}
    def apply(self,scale=None):return apply_dynamic_default_profile(self.item,self.entry,scale,self.pivot)
    def test_body_offset_is_rotated_without_mutating_api_box(self):
        raw=copy.deepcopy(self.item);box,error=self.apply();self.assertIsNone(error)
        np.testing.assert_allclose(box['training_center_world_xyz'],[10,20.1,1.1]);np.testing.assert_allclose(box['training_size_wlh_m'],[.6,1.,1.8]);self.assertEqual(raw,self.item)
    def test_full_tilted_pose_reaches_manifest(self):
        rpy=[3.,-4.,179.];rotation=rotation_matrix(rpy);pivot=np.array([10.,20.,1.])
        self.item.update(instance_key='probe',actor_pivot_world_xyz=pivot.tolist(),actor_rotation_rpy_deg=rpy,rotation_rpy_deg=rpy,heading_deg=179.,position=(pivot+rotation@self.profile['raw_offset_flu_m']).tolist())
        self.pivot['max_actor_roll_pitch_deg']=4.
        result,error=self.apply();self.assertIsNone(error)
        expected=pivot+rotation@self.profile['center_offset_flu_m']
        np.testing.assert_allclose(result['training_center_world_xyz'],expected)
        result.update(instance_key='probe',api_box_geometry_candidate=True)
        target=objects_for_entry({'object_gt':{'objects':[self.item]}},{'objects':[result]})[0]
        np.testing.assert_allclose([target[k] for k in ('x','y','z')],expected)
        self.assertEqual(target['rotation_rpy_deg'],rpy)
        self.item['rotation_rpy_deg']=[6.,-4.,179.];self.assertIsNone(self.apply()[0])
    def test_interpolation_preserves_full_pose_and_short_rotation_arc(self):
        row=dict(self.item,id='1',type='pedestrian',name='Man1',acceleration_mps2=[0,0,0],actor_pivot_world_xyz=[10,20,1],actor_rotation_rpy_deg=[2,-4,179],rotation_rpy_deg=[2,-4,179],heading_deg=179.)
        end=copy.deepcopy(row);end.update(actor_pivot_world_xyz=[12,22,3],actor_rotation_rpy_deg=[4,-2,-179],rotation_rpy_deg=[4,-2,-179],heading_deg=-179.)
        def snap(t,item):return {'source_ns':t,'conflicting_values':[],'value':{'objects':{'p':item},'counts':{'pedestrian':1},'possibly_truncated':False}}
        result=interpolate_object_value(snap(0,row),snap(100,end),50)['objects']['p']
        np.testing.assert_allclose(result['actor_pivot_world_xyz'],[11,21,2]);np.testing.assert_allclose(result['actor_rotation_rpy_deg'],[3,-3,180]);self.assertEqual(result['rotation_rpy_deg'],result['actor_rotation_rpy_deg'])
        end.pop('actor_rotation_rpy_deg');result=interpolate_object_value(snap(0,row),snap(100,end),50)['objects']['p'];self.assertNotIn('actor_pivot_world_xyz',result)
    def test_unreviewed_animation_speed_is_masked(self):
        self.pivot['pedestrian_velocity_parameters_kmh']=[18.];self.assertIsNone(self.apply()[0])
        self.pivot['pedestrian_velocity_parameters_kmh']=[0.,9.];self.assertIsNotNone(self.apply()[0])
    def test_noisy_differenced_motion_does_not_override_native_animation_parameter(self):
        self.item['velocity_kmh']=[18.,0.,0.];self.assertIsNotNone(self.apply()[0])
        self.pivot['pedestrian_velocity_parameters_kmh']=None;self.assertIsNone(self.apply()[0])
    def test_failed_motion_review_masks_old_valid_correction(self):
        self.profile['motion_pose_review_pass']=False
        registry={'models':{model_key(self.item):self.entry}}
        result=check_fused_object_geometry(self.item,{'reasons':[],'api_box_geometry_candidate':True,'training_center_world_xy':[99,99]},registry,None,None,pivot_evidence=self.pivot)
        self.assertFalse(result['api_box_geometry_candidate']);self.assertNotIn('training_center_world_xy',result)
    def test_vehicle_and_pedestrian_evidence_cannot_be_interchanged(self):
        self.item['kind']='npc';self.assertIsNone(self.apply()[0]);self.profile['kind']='vehicle';self.assertIsNotNone(self.apply()[0]);self.item['kind']='obstacle';self.assertIsNone(self.apply()[0])
    def test_unsupported_geometry_and_unreviewed_motion_fail_closed(self):
        for key,value in [('approved',False),('motion_pose_review_pass',False),('camera_review_pass',False),('default_scale_contract',''),('raw_size_lwh_m',[2.5,1.8,1.8])]:
            old=self.profile[key];self.profile[key]=value;self.assertIsNone(self.apply()[0],key);self.profile[key]=old
        for scale in [[2,2,2],[1,1,float('nan')]]:self.assertIsNone(self.apply(scale)[0])
        self.pivot['max_actor_roll_pitch_deg']=2.;self.assertIsNone(self.apply()[0])

if __name__=='__main__':unittest.main()
