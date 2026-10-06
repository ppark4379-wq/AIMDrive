import copy
import math
import numpy as np
import unittest
from scipy.spatial.transform import Rotation
from data.src.loaders.morai_map_geometry import MapGeometry, BASIS_SCHEMA, grid_yaw, manifest_map_basis
from data.src.loaders.morai_map_derivation import rotate_waypoints, correct_entry
from data.src.loaders.morai_encoder_contract import (world_target_to_local, LiveInputAdapter,
                                                   sensor_record_encoder_contract)
from data.src.loaders.morai_transfer_adapter import MoraiVisionAdapterConfig

BASIS = dict(schema=BASIS_SCHEMA, crs='EPSG:32652', origin_xyz=[302595.,4124145.,0.],
             position_frame='projected_grid', source_orientation_frame='true_enu', orientation_frame='projected_grid')


def test_convergence_and_no_double_rotation():
    geo = MapGeometry(BASIS['crs'], BASIS['origin_xyz'])
    p = [-55., -30., 28.]
    assert -1.36 < geo.convergence_degrees(p) < -1.33
    r = Rotation.from_euler('xyz', [4, -3, 85], degrees=True).as_matrix()
    grid = geo.orientation(r, p, source_frame='true_enu')
    np.testing.assert_allclose(geo.orientation(grid, p, source_frame='projected_grid'), grid)
    with unittest.TestCase().assertRaises(ValueError):
        geo.orientation(r, p, source_frame='guess')


def test_waypoint_rotation_equals_direct_world_projection():
    for yaw in [-math.pi, -1., 0., 1.5, math.pi]:
        check_waypoint_rotation(yaw)


def check_waypoint_rotation(yaw):
    ego = [40., -70., 29.]
    future = np.array([[50.,-65.],[60.,-63.],[70.,-60.],[80.,-55.],[90.,-54.]])
    old = [world_target_to_local(p, ego[:2], yaw) for p in future]
    new_yaw = grid_yaw(yaw, ego, BASIS, source_frame='true_enu')
    direct = [world_target_to_local(p, ego[:2], new_yaw) for p in future]
    corrected = rotate_waypoints(old, math.degrees(new_yaw-yaw))
    np.testing.assert_allclose(corrected, direct, atol=5e-6, rtol=1e-6)


def test_live_target_and_training_use_same_basis():
    contract = sensor_record_encoder_contract([MoraiVisionAdapterConfig(use_previous_lidar_sweep=False)],
                                             'ego_topic', map_basis=BASIS)
    live = LiveInputAdapter(contract, lidar_frame='base_flu')
    p, target, yaw = [40.,-70.,29.], [64.,-64.], 1.2
    gy = grid_yaw(yaw, p, BASIS, source_frame='true_enu')
    direct = world_target_to_local(target, p[:2], gy)
    np.testing.assert_allclose(live.world_target(target,p,yaw,orientation_frame='true_enu'),direct)
    np.testing.assert_allclose(live.world_target(target,p,gy,orientation_frame='projected_grid'),direct)
    with unittest.TestCase().assertRaises(ValueError):
        manifest_map_basis([(None, {'map_basis':BASIS}), (None, {})])


def fixture():
    pivot = np.array([10., 20., 28.]);rpy=[2., -3., 95.];offset=[1.3, .2, .8]
    c = pivot+Rotation.from_euler('xyz',rpy,degrees=True).apply(offset)
    obj=dict(id='1',instance_key='a',model_name='car', x=c[0],y=c[1],z=c[2],
             heading_deg=rpy[2],rotation_rpy_deg=rpy,width=2.,length=4.,height=1.5,
             vel_x=3.,vel_y=-1.,geometry_valid=True)
    before=dict(frame_id=1,sensor_time_ns=123,ego_pose=dict(x=10.,y=0.,yaw=1.5,speed=4.),
                waypoints=[[0.,-float(i)] for i in range(1,6)],objects=[obj],object_valid=True,
                object_positive=True,traffic_light=-1,navigation={'target_x':20.,'target_y':30.})
    raw=dict(instance_key='a',model_name='car',position=c.tolist(),actor_pivot_world_xyz=pivot.tolist(),
             actor_rotation_rpy_deg=rpy,heading_deg=rpy[2])
    audit=dict(instance_key='a',api_box_geometry_candidate=True,training_center_world_xyz=c.tolist(),
               training_center_world_xy=c[:2].tolist(),geometry_sources={'center':'dynamic_default_surface_3d'},
               geometry_pose_source='recorded_full_actor_transform')
    return before, dict(index=1,lidar_source_ns=123,object_gt={'objects':[raw]}), dict(index=1,lidar_source_ns=123,objects=[audit]), offset


def test_actor_center_reconstructed_about_pivot_and_immutable_fields():
    before, t, g, offset=fixture();saved=copy.deepcopy(before)
    after, _=correct_entry(before,t,g,BASIS,{})
    raw=t['object_gt']['objects'][0];geo=MapGeometry(BASIS['crs'],BASIS['origin_xyz'])
    r=Rotation.from_euler('xyz',raw['actor_rotation_rpy_deg'],degrees=True).as_matrix()
    expected=np.array(raw['actor_pivot_world_xyz'])+geo.orientation(r,raw['actor_pivot_world_xyz'],source_frame='true_enu')@offset
    np.testing.assert_allclose([after['objects'][0][k] for k in 'xyz'],expected)
    assert before == saved
    for k in ('width','length','height','vel_x','vel_y'):
        assert before['objects'][0][k] == after['objects'][0][k]
    assert before['sensor_time_ns']==after['sensor_time_ns']
    assert before['navigation']==after['navigation']


def test_world_center_is_not_rotated_and_unsafe_exclusion_masks():
    before,t,g,_=fixture();g['objects'][0]['geometry_sources']['center']='grpc'
    t['object_gt']['objects'].append(dict(instance_key='b',position=[10.,20.,28.]))
    g['objects'].append(dict(instance_key='b',bev_exclusion_proven=True,enclosing_radius_m=2.,boundary_margin_m=1.))
    after, review=correct_entry(before,t,g,BASIS,{})
    for k in 'xyz':assert after['objects'][0][k]==before['objects'][0][k]
    assert not after['object_valid'] and not after['object_positive']
    assert review['object_mask_reasons']


def test_light_heading_boundary_is_masked_and_time_mismatch_rejected():
    before,t,g,_=fixture();before['traffic_light']=0
    signal=dict(index=1,sensor_source_ns=123,training_label=0,heading_error_deg=19.5)
    after, review=correct_entry(before,t,g,BASIS,signal)
    assert after['traffic_light']==-1 and review['light_masked']
    t['lidar_source_ns']=124
    with unittest.TestCase().assertRaises(ValueError):correct_entry(before,t,g,BASIS,signal)



def test_static_nonzero_api_pivot_offset_uses_measured_profile():
    before,t,g,offset=fixture()
    rpy=[0.,0.,95.];pivot=np.array([10.,20.,28.]);r=Rotation.from_euler('xyz',rpy,degrees=True)
    observed=np.array([.02,.1,.45]);raw_center=pivot+r.apply(observed)
    fitted=np.array([.2,-.05,.8]);old_center=raw_center+r.apply(fitted-observed)
    raw=t['object_gt']['objects'][0]
    raw.update(position=raw_center.tolist(),actor_rotation_rpy_deg=rpy,heading_deg=95.)
    obj=before['objects'][0];obj.update(zip('xyz',old_center));obj.update(rotation_rpy_deg=rpy,heading_deg=95.)
    g['objects'][0].update(geometry_sources={'center':'static_unit_surface_3d'},
                           training_center_world_xyz=old_center.tolist(),training_center_world_xy=old_center[:2].tolist())
    pivots={'a':dict(valid=True,offset_xyz_m=observed.tolist(),offset_range_xyz_m=[0.,0.,0.])}
    after,_=correct_entry(before,t,g,BASIS,{},pivots)
    geo=MapGeometry(BASIS['crs'],BASIS['origin_xyz'])
    expected=pivot+geo.orientation(r.as_matrix(),pivot,source_frame='true_enu')@fitted
    np.testing.assert_allclose([after['objects'][0][k] for k in 'xyz'],expected,atol=1e-10)
    with unittest.TestCase().assertRaises(ValueError):
        correct_entry(before,t,g,BASIS,{})


def test_invalid_map_units_and_unknown_basis_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        MapGeometry('EPSG:4326', [0.,0.,0.])
    altered=dict(BASIS, source_orientation_frame='unknown')
    with unittest.TestCase().assertRaises(ValueError):
        grid_yaw(0.,[0.,0.,0.],altered,source_frame='unknown')


def test_training_rejects_legacy_targets_before_loading_sensors():
    import argparse
    from unittest.mock import patch
    from training.vision_encoder.recipes import transfer
    args=transfer.add_arguments(argparse.ArgumentParser()).parse_args(['--device','cpu'])
    args.manifest_index=['unused']
    with patch.object(transfer,'resolve_args',return_value=None), \
         patch.object(transfer,'load_publications',return_value=({}, {'train':[(None,{})], 'val':[(None,{})]})), \
         patch.object(transfer,'SensorRecordDataset') as dataset:
        with unittest.TestCase().assertRaisesRegex(ValueError,'Legacy targets'):
            transfer.run(args)
        dataset.assert_not_called()

def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items() if name.startswith('test_') and callable(fn))
