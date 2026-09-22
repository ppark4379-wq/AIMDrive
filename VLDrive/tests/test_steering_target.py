"""Remaining model path selection: passed points, turns and short-path rejection."""
import copy
from pathlib import Path

import numpy as np
import pytest

from control.config import load_config, validate_config
from control.expert_controller import ExpertWaypointController
from control.steering_target import select_target


@pytest.fixture
def cfg():
    return load_config(Path(__file__).resolve().parents[1]/'configs/vldrive_control.json')


def test_passed_points_are_skipped_and_segment_is_interpolated(cfg):
    out=select_target([[0,-6],[0,-2],[0,2],[0,6],[0,10]],5.,cfg['lateral']['target'])
    assert out['aim_left_forward']==pytest.approx([0,4])
    assert out['target_waypoint_indices']==[2,3]
    assert out['target_segment_fraction']==pytest.approx(.5)
    assert out['path_progress_m']==pytest.approx(6)
    assert out['remaining_path_m']==pytest.approx(10)


@pytest.mark.parametrize('speed,distance',[(0,4),(2,4),(10,8),(30,12)])
def test_speed_dependent_distance(cfg,speed,distance):
    out=select_target([[0,-4],[0,0],[0,5],[0,10],[0,20]],speed,cfg['lateral']['target'])
    assert out['aim_left_forward']==pytest.approx([0,distance])


def test_far_first_point_is_not_replaced_with_invented_ego_segment(cfg):
    out=select_target([[0,6],[0,10],[0,15],[0,20],[0,25]],1.,cfg['lateral']['target'])
    assert out['aim_left_forward']==[0.,6.]


@pytest.mark.parametrize('side',[-1,1])
def test_curved_path_target_stays_on_original_segment_and_ahead(cfg,side):
    points=np.array([[0,-2],[0,1],[side*2,3],[side*5,4],[side*8,4]])
    out=select_target(points,5.,cfg['lateral']['target'])
    i,j=out['target_waypoint_indices'];t=out['target_segment_fraction']
    aim=np.array(out['aim_left_forward'])
    np.testing.assert_allclose(aim,points[i]*(1-t)+points[j]*t)
    assert np.linalg.norm(aim)==pytest.approx(4)
    assert aim[1]>=.5 and side*aim[0]>0


@pytest.mark.parametrize('points',[
    [[0,-10],[0,-8],[0,-6],[0,-4],[0,-2]],
    [[0,-4],[0,-2],[0,0],[0,1],[0,3]],
    [[0,0]]*5,
    [[1,-.1],[3,-.1],[5,-.1],[7,-.1],[9,-.1]],
])
def test_missing_forward_extent_never_extrapolates(cfg,points):
    with pytest.raises(ValueError,match='insufficient_remaining_path'):
        select_target(points,5.,cfg['lateral']['target'])


def test_duplicate_segments_and_non_monotonic_forward_coordinates(cfg):
    out=select_target([[0,-2],[0,-2],[0,2],[5,1],[7,-1]],4.,cfg['lateral']['target'])
    assert out['aim_left_forward'][1]>=.5
    assert np.linalg.norm(out['aim_left_forward'])>=4-1e-8


def test_valid_geometry_leaves_longitudinal_outputs_exactly_unchanged(cfg):
    original=copy.deepcopy(cfg);original['lateral']['target']['mode']='first_two_midpoint'
    before,after=ExpertWaypointController(original),ExpertWaypointController(cfg)
    for k in range(100):
        points=np.array([[np.sin(k/10)*i,4*i] for i in range(1,6)])
        speed=abs(np.sin(k/8))*6
        a=before.step(points,speed,[.5,1,1.5,2,2.5],now=k/30)
        b=after.step(points,speed,[.5,1,1.5,2,2.5],now=k/30)
        assert [a[key] for key in ('accel','brake','desired_speed_mps')]==[
            b[key] for key in ('accel','brake','desired_speed_mps')]


@pytest.mark.parametrize('key,value',[('mode','oops'),('time_s',0),('min_distance_m',20),
    ('max_distance_m',2),('min_forward_m',5),('time_s',float('nan'))])
def test_invalid_target_configuration(cfg,key,value):
    cfg['lateral']['target'][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('predicted,expected',[(8.,5/3.6),(.5,.5),(0.,0.)])
def test_active_speed_cap_preserves_slower_predictions_and_stop(cfg,predicted,expected):
    assert cfg['max_speed_kmh']==5.
    cfg['lateral']['target']['mode']='first_two_midpoint'
    controller=ExpertWaypointController(cfg)
    points=[[0.,5+i*predicted*.5] for i in range(5)]
    out=controller.step(points,2.,[.5,1,1.5,2,2.5])
    assert out['predicted_speed_mps']==pytest.approx(predicted)
    assert out['desired_speed_mps']==pytest.approx(expected)
    assert out['accel']==0 and out['brake']>0
    if predicted==0:assert out['brake']==1.


@pytest.mark.parametrize('value',[0,-5,float('nan'),True])
def test_invalid_speed_cap_rejected(cfg,value):
    cfg['max_speed_kmh']=value
    with pytest.raises(ValueError):validate_config(cfg)
