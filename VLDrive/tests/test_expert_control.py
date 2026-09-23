"""Unchanged expert longitudinal output and reference LMDrive lateral output."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control.config import load_config,validate_config
from control.expert_controller import ExpertWaypointController
from control.expert_source import expert_module,SOURCE
from control.waypoint_pid import current_waypoints

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    cfg=load_config(ROOT/'configs/vldrive_control.json')
    cfg['max_speed_kmh']=None  # Historical uncapped expert parity.
    cfg['lateral']['target']['mode']='first_two_midpoint'  # Original-reference regression.
    return cfg


def test_config_reads_expert_values_instead_of_duplicate_tuning(cfg):
    original=json.loads((SOURCE/'config/config.json').read_text())
    assert cfg['expert_config']=={k:original[k] for k in ('common','control')}
    assert cfg['runtime']['pid_hz']==cfg['runtime']['send_hz']==original['common']['realtime_sampling_rate']
    cfg['longitudinal']={'p_gain':.15}
    with pytest.raises(ValueError,match='expert config only'):validate_config(cfg)


def test_sparse_path_uses_first_two_midpoint_without_pursuit_denominator(cfg):
    controller=ExpertWaypointController(cfg)
    points=[[1,4],[2,8],[3,12],[4,16],[5,20]]
    out=controller.step(points,1.,[.5,1,1.5,2,2.5])
    assert out['aim_left_forward']==[1.5,6.] and out['target_waypoint_indices']==[0,1]
    assert out['aim_distance_m']==pytest.approx(math.hypot(1.5,6.)) and 'lookahead_m' not in out
    error=math.atan2(1.5,6)/(math.pi/2)
    assert out['steer']==pytest.approx((1.25+.75/40+.3)*error*.8)
    assert out['steering_requested_deg']==pytest.approx(-out['steer']*40)
    assert len(out['tracking_polyline_left_forward'])==5


def test_model_target_speed_and_throttle_have_no_added_vldrive_caps(cfg):
    controller=ExpertWaypointController(cfg)
    points=[[0,4*i] for i in range(1,6)]
    for _ in range(15):out=controller.step(points,0.,[.5,1,1.5,2,2.5])
    assert out['desired_speed_mps']==8.  # Old 1/5 m/s cap is gone.
    controller.reset()
    for _ in range(15):out=controller.step([[0,10*i] for i in range(1,6)],0.,[.5,1,1.5,2,2.5])
    assert out['accel']==1.  # Expert saturation, not the former .75 limit.


def test_lmdrive_rear_target_and_zero_speed_behavior(cfg):
    controller=ExpertWaypointController(cfg)
    out=controller.step([[0,-4*i] for i in range(1,6)],1.,[.5,1,1.5,2,2.5])
    assert out['aim_left_forward']==[0.,-6.] and out['steer']==.8
    controller.reset()
    stopped=controller.step([[1,4]]*5,0.,[.5,1,1.5,2,2.5])
    assert stopped['brake']==1. and stopped['steer']==0
    assert stopped['angle_error']==0  # Upstream's zero-speed error override.


def test_inference_longitudinal_is_identical_to_expert_over_sequence(cfg):
    module=expert_module('control.path_controller')
    reference=module.PathController(copy.deepcopy(cfg['expert_config']))
    controller=ExpertWaypointController(cfg)
    Point=expert_module('localization.point').Point
    Vehicle=expert_module('vehicle_state').VehicleState
    ctx=SimpleNamespace(mode='model_waypoints',active=False,reason='model_waypoints',debug_details={},highway_state='OFF')
    for k in range(80):
        scale=0. if k in (25,26) else (1. if k<40 else .2)
        points=np.array([[math.sin(k/9)*i,4*i] for i in range(1,6)])*scale
        speed=abs(math.sin(k/13))*5
        desired=np.linalg.norm(points[1]-points[0])/.5
        expected=reference.control_path([Point(f,l) for l,f in points],Vehicle(0,0,0,speed),desired,ctx)
        actual=controller.step(points,speed,[.5,1,1.5,2,2.5])
        assert actual['accel']==expected.accel
        assert actual['brake']==expected.brake


def test_coordinate_adapter_does_not_insert_origin_or_prune_elapsed_points():
    result=dict(coordinate_convention='lmdrive_left_negative_forward',
        waypoints_raw=[[.2*i,-4*i] for i in range(1,6)],horizons_s=[.5,1,1.5,2,2.5],
        navigation={'pose':dict(x=0.,y=0.,yaw_rad=0.)})
    points,h,indices=current_waypoints(result,dict(x=.6,y=0.,yaw_rad=0.),.9,selection='expert')
    assert indices==list(range(5)) and h.tolist()==[.5,1,1.5,2,2.5]
    np.testing.assert_allclose(points[0],[.2,3.4])


@pytest.mark.parametrize('field,value',[('wheelbase',0),('max_steering',90),('realtime_sampling_rate',0)])
def test_bad_expert_config_rejected(cfg,field,value):
    cfg['expert_config']['common'][field]=value
    with pytest.raises(ValueError):validate_config(cfg)
