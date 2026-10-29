"""Regressions for abort/stop/rear-conflict events in the 03:16 highway run."""
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from AIM2026.Expert.src.control.path_controller import PathController
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.planning.highway.common.types import HighwayPlanningResult
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner


ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def lanes():
    return HighwayLaneMap(str(ROOT/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))


def recorded(lanes):
    planner=HighwayLanePlanner(lane_map=lanes)
    ego=VehicleState(66.96687389723957,133.5231966045685,-1.6172105711360256,.2776923113399082)
    match=lanes.project(ego.position,lanes.links['A2256W000409'])
    rear=ObjectInfo(66.83726193653412,149.22079420777996,9.445049285888672,2,unique_id=49,
                   size_x=6.049566030502319,size_y=2.5232014656066895,heading_deg=-81.2146055987036)
    rear.motion_uncertain=True
    lead=ObjectInfo(67.23079995309935,104.66423708671711,8.527131043530973,2,unique_id=47,
                   size_x=5.031945109367371,size_y=2.3255186080932617,heading_deg=-90.7386171371461)
    entries=[{'object_info':rear},{'object_info':lead}]
    path=[r[0] for r in lanes.route_points(match,180.,.5)]
    return planner,ego,match,entries,path


def test_first_merge_recorded_rear_risk_checks_stop_instead_of_blind_fallback(lanes):
    planner,ego,match,entries,path=recorded(lanes)
    safety=planner.merge_safety
    assert safety.zone_for(match)=='FIRST_PREDICTIVE'
    target=safety.limit(ego,match,planner._match_objects(entries),entries,path,desired_velocity=22.222)
    # Any remaining STOP must explicitly show why stopping was selected;
    # a rear-only unavoidable case may instead select verified acceleration.
    assert target > 0., safety.details
    assert safety.details['decision'] in ('GO','YIELD','MITIGATE')
    if safety.details.get('launch_verified'):
        assert safety.details['cause']=='first_merge_rear_verified_launch'


def test_new_front_body_still_blocks_rear_acceleration(lanes):
    planner,ego,match,entries,path=recorded(lanes)
    pt=lanes.advance_default(match,4.)[0]
    front=ObjectInfo(pt.x,pt.y,0.,2,unique_id=99,size_x=5.,size_y=2.4,
                     heading_deg=math.degrees(match.yaw))
    entries.append({'object_info':front})
    safety=planner.merge_safety
    assert safety.limit(ego,match,planner._match_objects(entries),entries,path,desired_velocity=22.222)==0.
    assert not safety.details.get('launch_verified')
    assert safety.details['cause']=='no_safe_speed'


def test_first_merge_follower_exemption_never_hides_body_collision(lanes):
    planner,ego,match,entries,path=recorded(lanes)
    safety=planner.merge_safety
    safety.zone='FIRST_PREDICTIVE';safety.ego_lane_match=match
    rear=entries[0]['object_info']
    key='npc:49'; safety.second_rear_gap_ids={key}; safety.spacing_ids={key}
    tracks=[(key,(rear,match,0.))]
    predictions=[[[ (ego.position,ego.yaw) ]]]
    hazard=safety._conflict(ego,path,planner._cumulative_distance(path),ego.velocity,tracks,predictions)
    assert hazard['kind']=='obb'




def test_recovery_keeps_steering_limit_after_change_state_ends_and_releases_it():
    controller=PathController(json.loads((ROOT/'AIM2026/Expert/src/config/config.json').read_text()))
    ego=VehicleState(66.78333711065352,138.64361212821677,-1.508640298024612,6.323831346299913)
    path=[Point(66.536,138.642),Point(66.539,138.142)]
    change=HighwayPlanningResult(path,highway_state='CHANGE_RIGHT')
    wait=HighwayPlanningResult(path,highway_state='ACC_FOLLOW')
    with patch.object(controller.pure_pursuit,'calculate_steering_angle',return_value=-.003):
        controller._calculate_control(ego,path,6.,change)
    previous=controller.previous_front_steer_rad
    with patch.object(controller.pure_pursuit,'calculate_steering_angle',return_value=-math.radians(33.1)):
        for _ in range(90):
            _,_,angle=controller._calculate_control(ego,path,0.,wait)
            assert abs(angle-previous)<=math.radians(30.)*controller.control_dt+1e-9
            assert abs(angle)<=math.radians(5.)+1e-9
            previous=angle
    assert wait.debug_details['highway_steering']['settling']
    with patch.object(controller.pure_pursuit,'calculate_steering_angle',return_value=0.):
        for _ in range(30):controller._calculate_control(ego,path,5.,wait)
    assert not wait.debug_details['highway_steering']['settling']


def test_verified_first_merge_launch_is_honored_but_lower_safety_target_wins():
    result=HighwayPlanningResult([],debug_details={'merge_safety':dict(zone='FIRST_PREDICTIVE',
        decision='GO',launch_verified=True,launch_target=12.)})
    ego=VehicleState(0.,0.,0.,1.)
    assert PathController._second_merge_launch_acceleration_gain(12.,ego,result) is not None
    assert PathController._second_merge_launch_acceleration_gain(0.,ego,result) is None
    assert PathController._second_merge_launch_acceleration_gain(5.,ego,result) is None


def test_first_merge_mitigation_survives_route_curvature_speed_processing():
    from AIM2026.Expert.src.planning.highway.common.speed_limits import _merge_ego_curvature_speed_floor
    result=HighwayPlanningResult([],debug_details={'merge_safety':dict(zone='FIRST_PREDICTIVE',
        decision='MITIGATE',speed_floor=8.,rear_mitigation_verified=True,launch_target=8.)})
    assert _merge_ego_curvature_speed_floor(2.,.3,result)==8.
    ego=VehicleState(0.,0.,0.,.3)
    assert PathController._second_merge_launch_acceleration_gain(8.,ego,result) is not None
    assert PathController._second_merge_launch_acceleration_gain(0.,ego,result) is None
