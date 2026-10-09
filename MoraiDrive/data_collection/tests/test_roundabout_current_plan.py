"""Behavioural regression tests; synthetic receipt cases are not data validation."""
import copy
import json
import math
from pathlib import Path

import pytest

from MoraiDrive.data_collection.situation_labeling.expert_instructions import (
    apply_expert_instructions, canonical_state, load_config)
from MoraiDrive.data_collection.situation_labeling.mapper import attach_rule_instructions
from MoraiDrive.data_collection.situation_labeling.instruction_quality import finalize_instruction_quality
from MoraiDrive.data_collection.situation_labeling.renderer import load_catalog
from MoraiDrive.data_collection.situation_labeling.roundabout_current import (
    align_current_roundabout_cruise, validate_current_roundabout_cruise)
from MoraiDrive.data_collection.situation_labeling.consistency import drop_contradicted_instructions

FIXTURE = Path(__file__).parent / 'fixtures/roundabout_current_plan_actual.json'


def raw(source='existing_roundabout_text_log', **kw):
    result = dict(mode='roundabout_entry', roundabout_phase='follow', roundabout_before_merge=True,
                  planner_target_speed_mps=1., ego_speed_mps=4., source=source)
    result.update(kw)
    return result


@pytest.mark.parametrize('age', [0., .1, 1., 4., 30.])
def test_transition_numeric_is_historical_even_near_estimated_onset(age):
    state, error = canonical_state('roundabout', raw(legacy_transition_observation={
        'transition_estimated_age_seconds': age}), load_config()[0])
    assert error is None and state['state'] == 'FOLLOW_MERGE'


@pytest.mark.parametrize('source', ['explicit_ros', 'existing_roundabout_debug'])
def test_exact_current_target_deceleration_is_preserved(source):
    assert canonical_state('roundabout', raw(source), load_config()[0])[0]['state'] == 'FOLLOW_DECEL'


@pytest.mark.parametrize('phase, expected', [('follow', 'FOLLOW_MERGE'), ('yield', 'YIELD_MOVING'),
                                             ('collision_slowdown', 'COLLISION_SLOWDOWN'),
                                             ('emergency', 'EMERGENCY')])
def test_historical_zero_target_cannot_prove_current_stop_plan(phase, expected):
    assert canonical_state('roundabout', raw(roundabout_phase=phase, planner_target_speed_mps=0.,
        ego_speed_mps=0.), load_config()[0])[0]['state'] == expected


@pytest.mark.parametrize('phase, expected', [('follow', 'FOLLOW_STOPPED'), ('yield', 'YIELD_STOPPED'),
                                             ('collision_slowdown', 'COLLISION_STOPPED'),
                                             ('emergency', 'EMERGENCY_STOPPED')])
def test_exact_zero_target_stop_variants_are_preserved(phase, expected):
    assert canonical_state('roundabout', raw('explicit_ros', roundabout_phase=phase,
        planner_target_speed_mps=0., ego_speed_mps=0.), load_config()[0])[0]['state'] == expected


def test_speed_cap_description_does_not_assert_acceleration_direction():
    text = load_catalog()['roundabout']['collision_slowdown']['instruction']
    assert 'Adjust speed in response to the vehicle.' in text
    assert 'Decelerate' not in text and 'Accelerate' not in text


def case():
    data = json.loads(FIXTURE.read_text())
    frames, objects, rules = data['frames'], data['objects'], data['rules']
    f, r = frames[-1], rules[0]
    f['ego_heading_deg'] = 90.
    t = f['lidar_source_ns']; anchor = 1_000_000_000
    def receipt(record_id, **values):
        return dict(record_id=record_id, payload_sha256='a'*64, receive_age_ms=20.,
                    receive_monotonic_ns=anchor-20_000_000,
                    preceding_received_clock_ns=t-20_000_000,
                    source_timestamp_ns=None, raw_chunk='chunks/chunk_000.bin', **values)
    x, y = f['ego_world_x_m'], f['ego_world_y_m']
    f['expert_publications'] = dict(status='RECEIVE_BOUND_CURRENT_CONTEXT', sensor_ns=t,
        lidar_record_id=1000, lidar_receive_monotonic_ns=anchor,
        source_sha256={'quality.json': 'f'*64},
        ego=receipt(1, speed_mps=4.4, yaw_rad=math.pi/2, world_xyz=[x,y,0.], ego_pose_source_age_ms=20.),
        control=receipt(2,longlCmdType=1,accel=.2,brake=0.),
        target_velocity=receipt(3,value_mps=4.6),
        local_path=receipt(4, world_xyz=[[x,y+.1,0.],[x,y+6.,0.]]))
    f['expert_publications']['ego']['source_timestamp_ns']=t-20_000_000
    r['expert_raw_state'] = raw(roundabout_front_id='1', roundabout_committed=False)
    r['evidence'] = {}
    r['expert_state']='FOLLOW_MERGE';r['instruction_id']='follow_merge'
    r['instruction']='There is a vehicle in front of the ego vehicle. Merge into the roundabout while following the lead vehicle.'
    r['status']='unknown';r['unknown_reasons']=['conflict_response_unverified']
    r['instruction_state']=None;r['label']='follow'
    r['expert_raw_state']['legacy_transition_observation'] = {'numeric_target_current_verified':False}
    return frames, objects, rules


def test_current_path_plan_recovers_without_a_fabricated_follow_actor_cause():
    fs, obs, rs = case(); r=rs[0]
    align_current_roundabout_cruise(fs,obs,rs)
    attach_rule_instructions(rs)
    drop_contradicted_instructions(fs,rs,objects=obs)
    finalize_instruction_quality([fs[-1]],rs)
    assert r['action_valid'] is True and r['observation_valid'] is True
    assert r['instruction']=='Continue driving along the current path.'
    assert r['expert_state'] is None and r['label'] is None and r['instruction_variables']=={}
    proof=r['evidence']['current_roundabout_cruise']
    assert proof['physical_simultaneity_verified'] is False
    assert proof['physical_front_safety_certified'] is False
    assert proof['current_expert_target'] is None
    assert validate_current_roundabout_cruise(fs,obs,r)==(True,[])


def test_exact_sealed_gt_ego_is_not_required_to_arrive_with_unstamped_control():
    fs,obs,rs=case();c=fs[-1]['expert_publications'];e=c['ego']
    e.update(source_timestamp_ns=fs[-1]['lidar_source_ns'], ego_pose_source_age_ms=0.,
             source_association_basis='exact_sealed_gt_scene_source_hypothesis_and_prior_receipt',
             receive_age_ms=154.,receive_monotonic_ns=c['lidar_receive_monotonic_ns']-154_000_000)
    align_current_roundabout_cruise(fs,obs,rs)
    assert rs[0]['instruction']=='Continue driving along the current path.'
    assert rs[0]['evidence']['current_roundabout_cruise']['physical_simultaneity_verified'] is False


def test_unstamped_control_does_not_inherit_the_exact_ego_source_exception():
    fs,obs,rs=case();c=fs[-1]['expert_publications'];item=c['control']
    item.update(source_timestamp_ns=fs[-1]['lidar_source_ns'], ego_pose_source_age_ms=0.,
                source_association_basis='exact_sealed_gt_scene_source_hypothesis_and_prior_receipt',
                receive_age_ms=154.,receive_monotonic_ns=c['lidar_receive_monotonic_ns']-154_000_000)
    align_current_roundabout_cruise(fs,obs,rs)
    assert rs[0]['instruction'] is None


@pytest.mark.parametrize('target, brake, text', [
    (1., .2, 'Slow down along the current path.'),
    (4.397, .001, 'Slow down along the current path.'),
    (0., 1., 'Stop along the current path.')])
def test_fresh_braking_plan_recovers_without_actor_or_safety_claim(target,brake,text):
    fs,obs,rs=case();c=fs[-1]['expert_publications']
    c['control'].update(accel=0.,brake=brake);c['target_velocity']['value_mps']=target
    align_current_roundabout_cruise(fs,obs,rs);attach_rule_instructions(rs)
    drop_contradicted_instructions(fs,rs,objects=obs)
    finalize_instruction_quality([fs[-1]],rs)
    assert rs[0]['instruction']==text and rs[0]['action_valid'] is True
    assert rs[0]['expert_state'] is None and rs[0]['instruction_variables']=={}
    assert validate_current_roundabout_cruise(fs,obs,rs[0])==(True,[])


@pytest.mark.parametrize('mutation', ['future_receipt', 'stale_receipt', 'future_clock', 'stale_clock',
    'brake', 'no_accel_target_below_ego', 'zero_target', 'stopped_ego', 'wrong_mode', 'path_behind',
    'path_far_start', 'missing_path', 'missing_control', 'wrong_sensor', 'bad_hash',
    'frame_not_admitted', 'observation_mask', 'action_mask', 'unrelated_unknown', 'emergency'])
def test_incomplete_conflicting_or_excluded_sources_never_create_drive_labels(mutation):
    fs,obs,rs=case();f,r=fs[-1],rs[0];c=f['expert_publications']
    if mutation=='future_receipt': c['control']['receive_age_ms']=-.01
    elif mutation=='stale_receipt': c['control']['receive_age_ms']=100.01
    elif mutation=='future_clock': c['control']['preceding_received_clock_ns']=f['lidar_source_ns']+1
    elif mutation=='stale_clock': c['control']['preceding_received_clock_ns']=f['lidar_source_ns']-100000001
    elif mutation=='brake': c['control']['brake']=.2
    elif mutation=='no_accel_target_below_ego': c['control']['accel']=0.;c['target_velocity']['value_mps']=1.
    elif mutation=='zero_target': c['target_velocity']['value_mps']=0.
    elif mutation=='stopped_ego': c['ego']['speed_mps']=0.
    elif mutation=='wrong_mode': c['control']['longlCmdType']=2
    elif mutation=='path_behind': c['local_path']['world_xyz'][-1][1]=f['ego_world_y_m']-10
    elif mutation=='path_far_start': c['local_path']['world_xyz'][0][0]+=4.
    elif mutation=='missing_path': c['local_path']=None
    elif mutation=='missing_control': c['control']=None
    elif mutation=='wrong_sensor': c['sensor_ns']+=1
    elif mutation=='bad_hash': c['control']['payload_sha256']='0'
    elif mutation=='frame_not_admitted': f['dataset_build_candidate']=False
    elif mutation=='observation_mask': r['observation_valid']=False
    elif mutation=='action_mask': r['action_valid']=False
    elif mutation=='unrelated_unknown': r['unknown_reasons'].append('missing_geometry_actor')
    elif mutation=='emergency': r['expert_raw_state']['roundabout_phase']='emergency'
    align_current_roundabout_cruise(fs,obs,rs)
    assert r.get('instruction') is None and r.get('action_valid') is False


@pytest.mark.parametrize('tamper', ['source_hash', 'path', 'text', 'slot', 'receipt', 'source_pin'])
def test_final_validator_recomputes_plan_and_rendering(tamper):
    fs,obs,rs=case();r=rs[0];align_current_roundabout_cruise(fs,obs,rs)
    if tamper=='source_hash': r['evidence']['current_roundabout_cruise']['current_plan']['path_geometry_sha256']='0'*64
    elif tamper=='path': fs[-1]['expert_publications']['local_path']['world_xyz'][-1][1]+=.1
    elif tamper=='text': r['instruction']='Follow the lead vehicle.'
    elif tamper=='slot': r['instruction_variables']={'position':'in front of'}
    elif tamper=='receipt': fs[-1]['expert_publications']['control']['payload_sha256']='f'*64
    elif tamper=='source_pin': fs[-1]['expert_publications']['source_sha256']['quality.json']='0'*64
    valid,reasons=validate_current_roundabout_cruise(fs,obs,r)
    assert valid is False and reasons


def test_uncertain_emergency_transition_does_not_claim_immediate_braking():
    fs,obs,rs=case();f,r=fs[-1],rs[0]
    source=raw(roundabout_phase='emergency',roundabout_conflict_id='1',planner_target_speed_mps=0.,
               legacy_transition_observation={'phase_within_anchor_bounds':False})
    apply_expert_instructions(fs,rs,None,[],index_observations={f['index']:source},objects=obs)
    assert r['instruction'] is None
    assert r['expert_alignment']['reason']=='expert_roundabout_emergency_transition_time_unverified'


@pytest.mark.parametrize('module_name', ['expert_static_phase','expert_roundabout_phase'])
@pytest.mark.parametrize('times,transition,expected', [([],2.,None),([1.,2.],2.,1),
    ([1.,2.],2.000001,None),([1.,2.],.9,0)])
def test_phase_after_capture_end_has_no_frame(module_name,times,transition,expected):
    import importlib
    module=importlib.import_module('MoraiDrive.data_collection.situation_labeling.'+module_name)
    assert module._frame_index_for_time(times,transition)==expected


def test_future_roundabout_emergency_does_not_overwrite_last_captured_phase(tmp_path):
    from MoraiDrive.data_collection.situation_labeling.expert_roundabout_phase import build_roundabout_observations
    log=tmp_path/'supervisor.log'
    log.write_text('[INFO] [1.000, 1.000]: [ExpertInput]\n[CTRL] one\n'
        '[ROUNDABOUT] cruise target=10.0km/h conflict_id=None ttc=infs front_id=None gap=None hold_gap=5m entering=False merged=False\n'
        '[CTRL] two\n[ROUNDABOUT] emergency target=0.0km/h conflict_id=1 ttc=0.0s front_id=1 gap=1 hold_gap=5m entering=True merged=False\n'
        '[INFO] [3.000, 3.000]: [ExpertInput]\n')
    frames=[{'index':i,'zone_id':'roundabout','lidar_source_ns':1_000_000_000+i*500_000_000} for i in range(3)]
    assert build_roundabout_observations(frames,log)[2]['roundabout_phase']=='cruise'


def test_future_static_rejoin_does_not_overwrite_last_captured_follow(tmp_path):
    from MoraiDrive.data_collection.situation_labeling.expert_static_phase import attach_static_phase
    log=tmp_path/'supervisor.log'
    log.write_text('[INFO] [1.000, 1.000]: [ExpertInput]\n[CTRL] one\n'
        '[STATIC] static_recorded_follow offset=1.0 target=1.0\n'
        '[CTRL] two\n[STATIC] static_recorded_rejoined offset=0.0 target=2.0\n'
        '[INFO] [3.000, 3.000]: [ExpertInput]\n')
    frames=[{'index':i,'lidar_source_ns':1_000_000_000+i*500_000_000} for i in range(3)]
    attach_static_phase(frames,log)
    assert frames[-1]['expert_static_phase']=='static_recorded_follow'
