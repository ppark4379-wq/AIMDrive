"""PID math, MORAI wire format, stale-input braking and real loopback UDP."""
import copy
import ctypes
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from control.config import load_config, validate_config
from control.protocol import PACKET, decode_message, encode_message, pack_command
from control.runtime import ControllerSession
from control.udp_bridge import ControlBridge
from control.waypoint_pid import WaypointPID, current_waypoints

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    cfg = load_config(ROOT/'../AIM2026/Control/vldrive_control.json')
    cfg['max_speed_kmh']=None  # Historical uncapped longitudinal regression.
    cfg['lateral']['target']['mode']='first_two_midpoint'  # Historical midpoint regression.
    # Historical LMDrive math fixtures only. The live controller reads expert.
    cfg['pid'] = dict(turn=dict(K_P=1.25,K_I=.75,K_D=.3,n=40),
        speed=dict(K_P=5.,K_I=.5,K_D=1.,n=40),max_throttle=.75,
        brake_speed=.1,brake_ratio=1.1,clip_delta=.35,steering_scale=.8,
        morai_steering_sign=1.,max_speed_mps=5.)
    return cfg


def prediction():
    return dict(coordinate_convention='lmdrive_left_negative_forward',
                waypoints_raw=[[0., -2.*i] for i in range(1, 6)],
                horizons_s=[.5, 1., 1.5, 2., 2.5], anchor_received_at=9.5,
                sim_time=20., frame=100, history_generation=0,
                navigation=dict(valid=True, pose=dict(x=0., y=0., yaw_rad=0.)))


def test_pid_matches_original_lmdrive_math_over_window(cfg):
    # Independent transcription of original deque PID and agent control_pid.
    pid = WaypointPID(cfg['pid'])
    steering_errors, speed_errors = [0.]*40, [0.]*40
    for k in range(75):
        points = np.array([[np.sin(k/10), 2.], [np.sin(k/10)*2, 4.]])
        speed = 0. if k == 0 else 3.+np.cos(k/8)
        desired = min(np.linalg.norm(points[0]-points[1])*2, 5.)
        brake = desired < .1 or speed/desired > 1.1
        aim = points.mean(axis=0)
        angle = np.degrees(np.pi/2-np.arctan2(aim[1], aim[0]))/90
        if speed < .01: angle = 0.
        steering_errors = steering_errors[1:]+[angle]
        expected_steer = np.clip(1.25*angle+.75*np.mean(steering_errors)+
                                .3*(steering_errors[-1]-steering_errors[-2]), -1, 1)*.8
        delta = np.clip(desired-speed, 0, .35)
        speed_errors = speed_errors[1:]+[delta]
        expected_throttle = 0. if brake else np.clip(5*delta+.5*np.mean(speed_errors)+
                                                     (speed_errors[-1]-speed_errors[-2]), 0, .75)
        actual = pid.step(points, speed)
        assert actual['steer'] == pytest.approx(expected_steer)
        assert actual['accel'] == pytest.approx(expected_throttle)
        assert actual['brake'] == float(brake)


def test_steering_sign_speed_ceiling_and_stationary_prediction(cfg):
    assert cfg['pid']['morai_steering_sign'] == 1.  # Observed EgoCtrlCmd polarity.
    assert WaypointPID(cfg['pid']).step([[2, 4], [4, 8]], 2)['steer'] > 0  # left
    assert WaypointPID(cfg['pid']).step([[-2, 4], [-4, 8]], 2)['steer'] < 0  # right
    capped = WaypointPID(cfg['pid']).step([[0, 20], [0, 40]], 6)
    assert capped['desired_speed_mps'] == 5 and capped['accel'] == 0 and capped['brake'] == 1
    stopped = WaypointPID(cfg['pid']).step([[0, 0], [0, 0]], 0)
    assert stopped['brake'] == 1 and stopped['accel'] == stopped['steer'] == 0


def test_configurable_polarity_only_changes_wire_steering(cfg):
    other = copy.deepcopy(cfg['pid']);other['morai_steering_sign'] = -1.
    observed, alternative = WaypointPID(cfg['pid']), WaypointPID(other)
    for side in [1., 2., -.5, -2., 0.]:
        points = [[side, 4.], [2*side, 8.]]
        left = observed.step(points, 2.);right = alternative.step(points, 2.)
        assert left['steer'] == -right['steer']
        assert left['steer_pid_left_positive'] == right['steer_pid_left_positive']
        assert left['accel'] == right['accel'] and left['brake'] == right['brake']
        assert left['desired_speed_mps'] == right['desired_speed_mps']


def test_recorded_left_aim_is_positive_in_actual_udp_packet(cfg):
    # User's 02:29:40 run, frame 674: left aim but wire -0.8 and decreasing yaw.
    # Preserve this observed failure independently of the runtime config value.
    aim = np.array([1.935543178190159, .8574746125290273])
    observed_yaw_rate_deg_s = -38.99040508007397
    previous_wire_steer = -.8
    assert previous_wire_steer*observed_yaw_rate_deg_s > 0  # Negative actually turned right.
    points = np.stack([aim-[0.,1.], aim+[0.,1.]])
    command = WaypointPID(cfg['pid']).step(points, 4.408465226491292)
    assert command['angle_error'] == pytest.approx(.7345104462766784)
    wire = PACKET.unpack(pack_command(command))[-2]
    assert wire > 0 and wire == pytest.approx(.8)  # Correct left command at saturation.


def test_reproject_and_discard_elapsed_prediction_horizons():
    result = prediction()
    points, horizons, indices = current_waypoints(result, dict(x=1., y=0., yaw_rad=0.), .65)
    assert indices == [1, 2, 3, 4]
    np.testing.assert_allclose(points, [[0, 3], [0, 5], [0, 7], [0, 9]])
    result['navigation']['pose'] = dict(x=10., y=10., yaw_rad=np.pi/2)
    result['waypoints_raw'] = [[1., -2.*i] for i in range(1, 6)]
    points, _, _ = current_waypoints(result, dict(x=9., y=11., yaw_rad=0.), 0.)
    np.testing.assert_allclose(points, [[1, 0], [3, 0], [5, 0], [7, 0], [9, 0]], atol=1e-12)


@pytest.mark.parametrize('case', ['expired','negative_age','nan','jump','bad_horizon','behind','convention'])
def test_bad_waypoints_rejected(case):
    result = prediction(); pose = dict(x=0., y=0., yaw_rad=0.); age = .1
    if case == 'expired': age = 2.
    if case == 'negative_age': age = -.1
    if case == 'nan': result['waypoints_raw'][0][0] = float('nan')
    if case == 'jump': pose['x'] = 20
    if case == 'bad_horizon': result['horizons_s'][0] = .1
    if case == 'behind': pose['x'] = 5
    if case == 'convention': result['coordinate_convention'] = 'other'
    with pytest.raises(ValueError): current_waypoints(result, pose, age)


def test_packet_exactly_matches_vendored_morai_ctypes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'inference/third_party/morai_network_module'))
    from lib.define.EgoCtrlCmd import EgoCtrlCmd
    official = EgoCtrlCmd()
    official.ctrl_mode, official.gear, official.cmd_type = 2, 4, 1
    official.accel, official.brake, official.steer = .25, 0., -.3
    packet = pack_command(dict(accel=.25, brake=0., steer=-.3))
    assert len(packet) == ctypes.sizeof(official) == 55
    assert packet == bytes(official)


@pytest.mark.parametrize('competition',[False,True])
def test_morai_udp_speed_is_signed_longitudinal_mps(competition):
    from inference.sensors.morai_udp import VehicleStatusReceiver, VehicleStatusPacket, CompetitionVehicleStatusPacket
    cls = CompetitionVehicleStatusPacket if competition else VehicleStatusPacket
    packet = cls();packet.header = b'#MoraiInfo$'
    packet.signed_vel = 36.;packet.vel_x = -12.;packet.vel_y = 33.
    receiver = VehicleStatusReceiver.__new__(VehicleStatusReceiver)
    value = receiver._consume(bytes(packet))
    assert value['speed'] == 10. and value['signed_speed'] == 36.
    packet.signed_vel = -3.6
    assert receiver._consume(bytes(packet))['speed'] == pytest.approx(-1.)


@pytest.mark.parametrize('payload', [b'[]', b'null', b'garbage', b'{"schema":0}', b'{}'])
def test_bridge_rejects_malformed_envelopes(payload):
    with pytest.raises((ValueError, KeyError, TypeError)):
        decode_message(payload, 'a'*32)


@pytest.mark.parametrize('command', [dict(accel=float('nan'),brake=0,steer=0),
    dict(accel=.1,brake=1,steer=0),dict(accel=0,brake=0,steer=1.1)])
def test_invalid_controls_never_packed(command):
    with pytest.raises(ValueError): pack_command(command)


def bound_udp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0)); sock.settimeout(1.)
    return sock


def test_bridge_actual_udp_handshake_watchdog_and_shutdown(cfg):
    target = bound_udp(); client = bound_udp(); reservation = bound_udp()
    port = reservation.getsockname()[1]; reservation.close()
    cfg['bridge'].update(listen_host='127.0.0.1',listen_port=port,
                         target_port=target.getsockname()[1],watchdog_s=.12,idle_exit_s=.6)
    bridge = ControlBridge(cfg, 'a'*32)
    thread = threading.Thread(target=bridge.run, daemon=True); thread.start()
    address = ('127.0.0.1', port)
    try:
        client.sendto(encode_message('a'*32, 1, 'hello'), address)
        assert json.loads(client.recv(2048))['accepted']
        target.settimeout(.08)
        with pytest.raises(socket.timeout): target.recv(2048)  # hello cannot move vehicle
        target.settimeout(1.)
        client.sendto(b'[]', address)
        client.sendto(encode_message('b'*32, 2, 'command', dict(accel=.5,brake=0,steer=0)), address)
        client.sendto(encode_message('a'*32, 2, 'command', dict(accel=.25,brake=0,steer=-.3)), address)
        assert PACKET.unpack(target.recv(2048))[-4:-1] == pytest.approx((.25,0.,-.3))
        client.recv(2048)
        client.sendto(encode_message('a'*32, 2, 'command', dict(accel=.6,brake=0,steer=0)), address)
        # Duplicate/foreign/malformed traffic cannot keep a throttle alive.
        assert PACKET.unpack(target.recv(2048))[-4:-1] == (0.,1.,0.)
        client.sendto(encode_message('a'*32, 3, 'stop'), address)
        thread.join(2.)
        assert not thread.is_alive()
        assert bridge.report['watchdog_brakes'] >= 1 and bridge.report['rejected'] == 3
    finally:
        target.close(); client.close()


@pytest.fixture
def controller(cfg, tmp_path):
    cfg.pop('pid')
    session = ControllerSession(cfg, ROOT/'../AIM2026/Control/vldrive_control.json', tmp_path, clock=lambda:10.)
    gps = SimpleNamespace(snapshot=lambda: (dict(latitude=0.,longitude=0.),.01,1,9.99))
    vehicle = dict(yaw=0.,speed=2.,sim_time=20.5)
    status = SimpleNamespace(snapshot=lambda: (vehicle,.01,1,9.99))
    session.source = SimpleNamespace(gps=gps,status=status,
                                     stream_report=lambda: [('lidar',2368,1,1,.01)])
    session.navigation = SimpleNamespace(converter=SimpleNamespace(transform=lambda lon,lat:(0.,0.)),
                                         position=dict(east_offset_m=0.,north_offset_m=0.))
    session.history = SimpleNamespace(resets=0)
    session.pipeline_state = dict(navigation=dict(valid=True),error=None,missing=[])
    session.sensor_max_age = .3
    session.client = SimpleNamespace(poll=lambda:None,last_ack=9.99)
    session.submit(prediction())
    return session


def test_controller_tracks_then_holds_on_invalid_input(controller):
    assert controller.state['steering_positive_direction'] == 'left'
    actual = controller.calculate(10.)
    assert actual['reason'] == 'tracking' and actual['accel'] > 0
    assert actual['waypoint_indices'] == [0, 1, 2, 3, 4]
    assert controller.counts['pid_updates'] == 1
    controller.calculate(10.05)
    assert controller.counts['pid_updates'] == 2
    controller.calculate(10.10)
    assert controller.counts['pid_updates'] == 3
    controller.invalidate('test_stop')
    assert controller.calculate(10.15) == dict(accel=0.,brake=1.,steer=0.,reason='test_stop')


def test_packet_arriving_during_tick_is_not_marked_future_stale(controller):
    gps = controller.source.gps.snapshot()
    controller.source.gps.snapshot = lambda:(gps[0],0.,gps[2],10.001)
    controller.clock = lambda:10.002
    assert controller.calculate(10.)['reason'] == 'tracking'


@pytest.mark.parametrize('case', ['stale_prediction','reset','navigation','camera','gps',
                                 'status','ack','rewind','worker','nan','future_time'])
def test_controller_brakes_on_fault(controller, case):
    assert controller.calculate(10.)['reason'] == 'tracking'
    if case == 'stale_prediction': controller.result['anchor_received_at'] = 8.
    if case == 'future_time': controller.result['anchor_received_at'] = 11.
    if case == 'reset': controller.history.resets = 1
    if case == 'navigation': controller.pipeline_state['navigation'] = None
    if case == 'camera': controller.source.stream_report = lambda:[('rgb_front',10501,1,1,.8)]
    if case == 'gps': controller.source.gps.snapshot = lambda:(None,1,0,None)
    if case == 'status': controller.source.status.snapshot = lambda:(None,1,0,None)
    if case == 'ack': controller.client.last_ack = 8.
    if case == 'rewind': controller.result['sim_time'] = 30.
    if case == 'worker': controller.pipeline_state['error'] = 'worker error'
    if case == 'nan': controller.result['waypoints_raw'][0][0] = float('nan')
    actual = controller.calculate(10.1)
    assert actual['accel'] == 0 and actual['brake'] == 1 and actual['reason'] != 'tracking'


def test_expert_runtime_uses_original_point_and_latest_control_frame(controller):
    vehicle=controller.source.status.snapshot()[0]
    vehicle.update(speed=1.12,steer=-12.5)
    controller.result['waypoints_raw']=[[.3*i,-2.*i] for i in range(1,6)]
    out=controller.calculate(10.)
    assert out['reason']=='tracking' and out['waypoint_indices']==[0,1,2,3,4]
    assert out['aim_left_forward']==pytest.approx([.45,3.]) and out['steer']>0
    assert out['aim_distance_m']==pytest.approx(np.hypot(.45,3.))
    assert out['steering_measured_deg']==-12.5 and out['steering_requested_deg']<0
    assert out['expert_sampling_time_s']==pytest.approx(1/30)
    assert out['desired_speed_mps']==pytest.approx(np.hypot(.3,2.)/.5)
    next_prediction=copy.deepcopy(controller.result)
    next_prediction.update(frame=101,anchor_received_at=9.8)
    controller.submit(next_prediction)
    held=controller.calculate(10.05)
    assert held['prediction_frame']==101 and held['prediction_age_s']==pytest.approx(.25)
    assert held['lateral_prediction_frame']==100 and not held['lateral_updated']
    assert held['steer']==out['steer']
    updated=controller.calculate(10.1)
    assert updated['prediction_frame']==101 and updated['prediction_age_s']==pytest.approx(.3)
    assert updated['lateral_prediction_frame']==101 and updated['lateral_updated']


def test_new_prediction_cannot_extend_expired_held_command(controller):
    controller.result['anchor_received_at']=8.78
    assert controller.calculate(10.)['prediction_age_s']==pytest.approx(1.22)
    next_prediction=copy.deepcopy(controller.result)
    next_prediction.update(frame=101,anchor_received_at=9.8)
    controller.submit(next_prediction)
    out=controller.calculate(10.05)
    assert out['prediction_frame']==101 and out['prediction_age_s']==pytest.approx(.25)
    assert out['lateral_prediction_frame']==101 and out['lateral_updated']
    assert controller.counts['pid_updates']==2


def test_held_lateral_target_is_reprojected_to_current_pose(controller):
    first=controller.calculate(10.)
    controller.navigation.converter.transform=lambda lon,lat:(1.,0.)
    held=controller.calculate(10.04)
    assert not held['lateral_updated'] and held['steer']==first['steer']
    assert held['aim_left_forward']==pytest.approx([0.,2.])
    assert held['aim_distance_m']==pytest.approx(2.)
    assert held['tracking_polyline_left_forward'][0]==pytest.approx([0.,1.])


def test_invalid_input_resets_lateral_window_and_hold_state(controller):
    controller.result['waypoints_raw']=[[i,-2.*i] for i in range(1,6)]
    first=controller.calculate(10.)
    controller.invalidate('test_stop')
    controller.calculate(10.02)
    assert controller.lateral_context is None
    new=prediction();new['waypoints_raw']=[[i,-2.*i] for i in range(1,6)]
    controller.submit(new)
    restarted=controller.calculate(10.03)
    assert restarted['lateral_updated'] and restarted['steer']==first['steer']


def test_remaining_target_already_passed_during_hold_is_refreshed_immediately(controller):
    controller.cfg['lateral']['target']['mode']='remaining_path'
    first=controller.calculate(10.)
    assert first['aim_left_forward']==pytest.approx([0,4])
    controller.navigation.converter.transform=lambda lon,lat:(4.1,0.)
    next_command=controller.calculate(10.04)
    assert next_command['lateral_updated']
    assert next_command['aim_left_forward']==pytest.approx([0,4])
    assert not next_command['aim_is_behind']


def test_insufficient_remaining_path_stops_even_between_lateral_ticks(controller):
    controller.cfg['lateral']['target']['mode']='remaining_path'
    assert controller.calculate(10.)['reason']=='tracking'
    controller.navigation.converter.transform=lambda lon,lat:(7.,0.)
    stopped=controller.calculate(10.04)
    assert stopped==dict(reason='insufficient_remaining_path',accel=0.,brake=1.,steer=0.)
    assert controller.lateral_context is None
    controller.navigation.converter.transform=lambda lon,lat:(0.,0.)
    restarted=controller.calculate(10.08)
    assert restarted['lateral_updated'] and restarted['reason']=='tracking'


def test_control_cli_configuration_requires_live_pose():
    from inference.realtime.realtime_vldrive import validate_config as validate_live
    cfg = json.loads((ROOT/'../AIM2026/Inference/vldrive_realtime.json').read_text())
    cfg['control']['enabled'] = True
    validate_live(cfg)
    cfg['navigation']['mode'] = 'fixed'
    with pytest.raises(ValueError, match='requires live route'): validate_live(cfg)


def test_live_pipeline_publishes_actual_predict_result_and_closes_control_first(monkeypatch, tmp_path):
    import torch
    from inference.realtime import live_pipeline as pipeline
    cfg = json.loads((ROOT/'../AIM2026/Inference/vldrive_realtime.json').read_text())
    cfg.pop('navigation_token_config', None)  # No real tokenizer in this pipeline fixture.
    cfg['runtime'].update(device='cpu',history_tolerance_s=.019,max_age_s=1.,resample_history=False)
    cfg['visualization']['enabled'] = False
    events = []; published = []; predictions = []
    started = time.monotonic()
    class Source:
        def __init__(self, *args, **kwargs): self.lidar = self
        def snapshot(self):
            index = int((time.monotonic()-started)/.02)+1
            return None, 0., index, started+(index-1)*.02
        def stream_report(self): return [('lidar',2368,1,1,0.)]
        def build(self, *args):
            _, _, index, stamp = self.snapshot()
            return {'x':torch.zeros(1)}, dict(missing=[],received_at={'lidar':stamp},
                sample_ids={'lidar':index},sim_time=stamp,front_jpeg=b'jpeg')
        def close(self): events.append('source_close')
    class Control:
        state = {'enabled':True}
        def start(self, *args): events.append('control_start')
        def check_error(self): pass
        def invalidate(self, reason): pass
        def submit(self, value): published.append(value)
        def close(self): events.append('control_close')
    def predict(*args, **kwargs):
        result = prediction(); result.update(selected_tokens=32,vldrive_ms=0.)
        predictions.append(result)
        return result
    monkeypatch.setattr(pipeline, 'LiveFrameSource', Source)
    monkeypatch.setattr(pipeline, 'load_stage2', lambda *args:(object(),{}))
    monkeypatch.setattr(pipeline, 'load_encoder', lambda *args:object())
    monkeypatch.setattr(pipeline, 'encode_frame', lambda *args:(torch.zeros(106,256),0.))
    monkeypatch.setattr(pipeline, 'predict_window', predict)
    monkeypatch.setattr(pipeline, 'navigation_for_frame', lambda *args:dict(valid=True,
        instruction={'text':'Turn left.','rule':'test'},pose=dict(x=0.,y=0.,yaw_rad=0.)))
    report = {}
    pipeline.run_live(cfg,dict(meta={'encoder_contract':{}},history_interval_s=.02),
                      tmp_path,report,duration=.8,navigation=object(),controller=Control())
    assert report['outputs'] >= 1 and len(published) == report['outputs']
    assert all(any(result is original for original in predictions) for result in published)
    assert all(result['instruction']=='Turn left.' and result['history_ready'] for result in published)
    assert all('history_generation' in result and len(result['history_sample_ids'])==11 for result in published)
    assert events.index('control_close') < events.index('source_close')
