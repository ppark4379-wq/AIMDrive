"""Simulator-free checks of command restrictions, causality and PID axes."""
import ctypes
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'VLDrive'))
from AIM2026.Control.waypoint import WaypointController, to_world, from_world, remaining_waypoints
from AIM2026.UDP.Cmd import Command, FORMAT, CmdSender
from AIM2026.UDP.Common import ReceivedSample
from AIM2026.UDP.VehicleStatus import CompetitionVehicleStatusPacket, VehicleStatusPacket, VehicleStatusReceiver
from AIM2026.UDP.GPS import GpsReceiver
from AIM2026.Inference.runtime.history import FeatureFrame, InstructionHistory, fit_context
from AIM2026.Inference.sensors.synchronizer import causal_sample, SensorSynchronizer
from AIM2026.Inference.runtime.runner import LatestSlot, LiveRunner
from AIM2026.Inference.navigation.route import RouteZones


def frame(i, segment='straight', source_time=None):
    return FeatureFrame(i*.1 if source_time is None else source_time, i*.1, i,
        dict(x=0., y=0., yaw_rad=0.), np.zeros((106, 256), np.float16),
        dict(instruction=segment, segment=segment, zone_id=0))


def test_command_format_and_competition_modes():
    packed = Command(accel=.2, brake=0., steer=-.3).pack()
    assert len(packed) == 55
    fields = FORMAT.unpack(packed)
    assert fields[0].rstrip(b'\0') == b'#MoraiCtrlCmd$'
    assert fields[5:8] == (2, 4, 1)
    assert fields[-1] == b'\r\n'


@pytest.mark.parametrize('changes', [dict(ctrl_mode=1), dict(cmd_type=2), dict(cmd_type=3),
    dict(accel=math.nan), dict(steer=1.1), dict(brake=-1.)])
def test_command_rejects_disallowed_controls(changes):
    with pytest.raises(ValueError):
        Command(**changes).pack()


def test_arbitrary_double_packet_protocol_is_not_available():
    with pytest.raises(ValueError):
        CmdSender('127.0.0.1', 7600, protocol='manual_doubles')


def test_status_accepts_only_competition_layout_and_does_not_expose_position():
    receiver = object.__new__(VehicleStatusReceiver)
    packet = CompetitionVehicleStatusPacket()
    packet.header, packet.tail = b'#MoraiInfo$', b'\r\n'
    packet.sec, packet.nsec, packet.signed_vel = 10, 100000000, 36.
    packet.yaw, packet.pos_x = 90., 12345.
    result = receiver._consume(bytes(packet))
    assert result['speed'] == 10.
    assert result['yaw'] == pytest.approx(math.pi/2)
    assert result['sim_time'] == pytest.approx(10.1)
    assert result['competition_variant'] is True
    assert not any(key.startswith(('pos_', 'vel_', 'tire_', 'accel_')) for key in result)
    assert receiver._consume(bytes(VehicleStatusPacket())) is None
    assert receiver._consume(bytes(packet) + b'\0') is None


def test_gps_rejects_invalid_fix():
    receiver = object.__new__(GpsReceiver)
    receiver._fix = dict(latitude=None, longitude=None, altitude=None)
    assert receiver._consume(b'$GPGGA,120000,3724.1,N,12646.1,E,0,00,1,10,M,0,M,,*00') is None
    assert receiver._consume(b'$GPRMC,120000,V,3724.1,N,12646.1,E,0,0,010126,,,A*00') is None


def test_pid_matches_lmdrive_first_step_and_repeated_integral():
    controller = WaypointController(positive_steer='right')
    waypoints = np.column_stack(([.5, 1., 1.5, 2., 2.5], -np.arange(1, 6)*2.))
    # Original LMDrive coordinates [right, -forward], CARLA steer right+.
    lm = waypoints.copy()
    lm[:, 0] *= -1
    lm[:, 1] *= -1
    aim = lm[:2].mean(0)
    angle = (math.pi/2 - math.atan2(aim[1], aim[0])) / (math.pi/2)
    for step in range(1, 5):
        result = controller.step(waypoints, 2.)
        derivative = angle if step == 1 else 0.
        expected = np.clip(1.25*angle + .75*angle*step/40 + .3*derivative, -1., 1.) * .8
        assert result.steer == pytest.approx(expected)
        assert result.accel == .75
        assert result.desired_speed_mps == pytest.approx(np.linalg.norm(lm[1]-lm[0])*2)


def test_turn_sign_and_braking():
    waypoints = np.column_stack((np.arange(1, 6), -np.arange(1, 6)*2))
    left = WaypointController(positive_steer='left')
    right = WaypointController(positive_steer='right')
    assert left.step(waypoints, 1.).steer > 0
    assert right.step(waypoints, 1.).steer < 0
    assert left.step(waypoints, 10.).accel == 0.
    assert left.step(waypoints, 10.).brake == 1.
    assert left.step(np.zeros((5, 2)), 0.).brake == 1.
    assert left.step(waypoints, 0.).steer == pytest.approx(0., abs=.1)
    assert left.step(waypoints, -1.).reason == 'reverse_motion'


def test_prediction_coordinates_and_latency_rebasing():
    pose = dict(x=10., y=20., yaw_rad=math.pi/2)
    points = np.column_stack((np.zeros(5), -np.arange(1, 6)*2.))
    world = to_world(points, pose)
    assert np.allclose(from_world(world, pose), points)
    moved = dict(pose, y=22.)  # 4 m/s, 0.5 s after acquisition
    remaining = remaining_waypoints(world, moved, .5)
    assert np.allclose(remaining[:2], points[:2])
    with pytest.raises(ValueError):
        remaining_waypoints(world, moved, 1.51)


def test_history_is_variable_instruction_bounded_and_keeps_latest():
    history = InstructionHistory(stride=2)
    for i in range(32):
        history.append(frame(i))
    generation, selected = history.snapshot()
    assert len(selected) == 17  # more than the training window's eleven
    assert selected[-1].sequence == 31
    history.append(frame(32, segment='left'))
    new_generation, selected = history.snapshot()
    assert new_generation == generation + 1
    assert [f.sequence for f in selected] == [32]


@pytest.mark.parametrize('stamp', [0., .9])
def test_source_gap_or_clock_reset_discards_old_history(stamp):
    history = InstructionHistory()
    history.append(frame(1))
    history.append(frame(2, source_time=stamp))
    assert len(history.snapshot()[1]) == 1


def test_context_budget_drops_whole_old_frames_not_text():
    frames = [frame(i) for i in range(30)]
    kept, bound = fit_context(frames, 6, [30]*30, [40]*30, 2048)
    assert len(kept) == 26 and kept[-1] is frames[-1]
    assert bound <= 2048
    with pytest.raises(ValueError):
        fit_context([frame(0)], 6, [2048], [1], 2048)


def test_causal_speed_uses_source_time_not_newest_receipt():
    anchor = ReceivedSample(None, 20., 10, 10.)
    old = ReceivedSample('old', 19.98, 1, 9.98)
    future = ReceivedSample('future', 19.99, 2, 10.01)
    assert causal_sample([old, future], anchor, .05) is old
    assert causal_sample([ReceivedSample(None, 19.99, 1, 9.94)], anchor, .05) is None
    # GPS/sweep use receipt time without claiming common simulator time.
    assert causal_sample([ReceivedSample('gps', 20.01, 1, None)], anchor, .15) is None


def test_latest_slot_drops_backlog():
    slot = LatestSlot()
    slot.put(1)
    slot.put(2)
    assert slot.take() == 2
    assert slot.take() is None


def test_live_sensor_frame_uses_only_causal_samples_and_never_reuses_anchor():
    import time
    now = time.monotonic()
    records = {}
    for name in ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'lidar', 'gps', 'imu', 'vehicle_status'):
        value = dict(speed=3., competition_variant=True) if name == 'vehicle_status' else name
        source = None if name in ('gps', 'lidar') else 100.
        records[name] = [ReceivedSample(value, now-.01, 1, source)]
    # A later speed packet must not replace the causally paired speed.
    records['vehicle_status'].append(ReceivedSample(dict(speed=99., competition_variant=True), now, 2, 100.1))
    sensors = SensorSynchronizer(SimpleNamespace(recent=lambda: records),
        SimpleNamespace(project=lambda gps, imu: dict(x=1., y=2., yaw_rad=0.)))
    result = sensors.next_frame()
    assert result.speed_mps == 3.
    assert result.source_time == 100.
    assert set(result.images) == {'rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear'}
    assert sensors.next_frame() is None


def test_missing_causal_speed_does_not_fall_back_to_zero_or_future():
    import time
    now = time.monotonic()
    records = {name:[ReceivedSample(name, now-.01, 1, 100.)] for name in
        ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'lidar', 'gps', 'imu')}
    records['vehicle_status'] = [ReceivedSample(dict(speed=99., competition_variant=True), now, 1, 100.1)]
    sensors = SensorSynchronizer(SimpleNamespace(recent=lambda:records), None)
    assert sensors.next_frame() is None
    assert sensors.reason == 'waiting_for_causal_vehicle_status'


def test_stop_for_stale_prediction_and_changed_instruction():
    import threading
    runner = object.__new__(LiveRunner)
    runner.controller = WaypointController(positive_steer='left')
    runner.stop_event = threading.Event()
    runner.error = None
    runner.options = dict(max_prediction_age_s=.75)
    runner.history = InstructionHistory()
    runner.history.append(frame(0))
    runner.sensors = SimpleNamespace(current_state=lambda: (dict(pose=frame(0).pose, speed_mps=1.), None))
    prediction = dict(generation=0, anchor=frame(0), world_points=np.zeros((5, 2)))
    assert runner._command(prediction, 1.).reason == 'stale_prediction'
    runner.history.reset('new_instruction')
    assert runner._command(prediction, .1).reason == 'instruction_history_changed'


def test_control_advances_by_simulator_time_not_gpu_wall_time():
    import threading
    runner = object.__new__(LiveRunner)
    runner.controller = WaypointController(positive_steer='left')
    runner.stop_event = threading.Event()
    runner.error = None
    runner.options = dict(max_prediction_age_s=.75)
    runner.history = InstructionHistory()
    runner.history.append(frame(0))
    state = dict(pose=dict(x=2., y=0., yaw_rad=0.), speed_mps=4., source_time=.5)
    runner.sensors = SimpleNamespace(current_state=lambda: (state, None))
    points = np.column_stack((np.zeros(5), -np.arange(1, 6)*2.))
    prediction = dict(generation=0, anchor=frame(0), world_points=to_world(points, frame(0).pose))
    result = runner._command(prediction, .1)
    assert result.desired_speed_mps == pytest.approx(4.)
    state['source_time'] = 1.6
    assert runner._command(prediction, .1).reason == 'prediction_sim_time_expired'


def test_official_zone_boundaries_match_existing_labeler():
    import yaml
    from MoraiDrive.data_collection.situation_labeling.build_labels import _zone_intervals
    path = ROOT / 'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt'
    definition = yaml.safe_load((ROOT/'MoraiDrive/scenario_runner/config/full_loop_links.yaml').read_text(encoding='utf-8'))['full_loop']
    points = np.loadtxt(path)
    zones = RouteZones(points, definition)
    route = dict(definition, nav_waypoints=points.tolist())
    ids = dict(static_obstacle=1, dynamic_obstacle=2, roundabout=3, highway=4, tunnel=5)
    for name, start, end in _zone_intervals(route):
        assert zones.at(zones.arc[(start + end)//2]) == ids[name]
        assert zones.at(zones.arc[(start + end)//2] + zones.arc[-1]) == ids[name]
    assert zones.at(0.) == 0


def test_check_cli_never_opens_udp_or_loads_weights(monkeypatch, capsys):
    import socket
    import yaml
    from AIM2026.Inference import main as entry
    config = yaml.safe_load((ROOT / 'AIM2026/config.yaml').read_text(encoding='utf-8'))
    config['Ego'].update(ip='127.0.0.1', vehicle_status_port=16001, cmd_port=16002)
    for port, name in enumerate(('GPS', 'IMU', 'LiDAR'), 16003):
        config[name]['dest_port'] = port
    monkeypatch.setattr(entry, 'read_config', lambda _:config)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--check', '--control'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Configuration check attempted network I/O')
    monkeypatch.setattr(socket, 'socket', forbidden)
    entry.main()
    assert 'Configuration OK' in capsys.readouterr().out
