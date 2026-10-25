import json
import copy
from pathlib import Path
import socket
import struct
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AIM2026.Expert.runtime import Inputs, wire_command
from AIM2026.Expert.adapter import planner_inputs
from AIM2026.UDP.Detection import SCHEMA
from AIM2026.UDP.Pose import SCHEMA as POSE_SCHEMA
from AIM2026.UDP.VehicleStatus import VehicleStatusReceiver
from inference.ve_object_tracking import VeObjectTracker


def packets():
    now = time.monotonic_ns()
    received = now-10_000_000
    shared = dict(generation='test', checkpoint_sha256='sha', sequence=1,
                  ready=True, sent_ns=now, sensor_received_ns=received,
                  output_frame='projected_grid', ego_position_source='GPS',
                  ego_status_source='UDP_status_speed_IMU_heading',
                  ego_pose=dict(x=10.,y=20.,yaw_raw_deg=0.,yaw_grid_rad=0.,speed_mps=3.))
    detection = dict(shared, schema=SCHEMA, scene_assumption='unknown', boxes=[dict(
        center_world=[15.,20.],heading_deg=0.,width_m=2.,length_m=4.,speed_mps=0.)])
    pose = dict(shared, schema=POSE_SCHEMA, selected_samples=dict(
        gps=dict(received_ns=received), vehicle_status=dict(received_ns=received,imu_received_ns=received)))
    pose['ego_pose'] = dict(shared['ego_pose'],x=12.)
    return detection,pose


def test_loopback_preserves_object_frame_and_expires_inputs():
    bridge = Inputs('test','sha',VeObjectTracker())
    detection,pose = packets()
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver, socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
        receiver.bind(('127.0.0.1',0)); receiver.settimeout(1.)
        for row in (detection,pose):
            sender.sendto(json.dumps(row).encode(),receiver.getsockname())
            bridge.accept(receiver.recv(65536))
    assert bridge.ready()
    vehicle,objects = planner_inputs(bridge.packet,bridge.pose)
    assert vehicle.position.x == 12.
    assert objects[0].position.x == 15.  # Never shift old boxes with newer ego pose.
    assert not bridge.ready(pose['sensor_received_ns']+151_000_000)
    assert bridge.accept(json.dumps(pose).encode()) is None  # Replay ignored.
    bad = dict(pose,generation='foreign',sequence=20,ready=False)
    assert bridge.accept(json.dumps(bad).encode()) is None
    assert bridge.pose is not None
    bad.update(generation='test')
    assert bridge.accept(json.dumps(bad).encode()) == 'invalid_pose'
    assert not bridge.ready()


def test_control_units_and_fail_closed():
    assert wire_command(None,.5).brake == 1.
    command = SimpleNamespace(accel=.2,brake=0.,front_steer=.25,rear_steer=0.)
    encoded = wire_command(command,.5)
    assert encoded.steer == .5
    assert encoded.pack()[30:33] == bytes((2,4,1))
    command.front_steer = float('nan')
    with pytest.raises(ValueError): wire_command(command,.5)


def test_new_rear_vehicle_keeps_model_speed_through_udp_tracker_and_adapter():
    bridge = Inputs('test', 'sha', VeObjectTracker())
    detection, pose = packets()
    detection['boxes'][0].update(center_world=[-10., 20.], speed_mps=9.56)
    bridge.accept(json.dumps(detection).encode())
    bridge.accept(json.dumps(pose).encode())
    assert bridge.ready()
    assert bridge.packet['boxes'][0]['motion_state'] == 'uncertain'
    _, objects = planner_inputs(bridge.packet, bridge.pose)
    assert objects[0].velocity == 9.56
    assert not objects[0].is_static
    assert objects[0].motion_uncertain


def aligned_packets():
    detection, pose = packets()
    latest = pose['selected_samples']['gps']['received_ns']
    detection['sensor_received_ns'] = latest - 300_000_000
    detection['object_received_ns'] = latest - 200_000_000
    detection['boxes'][0].update(speed_mps=10., heading_deg=90.)
    bridge = Inputs('test', 'sha', VeObjectTracker())
    bridge.accept(json.dumps(detection).encode())
    bridge.accept(json.dumps(pose).encode())
    assert bridge.ready()
    return bridge


def test_objects_advance_to_position_time_once_not_oldest_sensor_or_send_time():
    bridge = aligned_packets()
    original = copy.deepcopy(bridge.packet)
    vehicle, objects = planner_inputs(bridge.packet, bridge.pose)
    assert (vehicle.position.x, vehicle.position.y) == (12., 20.)
    # 10 m/s north for 0.2 s, not 0.3 s from the oldest sensor receipt.
    assert objects[0].position.x == pytest.approx(15.)
    assert objects[0].position.y == pytest.approx(22.)
    assert objects[0].position_projection_s == pytest.approx(.2)
    assert bridge.packet == original
    _, repeated = planner_inputs(bridge.packet, bridge.pose)
    assert repeated[0].position.y == pytest.approx(22.)  # No cumulative drift.
    later = copy.deepcopy(bridge.pose)
    later['selected_samples']['gps']['received_ns'] += 50_000_000
    _, advanced = planner_inputs(bridge.packet, later)
    assert advanced[0].position.y == pytest.approx(22.5)


@pytest.mark.parametrize('kind', ['stationary', 'unknown', 'legacy', 'no_pose', 'older_pose'])
def test_motion_alignment_never_invents_static_unknown_or_unstamped_motion(kind):
    bridge = aligned_packets()
    packet, pose = bridge.packet, bridge.pose
    if kind == 'stationary':
        packet['boxes'][0]['motion_state'] = 'stationary'
    elif kind == 'unknown':
        packet['boxes'][0]['planning_speed_mps'] = None
    elif kind == 'legacy':
        packet.pop('object_received_ns')
    elif kind == 'no_pose':
        pose = None
    else:
        pose['selected_samples']['gps']['received_ns'] = packet['object_received_ns'] - 1
    _, objects = planner_inputs(packet, pose)
    assert (objects[0].position.x, objects[0].position.y) == (15., 20.)
    assert objects[0].position_projection_s == 0.


def test_motion_alignment_does_not_extend_input_expiry_or_hide_bad_frame_time():
    bridge = aligned_packets()
    assert not bridge.ready(bridge.packet['sensor_received_ns'] + 500_000_001)
    detection, _ = packets()
    detection['object_received_ns'] = detection['sent_ns'] + 1
    bad = Inputs('test', 'sha', VeObjectTracker())
    assert bad.accept(json.dumps(detection).encode()) == 'invalid_detection'
    assert bad.packet is None


def test_rviz_snapshot_receives_the_same_aligned_objects_as_the_planner(tmp_path):
    from AIM2026.Visualization.snapshot import write_snapshot, decode
    bridge = aligned_packets()
    vehicle, objects = planner_inputs(bridge.packet, bridge.pose)
    target = tmp_path / 'visualization.json'
    write_snapshot(target, SimpleNamespace(candidates=[], path=[]), vehicle, objects, [])
    state = decode(json.loads(target.read_text(encoding='utf-8')))
    assert state['objects'][0].position.y == objects[0].position.y == pytest.approx(22.)
    assert state['objects'][0].observation_y == 20.
    assert state['objects'][0].position_projection_s == pytest.approx(.2)
    assert state['vehicle'].position.x == vehicle.position.x


def test_missing_model_speed_remains_unknown_instead_of_becoming_zero():
    bridge = Inputs('test', 'sha', VeObjectTracker())
    detection, pose = packets()
    detection['boxes'][0]['speed_mps'] = None
    bridge.accept(json.dumps(detection).encode())
    bridge.accept(json.dumps(pose).encode())
    assert bridge.packet['boxes'][0]['planning_speed_mps'] is None
    assert bridge.ready()  # One object's missing speed must not trigger global full braking.
    _, objects = planner_inputs(bridge.packet, bridge.pose)
    assert objects[0].speed_unknown and objects[0].motion_uncertain
    assert objects[0].velocity > 0.  # Explicit reachable-speed bound, not a measured speed.


def test_confirmed_mover_does_not_become_a_stationary_hypothesis_on_one_weak_frame():
    tracker = VeObjectTracker(moving_carry_s=.75)
    detection, pose = packets()
    start = detection['sent_ns']
    for i in range(14):
        detection['sent_ns'] = start + i*100_000_000
        detection['boxes'][0].update(center_world=[i*.16, 20.], speed_mps=1.6)
        tracked = tracker.update(detection)
    for k, x in enumerate((2.98, 1.58)):
        detection['sent_ns'] = start + 1_400_000_000 + k*100_000_000
        detection['boxes'][0]['center_world'] = [x, 20.]
        tracked = tracker.update(detection)
    assert tracked['boxes'][0]['planning_speed_source'] == 'carry_over'
    _, objects = planner_inputs(tracked, pose)
    assert objects[0].velocity > 1.
    assert not objects[0].motion_uncertain


def test_local_steering_profile_reuses_existing_sign_adapter():
    from AIM2026.Expert.adapter import configure_udp_steering
    config = {'common': {}}
    configure_udp_steering(config, 'normal')
    # Positive Pure Pursuit requests a left turn; this simulator needs negative wire steering.
    angle = config['common']['realtime_steering_sign'] * .25
    command = SimpleNamespace(accel=0.,brake=0.,front_steer=angle,rear_steer=0.)
    assert wire_command(command,.5).steer == -.5
    configure_udp_steering(config, 'competition')
    assert config['common']['realtime_steering_sign'] == 1.


def test_explicit_local_245_layout_never_reads_gt_pose():
    receiver = object.__new__(VehicleStatusReceiver)
    receiver.local_status_bytes = 245
    packet = bytearray(245)
    packet[:11] = b'#MoraiInfo$'; packet[-2:] = b'\r\n'
    struct.pack_into('<i',packet,11,216)
    struct.pack_into('<ii',packet,27,100,0)
    packet[36] = 4
    struct.pack_into('<f',packet,37,36.)
    struct.pack_into('<15f',packet,77,*([float('nan')]*15))
    result = receiver._consume(packet)
    assert result['speed'] == 10.
    assert result['competition_variant'] is False
    assert result['local_status_allowed'] is True
    assert 'position' not in result and 'yaw' not in result
    packet[11] = 0
    assert receiver._consume(packet) is None
