import json
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


def test_local_steering_profile_reuses_existing_sign_adapter():
    import yaml
    from AIM2026.Expert.adapter import configure_udp_steering
    root = Path(__file__).resolve().parents[2]
    settings = yaml.safe_load((root/'AIM2026/config_local.yaml').read_text())
    assert settings['Control']['positive_steer'] == 'right'
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
