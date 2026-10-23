"""Actual shared UDP -> DR -> existing Expert wire, using loopback only."""
import importlib.util
import json
import math
from pathlib import Path
import socket
import time

import yaml

from AIM2026.UDP.Manager import UDPManager
from AIM2026.UDP.IMU import ImuPacket
from AIM2026.UDP.VehicleStatus import CompetitionVehicleStatusPacket
from AIM2026.UDP.Pose import decode_pose


ROOT = Path(__file__).resolve().parents[2]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _gps_sentence(latitude, longitude):
    def coordinate(value, width):
        degree = int(abs(value))
        return f'{degree:0{width}d}{(abs(value)-degree)*60:013.10f}'
    body = ('GPGGA,120000,' + coordinate(latitude, 2) + ',N,' + coordinate(longitude, 3)
            + ',E,1,10,1.0,0.0,M,0.0,M,,')
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return f'${body}*{checksum:02X}\r\n'.encode()


def test_real_receivers_blackout_recovery_and_common_expert_decoder():
    config = yaml.safe_load((ROOT / 'AIM2026/config.yaml').read_text())
    ports = set()
    while len(ports) < 3:
        ports.add(_free_port())
    gps_port, imu_port, status_port = ports
    config['GPS'].update(dest_port=gps_port, bind='127.0.0.1')
    config['IMU'].update(dest_port=imu_port, bind='127.0.0.1')
    config['Ego'].update(vehicle_status_port=status_port, bind='127.0.0.1', cmd_port=None)
    config['Localization']['record_dir'] = None
    consumer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    consumer.bind(('127.0.0.1', 0))
    consumer.settimeout(.03)
    config['Detection'].update(ip='127.0.0.1', port=consumer.getsockname()[1], checkpoint_sha256='loopback-test')
    manager = UDPManager(config)
    manager.sensor_config = {key: value for key, value in manager.sensor_config.items()
                             if key in ('gps', 'imu', 'vehicle_status')}
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        manager.start()
        service = manager.localization
        lon, lat = service.hold.geometry.inverse.transform(*service.basis['origin_xyz'][:2])
        gps = _gps_sentence(lat, lon)
        imu, status = ImuPacket(), CompetitionVehicleStatusPacket()
        imu.header, imu.tail, imu.ori_w = b'#IMUData$', b'\r\n', 1.
        status.header, status.tail = b'#MoraiInfo$', b'\r\n'
        status.signed_vel, status.gear = 36., 4
        tick = 0
        def wait_mode(mode, send_gps):
            nonlocal tick
            deadline = time.monotonic() + 2.
            while time.monotonic() < deadline:
                tick += 1
                ns = 100_000_000_000 + tick * 20_000_000
                imu.sec = status.sec = ns // 1_000_000_000
                imu.nsec = status.nsec = ns % 1_000_000_000
                sender.sendto(bytes(imu), ('127.0.0.1', imu_port))
                sender.sendto(bytes(status), ('127.0.0.1', status_port))
                if send_gps:
                    sender.sendto(gps, ('127.0.0.1', gps_port))
                try:
                    packet = json.loads(consumer.recvfrom(8001)[0])
                except socket.timeout:
                    continue
                if packet.get('ready') and packet.get('selected_samples', {}).get('gps', {}).get('position_mode') == mode:
                    return packet
                time.sleep(.01)
            raise AssertionError('control pose did not reach mode ' + mode + ': ' + str(service.diagnostics()))
        normal = wait_mode('gps', True)
        held = wait_mode('dead_reckoning', False)
        assert normal['ego_position_source'] == 'GPS'
        assert held['ego_position_source'] == 'GPS_dead_reckoning'
        assert math.isfinite(held['ego_pose']['x'])
        assert held['ego_pose']['speed_mps'] == 10.
        assert held['selected_samples']['gps']['last_gps_received_ns'] < held['selected_samples']['gps']['received_ns']
        decode_pose(json.dumps(held).encode())
        # Test the existing consumer directly; it is not copied into DR runtime.
        spec = importlib.util.spec_from_file_location('existing_expert_pose_wire', ROOT / 'VLDrive/utils/common/ve_control_pose.py')
        decoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(decoder)
        decoded = decoder.decode_control_pose(held, held['generation'], 'loopback-test',
            time.monotonic_ns(), position_sources=('GPS', 'GPS_dead_reckoning'))
        assert decoded['ego_position_source'] == 'GPS_dead_reckoning'
        assert wait_mode('gps', True)['ready'] is True
        assert len(manager.receivers) == 3
        assert manager.cmd is None
    finally:
        manager.close()
        sender.close()
        consumer.close()
