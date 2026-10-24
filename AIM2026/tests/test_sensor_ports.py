"""Authoritative sensor export + real loopback shared-port routing (no control)."""
import json
from pathlib import Path
import socket
import struct
import time

import pytest
import yaml

from AIM2026.UDP.Manager import UDPManager
from AIM2026.UDP.IMU import ImuPacket

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('filename', ['config.yaml', 'config_local.yaml'])
def test_configs_match_export(filename):
    export = json.loads((ROOT/'sensor_set_competition.json').read_text(encoding='utf-8-sig'))
    cfg = yaml.safe_load((ROOT/filename).read_text(encoding='utf8'))
    rows = [(cfg['Camera'][key], sensor, 'cc') for key, sensor in zip(
        ('Cam_Front', 'Cam_Left', 'Cam_Right', 'Cam_Rear'), export['cameraList'])]
    rows += [(cfg[key], export[group][0], kind) for key, group, kind in
             [('GPS', 'GPSList', 'gc'), ('IMU', 'IMUList', 'ic'), ('LiDAR', 'Lidar3DList', 'lc')]]
    for config, sensor, kind in rows:
        net = sensor[kind]['udpConfig']
        assert (config['ip'], config['host_port'], config['bind'], config['dest_port']) == (
            net['hostIP'], net['hostPort'], net['destinationIP'], net['destinationPort'])
        assert config['pos'] == [float(sensor['pos'][k]) for k in ('x', 'y', 'z')]
        assert config['rot_deg'] == [float(sensor['rot'][k]) for k in ('roll', 'pitch', 'yaw')]
        if kind == 'cc':
            assert (config['width'], config['height'], config['fov']) == (
                sensor[kind]['cameraResWidth'], sensor[kind]['cameraResHeight'], sensor[kind]['cameraFOV'])


def udp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    return sock


def test_shared_destination_isolates_interleaved_camera_gps_and_imu():
    cfg = yaml.safe_load((ROOT/'config_local.yaml').read_text(encoding='utf8'))
    cfg['Ego']['cmd_port'] = cfg['Ego']['cmd_source_port'] = None
    cfg['Detection'] = {}
    cfg['Localization']['enabled'] = False
    senders = [udp_socket() for _ in range(7)]
    reservations = [udp_socket() for _ in range(3)]
    ports = [s.getsockname()[1] for s in reservations]
    for sock in reservations:
        sock.close()
    rows = list(cfg['Camera'].values()) + [cfg['GPS'], cfg['IMU']]
    for conf, sender in zip(rows, senders):
        conf.update(dest_port=ports[0], host_port=sender.getsockname()[1])
    cfg['LiDAR']['dest_port'] = ports[1]
    cfg['Ego']['vehicle_status_port'] = ports[2]
    cfg['UDP'] = {'isolate_heavy_sensors': False}  # this test covers the in-process shared-port demultiplexer
    manager = UDPManager(cfg)
    names = ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear')
    images = [b'\xff\xd8'+bytes([i])*30+b'\xff\xd9' for i in range(4)]
    target = ('127.0.0.1', ports[0])
    try:
        manager.start()
        assert len(manager.shared_ports) == 3  # Six sensors share a single socket.
        for index in range(2):
            for sender, image in zip(senders[:4], images):
                body = image[:17] if index == 0 else image[17:]
                sender.sendto(struct.pack('<3s4i', b'MOR', 1, 1, index, len(body))+
                              body+(b'EI' if index else b'\r\n'), target)
        senders[4].sendto(b'$GPGGA,123519,3723.000,N,12646.000,E,1,08,0.9,1.0,M,0,M,,\r\n', target)
        imu = ImuPacket()
        imu.header, imu.sec, imu.ori_w = b'#IMUData$', 1, 1.
        senders[5].sendto(bytes(imu), target)
        # A sender absent from the export cannot inject a camera frame.
        senders[6].sendto(struct.pack('<3s4i', b'MOR', 2, 1, 0, len(images[0]))+images[0]+b'EI', target)
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            if all(manager.receivers[name].samples == 1 for name in (*names, 'gps', 'imu')):
                break
            time.sleep(.01)
        for name, image in zip(names, images):
            assert manager.receivers[name].snapshot()[0] == image
            assert manager.receivers[name].samples == 1
            assert manager.receivers[name].packets == 2
        assert manager.receivers['gps'].snapshot()[0]['latitude'] == pytest.approx(37+23/60)
        assert manager.receivers['imu'].snapshot()[0]['yaw'] == 0.
    finally:
        manager.close()
        for sender in senders:
            sender.close()
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(('127.0.0.1', port))


def test_shared_destination_with_child_process_keeps_every_sensor_working():
    """Cameras demultiplexed in a child, GPS/IMU relayed to their in-process receivers (MORAI default layout)."""
    cfg = yaml.safe_load((ROOT/'config_local.yaml').read_text(encoding='utf8'))
    cfg['Ego']['cmd_port'] = cfg['Ego']['cmd_source_port'] = None
    cfg['Detection'] = {}
    cfg['Localization']['enabled'] = False
    senders = [udp_socket() for _ in range(7)]
    reservations = [udp_socket() for _ in range(3)]
    ports = [s.getsockname()[1] for s in reservations]
    for sock in reservations:
        sock.close()
    rows = list(cfg['Camera'].values()) + [cfg['GPS'], cfg['IMU']]
    for conf, sender in zip(rows, senders):
        conf.update(dest_port=ports[0], host_port=sender.getsockname()[1])
    cfg['LiDAR']['dest_port'] = ports[1]
    cfg['Ego']['vehicle_status_port'] = ports[2]
    manager = UDPManager(cfg)
    names = ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear')
    images = [b'\xff\xd8'+bytes([i])*30+b'\xff\xd9' for i in range(4)]
    target = ('127.0.0.1', ports[0])
    try:
        manager.start()
        assert len(manager.process_groups) == 2  # camera group (+ GPS/IMU relay) and LiDAR
        assert {port for _bind, port in manager.shared_ports} == {ports[2]}  # only the vehicle-status socket stays
        assert all(manager.receivers[n].diagnostics().get('process_isolated') for n in (*names, 'lidar'))
        time.sleep(2.5)  # spawned children import the packages before they read the port
        for index in range(2):
            for sender, image in zip(senders[:4], images):
                body = image[:17] if index == 0 else image[17:]
                sender.sendto(struct.pack('<3s4i', b'MOR', 1, 1, index, len(body))+
                              body+(b'EI' if index else b'\r\n'), target)
        senders[4].sendto(b'$GPGGA,123519,3723.000,N,12646.000,E,1,08,0.9,1.0,M,0,M,,\r\n', target)
        imu = ImuPacket()
        imu.header, imu.sec, imu.ori_w = b'#IMUData$', 1, 1.
        senders[5].sendto(bytes(imu), target)
        senders[6].sendto(struct.pack('<3s4i', b'MOR', 2, 1, 0, len(images[0]))+images[0]+b'EI', target)
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if (all(manager.receivers[name].snapshot()[2] == 1 for name in names)
                    and manager.receivers['gps'].samples == 1 and manager.receivers['imu'].samples == 1):
                break
            time.sleep(.02)
        for name, image in zip(names, images):
            assert manager.receivers[name].snapshot()[0] == image
        assert manager.receivers['gps'].snapshot()[0]['latitude'] == pytest.approx(37+23/60)
        assert manager.receivers['imu'].snapshot()[0]['yaw'] == 0.
    finally:
        manager.close()
        for sender in senders:
            sender.close()


def test_shared_destination_rejects_ambiguous_sender_ports():
    cfg = yaml.safe_load((ROOT/'config_local.yaml').read_text(encoding='utf8'))
    cfg['Detection'] = {}
    cfg['Camera']['Cam_Left']['host_port'] = cfg['Camera']['Cam_Front']['host_port']
    with pytest.raises(ValueError, match='distinct host_port'):
        UDPManager(cfg)._validate_ports()
