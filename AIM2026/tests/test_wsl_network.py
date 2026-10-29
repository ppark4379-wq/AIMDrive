"""wsl_network: per-PC UDP settings for WSL runs (pure file logic; no WSL, no MORAI)."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from AIM2026.Expert import wsl_network

ROOT = Path(__file__).resolve().parents[2]
WSL, WINDOWS = '172.30.144.83', '172.30.144.1'


def sensor_file(directory, *, host=WINDOWS, destination=WSL, shared_port=None):
    """A MORAI sensor file for the competition config's sensors (cameras 7501-7504, GPS 7506, IMU 7507, LiDAR 2369)."""
    ports = {'cameraList': (7501, 7502, 7503, 7504), 'GPSList': (7506,), 'IMUList': (7507,), 'Lidar3DList': (2369,)}
    destinations = {7501: 10501, 7502: 10502, 7503: 10503, 7504: 10504, 7506: 10506, 7507: 10507, 2369: 2368}
    document = {}
    for kind, hosts in ports.items():
        document[kind] = [{'cfg': {'udpConfig': dict(hostIP=host, hostPort=port, destinationIP=destination,
                                                     destinationPort=shared_port or destinations[port])}}
                          for port in hosts]
    path = Path(directory) / 'aim_sensor.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


class PathTests(unittest.TestCase):
    def test_windows_paths_become_wsl_paths(self):
        self.assertEqual(wsl_network.to_wsl_path('C:\\Morai\\Save File\\aim_sensor.json'),
                         Path('/mnt/c/Morai/Save File/aim_sensor.json'))
        self.assertEqual(wsl_network.to_wsl_path('"D:/x/y.json"'), Path('/mnt/d/x/y.json'))
        self.assertEqual(wsl_network.to_wsl_path('/home/me/aim_sensor.json'), Path('/home/me/aim_sensor.json'))

    def test_the_sensor_file_comes_from_the_option_then_the_environment_then_the_note(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            root = Path(folder)
            (root / 'AIM2026/outputs/network').mkdir(parents=True)
            (root / wsl_network.SENSOR_PATH_NOTE).write_text('C:\\from\\note.json\n', encoding='utf-8')
            args = type('Args', (), {'morai_sensor': None})()
            os.environ.pop('MORAI_SENSOR_FILE', None)
            self.assertEqual(wsl_network.saved_sensor_path(args, root), Path('/mnt/c/from/note.json'))
            os.environ['MORAI_SENSOR_FILE'] = '/env/sensor.json'
            self.assertEqual(wsl_network.saved_sensor_path(args, root), Path('/env/sensor.json'))
            args.morai_sensor = '/option/sensor.json'
            self.assertEqual(wsl_network.saved_sensor_path(args, root), Path('/option/sensor.json'))

    def test_nothing_configured_means_no_saved_sensor_file(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            os.environ.pop('MORAI_SENSOR_FILE', None)
            self.assertIsNone(wsl_network.saved_sensor_path(type('Args', (), {'morai_sensor': None})(), folder))


class PrepareTests(unittest.TestCase):
    def prepare(self, directory, **kwargs):
        return wsl_network.prepare(ROOT / 'AIM2026/config_competition.yaml', ROOT / 'AIM2026/aim_sensor.json', directory,
                                   wsl_ip=WSL, windows_ip=WINDOWS, competition_build=True, **kwargs)

    def test_the_generated_sensor_set_only_changes_the_addresses(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, sensor_path = self.prepare(folder)
            generated = json.loads(sensor_path.read_text(encoding='utf-8'))
            original = json.loads((ROOT / 'AIM2026/aim_sensor.json').read_text(encoding='utf-8-sig'))

            def strip(value):
                if isinstance(value, dict):
                    return {k: strip(v) for k, v in value.items() if k not in ('hostIP', 'destinationIP')}
                if isinstance(value, list):
                    return [strip(v) for v in value]
                return value
            self.assertEqual(strip(generated), strip(original))
            udp = generated['GPSList'][0]['gc']['udpConfig']
            self.assertEqual((udp['hostIP'], udp['destinationIP'], udp['destinationPort']), (WINDOWS, WSL, 10506))
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            self.assertEqual(config['Ego']['ip'], WINDOWS)
            self.assertEqual(config['Ego']['local_status_bytes'], 229)
            self.assertEqual(config['Control']['positive_steer'], 'left')
            self.assertIsNone(config['GPS']['ip'])

    def test_status_bytes_181_removes_the_full_status_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, _ = self.prepare(folder, status_bytes=181)
            self.assertNotIn('local_status_bytes', yaml.safe_load(config_path.read_text(encoding='utf-8'))['Ego'])

    def test_competition_addresses_keep_the_committed_files_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, sensor_path = self.prepare(folder, keep_addresses=True)
            self.assertEqual(sensor_path.read_bytes(), (ROOT / 'AIM2026/aim_sensor.json').read_bytes())
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            self.assertEqual(config['Ego']['ip'], '192.168.0.1')
            self.assertEqual(config['GPS']['ip'], '192.168.0.1')

    def test_receiver_ports_follow_the_saved_sensor_file(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = sensor_file(folder, shared_port=9090)
            config_path, used = self.prepare(Path(folder) / 'out', saved_sensor=saved)
            self.assertEqual(used, saved)                                   # no sensor set is written, the file is used
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            self.assertEqual([config['GPS']['dest_port'], config['IMU']['dest_port'],
                              *[c['dest_port'] for c in config['Camera'].values()]], [9090] * 6)

    def test_a_saved_sensor_file_with_one_port_per_sensor_is_followed_too(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = sensor_file(folder)
            config_path, _ = self.prepare(Path(folder) / 'out', saved_sensor=saved)
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            self.assertEqual((config['GPS']['dest_port'], config['IMU']['dest_port'], config['LiDAR']['dest_port']),
                             (10506, 10507, 2368))

    def test_a_saved_sensor_file_with_other_addresses_is_refused_with_a_hint(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = sensor_file(folder, host='127.0.0.1', destination='127.0.0.1')
            with self.assertRaisesRegex(ValueError, r'127\.0\.0\.1 -> 127\.0\.0\.1.*172\.30\.144\.1 -> 172\.30\.144\.83'):
                self.prepare(Path(folder) / 'out', saved_sensor=saved)

    def test_a_saved_sensor_file_missing_a_sensor_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = sensor_file(folder)
            document = json.loads(saved.read_text(encoding='utf-8'))
            document['cameraList'] = document['cameraList'][:3]
            saved.write_text(json.dumps(document), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'host port 7504'):
                self.prepare(Path(folder) / 'out', saved_sensor=saved)


if __name__ == '__main__':
    unittest.main()
