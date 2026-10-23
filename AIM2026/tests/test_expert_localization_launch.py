"""Common Expert selects one AIM2026 publisher without changing control policy."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from AIM2026.Inference.settings import read_config
from MoraiDrive.expert.tools import run_ve_gps_expert as launch

ROOT = Path(__file__).resolve().parents[2]


def args(**changes):
    values = dict(aim2026_config=ROOT/'AIM2026/config.yaml', simulate_gps_blackout=False,
                  gps_latency_s=0., gps_blackout_hold=True, control_source_port=7600)
    values.update(changes)
    return SimpleNamespace(**values)


def test_private_config_has_one_sensor_owner_no_child_control_and_unique_recording(tmp_path):
    original = read_config(ROOT/'AIM2026/config.yaml')
    original['Ego'].update(cmd_port=7601, cmd_source_port=7600)
    original['Localization']['record_dir'] = '/old/shared/path'
    saved = copy.deepcopy(original)
    result = launch.aim2026_publisher_config(args(), original, tmp_path)
    assert original == saved
    assert result['Ego']['cmd_port'] is result['Ego']['cmd_source_port'] is None
    assert result['Localization']['record_dir'] == str(tmp_path/'localization')
    assert result['Inference']['status_file'] == str(tmp_path/'aim2026_status.json')
    assert result['GPS'] == original['GPS']


@pytest.mark.parametrize('change,message', [
    ({'gps_blackout_hold':False}, '--gps-blackout-hold'),
    ({'simulate_gps_blackout':True}, 'test_blackout.enabled'),
    ({'gps_latency_s':.02}, '--gps-latency-s'),
    ({'control_source_port':10506}, 'conflicts'),
])
def test_config_rejects_silent_option_mismatch_or_port_collision(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        launch.aim2026_publisher_config(args(**change), read_config(ROOT/'AIM2026/config.yaml'), tmp_path)


def test_config_requires_expert_loopback_port(tmp_path):
    config = read_config(ROOT/'AIM2026/config.yaml')
    config['Detection']['port'] = 12051
    with pytest.raises(ValueError, match='12050'):
        launch.aim2026_publisher_config(args(), config, tmp_path)


def test_publisher_commands_select_exactly_one_backend_and_share_generation(tmp_path):
    new = launch.publisher_command(args(), tmp_path, '/python', 'shared-id')
    assert new == ['/python', '-u', '-B', str(ROOT/'AIM2026/Inference/main.py'),
                   '--config', str(tmp_path/'aim2026_publisher.yaml'), '--generation', 'shared-id', '--seconds', '0']
    old = launch.publisher_command(args(aim2026_config=None), tmp_path, '/python', 'legacy-id')
    assert old == ['/python', '-u', '-B', str(ROOT/'VLDrive/inference/ve_objects_udp.py'),
                   '--config', str(tmp_path/'publisher_config.json'), '--generation', 'legacy-id', '--seconds', '0',
                   '--audit-dir', str(tmp_path)]


def test_full_expert_check_uses_real_route_without_torch_socket_or_child(tmp_path):
    # A fresh interpreter catches accidental eager imports hidden by pytest's imports.
    # Only checkpoint presence is faked: this is not a model loading/hash test.
    script = '''
import sys
from pathlib import Path
from unittest.mock import patch
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('configuration check imported torch')
sys.meta_path.insert(0, NoTorch())
from AIM2026.Inference.vision import checkpoint
from MoraiDrive.expert.tools import run_ve_gps_expert as launch
base = Path(sys.argv[1]); weights=base/'weights'; weights.write_text('presence fixture')
digest=checkpoint.CHECKPOINTS['P1']['sha256']
with patch.object(checkpoint, 'resolve', return_value=(weights,digest,weights)), patch('socket.socket', side_effect=AssertionError('opened socket')), patch('subprocess.Popen', side_effect=AssertionError('spawned publisher')):
    assert launch.main(['--aim2026-config','AIM2026/config.yaml','--gps-blackout-hold','--check','--output',str(base/'out')]) == 0
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path)], cwd=str(ROOT),
                            capture_output=True, text=True, check=True)
    assert 'CHECKED' in result.stdout
    plan = json.loads((tmp_path/'out/plan.json').read_text())
    assert plan['publisher_backend'] == 'AIM2026'
    private = yaml.safe_load((tmp_path/'out/aim2026_publisher.yaml').read_text())
    assert private['Localization']['record_dir'] == str(tmp_path/'out/localization')
    assert private['Ego']['cmd_port'] is None


def test_inference_generation_cli_is_forwarded_to_manager_without_start(monkeypatch, tmp_path):
    entry = importlib.import_module('AIM2026.Inference.main')
    udp = importlib.import_module('AIM2026.UDP')
    weights = tmp_path/'weights'; weights.write_text('presence fixture')
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(udp, 'UDPManager', factory)
    monkeypatch.setattr('AIM2026.Inference.vision.checkpoint.resolve', lambda _: (weights, 'sha', weights))
    monkeypatch.setattr(sys, 'argv', ['main.py', '--config', str(ROOT/'AIM2026/config.yaml'),
                                    '--generation', 'expert-run-identity', '--check'])
    assert entry.main() == 0
    assert factory.call_args.args[0]['Detection']['generation'] == 'expert-run-identity'
    factory.return_value.start.assert_not_called()


def test_manager_gives_pose_and_detection_same_generation(monkeypatch):
    module = importlib.import_module('AIM2026.UDP.Manager')
    config = read_config(ROOT/'AIM2026/config.yaml')
    config['Localization']['enabled'] = False
    config['Detection']['generation'] = 'shared-generation'
    for name in ('CameraReceiver', 'GpsReceiver', 'ImuReceiver', 'LidarReceiver', 'VehicleStatusReceiver'):
        monkeypatch.setattr(module, name, Mock())
    detection = Mock(return_value=SimpleNamespace(generation='shared-generation', close=lambda:None))
    pose = Mock()
    monkeypatch.setattr(module, 'DetectionSender', detection)
    monkeypatch.setattr(module, 'PoseSender', pose)
    manager = module.UDPManager(config)
    monkeypatch.setattr(manager, '_probe_ports', lambda:None)
    manager.start()
    try:
        assert detection.call_args.kwargs['generation'] == 'shared-generation'
        assert 'shared-generation' in pose.call_args.args
    finally:
        manager.close()
