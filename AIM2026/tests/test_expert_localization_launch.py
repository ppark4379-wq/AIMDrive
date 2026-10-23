"""Shared DR integration through the current refactored Expert entry point."""
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from AIM2026.Inference.settings import read_config
from AIM2026.Expert.runtime import publisher_config

ROOT = Path(__file__).resolve().parents[2]


def test_private_config_has_one_sensor_owner_no_child_control_and_unique_recording(tmp_path):
    original = read_config(ROOT/'AIM2026/config_local.yaml')
    original['Localization']['record_dir'] = '/old/shared/path'
    saved = copy.deepcopy(original)
    result = publisher_config(original, tmp_path, 'shared-generation')
    assert original == saved
    assert result['Ego']['cmd_port'] is result['Ego']['cmd_source_port'] is None
    assert result['Localization']['record_dir'] == str(tmp_path/'localization')
    assert result['Inference']['status_file'] == str(tmp_path/'inference_status.json')
    assert result['Detection']['generation'] == 'shared-generation'
    assert result['GPS'] == original['GPS']


def test_private_config_rejects_unsupported_dr_mount(tmp_path):
    import pytest
    config = read_config(ROOT/'AIM2026/config_local.yaml')
    config['GPS']['pos'][0] = 1.
    with pytest.raises(ValueError, match='zero-mount'):
        publisher_config(config, tmp_path, 'generation')


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
    monkeypatch.setattr(module, 'SharedPort', Mock())
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
