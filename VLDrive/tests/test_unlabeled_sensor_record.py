"""Basic-drive inputs keep real frames, empty Situation text and default Zone 0."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from data.src.loaders.situation_dataset import labeled_episodes
from data.src.processing import joint_preprocessing as joint
from data.src.processing.cache_velocity_tokens import compatible_export_sources
from .test_joint_preprocessing import raw_episode, fake_vision


@pytest.fixture
def basic(tmp_path, monkeypatch):
    monkeypatch.setattr(joint, 'ROOT', tmp_path)
    root = tmp_path / 'raw'
    ep = raw_episode(root)
    # A genuinely unlabeled source: delete only this fixture's synthetic labels.
    for path in (ep / 'situation_labels').iterdir():
        path.unlink()
    (ep / 'situation_labels').rmdir()
    joint.write(ep.parent / 'sensor_contract.json', {})
    joint.write(ep / 'raw/raw_manifest.json', {})
    assets = {}
    for role in ('camera_1', 'camera_2', 'camera_3', 'camera_4', 'lidar'):
        path = ep.parent / (role + '.bin')
        path.write_bytes(b'sensor')
        assets[role] = dict(relative_path=path.name, size_bytes=6, sha256=joint.digest(path))
    rows = [dict(index=i, lidar_source_time_ns=(i + 10) * 100_000_000,
                 sensor_bundle_valid=i != 4, assets=assets) for i in range(60)]
    joint.write(ep / 'admission_v2/frames.json', rows)
    joint.write(ep / 'admission_v2/COMPLETE.json', dict(files={
        'frames.json': joint.digest(ep / 'admission_v2/frames.json')}))
    spec = dict(root=str(root), namespace='basic_driving', frame_source='sensor_record')
    return ep, spec


def test_rows_keep_identity_empty_text_and_default_normal_zone(basic):
    ep, spec = basic
    episode = labeled_episodes([spec])[0]
    assert len(episode['rows']) == 60
    assert all(row['instruction'] is None for row in episode['rows'])
    assert all(row['zone_instruction'] == 0 and row['zone'] == 'normal' for row in episode['rows'])
    assert episode['rows'][4]['sensor_input_valid'] is False
    assert episode['rows'][4]['frame_id'] == 4  # Never join across missing frames.
    assert not (ep / 'situation_labels').exists()
    from VLDrive.data.src.loaders.situation_instructions import SituationInstructionIndex
    index = SituationInstructionIndex([spec], include_zone=True)
    refs = [dict(frame_index=r['frame_id'], sensor_source_ns=r['sensor_time_ns']) for r in episode['rows']]
    assert index.sample(episode['identity'], refs) == [None] * 60
    assert index.sample_zone_ids(episode['identity'], refs) == [0] * 60
    signals = joint.read(ep / 'quality/active_signals.json')
    # Changing Ego position must not replace the default with a geographic Zone.
    signals[26]['signal']['position'] = [99, 0, 0]
    joint.write(ep / 'quality/active_signals.json', signals)
    joint.write(ep / 'quality/QUALITY_COMPLETE.json', dict(files={
        'active_signals.json': joint.digest(ep / 'quality/active_signals.json')}))
    updated = labeled_episodes([spec])[0]
    assert all(row['zone_instruction'] == 0 for row in updated['rows'])
    assert updated['label_sha256'] != episode['label_sha256']


def test_unlabeled_source_joins_actual_stage2_loader(basic, tmp_path):
    ep, spec = basic
    from navigation.engine import DEFAULT_POLICY
    cfg = dict(model=dict(visual_sha256='a' * 64, path_navigation=True,
                         situation_instruction=True, zone_instruction=True),
        navigation_sampling={}, path_instruction={}, situation_instruction={}, zone_instruction={},
        preprocessing=dict(datasets=[spec], navigation=dict(policy=DEFAULT_POLICY)),
        sequence_length=11, frame_sampling_interval=2)
    snapshot = joint.frozen_config(cfg, 'v1')
    folder = tmp_path / 'data/processed/v1'
    joint.write(folder / 'processing_config.json', snapshot)
    joint.prepare(snapshot, folder, {})
    fake_vision(snapshot)
    result = joint.finish(snapshot, folder)
    assert result['sample_visual_shape'] == [11, 106, 256]
    assert result['situation_windows']['without_text'] == result['windows'] > 0


def test_source_rejects_seal_damage_and_unsafe_assets(basic):
    ep, spec = basic
    path = ep / 'admission_v2/frames.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='Changed sealed'):
        labeled_episodes([spec])
    rows = joint.read(path)
    rows[0]['assets']['lidar']['relative_path'] = '../outside.bin'
    joint.write(path, rows)
    joint.write(ep / 'admission_v2/COMPLETE.json', dict(files={'frames.json': joint.digest(path)}))
    with pytest.raises(ValueError, match='Unsafe relative'):
        labeled_episodes([spec])


def test_cache_compatibility_accepts_only_pinned_reader_versions():
    from utils.common.source_compatibility import _manifest
    root = Path(__file__).resolve().parents[1]
    names = ('data/src/loaders/situation_dataset.py', 'data/src/processing/cache_velocity_tokens.py')
    old = {name: _manifest()['cache_reader_extensions'][name]['before'][0] for name in names}
    current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    assert compatible_export_sources(old, current)
    damaged = copy.deepcopy(current)
    damaged[names[0]] = 'f' * 64
    assert not compatible_export_sources(old, damaged)
    old[names[0]] = 'b' * 64
    assert not compatible_export_sources(old, current)
