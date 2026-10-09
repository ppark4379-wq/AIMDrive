"""Incremental versions must preserve raw membership, targets and cached tensors."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.src.processing import joint_preprocessing as joint
from data.src.processing.cache_sensor_record_tokens import tensor_ref
from data.src.processing.cache_velocity_tokens import reuse_situation_records
from data.src.loaders.situation_dataset import labeled_episodes
from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
from navigation.engine import DEFAULT_POLICY
from instruction.situation.text import load_catalog


def raw_episode(root, name='lap/episode_000'):
    folder = root / name
    path = folder / 'runtime/expert/path.csv'
    path.parent.mkdir(parents=True)
    path.write_text('x,y\n' + ''.join(f'{i},0\n' for i in range(100)))
    joint.write(folder / 'episode.json', {'expert_inputs': {'runtime/expert/path.csv': joint.digest(path)}})
    signals, gt, labels = [], [], []
    template = load_catalog()['static_obstacle']['static_normal']['instruction']
    for i in range(60):
        stamp = (i + 10) * 100_000_000
        signals.append(dict(record_id=i, role='ego', source_timestamp_ns=stamp,
                            signal=dict(position=[i / 2, 0, 0], yaw=0, speed=5)))
        gt.append(dict(index=i, lidar_source_ns=stamp, ego_xy=[i / 2, 0]))
        labels.append(dict(frame_id=i, sensor_time_ns=stamp, frame_valid=i != 4,
            zone='static_obstacle', zone_instruction=1,
            labeling=dict(index=i, zone_id='static_obstacle', instruction_id='static_normal',
                          instruction=template, instruction_variables={}, instruction_consistency='consistent')))
    for name, seal, rows in [('quality/active_signals.json', 'quality/QUALITY_COMPLETE.json', signals),
                             ('gt_timeline_v1/frames.json', 'gt_timeline_v1/COMPLETE.json', gt)]:
        joint.write(folder / name, rows)
        joint.write(folder / seal, {'files': {Path(name).name: joint.digest(folder / name)}})
    path = folder / 'situation_labels/labeled_frames.jsonl'
    path.parent.mkdir()
    joint.write_lines(path, labels)
    joint.write(path.parent / 'COMPLETE.json', {'files': {path.name: joint.digest(path)}})
    return folder


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(joint, 'ROOT', tmp_path)
    root = tmp_path / 'raw'
    raw_episode(root)
    cfg = dict(model=dict(visual_sha256='a' * 64, path_navigation=True,
                          situation_instruction=True, zone_instruction=True),
               navigation_sampling={}, path_instruction={}, situation_instruction={}, zone_instruction={},
               preprocessing=dict(datasets=[dict(root=str(root), namespace='static_obstacle')],
                                  navigation=dict(policy=DEFAULT_POLICY)),
               sequence_length=11, frame_sampling_interval=2,
               validation_enabled=False, include_validation_in_train=True)
    return tmp_path, root, cfg


def fake_vision(cfg):
    folder = joint.ROOT / cfg['feature_cache']
    folder.mkdir(parents=True, exist_ok=True)
    episodes = labeled_episodes(cfg['situation_instruction']['datasets'])
    records = []
    for episode in episodes:
        for row in episode['rows']:
            i = row['frame_id']
            key = hashlib.sha256(f'{episode["identity"]}|{i}'.encode()).hexdigest()
            ref = tensor_ref(folder, f'visual_feature/{key}.npy', np.full((106, 256), i, dtype=np.float16))
            record = dict(episode_id=episode['identity'], frame_index=i,
                sensor_source_ns=row['sensor_time_ns'], visual_feature=ref)
            joint.write(folder / 'records' / f'{key}.json', record)
            records.append(record)
    joint.write_lines(folder / 'manifest.jsonl', records)
    joint.write(folder / 'source.json', dict(episodes={e['identity']: {'labels_sha256': e['label_sha256']} for e in episodes}))


def test_prepare_and_finish_joins_real_loader_without_removing_empty_situation(source):
    tmp, root, cfg = source
    cfg = joint.frozen_config(cfg, 'v1')
    folder = tmp / 'data/processed/v1'
    joint.write(folder / 'processing_config.json', cfg)
    result = joint.prepare(cfg, folder, {})
    assert result['episodes'][0]['target_frames'] == 35  # Last 2.5 s has no future GT.
    fake_vision(cfg)
    summary = joint.finish(cfg, folder)
    assert summary['sample_visual_shape'] == [11, 106, 256]
    assert summary['sample_target_shape'] == [11, 5, 2]
    dataset = NavigationCacheDataset(tmp / cfg['feature_cache'], tmp / cfg['instructions'], 'train',
        sequence_length=11, frame_sampling_interval=2, include_validation_in_train=True,
        path_feature_cache=tmp / cfg['path_instruction']['feature_cache'],
        situation_datasets=cfg['situation_instruction']['datasets'], zone_datasets=cfg['zone_instruction']['datasets'])
    sample = dataset[0]
    expected = np.array([[0, -2.5], [0, -5], [0, -7.5], [0, -10], [0, -12.5]])
    np.testing.assert_allclose(sample['training_targets']['future_trajectory'][0], expected)
    assert sample['model_inputs']['situation_instructions'][2] is None  # Frame 4 retained.
    assert dataset.windows[0][2]['frame_index'] == 4
    assert (sample['model_inputs']['zone_ids'] == 1).all()
    before = joint.digest(folder / 'instructions/manifest.json')
    joint.prepare(cfg, folder, {})  # Interrupted preparation can resume deterministically.
    assert joint.digest(folder / 'instructions/manifest.json') == before


def test_snapshot_excludes_later_episodes_until_new_version(source):
    tmp, root, cfg = source
    first = joint.frozen_config(cfg, 'v1')
    raw_episode(root, 'added/episode_000')
    assert len(labeled_episodes(first['situation_instruction']['datasets'])) == 1
    second = joint.frozen_config(cfg, 'v2')
    assert len(labeled_episodes(second['situation_instruction']['datasets'])) == 2
    with pytest.raises(ValueError, match='Unsafe'):
        labeled_episodes([dict(root=str(root), namespace='x', include_episodes=['../outside'])])


def cache_source(folder):
    source = dict(schema='situation-ve-cache-v1', checkpoint_sha256='a' * 64,
        encoder_contract={'velocity_input': 'ego'}, implementations={'encoder': 'abc'},
        inference_precision='float32', storage_precision='float16', diagnostic_frame_limit=None,
        episodes={'unchanged': {'labels': 'a'}, 'changed': {'labels': 'b'}})
    joint.write(folder / 'source.json', source)
    for i, name in enumerate(source['episodes']):
        ref = tensor_ref(folder, f'visual_feature/{i}.npy', np.full((106, 256), i, dtype=np.float16))
        joint.write(folder / 'records' / f'{i}.json', dict(episode_id=name, visual_feature=ref))
    return source


def test_reuse_verifies_recipe_episode_and_tensor_content(tmp_path):
    previous, output = tmp_path / 'old', tmp_path / 'new'
    source = cache_source(previous)
    current = copy.deepcopy(source)
    current['episodes']['changed']['labels'] = 'new'
    old_bytes = (previous / 'visual_feature/0.npy').read_bytes()
    output.mkdir()
    assert reuse_situation_records(output, current, previous) == 1
    assert (output / 'visual_feature/0.npy').read_bytes() == old_bytes
    assert not (output / 'records/1.json').exists()
    assert (previous / 'visual_feature/0.npy').read_bytes() == old_bytes
    incompatible = copy.deepcopy(current)
    incompatible['checkpoint_sha256'] = 'b' * 64
    with pytest.raises(ValueError):
        reuse_situation_records(output, incompatible, previous)
    (previous / 'visual_feature/1.npy').write_bytes(b'corrupt')
    fresh = tmp_path / 'fresh'
    fresh.mkdir()
    assert reuse_situation_records(fresh, source, previous) == 1


def test_version_restores_path_and_frozen_label_sources(source, monkeypatch):
    tmp, root, cfg = source
    from utils.config import config_files
    monkeypatch.setattr(config_files, 'ROOT', tmp)
    snapshot = joint.frozen_config(cfg, 'v1')
    joint.write(tmp / 'version.json', dict(schema='vldrive-data-version-v1', config=snapshot, fingerprints={}))
    cfg['data_version'] = 'version.json'
    joint.write(tmp / 'training.json', cfg)
    restored = config_files.load_config(tmp / 'training.json')
    assert restored['path_instruction'] == snapshot['path_instruction']
    assert restored['situation_instruction']['datasets'] == snapshot['situation_instruction']['datasets']
    assert restored['zone_instruction']['datasets'] == snapshot['zone_instruction']['datasets']


def test_process_prepare_resume_activate_and_add_version(source, monkeypatch):
    tmp, root, cfg = source
    from data.src.processing import cache_velocity_tokens
    monkeypatch.setattr(joint, 'encoder_recipe', lambda config: {})
    monkeypatch.setattr(joint, 'vision_config', lambda config: config)
    monkeypatch.setattr(cache_velocity_tokens, 'cache_situation', lambda config, **kw: fake_vision(config))
    config = tmp / 'training.json'
    joint.write(config, cfg)
    state = joint.process(config, 'v1', prepare_only=True)
    assert joint.read(state)['status'] == 'prepared'
    assert 'data_version' not in joint.read(config)
    manifest = joint.process(config, 'v1', resume=True)
    assert joint.read(state)['status'] == 'complete'
    assert joint.read(config)['data_version'] == 'data/processed/v1/manifest.json'
    old_digest = joint.digest(manifest)
    with pytest.raises(ValueError, match='unfinished'):
        joint.process(config, 'v1', resume=True)
    joint.process(config, 'v2', reuse_version='v1', prepare_only=True)
    assert joint.digest(manifest) == old_digest
    assert joint.read(tmp / 'data/processed/v2/status.json')['reuse_version'] == 'v1'


@pytest.fixture
def completed(source, monkeypatch):
    tmp, root, cfg = source
    from data.src.processing import cache_velocity_tokens
    monkeypatch.setattr(joint, 'encoder_recipe', lambda config: {})
    monkeypatch.setattr(joint, 'vision_config', lambda config: config)
    reused = []

    def cache(config, reuse_cache=None):
        reused.append(reuse_cache)
        fake_vision(config)

    monkeypatch.setattr(cache_velocity_tokens, 'cache_situation', cache)
    config = tmp / 'training.json'
    joint.write(config, cfg)
    manifest = joint.process(config, 'v1')
    return tmp, root, config, manifest, reused


def test_extend_keeps_training_path_and_publishes_all_episodes(completed, monkeypatch):
    tmp, root, config, manifest, reused = completed
    from utils.config import config_files
    monkeypatch.setattr(config_files, 'ROOT', tmp)
    old_config = config.read_bytes()
    old_manifest = manifest.read_bytes()
    raw_episode(root, 'added/episode_000')
    state = joint.process(config, 'v1', extend=True, prepare_only=True)
    assert joint.read(state)['status'] == 'prepared'
    assert manifest.read_bytes() == old_manifest
    assert config.read_bytes() == old_config
    result = joint.process(config, 'v1', extend=True, resume=True)
    assert result == manifest
    assert config.read_bytes() == old_config
    bundle = joint.read(manifest)
    assert bundle['visual_frames'] == 120
    assert bundle['training_frames'] == 70
    assert bundle['name'] == 'v1'
    assert reused[-1] is None  # Only new episodes are exported; old tensors stay untouched.
    restored = config_files.load_config(config)
    assert restored['feature_cache'] == 'data/processed/v1/vision_tokens'
    assert restored['path_instruction']['feature_cache'].startswith('data/processed/v1/')
    assert len(labeled_episodes(restored['situation_instruction']['datasets'])) == 2
    assert not (manifest.parent.parent / 'v1_extend_pending').exists()
    assert not (manifest.parent.parent / 'v1_extend_previous').exists()
    assert 'extend_pending' not in manifest.read_text()
    visual = manifest.parent / 'vision_tokens'
    rows = [json.loads(line) for line in (visual / 'manifest.jsonl').read_text().splitlines()]
    assert all((visual / row['visual_feature']['path']).is_file() for row in rows)
    dataset = NavigationCacheDataset(visual, manifest.parent / 'instructions', 'train',
        frame_sampling_interval=2, include_validation_in_train=True,
        path_feature_cache=manifest.parent / 'navigation_tokens/path_features.json')
    for window in (dataset[0], dataset[len(dataset) - 1]):
        assert list(window['model_inputs']['visual_feature'].shape) == [11, 106, 256]


def test_failed_extension_preserves_original_and_can_resume(completed, monkeypatch):
    tmp, root, config, manifest, reused = completed
    from data.src.processing import cache_velocity_tokens
    original = manifest.read_bytes()
    raw_episode(root, 'added/episode_000')
    cache = cache_velocity_tokens.cache_situation

    def fail(*args, **kwargs):
        raise RuntimeError('GPU interrupted')

    monkeypatch.setattr(cache_velocity_tokens, 'cache_situation', fail)
    with pytest.raises(RuntimeError, match='GPU interrupted'):
        joint.process(config, 'v1', extend=True)
    assert manifest.read_bytes() == original
    joint._verify_version(manifest.parent)
    monkeypatch.setattr(cache_velocity_tokens, 'cache_situation', cache)
    joint.process(config, 'v1', extend=True, resume=True)
    assert joint.read(manifest)['visual_frames'] == 120


def test_extend_rejects_removed_episodes_and_changed_base(completed):
    tmp, root, config, manifest, reused = completed
    cfg = joint.read(config)
    cfg['preprocessing']['datasets'][0]['namespace'] = 'renamed'
    joint.write(config, cfg)
    with pytest.raises(ValueError, match='remove or rename'):
        joint.process(config, 'v1', extend=True)
    cfg['preprocessing']['datasets'][0]['namespace'] = 'static_obstacle'
    joint.write(config, cfg)
    joint.process(config, 'v1', extend=True, prepare_only=True)
    bundle = joint.read(manifest)
    bundle['changed'] = True
    joint.write(manifest, bundle)
    with pytest.raises(ValueError, match='base dataset changed'):
        joint.process(config, 'v1', extend=True, resume=True)


def test_publication_failure_restores_original_and_retries(completed, monkeypatch):
    tmp, root, config, manifest, reused = completed
    original = manifest.read_bytes()
    raw_episode(root, 'added/episode_000')
    rename = Path.rename

    def fail_pending(path, destination):
        if path.name == 'v1_extend_pending':
            raise PermissionError('folder busy')
        return rename(path, destination)

    monkeypatch.setattr(Path, 'rename', fail_pending)
    with pytest.raises(PermissionError, match='folder busy'):
        joint.process(config, 'v1', extend=True)
    assert manifest.read_bytes() == original
    joint._verify_version(manifest.parent)
    monkeypatch.setattr(Path, 'rename', rename)
    joint.process(config, 'v1', extend=True, resume=True)
    assert joint.read(manifest)['visual_frames'] == 120


def test_cleanup_failure_can_resume_without_repeating_cache(completed, monkeypatch):
    tmp, root, config, manifest, reused = completed
    raw_episode(root, 'added/episode_000')
    remove = joint._remove_extension_backup

    def fail(path):
        raise PermissionError('cleanup interrupted')

    monkeypatch.setattr(joint, '_remove_extension_backup', fail)
    with pytest.raises(PermissionError, match='cleanup interrupted'):
        joint.process(config, 'v1', extend=True)
    assert joint.read(manifest)['visual_frames'] == 120
    calls = len(reused)
    monkeypatch.setattr(joint, '_remove_extension_backup', remove)
    assert joint.process(config, 'v1', extend=True, resume=True) == manifest
    assert len(reused) == calls
    assert not (manifest.parent.parent / 'v1_extend_previous').exists()


def test_append_never_opens_old_tensor_or_record_files(completed, monkeypatch):
    tmp, root, config, manifest, reused = completed
    raw_episode(root, 'added/episode_000')
    old_root = manifest.parent / 'vision_tokens'
    tensor = next((old_root / 'visual_feature').glob('*.npy'))
    stamp = tensor.stat().st_mtime_ns
    open_file = Path.open

    def guard(path, *args, **kwargs):
        if old_root in path.parents and (path.suffix == '.npy' or path.parent.name == 'records'):
            raise AssertionError('Append reopened an old cache file: ' + str(path))
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guard)
    joint.process(config, 'v1', extend=True)
    assert tensor.stat().st_mtime_ns == stamp
    monkeypatch.setattr(Path, 'open', open_file)
    # A subsequent addition must keep the first addition's nested tensors too.
    raw_episode(root, 'third/episode_000')
    joint.process(config, 'v1', extend=True)
    rows = [json.loads(line) for line in (old_root / 'manifest.jsonl').read_text().splitlines()]
    assert len(rows) == 105
    assert all((old_root / row['visual_feature']['path']).is_file() for row in rows)


def test_append_recovers_interrupted_storage_transfer(completed, monkeypatch):
    from data.src.processing.joint_incremental import transfer_storage
    tmp, root, config, manifest, reused = completed
    raw_episode(root, 'added/episode_000')
    pending = manifest.parent.with_name('v1_extend_pending')
    extension = dict(target='v1', base_manifest_sha256=joint.digest(manifest))
    joint.process(config, pending.name, reuse_version='v1', activate=False, _extension=extension)
    backup = manifest.parent.with_name('v1_extend_previous')
    manifest.parent.rename(backup)
    transfer_storage(backup, pending)  # Simulate interruption before the final rename.
    assert not manifest.parent.exists()
    joint.process(config, 'v1', extend=True, resume=True)
    assert joint.read(manifest)['visual_frames'] == 120
    assert (manifest.parent / 'vision_tokens/visual_feature').is_dir()
