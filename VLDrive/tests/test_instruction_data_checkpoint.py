"""Actual dataset join and stage/checkpoint transfer regression tests."""
from dataclasses import dataclass

import pytest
import torch

from test_navigation_cache import cache
from data.src.processing.build_path_features import build_path_features
from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
from training.common.checkpoints import (save_model_checkpoint, restore_model_checkpoint,
    warm_start_model_checkpoint, save_checkpoint, restore_checkpoint, parameter_hash)
from training.connector_llm.stages import apply_instruction_stage, PATH_PREFIXES
from training.common.runtime.batching import batch_samples


def test_new_dataset_joins_every_frame_and_keeps_legacy_samples(cache, tmp_path):
    folder, labels, _, _ = cache
    path = tmp_path / 'path_features.json'
    build_path_features(labels, path)
    old = NavigationCacheDataset(folder, labels, 'train', sequence_length=3, frame_sampling_interval=2)
    new = NavigationCacheDataset(folder, labels, 'train', sequence_length=3, frame_sampling_interval=2,
                                 path_feature_cache=path)
    assert 'path_features' not in old[0]['model_inputs']
    sample = new[0]
    assert sample['model_inputs']['instruction'] == 'Follow the straight path.'
    assert sample['metadata']['sample_ids'] == ['train_episode/0', 'train_episode/200000000', 'train_episode/400000000']
    torch.testing.assert_close(sample['training_targets']['future_trajectory'], old[0]['training_targets']['future_trajectory'])
    data = batch_samples([new[0], new[1]], 'cpu')['model_inputs']['path_features']
    assert data['ego_relation'].shape == (2, 3, 3)
    assert data['path_context'].shape == (2, 3, 5)
    assert data['path_valid'].dtype == torch.bool
    assert 'path_feature_cache_sha256' in new.describe()


def test_new_mode_does_not_require_common_legacy_candidate(cache, tmp_path, monkeypatch):
    folder, labels, _, _ = cache
    path = tmp_path / 'path_features.json'
    build_path_features(labels, path)
    from navigation.sampling import NavigationLabels
    monkeypatch.setattr(NavigationLabels, 'eligible', lambda *a, **k: [])
    with pytest.raises(ValueError, match='No valid'):
        NavigationCacheDataset(folder, labels, 'train', sequence_length=3)
    assert len(NavigationCacheDataset(folder, labels, 'train', sequence_length=3, path_feature_cache=path)) > 0


@dataclass
class Config:
    path_navigation: bool = False
    visual_sha256: str = 'unchanged-vision'


class CheckpointModel(torch.nn.Module):
    checkpoint_schema = 'test-path-checkpoint-v1'

    def __init__(self, enabled=False):
        super().__init__()
        self.config = Config(enabled)
        self.language_model = torch.nn.Module()
        self.language_model.base = torch.nn.Linear(2, 2)
        self.frozen_asset = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        if enabled:
            self.language_model.path_tokenizer = torch.nn.Linear(2, 2)
            self.language_model.navigation_cross_attention = torch.nn.Linear(2, 2)


def save(path, model):
    save_model_checkpoint(path, model, updates=3, provenance={}, settings={}, metrics={})


def test_legacy_warm_start_allows_only_explicit_new_prefixes(tmp_path):
    old, new = CheckpointModel(), CheckpointModel(True)
    path = tmp_path / 'base.pt'
    save(path, old)
    with pytest.raises(ValueError, match='parameter set'):
        warm_start_model_checkpoint(path, new, provenance={}, transfer_reason='path extension')
    initialized = new.language_model.path_tokenizer.weight.detach().clone()
    audit = warm_start_model_checkpoint(path, new, provenance={}, transfer_reason='path extension', new_prefixes=PATH_PREFIXES)
    assert len(audit['initialized_new_keys']) == 4
    torch.testing.assert_close(old.language_model.base.weight, new.language_model.base.weight)
    torch.testing.assert_close(initialized, new.language_model.path_tokenizer.weight)
    with pytest.raises(ValueError, match='only in the destination'):
        warm_start_model_checkpoint(path, new, provenance={}, transfer_reason='bad whitelist', new_prefixes=('language_model.base.',))


def test_warmup_checkpoint_retains_frozen_base_and_transfers_to_joint(tmp_path):
    old = CheckpointModel(True)
    apply_instruction_stage(old, 'path_warmup')
    path = tmp_path / 'warmup.pt'
    save(path, old)
    destination = CheckpointModel(True)
    apply_instruction_stage(destination, 'joint')
    warm_start_model_checkpoint(path, destination, provenance={}, transfer_reason='warmup to joint')
    for name, p in old.named_parameters():
        torch.testing.assert_close(p, dict(destination.named_parameters())[name])
    restored = CheckpointModel(True)
    apply_instruction_stage(restored, 'path_warmup')
    restore_model_checkpoint(path, restored, provenance={})
    assert parameter_hash(old, trainable=True) == parameter_hash(restored, trainable=True)
    optimizer = torch.optim.AdamW([p for p in old.parameters() if p.requires_grad])
    full = tmp_path / 'resume.pt'
    save_checkpoint(full, old, optimizer, updates=7, provenance={}, settings={})
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
    assert restore_checkpoint(full, restored, restored_optimizer, provenance={})['updates'] == 7


def test_path_cache_resolves_against_explicit_path_root(tmp_path):
    from types import SimpleNamespace
    from utils.common.runtime import resolve_settings
    args = SimpleNamespace(path_root=tmp_path, dataset_root=None, asset_root=None)
    result = resolve_settings(dict(dataset='data', model={},
                                   path_instruction={'feature_cache': 'caches/path.json'}), args)
    assert result['path_instruction']['feature_cache'] == str((tmp_path / 'caches/path.json').resolve())


@pytest.mark.parametrize('enabled', [False, True])
def test_versioned_data_does_not_force_legacy_token_cache_for_new_mode(monkeypatch, enabled):
    import copy
    from utils.config import config_files
    config = dict(data_version='version.json', model=dict(visual_sha256='same', path_navigation=enabled),
                  navigation_sampling={})
    version = dict(schema='vldrive-data-version-v1', fingerprints={}, config=dict(
        model=dict(visual_sha256='same'), instructions='labels', feature_cache='visual', split_file='splits',
        navigation_sampling=dict(token_cache='legacy-text')))
    monkeypatch.setattr(config_files, 'read_document', lambda path: copy.deepcopy(config if str(path) == 'input.json' else version))
    result = config_files.load_config('input.json')
    assert result['feature_cache'] == 'visual'
    assert result['navigation_sampling'].get('token_cache') == (None if enabled else 'legacy-text')
