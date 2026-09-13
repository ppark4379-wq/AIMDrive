from dataclasses import dataclass
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from lavis.runners.vldrive_trainer import (
    restore_model_checkpoint,
    save_model_checkpoint,
    stride_indices,
    make_optimizer, train, save_checkpoint, restore_checkpoint,
    parameter_hash, preserve_rng_state, seed_all, periodic_checkpoint_path,
    balanced_training_indices, warm_start_model_checkpoint, set_cosine_learning_rate,
    evaluate,
)


def test_stride_indices_uniformly_covers_both_ends():
    assert stride_indices(11, 5) == [0, 5, 10]
    assert stride_indices(12, 5) == [0, 5, 10, 11]


def test_periodic_checkpoint_two_slots_preserve_previous_snapshot(tmp_path):
    paths = [periodic_checkpoint_path(tmp_path, step, 200, 2) for step in (200, 400, 600)]
    assert paths[0] == paths[2] and paths[0] != paths[1]
    assert periodic_checkpoint_path(tmp_path, 200, 200).name == 'checkpoint_000200.pt'
    with pytest.raises(ValueError):
        periodic_checkpoint_path(tmp_path, 200, 200, 1)
    model = _Model()
    optimizer = make_optimizer(model)
    for step, path in zip((200, 400, 600), paths):
        save_checkpoint(path, model, optimizer, updates=step, provenance={}, settings={})
    # A failed newest snapshot must not invalidate the preceding slot.
    paths[2].with_suffix('.sha256.json').write_text('{"bytes":0,"sha256":"bad"}')
    with pytest.raises(ValueError, match='integrity'):
        restore_checkpoint(paths[2], model, optimizer, provenance={})
    assert restore_checkpoint(paths[1], model, optimizer, provenance={})['updates'] == 400


@pytest.mark.parametrize("length,stride", [(0, 1), (1, 0), (-1, 2)])
def test_stride_indices_rejects_invalid_ranges(length, stride):
    with pytest.raises(ValueError):
        stride_indices(length, stride)


@dataclass
class _Config:
    width: int = 2
    precision: str = 'fp32'
    visual_sha256: str = 'vision-a'
    loss_weight: float = 1.0


class _Model(nn.Module):
    checkpoint_schema = "test-model-v1"

    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.trainable = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.frozen = nn.Parameter(torch.tensor([3.0]), requires_grad=False)

    def forward(self, sample):
        noise = torch.rand_like(self.trainable)
        return {'loss': ((self.trainable * noise - sample['model_inputs']['x']) ** 2).mean()}


def test_model_only_checkpoint_round_trip_and_frozen_guard(tmp_path):
    path = tmp_path / "model-best.pt"
    model = _Model()
    provenance = {"cache": "hash", "backend": {"torch": torch.__version__}}
    save_model_checkpoint(
        path, model, updates=7, provenance=provenance,
        settings={"epochs": 2}, metrics={"fde_m": 1.2},
    )
    with torch.no_grad():
        model.trainable.zero_()
    state = restore_model_checkpoint(path, model, provenance=provenance)
    assert model.trainable.tolist() == [1.0, 2.0]
    assert state["updates"] == 7

    with torch.no_grad():
        model.frozen.zero_()
    with pytest.raises(ValueError, match="frozen asset mismatch"):
        restore_model_checkpoint(path, model, provenance=provenance)


def test_full_checkpoint_preserves_stochastic_next_update(tmp_path):
    seed_all(19)
    model = _Model()
    optimizer = make_optimizer(model, lr=0.01)
    data = [{'model_inputs': {'x': torch.tensor([0.2, -0.1])}}]
    options = dict(schedule_updates=4, seed=19, device='cpu', lr=0.01)
    train(model, optimizer, data, [0], start_update=0, updates=1,
          log_path=tmp_path / 'first.jsonl', **options)
    path = tmp_path / 'checkpoint.pt'
    provenance = {'backend': {'torch': torch.__version__}}
    save_checkpoint(path, model, optimizer, updates=1, provenance=provenance, settings={})
    train(model, optimizer, data, [0], start_update=1, updates=2,
          log_path=tmp_path / 'uninterrupted.jsonl', **options)
    expected_hash = parameter_hash(model, trainable=True)
    expected_random = (random.random(), np.random.rand(), torch.rand(()).item())

    restore_checkpoint(path, model, optimizer, provenance=provenance)
    # Preflight/evaluation are decorated with this scope: Gumbel/dropout probes
    # must leave Python, NumPy and torch random streams unchanged.
    with preserve_rng_state():
        random.random()
        np.random.rand(8)
        torch.rand(8)
    train(model, optimizer, data, [0], start_update=1, updates=2,
          log_path=tmp_path / 'resumed.jsonl', **options)
    assert parameter_hash(model, trainable=True) == expected_hash
    assert (random.random(), np.random.rand(), torch.rand(()).item()) == expected_random
    uninterrupted = json.loads((tmp_path / 'uninterrupted.jsonl').read_text())
    resumed = json.loads((tmp_path / 'resumed.jsonl').read_text())
    assert uninterrupted['loss'] == resumed['loss']
    # A full checkpoint is also usable for evaluation without allocating a new optimizer.
    state = restore_model_checkpoint(path, model, provenance=provenance)
    assert state['updates'] == 1


class _TransferModel(_Model):
    def __init__(self, *, selector_width=2, width=2, extra=False):
        super().__init__()
        self.trainable = nn.Parameter(torch.arange(width, dtype=torch.float32))
        self.connector = nn.Module()
        self.connector.ccdp = nn.Linear(selector_width, selector_width)
        if extra:
            self.extra = nn.Parameter(torch.ones(1))


def _save_transfer(tmp_path, model=None):
    model = _TransferModel() if model is None else model
    path = tmp_path/'transfer.pt'
    provenance = dict(source={'old.py': 'old'}, datasets={'train': {'checkpoint_sha256': 'vision-a'}})
    save_model_checkpoint(path, model, updates=99, provenance=provenance,
                          settings={'epochs': 10}, metrics={})
    return path, model, provenance


def test_warm_start_explicit_transfer_resets_only_named_module_and_keeps_rng(tmp_path):
    path, source, old_provenance = _save_transfer(tmp_path)
    model = _TransferModel()
    with torch.no_grad():
        model.trainable.fill_(77)
        model.connector.ccdp.weight.fill_(23)
        model.connector.ccdp.bias.fill_(24)
    model.config.loss_weight = 2.
    new_provenance = dict(source={'new.py': 'new'}, datasets={'train': {'checkpoint_sha256': 'vision-a', 'windows': 20}})
    seed_all(818)
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    audit = warm_start_model_checkpoint(path, model, provenance=new_provenance,
        reset_prefixes=('connector.ccdp.',), transfer_reason='Repair collapsed selector with a new objective')
    assert torch.equal(model.trainable, source.trainable)
    assert torch.all(model.connector.ccdp.weight == 23)
    assert torch.all(model.connector.ccdp.bias == 24)
    assert audit['loaded_keys'] == ['trainable']
    assert audit['reset_new_keys'] == ['connector.ccdp.bias', 'connector.ccdp.weight']
    assert audit['old_provenance'] == old_provenance and audit['new_provenance'] == new_provenance
    assert audit['old_model_config']['loss_weight'] == 1 and audit['new_model_config']['loss_weight'] == 2
    assert not audit['optimizer_restored'] and not audit['rng_restored']
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng
    # Changes accepted for transfer are still rejected by strict exact restore.
    with pytest.raises(ValueError, match='contract/provenance'):
        restore_model_checkpoint(path, model, provenance=new_provenance)


@pytest.mark.parametrize('change', ['config_vision', 'cache_vision', 'frozen', 'schema'])
def test_warm_start_rejects_changed_frozen_assets_and_schema(tmp_path, change):
    path, _, provenance = _save_transfer(tmp_path)
    model = _TransferModel()
    if change == 'config_vision':
        model.config.visual_sha256 = 'vision-b'
    elif change == 'cache_vision':
        provenance['datasets']['train']['checkpoint_sha256'] = 'vision-b'
    elif change == 'frozen':
        with torch.no_grad():
            model.frozen.zero_()
    else:
        model.checkpoint_schema = 'other-v2'
    before = parameter_hash(model, trainable=True)
    with pytest.raises(ValueError):
        warm_start_model_checkpoint(path, model, provenance=provenance,
            reset_prefixes=('connector.ccdp.',), transfer_reason='Test transfer guard')
    assert parameter_hash(model, trainable=True) == before


@pytest.mark.parametrize('prefixes', [('',), ('connector',), ('connector.no_such_module.',),
                                     ('connector.', 'connector.ccdp.'), ('connector.ccdp.', 'connector.ccdp.')])
def test_warm_start_rejects_invalid_or_empty_reset_matches(tmp_path, prefixes):
    path, _, provenance = _save_transfer(tmp_path)
    with pytest.raises(ValueError, match='prefix'):
        warm_start_model_checkpoint(path, _TransferModel(), provenance=provenance,
            reset_prefixes=prefixes, transfer_reason='Test reset guard')


@pytest.mark.parametrize('model,error', [(_TransferModel(width=3), 'shape/dtype'),
                                        (_TransferModel(extra=True), 'parameter set')])
def test_warm_start_rejects_nonreset_parameter_mismatches_without_partial_load(tmp_path, model, error):
    path, _, provenance = _save_transfer(tmp_path)
    before = parameter_hash(model, trainable=True)
    with pytest.raises(ValueError, match=error):
        warm_start_model_checkpoint(path, model, provenance=provenance,
            reset_prefixes=('connector.ccdp.',), transfer_reason='Test shape guard')
    assert parameter_hash(model, trainable=True) == before


def test_warm_start_allows_shape_changes_only_inside_explicit_reset(tmp_path):
    path, source, provenance = _save_transfer(tmp_path)
    model = _TransferModel(selector_width=3)
    initialized = model.connector.ccdp.weight.detach().clone()
    warm_start_model_checkpoint(path, model, provenance=provenance,
        reset_prefixes=('connector.ccdp.',), transfer_reason='Explicitly rebuild selector dimensions')
    assert torch.equal(model.trainable, source.trainable)
    assert torch.equal(model.connector.ccdp.weight, initialized)


def test_warm_start_requires_reason_and_verified_sidecar(tmp_path):
    path, _, provenance = _save_transfer(tmp_path)
    with pytest.raises(ValueError, match='transfer_reason'):
        warm_start_model_checkpoint(path, _TransferModel(), provenance=provenance, transfer_reason='')
    path.with_suffix('.pt.sha256.json').write_text('{"sha256":"bad","bytes":0}')
    with pytest.raises(ValueError, match='integrity'):
        warm_start_model_checkpoint(path, _TransferModel(), provenance=provenance,
            transfer_reason='Test integrity guard')


class _TrajectoryDataset:
    split = 'train'
    trajectory_coordinate_convention = 'lmdrive_left_negative_forward'

    def __init__(self, targets=None):
        # Stop, forward, left and right; one clip each plus a second straight.
        endpoints = targets or [(0, 0), (0, -10), (3, -10), (-3, -10), (.4, -8)]
        self.targets = [torch.tensor(point, dtype=torch.float32).expand(1, 5, 2).clone() for point in endpoints]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {'training_targets': {'future_trajectory': self.targets[index]}}


def test_balanced_training_indices_are_reproducible_and_disclose_oversampling():
    dataset = _TrajectoryDataset()
    original = list(range(len(dataset)))
    chosen, audit = balanced_training_indices(dataset, indices=original, draws=640, seed=17)
    repeated, repeated_audit = balanced_training_indices(dataset, indices=original, draws=640, seed=17)
    assert chosen == repeated and audit == repeated_audit and original == list(range(5))
    assert len(chosen) == 640 and set(chosen) <= set(original)
    assert audit['candidate_counts'] == dict(stationary=1, moving_straight=2, moving_left=1, moving_right=1)
    assert audit['draw_counts'] == dict.fromkeys(audit['candidate_counts'], 160)
    assert audit['unique_sampled'] == 5
    assert chosen.count(0) == chosen.count(2) == chosen.count(3) == 160
    assert chosen != balanced_training_indices(dataset, draws=640, seed=18)[0]


@pytest.mark.parametrize('bad_split', ['validation', 'val', 'test', None])
def test_balanced_sampling_never_accepts_validation(bad_split):
    dataset = _TrajectoryDataset()
    dataset.split = bad_split
    with pytest.raises(ValueError, match='training split'):
        balanced_training_indices(dataset)


def test_balanced_sampling_thresholds_and_empty_category_guards():
    dataset = _TrajectoryDataset([(0, .499), (0, .5), (2, -3), (-2, -3)])
    _, audit = balanced_training_indices(dataset, draws=8)
    assert audit['category_indices'] == dict(stationary=[0], moving_straight=[1], moving_left=[2], moving_right=[3])
    with pytest.raises(ValueError, match='empty'):
        balanced_training_indices(dataset, indices=[0, 1, 2])
    with pytest.raises(ValueError, match='unique'):
        balanced_training_indices(dataset, indices=[0, 0, 1, 2, 3])
    with pytest.raises(ValueError, match='divisible'):
        balanced_training_indices(dataset, draws=7)


@pytest.mark.parametrize('indices', [[0, 0, 1], [1, 0]])
def test_evaluation_rejects_balanced_duplicates_or_nonchronological_indices(indices):
    with pytest.raises(ValueError, match='unique chronological'):
        evaluate(_Model(), _TrajectoryDataset(), indices, 'cpu')


@pytest.mark.parametrize('state_device', ['cpu', 'model'])
def test_optimizer_lr_prefix_multipliers_survive_cosine_and_state_roundtrip(state_device):
    model = _TransferModel()
    optimizer = make_optimizer(model, lr=.01, weight_decay=0, state_device=state_device,
                               lr_multipliers={'connector.ccdp.': 10})
    groups = {group['group_name']: group for group in optimizer.param_groups}
    assert groups['default']['lr'] == .01 and groups['connector.ccdp.']['lr'] == .1
    assert set_cosine_learning_rate(optimizer, lr=.01, step=5, schedule_updates=10) == .005
    assert groups['default']['lr'] == .005 and groups['connector.ccdp.']['lr'] == .05
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()
    restored = make_optimizer(model, lr=.01, weight_decay=0, state_device=state_device,
                              lr_multipliers={'connector.ccdp.': 10})
    restored.load_state_dict(state)
    set_cosine_learning_rate(restored, lr=.01, step=0, schedule_updates=10)
    assert [group['lr'] for group in restored.param_groups] == [.01, .1]
    assert [group['base_lr_mult'] for group in restored.param_groups] == [1, 10]


@pytest.mark.parametrize('multipliers', [{'not_present.': 10}, {'connector.': 2, 'connector.ccdp.': 10},
                                        {'connector.ccdp.': 0}, {'': 2}])
def test_optimizer_rejects_unmatched_overlapping_or_invalid_lr_prefixes(multipliers):
    with pytest.raises(ValueError):
        make_optimizer(_TransferModel(), lr_multipliers=multipliers)


def test_training_local_sampling_offset_keeps_global_update_and_cosine_lr(tmp_path):
    model = _Model()
    optimizer = make_optimizer(model, lr=.01)
    data = [{'model_inputs': {'x': torch.tensor([float(i), -.1])}} for i in range(3)]
    callbacks = []
    train(model, optimizer, data, [0, 1, 2], start_update=5, updates=6, schedule_updates=20,
          sampling_step_offset=5, seed=17, device='cpu', lr=.01,
          log_path=tmp_path/'local_sampling.jsonl', on_update=callbacks.append)
    row = json.loads((tmp_path/'local_sampling.jsonl').read_text())
    expected_index = torch.randperm(3, generator=torch.Generator().manual_seed(17)).tolist()[0]
    assert row['index'] == expected_index and row['update'] == 6 and callbacks == [6]
    assert row['lr'] == pytest.approx(.01 * .5 * (1 + np.cos(np.pi * 5 / 20)))
    with pytest.raises(ValueError):
        train(model, optimizer, data, [0], start_update=5, updates=6, schedule_updates=20,
              sampling_step_offset=6, seed=17, device='cpu', log_path=tmp_path/'invalid.jsonl')
