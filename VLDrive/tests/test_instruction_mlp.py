"""MLP baseline shares features, temporal masks and checkpoint discipline with CA."""
from types import SimpleNamespace
from dataclasses import dataclass

import pytest
import torch

from instruction.navigation.navigation_mlp import NavigationMLP
from .test_instruction_crossattention import features
from .test_instruction_integration import make_model, sample
from training.connector_llm.stages import apply_instruction_stage, MLP_PREFIXES
from models.vldrive import VLDriveTrainingConfig
from training.common.checkpoints import (save_model_checkpoint, restore_model_checkpoint,
    warm_start_model_checkpoint, save_checkpoint, restore_checkpoint)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_mlp_batches_frames_without_temporal_mixing(dtype):
    torch.manual_seed(10)
    fusion = NavigationMLP(32, 16).to(dtype)
    data, text, mask = features(), torch.randn(2, 5, 32).to(dtype), torch.ones(2, 5, dtype=torch.bool)
    output = fusion(text, data, mask)
    reference = torch.cat([fusion(text, {k: v[:, i:i+1] for k, v in data.items()}, mask) for i in range(3)], 1)
    torch.testing.assert_close(output, reference)
    assert output.shape == (2, 3, 5, 32)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
    assert fusion.network[0].weight.grad.abs().sum() > 0


def test_mlp_invalid_paths_and_padded_text_fall_back_exactly():
    data = features()
    data['path_valid'][0].zero_()
    text = torch.randn(2, 4, 32)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    fusion = NavigationMLP(32, 16)
    output = fusion(text, data, mask)
    torch.testing.assert_close(output[0], text[0, None].expand(3, -1, -1), rtol=0, atol=0)
    torch.testing.assert_close(output[1, :, 2:], text[1, None, 2:].expand(3, -1, -1), rtol=0, atol=0)
    data['ego_relation'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        fusion(text, data, mask)
    data = features()
    data['path_context'] = data['path_context'][..., :4]
    with pytest.raises(ValueError, match='Expected'):
        fusion(text, data, mask)


def mlp_model(monkeypatch):
    return make_model(monkeypatch, fusion='mlp')


@pytest.mark.parametrize('checkpointed', [False, True])
def test_mlp_decoder_backprop_and_future_isolation(monkeypatch, checkpointed):
    torch.manual_seed(21)
    model = mlp_model(monkeypatch)
    assert hasattr(model, 'navigation_mlp')
    assert not hasattr(model, 'path_tokenizer')
    assert not hasattr(model, 'navigation_cross_attention')
    if checkpointed:
        model.tinyllama.model.gradient_checkpointing_enable()
    visual, data = torch.randn(2, 3, 4, 32), sample()
    result = model(['follow'] * 2, visual, predict_per_frame=True, path_features=data)
    result.predicted_waypoints.square().mean().backward()
    assert model.navigation_mlp.network[0].weight.grad.abs().sum() > 0
    assert model.tinyllama.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert result.sequence.inputs_embeds.shape == (2, 21, 32)
    assert result.predicted_waypoints.shape == (2, 3, 10)
    model.eval()
    with torch.no_grad():
        baseline = model(['follow'] * 2, visual, predict_per_frame=True, path_features=data)
        changed = {k: v.clone() for k, v in data.items()}
        changed['ego_relation'][:, -1] += 20
        changed['path_context'][:, -1] -= 20
        future = model(['follow'] * 2, visual, predict_per_frame=True, path_features=changed)
    torch.testing.assert_close(baseline.predicted_waypoints[:, :-1], future.predicted_waypoints[:, :-1], rtol=0, atol=1e-6)
    assert not torch.allclose(baseline.predicted_waypoints[:, -1], future.predicted_waypoints[:, -1])


def test_mlp_initialization_preserves_base_weights(monkeypatch):
    torch.manual_seed(607)
    original = make_model(monkeypatch, enabled=False)
    torch.manual_seed(607)
    added = mlp_model(monkeypatch)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, added.state_dict()[name], rtol=0, atol=0)


def test_mlp_warmup_only_updates_new_branch(monkeypatch):
    wrapper = torch.nn.Module()
    wrapper.config = SimpleNamespace(path_navigation=True, path_fusion='mlp')
    wrapper.language_model = mlp_model(monkeypatch)
    wrapper.language_model.tinyllama.lm_head.requires_grad_(False)
    wrapper.language_model.end_predictor.requires_grad_(False)
    apply_instruction_stage(wrapper, 'path_warmup')
    for name, parameter in wrapper.named_parameters():
        assert parameter.requires_grad == name.startswith(MLP_PREFIXES)
    output = wrapper.language_model(['follow'] * 2, torch.randn(2, 3, 4, 32),
                                    path_features=sample(), predict_per_frame=True)
    output.predicted_waypoints.square().mean().backward()
    assert wrapper.language_model.navigation_mlp.network[0].weight.grad.abs().sum() > 0
    assert wrapper.language_model.tinyllama.model.layers[0].self_attn.q_proj.weight.grad is None
    apply_instruction_stage(wrapper, 'joint')
    assert wrapper.language_model.tinyllama.model.layers[0].self_attn.q_proj.weight.requires_grad
    assert not wrapper.language_model.tinyllama.lm_head.weight.requires_grad


def test_mlp_config_does_not_require_attention_head_divisibility():
    VLDriveTrainingConfig(path_navigation=True, path_fusion='mlp', path_width=17).validate()
    with pytest.raises(ValueError, match='path navigation'):
        VLDriveTrainingConfig(path_navigation=True, path_fusion='cross_attention', path_width=17).validate()
    with pytest.raises(ValueError, match='path navigation'):
        VLDriveTrainingConfig(path_fusion='typo').validate()


def test_mlp_four_windows_eleven_frames_share_one_network(monkeypatch):
    model = mlp_model(monkeypatch)
    data = features(batch=4, frames=11)
    result = model(['follow'] * 4, torch.randn(4, 11, 4, 32), predict_per_frame=True,
                   path_features=data)
    assert result.predicted_waypoints.shape == (4, 11, 10)
    assert result.sequence.inputs_embeds.shape == (4, 77, 32)
    assert len([name for name, _ in model.named_modules() if name.endswith('navigation_mlp')]) == 1
    result.predicted_waypoints.square().mean().backward()
    assert model.navigation_mlp.network[0].weight.grad.abs().sum() > 0


@dataclass
class MLPCheckpointConfig:
    path_navigation: bool = False
    path_fusion: str = 'mlp'
    visual_sha256: str = 'unchanged-vision'


class MLPCheckpointModel(torch.nn.Module):
    checkpoint_schema = 'test-path-checkpoint-v1'

    def __init__(self, enabled=False):
        super().__init__()
        self.config = MLPCheckpointConfig(enabled)
        self.language_model = torch.nn.Module()
        self.language_model.base = torch.nn.Linear(2, 2)
        self.frozen_asset = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        if enabled:
            self.language_model.navigation_mlp = NavigationMLP(32, 16)


def test_mlp_warm_start_checkpoint_resume_and_joint_transfer(tmp_path):
    original, added = MLPCheckpointModel(), MLPCheckpointModel(True)
    base_path = tmp_path / 'base.pt'
    save_model_checkpoint(base_path, original, updates=3, provenance={}, settings={}, metrics={})
    with pytest.raises(ValueError, match='parameter set'):
        warm_start_model_checkpoint(base_path, added, provenance={}, transfer_reason='new MLP branch')
    audit = warm_start_model_checkpoint(base_path, added, provenance={}, transfer_reason='new MLP branch',
                                       new_prefixes=MLP_PREFIXES)
    assert len(audit['initialized_new_keys']) == 5
    torch.testing.assert_close(original.language_model.base.weight, added.language_model.base.weight)
    apply_instruction_stage(added, 'path_warmup')
    optimizer = torch.optim.AdamW([p for p in added.parameters() if p.requires_grad], lr=.001)
    added.language_model.navigation_mlp(torch.randn(2, 3, 32), features(), torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    full_path, weights_path = tmp_path / 'resume.pt', tmp_path / 'warmup.pt'
    save_checkpoint(full_path, added, optimizer, updates=1, provenance={}, settings={})
    save_model_checkpoint(weights_path, added, updates=1, provenance={}, settings={}, metrics={})
    resumed = MLPCheckpointModel(True)
    apply_instruction_stage(resumed, 'path_warmup')
    resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.001)
    assert restore_checkpoint(full_path, resumed, resumed_optimizer, provenance={})['updates'] == 1
    restored = MLPCheckpointModel(True)
    apply_instruction_stage(restored, 'path_warmup')
    restore_model_checkpoint(weights_path, restored, provenance={})
    joint = MLPCheckpointModel(True)
    apply_instruction_stage(joint, 'joint')
    warm_start_model_checkpoint(weights_path, joint, provenance={}, transfer_reason='MLP warmup to joint')
    for destination in (resumed, restored, joint):
        for name, value in added.state_dict().items():
            torch.testing.assert_close(value, destination.state_dict()[name], rtol=0, atol=0)


def test_cross_attention_checkpoint_cannot_silently_load_into_mlp(tmp_path):
    from .test_instruction_data_checkpoint import CheckpointModel
    source, destination = CheckpointModel(True), MLPCheckpointModel(True)
    path = tmp_path / 'attention.pt'
    save_model_checkpoint(path, source, updates=3, provenance={}, settings={}, metrics={})
    before = destination.language_model.base.weight.detach().clone()
    with pytest.raises(ValueError, match='parameter set'):
        warm_start_model_checkpoint(path, destination, provenance={}, transfer_reason='incompatible architectures',
                                   new_prefixes=MLP_PREFIXES)
    torch.testing.assert_close(before, destination.language_model.base.weight, rtol=0, atol=0)
