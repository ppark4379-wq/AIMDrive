"""Small real Llama decoder tests; no downloaded model weights or GPU needed."""
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from models.llm import language_model
from models.llm.ddia_attention import install_ddia_attention
from training.connector_llm.stages import apply_instruction_stage, PATH_PREFIXES


class TextTokenizer:
    def __call__(self, texts, **kwargs):
        return dict(input_ids=torch.tensor([[1, 2, 3]] * len(texts)),
                    attention_mask=torch.ones(len(texts), 3, dtype=torch.long))


def make_model(monkeypatch, enabled=True, fusion='cross_attention'):
    def load(*args):
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                            num_key_value_heads=2, num_hidden_layers=2, vocab_size=32,
                            max_position_embeddings=128)
        return SimpleNamespace(tokenizer=TextTokenizer(), model=install_ddia_attention(LlamaForCausalLM(config)))
    monkeypatch.setattr(language_model, 'load_tinyllama', load)
    return language_model.VLDriveModel(visual_feature_dim=32, precision='fp32', path_navigation=enabled,
                                       path_width=16, path_heads=4, path_fusion=fusion)


def sample():
    return dict(ego_relation=torch.randn(2, 3, 3), path_context=torch.randn(2, 3, 5),
                path_valid=torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize('checkpointed', [False, True])
def test_end_to_end_gradient_and_future_path_causality(monkeypatch, checkpointed):
    torch.manual_seed(31)
    model = make_model(monkeypatch)
    if checkpointed:
        model.tinyllama.model.gradient_checkpointing_enable()
    visual, path = torch.randn(2, 3, 4, 32), sample()
    model.train()
    result = model(['follow'] * 2, visual, predict_per_frame=True, path_features=path)
    result.predicted_waypoints.square().mean().backward()
    for module in (model.path_tokenizer, model.navigation_cross_attention, model.tinyllama.model.layers[0]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert result.predicted_waypoints.shape == (2, 3, 10)
    model.eval()
    with torch.no_grad():
        baseline = model(['follow'] * 2, visual, predict_per_frame=True, path_features=path)
        changed = {k: v.clone() for k, v in path.items()}
        changed['ego_relation'][:, -1] += 3
        changed['path_context'][:, -1] -= 5
        changed_visual = visual.clone()
        changed_visual[:, -1] += 10
        future = model(['follow'] * 2, changed_visual, predict_per_frame=True, path_features=changed)
    torch.testing.assert_close(baseline.predicted_waypoints[:, :-1], future.predicted_waypoints[:, :-1], rtol=0, atol=1e-6)
    assert not torch.allclose(baseline.predicted_waypoints[:, -1], future.predicted_waypoints[:, -1])


def test_feature_off_keeps_legacy_packing(monkeypatch):
    model = make_model(monkeypatch, enabled=False).eval()
    visual = torch.randn(1, 3, 4, 32)
    result = model(['follow'], visual, predict_per_frame=True)
    assert result.sequence.inputs_embeds.shape == (1, 15, 32)
    assert result.sequence.frame_ids is None
    assert not any('path_tokenizer' in k or 'navigation_cross_attention' in k for k in model.state_dict())
    with pytest.raises(ValueError, match='disabled'):
        model(['follow'], visual, path_features=sample())


def test_new_module_initialization_preserves_existing_weights(monkeypatch):
    torch.manual_seed(607)
    original = make_model(monkeypatch, enabled=False)
    torch.manual_seed(607)
    added = make_model(monkeypatch, enabled=True)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, added.state_dict()[name], rtol=0, atol=0)


def test_warmup_freezes_base_but_backpropagates_through_decoder(monkeypatch):
    wrapper = torch.nn.Module()
    wrapper.config = SimpleNamespace(path_navigation=True)
    wrapper.language_model = make_model(monkeypatch)
    wrapper.language_model.tinyllama.lm_head.requires_grad_(False)
    wrapper.language_model.end_predictor.requires_grad_(False)
    apply_instruction_stage(wrapper, 'path_warmup')
    for name, p in wrapper.named_parameters():
        assert p.requires_grad == name.startswith(PATH_PREFIXES)
    result = wrapper.language_model(['follow'] * 2, torch.randn(2, 3, 4, 32),
                                     path_features=sample(), predict_per_frame=True)
    result.predicted_waypoints.square().mean().backward()
    assert wrapper.language_model.path_tokenizer.ego[0].weight.grad.abs().sum() > 0
    assert wrapper.language_model.tinyllama.model.layers[0].self_attn.q_proj.weight.grad is None
    apply_instruction_stage(wrapper, 'joint')
    assert wrapper.language_model.tinyllama.model.layers[0].self_attn.q_proj.weight.requires_grad
    assert not wrapper.language_model.tinyllama.lm_head.weight.requires_grad
