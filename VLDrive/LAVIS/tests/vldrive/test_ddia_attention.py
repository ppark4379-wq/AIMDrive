import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaRotaryEmbedding, apply_rotary_pos_emb,
)

from lavis.models.vldrive.ddia_attention import (
    apply_rope,
    DDIALlamaAttention,
    build_ddia_attention_mask,
    compose_ddia_logits,
    repeat_kv,
)
from lavis.models.vldrive.multimodal_sequence import INSTRUCTION, PADDING, VISUAL


def test_ddia_modality_mask_and_padding():
    ids = torch.tensor([[INSTRUCTION, INSTRUCTION, VISUAL, VISUAL, VISUAL]])
    mask = build_ddia_attention_mask(ids)[0]
    assert mask[0].tolist() == [True, True, False, False, False]
    assert mask[2].tolist() == [True, True, False, False, False]
    assert mask[3].tolist() == [True, True, True, False, False]
    assert mask[4].tolist() == [True, True, True, True, False]
    padded = build_ddia_attention_mask(torch.tensor([[INSTRUCTION, PADDING, VISUAL]]))[0]
    assert not padded[:, 1].any() and not padded[1].any()


def test_cross_modal_raw_and_other_blocks_rope():
    raw_q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
    raw_k = raw_q.clone()
    positions = torch.tensor([[0, 1, 2]])
    rope_q = apply_rope(raw_q, positions)
    rope_k = apply_rope(raw_k, positions)
    ids = torch.tensor([[INSTRUCTION, INSTRUCTION, VISUAL]])
    logits, _ = compose_ddia_logits(raw_q, raw_k, rope_q, rope_k, ids)
    raw = raw_q @ raw_k.transpose(-1, -2) / (2.0 ** 0.5)
    rotated = rope_q @ rope_k.transpose(-1, -2) / (2.0 ** 0.5)
    assert torch.allclose(logits[:, :, 2, :2], raw[:, :, 2, :2])  # V -> I is raw
    assert torch.allclose(logits[:, :, 0, 1], rotated[:, :, 0, 1])  # I -> I is RoPE
    assert torch.allclose(logits[:, :, 2, 2], rotated[:, :, 2, 2])  # V -> V is RoPE
    assert not torch.allclose(raw[:, :, 2, 0], rotated[:, :, 2, 0])


def test_gqa_repeat_kv_dimensions():
    keys = torch.randn(2, 2, 5, 8)
    assert repeat_kv(keys, 4).shape == (2, 8, 5, 8)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_rope_long_positions_match_hf_431_trig_before_cast(dtype):
    """HF 4.31 constructs FP32 sin/cos, not sin/cos of BF16 angles."""
    torch.manual_seed(81)
    positions = torch.tensor([[0, 1, 1199, 1200, 1201, 1305, 2047]])
    values = torch.randn(1, 2, positions.shape[1], 64).to(dtype)
    rotary = LlamaRotaryEmbedding(64, max_position_embeddings=2048)
    cosine, sine = rotary(values, seq_len=2048)
    reference, _ = apply_rotary_pos_emb(values, values, cosine, sine, positions)
    actual = apply_rope(values, positions)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert actual.dtype == dtype
    if dtype == torch.bfloat16:
        frequency = 1 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
        angles = positions.float().unsqueeze(-1) * frequency
        angles = torch.cat((angles, angles), -1).unsqueeze(1).to(dtype)
        first, second = values.chunk(2, -1)
        legacy = values * angles.cos() + torch.cat((-second, first), -1) * angles.sin()
        # This regression must distinguish the previously broken implementation.
        assert (legacy.float() - reference.float()).abs().max() > .1


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_ddia_long_positions_preserve_visual_causality_and_padding(dtype):
    torch.manual_seed(82)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    attention = DDIALlamaAttention(LlamaAttention(config), config).to(dtype).eval()
    attention.modality_ids = torch.tensor([[INSTRUCTION, INSTRUCTION, VISUAL, VISUAL, VISUAL, PADDING]])
    positions = torch.tensor([[0, 1, 1199, 1200, 1305, 1306]])
    values = torch.randn(1, 6, 32).to(dtype)
    baseline, probabilities, _ = attention(values, position_ids=positions, output_attentions=True)
    changed = values.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 5
    future_changed, _, _ = attention(changed, position_ids=positions)
    torch.testing.assert_close(baseline[:, :4], future_changed[:, :4], rtol=0, atol=0)
    assert not baseline[:, 5].any()
    assert not probabilities[:, :, :5, 5].any()
    assert not probabilities[:, :, 2:5, 2:5].triu().any()
    assert torch.isfinite(baseline).all()
