import torch

from lavis.models.vldrive.ddia_attention import (
    apply_rope,
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
