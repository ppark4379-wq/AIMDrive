"""Evaluation-only full-window optimizations; training/checkpoint code is untouched.

MEFA and Gumbel change historical tokens when a window moves. Therefore visual
KV caches are deliberately NOT carried between windows. Shared DDIA geometry
is computed once per window, keeping the original GEMM shapes and BF16 math.
"""
from contextlib import contextmanager
import math

import torch

from models.llm.ddia_attention import (DDIALlamaAttention, build_ddia_attention_mask,
                                   repeat_kv, rotate_half)


class _Layout:
    def __init__(self):
        self.identity = None

    def prepare(self, attention, positions, dtype):
        ids = attention.modality_ids
        identity = (id(ids), id(positions), dtype, attention.head_dim, attention.rope_theta)
        if identity != self.identity:
            if ids.shape[0] != 1:
                raise ValueError('Fast DDIA requires a single sequence')
            frequency = 1. / (attention.rope_theta ** (
                torch.arange(0, attention.head_dim, 2, device=ids.device).float() / attention.head_dim))
            angles = positions.float().unsqueeze(-1) * frequency
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            self.cosine, self.sine = angles.cos().to(dtype), angles.sin().to(dtype)
            self.allowed = build_ddia_attention_mask(ids).unsqueeze(1)
            no_key = ~self.allowed.any(dim=-1, keepdim=True)
            self.safe_allowed = self.allowed | (no_key & torch.eye(ids.shape[1], device=ids.device, dtype=torch.bool)[None, None])
            self.visual_instruction = (ids[:, None, :, None] == 1) & (ids[:, None, None, :] == 0)
            self.valid_query = (ids != -1)[:, None, :, None]
            # Hold tensor references so Python cannot recycle an identity.
            self.ids, self.positions, self.identity = ids, positions, identity
        return self


def _attention_forward(attention, layout, hidden_states, attention_mask=None,
                       position_ids=None, past_key_value=None, output_attentions=False,
                       use_cache=False, **kwargs):
    if attention.training or torch.is_grad_enabled() or past_key_value is not None or use_cache:
        raise ValueError('Fast DDIA is uncached inference only')
    batch, length, _ = hidden_states.shape
    q = attention.q_proj(hidden_states).view(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    k = attention.k_proj(hidden_states).view(batch, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    v = attention.v_proj(hidden_states).view(batch, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    if position_ids is None:
        position_ids = torch.arange(length, device=q.device).unsqueeze(0)
    geometry = layout.prepare(attention, position_ids, q.dtype)
    qr = q * geometry.cosine + rotate_half(q) * geometry.sine
    kr = k * geometry.cosine + rotate_half(k) * geometry.sine
    kr = repeat_kv(kr, attention.num_key_value_groups)
    v = repeat_kv(v, attention.num_key_value_groups)
    scale = 1. / math.sqrt(attention.head_dim)
    # Same GEMM shapes/order as the reference, for BF16 output parity.
    k = repeat_kv(k, attention.num_key_value_groups)
    raw_logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    rope_logits = torch.matmul(qr, kr.transpose(-1, -2)) * scale
    logits = torch.where(geometry.visual_instruction, raw_logits, rope_logits)
    logits.masked_fill_(~geometry.safe_allowed, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    output = torch.matmul(probabilities, v) * geometry.valid_query
    output = output.transpose(1, 2).reshape(batch, length, attention.hidden_size)
    return attention.o_proj(output), (probabilities if output_attentions else None), None


@contextmanager
def shared_ddia(model):
    """Temporarily share one layout across decoder layers, without changing weights."""
    layout, saved = _Layout(), []
    try:
        for module in model.modules():
            if isinstance(module, DDIALlamaAttention) and module.use_ddia:
                saved.append((module, module.__dict__.get('forward')))
                def forward(*args, _attention=module, **kwargs):
                    return _attention_forward(_attention, layout, *args, **kwargs)
                module.forward = forward
        yield
    finally:
        for module, previous in saved:
            if previous is None:
                del module.forward
            else:
                module.forward = previous


@torch.inference_mode()
def predict_latest(model, features, instruction):
    """Same full-window forward and fresh Gumbel draw; return the latest frame."""
    if model.training or features.shape != (1, 11, 106, 256):
        raise ValueError('Fast window requires eval mode and [1,11,106,256]')
    with shared_ddia(model.language_model):
        output = model.predict({'visual_feature': features, 'instruction': [instruction]},
                               compute_auxiliary=False, token_selection_mode='gumbel')
    return dict(trajectory=output['trajectory'][:, -1:], selected_count=output['selected_count'])
