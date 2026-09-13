import math
from typing import Optional, Tuple

import torch
from torch import nn

from .multimodal_sequence import INSTRUCTION, PADDING, VISUAL


def build_ddia_attention_mask(modality_ids: torch.Tensor) -> torch.Tensor:
    """Return [B,Q,K] allowed-key mask implementing VLDrive Eq. (9)."""
    if modality_ids.ndim != 2:
        raise ValueError("modality_ids must have shape [B,S]")
    query = modality_ids.unsqueeze(2)
    key = modality_ids.unsqueeze(1)
    sequence_length = modality_ids.shape[1]
    query_index = torch.arange(sequence_length, device=modality_ids.device).view(1, -1, 1)
    key_index = torch.arange(sequence_length, device=modality_ids.device).view(1, 1, -1)

    instruction_to_instruction = (query == INSTRUCTION) & (key == INSTRUCTION)
    visual_to_instruction = (query == VISUAL) & (key == INSTRUCTION)
    # Paper notation {V | < j}: a visual token does not attend to itself.
    visual_to_earlier_visual = (query == VISUAL) & (key == VISUAL) & (key_index < query_index)
    allowed = instruction_to_instruction | visual_to_instruction | visual_to_earlier_visual
    return allowed & (query != PADDING) & (key != PADDING)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, rope_theta: float = 10000.0) -> torch.Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frequency = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    angles = position_ids.float().unsqueeze(-1) * frequency
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1).to(x.dtype)
    return x * angles.cos() + rotate_half(x) * angles.sin()


def repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return hidden_states
    batch, heads, length, dim = hidden_states.shape
    return hidden_states[:, :, None].expand(batch, heads, repetitions, length, dim).reshape(
        batch, heads * repetitions, length, dim
    )


def compose_ddia_logits(
    query_raw: torch.Tensor,
    key_raw: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    modality_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose raw/rotated blocks before one shared softmax."""
    scale = 1.0 / math.sqrt(query_raw.shape[-1])
    raw_logits = torch.matmul(query_raw, key_raw.transpose(-1, -2)) * scale
    rope_logits = torch.matmul(query_rope, key_rope.transpose(-1, -2)) * scale
    visual_instruction = (modality_ids[:, None, :, None] == VISUAL) & (
        modality_ids[:, None, None, :] == INSTRUCTION
    )
    logits = torch.where(visual_instruction, raw_logits, rope_logits)
    allowed = build_ddia_attention_mask(modality_ids).unsqueeze(1)
    return logits, allowed


class DDIALlamaAttention(nn.Module):
    """LLaMA/GQA attention using VLDrive's distance-decoupled logits."""

    def __init__(self, original_attention: nn.Module, config, use_ddia: bool = True):
        super().__init__()
        # Keep the ablation delegate without registering a second copy of all
        # pretrained projection parameters in state_dict.
        object.__setattr__(self, "original_attention", original_attention)
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.use_ddia = use_ddia
        self.modality_ids: Optional[torch.Tensor] = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, **kwargs):
        if not self.use_ddia:
            return self.original_attention(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache, **kwargs
            )
        if past_key_value is not None:
            raise NotImplementedError("DDIA cached decoding is not supported")
        if self.modality_ids is None:
            raise ValueError("modality_ids must be set before a DDIA forward pass")

        batch, length, _ = hidden_states.shape
        query_raw = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key_raw = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0).expand(batch, -1)
        query_rope = apply_rope(query_raw, position_ids, self.rope_theta)
        key_rope = apply_rope(key_raw, position_ids, self.rope_theta)
        key_raw = repeat_kv(key_raw, self.num_key_value_groups)
        key_rope = repeat_kv(key_rope, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        logits, allowed = compose_ddia_logits(query_raw, key_raw, query_rope, key_rope, self.modality_ids)
        # Avoid NaNs for padding queries; their output is zeroed immediately after attention.
        no_key = ~allowed.any(dim=-1, keepdim=True)
        safe_allowed = allowed | (no_key & torch.eye(length, device=allowed.device, dtype=torch.bool)[None, None])
        logits = logits.masked_fill(~safe_allowed, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits.float(), dim=-1).to(query_raw.dtype)
        probabilities = nn.functional.dropout(probabilities, p=self.attention_dropout, training=self.training)
        output = torch.matmul(probabilities, value)
        valid_query = (self.modality_ids != PADDING)[:, None, :, None]
        output = (output * valid_query).transpose(1, 2).reshape(batch, length, self.hidden_size)
        output = self.o_proj(output)
        return output, (probabilities if output_attentions else None), (None if use_cache else None)


def install_ddia_attention(model: nn.Module, use_ddia: bool = True):
    """Replace HF LLaMA layer attention while retaining pretrained projections."""
    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("Expected a Hugging Face LLaMA-family model with model.layers")
    for layer in layers:
        if not isinstance(layer.self_attn, DDIALlamaAttention):
            layer.self_attn = DDIALlamaAttention(layer.self_attn, model.config, use_ddia)
        else:
            layer.self_attn.use_ddia = use_ddia
    return model


def set_ddia_modality_ids(model: nn.Module, modality_ids: torch.Tensor) -> None:
    decoder = getattr(model, "model", model)
    for layer in decoder.layers:
        if isinstance(layer.self_attn, DDIALlamaAttention):
            layer.self_attn.modality_ids = modality_ids
