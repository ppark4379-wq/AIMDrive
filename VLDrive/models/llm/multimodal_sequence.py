from dataclasses import dataclass

import torch

INSTRUCTION = 0
VISUAL = 1
PADDING = -1


@dataclass
class MultimodalSequence:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    modality_ids: torch.Tensor
    visual_start_index: int
    frame_ids: torch.Tensor = None
    frame_readout_indices: torch.Tensor = None


def build_multimodal_sequence(
    instruction_embeddings: torch.Tensor,
    instruction_token_mask: torch.Tensor,
    visual_embeddings: torch.Tensor,
    visual_token_mask: torch.Tensor,
) -> MultimodalSequence:
    """Build [padded instruction, visual] sequences with explicit modalities.

    Keeping a fixed instruction block makes batching deterministic. Padding keys
    are excluded by DDIA; valid visual tokens follow in chronological order.
    """
    if instruction_embeddings.ndim != 3 or visual_embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [B,S,D]")
    if instruction_embeddings.shape[0] != visual_embeddings.shape[0]:
        raise ValueError("instruction and visual batch sizes differ")
    if instruction_embeddings.shape[2] != visual_embeddings.shape[2]:
        raise ValueError("instruction and visual hidden dimensions differ")

    instruction_token_mask = instruction_token_mask.bool()
    visual_token_mask = visual_token_mask.bool()
    attention_mask = torch.cat((instruction_token_mask, visual_token_mask), dim=1)
    modality_ids = torch.cat(
        (
            torch.full_like(instruction_token_mask, INSTRUCTION, dtype=torch.long),
            torch.full_like(visual_token_mask, VISUAL, dtype=torch.long),
        ),
        dim=1,
    )
    modality_ids = modality_ids.masked_fill(~attention_mask, PADDING)
    position_ids = attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
    return MultimodalSequence(
        inputs_embeds=torch.cat((instruction_embeddings, visual_embeddings), dim=1),
        attention_mask=attention_mask.long(),
        position_ids=position_ids,
        modality_ids=modality_ids,
        visual_start_index=instruction_embeddings.shape[1],
    )


def build_frame_multimodal_sequence(instruction_embeddings, instruction_token_mask,
                                    visual_embeddings, visual_token_mask):
    """Pack [navigation_i, optional Zone_i/Situation_i, visual_i] with frame-local DDIA."""
    from ..connector.visual_token_interface import last_valid_indices
    if instruction_embeddings.ndim != 4 or visual_embeddings.ndim != 4:
        raise ValueError('Frame embeddings must have shape [B,T,L/K,D]')
    b, t, length, d = instruction_embeddings.shape
    if visual_embeddings.shape[:2] != (b, t) or visual_embeddings.shape[-1] != d:
        raise ValueError('Navigation and visual frame dimensions differ')
    retained = visual_embeddings.shape[2]
    if instruction_token_mask.shape not in ((b, length), (b, t, length)) or visual_token_mask.shape != (b, t, retained):
        raise ValueError('Frame token mask shapes differ')
    text_mask = instruction_token_mask.bool()
    if text_mask.ndim == 2:
        text_mask = text_mask[:, None].expand(-1, t, -1)
    visual_mask = visual_token_mask.bool()
    if not text_mask.any(-1).all() or not visual_mask.any(-1).all():
        raise ValueError('Every frame requires navigation and visual tokens')
    mask = torch.cat((text_mask, visual_mask), -1).reshape(b, -1)
    modalities = torch.cat((torch.full_like(text_mask, INSTRUCTION, dtype=torch.long),
                            torch.full_like(visual_mask, VISUAL, dtype=torch.long)), -1).reshape(b, -1)
    frame_ids = torch.arange(t, device=visual_embeddings.device)[None, :, None].expand(b, t, length + retained).reshape(b, -1)
    indices = last_valid_indices(visual_mask) + length + torch.arange(t, device=visual_embeddings.device) * (length + retained)
    return MultimodalSequence(
        inputs_embeds=torch.cat((instruction_embeddings, visual_embeddings), 2).reshape(b, -1, d),
        attention_mask=mask.long(), position_ids=mask.long().cumsum(1).sub(1).clamp_min(0),
        modality_ids=modalities.masked_fill(~mask, PADDING), visual_start_index=length,
        frame_ids=frame_ids.masked_fill(~mask, -1), frame_readout_indices=indices)
