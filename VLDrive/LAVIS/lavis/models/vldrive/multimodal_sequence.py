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
