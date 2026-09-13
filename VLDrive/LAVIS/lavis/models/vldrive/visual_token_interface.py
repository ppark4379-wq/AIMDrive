from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VisualTokenBatch:
    """Chronological CCDP/MEFA tokens with explicit per-token validity."""

    visual_tokens: torch.Tensor
    visual_attention_mask: Optional[torch.Tensor] = None

    def flattened(self) -> "VisualTokenBatch":
        if self.visual_tokens.ndim == 3:
            tokens = self.visual_tokens
        elif self.visual_tokens.ndim == 4:
            batch, time, retained, dim = self.visual_tokens.shape
            tokens = self.visual_tokens.reshape(batch, time * retained, dim)
        else:
            raise ValueError("visual_tokens must have shape [B,V,D] or [B,T,M,D]")

        if self.visual_attention_mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device)
        elif tuple(self.visual_attention_mask.shape) == tuple(self.visual_tokens.shape[:-1]):
            if self.visual_attention_mask.device != tokens.device:
                raise ValueError("visual token mask is on a different device")
            if not bool(((self.visual_attention_mask == 0) | (self.visual_attention_mask == 1)).all()):
                raise ValueError("visual_attention_mask must be finite binary 0/1")
            mask = self.visual_attention_mask.reshape(tokens.shape[:2]).long()
        else:
            raise ValueError("visual_attention_mask must have shape [B,V] or [B,T,M]")
        if mask.shape != tokens.shape[:2]:
            raise ValueError("visual_attention_mask does not match visual_tokens")
        return VisualTokenBatch(tokens, mask)


def last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Locate the actual last valid position, allowing holes inside a sequence."""
    if mask.ndim < 2 or mask.shape[-1] == 0:
        raise ValueError("Visual sequences must be nonempty")
    positions = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask.bool(), -1).amax(dim=-1)
    if bool((last < 0).any()):
        raise ValueError("Every sample/frame needs at least one valid visual token")
    return last
