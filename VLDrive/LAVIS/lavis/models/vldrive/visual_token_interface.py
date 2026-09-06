from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VisualTokenBatch:
    """Output contract for the future CCDP + MEFA connector."""

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
        elif self.visual_attention_mask.ndim in (2, 3):
            mask = self.visual_attention_mask.reshape(tokens.shape[:2]).long()
        else:
            raise ValueError("visual_attention_mask must have shape [B,V] or [B,T,M]")
        if mask.shape != tokens.shape[:2]:
            raise ValueError("visual_attention_mask does not match visual_tokens")
        return VisualTokenBatch(tokens, mask)
