"""
Route-ready CCDP token sparsification for the VLDrive reproduction project.

Paper-derived core:
    L_i = MLP(F_i)                         [N, C/2]
    G_i = Avg(L_i)                         [1, C/2]
    S_i = Softmax(MLP([G_i; L_i]))         [N, 2]
    M_i = Gumbel-Softmax(S_i)[..., keep]   [N]
    L_prun_i = (R - mean_j M_ij)^2

Independent implementation choices are explicitly documented:
- optional route/ego context is projected additively into selector features;
- training uses hard straight-through Gumbel-Softmax;
- deterministic top-k and argmax policies are available for evaluation;
- L_prun is computed frame-wise, then averaged.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCDPTokenSparsifier(nn.Module):
    """Dynamic keep/drop selector for visual tokens."""

    def __init__(
        self,
        dim: int = 256,
        target_retention: float = 0.30,
        temperature: float = 1.0,
        route_context_dim: int = 0,
        ego_context_dim: int = 0,
    ) -> None:
        super().__init__()

        if dim <= 1 or dim % 2 != 0:
            raise ValueError("dim must be an even integer greater than 1")
        if not 0.0 < target_retention <= 1.0:
            raise ValueError("target_retention must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if route_context_dim < 0 or ego_context_dim < 0:
            raise ValueError("context dimensions cannot be negative")

        hidden = dim // 2
        selector_dim = hidden * 2

        # Paper Eq. (1): F_i -> L_i in C/2.
        self.local_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
        )

        # Paper Eq. (3). Exact MLP depth is not stated in the paper.
        self.score_mlp = nn.Sequential(
            nn.Linear(selector_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

        # Our route-ready extension. Additive conditioning preserves the
        # paper baseline and keeps the no-context checkpoint shape compatible.
        self.route_context_projection = (
            nn.Linear(route_context_dim, selector_dim, bias=False)
            if route_context_dim > 0
            else None
        )
        self.ego_context_projection = (
            nn.Linear(ego_context_dim, selector_dim, bias=False)
            if ego_context_dim > 0
            else None
        )

        self.dim = int(dim)
        self.target_retention = float(target_retention)
        self.temperature = float(temperature)
        self.route_context_dim = int(route_context_dim)
        self.ego_context_dim = int(ego_context_dim)

    @staticmethod
    def _flatten_context(
        context: Optional[torch.Tensor],
        leading_shape: Tuple[int, ...],
        expected_dim: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if expected_dim <= 0:
            raise ValueError(
                "%s was supplied, but its configured dimension is zero" % name
            )
        expected_shape = leading_shape + (expected_dim,)
        if tuple(context.shape) != expected_shape:
            raise ValueError(
                "%s must have shape %s, got %s"
                % (name, expected_shape, tuple(context.shape))
            )
        return context.reshape(-1, expected_dim)

    @staticmethod
    def _rescue_empty_frames(
        mask: torch.Tensor,
        keep_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Ensure every frame has at least one retained token."""
        empty = mask.sum(dim=-1).eq(0)
        if not bool(empty.any()):
            return mask

        best = keep_probability.argmax(dim=-1)
        rescue = torch.zeros_like(mask)
        rescue.scatter_(1, best.unsqueeze(1), 1.0)

        # Keep one token in the forward pass, but preserve the Gumbel surrogate
        # gradient. Otherwise an all-drop frame cannot learn from either the
        # pruning loss or the retained-query path.
        rescue_st = mask + (rescue - mask).detach()
        return torch.where(empty.unsqueeze(1), rescue_st, mask)

    def _select_mask(
        self,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        selection_mode: str,
        hard: bool,
    ) -> torch.Tensor:
        if selection_mode == "gumbel":
            one_hot = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=hard,
                dim=-1,
            )
            return one_hot[..., 1]

        if selection_mode == "argmax":
            return logits.argmax(dim=-1).eq(1).to(logits.dtype)

        if selection_mode == "topk":
            num_tokens = logits.shape[1]
            keep_count = max(
                1,
                min(
                    num_tokens,
                    int(round(self.target_retention * num_tokens)),
                ),
            )
            indices = probabilities[..., 1].topk(
                k=keep_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            mask = torch.zeros_like(probabilities[..., 1])
            mask.scatter_(1, indices, 1.0)
            return mask

        raise ValueError(
            "selection_mode must be one of: gumbel, argmax, topk"
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        route_context: Optional[torch.Tensor] = None,
        ego_context: Optional[torch.Tensor] = None,
        hard: bool = True,
        selection_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_tokens:
                [B, N, C] or [B, T, N, C].
            route_context:
                [B, D_r] or [B, T, D_r], matching leading dimensions.
            ego_context:
                [B, D_e] or [B, T, D_e], matching leading dimensions.
            hard:
                Straight-through hard Gumbel samples when mode is "gumbel".
            selection_mode:
                None -> gumbel in train, argmax in eval.
                Explicit options: gumbel, argmax, topk.
        """
        if visual_tokens.ndim not in (3, 4):
            raise ValueError(
                "visual_tokens must be [B,N,C] or [B,T,N,C], got %s"
                % (tuple(visual_tokens.shape),)
            )
        if visual_tokens.shape[-1] != self.dim:
            raise ValueError(
                "expected feature dim %d, got %d"
                % (self.dim, visual_tokens.shape[-1])
            )

        leading_shape = tuple(visual_tokens.shape[:-2])
        num_tokens = int(visual_tokens.shape[-2])
        flat = visual_tokens.reshape(-1, num_tokens, self.dim)

        local_feature = self.local_mlp(flat)
        global_feature = local_feature.mean(dim=1, keepdim=True)
        global_broadcast = global_feature.expand(-1, num_tokens, -1)

        selector_feature = torch.cat(
            [global_broadcast, local_feature],
            dim=-1,
        )

        route_flat = self._flatten_context(
            route_context,
            leading_shape,
            self.route_context_dim,
            "route_context",
        )
        if route_flat is not None:
            selector_feature = selector_feature + (
                self.route_context_projection(route_flat).unsqueeze(1)
            )

        ego_flat = self._flatten_context(
            ego_context,
            leading_shape,
            self.ego_context_dim,
            "ego_context",
        )
        if ego_flat is not None:
            selector_feature = selector_feature + (
                self.ego_context_projection(ego_flat).unsqueeze(1)
            )

        logits_flat = self.score_mlp(selector_feature)
        probabilities_flat = logits_flat.softmax(dim=-1)

        mode = selection_mode
        if mode is None:
            mode = "gumbel" if self.training else "argmax"

        mask_flat = self._select_mask(
            logits_flat,
            probabilities_flat,
            selection_mode=mode,
            hard=hard,
        )
        mask_flat = self._rescue_empty_frames(
            mask_flat,
            probabilities_flat[..., 1],
        )

        # Paper Eq. (12): each frame receives its own retention penalty.
        retention_per_frame_flat = mask_flat.mean(dim=-1)
        target = retention_per_frame_flat.new_tensor(
            self.target_retention
        )
        pruning_loss_per_frame_flat = (
            retention_per_frame_flat - target
        ).pow(2)
        pruning_loss = pruning_loss_per_frame_flat.mean()

        soft_retention_per_frame_flat = (
            probabilities_flat[..., 1].mean(dim=-1)
        )

        output_shape_token = leading_shape + (num_tokens,)
        output_shape_logits = leading_shape + (num_tokens, 2)
        output_shape_local = leading_shape + (
            num_tokens,
            local_feature.shape[-1],
        )
        output_shape_global = leading_shape + (
            1,
            global_feature.shape[-1],
        )

        mask = mask_flat.reshape(output_shape_token)
        probabilities = probabilities_flat.reshape(output_shape_logits)
        logits = logits_flat.reshape(output_shape_logits)

        return {
            "local_feature": local_feature.reshape(output_shape_local),
            "global_feature": global_feature.reshape(output_shape_global),
            "logits": logits,
            "probabilities": probabilities,
            "mask": mask,
            "retention_per_frame": retention_per_frame_flat.reshape(
                leading_shape
            ),
            "soft_retention_per_frame": (
                soft_retention_per_frame_flat.reshape(leading_shape)
            ),
            "pruning_loss_per_frame": (
                pruning_loss_per_frame_flat.reshape(leading_shape)
            ),
            "retention": retention_per_frame_flat.mean(),
            "soft_retention": soft_retention_per_frame_flat.mean(),
            "pruning_loss": pruning_loss,
        }
