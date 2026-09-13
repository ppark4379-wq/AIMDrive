"""
Stage 6D-3 route context encoder.

Input
-----
ego_frame_route_xy: [B,T,P,2], meters
    +x = ego forward
    +y = ego left

Output
------
route_context: [B,T,D]

This flatten-MLP encoder deliberately preserves point order and is only an
integration baseline for Route-Conditioned CCDP.  It is NOT a frozen final
route-token architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FlattenRouteContextEncoder(nn.Module):
    def __init__(
        self,
        point_count: int = 41,
        context_dim: int = 128,
        hidden_dim: int = 256,
        normalization_m: float = 40.0,
    ) -> None:
        super().__init__()

        if point_count < 2:
            raise ValueError("point_count must be >= 2")
        if context_dim < 1 or hidden_dim < 1:
            raise ValueError("encoder dimensions must be positive")
        if normalization_m <= 0.0:
            raise ValueError("normalization_m must be positive")

        self.point_count = int(point_count)
        self.context_dim = int(context_dim)
        self.normalization_m = float(normalization_m)

        flat_dim = self.point_count * 2

        self.network = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self,
        ego_frame_route_xy: torch.Tensor,
    ) -> torch.Tensor:
        if ego_frame_route_xy.ndim != 4:
            raise ValueError(
                "ego_frame_route_xy must be [B,T,P,2]"
            )
        if (
            int(ego_frame_route_xy.shape[2])
            != self.point_count
        ):
            raise ValueError(
                "expected P={}, got {}".format(
                    self.point_count,
                    int(ego_frame_route_xy.shape[2]),
                )
            )
        if int(ego_frame_route_xy.shape[3]) != 2:
            raise ValueError("last route dimension must be XY")

        normalized = (
            ego_frame_route_xy
            / self.normalization_m
        )

        flat = normalized.reshape(
            int(normalized.shape[0]),
            int(normalized.shape[1]),
            self.point_count * 2,
        )

        return self.network(flat)
