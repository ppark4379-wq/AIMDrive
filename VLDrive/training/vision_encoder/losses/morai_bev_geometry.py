"""Experimental bbox branch on frozen, precompression BEV+speed features.

Not wired into production. Confidence/speed are copied bit-exactly. Bounds
limit residual geometry, not confidence filtering or planner safety margins.
"""
import math
import torch
from torch import nn


class BevGeometryDelta(nn.Module):
    def __init__(self, in_features=288, width=128):
        super().__init__()
        self.in_features = in_features
        self.delta = nn.Sequential(nn.LayerNorm(in_features), nn.Linear(in_features, width),
                                   nn.SiLU(), nn.Linear(width, 5))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, features, baseline):
        if (features.shape[:-1] != baseline.shape[:-1] or features.shape[-1] != self.in_features
                or baseline.shape[-1] != 8):
            raise ValueError('Expected aligned precompression features and eight detection channels')
        base = baseline.detach()
        residual = self.delta(features.detach())
        center = base[..., 1:3] + (1.5 / 3.5) * torch.tanh(residual[..., :2])
        angle = (math.pi / 2) * torch.tanh(residual[..., 2:3])
        c, s = torch.cos(angle), torch.sin(angle)
        # Rotate the native two-component heading; preserve its norm.
        heading = torch.cat((c*base[..., 3:4]-s*base[..., 4:5],
                             s*base[..., 3:4]+c*base[..., 4:5]), -1)
        sizes = base[..., 5:7] * torch.exp(math.log(2) * torch.tanh(residual[..., 3:5]))
        return torch.cat((base[..., :1], center, heading, sizes, base[..., 7:]), -1)
