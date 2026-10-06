"""Two typed tokens per frame, shared across every frame and batch sample."""
import torch
from torch import nn


class PathTokenizer(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.ego = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, width))
        self.path = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, width))
        self.token_type = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.token_type, std=.02)

    def forward(self, features):
        if set(features) != {'ego_relation', 'path_context', 'path_valid'}:
            raise ValueError('Path input accepts only ego_relation, path_context, path_valid')
        ego, path, valid = (features[k] for k in ('ego_relation', 'path_context', 'path_valid'))
        if (ego.ndim != 3 or ego.shape[-1] != 3 or path.shape != (*ego.shape[:2], 5)
                or valid.shape != ego.shape[:2] or valid.dtype != torch.bool):
            raise ValueError('Expected path shapes [B,T,3], [B,T,5], bool[B,T]')
        if not torch.isfinite(ego).all() or not torch.isfinite(path).all():
            raise ValueError('Path inputs must be finite, including invalid-frame placeholders')
        dtype = self.token_type.dtype
        tokens = torch.stack((self.ego(ego.to(dtype)), self.path(path.to(dtype))), dim=-2)
        return (tokens + self.token_type) * valid[..., None, None]
