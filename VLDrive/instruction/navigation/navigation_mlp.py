"""Eight frame-local numeric features -> a residual navigation embedding.

No attention, event selection, text generation, or temporal mixing happens here.
Geometry and feature normalization are shared with the cross-attention baseline.
"""
import torch
from torch import nn


class NavigationMLP(nn.Module):
    def __init__(self, hidden_size, width=256, gate_init=.01):
        super().__init__()
        if hidden_size < 1 or width < 1 or not 0 < gate_init <= 1:
            raise ValueError('Invalid navigation MLP dimensions or residual gate')
        self.network = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, hidden_size))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, text, features, text_mask):
        if set(features) != {'ego_relation', 'path_context', 'path_valid'}:
            raise ValueError('Path input accepts only ego_relation, path_context, path_valid')
        ego, path, valid = (features[k] for k in ('ego_relation', 'path_context', 'path_valid'))
        if (text.ndim != 3 or ego.ndim != 3 or ego.shape[-1] != 3
                or path.shape != (*ego.shape[:2], 5) or valid.shape != ego.shape[:2]
                or valid.dtype != torch.bool or ego.shape[0] != text.shape[0]
                or text_mask.shape != text.shape[:2]):
            raise ValueError('Expected text[B,L,D], ego[B,T,3], path[B,T,5], bool valid[B,T]')
        if not torch.isfinite(ego).all() or not torch.isfinite(path).all():
            raise ValueError('Path inputs must be finite, including invalid-frame placeholders')
        numeric = torch.cat((ego, path), -1).to(self.network[0].weight.dtype)
        delta = self.network(numeric)[:, :, None, :]
        delta = delta * valid[..., None, None] * text_mask[:, None, :, None].bool()
        # Broadcast one numeric representation over the common sentence's tokens.
        # The existing frame-aware decoder consumes exactly the same layout as CA.
        return text[:, None] + (self.gate * delta).to(text.dtype)
