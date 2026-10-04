"""Direction queries attend to two numeric tokens independently per frame."""
import torch
from torch import nn


class NavigationCrossAttention(nn.Module):
    def __init__(self, hidden_size, width=256, heads=4, gate_init=.01):
        super().__init__()
        if width < 1 or heads < 1 or width % heads or not 0 < gate_init <= 1:
            raise ValueError('Invalid path attention dimensions or residual gate')
        self.query_projection = nn.Linear(hidden_size, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output_projection = nn.Linear(width, hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, text, path, text_mask, path_valid):
        if path.ndim != 4 or path.shape[2] != 2 or text.ndim != 3:
            raise ValueError('Expected text[B,L,D] and path[B,T,2,d]')
        batch, frames, _, width = path.shape
        if text.shape[0] != batch or text_mask.shape != text.shape[:2] or path_valid.shape != (batch, frames):
            raise ValueError('Text, path and mask dimensions differ')
        length = text.shape[1]
        dtype = self.query_projection.weight.dtype
        common = text[:, None].expand(-1, frames, -1, -1)
        query = self.query_projection(text.to(dtype))[:, None].expand(-1, frames, -1, -1)
        values = path.to(dtype).reshape(batch * frames, 2, width)
        fused, _ = self.attention(query.reshape(batch * frames, length, width), values, values,
                                  need_weights=False)
        delta = self.output_projection(fused).reshape(batch, frames, length, -1)
        # Do not mask both K/V tokens to -inf: all-invalid frames must not NaN.
        delta = delta * path_valid[..., None, None] * text_mask[:, None, :, None]
        return common + (self.gate * delta).to(text.dtype)
