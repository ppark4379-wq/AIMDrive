import torch
from torch import nn

from .zone_embedding import ZoneEmbedding


class SituationHead(nn.Module):
    """Read per-frame VE features; tensors alone do not prove their source."""

    def __init__(self, class_counts, visual_dim=256, hidden_dim=256,
                 output_dim=2048, layers=2, heads=8, dropout=0.1):
        super().__init__()
        if (any(type(x) is not int or x < 1 for x in
                (visual_dim, hidden_dim, output_dim, layers, heads))
                or hidden_dim % heads):
            raise ValueError('Invalid SituationHead dimensions')
        if (not 0 <= dropout < 1 or not class_counts
                or any(type(c) is not int or c < 0 for c in class_counts.values())):
            raise ValueError('Invalid dropout or class counts')
        self.config = dict(class_counts=dict(class_counts), visual_dim=visual_dim,
                           hidden_dim=hidden_dim, output_dim=output_dim, layers=layers,
                           heads=heads, dropout=dropout)
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.zone_embedding = ZoneEmbedding(hidden_dim)
        self.time_projection = nn.Linear(1, hidden_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, 4 * hidden_dim,
                                           dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.embedding = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        self.classifiers = nn.ModuleDict({name: nn.Linear(output_dim, count)
                                         for name, count in class_counts.items() if count > 0})

    def forward(self, visual_tokens, visual_mask, timestamps, zone_ids=None,
                zone_embeddings=None):
        if visual_tokens.ndim != 4 or visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(f'Expected visual tokens [B,T,K,{self.visual_dim}]')
        b, t, k, _ = visual_tokens.shape
        if visual_mask.dtype != torch.bool or visual_mask.shape != (b, t, k):
            raise ValueError('visual_mask must be boolean [B,T,K]')
        if min(b, t, k) < 1 or not visual_tokens.is_floating_point():
            raise ValueError('Visual tokens must be a nonempty floating point tensor')
        if timestamps.shape != (b, t) or not timestamps.is_floating_point():
            raise ValueError('timestamps must be floating point [B,T]')
        if not torch.isfinite(timestamps).all() or not (timestamps[:, 1:] > timestamps[:, :-1]).all():
            raise ValueError('Use strictly increasing timestamps within each observed window')
        if (zone_ids is None) == (zone_embeddings is None):
            raise ValueError('Supply exactly one of zone_ids or zone_embeddings')
        if zone_ids is not None:
            if zone_ids.shape != (b, t):
                raise ValueError('zone_ids must be [B,T]')
            fz = self.zone_embedding(zone_ids)
        else:
            if zone_embeddings.shape != (b, t, self.hidden_dim):
                raise ValueError('zone_embeddings must be [B,T,hidden_dim]')
            if not torch.isfinite(zone_embeddings).all():
                raise ValueError('Nonfinite zone embeddings')
            fz = zone_embeddings
        if not visual_mask.any(dim=2).all():
            raise ValueError('Every observed frame must contain at least one valid token')
        if not torch.isfinite(visual_tokens[visual_mask]).all():
            raise ValueError('Nonfinite valid visual tokens')
        parameter_dtype = self.visual_projection.weight.dtype
        relative_time = (timestamps - timestamps[:, -1:]).to(parameter_dtype)
        temporal = self.time_projection(relative_time.unsqueeze(-1))

        fv = visual_tokens.masked_fill(~visual_mask.unsqueeze(-1), 0).to(parameter_dtype)
        fv = self.visual_projection(fv) + temporal.unsqueeze(2)
        fz = fz.to(parameter_dtype) + temporal
        sequence = torch.cat((self.query.expand(b, -1, -1), fz, fv.flatten(1, 2)), dim=1)
        padding = torch.cat((torch.zeros(b, 1 + t, device=fv.device, dtype=torch.bool),
                             ~visual_mask.flatten(1, 2)), dim=1)

        state = self.transformer(sequence, src_key_padding_mask=padding)[:, 0]
        fs = self.embedding(state)
        return {'situation_embedding': fs.unsqueeze(1),
                'logits': {name: head(fs) for name, head in self.classifiers.items()}}
