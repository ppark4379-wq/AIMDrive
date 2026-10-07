import torch
from torch import nn


# Stage 2 uses top-level models/; the standalone Situation trainer uses VLDrive/.
if __package__.startswith('VLDrive.'):
    from ...instruction.zone.encoding import ZONE_NAMES, ZONE_ID_SCHEMA as ZONE_SCHEMA
else:
    from instruction.zone.encoding import ZONE_NAMES, ZONE_ID_SCHEMA as ZONE_SCHEMA

class ZoneEmbedding(nn.Module):
    def __init__(self, output_dim=256, embedding_dim=32):
        super().__init__()
        for name, value in (('output_dim', output_dim), ('embedding_dim', embedding_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.output_dim = output_dim
        self.zone_embedding = nn.Embedding(len(ZONE_NAMES), embedding_dim)
        self.projection = nn.Linear(embedding_dim, output_dim)

    def forward(self, zone_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(zone_ids, torch.Tensor):
            raise TypeError('zone_ids must be an integer torch.Tensor')
        if zone_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError('zone_ids must use torch.int32 or torch.int64')
        if zone_ids.ndim not in (1, 2) or zone_ids.numel() == 0:
            raise ValueError('zone_ids must have nonempty shape [B] or [B, T]')
        if zone_ids.device != self.zone_embedding.weight.device:
            raise ValueError('zone_ids and ZoneEmbedding must be on the same device')
        if bool(((zone_ids < 0) | (zone_ids >= len(ZONE_NAMES))).any()):
            raise ValueError('zone_ids must be in 0..5; unknown zones need explicit handling')
        ids = zone_ids.long()
        return self.projection(self.zone_embedding(ids))
