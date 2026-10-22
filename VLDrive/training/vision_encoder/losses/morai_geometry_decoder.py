"""Isolated bbox branch copied from P1's final decoder camera-fusion block.

The base encoder/decoder, self-attention, confidence, speed and signal outputs
remain on the original path. This module is experimental and not auto-loaded
by the live model. Cache inputs contain no GT or actor/model identifiers.
"""
from copy import deepcopy
import torch
from torch import nn


class GeometryDecoderBranch(nn.Module):
    def __init__(self, layer, decoder_norm, object_head, mode='fusion'):
        super().__init__()
        if layer.normalize_before:
            raise ValueError('Only the P1 post-norm decoder is supported')
        if mode not in ('fusion', 'head_only'):
            raise ValueError('Unknown geometry branch mode')
        self.mode = mode
        for name in ('multihead_attn', 'norm2', 'linear1', 'linear2', 'norm3', 'activation'):
            setattr(self, name, deepcopy(getattr(layer, name)))
        self.decoder_norm = deepcopy(decoder_norm)
        self.head = deepcopy(object_head)
        self.requires_grad_(mode == 'fusion')
        self.head.requires_grad_(True)
        self.eval()

    def train(self, mode=True):
        # Frozen cached inputs and baseline were produced in eval mode.
        # Gradients are enabled independently; dropout must remain disabled.
        return super().train(False)

    def forward(self, residual, query, memory, velocity, baseline):
        """Batch-first tensors; residual is the final layer's norm1 output.

        Query is that exact residual plus positional embedding. Memory is the
        frozen RGB encoder output; this is the last camera cross-attention.
        """
        if residual.shape != query.shape or baseline.shape[:-1] != query.shape[:-1]:
            raise ValueError('Mismatched query/residual/baseline shapes')
        fused = self.multihead_attn(query.transpose(0, 1), memory.transpose(0, 1),
                                   memory.transpose(0, 1))[0].transpose(0, 1)
        x = self.norm2(residual + fused)
        x = self.norm3(x + self.linear2(self.activation(self.linear1(x))))
        x = self.decoder_norm(x)
        pred = self.head(torch.cat((x, velocity), dim=-1))
        base = baseline.detach()
        return torch.cat((base[..., :1], pred[..., 1:7], base[..., 7:]), dim=-1)
