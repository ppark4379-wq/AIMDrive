"""Explicit fixed-budget CCDP adaptation for matched train/inference token counts.

The reference independent implementation samples N binary keep/drop decisions.
That policy is retained as an ablation. Here the SAME route-conditioned scores
rank tokens, with Gumbel noise during training and deterministic ranking at eval.
The forward mask has exactly K tokens. A K-scaled softmax provides a straight-
through surrogate for ranking; this is an engineering adaptation, not a claim
about the unpublished authors' source or an exact derivative of hard top-k.
"""
import torch
from torch import nn

from .ccdp import CCDPTokenSparsifier


class FixedBudgetCCDP(CCDPTokenSparsifier):
    def __init__(self, original):
        # Reuse actual initialized selector modules and names, without extra RNG
        # draws or changing matched initialization between ablation arms.
        nn.Module.__init__(self)
        for name, module in original._modules.items():
            self.add_module(name, module)
        for name in ('dim', 'target_retention', 'temperature', 'route_context_dim', 'ego_context_dim'):
            setattr(self, name, getattr(original, name))

    def _select_mask(self, logits, probabilities, selection_mode, hard):
        if selection_mode not in ('gumbel', 'topk') or not hard:
            raise ValueError('Fixed-budget CCDP requires a hard gumbel/topk forward mask')
        count = max(1, min(logits.shape[1], round(self.target_retention * logits.shape[1])))
        # Ranking log-odds avoids two-class softmax saturation/ties under AMP.
        scores = logits[..., 1].float() - logits[..., 0].float()
        if selection_mode == 'gumbel':
            uniform = torch.rand_like(scores).clamp(1e-6, 1 - 1e-6)
            scores = scores - torch.log(-torch.log(uniform))
        indices = scores.topk(count, dim=-1, sorted=False).indices
        mask = torch.zeros_like(scores).scatter(1, indices, 1.)
        if selection_mode == 'gumbel':
            surrogate = (scores / self.temperature).softmax(-1) * count
            mask = mask + (surrogate - surrogate.detach())
        return mask
