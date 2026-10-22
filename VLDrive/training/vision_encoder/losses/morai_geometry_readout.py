"""Opt-in final-layer geometry tuning on frozen LAV hidden features.

No production model or loss is changed by importing this module. The caller
must supply detached features and the corresponding frozen eight-channel
prediction. Only channels 1:7 receive a learned residual; confidence (0) and
speed (7) are copied exactly, including when an optimizer uses weight decay.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .morai_bbox_corner import _instance_frame_mean, corner_bbox_loss


class GeometryReadoutDelta(nn.Module):
    def __init__(self, in_features=64):
        super().__init__()
        self.delta = nn.Linear(in_features, 6)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, hidden, baseline):
        if hidden.shape[:-1] != baseline.shape[:-1] or baseline.shape[-1] != 8:
            raise ValueError('Expected aligned hidden features and eight LAV channels')
        base = baseline.detach()
        geometry = base[..., 1:7] + self.delta(hidden.detach())
        return torch.cat((base[..., :1], geometry, base[..., 7:]), dim=-1)

    @torch.no_grad()
    def fold_into(self, linear):
        """Return final-layer tensors; do not mutate the supplied checkpoint."""
        if linear.weight.shape != (8, self.delta.in_features) or linear.bias is None:
            raise ValueError('Expected a biased eight-channel final linear layer')
        weight, bias = linear.weight.detach().clone(), linear.bias.detach().clone()
        weight[1:7] += self.delta.weight.to(weight)
        bias[1:7] += self.delta.bias.to(bias)
        return weight, bias


def geometry_readout_loss(prediction, target, instance_ids, baseline, eligible):
    """Instance-balanced geometry objective with a weak output-change penalty.

Eligible selects GT objects, never predicted classes. Center/size terms are
in metres. The corner term handles rectangle rotational equivalence; native
heading components anchor a nonzero direction. Deferred objects and other
confident proposals participate only in the baseline-preservation penalty.
"""
    corner, _ = corner_bbox_loss(prediction, target, instance_ids, eligible)
    assigned = (instance_ids >= 0) & eligible
    zero = prediction.reshape(-1)[:0].sum()
    if assigned.any():
        p, t = prediction[assigned], target[assigned]
        center = F.smooth_l1_loss(p[:, 1:3] * 3.5, t[:, 1:3] * 3.5,
                                 beta=.2, reduction='none').mean(-1)
        scale = p.new_tensor([7., 4.])
        size = F.smooth_l1_loss(p[:, 5:7] * scale, t[:, 5:7] * scale,
                               beta=.2, reduction='none').mean(-1)
        heading = F.smooth_l1_loss(p[:, 3:5], t[:, 3:5], beta=.2,
                                  reduction='none').mean(-1)
        terms = _instance_frame_mean(torch.stack((center, size, heading), -1),
                                     t[:, 0], assigned, instance_ids)
        center, size, heading = terms.unbind()
    else:
        center = size = heading = zero
    anchor_mask = (baseline[..., 0] >= 0) | (instance_ids >= 0)
    anchor = (F.mse_loss(prediction[..., 1:7][anchor_mask],
                         baseline.detach()[..., 1:7][anchor_mask])
              if anchor_mask.any() else zero)
    loss = .5 * center + .25 * size + .25 * corner + .1 * heading + .1 * anchor
    return loss, dict(center_m_loss=center, size_m_loss=size,
                      corner_loss=corner, heading_loss=heading, anchor_loss=anchor)
