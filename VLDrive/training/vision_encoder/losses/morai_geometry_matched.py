"""Experimental geometry-only tuning at frozen decoded detection queries.

Targets are created offline from training GT. Inference consumes only the
frozen visual features and predictions, never GT, model names or actor IDs.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .morai_bbox_corner import _corners, _instance_frame_mean


class GeometryMLPDelta(nn.Module):
    """Zero-initialized geometry residual; confidence and speed stay exact."""
    def __init__(self, in_features=64, width=32):
        super().__init__()
        self.delta = nn.Sequential(nn.Linear(in_features, width), nn.ReLU(),
                                   nn.Linear(width, 6))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, hidden, baseline):
        base = baseline.detach()
        return torch.cat((base[..., :1], base[..., 1:7] +
                          self.delta(hidden.detach()), base[..., 7:]), dim=-1)


def encode_query_target(box, cell, *, dtype=torch.float32):
    """Encode a right/forward metric rectangle at a chosen 50x50 query."""
    if not 0 <= cell < 2500:
        raise ValueError('Query index outside BEV')
    row, col = divmod(int(cell), 50)
    center = torch.as_tensor(box['center'], dtype=dtype)
    heading = torch.as_tensor(box['heading'], dtype=dtype)
    size = torch.tensor([box['length'], box['width']], dtype=dtype)
    if not (torch.isfinite(center).all() and torch.isfinite(heading).all()
            and torch.isfinite(size).all() and (size > 0).all()
            and heading.norm() > 0):
        raise ValueError('Invalid metric GT rectangle')
    heading = heading / heading.norm()
    target = torch.zeros(8, dtype=dtype)
    target[0] = 1.
    target[1:3] = (center - center.new_tensor([col - 24.5, 29.5 - row])) / 3.5
    target[3:5] = heading * heading.new_tensor([1., -1.])
    target[5:7] = size / size.new_tensor([7., 4.])
    return target


def matched_geometry_loss(prediction, target, instance_ids, baseline, anchor_weight=.1):
    """Metre corner/center loss on fixed baseline NMS queries matched to GT.

One target per matched detection. Mean objects within each frame, then mean
positive frames. Corner cyclic correspondence makes equivalent rectangle
orientations harmless. Sizes are regularized positive. Native heading is
weakly retained to avoid zero directions. Other proposals retain an anchor.
"""
    selected = instance_ids >= 0
    zero = prediction.reshape(-1)[:0].sum()
    center = corner = validity = heading_loss = zero
    if selected.any():
        p, t = prediction[selected], target[selected]
        if not torch.isfinite(p[:, 1:7]).all() or not torch.isfinite(t[:, :7]).all():
            raise ValueError('Nonfinite geometry')
        signs = p.new_tensor([1., -1.])
        ph, th = p[:, 3:5] * signs, t[:, 3:5] * signs
        ph = F.normalize(ph, dim=-1, eps=1e-3)
        th = F.normalize(th, dim=-1, eps=1e-3)
        scales = p.new_tensor([7., 4.])
        ps, ts = p[:, 5:7] * scales, t[:, 5:7] * scales
        shift = (p[:, 1:3] - t[:, 1:3]) * 3.5
        pc = _corners(ph, ps) + shift[:, None, :]
        tc = _corners(th, ts)
        alternatives = [torch.linalg.vector_norm(pc.roll(k, 1) - tc, dim=-1).mean(-1)
                        for k in range(4)]
        costs = torch.stack((torch.linalg.vector_norm(shift, dim=-1),
                             torch.stack(alternatives, -1).min(-1).values,
                             F.relu(.02 - ps).mean(-1),
                             F.smooth_l1_loss(p[:, 3:5], t[:, 3:5],
                                              beta=.2, reduction='none').mean(-1)), -1)
        center, corner, validity, heading_loss = _instance_frame_mean(
            costs, t[:, 0], selected, instance_ids).unbind()
    mask = (baseline[..., 0] >= 0) | selected
    anchor = (F.mse_loss(prediction[..., 1:7][mask], baseline.detach()[..., 1:7][mask])
              if mask.any() else zero)
    loss = corner + .5 * center + 2. * validity + .02 * heading_loss + anchor_weight * anchor
    return loss, dict(corner_m=corner, center_m=center, size_validity_m=validity,
                      heading=heading_loss, anchor=anchor)
