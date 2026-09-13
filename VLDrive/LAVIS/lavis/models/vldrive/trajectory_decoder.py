"""Candidate metric decoder: route progress + learned XY correction.

Coordinates are metres, +X forward/+Y left. Future speeds (m/s) use right-end
integration over the five intervals. Progress beyond the finite route is
extrapolated along its last tangent, retaining speed gradients. The path starts
at the ego origin and joins the first route point before following the route;
the projected first route point is not assumed to be the current vehicle pose.
This is a
learnable planning representation, not a collision or kinematic guarantee.
"""
import torch
from torch import nn
from torch.nn import functional as F


def sample_route(route, distance):
    if route.shape[:-2] != distance.shape[:-1] or route.shape[-2:] != (41, 2):
        raise ValueError('Expected matching leading dimensions and metric route [41,2]')
    segments = route[..., 1:, :] - route[..., :-1, :]
    lengths = segments.norm(dim=-1)
    if not bool(torch.isfinite(route).all()) or bool((lengths < 1e-5).any()):
        raise ValueError('Route contains nonfinite or duplicate adjacent points')
    cumulative = torch.cat((torch.zeros_like(lengths[..., :1]), lengths.cumsum(-1)), -1)
    index = torch.searchsorted(cumulative.contiguous(), distance.contiguous(), right=True) - 1
    index = index.clamp(0, lengths.shape[-1] - 1)
    xy_index = index.unsqueeze(-1).expand(*index.shape, 2)
    base = torch.gather(route[..., :-1, :], -2, xy_index)
    tangent = torch.gather(segments / lengths.unsqueeze(-1), -2, xy_index)
    start = torch.gather(cumulative[..., :-1], -1, index)
    return base + tangent * (distance - start).unsqueeze(-1)


def sample_ego_route(route, distance):
    """Arc-length sample the ego-to-route join followed by the unchanged route.

    A straight join is an explicit geometric prior, not a feasible steering
    manoeuvre. Learned residuals remain responsible for departing from it.
    """
    first = route[..., 0, :]
    join_length = first.norm(dim=-1, keepdim=True)
    on_route = sample_route(route, (distance - join_length).clamp_min(0))
    on_join = first.unsqueeze(-2) * (distance / join_length.clamp_min(1e-5)).unsqueeze(-1)
    return torch.where((distance < join_length).unsqueeze(-1), on_join, on_route)


class MetricTrajectoryDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 256), nn.GELU())
        self.speed_head = nn.Linear(256, 5)
        self.residual_head = nn.Linear(256, 10)
        self.register_buffer('horizons_s', torch.tensor([.5, 1., 1.5, 2., 2.5]))

    def forward(self, context, route):
        context = self.context(context.float())
        speed = F.softplus(self.speed_head(context).float())
        dt = torch.diff(self.horizons_s, prepend=self.horizons_s.new_zeros(1))
        progress = (speed * dt).cumsum(-1)
        residual = self.residual_head(context).float().reshape(*context.shape[:-1], 5, 2)
        trajectory = sample_ego_route(route.float(), progress) + residual
        return dict(trajectory=trajectory, speed=speed, route_progress_m=progress,
                    metric_residual_xy=residual)
