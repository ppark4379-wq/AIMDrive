"""Small current-frame MORAI baseline, with optional ordered metric route.

No instruction, pruning or temporal module is required. A temporal feature
window may be supplied but only its LAST frame is used. This gives later
MEFA experiments an explicit current-only baseline. Output is [B,H,2] metres
in the dataset's +x forward/+y left frame; no physical constraint is claimed.
"""
import torch
from torch import nn


class TrajectoryBaseline(nn.Module):
    def __init__(self, *, use_route=True, horizon_points=5):
        super().__init__()
        if horizon_points<1:raise ValueError("horizon_points must be positive")
        self.use_route=bool(use_route);self.horizon_points=int(horizon_points)
        self.visual_projection=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,32),nn.GELU())
        self.visual_readout=nn.Sequential(nn.Linear(106*32,256),nn.GELU())
        self.route_encoder=nn.Sequential(nn.Linear(41*2,128),nn.GELU()) if self.use_route else None
        self.head=nn.Sequential(nn.Linear(384 if self.use_route else 256,256),nn.GELU(),nn.Linear(256,self.horizon_points*2))

    def forward(self, visual_feature, route=None):
        if visual_feature.ndim!=4 or tuple(visual_feature.shape[2:])!=(106,256) or visual_feature.shape[1]<1:
            raise ValueError("visual_feature must be [B,T,106,256] with nonempty T")
        batch=visual_feature.shape[0]
        visual=self.visual_readout(self.visual_projection(visual_feature[:,-1]).reshape(batch,-1))
        if self.use_route:
            if route is None or tuple(route.shape)!=(batch,visual_feature.shape[1],41,2):
                raise ValueError("route must be [B,T,41,2] for route baseline")
            context=self.route_encoder((route[:,-1]/40.0).reshape(batch,82))
            visual=torch.cat((visual,context),dim=-1)
        return self.head(visual).reshape(batch,self.horizon_points,2)
