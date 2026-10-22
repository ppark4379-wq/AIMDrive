"""Experimental proposal-aligned BEV geometry refinement; no GT input at inference."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def pool_proposal_bev(bev, baseline, indices, grid_size=5, margin_m=1.):
    """Sample a metric, heading-aligned ROI from one 50x50 right/forward BEV.

    bev: [2500,256], baseline: [N,8], indices: [N]. Positive forward is
    decreasing raster row. Returns detached [N,25,256] and support mask.
    """
    if bev.shape != (2500,256) or baseline.shape != (len(indices),8):
        raise ValueError('Expected one 50x50 BEV and aligned detection queries')
    if ((indices<0)|(indices>=2500)).any():raise ValueError('Query out of bounds')
    if not torch.isfinite(baseline).all():raise ValueError('Nonfinite geometry')
    with torch.no_grad():
        b=baseline.detach();index=indices.to(b.device)
        center=torch.stack((index%50-24.5+b[:,1]*3.5,29.5-index//50+b[:,2]*3.5),-1)
        direction=torch.stack((b[:,3],-b[:,4]),-1)
        norm=direction.norm(dim=-1);direction=F.normalize(direction,dim=-1,eps=1e-8)
        lateral=torch.stack((-direction[:,1],direction[:,0]),-1)
        axis=torch.linspace(-1,1,grid_size,device=b.device,dtype=b.dtype)
        u,v=torch.meshgrid(axis,axis,indexing='ij')
        half=torch.stack(((b[:,5]*7).clamp(.1,50)/2+margin_m,(b[:,6]*4).clamp(.1,50)/2+margin_m),-1)
        coords=center[:,None,:]+u.flatten()[None,:,None]*half[:,0,None,None]*direction[:,None,:]+v.flatten()[None,:,None]*half[:,1,None,None]*lateral[:,None,:]
        pixels=torch.stack((coords[...,0]+24.5,29.5-coords[...,1]),-1)
        valid=((pixels>=0)&(pixels<=49)).all(-1)&((norm>1e-8)&(b[:,5]>0)&(b[:,6]>0))[:,None]
        grid=(pixels/49*2-1)[None]
        featuremap=bev.detach().T.reshape(1,256,50,50)
        roi=F.grid_sample(featuremap,grid,align_corners=True,padding_mode='zeros',mode='bilinear')[0].permute(1,2,0)
        return roi*valid[...,None],valid


class SpatialBevGeometryDelta(nn.Module):
    def __init__(self,mode='spatial'):
        super().__init__()
        if mode not in ['spatial','repeat_center']:raise ValueError('Unknown ablation mode')
        self.mode=mode
        self.roi=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,16),nn.SiLU())
        self.query=nn.LayerNorm(288)
        self.delta=nn.Sequential(nn.Linear(288+25*16+25+6,128),nn.SiLU(),nn.Linear(128,5))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)

    def forward(self,features,baseline):
        if features.shape[:-1]!=baseline.shape[:-1] or features.shape[-1]!=288+25*256+25 or baseline.shape[-1]!=8:
            raise ValueError('Expected query, 5x5 ROI features, support mask and bbox')
        f=features.detach();b=baseline.detach();query=f[...,:288]
        roi=f[...,288:288+25*256].reshape(*f.shape[:-1],25,256);mask=f[...,-25:]
        if self.mode=='repeat_center':
            roi=query[...,:256].half().float().unsqueeze(-2).expand_as(roi)
        local=self.roi(roi)*mask[...,None]
        r=self.delta(torch.cat((self.query(query),local.flatten(-2),mask,b[...,1:7]),-1))
        center=b[...,1:3]+(1.5/3.5)*r[...,:2].tanh()
        angle=(math.pi/2)*r[...,2:3].tanh();c,s=angle.cos(),angle.sin()
        heading=torch.cat((c*b[...,3:4]-s*b[...,4:5],s*b[...,3:4]+c*b[...,4:5]),-1)
        size=b[...,5:7]*torch.exp(math.log(2)*r[...,3:5].tanh())
        return torch.cat((b[...,:1],center,heading,size,b[...,7:]),-1)
