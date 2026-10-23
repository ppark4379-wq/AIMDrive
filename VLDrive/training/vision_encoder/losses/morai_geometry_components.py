"""Opt-in bbox-only objective with independent physical size/axis terms."""
import torch
from torch.nn import functional as F
from .morai_bbox_corner import _corners, _instance_frame_mean


def _cyclic_distance(a, b):
    return torch.stack([torch.linalg.vector_norm(a.roll(k, 1)-b,dim=-1).mean(-1)
                        for k in range(4)],-1).min(-1).values


def component_geometry_loss(prediction, target, instance_ids, baseline, anchor_weight=.1):
    selected=instance_ids>=0
    zero=prediction.reshape(-1)[:0].sum()
    center=size=heading=validity=direction=zero
    if selected.any():
        p,t=prediction[selected],target[selected]
        if not torch.isfinite(p[:,1:7]).all() or not torch.isfinite(t[:,:7]).all():
            raise ValueError('Nonfinite selected geometry')
        if (t[:,5:7]<=0).any() or (t[:,0]<=0).any() or (t[:,3:5].norm(dim=-1)==0).any():
            raise ValueError('Invalid target geometry')
        signs=p.new_tensor([1.,-1.]);scale=p.new_tensor([7.,4.])
        phraw=p[:,3:5]*signs;th=F.normalize(t[:,3:5]*signs,dim=-1)
        ps=p[:,5:7]*scale;ts=t[:,5:7]*scale
        transverse=torch.stack((-phraw[:,1],phraw[:,0]),-1)
        # Choose an equivalent axis encoding; this does not alter the rectangle.
        swap=((transverse.detach()*th).sum(-1).abs()>(phraw.detach()*th).sum(-1).abs())
        phraw=torch.where(swap[:,None],transverse,phraw)
        ps=torch.where(swap[:,None],ps.flip(-1),ps)
        phraw=torch.where(((phraw.detach()*th).sum(-1)<0)[:,None],-phraw,phraw)
        ph=F.normalize(phraw,dim=-1,eps=1e-3)
        costs=torch.stack((torch.linalg.vector_norm((p[:,1:3]-t[:,1:3])*3.5,dim=-1),
                           _cyclic_distance(_corners(th,ps),_corners(th,ts)),
                           _cyclic_distance(_corners(ph,ts),_corners(th,ts)),
                           F.relu(.02-ps).mean(-1),
                           F.smooth_l1_loss(phraw,th,beta=.2,reduction='none').mean(-1)),dim=-1)
        center,size,heading,validity,direction=_instance_frame_mean(costs,t[:,0],selected,instance_ids).unbind()
    mask=(baseline[...,0]>=0)|selected
    anchor=F.mse_loss(prediction[...,1:7][mask],baseline.detach()[...,1:7][mask]) if mask.any() else zero
    loss=.5*center+size+heading+2.*validity+.02*direction+anchor_weight*anchor
    return loss,dict(center_m=center,size_corner_m=size,heading_corner_m=heading,
                     size_validity_m=validity,direction_anchor=direction,anchor=anchor)
