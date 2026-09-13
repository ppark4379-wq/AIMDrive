"""Small diagnostic readout for the existing real visual encoder.

No pruning, Q-Former, LM or route-arc decoder is instantiated. The existing
sensor preparation, teacher validation, objective and evaluation are reused.
This probe isolates trainability; it is not a replacement target architecture.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lavis.common.morai_runtime import complete_visual_checkpoint_initialization
from lavis.models.vldrive.visual_encoder import MoraiVisualEncoder


@dataclass
class VisualTrajectoryConfig:
    encoder_policy: str = 'frozen'
    visual_checkpoint: str = ''
    visual_sha256: str = ''
    precision: str = 'bf16'
    speed_weight: float = 1.0
    pruning_weight: float = 0.0
    reconstruction_weight: float = 0.0
    xy_output_scale_m: float = 40.
    speed_output_scale_mps: float = 20.


class VisualTrajectoryReadout(nn.Module):
    """Ordered per-frame tokens -> causal GRU -> direct metric XY and speed."""
    def __init__(self, config):
        super().__init__()
        self.token = nn.Sequential(nn.LayerNorm(256),nn.Linear(256,16),nn.GELU())
        self.visual = nn.Sequential(nn.Linear(106*16,256),nn.GELU())
        self.route = nn.Sequential(nn.Linear(82,64),nn.GELU())
        self.temporal = nn.GRU(322,128,batch_first=True)
        self.context = nn.Sequential(nn.Linear(128,128),nn.GELU())
        self.xy = nn.Linear(128,10)
        self.speed = nn.Linear(128,5)

        if config.xy_output_scale_m <= 0 or config.speed_output_scale_mps <= 0:
            raise ValueError('Output unit scales must be positive')
        self.visual_norm = nn.LayerNorm(256)
        self.route_norm = nn.LayerNorm(64)
        self.xy_scale = config.xy_output_scale_m
        self.speed_scale = config.speed_output_scale_mps

    def forward(self,visual,route,elapsed):
        if visual.ndim!=4 or visual.shape[-2:]!=(106,256):raise ValueError('Invalid visual token contract')
        if route.shape!=visual.shape[:2]+(41,2) or elapsed.shape!=visual.shape[:2]:raise ValueError('Input shape mismatch')
        if not all(bool(torch.isfinite(x).all()) for x in (visual,route,elapsed)):raise ValueError('Nonfinite input')
        if bool((elapsed[:,0]!=0).any()) or bool((torch.diff(elapsed,dim=1)<=0).any()):raise ValueError('Invalid receive elapsed time')
        with torch.autocast(device_type=visual.device.type,enabled=False):
            v=self.visual_norm(self.visual(self.token(visual.float()).flatten(-2)))
            r=self.route_norm(self.route((route.float()/40.).flatten(-2)))
            dt=torch.diff(elapsed,prepend=elapsed[:,:1],dim=1).float()
            context,_=self.temporal(torch.cat((v,r,elapsed.float().unsqueeze(-1),dt.unsqueeze(-1)),-1))
            context=self.context(context)
            xy=self.xy(context).reshape(*visual.shape[:2],5,2)*self.xy_scale
            speed=F.softplus(self.speed(context))*self.speed_scale
        return dict(trajectory=xy,speed=speed)


class VisualTrajectoryModel(nn.Module):
    def __init__(self,config,*,device='cuda'):
        super().__init__()
        if config.encoder_policy not in ('frozen','fusion','all_features') or config.precision not in ('fp32','bf16'):
            raise ValueError('Unsupported probe policy/precision')
        if config.pruning_weight or config.reconstruction_weight or config.speed_weight<0:
            raise ValueError('Probe uses only XY and speed supervision')
        self.config=config
        with complete_visual_checkpoint_initialization():
            self.visual_encoder=MoraiVisualEncoder(config.visual_checkpoint,policy=config.encoder_policy,
                device=device,expected_sha256=config.visual_sha256)
        self.readout=VisualTrajectoryReadout(config)
        self.to(device)

    def forward(self,inputs,*,selection_mode=None,compute_auxiliary=False,retain_visual_grad=False):
        if set(inputs)!={'sensors','route','elapsed_s','instruction'}:
            raise ValueError('Probe allowlist rejects teacher, targets and localization metadata')
        if not isinstance(inputs['instruction'],(list,tuple)) or any(inputs['instruction']):
            raise ValueError('This visual-only diagnostic requires empty instructions')
        sensors=inputs['sensors'];keys=sorted(sensors)
        features=[]
        for t in range(sensors['rgb_front'].shape[1]):
            values=tuple(sensors[k][:,t:t+1] for k in keys)
            def encode(*args):
                return self.visual_encoder(dict(zip(keys,args)))
            enabled=self.training and self.config.encoder_policy!='frozen' and torch.is_grad_enabled()
            features.append(checkpoint(encode,*values,use_reentrant=False) if enabled else encode(*values))
        visual=torch.cat(features,1)
        if retain_visual_grad and visual.requires_grad:visual.retain_grad()
        output=self.readout(visual,inputs['route'],inputs['elapsed_s'])
        output.update(visual_feature=visual,pruning_loss=visual.new_zeros(()),reconstruction_loss=visual.new_zeros(()),
            selected_count=torch.full(visual.shape[:2],106,device=visual.device,dtype=torch.long))
        return output

    def gradient_groups(self):
        groups=dict(visual_readout='readout.visual.',token_projection='readout.token.',route='readout.route.',
            temporal='readout.temporal.',trajectory='readout.xy.',speed='readout.speed.')
        if self.config.encoder_policy!='frozen':
            groups.update(encoder_fusion='visual_encoder.encoder.encoder.',encoder_decoder='visual_encoder.encoder.decoder.')
        if self.config.encoder_policy=='all_features':
            groups.update(rgb_backbone='visual_encoder.encoder.rgb_backbone.',lidar_backbone='visual_encoder.encoder.lidar_backbone.',
                rgb_projection='visual_encoder.encoder.rgb_patch_embed.proj.')
        return groups

    def describe(self):
        return dict(schema='morai-visual-trajectory-v1', conditioning='LayerNorm before GRU',
            xy_output_scale_m=self.config.xy_output_scale_m, speed_output_scale_mps=self.config.speed_output_scale_mps,visual_encoder=self.visual_encoder.describe(),
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            readout='106 ordered tokens -> projection -> causal GRU128 -> direct metric XY and softplus speed',
            instantiated_ccdp=False,instantiated_qformer=False,instantiated_language_model=False,
            route_arc_prior=False,model_inputs='raw sensors, metric route, receive elapsed time')
