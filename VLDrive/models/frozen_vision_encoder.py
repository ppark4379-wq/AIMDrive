"""Explicit frozen/partial-adaptation wrapper; no teacher or transport inputs.

`fusion` is an experiment option, not a final adaptation policy. It trains the
visual transformer and view/query embeddings while retaining frozen RGB/LiDAR
backbones and their BatchNorm statistics. All policies keep eval-mode modules
for a controlled dropout/normalization comparison; eval does not disable grads.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from utils.common.runtime import deterministic_pointpillar_reductions

from data.src.loaders.sensor_adapter import load_frozen_lmdrive_vision_encoder


SENSOR_KEYS={"rgb_front","rgb_left","rgb_right","rgb_rear","rgb_center","lidar","num_points","velocity"}
FUSION_PREFIXES=("encoder.","decoder.","view_embed","global_embed","query_embed","query_pos_embed","velocity_fc.")


class _VisualEncoderBase(nn.Module):
    def __init__(self, checkpoint_path, *, policy="frozen", device="cuda", expected_sha256=None):
        super().__init__()
        if policy not in ("frozen","fusion"):raise ValueError("policy must be frozen or fusion")
        checkpoint=Path(checkpoint_path).expanduser().resolve()
        digest=hashlib.sha256()
        with checkpoint.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
        self.checkpoint_sha256=digest.hexdigest()
        if expected_sha256 is not None and self.checkpoint_sha256!=expected_sha256:
            raise ValueError("Raw encoder/cache checkpoint mismatch")
        self.encoder,self.load_result=load_frozen_lmdrive_vision_encoder(checkpoint_path=checkpoint,device=device)
        self.policy=policy
        for name,p in self.encoder.named_parameters():
            p.requires_grad_(policy=="fusion" and name.startswith(FUSION_PREFIXES))
        if policy=="fusion" and not any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError("No matching fusion parameters")
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, sensors):
        if set(sensors)!=SENSOR_KEYS:raise ValueError("Unexpected/missing sensor key")
        shape=sensors["rgb_front"].shape
        if len(shape)!=5:raise ValueError("Raw RGB input must be [B,T,C,H,W]")
        batch,time=shape[:2]
        if any(tuple(v.shape[:2])!=(batch,time) for v in sensors.values()):
            raise ValueError("Sensor batch/time mismatch")
        velocity = sensors["velocity"]
        if not bool(torch.isfinite(velocity).all()) or bool((velocity < 0).any()):
            raise ValueError("MORAI visual encoder requires finite current ego speed")
        flat={k:v.reshape(batch*time,*v.shape[2:]) for k,v in sensors.items()}
        if self.policy=="frozen":
            with torch.no_grad():features=self.encoder(flat)
        else:features=self.encoder(flat)
        if tuple(features.shape)!=(batch*time,106,256) or not bool(torch.isfinite(features).all()):
            raise ValueError("Invalid encoder feature contract")
        return features.reshape(batch,time,106,256)

    def describe(self):
        return dict(policy=self.policy,checkpoint_sha256=self.checkpoint_sha256,strict_load=str(self.load_result),
            frozen_backbone_eval=True,trainable_names=[n for n,p in self.encoder.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in self.encoder.parameters()),velocity_source="explicit-zero")


class MoraiVisualEncoder(_VisualEncoderBase):
    """Explicit feature-backbone adaptation, with small-batch BN held in eval."""
    def __init__(self, path, *, policy, **kwargs):
        super().__init__(path, policy='fusion' if policy == 'all_features' else policy, **kwargs)
        if policy == 'all_features':
            prefixes = FUSION_PREFIXES + ('rgb_backbone.', 'rgb_patch_embed.proj.', 'lidar_backbone.')
            for name, parameter in self.encoder.named_parameters():
                parameter.requires_grad_(name.startswith(prefixes))
            self.policy = policy

    def forward(self, sensors):
        with deterministic_pointpillar_reductions():
            return super().forward(sensors)

    def describe(self):
        report = super().describe()
        report.pop('frozen_backbone_eval', None)
        report.update(backbone_normalization='eval, running statistics fixed',
                      backbone_weights_trainable=self.policy == 'all_features',
                      pointpillar_reduction='sorted deterministic segment_csr, mean/max')
        return report
