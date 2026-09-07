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

from .morai_lmdrive_vision_adapter import load_frozen_lmdrive_vision_encoder


SENSOR_KEYS={"rgb_front","rgb_left","rgb_right","rgb_rear","rgb_center","lidar","num_points","velocity"}
FUSION_PREFIXES=("encoder.","decoder.","view_embed","global_embed","query_embed","query_pos_embed","velocity_fc.")


class Stage6FVisualEncoder(nn.Module):
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
        if not bool(sensors["velocity"].eq(0).all()):
            raise ValueError("Stage6F baseline uses explicit zero velocity, never Ego teacher")
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
