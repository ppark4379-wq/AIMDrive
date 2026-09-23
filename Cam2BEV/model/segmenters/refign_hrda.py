"""Inference-only adapter for official Refign-HRDA ACDC weights.

Uses the upstream MiT-B5, DAFormer head and HRDA decorators unchanged.
Training-only teacher/alignment/optimizer components are not instantiated.
"""
import importlib
import os
from pathlib import Path
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
_runtime=None

def _official_modules(source):
    # Namespace the upstream files without importing its Lightning training package.
    for name,folder in [("refign_infer",source/"models"),
                        ("refign_infer.backbones",source/"models/backbones"),
                        ("refign_infer.heads",source/"models/heads")]:
        if name not in sys.modules:
            module=types.ModuleType(name)
            module.__path__=[str(folder)]
            sys.modules[name]=module
    backbone=importlib.import_module("refign_infer.backbones.mix_transformer")
    head=importlib.import_module("refign_infer.heads.daformer")
    attention=importlib.import_module("refign_infer.heads.segformer")
    hrda=importlib.import_module("refign_infer.hrda")
    return backbone,head,attention,hrda

def build_network(source, backbone_type="mit_b5"):
    import torch
    from torch import nn
    B,D,A,H=_official_modules(source)
    class RefignInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=B.MixVisionTransformer(model_type=backbone_type,pretrained=None)
            channels=B.MixVisionTransformer.arch_settings[backbone_type]["embed_dims"]
            settings=dict(in_channels=channels,in_index=[0,1,2,3],
                          num_classes=19,input_transform="multiple_select")
            self.head=D.DAFormerHead(**settings)
            self.hrda_scale_attention=A.SegFormerHead(**settings)
            base_forward=self.backbone.forward
            # Same eval math with sequential crops to bound peak CPU memory.
            def chunked(x):
                results=[base_forward(part) for part in x.split(1)]
                return [torch.cat([r[i] for r in results]) for i in range(4)]
            self.backbone.forward=H.hrda_backbone(self.backbone,4)(chunked)
            self.head.forward=H.hrda_head(self.head,self.hrda_scale_attention,4)(self.head.forward)
        def whole(self,x):
            logits=self.head(self.backbone(x))
            return nn.functional.interpolate(logits,x.shape[-2:],mode="bilinear",align_corners=False)
        def forward(self,x):
            # Official outer sliding-window settings: crop=1080, stride=420.
            b,_,height,width=x.shape
            output=x.new_zeros((b,19,height,width))
            count=x.new_zeros((b,1,height,width))
            nh=max(height-1080+419,0)//420+1
            nw=max(width-1080+419,0)//420+1
            for i in range(nh):
                for j in range(nw):
                    y2=min(i*420+1080,height);x2=min(j*420+1080,width)
                    y1=max(y2-1080,0);x1=max(x2-1080,0)
                    output[:,:,y1:y2,x1:x2]+=self.whole(x[:,:,y1:y2,x1:x2])
                    count[:,:,y1:y2,x1:x2]+=1
            if not (count>0).all():raise RuntimeError("Uncovered inference pixels")
            return output/count
    return RefignInference()

def _load():
    global _runtime
    if _runtime is None:
        import torch
        source=Path(os.environ.get("REFIGN_SOURCE",ROOT/"data/models/refign-source"))
        path=Path(os.environ.get("REFIGN_WEIGHTS",ROOT/"data/models/refign_hrda_star_acdc.ckpt"))
        if not path.is_file():
            raise FileNotFoundError("Official Refign-HRDA ACDC checkpoint required: "+str(path))
        torch.set_num_threads(4)
        network=build_network(source).eval()
        # The source checkpoint must be the verified official release.
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        state=checkpoint.get("state_dict",checkpoint)
        prefixes=("backbone.","head.","hrda_scale_attention.")
        state={k:v for k,v in state.items() if k.startswith(prefixes)}
        network.load_state_dict(state,strict=True)
        _runtime=(torch,network)
    return _runtime

def segment(rgb_image,camera_name=None):
    torch,network=_load()
    rgb=np.asarray(rgb_image)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError("Expected native RGB uint8 image")
    if rgb.shape[0]%8 or rgb.shape[1]%8:
        raise ValueError("This native-resolution HRDA adapter requires dimensions divisible by 8")
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255)[None]
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode():
        logits=network((x-mean)/std)
        if not torch.isfinite(logits).all():raise RuntimeError("Non-finite HRDA logits")
        ids=logits.argmax(1)[0].numpy()
    # Cityscapes 19 train IDs -> existing Cam2BEV ten input channels.
    lut=np.array([0,1,7,7,7,7,7,7,8,8,9,2,6,3,4,5,7,6,6],np.uint8)
    return lut[ids]
