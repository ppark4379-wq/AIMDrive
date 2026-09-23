"""Offline SegFormer processing of the four-camera capture; no new training."""
import argparse
import os
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model/segmenters"))
import segformer_cityscapes as seg


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gamma",type=float,default=1.0)
    parser.add_argument("--max-frames",type=int,default=0)
    parser.add_argument("--segmenter",choices=["segformer","refign"],default="segformer")
    args=parser.parse_args()
    if not np.isfinite(args.gamma) or args.gamma <= 0: raise ValueError("gamma must be positive")
    if args.max_frames < 0: raise ValueError("max-frames must be nonnegative")
    import torch
    torch.set_num_threads(4)
    if args.segmenter=="refign":
        import refign_hrda
        refign_hrda._load()  # fail before creating outputs if weights are missing
        segment_fn=refign_hrda.segment
    else:
        segment_fn=seg.segment
    names=sorted(p.name for p in (args.capture/"front").glob("*.jpg"))
    if not names: raise ValueError("No input JPEGs")
    for camera in ("front","rear","left","right"):
        if sorted(p.name for p in (args.capture/camera).glob("*.jpg"))!=names:
            raise ValueError("Unpaired inputs: "+camera)
    if args.max_frames: names=names[:args.max_frames]
    args.output.mkdir(parents=True,exist_ok=False)
    metadata={}
    for camera in ("front","rear","left","right"):
        for kind in ("ids","semantic","overlay"):
            (args.output/kind/camera).mkdir(parents=True)
        for name in names:
            rgb=np.asarray(Image.open(args.capture/camera/name).convert("RGB"))
            processed=np.rint(255*(rgb.astype(np.float32)/255)**args.gamma).astype(np.uint8)
            ids=segment_fn(processed,camera)
            colors=seg.PALETTE[ids]
            stem=Path(name).stem+".png"
            Image.fromarray(ids).save(args.output/"ids"/camera/stem)
            Image.fromarray(colors).save(args.output/"semantic"/camera/stem)
            Image.fromarray((.55*rgb+.45*colors).astype(np.uint8)).save(args.output/"overlay"/camera/stem)
            counts=np.bincount(ids.ravel(),minlength=10)
            metadata[camera+"/"+name]={"pixels_per_class":counts.tolist()}
            print(camera,name,"road_fraction",float((ids==0).mean()),flush=True)
    if args.segmenter=="refign":
        import subprocess
        info={"model":"Refign-HRDA-star Cityscapes-to-ACDC",
              "source_commit":subprocess.check_output(["git","-C",str(ROOT/"data/models/refign-source"),"rev-parse","HEAD"],text=True).strip(),
              "device":"cpu","gamma":args.gamma,"frames":metadata,
              "checkpoint":json.loads((ROOT/"data/models/refign_acdc_download.json").read_text()),
              "inference":"official HRDA decorators; native resolution; outer crops 1080 stride 420; sequential crop batches"}
    else:
        torch_module, processor, model, device=seg._runtime
        info={"model":os.environ.get("CAM2BEV_SEGMENTER_MODEL",seg.MODEL_ID),"model_commit":getattr(model.config,"_commit_hash",None),
              "device":str(device),"gamma":args.gamma,"frames":metadata}
    (args.output/"manifest.json").write_text(json.dumps(info,indent=2))


if __name__=="__main__": main()
