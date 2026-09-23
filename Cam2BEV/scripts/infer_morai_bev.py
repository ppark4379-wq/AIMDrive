"""Infer saved MORAI segmentation with frozen Cam2BEV weights; save diagnostic BEVs."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
os.environ.setdefault("TF_USE_LEGACY_KERAS","1")
import argparse
import json
import shutil
import hashlib
from pathlib import Path
import sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model"))
import utils
import morai_geometry as geometry


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--capture",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--source-map",type=Path,required=True)
    p.add_argument("--config",type=Path,default=ROOT/"model/config.morai_udp.json")
    p.add_argument("--segmentation",type=Path)
    p.add_argument("--weights",type=Path,help="Copy a checkpoint into a fresh run before inference")
    p.add_argument("--ground-z",type=float,default=0.0)
    p.add_argument("--legacy-unmasked",action="store_true")
    args=p.parse_args()
    segdir=args.segmentation or args.run/"segmentation"
    args.run.mkdir(parents=True,exist_ok=True)
    if (args.run/"inference_report.json").exists():
        raise ValueError("Use a new run directory; previous results are preserved")
    if args.weights:
        target=args.run/"frozen_best_road_weights.hdf5"
        if target.exists(): raise ValueError("Checkpoint already exists in run")
        before=args.weights.stat()
        shutil.copy2(args.weights,target)
        after=args.weights.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            target.unlink()
            raise RuntimeError("Checkpoint changed during copy; retry with a stable checkpoint")
        (args.run/"checkpoint.json").write_text(json.dumps({"source":str(args.weights.resolve()),
            "size":target.stat().st_size,"mtime":before.st_mtime,
            "sha256":hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
    mapping=json.loads(args.source_map.read_text())
    if args.source_map.resolve() != (args.run/"source_mapping.json").resolve():
        shutil.copy2(args.source_map,args.run/"source_mapping.json")
    config=json.loads(args.config.read_text())
    if set(mapping["camera_to_capture_folder"]) != set(geometry.ORDER) or set(mapping["camera_to_capture_folder"].values()) != set(geometry.ORDER):
        raise ValueError("Camera mapping must be a permutation of front/rear/left/right")
    matrices,uv,masks=geometry.build_geometry(config["cameras"],args.ground_z)
    palette=utils.parse_convert_xml(str(ROOT/"model/one_hot_conversion/convert_9+occl.xml"))
    inp_palette=utils.parse_convert_xml(str(ROOT/"model/one_hot_conversion/convert_10.xml"))
    colors=np.array([group[0] for group in palette],np.uint8)
    in_colors=np.array([group[0] for group in inp_palette],np.uint8)
    architecture=utils.load_module(str(ROOT/"model/architecture/uNetXST.py"))
    model=architecture.get_network((256,512,10),10,n_inputs=4,thetas=matrices,visibility_masks=None if args.legacy_unmasked else masks)
    model.load_weights(str(args.run/"frozen_best_road_weights.hdf5"))
    folders=("bev_semantic","road_probability","road_mask","road_mask_visible","ipm_rgb","ipm_semantic","overview","road_supported","road_observed","road_disagreement")
    for folder in folders:(args.run/folder).mkdir(exist_ok=True)
    coverage=np.any(masks,axis=0)
    cv2.imwrite(str(args.run/"coverage.png"),coverage.astype(np.uint8)*255)
    summaries=[]
    names=sorted(p.stem for p in (args.capture/"front").glob("*.jpg"))
    for stem in names:
        if not all((segdir/"ids"/c/(stem+".png")).exists() for c in geometry.ORDER): continue
        inputs=[]
        ipm=np.zeros((256,512,3),np.uint8)
        ipm_ids=np.full((256,512),7,np.uint8)
        best=np.full((256,512),-1.,np.float32)
        for i,camera in enumerate(geometry.ORDER):
            source=mapping["camera_to_capture_folder"][camera]
            rgb=cv2.imread(str(args.capture/source/(stem+".jpg")))
            ids=cv2.imread(str(segdir/"ids"/source/(stem+".png")),cv2.IMREAD_GRAYSCALE)
            cam=config["cameras"][camera]
            assert ids.shape==(cam["height"],cam["width"]),(camera,ids.shape)
            tensor=utils.resize_image_op(ids[...,None].astype(np.float32),ids.shape,(256,512),interpolation="nearest")
            net_ids=tensor.numpy()[...,0].astype(np.int32)
            inputs.append(np.eye(10,dtype=np.float32)[net_ids][None])
            x,y=uv[i]
            # Prefer the view closer to its image center in overlapping ground cells.
            score=1-abs(x-cam["width"]/2)/(cam["width"]/2)
            use=masks[i] & (score>best)
            warped=cv2.remap(rgb,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            warped_ids=cv2.remap(ids,x,y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=7)
            ipm[use]=warped[use]; ipm_ids[use]=warped_ids[use]; best[use]=score[use]
        pred=model.predict(inputs,verbose=0)[0]
        assert pred.shape==(256,512,10) and np.isfinite(pred).all()
        labels=pred.argmax(-1)
        road=(labels==0)&(pred[...,0]>=.7)
        visible_road=road&coverage
        observed_road=(ipm_ids==0)&coverage
        supported_road=visible_road&observed_road
        outputs={
            "road_observed":observed_road.astype(np.uint8)*255,
            "road_supported":supported_road.astype(np.uint8)*255,
            "road_disagreement":(visible_road^observed_road).astype(np.uint8)*255,
            "bev_semantic":cv2.cvtColor(colors[labels],cv2.COLOR_RGB2BGR),
            "road_probability":np.rint(pred[...,0]*255).astype(np.uint8),
            "road_mask":road.astype(np.uint8)*255,
            "road_mask_visible":visible_road.astype(np.uint8)*255,
            "ipm_rgb":ipm,"ipm_semantic":cv2.cvtColor(in_colors[ipm_ids],cv2.COLOR_RGB2BGR)}
        for folder,im in outputs.items():
            if not cv2.imwrite(str(args.run/folder/(stem+".png")),im):raise IOError(folder)
        overlay=ipm.copy()
        overlay[supported_road]=(.45*overlay[supported_road]+.55*np.array([0,220,0])).astype(np.uint8)
        panels=[]
        for title,im in (("Ground-plane RGB",ipm),("Cam2BEV classes",outputs["bev_semantic"]),("Road: model AND RGB labels",overlay)):
            upright=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)
            panel=np.full((580,300,3),245,np.uint8)
            panel[45:557,22:278]=upright
            cv2.putText(panel,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(30,30,30),1,cv2.LINE_AA)
            cv2.putText(panel,"FRONT UP | rear down",(10,577),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
            panels.append(panel)
        cv2.imwrite(str(args.run/"overview"/(stem+".jpg")),np.hstack(panels))
        summaries.append({"frame":stem,"road_fraction":float(road.mean()),"visible_road_fraction":float(visible_road.mean()),
                          "mean_road_probability":float(pred[...,0].mean()),"supported_road_fraction":float(supported_road.mean())})
        print(summaries[-1],flush=True)
    if not summaries: raise ValueError("No paired segmentation frames")
    calibration={"cameras":config["cameras"],"ground_z":args.ground_z,
        "meters_per_pixel":geometry.METERS_PER_PIXEL,"shape":list(geometry.SHAPE),
        "assumptions":"FLU; positive pitch down; flat ground; centered pinhole; requires simulator landmark validation"}
    (args.run/"calibration.json").write_text(json.dumps(calibration,indent=2))
    np.savez(args.run/"geometry.npz",H=np.array(matrices),uv=np.array(uv),visible=np.array(masks))
    report={"visibility_masking":not args.legacy_unmasked,
        "segmentation":str(segdir.resolve()),
        "segmentation_manifest":json.loads((segdir/"manifest.json").read_text()),
        "road_supported_definition":"intersection of BEV road >=0.7 and selected camera ground-projected road; not ground truth or occlusion reasoning",
        "checkpoint":json.loads((args.run/"checkpoint.json").read_text()),"source_mapping":mapping,
        "geometry":json.loads((args.run/"calibration.json").read_text()),"threshold":.7,"frames":summaries,
        "limitations":["No MORAI BEV ground truth: no quantitative accuracy measured",
                       "Port mapping provisional unless confirmed by user",
                       "Flat-ground projection and pitch sign require landmark validation",
                       "Visible-road mask only removes geometric out-of-FOV cells, not occlusion",
                       "RGB IPM distorts raised objects; not a ground truth comparison"]}
    (args.run/"inference_report.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":main()
