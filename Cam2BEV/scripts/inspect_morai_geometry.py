"""Project a metric ground grid into RGB images to inspect calibration."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model"))
import morai_geometry as g
p=argparse.ArgumentParser()
p.add_argument("--capture",type=Path,required=True)
p.add_argument("--source-map",type=Path,required=True)
p.add_argument("--config",type=Path,default=ROOT/"model/config.morai_udp.json")
p.add_argument("--ground-z",type=float,default=0)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
cfg=json.loads(a.config.read_text())
mapping=json.loads(a.source_map.read_text())["camera_to_capture_folder"]
a.output.mkdir(parents=True,exist_ok=True)
for name in g.ORDER:
    src=mapping[name]
    rgb=cv2.imread(str(a.capture/src/"00000000.jpg"))
    P=g.camera_projection(cfg["cameras"][name])
    def line(points,color):
        uv=P@points
        good=uv[2]>0.2
        xy=(uv[:2]/np.maximum(uv[2],.001)).T
        # Fine sampling avoids connecting disjoint portions across the horizon.
        for i in range(len(xy)-1):
            if good[i] and good[i+1] and np.max(np.abs(xy[i:i+2]))<100000:
                ok,v,w=cv2.clipLine((0,0,rgb.shape[1],rgb.shape[0]),tuple(np.rint(xy[i]).astype(int)),tuple(np.rint(xy[i+1]).astype(int)))
                if ok:cv2.line(rgb,v,w,color,1,cv2.LINE_AA)
    t=np.linspace(-35,35,701)
    for y in range(-15,16,5):
        line(np.array([t,np.full_like(t,y),np.full_like(t,a.ground_z),np.ones_like(t)]),(0,255,255))
    for x in range(-30,31,5):
        line(np.array([np.full_like(t,x),t,np.full_like(t,a.ground_z),np.ones_like(t)]),(255,255,0))
    cv2.putText(rgb,name+" <- "+src+" | grid 5m | UNCONFIRMED",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
    cv2.imwrite(str(a.output/(name+".jpg")),rgb)
(a.output/"settings.json").write_text(json.dumps({"config":cfg,"mapping":mapping,"ground_z":a.ground_z,"status":"Requires known ground landmarks"},indent=2))
