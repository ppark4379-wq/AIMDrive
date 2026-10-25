import sys,time,json,traceback,builtins
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'expert/src'),'/usr/lib/python3/dist-packages']
from expert.src.planning.highway.common.lane_map import HighwayLaneMap
from expert.src.localization.point import Point
import numpy as np
m=HighwayLaneMap(str(root/'expert/map_data/R_KR_PG_KATRI_2025'))
position=Point(-94.7943344116211,-360.8637390136719)
link=m.links['A2256W000179']
first=m.project(position,link)
expected=(first.s,first.distance,first.x,first.y,first.yaw)
start=time.monotonic();i=0
try:
 while time.monotonic()-start<15:
  obj=m.project(position,link)
  got=(obj.s,obj.distance,obj.x,obj.y,obj.yaw)
  if got!=expected:raise ValueError('projection changed: '+repr((expected,got)))
  i+=1
 print(json.dumps(dict(status='pass',iterations=i,seconds=time.monotonic()-start,python=sys.version,numpy=np.__version__)))
except Exception:
 print(json.dumps(dict(status='fail',iterations=i,seconds=time.monotonic()-start,traceback=traceback.format_exc(),getattr_binding=repr(builtins.getattr),float_binding=repr(builtins.float))))
 sys.exit(1)
