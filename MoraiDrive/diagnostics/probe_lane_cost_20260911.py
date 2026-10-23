"""Bounded offline map scan: 5 calls, no ROS/MORAI/NAS."""
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'expert/src'))
from expert.src.planning.highway.lane_map import HighwayLaneMap
from expert.src.localization.point import Point
lane=HighwayLaneMap(str(ROOT/'scenario_runner/data/R_KR_PR_K-city_2025'))
point=Point(-131.486, -427.961)
start=time.process_time()
results=[lane.match(point,heading=math.radians(61.3)) for _ in range(5)]
cpu=time.process_time()-start
segments=sum(len(link.points)-1 for link in lane.links.values())
print(json.dumps({'links':len(lane.links),'segments_per_match':segments,
 'outside_matches_return_none':all(item is None for item in results),
 'calls':5,'cpu_seconds':cpu,'cpu_ms_per_match':cpu/5*1000}))
