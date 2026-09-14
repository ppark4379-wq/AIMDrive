#!/usr/bin/env python3
"""Read-only audit and offline replay of the recorded static-route library."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from expert.src.planning.정적장애물.recorded_planner import RecordedStaticPlanner
from expert.src.planning.lattice_planning.lattice_planner import LatticePlanner
from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.obstacle.object_info import ObjectInfo


def main():
    planner = RecordedStaticPlanner(LatticePlanner())
    if planner.error:
        raise ValueError(planner.error)
    with (ROOT / ".runtime/path.csv").open(newline="") as stream:
        route = [Point(float(r["x"]), float(r["y"])) for r in csv.DictReader(stream)]
    scene = json.loads((ROOT / "src/expert/tests/static_scene_20260914.json").read_text())
    obstacles = [ObjectInfo(**row) for row in scene["objects"]]
    ego = VehicleState(-60.1, -162.0, np.pi/2, 0.)
    nearest = min(range(len(route)), key=lambda i: ego.position.distance(route[i]))
    preview = route[nearest:nearest+200]
    planner._setup(preview, ego)
    for record, curve in zip(planner.audit, planner.curves):
        ys = np.arange(max(record["start"][1], ego.position.y), planner.goal_y, .1)
        record["max_curvature"] = float(np.max(np.abs(curve(ys, 2))/(1+curve(ys, 1)**2)**1.5))
        record["whole_route_clear"] = planner._edge(curve, ys[0], planner.goal_y, obstacles) is not None
    start = time.monotonic()
    result = planner.plan(ego, preview[:60], obstacles, preview)
    print(json.dumps(dict(audit=planner.audit, elapsed_s=time.monotonic()-start,
        result=result.reason, path_points=len(result.path), target_kph=result.acc_target_velocity*3.6,
        rejected=planner.rejections, progress=planner.search_progress,
        details=result.debug_details), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
