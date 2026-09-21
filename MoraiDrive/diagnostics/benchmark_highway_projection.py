#!/usr/bin/env python3
"""Offline, paired scalar/optimized highway planner timing and output comparison."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

MORAI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MORAI_ROOT / "expert" / "src"))

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.planning.highway_lane_planning.structures import HighwayLaneLink
from expert.src.vehicle_state import VehicleState


class ScalarLaneMap(HighwayLaneMap):
    project = HighwayLaneMap._project_scalar


def snapshot(value):
    """Compare the complete result, including diagnostics and candidate paths."""
    if isinstance(value, HighwayLaneLink):
        return value.id
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    if hasattr(value, "__dict__"):
        return snapshot(vars(value))
    return value


def scene(lane_map, link_id, s, count):
    link = lane_map.links[link_id]
    point, yaw, _ = lane_map.point_at_s(link, s)
    ego = VehicleState(point.x, point.y, yaw, 12.)
    match = lane_map.project(point, link)
    path = [pose[0] for pose in lane_map.route_points(match, 100., .5)]
    entries = []
    for index in range(count):
        center, angle, _ = lane_map.point_at_s(link, s + 20. + index * 8.)
        obj = ObjectInfo(center.x, center.y, 10., 1, name="synthetic_{}".format(index),
                         size_x=4.5, size_y=2., heading_deg=math.degrees(angle), unique_id=index + 1)
        dx, dy = center.x - point.x, center.y - point.y
        entries.append({"object_info": obj, "local_position": Point(
            math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)})
    return ego, path, entries


def run(iterations):
    map_dir = str(MORAI_ROOT / "expert" / "map_data" / "R_KR_PG_KATRI_2025")
    fast_map, reference_map = HighwayLaneMap(map_dir), ScalarLaneMap(map_dir)
    rows = []
    for count in (0, 5, 12):
        planners = {"scalar": HighwayLanePlanner(lane_map=reference_map),
                    "optimized": HighwayLanePlanner(lane_map=fast_map)}
        times = {name: [] for name in planners}
        for step in range(iterations + 2):
            inputs = scene(fast_map, "A2256W000411", 30. + step * .2, count)
            outputs = {}
            # Alternate ordering to reduce warm-cache/timing bias. Do not time
            # construction, comparison or simulator/network/rendering work.
            order = ("scalar", "optimized") if step % 2 else ("optimized", "scalar")
            for name in order:
                start = time.perf_counter()
                result = planners[name].plan(*inputs, desired_velocity=20.)
                duration = (time.perf_counter() - start) * 1000.
                if step >= 2:
                    times[name].append(duration)
                outputs[name] = snapshot(result)
            if outputs["scalar"] != outputs["optimized"]:
                raise AssertionError("Planner output changed: NPCs={}, step={}".format(count, step))
        row = {"npc_count": count, "equal_output_frames": iterations + 2}
        for name, values in times.items():
            row[name + "_median_ms"] = round(statistics.median(values), 3)
            row[name + "_p95_ms"] = round(float(np.percentile(values, 95)), 3)
        row["speedup"] = round(row["scalar_median_ms"] / row["optimized_median_ms"], 2)
        rows.append(row)
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scope": "Offline planner only; synthetic traffic; not end-to-end ROS or MORAI latency",
            "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    result = run(args.iterations)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
