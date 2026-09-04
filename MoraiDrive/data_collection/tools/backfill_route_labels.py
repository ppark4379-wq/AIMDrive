#!/usr/bin/env python3
"""Backfill route progress and LMDrive commands into an existing collection."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mgeo_navigation import MGeoMap, MGeoRouteTracker, GNSSMotionEstimator


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--map-dir", required=True)
    args = parser.parse_args()
    with open(os.path.join(args.route_dir, "route.json"), encoding="utf-8") as f:
        route = json.load(f)
    tracker = MGeoRouteTracker(
        MGeoMap(os.path.join(args.map_dir, "link_set.json")), route
    )
    estimator = GNSSMotionEstimator()
    full_dir = os.path.join(args.route_dir, "measurements_full")
    paths = sorted(os.path.join(full_dir, name) for name in os.listdir(full_dir)
                   if name.endswith(".json"))
    all_frames = []
    command_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    previous_index = -1
    for full_path in paths:
        with open(full_path, encoding="utf-8") as f:
            full = json.load(f)
        gps = full.get("gnss_mgeo_local")
        if gps is None:
            raise RuntimeError(f"{full_path}: gnss_mgeo_local missing")
        localization = estimator.update(
            float(gps["x"]), float(gps["y"]),
            int(full["sensor_timestamps_ns"]["gnss"]),
            float(full["ego_raw"]["yaw"]), float(full["ego_raw"]["speed"]),
        )
        progress = tracker.update(localization.x, localization.y,
                                  localization.heading_rad)
        command_counts[progress.command] += 1
        if progress.route_index < previous_index:
            raise RuntimeError("route_index moved backward")
        previous_index = progress.route_index
        full.update({"x": localization.x, "y": localization.y,
                     "gps_x": localization.x, "gps_y": localization.y,
                     "theta": localization.heading_rad,
                     "speed": localization.speed_mps,
                     "command": progress.command,
                     "x_command": progress.target_x,
                     "y_command": progress.target_y})
        full["localization"] = {
            "source": "gnss_mgeo", "heading_source": localization.heading_source,
            "x": localization.x, "y": localization.y,
            "heading_rad": localization.heading_rad,
            "speed_mps": localization.speed_mps,
        }
        full.setdefault("navigation", {}).update({
            "current_link_id": progress.current_link_id,
            "distance_to_link_m": progress.distance_to_route_m,
            "route_index": progress.route_index,
            "route_link_index": progress.route_link_index,
            "target_route_index": progress.target_route_index,
            "command_source": "mgeo_related_signal_30m",
            "route_valid": progress.distance_to_route_m <= 3.0,
        })
        write_json(full_path, full)
        measurement_path = os.path.join(
            args.route_dir, "measurements", os.path.basename(full_path)
        )
        with open(measurement_path, encoding="utf-8") as f:
            measurement = json.load(f)
        measurement.update({"x": localization.x, "y": localization.y,
                            "gps_x": localization.x, "gps_y": localization.y,
                            "theta": localization.heading_rad,
                            "speed": localization.speed_mps,
                            "command": progress.command,
                            "x_command": progress.target_x,
                            "y_command": progress.target_y})
        write_json(measurement_path, measurement)
        all_frames.append(full)
    write_json(os.path.join(args.route_dir, "measurements_all.json"), all_frames)
    print(f"updated frames: {len(paths)}")
    print(f"command counts: {command_counts}")
    if all_frames:
        print(f"route_index: {all_frames[0]['navigation']['route_index']} -> "
              f"{all_frames[-1]['navigation']['route_index']}")


if __name__ == "__main__":
    main()
