#!/usr/bin/env python3
"""Backfill route progress and LMDrive commands into an existing collection."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mgeo_navigation import (
    MGeoMap, MGeoRouteTracker, GNSSMotionEstimator, load_official_route,
)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--route-definition", default="")
    parser.add_argument("--global-path", default="")
    parser.add_argument("--route-definition-key", default="full_loop")
    args = parser.parse_args()
    with open(os.path.join(args.route_dir, "route.json"), encoding="utf-8") as f:
        stored_route = json.load(f)
    has_official_route = bool(args.route_definition or args.global_path)
    if has_official_route and not (args.route_definition and args.global_path):
        raise RuntimeError("--route-definition과 --global-path는 함께 지정해야 합니다")
    route = (
        load_official_route(
            args.route_definition, args.global_path, args.route_definition_key
        )
        if has_official_route else stored_route
    )
    node_set_path = os.path.join(args.map_dir, "node_set.json")
    mgeo_map = MGeoMap(
        os.path.join(args.map_dir, "link_set.json"),
        node_set_path=node_set_path if os.path.isfile(node_set_path) else None,
    )
    tracker = MGeoRouteTracker(mgeo_map, route)
    if has_official_route:
        route["map_id"] = stored_route.get("map_id", route.get("map_id"))
        route["map_path"] = stored_route.get("map_path", route.get("map_path"))
        route["nav_node_ids"] = [
            str(mgeo_map.links[route["nav_link_ids"][0]].get("from_node_idx"))
        ] + [
            str(mgeo_map.links[link_id].get("to_node_idx"))
            for link_id in route["nav_link_ids"]
        ]
        route["nav_road_options"] = list(tracker.point_road_options)
        write_json(os.path.join(args.route_dir, "route.json"), route)
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
                     "y_command": progress.target_y,
                     # 기존 수집 프레임의 실제 Ego trajectory를 사용하지 않는다.
                     # 현재 official dense route의 앞쪽 최대 50개 planned point다.
                     "future_waypoints": progress.future_waypoints})
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
            "command_source": "route_link_annotation",
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
                            "y_command": progress.target_y,
                            "future_waypoints": progress.future_waypoints})
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
