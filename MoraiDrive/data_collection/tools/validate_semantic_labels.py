#!/usr/bin/env python3
"""Summarize junction, stop-line and traffic-light labels in a route capture."""
import argparse
import collections
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route_dir")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(
        args.route_dir, "measurements_full", "*.json"
    )))
    if not paths:
        raise SystemExit("measurements_full JSON이 없습니다")
    frames = [json.load(open(path, encoding="utf-8")) for path in paths]
    on_route = [f for f in frames if f.get("navigation", {}).get("route_valid") is not False]
    junction = [f for f in on_route if f.get("is_junction")]
    affected = [f for f in on_route if f.get("affected_light_id", -1) != -1]
    observed = [f for f in affected
                if f.get("navigation", {}).get("affected_light_state_raw") is not None]
    red = [f for f in on_route if f.get("is_red_light_present")]
    distances = [f["navigation"]["distance_to_stopline"] for f in on_route
                 if f.get("navigation", {}).get("distance_to_stopline") is not None]
    light_ids = collections.Counter(
        f["navigation"].get("affected_light_id_raw") for f in affected
    )
    print(f"frames: {len(frames)} (on-route {len(on_route)})")
    print(f"junction frames: {len(junction)}")
    print(f"affected-light frames: {len(affected)}")
    print(f"matched live light-state frames: {len(observed)}")
    print(f"red-light frames: {len(red)}")
    if distances:
        print(f"distance_to_stopline m: min={min(distances):.2f} max={max(distances):.2f}")
    print("affected light IDs:", dict(light_ids))
    failures = []
    if not junction:
        failures.append("junction 라벨 없음")
    if not affected:
        failures.append("affected_light_id 없음")
    if affected and not observed:
        failures.append("route 신호등 ID와 ROS live 신호 ID가 매칭되지 않음")
    print("result:", "FAIL - " + "; ".join(failures) if failures else "PASS")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
