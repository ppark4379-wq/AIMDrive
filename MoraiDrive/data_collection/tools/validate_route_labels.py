#!/usr/bin/env python3
"""Validate MGeo route progress and LMDrive command labels in a collection."""
import argparse
import collections
import glob
import json
import math
import os
import statistics


COMMAND_NAMES = {1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "LANEFOLLOW"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route_dir")
    parser.add_argument("--max-route-error", type=float, default=3.0)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.route_dir, "measurements_full", "*.json")))
    if not paths:
        raise SystemExit("measurements_full JSON이 없습니다")
    frames = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            frames.append(json.load(f))

    indices = [f["navigation"]["route_index"] for f in frames]
    errors = [float(f["navigation"]["distance_to_link_m"]) for f in frames]
    target_distances = [math.hypot(f["x_command"] - f["gps_x"],
                                   f["y_command"] - f["gps_y"]) for f in frames]
    commands = [int(f["command"]) for f in frames]
    valid_mask = [error <= args.max_route_error for error in errors]
    transitions = []
    previous = None
    for index, command in enumerate(commands):
        if command != previous:
            transitions.append((index, command))
            previous = command

    monotonic = all(b >= a for a, b in zip(indices, indices[1:]))
    valid_commands = all(command in COMMAND_NAMES for command in commands)
    route_error_ok = max(errors) <= args.max_route_error
    print(f"frames: {len(frames)}")
    print(f"route_index: {indices[0]} -> {indices[-1]}")
    print(f"route_index monotonic: {monotonic}")
    print(f"route error m: median={statistics.median(errors):.2f} max={max(errors):.2f}")
    first_off_route = next((i for i, error in enumerate(errors)
                            if error > args.max_route_error), None)
    if first_off_route is not None:
        frame = frames[first_off_route]
        print(
            f"first off-route: frame={first_off_route:04d} "
            f"xy=({frame['gps_x']:.2f},{frame['gps_y']:.2f}) "
            f"error={errors[first_off_route]:.2f}m"
        )
    print(f"target distance m: median={statistics.median(target_distances):.2f} "
          f"min={min(target_distances):.2f} max={max(target_distances):.2f}")
    counts = collections.Counter(commands)
    print("command counts:", {COMMAND_NAMES[k]: counts[k] for k in sorted(counts)})
    valid_counts = collections.Counter(
        command for command, valid in zip(commands, valid_mask) if valid
    )
    print("on-route command counts:", {
        COMMAND_NAMES[k]: valid_counts[k] for k in sorted(valid_counts)
    })
    print(f"on-route frames: {sum(valid_mask)}/{len(valid_mask)}")
    print("command transitions:")
    for frame_index, command in transitions:
        print(f"  frame={frame_index:04d} route_index={indices[frame_index]} "
              f"command={command}({COMMAND_NAMES[command]})")

    failures = []
    if not monotonic:
        failures.append("route_index가 역행함")
    if not valid_commands:
        failures.append("command 범위가 1~4가 아님")
    if not route_error_ok:
        failures.append(f"route 오차가 {args.max_route_error:.1f}m 초과")
    if indices[-1] == indices[0]:
        failures.append("route_index가 변하지 않음")
    print("result:", "FAIL - " + "; ".join(failures) if failures else "PASS")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
