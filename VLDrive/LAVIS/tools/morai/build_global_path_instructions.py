#!/usr/bin/env python3
"""Build deterministic language instructions from the recorded global path."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "morai-global-path-instructions-v1"
LOOKAHEAD_SCHEMA = "morai-global-path-instructions-v2"
COMMAND_TEXT = {
    1: "Proceed ahead and make a left turn.",
    2: "Proceed ahead and make a right turn.",
    3: "Please proceed straight at the next intersection.",
    4: "Proceed along this route.",
    5: "Transition to the left lane for travel.",
    6: "Transition to the right lane for travel.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_routes(root):
    index = Path(root) / "dataset_index.txt"
    rows = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        relative, count = line.split()
        relative = relative.replace("\\", "/").strip("/")
        if relative.startswith("data/"):
            relative = relative[len("data/"):]
        route = (Path(root) / "data" / relative).resolve()
        if (Path(root) / "data").resolve() not in route.parents:
            raise ValueError("Route escapes dataset root")
        rows.append((relative, route, int(count)))
    if not rows:
        raise ValueError("Dataset index has no routes")
    return index, rows


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RouteLookaheadPolicy:
    """Announce the next route maneuver using only the *current* ego pose.

    Recorded route_index is a causal progress hint, needed to disambiguate road
    crossings and repeated route geometry. Arc length is projected onto its
    adjacent segments, never inferred from future ego positions. RoadOption 3
    means crossing an intersection straight, not that the physical road is a
    straight line; use route-follow text unless a turn/lane change is requested.
    """

    def __init__(self, route_data, lookahead_m=30.0, curve_lookahead_m=20.0,
                 curve_threshold_deg=20.0):
        self.lookahead_m = float(lookahead_m)
        if not math.isfinite(self.lookahead_m) or self.lookahead_m <= 0:
            raise ValueError("lookahead_m must be finite and positive")
        self.curve_lookahead_m = float(curve_lookahead_m)
        self.curve_threshold_deg = float(curve_threshold_deg)
        if (not math.isfinite(self.curve_lookahead_m) or self.curve_lookahead_m <= 0
                or not math.isfinite(self.curve_threshold_deg)
                or not 0 < self.curve_threshold_deg < 180):
            raise ValueError("Curve horizon must be positive and threshold between 0 and 180 degrees")
        self.options = [int(value) for value in route_data["nav_road_options"]]
        self.points = [(float(point[0]), float(point[1]))
                       for point in route_data["nav_waypoints"]]
        if len(self.points) < 2 or len(self.options) != len(self.points):
            raise ValueError("Route geometry and road options must align")
        if any(not all(math.isfinite(value) for value in point) for point in self.points):
            raise ValueError("Route geometry must be finite")
        if any(option not in COMMAND_TEXT for option in self.options):
            raise ValueError("Unsupported global-path command")
        self.cumulative_m = [0.0]
        for before, after in zip(self.points, self.points[1:]):
            self.cumulative_m.append(self.cumulative_m[-1] + math.hypot(
                after[0] - before[0], after[1] - before[1]))
        self.turn_starts = [(index, option) for index, option in enumerate(self.options)
                            if option in (1, 2)
                            and (index == 0 or self.options[index - 1] != option)]

    def _current_progress(self, measurement, route_index):
        # Deliberately no reads of future_waypoints, next_ego_raw or future frames.
        x, y = float(measurement["gps_x"]), float(measurement["gps_y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Current ego coordinates must be finite")
        projections = []
        for segment in range(max(0, route_index - 1), min(len(self.points) - 1, route_index + 1)):
            ax, ay = self.points[segment]
            bx, by = self.points[segment + 1]
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                continue
            fraction = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
            px, py = ax + fraction * dx, ay + fraction * dy
            projections.append(((x - px) ** 2 + (y - py) ** 2,
                                self.cumulative_m[segment] + fraction * math.sqrt(length_sq)))
        return min(projections)[1] if projections else self.cumulative_m[route_index]

    def command_for(self, measurement):
        navigation = measurement.get("navigation", {})
        route_index = int(navigation.get("route_index", -1))
        if not 0 <= route_index < len(self.options):
            raise ValueError("Current ego route index is outside the recorded route")
        if navigation.get("route_valid") is False:
            raise ValueError("Cannot generate route-lookahead instructions from invalid localization")
        current_command = self.options[route_index]
        # An active maneuver wins over a nearby following maneuver until exit.
        if current_command in (1, 2, 5, 6):
            return current_command
        progress = self._current_progress(measurement, route_index)
        for index, command in self.turn_starts:
            if index <= route_index:
                continue
            distance = self.cumulative_m[index] - progress
            if distance > self.lookahead_m:
                break
            if distance >= -1e-6:
                return command
        return 4


    def _point_at(self, progress):
        progress = min(self.cumulative_m[-1], max(0.0, progress))
        index = min(len(self.points) - 2, bisect.bisect_right(self.cumulative_m, progress) - 1)
        span = self.cumulative_m[index + 1] - self.cumulative_m[index]
        fraction = (progress - self.cumulative_m[index]) / span if span > 0 else 0.0
        before, after = self.points[index:index + 2]
        return (before[0] + fraction * (after[0] - before[0]),
                before[1] + fraction * (after[1] - before[1]))

    def instruction_for(self, measurement):
        command = self.command_for(measurement)
        if command != 4:
            return command, COMMAND_TEXT[command]
        route_index = int(measurement["navigation"]["route_index"])
        start = self._current_progress(measurement, route_index)
        stop = min(self.cumulative_m[-1], start + self.curve_lookahead_m)
        # Average the endpoint tangents over up to 3 m to suppress point noise.
        tangent_span = min(3.0, (stop - start) / 3.0)
        if tangent_span < 0.5:
            return command, COMMAND_TEXT[command]
        a, b = self._point_at(start), self._point_at(start + tangent_span)
        c, d = self._point_at(stop - tangent_span), self._point_at(stop)
        before_heading = math.atan2(b[1] - a[1], b[0] - a[0])
        after_heading = math.atan2(d[1] - c[1], d[0] - c[0])
        delta = math.degrees(math.atan2(math.sin(after_heading - before_heading),
                                        math.cos(after_heading - before_heading)))
        if abs(delta) >= self.curve_threshold_deg:
            side = "left" if delta > 0 else "right"
            return command, "Follow the road as it curves to the {}.".format(side)
        return command, COMMAND_TEXT[command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("current_link", "route_lookahead"), default="current_link")
    parser.add_argument("--lookahead-m", type=float, default=30.0,
                        help="Route arc distance for route_lookahead instructions (default: 30 m)")
    parser.add_argument("--curve-lookahead-m", type=float, default=20.0)
    parser.add_argument("--curve-threshold-deg", type=float, default=20.0)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    index, routes = dataset_routes(root)
    records = []
    for route_id, route, frame_count in routes:
        route_file = route / "route.json"
        route_data = load(route_file)
        options = [int(value) for value in route_data["nav_road_options"]]
        policy = (RouteLookaheadPolicy(route_data, args.lookahead_m, args.curve_lookahead_m,
                                       args.curve_threshold_deg)
                  if args.policy == "route_lookahead" else None)
        frames = []
        for frame in range(frame_count):
            measurement = load(route / "measurements_full" / ("{:04d}.json".format(frame)))
            if int(measurement.get("frame_id", -1)) != frame:
                raise ValueError("Measurement frame mismatch: {} {}".format(route_id, frame))
            navigation = measurement.get("navigation", {})
            route_index = int(navigation.get("route_index", -1))
            command = int(measurement.get("gt_command", -1))
            if route_index < 0 or route_index >= len(options) or options[route_index] != command:
                raise ValueError("Recorded command disagrees with global path: {} {}".format(route_id, frame))
            if command not in COMMAND_TEXT:
                raise ValueError("Unsupported global-path command: {}".format(command))
            text = COMMAND_TEXT[command]
            if policy is not None:
                command, text = policy.instruction_for(measurement)
            frames.append((frame, int(measurement["native_sync_frame"]), command, text))

        start = 0
        for cursor in range(1, len(frames) + 1):
            changed = cursor == len(frames) or frames[cursor][2:] != frames[start][2:]
            if not changed:
                continue
            command = frames[start][2]
            record = {
                "schema_version": LOOKAHEAD_SCHEMA if policy is not None else SCHEMA,
                "route_id": route_id,
                "start_frame": frames[start][0],
                "stop_frame": frames[cursor - 1][0],
                "command": command,
                "text": frames[start][3],
                "source": "recorded_global_path",
                "route_sha256": sha256(route_file),
            }
            if policy is not None:
                record.update(instruction_policy="route_lookahead", lookahead_m=policy.lookahead_m,
                              progress_source="current_ego_route_projection",
                              curve_lookahead_m=policy.curve_lookahead_m,
                              curve_threshold_deg=policy.curve_threshold_deg)
            records.append(record)
            start = cursor

    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in records
    )
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError("Existing instruction file differs; refusing overwrite: {}".format(output))
        status = "verified_existing"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_suffix(output.suffix + ".partial")
        with pending.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        pending.replace(output)
        status = "created"
    print(json.dumps({
        "schema_version": LOOKAHEAD_SCHEMA if args.policy == "route_lookahead" else SCHEMA,
        "dataset_index_sha256": sha256(index),
        "routes": len(routes),
        "instruction_ranges": len(records),
        "output": str(output),
        "sha256": sha256(output),
        "status": status,
    }, indent=2))


if __name__ == "__main__":
    main()
