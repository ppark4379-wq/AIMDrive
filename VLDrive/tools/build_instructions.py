#!/usr/bin/env python3
"""Build deterministic language instructions from the recorded global path."""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "morai-global-path-instructions-v1"
LOOKAHEAD_SCHEMA = "morai-global-path-instructions-v2"
SENSOR_RECORD_SCHEMA = "morai-sensor-record-instructions-v1"
CONFIG_SCHEMA = "morai-navigation-instructions-config-v1"
VLDRIVE_ROOT = Path(__file__).resolve().parents[1]
ARGUMENT_DEFAULTS = {
    "dataset_root": None, "output": None, "policy": None, "input_format": "auto",
    "map_root": None, "route_definition": None, "max_route_error_m": 10.0,
    "max_heading_error_deg": 60.0, "lookahead_m": 30.0,
    "curve_lookahead_m": 20.0, "curve_threshold_deg": 20.0, "max_sensor_gap_ms": 150.0,
}
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
        self.closed_route = bool(route_data.get("closed_route", False))
        if self.closed_route and (self.cumulative_m[-1] <= 0 or
                math.dist(self.points[0], self.points[-1]) > 1e-3):
            raise ValueError("Closed route must end at its first point")
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
        upcoming = []
        for index, command in self.turn_starts:
            distance = self.cumulative_m[index] - progress
            if index <= route_index:
                if not self.closed_route:
                    continue
                distance += self.cumulative_m[-1]
            if -1e-6 <= distance <= self.lookahead_m:
                upcoming.append((distance, index, command))
        if upcoming:
            return min(upcoming)[2]
        return 4


    def _point_at(self, progress):
        if self.closed_route:
            progress %= self.cumulative_m[-1]
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
        stop = start + self.curve_lookahead_m
        if self.closed_route:
            stop = min(stop, start + self.cumulative_m[-1])
        else:
            stop = min(self.cumulative_m[-1], stop)
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


def verified_source(candidates, expected_hash, label):
    """Relocate recorded Linux paths without accepting a different map/route."""
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Missing recorded SHA256 for " + label)
    for candidate in candidates:
        candidate = Path(candidate).expanduser().resolve()
        if not candidate.is_file():
            continue
        content = candidate.read_bytes()
        # Git can convert text files between LF and CRLF on Windows.
        lf = content.replace(b"\r\n", b"\n")
        if expected_hash in {hashlib.sha256(data).hexdigest()
                             for data in (content, lf, lf.replace(b"\n", b"\r\n"))}:
            return candidate
    raise ValueError("No SHA256-matching source for {}: {}".format(label, candidates))


def sensor_record_route(episode, args):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Sensor Record input requires PyYAML: python -m pip install pyyaml") from exc
    contract_path = episode.parent / "sensor_contract.json"
    report_path = episode / "route_signal_audit_v1/report.json"
    episode_path = episode / "episode.json"
    contract, report, metadata = load(contract_path), load(report_path), load(episode_path)
    traffic = contract["capture"]["traffic_light_ground_truth"]
    hashes = dict(traffic.get("source_sha256", {}))
    for key, value in report.get("route_source_sha256", {}).items():
        if key in hashes and hashes[key] != value:
            raise ValueError("Capture and route audit hashes disagree: " + key)
        hashes[key] = value
    for key, value in (("global_path", report["global_path_sha256"]),
                       ("route_definition", report["route_definition_sha256"])):
        if key in hashes and hashes[key] != value:
            raise ValueError("Route audit hashes disagree: " + key)
        hashes[key] = value
    if metadata.get("global_path_sha256") != hashes["global_path"]:
        raise ValueError("Episode and route audit global paths disagree")
    scenario = Path(__file__).resolve().parents[2] / "MoraiDrive/scenario_runner"
    recorded_map = Path(traffic["map_root"])
    map_roots = ([args.map_root] if args.map_root else
                 [recorded_map, scenario / "data" / recorded_map.name])
    sources = {}
    recorded_paths = report.get('route_source_resolution', {}).get('selected_paths', {})
    recorded_global = recorded_paths.get('global_path') or metadata.get('scenario_config', {}).get('global_path')
    global_name = Path(recorded_global).name if recorded_global else '2026_molit_comp_global_path.txt'
    for key, filename in (("global_path", global_name),
                          ("link_set", "link_set.json"), ("node_set", "node_set.json")):
        sources[key] = verified_source([Path(folder) / filename for folder in map_roots],
                                       hashes.get(key), key)
    definitions = ([args.route_definition] if args.route_definition else
                   [metadata.get("scenario_config", {}).get("route_definition_path", ""),
                    scenario / "config/full_loop_links.yaml"])
    sources["route_definition"] = verified_source(definitions, hashes["route_definition"], "route_definition")
    route_key = report["route_key"]
    if route_key != traffic.get("route_key"):
        raise ValueError("Capture and route audit route keys disagree")
    definition = yaml.safe_load(sources["route_definition"].read_text(encoding="utf-8-sig"))[route_key]
    links = {str(link["idx"]): link for link in load(sources["link_set"])}
    link_ids = list(definition["route_links"])
    points, options, point_links, point_link_indices = [], [], [], []
    for link_index, link_id in enumerate(link_ids):
        link = links[link_id]
        signal = str(link.get("related_signal") or "").lower()
        # Use only explicit turn semantics; straight metadata also appears on tunnels.
        command = 1 if signal in ("left", "left_unprotected") else (
            2 if signal in ("right", "right_unprotected") else 4)
        for point in link["points"]:
            points.append([float(point[0]), float(point[1])])
            options.append(command)
            point_links.append(link_id)
            point_link_indices.append(link_index)
    recorded = []
    for line in sources["global_path"].read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            recorded.append([float(value) for value in line.split()[:2]])
    if len(points) < 2 or len(points) != len(recorded) or any(
            math.dist(a, b) > 1e-3 for a, b in zip(points, recorded)):
        raise ValueError("Official global path does not match ordered MGeo link points")
    expert_path = episode / "runtime/expert/path.csv"
    if sha256(expert_path) != metadata.get("expert_inputs", {}).get("runtime/expert/path.csv"):
        raise ValueError("Recorded expert path hash mismatch")
    with expert_path.open(encoding="utf-8-sig", newline="") as stream:
        expert = [[float(row["x"]), float(row["y"])] for row in csv.DictReader(stream)]
    if len(expert) != len(recorded) or any(math.dist(a, b) > 1e-3 for a, b in zip(expert, recorded)):
        raise ValueError("Recorded expert path differs from the audited global path")
    route = {"nav_waypoints": recorded, "nav_road_options": options,
             "closed_route": math.dist(recorded[0], recorded[-1]) <= 1e-3}
    roundabout = definition.get("zones", {}).get("roundabout")
    roundabout_bounds = None
    if roundabout:
        checkpoints = {str(row["id"]): int(row["path_index"]) for row in definition["checkpoints"]}
        start = checkpoints[str(roundabout["ego_entry_checkpoint"])]
        stop = checkpoints[str(roundabout["ego_exit_checkpoint"])]
        if not 0 <= start < stop < len(points):
            raise ValueError("Invalid roundabout checkpoint range")
        roundabout_bounds = (start, stop)
    provenance = {key: {"path": str(path), "sha256": sha256(path),
                         "recorded_sha256": hashes[key]} for key, path in sources.items()}
    for key, path in (("episode", episode_path), ("sensor_contract", contract_path),
                      ("route_audit", report_path), ("expert_path", expert_path)):
        provenance[key] = {"path": str(path), "sha256": sha256(path)}
    return route, point_links, point_link_indices, roundabout_bounds, provenance


def indexed_timeline(path):
    rows = load(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty or invalid timeline: " + str(path))
    if any(type(row.get("index")) is not int or row["index"] != index
           for index, row in enumerate(rows)):
        raise ValueError("Timeline frame index mismatch: " + str(path))
    return rows


def sensor_record_episode(episode, args):
    route, point_links, link_indices, roundabout, provenance = sensor_record_route(episode, args)
    policy = RouteLookaheadPolicy(route, args.lookahead_m, args.curve_lookahead_m,
                                  args.curve_threshold_deg)
    timelines = {}
    for key, relative in (("route_frames", "route_signal_audit_v1/frames.json"),
                          ("gt_frames", "gt_timeline_v1/frames.json"),
                          ("associations", "native_associations/associations.json")):
        path = episode / relative
        timelines[key] = indexed_timeline(path)
        provenance[key] = {"path": str(path), "sha256": sha256(path)}
    if len({len(rows) for rows in timelines.values()}) != 1:
        raise ValueError("Sensor Record timelines have different lengths: " + str(episode))
    # Stable across selecting a basic-drive root, campaign, lap, or episode.
    episode_id = episode.parent.name + "/" + episode.name
    records, frames = [], []
    previous_stamp, previous_progress = None, None
    counts = Counter()
    for signal, gt, association in zip(timelines["route_frames"], timelines["gt_frames"],
                                        timelines["associations"]):
        index = signal["index"]
        stamp = int(signal["sensor_source_ns"])
        if (stamp != gt["lidar_source_ns"] or stamp != association["scene_time_hypothesis_ns"]
                or gt["native_filename_time_ns"] != association["native_filename_time_ns"]):
            raise ValueError("Sensor timestamp mismatch: {} frame {}".format(episode_id, index))
        if previous_stamp is not None and stamp <= previous_stamp:
            raise ValueError("Sensor source time must be strictly increasing")
        frame = {"index": index, "sensor_source_ns": stamp,
                 "native_filename_time_ns": association["native_filename_time_ns"],
                 "camera_files": association["camera_files"], "lidar_file": association["lidar_file"],
                 "route_index": signal.get("route_index"), "route_link_id": signal.get("route_link_id"),
                 "instruction_id": None, "navigation_valid": False}
        route_index = signal.get("route_index")
        reason = None
        progress = None
        if route_index is None:
            reason = "missing_audited_route_position"
        elif type(route_index) is not int or not 0 <= route_index < len(point_links):
            raise ValueError("Audited route index is out of range")
        elif (signal.get("route_link_id") != point_links[route_index]
              or signal.get("route_link_index") != link_indices[route_index]):
            raise ValueError("Audited route/link association differs from the recorded map")
        else:
            xy = gt.get("ego_xy")
            lane_error, heading_error = signal.get("lane_error_m"), signal.get("heading_error_deg")
            if (not isinstance(xy, list) or len(xy) != 2
                    or any(value is None or not math.isfinite(float(value))
                           for value in [*xy, lane_error, heading_error])):
                reason = "missing_or_nonfinite_localization"
            else:
                measurement = {"gps_x": xy[0], "gps_y": xy[1],
                               "navigation": {"route_index": route_index, "route_valid": True}}
                progress = policy._current_progress(measurement, route_index)
                projection = policy._point_at(progress)
                if max(abs(lane_error), math.dist(xy, projection)) > args.max_route_error_m:
                    reason = "route_distance_exceeds_navigation_gate"
                elif abs(heading_error) > args.max_heading_error_deg:
                    reason = "heading_exceeds_navigation_gate"
                elif previous_progress is not None and progress < previous_progress - 2.0:
                    # The traffic-light audit calls a valid closed-loop seam a reversal.
                    if not (policy.closed_route and previous_progress >= policy.cumulative_m[-1] - 25
                            and progress <= 50):
                        reason = "route_progress_reversed"
        if reason:
            frame["exclusion_reason"] = reason
            counts[reason] += 1
        else:
            command, text = policy.instruction_for(measurement)
            if roundabout and roundabout[0] <= route_index < roundabout[1]:
                command, text = 4, "Follow the planned route through the roundabout."
            # No signal.valid filter: that flag tests traffic-light binding, not navigation.
            continuous = (records and records[-1]["stop_frame"] == index - 1
                          and records[-1]["command"] == command and records[-1]["text"] == text
                          and stamp - previous_stamp <= round(args.max_sensor_gap_ms * 1_000_000))
            if continuous:
                records[-1].update(stop_frame=index, stop_sensor_source_ns=stamp)
            else:
                instruction_id = "{}:{:06d}".format(episode_id, index)
                records.append({
                    "schema_version": SENSOR_RECORD_SCHEMA, "instruction_id": instruction_id,
                    "episode_id": episode_id, "frame_domain": "native_association_index",
                    "start_frame": index, "stop_frame": index,
                    "start_sensor_source_ns": stamp, "stop_sensor_source_ns": stamp,
                    "command": command, "text": text, "source": "recorded_global_path",
                    "route_sha256": provenance["global_path"]["recorded_sha256"],
                    "instruction_policy": "route_lookahead", "lookahead_m": args.lookahead_m,
                    "curve_lookahead_m": args.curve_lookahead_m,
                    "curve_threshold_deg": args.curve_threshold_deg,
                    "progress_source": "audited_current_route_index_and_sensor_time_ego_xy",
                })
            frame.update(instruction_id=records[-1]["instruction_id"], navigation_valid=True)
            counts["labeled_frames"] += 1
            previous_progress = progress
        frames.append(frame)
        previous_stamp = stamp
    if not records:
        raise ValueError("Episode has no valid navigation frames: " + str(episode))
    return records, {"episode_id": episode_id, "episode_path": str(episode),
                     "total_frames": len(frames), "counts": dict(counts),
                     "instruction_ranges": len(records), "closed_route": policy.closed_route,
                     "sources": provenance, "frames": frames}


def build_sensor_record(root, output, args):
    if args.policy != "route_lookahead":
        raise ValueError("Sensor Record input supports --policy route_lookahead only")
    for name in ("max_route_error_m", "max_heading_error_deg", "max_sensor_gap_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    if args.max_heading_error_deg > 180:
        raise ValueError("max_heading_error_deg must not exceed 180")
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist: " + str(root))
    episodes = ([root] if (root / "episode.json").is_file() else
                sorted(path.parent for path in root.rglob("episode.json")))
    if not episodes:
        raise ValueError("No Sensor Record episode.json found under " + str(root))
    records, reports, ids = [], [], set()
    report_path = output.with_suffix(".report.json")
    if report_path == output:
        raise ValueError("Instruction output must not use the .report.json suffix")
    for episode in episodes:
        # Keep generated files out of sealed episode/capture inputs.
        if episode.parent == output.parent or episode.parent in output.parents:
            raise ValueError("Write instructions outside the original capture directory")
        rows, report = sensor_record_episode(episode, args)
        if report["episode_id"] in ids:
            raise ValueError("Duplicate episode identity: " + report["episode_id"])
        ids.add(report["episode_id"])
        records.extend(rows)
        reports.append(report)
    serialized = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records)
    report = {"schema_version": "morai-sensor-record-instruction-report-v1",
              "instruction_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
              "generator_sha256": sha256(Path(__file__)),
              "frame_domain": "native_association_index", "stop_frame_inclusive": True,
              "legacy_stage2_loader_compatible": False,
              "stage2_join": "Join by episode_id and sensor_source_ns; preserve exclusions and original indices.",
              "max_route_error_m": args.max_route_error_m,
              "max_heading_error_deg": args.max_heading_error_deg,
              "max_contiguous_sensor_gap_ns": round(args.max_sensor_gap_ms * 1_000_000),
              "roundabout_policy": "Generic route-follow inside recorded checkpoints; no inferred exit number.",
              "episodes": reports}
    if args.config is not None:
        report["configuration"] = {"path": str(args.config), "sha256": sha256(args.config),
                                   "effective_parameters": {
                                       key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items() if key != "config"}}
    contents = {output: serialized, report_path: json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"}
    # Check all outputs before writing any file; repeat runs are deterministic.
    for path, content in contents.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError("Existing instruction artifact differs; refusing overwrite: " + str(path))
    existed = all(path.exists() for path in contents)
    for path, content in contents.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(path.suffix + ".partial")
            with pending.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            pending.replace(path)
    print(json.dumps({"schema_version": SENSOR_RECORD_SCHEMA, "episodes": len(reports),
                      "instruction_ranges": len(records),
                      "labeled_frames": sum(row["counts"].get("labeled_frames", 0) for row in reports),
                      "excluded_frames": sum(row["total_frames"] - row["counts"].get("labeled_frames", 0)
                                             for row in reports),
                      "output": str(output), "report": str(report_path),
                      "status": "verified_existing" if existed else "created"}, indent=2))


def load_instruction_config(path):
    document = load(path)
    if not isinstance(document, dict) or document.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Expected config schema " + CONFIG_SCHEMA)
    expected = set(ARGUMENT_DEFAULTS) | {"schema"}
    if set(document) != expected:
        raise ValueError("Config keys mismatch (missing={}, unknown={})".format(
            sorted(expected - set(document)), sorted(set(document) - expected)))
    values = {key: document[key] for key in ARGUMENT_DEFAULTS}
    for key in ("dataset_root", "output", "map_root", "route_definition"):
        value = values[key]
        if value is None and key in ("map_root", "route_definition"):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} must be a nonempty path string".format(key))
        candidate = Path(value).expanduser()
        # Match the training configs: all relative data paths start at VLDrive/.
        values[key] = (VLDRIVE_ROOT / candidate).resolve()
    for key, choices in (("input_format", ("auto", "legacy", "sensor_record")),
                         ("policy", (None, "current_link", "route_lookahead"))):
        if values[key] not in choices:
            raise ValueError("Invalid {}: {!r}".format(key, values[key]))
    for key in ("max_route_error_m", "max_heading_error_deg", "lookahead_m",
                "curve_lookahead_m", "curve_threshold_deg", "max_sensor_gap_ms"):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be a finite positive number".format(key))
        values[key] = float(value)
    if values["max_heading_error_deg"] > 180 or values["curve_threshold_deg"] >= 180:
        raise ValueError("Heading limit must be <=180 and curve threshold must be <180 degrees")
    if round(values["max_sensor_gap_ms"] * 1_000_000) < 1:
        raise ValueError("max_sensor_gap_ms must represent at least one nanosecond")
    return values


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path,
                        help="JSON config; use alone, without individual parameter flags")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", choices=("current_link", "route_lookahead"),
                        help="Default: current_link for legacy data, route_lookahead for Sensor Record")
    parser.add_argument("--input-format", choices=("auto", "legacy", "sensor_record"))
    parser.add_argument("--map-root", type=Path, help="Relocated MGeo directory (verified against capture hashes)")
    parser.add_argument("--route-definition", type=Path, help="Relocated route YAML (verified against capture hash)")
    parser.add_argument("--max-route-error-m", type=float,
                        help="Sensor Record navigation localization gate; allows obstacle avoidance (default: 10 m)")
    parser.add_argument("--max-heading-error-deg", type=float)
    parser.add_argument("--lookahead-m", type=float,
                        help="Route arc distance for route_lookahead instructions (default: 30 m)")
    parser.add_argument("--curve-lookahead-m", type=float)
    parser.add_argument("--curve-threshold-deg", type=float)
    parser.add_argument("--max-sensor-gap-ms", type=float, help="Maximum gap within an instruction range (default: 150 ms)")
    supplied = vars(parser.parse_args(argv))
    config = supplied.pop("config", None)
    if config is not None:
        if supplied:
            parser.error("--config cannot be combined with parameter flags; edit the config file")
        config = config.expanduser().resolve()
        try:
            values = load_instruction_config(config)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    else:
        values = dict(ARGUMENT_DEFAULTS, **supplied)
        if values["dataset_root"] is None or values["output"] is None:
            parser.error("provide --config, or both --dataset-root and --output")
    return argparse.Namespace(config=config, **values)


def main():
    # The versioned LMDrive pipeline shares semantics with the online observer.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', type=Path)
    known, rest = probe.parse_known_args()
    if known.config and load(known.config).get('schema') == 'lmdrive-navigation-build-v1':
        if rest:
            probe.error('--config cannot be combined with parameter flags')
        import sys
        if str(VLDRIVE_ROOT) not in sys.path:
            sys.path.insert(0, str(VLDRIVE_ROOT))
        from navigation.build import build
        build(known.config)
        return
    args = parse_arguments()
    root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    input_format = args.input_format
    if input_format == "auto":
        input_format = "legacy" if (root / "dataset_index.txt").is_file() else "sensor_record"
    if input_format == "sensor_record":
        args.policy = args.policy or "route_lookahead"
        build_sensor_record(root, output, args)
        return
    args.policy = args.policy or "current_link"
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
