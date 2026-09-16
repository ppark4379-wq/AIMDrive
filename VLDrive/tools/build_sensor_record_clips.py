#!/usr/bin/env python3
"""Build causal navigation-instruction clips from sealed Sensor Record data.

Only the official route, current ``route_index``, frame identity and simulation
time are read.  Future waypoints, objects, traffic lights and future Ego poses
are deliberately outside this builder's input contract.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.morai_sensor_record_dataset import PathResolver, digest, load_publications
from datasets.morai_sensor_record_clip_dataset import (
    CLIP_META_SCHEMA, CLIP_SCHEMA, publication_set_sha256, sha256_file)


COMMAND_TEXT = {
    1: "Proceed ahead and make a left turn.",
    2: "Proceed ahead and make a right turn.",
    3: "Please proceed straight at the next intersection.",
    4: "Proceed along this route.",
}


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def resolve_inputs(args):
    config_path = args.config or ROOT / "configs/morai_transfer.local.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    manifests = list(args.manifest_index or config.get("manifest_indices", []))
    if not manifests and config.get("manifest_index"):
        manifests.append(config["manifest_index"])
    campaigns = list(config.get("manifest_campaigns", [])) if args.manifest_index is None else []
    campaigns += list(args.manifest_campaign)
    for campaign in campaigns:
        campaign = Path(campaign).expanduser()
        candidates = sorted(campaign.glob("validation_*/training_dataset*/manifest_index.json"),
                            key=lambda path: (path.stat().st_mtime_ns, str(path)))
        if not candidates:
            raise ValueError("Campaign has no sealed training publication: {}".format(campaign))
        manifests.append(candidates[-1])
    if not manifests:
        raise ValueError("Provide --manifest-index/--manifest-campaign or a MORAI transfer config")
    resolver = PathResolver(config.get("path_mappings", []))
    return [Path(path).expanduser().resolve() for path in manifests], resolver


def load_route_assets(map_root, route_definition, global_path, route_key):
    map_root = Path(map_root).expanduser().resolve()
    route_definition = Path(route_definition).expanduser().resolve()
    global_path = Path(global_path).expanduser().resolve()
    links_path = map_root / "link_set.json"
    for path in (links_path, route_definition, global_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    links = {str(row["idx"]): row for row in json.loads(links_path.read_text(encoding="utf-8"))}
    definitions = yaml.safe_load(route_definition.read_text(encoding="utf-8-sig")) or {}
    definition = definitions.get(route_key)
    if not isinstance(definition, dict):
        raise ValueError("Missing route definition: {}".format(route_key))
    link_ids = [str(value) for value in definition.get("route_links", [])]
    if len(link_ids) < 2 or any(link_id not in links for link_id in link_ids):
        raise ValueError("Official route contains missing MGeo links")
    points = []
    with global_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            fields = value.split()
            if len(fields) < 2:
                raise ValueError("{}:{} needs x y [z]".format(global_path, line_number))
            points.append((float(fields[0]), float(fields[1])))
    reference, options = [], []
    for link_id in link_ids:
        link = links[link_id]
        signal = str(link.get("related_signal") or "").lower()
        command = 1 if "left" in signal else 2 if "right" in signal else 3 if signal == "straight" else 4
        for point in link.get("points", []):
            reference.append((float(point[0]), float(point[1])))
            options.append(command)
    if len(points) != len(reference) or any(
            math.hypot(a[0] - b[0], a[1] - b[1]) > 1e-3
            for a, b in zip(points, reference)):
        raise ValueError("Official global path no longer exactly matches the sealed MGeo route links")
    return points, options, {
        "map_root": str(map_root),
        "link_set_sha256": digest(links_path),
        "route_definition": str(route_definition),
        "route_definition_sha256": digest(route_definition),
        "global_path": str(global_path),
        "global_path_sha256": digest(global_path),
        "route_key": route_key,
    }


class OfficialRouteInstructionPolicy:
    """Causal LMDrive-style instruction from route geometry and current progress."""

    def __init__(self, points, options, *, lookahead_m=30.0,
                 curve_lookahead_m=20.0, curve_threshold_deg=20.0):
        if len(points) < 2 or len(points) != len(options):
            raise ValueError("Route point/command lengths differ")
        self.points, self.options = list(points), list(options)
        self.lookahead_m = float(lookahead_m)
        self.curve_lookahead_m = float(curve_lookahead_m)
        self.curve_threshold_deg = float(curve_threshold_deg)
        if min(self.lookahead_m, self.curve_lookahead_m) <= 0:
            raise ValueError("Instruction lookaheads must be positive")
        self.cumulative = [0.0]
        for before, after in zip(self.points, self.points[1:]):
            self.cumulative.append(self.cumulative[-1] + math.hypot(
                after[0] - before[0], after[1] - before[1]))
        self.total_m = self.cumulative[-1]
        self.closed = self.total_m > 0 and math.hypot(
            self.points[0][0] - self.points[-1][0],
            self.points[0][1] - self.points[-1][1]) <= 1e-3
        self.turn_starts = [(index, command) for index, command in enumerate(self.options)
                            if command in (1, 2)
                            and (index == 0 or self.options[index - 1] != command)]

    def _forward_distance(self, start, stop):
        value = self.cumulative[stop] - self.cumulative[start]
        if value < 0 and self.closed:
            value += self.total_m
        return value

    def _index_ahead(self, start, distance):
        target = self.cumulative[start] + distance
        if self.closed:
            target %= self.total_m
        elif target >= self.total_m:
            return len(self.points) - 1
        return min(bisect.bisect_left(self.cumulative, target), len(self.points) - 1)

    def instruction(self, route_index):
        route_index = int(route_index)
        if not 0 <= route_index < len(self.points):
            raise ValueError("route_index outside sealed global path")
        current = int(self.options[route_index])
        if current in (1, 2):
            return current, COMMAND_TEXT[current]
        candidates = []
        for index, command in self.turn_starts:
            distance = self._forward_distance(route_index, index)
            if 1e-6 < distance <= self.lookahead_m:
                candidates.append((distance, command))
        if candidates:
            command = min(candidates)[1]
            return command, COMMAND_TEXT[command]
        end = self._index_ahead(route_index, self.curve_lookahead_m)
        before = self._index_ahead(route_index, min(3.0, self.curve_lookahead_m / 3.0))
        end_before_distance = max(0.0, self._forward_distance(route_index, end) - 3.0)
        end_before = self._index_ahead(route_index, end_before_distance)
        a, b = self.points[route_index], self.points[before]
        c, d = self.points[end_before], self.points[end]
        if before != route_index and end != end_before:
            heading_a = math.atan2(b[1] - a[1], b[0] - a[0])
            heading_b = math.atan2(d[1] - c[1], d[0] - c[0])
            delta = math.degrees(math.atan2(math.sin(heading_b - heading_a),
                                            math.cos(heading_b - heading_a)))
            if abs(delta) >= self.curve_threshold_deg:
                side = "left" if delta > 0 else "right"
                return 4, "Follow the road as it curves to the {}.".format(side)
        return 4, COMMAND_TEXT[4]


def build_clips(manifests, policy):
    clips, counts = [], {"train_frames": 0, "val_frames": 0,
                         "train_clips": 0, "val_clips": 0,
                         "time_discontinuities": 0, "instruction_boundaries": 0}
    for split in ("train", "val"):
        for manifest_path, document in manifests[split]:
            if document.get("source_clock") != "sensor_record_simulation_time":
                raise ValueError("Clip building requires Sensor Record simulation time")
            manifest_sha = digest(manifest_path)
            document_contract = document.get("navigation_contract", {})
            active_policy = policy
            if isinstance(policy, dict):
                route_identity = (
                    document_contract.get("global_path_sha256"),
                    document_contract.get("route_definition_sha256"),
                    document_contract.get("route_key"),
                )
                active_policy = policy.get(route_identity)
                if active_policy is None:
                    raise ValueError("No verified route assets for publication navigation contract")
            episode_id = "{}_{}".format(Path(document["episode"]).parent.name,
                                         manifest_sha[:12])
            entries = document["entries"]
            times = [int(row["sensor_time_ns"]) for row in entries]
            diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
            if not diffs:
                raise ValueError("Episode has no advancing Sensor Record time")
            period = int(statistics.median(diffs))
            if not 50_000_000 <= period <= 200_000_000:
                raise ValueError("Unexpected Sensor Record cadence: {} ns".format(period))
            tolerance = max(5_000_000, int(period * 0.1))
            annotated = []
            for entry in entries:
                # Intentionally inspect only present-time route progress and identities.
                navigation = entry["navigation"]
                command, text = active_policy.instruction(navigation["route_index"])
                annotated.append((entry, command, text))
            counts[split + "_frames"] += len(annotated)
            begin = 0
            for cursor in range(1, len(annotated) + 1):
                boundary = cursor == len(annotated)
                reason = None
                if not boundary:
                    before, after = annotated[cursor - 1], annotated[cursor]
                    delta = int(after[0]["sensor_time_ns"]) - int(before[0]["sensor_time_ns"])
                    if (int(after[0]["frame_id"]) != int(before[0]["frame_id"]) + 1
                            or abs(delta - period) > tolerance):
                        boundary, reason = True, "time"
                    elif after[1:] != before[1:]:
                        boundary, reason = True, "instruction"
                if not boundary:
                    continue
                segment = annotated[begin:cursor]
                first, last = segment[0][0], segment[-1][0]
                clip_number = counts[split + "_clips"]
                clips.append({
                    "schema_version": CLIP_SCHEMA,
                    "clip_id": "{}_{:05d}".format(episode_id, clip_number),
                    "episode_id": episode_id,
                    "split": split,
                    "source_manifest": str(manifest_path),
                    "source_manifest_sha256": manifest_sha,
                    "start_frame_id": int(first["frame_id"]),
                    "stop_frame_id": int(last["frame_id"]),
                    "start_sensor_time_ns": int(first["sensor_time_ns"]),
                    "stop_sensor_time_ns": int(last["sensor_time_ns"]),
                    "expected_period_ns": period,
                    "frame_count": len(segment),
                    "command": int(segment[0][1]),
                    "instruction": str(segment[0][2]),
                    "instruction_source": "sealed_official_global_path_plus_current_route_index_v1",
                })
                counts[split + "_clips"] += 1
                if reason == "time":
                    counts["time_discontinuities"] += 1
                elif reason == "instruction":
                    counts["instruction_boundaries"] += 1
                begin = cursor
    return clips, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest-index", action="append", type=Path)
    parser.add_argument("--manifest-campaign", action="append", type=Path, default=[])
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--route-definition", type=Path, required=True)
    parser.add_argument("--global-path", type=Path, required=True)
    parser.add_argument(
        "--additional-route-bundle", nargs=3, action="append", default=[],
        metavar=("MAP_ROOT", "ROUTE_DEFINITION", "GLOBAL_PATH"),
        help="Repeat for an older sealed route revision that is still present in publications")
    parser.add_argument("--route-key", default="full_loop")
    parser.add_argument("--lookahead-m", type=float, default=30.0)
    parser.add_argument("--curve-lookahead-m", type=float, default=20.0)
    parser.add_argument("--curve-threshold-deg", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexes, resolver = resolve_inputs(args)
    publication_info, manifests = load_publications(indexes, resolver)
    bundle_arguments = [(args.map_root, args.route_definition, args.global_path)] + [
        tuple(Path(value) for value in row) for row in args.additional_route_bundle]
    policies, route_identities = {}, []
    for map_root, route_definition, global_path in bundle_arguments:
        points, options, route_identity = load_route_assets(
            map_root, route_definition, global_path, args.route_key)
        key = (route_identity["global_path_sha256"],
               route_identity["route_definition_sha256"], args.route_key)
        if key in policies:
            continue
        policies[key] = OfficialRouteInstructionPolicy(
            points, options, lookahead_m=args.lookahead_m,
            curve_lookahead_m=args.curve_lookahead_m,
            curve_threshold_deg=args.curve_threshold_deg)
        route_identity["route_point_count"] = len(points)
        route_identities.append(route_identity)
    for split in manifests.values():
        for _, document in split:
            contract = document.get("navigation_contract", {})
            key = (contract.get("global_path_sha256"),
                   contract.get("route_definition_sha256"),
                   contract.get("route_key"))
            if key not in policies:
                raise ValueError("Publication navigation contract differs from every supplied route bundle")
            point_count = contract.get("route_point_count")
            if point_count is not None:
                matching = next(row for row in route_identities
                                if (row["global_path_sha256"],
                                    row["route_definition_sha256"], row["route_key"]) == key)
                if int(point_count) != int(matching["route_point_count"]):
                    raise ValueError("Publication route point count differs from supplied route")
    clips, counts = build_clips(manifests, policies)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Use a new semantic clip output directory: {}".format(output))
    output.mkdir(parents=True)
    clips_path = output / "clips.jsonl"
    write_jsonl(clips_path, clips)
    meta = {
        "schema_version": CLIP_META_SCHEMA,
        "status": "complete",
        "publication_set_sha256": publication_set_sha256(publication_info["publications"]),
        "publications": publication_info["publications"],
        "clips_sha256": sha256_file(clips_path),
        "routes": route_identities,
        "policy": {
            "name": "causal_route_lookahead_v1",
            "lookahead_m": args.lookahead_m,
            "curve_lookahead_m": args.curve_lookahead_m,
            "curve_threshold_deg": args.curve_threshold_deg,
            "teacher_inputs_used": [],
        },
        "counts": counts,
    }
    write_json(output / "clip_meta.json", meta)
    print(json.dumps({"output": str(output), **counts}, indent=2))


if __name__ == "__main__":
    main()
