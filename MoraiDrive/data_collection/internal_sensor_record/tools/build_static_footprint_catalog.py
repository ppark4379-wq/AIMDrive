#!/usr/bin/env python3
"""Export the LiDAR-verified footprint of each static model for preset generation.

Reads sealed static runs and their V5 geometry (read-only) and writes one JSON
for the Scenario Runner.  Each approved native-scale model records its V5
training box relative to the spawn pose written in the preset: centre offset
in the spawn frame (x along the preset yaw, y left), heading offset from the
preset yaw, and box size.  Only models the registry approved are exported.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.journal import digest
from async_capture.sensor_record_admission import verify_sealed_directory

SCHEMA = "morai_static_obstacle_footprints_v1"
NATIVE_SCALE = "1,1,1"
DEFAULT_OUTPUT = ROOT.parents[1] / "scenario_runner/config/static_obstacle_footprints_v1.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wrap_deg(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def preset_obstacles(episode):
    record = (load(episode / "raw/session.json").get("episode") or {}).get("static_obstacle_preset")
    if not record:
        raise ValueError("Run has no pinned static preset: " + str(episode))
    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError("Pinned static preset changed: " + str(path))
    return yaml.safe_load(path.read_text(encoding="utf-8"))["obstacles"], record


def observe_run(run, geometry_dir):
    episode = Path(run).resolve() / "episode_000"
    verify_sealed_directory(episode / "gt_timeline_v1")
    verify_sealed_directory(episode / geometry_dir)
    obstacles, preset = preset_obstacles(episode)
    headings = {}
    for frame in load(episode / "gt_timeline_v1/frames.json"):
        for item in frame.get("object_gt", {}).get("objects", []):
            if item.get("kind") == "obstacle":
                headings[(frame["index"], item["instance_key"])] = (
                    item["model_name"], float(item["heading_deg"]))
    rows = defaultdict(list)
    for frame in load(episode / geometry_dir / "frames.json"):
        for item in frame.get("objects", []):
            centre, size = item.get("training_center_world_xy"), item.get("training_size_wl_m")
            known = headings.get((frame["index"], item.get("instance_key")))
            if not (item.get("api_box_geometry_candidate") and centre and size and known):
                continue
            model, heading = known
            spawn = [row for row in obstacles if row["model"] == model]
            if len(spawn) != 1 or [float(v) for v in spawn[0]["scale"]] != [1.0, 1.0, 1.0]:
                continue  # ambiguous duplicate or non-native instance
            spawn = spawn[0]
            yaw = math.radians(float(spawn["yaw_deg"]))
            dx = centre[0] - float(spawn["position"]["x"])
            dy = centre[1] - float(spawn["position"]["y"])
            rows[model].append({
                "forward": dx * math.cos(yaw) + dy * math.sin(yaw),
                "left": -dx * math.sin(yaw) + dy * math.cos(yaw),
                "heading": wrap_deg(heading - float(spawn["yaw_deg"])),
                "width": float(size[0]), "length": float(size[1]),
            })
    return {"run": str(Path(run).resolve()), "preset": preset, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-observations", type=int, default=10)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Existing footprint catalog is preserved: " + str(args.output))
    registry = load(args.registry)
    tag = digest(args.registry)[:12]
    approved = {row["model_name"] for key, row in registry["models"].items()
                if key.startswith("obstacle|") and key.endswith("|" + NATIVE_SCALE)
                and row.get("approved")}
    by_model = defaultdict(list)
    runs = defaultdict(set)
    sources = []
    for run in args.run:
        result = observe_run(run, "object_geometry_audit_v5_" + tag)
        sources.append({"run": result["run"], "preset": result["preset"]})
        for model, rows in result["rows"].items():
            by_model[model].extend(rows)
            runs[model].add(Path(result["run"]).name)
    models, skipped = {}, {}
    for model in sorted(approved):
        rows = by_model.get(model, [])
        if len(rows) < args.min_observations:
            skipped[model] = "observations=%d" % len(rows)
            continue
        heading = statistics.median(row["heading"] for row in rows)
        if max(abs(wrap_deg(row["heading"] - heading)) for row in rows) > 2.0:
            skipped[model] = "heading_offset_unstable"
            continue
        models[model] = {
            "center_offset_spawn_xy": [round(statistics.median(r["forward"] for r in rows), 4),
                                       round(statistics.median(r["left"] for r in rows), 4)],
            "heading_offset_deg": round(heading, 3),
            "width_m": round(statistics.median(r["width"] for r in rows), 4),
            "length_m": round(statistics.median(r["length"] for r in rows), 4),
            "observations": len(rows),
            "runs": sorted(runs[model]),
        }
    document = {
        "schema": SCHEMA,
        "frame": ("Box centre offset in the preset spawn frame (x along yaw_deg, y left); "
                  "box heading = yaw_deg + heading_offset_deg; width across, length along it"),
        "scale_xyz": [1.0, 1.0, 1.0],
        "registry": {"path": str(args.registry.resolve()), "sha256": digest(args.registry)},
        "sources": sources,
        "models": models,
        "skipped_approved_models": skipped,
        "script_sha256": digest(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print("[FOOTPRINTS] %d models -> %s; skipped %s" % (len(models), args.output, skipped))


if __name__ == "__main__":
    main()
