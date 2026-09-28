#!/usr/bin/env python3
"""Build the per-model box-source registry from sealed runs and native LiDAR.

Read-only for every input run; writes one new registry JSON.  Each approved
model records which centre and size source contained the most LiDAR points
with the smallest footprint, and the evidence behind it.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.journal import digest, write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from morai_pretraining.fused_geometry import (
    DEFAULT_REGISTRY_POLICY, REGISTRY_SCHEMA, lidar_world_points, load_ros_object_sizes,
    fit_center, model_key, score_observation, select_model_geometry,
)
from morai_pretraining.sensor_record_manifest import dense_ego_timeline, interpolate_ego, read_native_lidar
from ssd_storage import SSD_ROOT


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def instance_scales(episode):
    """Hash-linked preset scale per obstacle instance, from the newest scale audit."""
    for name in ("object_geometry_audit_v4", "object_geometry_audit_v3"):
        folder = episode / name
        if (folder / "COMPLETE.json").is_file() and (folder / "scale_evidence.json").is_file():
            verify_sealed_directory(folder)
            evidence = load(folder / "scale_evidence.json")
            return {key: row["scale_xyz"] for key, row in evidence.get("instances", {}).items()
                    if row.get("valid") is True}, name
    return {}, None


def v4_centres(episode):
    folder = episode / "object_geometry_audit_v4"
    if not (folder / "COMPLETE.json").is_file():
        return {}
    verify_sealed_directory(folder)
    return {(frame["index"], obj.get("instance_key")): obj["training_center_world_xy"]
            for frame in load(folder / "frames.json") for obj in frame.get("objects", [])
            if obj.get("training_center_world_xy") is not None}


def pivot_offsets(episode):
    """Observed pivot-to-box offset per instance from the V4 pivot evidence."""
    path = episode / "object_geometry_audit_v4/pivot_evidence.json"
    if not path.is_file():
        return {}
    verify_sealed_directory(path.parent)
    return {key: row["offset_xyz_m"][:2] for key, row in load(path).get("instances", {}).items()
            if row.get("valid") is True}


def observe_run(run, policy=DEFAULT_REGISTRY_POLICY, fit_offsets=None):
    run = Path(run).resolve()
    episode = run / "episode_000"
    for folder, seal in (("quality", "QUALITY_COMPLETE.json"), ("gt_timeline_v1", "COMPLETE.json"),
                         ("native_associations", "COMPLETE.json"), ("actor_inventory_audit_v1", "COMPLETE.json")):
        verify_sealed_directory(episode / folder, seal)
    active = load(episode / "quality/active_signals.json")
    raw_hash = load(episode / "actor_inventory_audit_v1/report.json")["raw_manifest_sha256"]
    ros = load_ros_object_sizes(episode, active, raw_hash, SSD_ROOT)["actors"]
    scales, scale_source = instance_scales(episode)
    centres = v4_centres(episode)
    pivots = pivot_offsets(episode)
    times, values = dense_ego_timeline(active)
    associations = {row["index"]: row for row in load(episode / "native_associations/associations.json")}
    observations = defaultdict(list)
    for frame in load(episode / "gt_timeline_v1/frames.json"):
        association = associations.get(frame["index"])
        objects = frame.get("object_gt", {}).get("objects", [])
        if association is None or not objects:
            continue
        ego = interpolate_ego(times, values, int(frame["lidar_source_ns"]))
        chosen = []
        for item in objects:
            if math.hypot(item["position"][0] - ego["x"], item["position"][1] - ego["y"]) >= policy["max_range_m"]:
                continue
            scale = scales.get(item.get("instance_key")) if item.get("kind") == "obstacle" else None
            if item.get("kind") == "obstacle" and scale is None:
                continue  # unknown instance scale: the model size is not identified
            key = model_key(item, scale)
            if len(observations[key]) < policy["max_observations_per_model_per_run"]:
                chosen.append((key, item))
        if not chosen:
            continue
        world, height = lidar_world_points(read_native_lidar(run / association["lidar_file"]), ego)
        for key, item in chosen:
            evidence = ros.get(str(item.get("id")))
            if evidence is not None and evidence["kind"] != item.get("kind"):
                evidence = None  # ids are compared only within one actor type
            others = [other for other in objects if other is not item
                      and math.hypot(other["position"][0] - item["position"][0],
                                     other["position"][1] - item["position"][1]) < 12.0]
            extra = {"v4": centres.get((frame["index"], item.get("instance_key")))}
            offset = (fit_offsets or {}).get(item.get("model_name"))
            observed = pivots.get(item.get("instance_key"))
            if offset is not None and observed is not None:
                extra["fit"] = fit_center(item, observed, offset)
            row = score_observation(world, height, item, evidence, extra, policy, others)
            if row is not None:
                observations[key].append(dict(row, lidar_index=frame["index"]))
    return {"run": str(run), "raw_manifest_sha256": raw_hash,
            "gt_timeline_seal_sha256": digest(episode / "gt_timeline_v1/COMPLETE.json"),
            "native_associations_seal_sha256": digest(episode / "native_associations/COMPLETE.json"),
            "scale_source": scale_source, "observations": dict(observations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--center-candidates", type=Path,
                        help="Reviewed pivot-relative centre candidates (status CENTER_CORRECTION_CANDIDATE)")
    args = parser.parse_args()
    fit_offsets, fit_source = {}, None
    if args.center_candidates:
        document = load(args.center_candidates)
        fit_offsets = {row["model"]: row["candidate_actor_local_xyz_m"][:2] for row in document["models"]
                       if row.get("status") == "CENTER_CORRECTION_CANDIDATE"
                       and row.get("candidate_actor_local_xyz_m")}
        fit_source = {"path": str(args.center_candidates.resolve()),
                      "sha256": digest(args.center_candidates), "models": fit_offsets}
    if args.output.exists():
        raise FileExistsError(args.output)
    by_model = defaultdict(list)
    sources = []
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        from functools import partial
        for result in pool.map(partial(observe_run, fit_offsets=fit_offsets), args.run):
            sources.append({key: value for key, value in result.items() if key != "observations"})
            for key, rows in result["observations"].items():
                by_model[key].extend(rows)
            print("[REGISTRY] %s: %d models" % (Path(result["run"]).name, len(result["observations"])), flush=True)
    models = {}
    for key, rows in sorted(by_model.items()):
        entry = select_model_geometry(rows, DEFAULT_REGISTRY_POLICY)
        kind, model, scale = key.split("|")
        models[key] = dict(entry, kind=kind, model_name=model, scale_xyz=[float(v) for v in scale.split(",")])
        if entry["center_source"] == "fit":
            models[key]["center_offset_local_xy"] = fit_offsets[model]
    registry = {"schema": REGISTRY_SCHEMA, "policy": dict(DEFAULT_REGISTRY_POLICY),
                "sources": sources, "center_candidates": fit_source, "models": models,
                "approved_models": sorted(key for key, row in models.items() if row["approved"]),
                "script_sha256": digest(Path(__file__)),
                "module_sha256": digest(ROOT / "morai_pretraining/fused_geometry.py"),
                "training_visible": False,
                "limitation": ("LiDAR sees visible faces only; containment with the smallest footprint "
                               "selects among MORAI-reported boxes and never fits a new size.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, registry)
    for key in registry["approved_models"]:
        row = models[key]
        best = row["candidates"][row["center_source"] + "+" + row["size_source"]]
        print("[REGISTRY] approved %-40s centre=%s size=%s containment=%.2f n=%d" % (
            key, row["center_source"], row["size_source"], best["median_containment"], best["observations"]))
    print("[REGISTRY] %d/%d models approved -> %s" % (len(registry["approved_models"]), len(models), args.output))


if __name__ == "__main__":
    main()
