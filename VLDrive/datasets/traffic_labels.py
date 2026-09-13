"""Versioned, conservative traffic-loss masks; raw MORAI data stays immutable.

This sidecar never changes a class label, waypoint target, route association, or
distance threshold. Unknown affecting signals and red/green composite signals
are excluded from the traffic auxiliary loss only. It is not a GO/STOP policy.
The implementation is pure Python so provenance can be checked before CUDA.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .split_manifest import load_split_manifest, split_entries


SCHEMA_VERSION = "morai-traffic-labels-v2"
POLICY_VERSION = "preserve-label-mask-unknown-or-red-green-v1"
POLICY = {
    "version": POLICY_VERSION,
    "class_mapping": {"red": 0, "not_red": 1},
    "relabel": False,
    "unknown_affecting_signal": "mask_traffic_loss_only",
    "red_with_green_permission": "mask_traffic_loss_only",
    "no_affected_signal": "preserve_original_label_and_alignment_mask",
    "distance_and_route_association": "unchanged_no_distance_relabel",
    "waypoint_supervision": "unchanged_future_synchronized_ego_positions",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _route_path(dataset_root, relative):
    data = (Path(dataset_root).resolve() / "data").resolve()
    route = (data / relative).resolve()
    if data not in route.parents or any(
        part.startswith("_invalid") for part in route.relative_to(data).parts
    ):
        raise ValueError("Invalid traffic-sidecar route: {}".format(relative))
    return route


def traffic_record(relative, split, frame_id, measurement):
    """Deterministic policy decision without consulting future ego/control."""
    if int(measurement.get("frame_id", -1)) != frame_id:
        raise ValueError("Traffic source frame_id mismatch: {}:{}".format(relative, frame_id))
    native = int(measurement["native_sync_frame"])
    grpc_tick = measurement.get("grpc_label_tick_raw")
    nav = measurement.get("navigation", {})
    affected_id = nav.get("affected_light_id_raw")
    raw_state = nav.get("affected_light_state_raw")
    lights = measurement.get("traffic_lights", [])
    aligned = (measurement.get("simulator_labels_source") == "grpc"
               and grpc_tick == native)
    known_matching_state = any(
        light.get("tl_id") == affected_id and light.get("state") == raw_state
        for light in lights
    )
    available = bool(aligned)
    if not aligned:
        reason = "unaligned_or_non_grpc"
    elif affected_id and raw_state is None:
        available = False
        reason = "unknown_affected_signal"
    elif affected_id and not known_matching_state:
        available = False
        reason = "unverified_affected_signal"
    elif affected_id and (int(raw_state) & 1) and (int(raw_state) & (16 | 32)):
        available = False
        reason = "ambiguous_red_green_permission"
    elif affected_id:
        reason = "known_affected_signal_preserved"
    else:
        reason = "no_affected_signal_preserved"
    red = bool(measurement.get("is_red_light_present"))
    return {
        "route": relative,
        "split": split,
        "frame_id": frame_id,
        "native_sync_frame": native,
        "grpc_label_tick_raw": grpc_tick,
        "simulator_labels_source": measurement.get("simulator_labels_source"),
        "measurement_sha256": canonical_sha256(measurement),
        "affected_light_id_raw": affected_id,
        "affected_light_state_raw": raw_state,
        "affected_light_state_lmdrive": nav.get("affected_light_state_lmdrive"),
        "observed_signals": lights,
        "route_index": nav.get("route_index"),
        "route_link_index": nav.get("route_link_index"),
        "current_link_id": nav.get("current_link_id"),
        "gt_command": measurement.get("gt_command"),
        "near_command": measurement.get("command"),
        "original_red_flag": red,
        "original_available": bool(aligned),
        "traffic_light": 0 if red else 1,
        "traffic_available": available,
        "reason": reason,
    }


def build_traffic_label_sidecar(dataset_root, split_file):
    """Return an entirely reproducible artifact, without writing source data."""
    split_file = Path(split_file).resolve()
    split_spec = load_split_manifest(split_file)
    route_specs = {str(row["path"]): row for row in split_spec["routes"]}
    if len(route_specs) != len(split_spec["routes"]):
        raise ValueError("Duplicate route in split manifest")
    entries = {}
    for split in ("train", "validation"):
        for relative, frame in split_entries(split_spec, split):
            key = (relative, frame)
            if key in entries:
                raise ValueError("Overlapping/duplicate traffic-sidecar split frame")
            entries[key] = split
    sources, records = [], []
    for relative, spec in route_specs.items():
        route = _route_path(dataset_root, relative)
        aggregate_path = route / "measurements_all.json"
        measurements = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if not isinstance(measurements, list) or len(measurements) != int(spec["frame_count"]):
            raise ValueError("Traffic source frame count mismatch: {}".format(relative))
        sources.append({
            "route": relative,
            "frame_count": len(measurements),
            "measurements_all_sha256": sha256_file(aggregate_path),
            "route_json_sha256": sha256_file(route / "route.json"),
        })
        for (entry_route, frame), split in entries.items():
            if entry_route != relative:
                continue
            if not 0 <= frame < len(measurements):
                raise ValueError("Traffic source frame out of bounds")
            records.append(traffic_record(relative, split, frame, measurements[frame]))
    records.sort(key=lambda row: (row["route"], row["frame_id"]))
    if len(records) != len(entries):
        raise ValueError("Traffic sidecar does not cover all split frames")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "implementation_sha256": sha256_file(Path(__file__)),
        "split_sha256": sha256_file(split_file),
        "source_manifest_sha256": canonical_sha256(sources),
        "sources": sources,
        "records_sha256": canonical_sha256(records),
        "records": records,
    }


def sidecar_summary(artifact, split="train"):
    """Report selected split only; no validation performance is evaluated."""
    rows = [row for row in artifact["records"] if row["split"] == split]
    newly_masked = [row for row in rows
                    if row["original_available"] and not row["traffic_available"]]
    return {
        "schema_version": artifact["schema_version"],
        "policy_version": artifact["policy"]["version"],
        "split": split,
        "frames": len(rows),
        "reason_counts": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "original_available": sum(row["original_available"] for row in rows),
        "traffic_available": sum(row["traffic_available"] for row in rows),
        "newly_masked": len(newly_masked),
        "newly_masked_frame_ids": [row["frame_id"] for row in newly_masked],
        "relabeled": sum(row["traffic_light"] != (0 if row["original_red_flag"] else 1)
                         for row in rows),
        "split_sha256": artifact["split_sha256"],
        "source_manifest_sha256": artifact["source_manifest_sha256"],
        "records_sha256": artifact["records_sha256"],
        "sources": artifact["sources"],
        "waypoint_targets_changed": False,
        "distance_or_route_association_changed": False,
    }


def load_traffic_label_sidecar(path, dataset_root, split_file):
    """Fail closed on source/split/policy/coverage/tick/state/mask tampering."""
    path = Path(path).resolve()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("policy") != POLICY:
        raise ValueError("Traffic sidecar schema/policy mismatch")
    expected = build_traffic_label_sidecar(dataset_root, split_file)
    # Exact deterministic replay verifies values, reasons and original sources,
    # not just a digest which could have been updated alongside bad labels.
    if canonical_sha256(artifact) != canonical_sha256(expected):
        raise ValueError("Traffic sidecar source/split/provenance/records mismatch")
    records = {(row["route"], row["frame_id"]): row for row in artifact["records"]}
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "sidecar_sha256": sha256_file(path),
        "split_sha256": artifact["split_sha256"],
        "source_manifest_sha256": artifact["source_manifest_sha256"],
        "records_sha256": artifact["records_sha256"],
        "implementation_sha256": artifact["implementation_sha256"],
        "policy": artifact["policy"],
    }
    return records, provenance


def checked_traffic_target(records, relative, frame_id, measurement):
    """Also check the actual measurements_full record consumed by the loader."""
    row = records.get((relative, frame_id))
    if row is None:
        raise ValueError("Missing traffic-sidecar frame: {}:{}".format(relative, frame_id))
    if canonical_sha256(measurement) != row["measurement_sha256"]:
        raise ValueError("Traffic sidecar per-frame source mismatch: {}:{}".format(relative, frame_id))
    return int(row["traffic_light"]), bool(row["traffic_available"])
