#!/usr/bin/env python3
"""Independently verify direct gRPC light queries against the GT timeline.

This audit preserves query failures and transition uncertainty.  It proves
only that the offline timeline is a faithful, conservative reconstruction of
the timestamp-bracketed raw queries; it does not perform route/lane binding.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.alignment import PayloadReader
from async_capture.journal import digest, verify_run, write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from ssd_storage import require_ssd
from tools.build_sensor_record_gt_timeline import (
    direct_light_index, direct_light_value, selected_traffic_query,
)


def load(path):
    return json.loads(Path(path).read_text())


def canonical(values):
    return sorted(values, key=lambda row: row["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gt-timeline-dir", default="gt_timeline_v1")
    args = parser.parse_args()
    for path in (args.episode, args.output):
        require_ssd(path, reserve_bytes=0)
    if args.output.exists():
        raise FileExistsError(args.output)

    quality = args.episode / "quality"
    gt_dir = args.episode / args.gt_timeline_dir
    verify_sealed_directory(quality, "QUALITY_COMPLETE.json")
    gt_seal = verify_sealed_directory(gt_dir)
    index = load(quality / "INDEX_COMPLETE.json")
    raw = Path(index["raw_path"])
    manifest, _ = verify_run(raw, payloads=False)
    if manifest["run_id"] != index["raw_run_id"]:
        raise ValueError("Quality index and raw run differ")
    if gt_seal.get("raw_manifest_sha256") != index["raw_manifest_sha256"]:
        raise ValueError("GT timeline and raw run differ")

    acquisition = load(args.episode.parent / "sensor_contract.json")
    actor_source_units=int(acquisition.get('capture',{}).get('actor_ground_truth',{}).get(
        'source_timestamp_units_per_second',1000000))
    if actor_source_units<=0:raise ValueError('Invalid actor source clock units')
    light_contract = acquisition.get("capture", {}).get("traffic_light_ground_truth", {})
    required_ids = [str(value) for value in light_contract.get("light_ids", [])]
    if not required_ids or len(required_ids) != len(set(required_ids)):
        raise ValueError("Acquisition contract has no unique route light IDs")

    reader = PayloadReader(raw)
    query_rows = []
    attempt_failures = []
    final_failures = []
    recovered_queries = []
    batch_count = 0
    configured_mismatch = []
    schemas = Counter()
    total_attempts = 0
    try:
        for meta in load(quality / "active_signals.json"):
            if meta.get("role") != "traffic_light_inventory":
                continue
            batch_count += 1
            row = json.loads(reader.read(meta))
            schema=row.get("schema");schemas[str(schema)]+=1
            if schema not in ("morai_grpc_traffic_light_bracket_v2",
                              "morai_grpc_traffic_light_bracket_v3"):
                raise ValueError("Unexpected direct traffic schema")
            configured = [str(value) for value in row.get("configured_light_ids", [])]
            if configured != required_ids:
                configured_mismatch.append(meta["record_id"])
            before = row.get("before_actor_source_raw")
            after = row.get("after_actor_source_raw")
            if before is None or after is None or int(after) < int(before):
                raise ValueError("Invalid actor-source bracket in traffic batch")
            for requested, query in row.get("queries", {}).items():
                attempts=query.get('attempts',[query]);total_attempts+=len(attempts)
                for attempt in attempts:
                    attempt_response=attempt.get('response',{})
                    attempt_identity=attempt_response.get('id',{}).get('value')
                    if (attempt.get('valid_response_identity') is not True or
                            attempt_identity != requested):
                        attempt_failures.append({
                            "record_id":meta["record_id"],"seq":row.get("seq"),
                            "id":requested,"attempt_index":attempt.get('attempt_index',0),
                            "error":attempt.get("error"),"response_id":attempt_identity,
                        })
                if query.get('recovered_after_retry') is True:
                    recovered_queries.append({"record_id":meta["record_id"],
                                              "seq":row.get("seq"),"id":requested,
                                              "attempt_count":len(attempts)})
                selected=selected_traffic_query(query)
                response = selected.get("response", {})
                identity = response.get("id", {}).get("value")
                if selected.get("valid_response_identity") is not True or identity != requested:
                    final_failures.append({
                        "record_id": meta["record_id"],
                        "seq": row.get("seq"),
                        "id": requested,
                        "error": selected.get("error"),
                        "response_id": identity,
                    })
                    continue
                query_rows.append({
                    "before_ns": int(before) * 1_000_000_000 // actor_source_units,
                    "after_ns": int(after) * 1_000_000_000 // actor_source_units,
                    "record_id": meta["record_id"],
                    "value": {"id": str(identity), "color": int(response.get("color", 0))},
                })
    finally:
        reader.close()

    query_index = direct_light_index(query_rows)
    frames = load(gt_dir / "frames.json")
    reconstruction_mismatches = []
    missing_required = []
    for frame in frames:
        expected = canonical(direct_light_value(query_index, int(frame["lidar_source_ns"])))
        observed = canonical(frame["traffic_light_gt"].get("values", []))
        if expected != observed:
            reconstruction_mismatches.append(frame["index"])
        present = {row["id"] for row in observed}
        missing = [light_id for light_id in required_ids if light_id not in present]
        if missing:
            missing_required.append({"index": frame["index"], "ids": missing})

    transitions = []
    transition_anchors = 0
    incorrectly_kept = []
    for light_id, item in query_index.items():
        for before, after in zip(item["rows"], item["rows"][1:]):
            if before["value"]["color"] == after["value"]["color"]:
                continue
            affected = [frame for frame in frames
                        if before["after_ns"] <= int(frame["lidar_source_ns"]) <= after["before_ns"]]
            kept = [frame["index"] for frame in affected
                    if light_id in {value["id"] for value in frame["traffic_light_gt"].get("values", [])}]
            transition_anchors += len(affected)
            incorrectly_kept.extend({"index": index, "id": light_id} for index in kept)
            transitions.append({
                "id": light_id,
                "from_color": before["value"]["color"],
                "to_color": after["value"]["color"],
                "uncertainty_source_ns": [before["after_ns"], after["before_ns"]],
                "sensor_anchor_indices": [frame["index"] for frame in affected],
            })

    successful_by_id = Counter(row["value"]["id"] for row in query_rows)
    report = {
        "schema": "morai_sensor_record_direct_traffic_audit_v1",
        "raw_manifest_sha256": index["raw_manifest_sha256"],
        "gt_timeline_complete_sha256": digest(gt_dir / "COMPLETE.json"),
        "acquisition_contract_sha256": digest(args.episode.parent / "sensor_contract.json"),
        "required_route_light_ids": required_ids,
        "traffic_batches": batch_count,
        "traffic_schemas": dict(schemas),
        "query_attempts": total_attempts,
        "failed_attempts": len(attempt_failures),
        "failed_attempt_details": attempt_failures,
        "recovered_after_retry": len(recovered_queries),
        "recovered_query_details": recovered_queries,
        "valid_queries": len(query_rows),
        "failed_queries": len(final_failures),
        "failed_query_details": final_failures,
        "successful_queries_by_id": dict(successful_by_id),
        "configured_id_mismatch_batches": configured_mismatch,
        "sensor_frames": len(frames),
        "raw_to_timeline_reconstruction_mismatches": reconstruction_mismatches,
        "frames_missing_required_route_light_ids": missing_required,
        "detected_color_transitions": len(transitions),
        "sensor_anchors_inside_transition_uncertainty": transition_anchors,
        "transition_uncertainty_values_incorrectly_kept": incorrectly_kept,
        "structural_valid": not configured_mismatch and not reconstruction_mismatches and not incorrectly_kept,
        "complete_route_query_coverage": not missing_required,
        "query_failures_are_preserved": True,
        "route_signal_training_label_valid": False,
        "training_visible": False,
        "limitation": "Direct query transport/provenance audit only; lane/stopline binding and sensor/GT atomicity are not established.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "transitions.json", transitions)
    write_json(args.output / "COMPLETE.json", {
        "files": {name: digest(args.output / name) for name in ("report.json", "transitions.json")},
        "raw_manifest_sha256": index["raw_manifest_sha256"],
        "script_sha256": digest(Path(__file__)),
        "training_visible": False,
    })
    print(json.dumps(report), flush=True)
    if not report["structural_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
