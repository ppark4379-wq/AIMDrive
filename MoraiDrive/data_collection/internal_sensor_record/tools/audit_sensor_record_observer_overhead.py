#!/usr/bin/env python3
"""Measure passive actor/traffic observer RPC cost from one sealed raw run."""
import argparse
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.alignment import PayloadReader
from async_capture.journal import digest, verify_run, write_json
from async_capture.portable_paths import recorded_episode_directory
from ssd_storage import require_ssd, SSD_ROOT


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}

    def percentile(fraction):
        index = fraction * (len(values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "p95": percentile(.95),
        "max": values[-1],
    }


def correlation(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    return numerator / (dx * dy) ** .5 if dx and dy else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.episode, args.output):
        require_ssd(path, reserve_bytes=0)
    if args.output.exists():
        raise FileExistsError(args.output)

    quality = args.episode / "quality"
    seal = json.loads((quality / "QUALITY_COMPLETE.json").read_text())
    for name, expected in seal["files"].items():
        if digest(quality / name) != expected:
            raise ValueError("Changed quality evidence: " + name)
    index = json.loads((quality / "INDEX_COMPLETE.json").read_text())
    raw = recorded_episode_directory(index["raw_path"], args.episode, "raw", SSD_ROOT)
    manifest, _ = verify_run(raw, payloads=False)
    if digest(raw / "raw_manifest.json") != index["raw_manifest_sha256"]:
        raise ValueError("Quality index refers to another raw run")

    rows = json.loads((quality / "active_signals.json").read_text())
    reader = PayloadReader(raw)
    actor_rpc_ms = []
    actor_counts = []
    actor_sources = []
    traffic_rpc_ms = []
    traffic_query_rpc_ms = []
    traffic_query_count = 0
    traffic_query_failures = 0
    traffic_query_attempt_count = 0
    traffic_query_attempt_failures = 0
    traffic_queries_recovered = 0
    actor_identity_rpc_ms = []
    actor_identity_failures = 0
    payload_errors = []
    try:
        for row in rows:
            role = row.get("role")
            if role not in ("actor_inventory", "actor_identity", "traffic_light_inventory"):
                continue
            try:
                payload = json.loads(reader.read(row))
            except Exception as error:
                payload_errors.append({"record_id": row.get("record_id"),
                                       "error": type(error).__name__ + ": " + str(error)})
                continue
            if role == "actor_inventory":
                if "response" not in payload or "rpc_duration_ns" not in payload:
                    continue
                actor_rpc_ms.append(float(payload["rpc_duration_ns"]) / 1e6)
                actor_counts.append(len(payload["response"].get("states", [])))
                actor_sources.append(int(payload["response"]["timestamp"]))
            elif role == 'actor_identity':
                if 'rpc_duration_ns' in payload:
                    actor_identity_rpc_ms.append(float(payload['rpc_duration_ns'])/1e6)
                actor_identity_failures += int(
                    payload.get('valid_nonempty_model_name') is not True)
            else:
                if "rpc_duration_ns" in payload:
                    traffic_rpc_ms.append(float(payload["rpc_duration_ns"]) / 1e6)
                for query in payload.get("queries", {}).values():
                    traffic_query_count += 1
                    attempts=query.get('attempts',[query])
                    traffic_query_attempt_count += len(attempts)
                    for attempt in attempts:
                        if "rpc_duration_ns" in attempt:
                            traffic_query_rpc_ms.append(float(attempt["rpc_duration_ns"]) / 1e6)
                        traffic_query_attempt_failures += int(
                            attempt.get('valid_response_identity') is not True)
                    traffic_queries_recovered += int(query.get('recovered_after_retry') is True)
                    if query.get("valid_response_identity") is not True:
                        traffic_query_failures += 1
    finally:
        reader.close()

    consecutive_unique = []
    for source in actor_sources:
        if not consecutive_unique or source != consecutive_unique[-1]:
            consecutive_unique.append(source)
    wall_ns = int(manifest["wall_duration_ns"])
    actor_rpc_ns = int(sum(actor_rpc_ms) * 1e6)
    traffic_rpc_ns = int(sum(traffic_rpc_ms) * 1e6)
    report = {
        "schema": "morai_sensor_record_observer_overhead_v1",
        "actor_rpc_ms": distribution(actor_rpc_ms),
        "actor_count": distribution(actor_counts),
        "actor_count_rpc_duration_correlation": correlation(actor_counts, actor_rpc_ms),
        "actor_source_samples": len(actor_sources),
        "actor_source_consecutive_unique": len(consecutive_unique),
        "actor_source_duplicate_fraction": (
            1 - len(consecutive_unique) / len(actor_sources) if actor_sources else None),
        "traffic_batch_rpc_ms": distribution(traffic_rpc_ms),
        "traffic_query_rpc_ms": distribution(traffic_query_rpc_ms),
        "traffic_query_count": traffic_query_count,
        "traffic_query_failures": traffic_query_failures,
        "traffic_query_attempt_count": traffic_query_attempt_count,
        "traffic_query_attempt_failures": traffic_query_attempt_failures,
        "traffic_queries_recovered_after_retry": traffic_queries_recovered,
        "actor_identity_rpc_ms": distribution(actor_identity_rpc_ms),
        "actor_identity_failures": actor_identity_failures,
        "raw_wall_seconds": wall_ns / 1e9,
        "actor_rpc_wall_occupancy_fraction": actor_rpc_ns / wall_ns,
        "traffic_rpc_wall_occupancy_fraction": traffic_rpc_ns / wall_ns,
        "combined_observer_rpc_wall_occupancy_fraction": (
            actor_rpc_ns + traffic_rpc_ns) / wall_ns,
        "payload_errors": payload_errors,
        "interpretation": (
            "RPC occupancy is the passive observer thread's measured service time. "
            "It can overlap simulator/recorder work and is therefore not a predicted RTF gain."),
        "raw_manifest_sha256": index["raw_manifest_sha256"],
        "quality_seal_sha256": digest(quality / "QUALITY_COMPLETE.json"),
        "training_visible": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "COMPLETE.json", {
        "schema": report["schema"],
        "files": {"report.json": digest(args.output / "report.json")},
        "script_sha256": digest(Path(__file__)),
        "training_visible": False,
    })
    print(json.dumps(report), flush=True)
    if payload_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
