#!/usr/bin/env python3
"""Offline Expert Event Report; no training acceptance or exclusion decisions."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = PROJECT_ROOT / "dataset" / "sensor_record_auto_v1"
SCHEMA = "expert_driving_evaluation_v1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_frame_matching import load_frame_sources, match_collision
from expert_event_reporting import REPORT_COMMENTS, build_event_report


def directory_name(value):
    """Campaign and run IDs are single directory names on Windows and Linux."""
    if (not isinstance(value, str) or not value.strip()
            or value in {".", ".."} or any(c in value for c in '/\\:\0')):
        raise ValueError("expected a single directory name: {!r}".format(value))
    return value


def load_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def load_campaign_laps(campaign_dir):
    """Use the last ledger entry per lap, including failed/started entries."""
    path = campaign_dir / "runs.tsv"
    latest = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"lap", "run_name", "status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("{}: required columns: lap, run_name, status".format(path))
        for row in reader:
            try:
                lap = int(row["lap"])
                if lap < 1:
                    raise ValueError("lap must be positive")
                directory_name(row["run_name"])
                if not row["status"] or None in row or None in row.values():
                    raise ValueError("incomplete or extra TSV fields")
            except (TypeError, ValueError) as exc:
                raise ValueError("{}:{}: {}".format(path, reader.line_num, exc)) from exc
            latest[lap] = row
    if not latest:
        raise ValueError("{}: no laps recorded".format(path))
    return [latest[lap] for lap in sorted(latest)]


def read_collisions(path, errors, records=None):
    """Retain readable collision evidence even when a JSONL line is corrupt."""
    collisions = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                        raise ValueError("expected an object with an event name")
                except ValueError as exc:
                    errors.append("events.jsonl:{}: {}".format(line_number, exc))
                    if records is not None:
                        records.append((line_number, None))
                    continue
                if records is not None:
                    records.append((line_number, event))
                if event["event"] != "collision":
                    continue
                ego = event.get("ego") or {}
                objects = ego.get("collision_objects", []) if isinstance(ego, dict) else None
                if not isinstance(objects, list):
                    errors.append("events.jsonl:{}: invalid ego.collision_objects".format(line_number))
                    objects = []
                collisions.append({
                    "event_id": len(collisions) + 1,
                    "sensor_frame": None,
                    "elapsed_seconds": event.get("elapsed_seconds"),
                    "route_s": event.get("route_s"),
                    "collision_objects": [str(obj) for obj in objects],
                    "receive_monotonic_ns": event.get("receive_monotonic_ns"),
                    "event_sequence": event.get("sequence"),
                    "source_line": line_number,
                    "ego_xy": [ego.get("x"), ego.get("y")] if isinstance(ego, dict) else None,
                    "current_link": ego.get("current_link") if isinstance(ego, dict) else None,
                })
    except (OSError, UnicodeError) as exc:
        errors.append("events.jsonl read failed: {}".format(exc))
    return collisions


def evaluate_lap(base, row):
    episode = base / row["run_name"] / "episode_000"
    result = {
        "lap": int(row["lap"]),
        "run_name": row["run_name"],
        "episode": str(episode),
        "collection_status": row["status"],
        "completed": None,
        "collision_count": None,
        "collisions": [],
        "events_complete": False,
        "events": [],
        "event_counts": {},
        "event_report_warnings": [],
        "errors": [],
    }
    errors = result["errors"]
    if not episode.is_dir():
        errors.append("episode directory missing: {}".format(episode))
        return result

    outcome = None
    try:
        outcome = load_json(episode / "outcome.json")
        result["scenario_status"] = outcome.get("status")
        result["scenario_reason"] = outcome.get("reason")
        result["collision_summary"] = outcome.get("collision_summary")
        completed = outcome.get("full_mission_completed")
        if type(completed) is not bool:
            raise ValueError("full_mission_completed must be a JSON boolean")
        result["completed"] = completed
    except (OSError, ValueError) as exc:
        errors.append("outcome.json read failed: {}".format(exc))

    event_errors = []
    records = []
    result["collisions"] = read_collisions(episode / "events.jsonl", event_errors, records)
    errors.extend(event_errors)
    result["events_complete"] = not event_errors
    if result["events_complete"]:
        result["collision_count"] = len(result["collisions"])
        # The summary is corroboration, never a substitute for event records.
        summary = result.get("collision_summary")
        if isinstance(summary, dict) and "contact_events" in summary:
            count = summary["contact_events"]
            if type(count) is not int or count != result["collision_count"]:
                errors.append("collision_summary.contact_events disagrees with events.jsonl")
    sources = load_frame_sources(episode)
    for collision in result["collisions"]:
        collision.update(match_collision(collision, sources))
    result.update(build_event_report(episode, records, result["collisions"], outcome, sources))
    return result


def evaluate_campaign(base, campaign):
    directory_name(campaign)
    laps = [evaluate_lap(base, row) for row in load_campaign_laps(base / "campaigns" / campaign)]
    return {
        "_comments": REPORT_COMMENTS,
        "schema": SCHEMA,
        "frame_matching_schema": "expert_collision_frame_proximity_v2",
        "event_report_schema": "expert_event_report_v1",
        "campaign": campaign,
        "laps": laps,
        "summary": {
            "total_laps": len(laps),
            "completed_laps": sum(lap["completed"] is True for lap in laps),
            "collision_free_laps": sum(
                lap["completed"] is True and lap["collision_count"] == 0
                and not lap["errors"] for lap in laps
            ),
            "collision_events": sum(len(lap["collisions"]) for lap in laps),
            "laps_with_errors": sum(bool(lap["errors"]) for lap in laps),
            "evaluation_complete": all(not lap["errors"] for lap in laps),
        },
    }


def print_report(report):
    print("\n" + "=" * 60)
    print("Expert Event Report")
    print("Campaign : {}".format(report["campaign"]))
    print("=" * 60)
    for lap in report["laps"]:
        completed = {True: "YES", False: "NO", None: "UNKNOWN"}[lap["completed"]]
        count = lap["collision_count"]
        if count is None:
            count = "UNKNOWN ({} observed)".format(len(lap["collisions"]))
        print("\nLap {:03d}".format(lap["lap"]))
        print("  Run        : {}".format(lap["run_name"]))
        print("  Collection : {}".format(lap["collection_status"]))
        print("  Completed  : {}".format(completed))
        print("  Collisions : {}".format(count))
        for collision in lap["collisions"]:
            print("\n    Collision #{}".format(collision["event_id"]))
            print("      object  : {}".format(collision["collision_objects"]))
            print("      route_s : {} m".format(collision["route_s"]))
            print("      elapsed : {} s".format(collision["elapsed_seconds"]))
            print("      link    : {}".format(collision["current_link"]))
            index = collision["sensor_bundle_index"]
            if index is None:
                print("      sensor_bundle_index  : NOT MATCHED YET (not selected)")
            else:
                print("      sensor_bundle_index  : {} (0-based)".format(index))
                print("      sensor_bundle_number : {} (1-based ordinal)".format(
                    collision["sensor_bundle_number"]))
            print("      Time candidates      : {}".format(collision["receive_time_candidate_indices"]))
            print("      Position candidates  : {}".format(collision["position_candidate_indices"]))
            gap, error = collision["receive_time_gap_ms"], collision["position_error_m"]
            print("      Receive-time gap     : {}".format(
                "{:.2f} ms".format(gap) if gap is not None else "UNKNOWN"))
            print("      Position error       : {}".format(
                "{:.3f} m".format(error) if error is not None else "UNKNOWN"))
            print("      Match confidence     : {}".format(collision["match_confidence"]))
            print("      Match status         : {}".format(collision["match_status"]))
            print("      Interpretation       : {}".format(collision["match_interpretation"]))
            if collision["candidate_index_distance"] is not None:
                print("      Candidate index gap  : {}".format(collision["candidate_index_distance"]))
            if collision["candidate_source_time_gap_ms"] is not None:
                print("      Candidate source gap : {:.2f} ms".format(collision["candidate_source_time_gap_ms"]))
            for reason in collision["match_reasons"]:
                print("      Match note           : {}".format(reason))
            if collision["collision_between_frames"] is not None:
                print("      Estimated between    : {} (0-based indices)".format(
                    collision["collision_between_frames"]))
                print("      Estimated fraction   : {:.3f}".format(
                    collision["estimated_fraction_after_frame"]))
                print("      Estimated source time: {:.6f} s".format(
                    collision["estimated_collision_source_time_seconds"]))
        for event in lap["events"]:
            if event["type"] == "collision":
                continue
            print("\n    Event #{}: {}".format(event["report_event_id"], event["type"]))
            if event["type"] == "off_route" and "start_match" in event:
                print("      Bundle endpoints     : {} -> {} (0-based candidates)".format(
                    event["start_sensor_bundle_index"], event["end_sensor_bundle_index"]))
                print("      Duration / max CTE   : {:.3f} s / {:.3f} m".format(
                    event["duration_seconds"], event["max_cross_track_error_m"]))
                print("      Endpoint confidence  : {} / {}".format(
                    event["start_match"]["match_confidence"], event["end_match"]["match_confidence"]))
            else:
                print("      sensor_bundle_index  : {} (0-based candidate)".format(event["sensor_bundle_index"]))
                print("      Match status         : {}".format(event["match_status"]))
                print("      Confidence           : {}".format(event["match_confidence"]))
                if event["details"].get("light_id") is not None:
                    print("      Light ID             : {}".format(event["details"]["light_id"]))
                if event.get("reason") is not None:
                    print("      Reason               : {}".format(event["reason"]))
                if event["sensor_bundle_index"] is None:
                    print("      Match note           : {}".format(event["match_interpretation"]))
        print("  Event counts : {}".format(lap["event_counts"]))
        for warning in lap["event_report_warnings"]:
            print("  NOTE       : {}".format(warning))
        for error in lap["errors"]:
            print("  ERROR      : {}".format(error))
    summary = report["summary"]
    total = summary["total_laps"]
    print("\n" + "-" * 60)
    print("Total laps          : {}".format(total))
    print("Completed laps      : {}/{}".format(summary["completed_laps"], total))
    print("Collision-free laps : {}/{}".format(summary["collision_free_laps"], total))
    print("Collision events    : {}{}".format(
        summary["collision_events"],
        " (observed; evaluation incomplete)" if not summary["evaluation_complete"] else "",
    ))
    print("Laps with errors    : {}".format(summary["laps_with_errors"]))
    print("-" * 60)


def save_report(path, report):
    """Replace the report only after the new JSON has been written in full."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", help="campaign ID, e.g. campaign_01")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help="sensor_record_auto_v1 directory")
    args = parser.parse_args(argv)
    base = args.base.expanduser().resolve()
    try:
        report = evaluate_campaign(base, args.campaign)
        output = base / "campaigns" / args.campaign / SCHEMA / "report.json"
        save_report(output, report)
    except (OSError, ValueError) as exc:
        print("Evaluation failed: {}".format(exc), file=sys.stderr)
        return 2
    print_report(report)
    print("\nReport saved: {}".format(output))
    return 0 if report["summary"]["evaluation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
