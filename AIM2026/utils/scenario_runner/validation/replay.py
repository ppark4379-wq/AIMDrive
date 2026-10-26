"""Replay recorded Ego observations through EventMonitor without MORAI."""

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path

from .event_monitor import EgoObservation, EventMonitor


class ReplayError(ValueError):
    pass


OBSERVATION_FIELDS = {field.name for field in fields(EgoObservation)}


def _read_jsonl(path, digest):
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ReplayError(f"Missing or unreadable file: {path}") from error
    if hashlib.sha256(data).hexdigest() != digest:
        raise ReplayError(f"File checksum mismatch: {path}")
    if data and not data.endswith(b"\n"):
        raise ReplayError(f"Truncated JSONL file: {path}")
    try:
        return [json.loads(line) for line in data.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, ValueError) as error:
        raise ReplayError(f"Corrupt JSONL file: {path}") from error


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _observation(row, sequence, run_id, previous_lap):
    if not isinstance(row, dict) or set(row) != OBSERVATION_FIELDS | {"schema_version", "sequence"}:
        raise ReplayError(f"Observation fields missing or unexpected at sequence {sequence}")
    if row["schema_version"] != 1 or type(row["sequence"]) is not int or row["sequence"] != sequence:
        raise ReplayError(f"Observation sequence mismatch at {sequence}")
    if row["run_id"] != run_id:
        raise ReplayError(f"Observation run_id mismatch at sequence {sequence}")
    if (row["time_basis"] != "lap_elapsed_sec" or not isinstance(row["clock_domain"], str)
            or not row["clock_domain"]):
        raise ReplayError(f"Observation time basis missing at sequence {sequence}")
    lap = row["lap_index"]
    timestamp = row["timestamp"]
    # The Scenario Runner can reset its elapsed clock for a new editor trial;
    # sequence is the authoritative sample order, not elapsed seconds.
    if (type(lap) is not int or lap < 0 or lap < previous_lap or lap > previous_lap + 1
            or not _number(timestamp) or timestamp < 0):
        raise ReplayError(f"Observation time order invalid at sequence {sequence}")
    position = row["position_xyz"]
    if position is not None and (not isinstance(position, list) or len(position) != 3
                                 or not all(_number(value) for value in position)):
        raise ReplayError(f"Observation position invalid at sequence {sequence}")
    for field in ("speed_mps", "yaw_deg"):
        if row[field] is not None and not _number(row[field]):
            raise ReplayError(f"Observation {field} invalid at sequence {sequence}")
    collisions = row["collision_objects"]
    if collisions is not None and (not isinstance(collisions, list)
                                   or any(not isinstance(actor, str) or not actor.strip()
                                          for actor in collisions)):
        raise ReplayError(f"Observation collisions invalid at sequence {sequence}")
    values = {key: row[key] for key in OBSERVATION_FIELDS}
    values["position_xyz"] = None if position is None else tuple(position)
    values["collision_objects"] = None if collisions is None else tuple(collisions)
    return EgoObservation(**values), lap


def replay_run(run_directory):
    """Raise ReplayError unless the replayed event stream matches the saved one."""
    run_directory = Path(run_directory)
    run_id = run_directory.name
    summary_path = run_directory / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ReplayError(f"Missing or corrupt summary: {summary_path}") from error
    if not isinstance(summary, dict) or summary.get("run_id") != run_id:
        raise ReplayError("Summary run_id does not match directory")
    observations_path = run_directory / "ego_observations.jsonl"
    events_path = run_directory / "validation_events.jsonl"
    for key, path in (("ego_observations_path", observations_path),
                      ("validation_events_path", events_path)):
        recorded = summary.get(key)
        if (not isinstance(recorded, str)
                or recorded.replace("\\", "/").split("/")[-2:] != [run_id, path.name]):
            raise ReplayError(f"Summary {key} does not match run directory")
    rows = _read_jsonl(observations_path, summary.get("observation_sha256"))
    saved_events = _read_jsonl(events_path, summary.get("event_sha256"))
    if not rows or type(summary.get("observation_count")) is not int or len(rows) != summary["observation_count"]:
        raise ReplayError("Observation count missing or inconsistent")
    if type(summary.get("event_count")) is not int or len(saved_events) != summary["event_count"]:
        raise ReplayError("Event count missing or inconsistent")

    monitor = EventMonitor()
    replayed = []
    previous_lap = 0
    for sequence, row in enumerate(rows):
        observation, previous_lap = _observation(
            row, sequence, run_id, previous_lap,
        )
        try:
            replayed.extend(asdict(event) for event in monitor.observe(observation))
        except (TypeError, ValueError) as error:
            raise ReplayError(f"Invalid observation at sequence {sequence}: {error}") from error
    if replayed != saved_events:
        raise ReplayError("Replayed events differ from online events")
    return {"run_id": run_id, "observation_count": len(rows),
            "event_count": len(replayed), "status": "matched"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path,
                        help="scenario_runner/.runtime/validation_runs/<run_id>")
    args = parser.parse_args()
    try:
        result = replay_run(args.run_directory)
    except ReplayError as error:
        parser.exit(1, f"Replay failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
