#!/usr/bin/env python3
"""Audit that collection used the current team full-loop scenario unchanged.

This stage deliberately does not require old collector-side actor lifecycle or
traffic-light override events. Weather, time, obstacles and traffic belong to
the team scenario runner; the collector observes them and records provenance.
"""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.journal import digest, write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from ssd_storage import require_ssd


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.episode, args.output):
        require_ssd(path, reserve_bytes=0)
    if args.output.exists():
        raise FileExistsError(args.output)

    verify_sealed_directory(args.episode / "quality", "QUALITY_COMPLETE.json")
    verify_sealed_directory(args.episode / "native_audit")
    verify_sealed_directory(args.episode / "actor_inventory_audit_v1")

    episode = load(args.episode / "episode.json")
    outcome = load(args.episode / "outcome.json")
    native = load(args.episode / "native_audit" / "report.json")
    actor = load(args.episode / "actor_inventory_audit_v1" / "report.json")
    events = [
        json.loads(line)
        for line in (args.episode / "events.jsonl").read_text().splitlines()
    ]
    progress = [row for row in events if row.get("event") == "progress"]

    config = dict(episode.get("scenario_config", {}))
    profile = dict(episode.get("profile", {}))
    static_models = list(config.get("static_obstacle_models", []))
    dynamic_models = list(config.get("dynamic_obstacle_models", []))
    roundabout = dict(config.get("roundabout_npc", {}))
    profile_state_path = args.episode / "campaign_profile_state.json"
    if not profile_state_path.is_file():
        profile_state_path = args.episode / "profile_state.json"
    profile_state = load(profile_state_path) if profile_state_path.is_file() else {}
    selected_static = profile_state.get("static_obstacle_model")
    selected_dynamic = profile_state.get("dynamic_obstacle_model")

    observed_roundabout_counts = sorted({
        int(row["roundabout_actor_count"])
        for row in progress if row.get("roundabout_actor_count") is not None
    })
    dynamic_phases = sorted({
        str(row.get("dynamic_phase"))
        for row in progress if row.get("dynamic_phase") is not None
    })
    highway_samples = sum(bool(row.get("highway_active")) for row in progress)
    tunnel_phases = sorted({
        str(row.get("tunnel_phase"))
        for row in progress if row.get("tunnel_phase") is not None
    })

    checks = {
        "full_mission_completed": outcome.get("full_mission_completed") is True,
        "scenario_enabled": config.get("clean_road_mode") is False,
        "weather_contract": [
            str(row.get("weather", "")).upper()
            for row in config.get("profiles", [])
        ] == ["SUNNY", "FOGGY"],
        "time_contract": list(config.get("scenario_times", [])) == [11, 13, 15],
        "selected_weather": str(profile.get("weather", "")).upper()
            in ("SUNNY", "FOGGY"),
        "selected_time": int(episode.get("time_hour", -1)) in (11, 13, 15),
        "static_pool": bool(static_models),
        "tree_obstacles_excluded": not any(
            str(model).lower().startswith("tree") for model in static_models
        ),
        "selected_static_from_pool": selected_static in static_models,
        "dynamic_pool": bool(dynamic_models),
        "selected_dynamic_from_pool": selected_dynamic in dynamic_models,
        "roundabout_team_contract": (
            int(roundabout.get("count", 0)) == 6
            and float(roundabout.get("speed_mps", 0.0)) == 4.0
            and roundabout.get("model") == "DEFAULT_SEDAN"
        ),
        "roundabout_observed": bool(progress)
            and observed_roundabout_counts == [6],
        "native_files": native.get("files_valid") is True,
        "actor_transport": actor.get("structural_valid") is True,
    }
    report = {
        "schema": "morai_team_scenario_fidelity_audit_v1",
        "checks": checks,
        "selected_profile": {
            "weather": profile.get("weather"),
            "time_hour": episode.get("time_hour"),
            "static_obstacle_model": selected_static,
            "dynamic_obstacle_model": selected_dynamic,
        },
        "coverage_observations": {
            "progress_samples": len(progress),
            "roundabout_actor_counts": observed_roundabout_counts,
            "dynamic_phases": dynamic_phases,
            "highway_active_samples": highway_samples,
            "tunnel_phases": tunnel_phases,
        },
        "collector_distribution_overrides": [],
        "valid": all(checks.values()),
        "source_timestamps_modified": False,
        "physical_simultaneity_verified": False,
        "training_visible": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "COMPLETE.json", {
        "files": {"report.json": digest(args.output / "report.json")},
        "script_sha256": digest(Path(__file__)),
        "training_visible": False,
    })
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
