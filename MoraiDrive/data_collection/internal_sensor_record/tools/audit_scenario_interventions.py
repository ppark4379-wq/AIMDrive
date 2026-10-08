#!/usr/bin/env python3
"""Audit the selected main scenario and disclosed interventions.

Weather, time, obstacles and traffic belong to the team scenario runner.  The
collector observes them and records provenance.  A route-light release after a
prolonged stop is valid only when the command has exact-ID readback evidence;
it remains explicitly separate from naturally changing signal samples.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT_ROOT)]

from async_capture.journal import digest, write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from ssd_storage import require_ssd
from scenario_runner.scenarios.segment_modes.modes import (
    PRIMARY_SCENARIOS,
    get_scenario_zones,
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_traffic_interventions(events):
    commands = [
        row for row in events
        if row.get("event") == "traffic_light_override"
    ]
    delayed_verified = [
        row for row in events
        if row.get("event") == "traffic_light_override_verified"
    ]
    timeouts = [
        row for row in events
        if row.get("event") == "traffic_light_override_verification_timeout"
    ]
    immediate_verified = sum(
        bool(row.get("result", {}).get("verified")) for row in commands
    )
    verified = immediate_verified + len(delayed_verified)
    disclosed = all(
        row.get("intervention") is True
        and row.get("mission_distribution_changed") is True
        and row.get("training_visible") is False
        for row in commands + delayed_verified + timeouts
    )
    return {
        "commands": len(commands),
        "verified": verified,
        "verification_timeouts": len(timeouts),
        "all_commands_verified": verified == len(commands) and not timeouts,
        "all_events_disclosed": disclosed,
        "light_ids": sorted({
            str(row.get("light_id"))
            for row in commands
            if row.get("light_id")
        }),
    }


def summarize_roundabout_observation(events, configured):
    """Validate the configured nearby lifecycle from its explicit event ledger."""
    legacy_count = configured.get("count")
    minimum_count = int(
        configured.get(
            "min_count", legacy_count if legacy_count is not None else 0
        ) or 0
    )
    maximum_count = int(
        configured.get(
            "max_count", legacy_count if legacy_count is not None else 0
        ) or 0
    )
    expected_speed = float(configured.get("speed_mps", 0.0) or 0.0)
    spawns = [
        row for row in events
        if row.get("event") == "roundabout_spawn_contract"
    ]
    spawn_finishes = [
        row for row in events
        if row.get("event") == "actor_lifecycle_finished"
        and row.get("action") == "roundabout_spawn"
        and row.get("success") is True
    ]
    retires = [
        row for row in events
        if row.get("event") == "actor_lifecycle_finished"
        and row.get("action") == "roundabout_retire"
        and row.get("success") is True
    ]
    observed_counts = sorted({
        len(set(str(actor_id) for actor_id in row.get("actor_ids", [])))
        for row in spawns
    })
    spawn = spawns[0] if len(spawns) == 1 else {}
    actor_ids = {
        str(actor_id) for actor_id in spawn.get("actor_ids", [])
    }
    motion = list(spawn.get("actor_motion_contracts", []))
    motion_ids = {str(row.get("actor_id")) for row in motion}
    motion_acknowledged = (
        len(motion) == len(actor_ids)
        and motion_ids == actor_ids
        and all(
            row.get("speed_limit_acknowledged") is True
            and row.get("initial_velocity_acknowledged") is True
            for row in motion
        )
    )
    retire_ids = {
        str(actor_id)
        for row in retires
        for actor_id in row.get("actor_ids", [])
    }
    return {
        "configured_count": (
            minimum_count if minimum_count == maximum_count else None
        ),
        "configured_min_count": minimum_count,
        "configured_max_count": maximum_count,
        "configured_speed_mps": expected_speed,
        "observed_counts": observed_counts,
        "spawn_events": len(spawns),
        "spawn_lifecycle_successes": len(spawn_finishes),
        "retire_lifecycle_successes": len(retires),
        "actor_ids": sorted(actor_ids),
        "motion_acknowledged": motion_acknowledged,
        "spawn_matches_config": (
            len(spawns) == 1
            and minimum_count > 0
            and minimum_count <= len(actor_ids) <= maximum_count
            and abs(float(spawn.get("target_speed_mps", -1.0)) - expected_speed)
                < 1e-9
            and motion_acknowledged
        ),
        "lifecycle_complete": (
            len(spawn_finishes) == 1
            and len(retires) == 1
            and retire_ids == actor_ids
        ),
    }


def scenario_zone_contract(scenario_name):
    """Read the enabled-zone contract from the main Scenario Runner."""
    if scenario_name not in PRIMARY_SCENARIOS:
        raise ValueError("Unsupported scenario: " + str(scenario_name))
    return asdict(get_scenario_zones(scenario_name))


def zone_observation_checks(
    scenario_name, zones, *,
    selected_static, selected_dynamic,
    static_models, dynamic_models,
    dynamic_phases, roundabout_observation,
    highway_samples, tunnel_phases,
):
    """Evaluate only the events enabled by the selected main scenario mode."""
    dynamic_set = set(dynamic_phases)
    tunnel_set = set(tunnel_phases)
    if zones["static"]:
        static_ok = selected_static in static_models
    else:
        static_ok = selected_static == "OFF"

    if zones["dynamic"]:
        dynamic_profile_ok = selected_dynamic in dynamic_models
        dynamic_observed = bool(dynamic_set.intersection({"crossing", "cleared"}))
    else:
        dynamic_profile_ok = selected_dynamic == "OFF"
        dynamic_observed = bool(dynamic_set) and dynamic_set <= {"disabled"}

    if zones["roundabout"]:
        roundabout_observed = (
            roundabout_observation["spawn_matches_config"]
            and roundabout_observation["lifecycle_complete"]
        )
    else:
        roundabout_observed = (
            roundabout_observation["spawn_events"] == 0
            and roundabout_observation["spawn_lifecycle_successes"] == 0
            and roundabout_observation["retire_lifecycle_successes"] == 0
        )

    highway_observed = highway_samples > 0 if zones["highway"] else highway_samples == 0
    if scenario_name in ("full_loop_tunnel_dynamic", "full_loop_tunnel_vehicle"):
        tunnel_observed = "completed" in tunnel_set
    elif zones["tunnel"]:
        tunnel_observed = bool(tunnel_set - {"disabled"})
    else:
        tunnel_observed = bool(tunnel_set) and tunnel_set <= {"disabled"}

    checks = {
        "selected_static_from_pool": static_ok,
        "selected_dynamic_from_pool": dynamic_profile_ok,
        "roundabout_observed": roundabout_observed,
    }
    # Preserve the pre-feature full_loop admission contract exactly. Segment
    # modes additionally prove that enabled events ran and disabled events did not.
    if scenario_name != "full_loop":
        checks.update(
            dynamic_observed=dynamic_observed,
            highway_observed=highway_observed,
            tunnel_observed=tunnel_observed,
        )
    return checks


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

    legacy_roundabout_counts = sorted({
        int(row["roundabout_actor_count"])
        for row in progress if row.get("roundabout_actor_count") is not None
    })
    roundabout_observation = summarize_roundabout_observation(events, roundabout)
    observed_roundabout_counts = (
        roundabout_observation["observed_counts"] or legacy_roundabout_counts
    )
    dynamic_phases = sorted({
        str(row.get("dynamic_phase"))
        for row in progress if row.get("dynamic_phase") is not None
    })
    highway_samples = sum(bool(row.get("highway_active")) for row in progress)
    tunnel_phases = sorted({
        str(row.get("tunnel_phase"))
        for row in progress if row.get("tunnel_phase") is not None
    })
    traffic_interventions = summarize_traffic_interventions(events)
    traffic_override = dict(config.get("traffic_light_wait_override", {}))

    scenario_name = (
        episode.get("scenario_name")
        or config.get("mission_mode")
        or "full_loop"
    )
    scenario_known = scenario_name in PRIMARY_SCENARIOS
    zones = (
        scenario_zone_contract(scenario_name)
        if scenario_known
        else {name: False for name in ("static", "dynamic", "roundabout", "highway", "tunnel")}
    )
    tunnel_mission = outcome.get("tunnel_mission") or {}
    if (scenario_name in ("full_loop_tunnel_dynamic", "full_loop_tunnel_vehicle")
            and tunnel_mission.get("scenario_name") == scenario_name
            and tunnel_mission.get("completed") is True
            and tunnel_mission.get("phase") == "completed"):
        # Completion may occur during the same actor's continuation after Ego goal.
        tunnel_phases = sorted(set(tunnel_phases) | {"completed"})
    zone_checks = zone_observation_checks(
        scenario_name,
        zones,
        selected_static=selected_static,
        selected_dynamic=selected_dynamic,
        static_models=static_models,
        dynamic_models=dynamic_models,
        dynamic_phases=dynamic_phases,
        roundabout_observation=roundabout_observation,
        highway_samples=highway_samples,
        tunnel_phases=tunnel_phases,
    )
    if scenario_name in ("full_loop_tunnel_dynamic", "full_loop_tunnel_vehicle"):
        zone_checks["tunnel_completion_verified"] = (
            tunnel_mission.get("scenario_name") == scenario_name
            and tunnel_mission.get("completed") is True
            and tunnel_mission.get("phase") == "completed"
        )
    expected_mission_mode = None if scenario_name == "full_loop" else scenario_name

    checks = {
        "full_mission_completed": outcome.get("full_mission_completed") is True,
        "scenario_supported": scenario_known,
        "scenario_enabled": (
            config.get("mission_mode") == expected_mission_mode
            if expected_mission_mode is not None
            else (
                config.get("clean_road_mode") is False
                or (
                    "clean_road_mode" not in config
                    and selected_static in static_models
                    and selected_dynamic in dynamic_models
                )
            )
        ),
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
        "dynamic_pool": bool(dynamic_models),
        "traffic_override_contract": (
            traffic_override.get("enabled") is True
            and float(traffic_override.get("wait_wall_sec", -1.0)) == 10.0
            and int(traffic_override.get("target_color", -1)) == 16
        ),
        "traffic_override_commands_verified": (
            traffic_interventions["all_commands_verified"]
            and traffic_interventions["all_events_disclosed"]
        ),
        "roundabout_team_contract": (
            int(roundabout.get("min_count", 0)) == 3
            and int(roundabout.get("max_count", 0)) == 5
            and float(roundabout.get("speed_mps", 0.0)) == 4.0
            and bool(roundabout.get("models"))
            and "DEFAULT_SEDAN" in roundabout.get("models", [])
            and float(
                roundabout.get("nonuniform_position", {}).get(
                    "min_spawn_gap_m", 0.0
                )
            ) == 10.0
            and float(
                roundabout.get("speed_randomization", {}).get(
                    "emergency_gap_m", 0.0
                )
            ) == 6.0
        ),
        "native_files": native.get("files_valid") is True,
        "actor_transport": actor.get("structural_valid") is True,
        **zone_checks,
    }
    report = {
        "schema": "morai_team_scenario_fidelity_audit_v3",
        "scenario_name": scenario_name,
        "enabled_zones": zones,
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
            "roundabout": roundabout_observation,
            "dynamic_phases": dynamic_phases,
            "highway_active_samples": highway_samples,
            "tunnel_phases": tunnel_phases,
            "traffic_light_interventions": traffic_interventions,
        },
        "collector_distribution_overrides": [],
        "scenario_distribution_interventions": [
            "traffic_light_wait_override"
        ] if traffic_interventions["commands"] else [],
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
