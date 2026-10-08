#!/usr/bin/env python3
"""Verify every sealed local lap in one Sensor Record campaign."""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT
PROJECT_ROOT = Path(
    os.environ.get("MORAI_PROJECT_ROOT", str(ROOT.parents[1]))
).expanduser().resolve()
sys.path[:0] = [str(INTEGRATION), str(PROJECT_ROOT)]

from async_capture.journal import digest, verify_run, write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from async_capture.route_sources import resolve_route_sources, verify_route_report
from async_capture.route_review_rebuild import archive_route_review
from cleanup_failed_sensor_record_attempt import cleanup_failed_attempt
from ssd_storage import SSD_ROOT, require_ssd, windows_path
from async_capture.scenario_catalog import COLLECTION_SCENARIOS, TUNNEL_STATIC_SCENARIO, validate_tunnel_static_collection
from scenario_runner.tools import static_lane_compliance
from python_environment import (
    build_analysis_environment,
    probe_analysis_environment,
)
from dataset_layout import (
    campaign_root_from_metadata,
    collecting_campaign,
    final_campaign,
    finalize_collecting_campaign,
    find_campaign,
    is_direct_campaign_metadata,
    relative_path,
    run_path,
)


BASE = SSD_ROOT / "sensor_record_auto_v1"
SEALED_STAGES = {
    "quality": ("quality", "QUALITY_COMPLETE.json"),
    "native_audit": ("native_audit", "COMPLETE.json"),
    "native_associations": ("native_associations", "COMPLETE.json"),
    "gt_timeline": ("gt_timeline_v1", "COMPLETE.json"),
    "traffic_audit": ("traffic_audit_v1", "COMPLETE.json"),
    "actor_inventory": ("actor_inventory_audit_v1", "COMPLETE.json"),
    "object_geometry": ("object_geometry_audit_v1", "COMPLETE.json"),
    "observer_overhead": ("observer_overhead_v1", "COMPLETE.json"),
    "route_signal": ("route_signal_audit_v1", "COMPLETE.json"),
    "motion_time": ("motion_time_v1", "COMPLETE.json"),
    "exposure": ("exposure_v1", "COMPLETE.json"),
    "scenario_interventions": ("scenario_intervention_audit_v3", "COMPLETE.json"),
    "review": ("review_closeout", "COMPLETE.json"),
    "admission": ("admission_v1", "COMPLETE.json"),
}


def finalize_direct_local_publication(campaign, output, storage_root,
                                      scenario_name, plan):
    """Finalize a direct campaign and write its publication record once."""
    campaign = Path(campaign).resolve()
    output = Path(output).resolve()
    source = campaign_root_from_metadata(campaign)
    campaign_id = plan["campaign_id"]
    dataset_campaign_id = plan.get("dataset_campaign_id", campaign_id)
    expected_staging = collecting_campaign(
        storage_root, scenario_name, campaign_id).resolve()
    expected_final = final_campaign(
        storage_root, scenario_name, dataset_campaign_id).resolve()
    try:
        relative_output = output.relative_to(source)
    except ValueError as exc:
        raise ValueError("Validation output is outside its direct campaign") from exc

    promoted = False
    if source == expected_staging:
        finalized = finalize_collecting_campaign(
            campaign, storage_root, scenario_name,
            campaign_id, dataset_campaign_id)
        promoted = True
        campaign = finalized / "_campaign"
        output = finalized / relative_output
    elif source == expected_final:
        finalized = source
    else:
        raise ValueError("Direct campaign is outside staging/final dataset roots")

    publication = {
        "schema": "morai_local_dataset_campaign_v2",
        "scenario_folder": plan["dataset_scenario_folder"],
        "campaign_id": campaign_id,
        "dataset_campaign_id": dataset_campaign_id,
        "campaign": str(finalized),
        "hardlinked": False,
        "direct_capture": True,
    }
    write_json(output / "local_dataset_publication.json", publication)
    return campaign, output, publication, promoted


def offline_stages(run, episode):
    return [
        ("audit_sensor_record_probe.py", episode / "native_audit", [
            "--run", str(run), "--raw", str(episode / "raw"),
            "--output", str(episode / "native_audit")]),
        ("build_sensor_record_associations.py", episode / "native_associations", [
            "--episode", str(episode)]),
        ("build_sensor_record_gt_timeline.py", episode / "gt_timeline_v1", [
            "--episode", str(episode), "--output", str(episode / "gt_timeline_v1")]),
        ("audit_sensor_record_direct_traffic.py", episode / "traffic_audit_v1", [
            "--episode", str(episode), "--output", str(episode / "traffic_audit_v1")]),
        ("audit_sensor_record_actor_inventory.py", episode / "actor_inventory_audit_v1", [
            "--episode", str(episode), "--output", str(episode / "actor_inventory_audit_v1")]),
        ("audit_sensor_record_object_geometry.py", episode / "object_geometry_audit_v1", [
            "--episode", str(episode), "--output", str(episode / "object_geometry_audit_v1")]),
        ("audit_sensor_record_observer_overhead.py", episode / "observer_overhead_v1", [
            "--episode", str(episode), "--output", str(episode / "observer_overhead_v1")]),
        ("audit_sensor_record_route_signals.py", episode / "route_signal_audit_v1", [
            "--episode", str(episode), "--output", str(episode / "route_signal_audit_v1")]),
        ("audit_sensor_record_motion_time.py", episode / "motion_time_v1", [
            "--episode", str(episode), "--output", str(episode / "motion_time_v1")]),
        ("audit_sensor_record_exposure.py", episode / "exposure_v1", [
            "--audit", str(episode / "native_audit"), "--quality", str(episode / "quality"),
            "--output", str(episode / "exposure_v1")]),
        ("audit_scenario_interventions.py", episode / "scenario_intervention_audit_v3", [
            "--episode", str(episode), "--output", str(episode / "scenario_intervention_audit_v3")]),
        ("summarize_sensor_record_review.py", episode / "review_closeout", [
            "--episode", str(episode), "--output", str(episode / "review_closeout")]),
        ("admit_sensor_record_collection.py", episode / "admission_v1", [
            "--episode", str(episode), "--capture-contract", str(run / "capture_contract.json"),
            "--run-plan", str(run / "plan.json"), "--output", str(episode / "admission_v1")]),
    ]


def complete_offline_review(run, *, analysis_python, env, log):
    episode = run / "episode_000"
    results = []
    for tool, output, arguments in offline_stages(run, episode):
        seal = output / "COMPLETE.json"
        if seal.is_file():
            if tool == "audit_sensor_record_route_signals.py":
                verify_sealed_directory(output, "COMPLETE.json")
                sources = resolve_route_sources(episode, load(run / "sensor_contract.json"))
                verify_route_report(sources, load(output / "report.json"))
            print("[SKIP] %s is already sealed" % tool, flush=True)
            results.append({"tool": tool, "return_code": 0, "already_sealed": True})
            continue
        if output.exists():
            raise RuntimeError(
                "Unsealed offline output exists; preserving it for diagnosis: " + str(output))
        print("[REVIEW] starting %s" % tool, flush=True)
        started = time.monotonic()
        next_heartbeat = started + 30
        process = subprocess.Popen(
            [str(analysis_python), str(INTEGRATION / "tools" / tool)] + arguments,
            env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(1)
            now = time.monotonic()
            if now >= next_heartbeat:
                print("[REVIEW] %s running: %.0fs elapsed" % (tool, now-started), flush=True)
                next_heartbeat = now + 30
        code = int(process.returncode)
        results.append({"tool": tool, "return_code": code, "already_sealed": False})
        if code:
            raise RuntimeError("Offline stage failed (%d): %s; see %s" %
                               (code, tool, log.name))
        print("[PASS] %s: %.1fs" % (tool, time.monotonic()-started), flush=True)
    return results


def complete_bev_geometry_recovery(run, *, analysis_python, env, log):
    """Add V2 BEV-scoped geometry evidence without rewriting V1 seals."""
    from recover_sensor_record_bev_geometry import recovery_selection, validate_recovery
    selection = recovery_selection(run)
    recovery = selection["path"]
    if recovery.is_file():
        existing = load(recovery)
        if selection["tunnel_revision"] is None and existing.get("corrected_object_geometry") is None:
            print("[SKIP] existing legacy admission recovery keeps V1 geometry", flush=True)
            return
    print("[REVIEW] checking BEV-aware object geometry recovery", flush=True)
    started = time.monotonic()
    process = subprocess.Popen([
        str(analysis_python),
        str(INTEGRATION / "tools" / "recover_sensor_record_bev_geometry.py"),
        "--run", str(run),
    ], env=env, stdout=log, stderr=subprocess.STDOUT)
    next_heartbeat = started + 30
    while process.poll() is None:
        time.sleep(1)
        now = time.monotonic()
        if now >= next_heartbeat:
            print("[REVIEW] BEV geometry recovery running: %.0fs elapsed" %
                  (now - started), flush=True)
            next_heartbeat = now + 30
    if process.returncode:
        raise RuntimeError("BEV geometry recovery failed; see " + log.name)
    if not recovery.is_file():
        raise RuntimeError("BEV geometry recovery did not produce a record")
    if selection["tunnel_revision"] is not None:
        validate_recovery(run,recovery,selection=selection)
    print("[PASS] BEV geometry recovery: %.1fs" %
          (time.monotonic() - started), flush=True)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def effective_admission(run, episode, errors, seals):
    """Use an additive, sealed recalculation without changing admission_v1."""
    original_path = episode / "admission_v1" / "report.json"
    original = load(original_path)
    from recover_sensor_record_bev_geometry import recovery_selection, validate_recovery
    try:
        selection = recovery_selection(run)
    except Exception as exc:
        errors.append("admission_recovery:" + str(exc))
        return original
    recovery_path = selection["path"]
    if not recovery_path.is_file():
        return original
    try:
        recovery = load(recovery_path)
        if selection["tunnel_revision"] is not None:
            validate_recovery(run,recovery_path,selection=selection)
        if not (
            recovery.get("schema") == "morai_sensor_record_admission_recovery_v1"
            and recovery.get("status") == "completed"
            and recovery.get("recovery_is_additive") is True
            and recovery.get("source_timestamps_modified") is False
            and recovery.get("training_visible") is False
            and recovery.get("original_admission", {}).get("sha256") == digest(original_path)
        ):
            raise ValueError("invalid admission recovery record")
        corrected_relative = recovery.get("corrected_admission", {}).get("path")
        corrected_path = (run / str(corrected_relative)).resolve()
        if run.resolve() not in corrected_path.parents:
            raise ValueError("admission recovery path escapes run")
        corrected_dir = corrected_path.parent
        verify_sealed_directory(corrected_dir, "COMPLETE.json")
        corrected = load(corrected_path)
        if not (
            recovery["corrected_admission"].get("sha256") == digest(corrected_path)
            and recovery["corrected_admission"].get("seal_sha256")
                == digest(corrected_dir / "COMPLETE.json")
            and corrected.get("raw_manifest_sha256") == original.get("raw_manifest_sha256")
            and corrected.get("runtime_streams_valid") is True
        ):
            raise ValueError("corrected admission evidence mismatch")
        seals["admission_recovery"] = digest(recovery_path)
        seals["effective_admission"] = digest(corrected_dir / "COMPLETE.json")
        return corrected
    except Exception as exc:
        errors.append("admission_recovery:" + str(exc))
        return original


def admission_recovery_build_arguments(runs):
    """Pass one verified tunnel revision to the existing downstream reader."""
    from recover_sensor_record_bev_geometry import recovery_selection, validate_recovery
    selections = [(Path(run),recovery_selection(run)) for run in runs]
    if not any(selection["tunnel_revision"] is not None for _,selection in selections):
        return []
    names = {selection["path"].name for _,selection in selections}
    revisions = [selection["tunnel_revision"] for _,selection in selections]
    if len(names) != 1 or any(revision is None or revision != revisions[0] for revision in revisions):
        raise ValueError("Validated tunnel runs do not share one current scale recovery revision")
    for run,selection in selections:
        if not selection["path"].is_file():
            raise ValueError("Current tunnel scale recovery is missing before dataset build")
        validate_recovery(run,selection["path"],selection=selection)
    return ["--admission-recovery-name",next(iter(names))]


def resolve_campaign(value):
    path = find_campaign(SSD_ROOT, value)
    require_ssd(path, reserve_bytes=0)
    return path


def campaign_run(campaign, run_name):
    if is_direct_campaign_metadata(campaign, SSD_ROOT):
        return run_path(campaign, run_name)
    return BASE / run_name


def completed_rows(path):
    latest = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            latest[int(row["lap"])] = row
    return latest


def _cleanup_summary(marker):
    return {
        "status": marker.get("cleanup_status"),
        "failure_kind": marker.get("failure_kind"),
        "deleted_files": int(marker.get("deleted_files", 0) or 0),
        "deleted_bytes": int(marker.get("deleted_bytes", 0) or 0),
    }


def _failed_attempt_marker(run, expected_kind):
    marker_path = Path(run) / "FAILED_ATTEMPT.json"
    if not marker_path.is_file():
        return None
    marker = load(marker_path)
    if not (
        marker.get("schema") == "morai_sensor_record_failed_attempt_v1"
        and marker.get("cleanup_status") == "completed"
        and marker.get("failure_kind") == expected_kind
        and marker.get("training_visible") is False
    ):
        return None
    return marker


def _write_lap_results_csv(campaign, laps):
    path = Path(campaign) / "lap_results.csv"
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=(
            "lap", "run_name", "capture_status", "validation_status",
            "training_usable", "reason",
        ))
        writer.writeheader()
        for lap in laps:
            writer.writerow({
                "lap": lap.get("lap"),
                "run_name": lap.get("run_name", ""),
                "capture_status": lap.get("capture_status", "missing"),
                "validation_status": lap.get("validation_status", "not_run"),
                "training_usable": str(
                    bool(lap.get("training_usable"))).lower(),
                "reason": "; ".join(str(value) for value in lap.get("errors", [])),
            })
    os.replace(str(temporary), str(path))
    return path


def _lap_numbers(laps):
    numbers = [int(lap["lap"]) for lap in laps]
    return ",".join("%03d" % number for number in numbers) if numbers else "-"


def collision_evidence(outcome, active_signals):
    """Require no contact in the supervisor samples and in every /CollisionData message.

    The supervisor samples contacts once per control loop, so a brief touch can
    be missed there; the raw recorder keeps every message.  An absent or
    undecoded stream cannot prove a collision-free lap.
    """
    summary = outcome.get("collision_summary")
    stream = [row.get("signal") for row in active_signals if row.get("role") == "collision"]
    decoded = [signal for signal in stream if isinstance(signal, dict)]
    contacts = sum(1 for signal in decoded
                   if int(signal.get("objects", 0) or 0) or int(signal.get("objects_a", 0) or 0))
    supervisor_clear = bool(
        isinstance(summary, dict)
        and summary.get("contact_events") == 0
        and summary.get("nonempty_contact_observations") == 0
        and outcome.get("status") != "diagnostic_lap_completed_with_collisions"
    )
    return {
        "supervisor_contact_events": (summary or {}).get("contact_events"),
        "supervisor_contact_observations": (summary or {}).get("nonempty_contact_observations"),
        "collision_stream_messages": len(stream),
        "collision_stream_undecoded": len(stream) - len(decoded),
        "collision_stream_contacts": contacts,
        "collision_free": bool(supervisor_clear and stream
                               and len(decoded) == len(stream) and contacts == 0),
    }


def static_lane_evidence(scenario_name, active_signals):
    """Ego body against the static segment's painted no-cross lines.

    Only full_loop_static drives the segment the pinned markings cover.  A lap
    whose body touched a solid line or swung beyond the deviation limit is not
    kept as an Expert demonstration.
    """
    if scenario_name != "full_loop_static":
        return None
    result = static_lane_compliance.evaluate_trace(
        static_lane_compliance.ego_trace_from_active_signals(active_signals))
    keys = ("passed", "reasons", "min_marking_clearance_m", "closest_marking",
            "touched_markings", "max_right_deviation_m", "max_left_deviation_m",
            "avoidance_class", "trajectory_class", "swing_onset_route_index",
            "peak_route_index", "markings_source_sha256", "policy", "ego_body")
    return {key: result.get(key) for key in keys}


def configured_environment(setup):
    """Weather and hour options the run's own scenario configuration allowed."""
    config = setup.get("scenario_config") or {}
    weathers = {str(profile.get("weather")) for profile in config.get("profiles", [])
                if isinstance(profile, dict) and profile.get("weather")}
    hours = {int(hour) for hour in config.get("scenario_times", [])}
    # Runs recorded before the configuration was embedded used these values.
    return (weathers or {"SUNNY", "FOGGY"}), (hours or {11, 13, 15})


def validate_lap(
    row, expected_scenario="full_loop", campaign=None,
    planned_static_run=None,
):
    run = campaign_run(campaign, row["run_name"]) if campaign else BASE / row["run_name"]
    episode = run / "episode_000"
    errors = []
    seals = {}
    try:
        raw_manifest, _ = verify_run(episode / "raw", payloads=False)
        seals["raw"] = digest(episode / "raw" / "RAW_COMPLETE.json")
    except Exception as exc:
        errors.append("raw:" + str(exc))
        raw_manifest = {}
    for name, (relative, seal_name) in SEALED_STAGES.items():
        try:
            verify_sealed_directory(episode / relative, seal_name)
            seals[name] = digest(episode / relative / seal_name)
        except Exception as exc:
            errors.append(name + ":" + str(exc))

    def safe(relative, default=None):
        try:
            return load(run / relative)
        except Exception as exc:
            errors.append(relative + ":" + str(exc))
            return {} if default is None else default

    sources = None
    try:
        sources = resolve_route_sources(episode, load(run / "sensor_contract.json"))
        verify_route_report(sources, load(episode / "route_signal_audit_v1/report.json"))
    except Exception as exc:
        errors.append("route_source:" + str(exc))

    result = safe("NATIVE_CAPTURE_RESULT.json")
    setup = safe("episode_000/episode.json")
    profile_state = safe("episode_000/campaign_profile_state.json")
    outcome = safe("episode_000/outcome.json")
    native = safe("episode_000/native_audit/report.json")
    admission = effective_admission(run, episode, errors, seals)
    traffic = safe("episode_000/traffic_audit_v1/report.json")
    actors = safe("episode_000/actor_inventory_audit_v1/report.json")
    geometry = safe("episode_000/object_geometry_audit_v1/report.json")
    motion = safe("episode_000/motion_time_v1/report.json")
    review = safe("episode_000/review_closeout/report.json")
    intervention = safe("episode_000/scenario_intervention_audit_v3/report.json")
    active_signals = safe("episode_000/quality/active_signals.json", default=[])
    collision = collision_evidence(outcome, active_signals)
    lane_markings = static_lane_evidence(expected_scenario, active_signals)
    allowed_weather, allowed_hours = configured_environment(setup)

    frames = int(admission.get("frames", 0) or 0)
    counts = admission.get("counts", {})
    native_counts = native.get("counts", {})
    observed_scenario = setup.get("scenario_name", "full_loop")
    intervention_scenario = intervention.get("scenario_name", "full_loop")
    checks = {
        "supervisor_completed": result.get("error") is None,
        "scenario_name": (
            expected_scenario in COLLECTION_SCENARIOS
            and observed_scenario == expected_scenario
            and intervention_scenario == expected_scenario
        ),
        "campaign_seed": (
            int(setup.get("seed", -1)) == int(row["seed"])
            and int(profile_state.get("run_seed", -2)) == int(row["seed"])
        ),
        "campaign_environment": (
            row.get("weather") in allowed_weather
            and int(row["time_hour"]) in allowed_hours
            and setup.get("profile", {}).get("weather") == row.get("weather")
            and setup.get("time_hour") == int(row["time_hour"])
            and profile_state.get("weather") == row.get("weather")
            and profile_state.get("time_hour") == int(row["time_hour"])
            and profile_state.get("static_obstacle_model") == row.get("static_obstacle")
            and profile_state.get("dynamic_obstacle_model") == row.get("dynamic_obstacle")
        ),
        "static_collection_plan_environment": (
            planned_static_run is None
            or (
                setup.get("profile", {}).get("weather")
                == planned_static_run.get("weather")
                and setup.get("time_hour")
                == int(planned_static_run.get("time_hour", -1))
            )
        ),
        "full_lap_completed": outcome.get("full_mission_completed") is True,
        # A lap that touched an actor is not an expert demonstration.
        "collision_free": collision["collision_free"],
        "static_lane_markings_clear": lane_markings is None or lane_markings["passed"] is True,
        "capture_contract": admission.get("capture_contract_valid") is True,
        "runtime_streams": admission.get("runtime_streams_valid") is True,
        "sensor_capture": admission.get("sensor_capture_status") == "admitted",
        "teacher_transport": admission.get("teacher_transport_status") == "admitted",
        "all_native_bundles": (
            native.get("files_valid") is True
            and native.get("complete_cadence") is True
            and not native.get("errors")
            and frames > 0
            and int(native.get("complete_bundles", -1)) == frames
            and all(int(native_counts.get(role, -1)) == frames for role in
                    ("camera_1", "camera_2", "camera_3", "camera_4", "lidar", "gps", "imu"))
        ),
        "teacher_frames": int(counts.get("ego_teacher_time_valid", -1)) == frames,
        "ros_native_lidar_identity": int(review.get("exact_lidar_frames", -1)) == frames,
        "teacher_kinematics": (
            motion.get("all_native_intervals_consistent") is True
            and int(review.get("teacher_consistent_frames", -1)) == frames
            and int(review.get("same_source_gps_teacher_frames", -1)) == frames
            and motion.get("full_ego_stream", {}).get("source_gap_count") == 0
            and motion.get("full_ego_stream", {}).get("moving_position_holds") == 0
        ),
        "dataset_candidates": int(counts.get("dataset_build_candidate", 0)) > 0,
        "traffic_queries": traffic.get("structural_valid") is True,
        "actor_inventory": actors.get("structural_valid") is True,
        "object_geometry_audit": geometry.get("structural_valid") is True,
        "scenario_interventions": intervention.get("valid") is True,
        "timestamps_preserved": admission.get("source_timestamps_modified") is False,
        "not_published_to_training": (
            admission.get("training_visible") is False
            and raw_manifest.get("training_visible") is False
        ),
    }
    if errors:
        checks["all_seals_readable"] = False
    else:
        checks["all_seals_readable"] = True
    tunnel_guard = validate_tunnel_static_collection(episode, expected_scenario, map_root=sources["map_root"] if sources else None) if expected_scenario == TUNNEL_STATIC_SCENARIO else None
    if tunnel_guard is not None:
        checks["tunnel_static_guarded_collection"] = tunnel_guard["valid"]
    return {
        "lap": int(row["lap"]),
        "seed": int(row["seed"]),
        "run_name": row["run_name"],
        "route_source_resolution": sources["provenance"] if sources else None,
        "run": str(run),
        "scenario_name": observed_scenario,
        "weather": setup.get("profile", {}).get("weather"),
        "time_hour": setup.get("time_hour"),
        "static_obstacle": row.get("static_obstacle"),
        "dynamic_obstacle": row.get("dynamic_obstacle"),
        "frames": frames,
        "dataset_build_candidates": int(counts.get("dataset_build_candidate", 0) or 0),
        "excluded_frames": int(counts.get("excluded_from_dataset_build", 0) or 0),
        "native_files": sum(int(value) for value in native_counts.values()) if native_counts else 0,
        "native_bytes": sum(int(item.get("size_bytes", 0)) for item in native.get("assets", [])),
        "observed_rtf": native.get("observed_rtf"),
        "collision": collision,
        "static_lane_compliance": lane_markings,
        "tunnel_static_guarded_collection": tunnel_guard,
        "checks": checks,
        "errors": errors,
        "input_seals": seals,
        "campaign_profile_state_sha256": (
            digest(episode / "campaign_profile_state.json")
            if (episode / "campaign_profile_state.json").is_file() else None
        ),
        "valid": all(checks.values()),
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", help="campaign_id or local campaign directory")
    parser.add_argument("--no-prepare-training", action="store_true",
                        help="Only validate capture; do not build the direct-reference training manifests")
    parser.add_argument("--val-lap", type=int, action="append", default=[],
                        help="Explicit whole-run validation holdout for campaigns without a pinned split plan")
    parser.add_argument("--rebuild-route-review", action="store_true",
                        help="Archive route audit, review and admission, then rebuild from captured route inputs")
    parser.add_argument("--allow-valid-subset", action="store_true",
                        help="Publish only valid complete laps when failed laps are sealed and excluded")
    parser.add_argument("--delete-invalid-payload", action="store_true",
                        help="Delete the native payload of laps that fail validation; "
                             "by default it is kept for review and excluded from training and NAS")
    nas = parser.add_mutually_exclusive_group()
    nas.add_argument("--no-nas-copy", action="store_true", help="Finish with local validation only")
    nas.add_argument("--nas-plan-only", action="store_true", help="Plan export without accessing NAS")
    nas.add_argument("--nas-only", metavar="REPORT", help="Resume export from a passed sealed report")
    parser.add_argument("--nas-root", help="Windows NAS root; overrides MORAI_NAS_ROOT/local config")
    args = parser.parse_args()
    campaign = resolve_campaign(args.campaign)
    direct_layout = is_direct_campaign_metadata(campaign, SSD_ROOT)
    if args.nas_only:
        report_path = Path(args.nas_only).resolve()
        # Legacy: <campaign>/validation_*/report.json.
        # Direct: <campaign>/validations/validation_*/report.json.
        owner = report_path.parent.parent
        if direct_layout and owner.name == "validations":
            owner = owner.parent
        if owner != campaign.resolve():
            parser.error("--nas-only report must belong to the requested campaign")
        from nas_publish import publish_validated
        try:
            print(json.dumps(publish_validated(report_path, args.nas_root), indent=2))
        except Exception as exc:
            print("[NAS] Local data retained: " + str(exc), file=sys.stderr)
            raise SystemExit(4)
        return
    configured_python = os.environ.get("MORAI_ANALYSIS_PYTHON", "").strip()
    candidates = [
        Path(configured_python).expanduser() if configured_python else None,
        Path(os.environ["CONDA_PREFIX"]) / "bin/python"
        if os.environ.get("CONDA_PREFIX") else None,
        Path.home() / "miniconda3/envs/lmdrive/bin/python3.8",
        Path(shutil.which("python3") or sys.executable),
    ]
    analysis_python = next(
        (path.resolve() for path in candidates if path is not None and path.is_file()),
        Path(sys.executable).resolve(),
    )
    review_env = build_analysis_environment(
        analysis_python, os.environ,
        OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
    )
    analysis_environment_report = probe_analysis_environment(
        analysis_python, review_env)
    plan = load(campaign / "campaign_plan.json")
    if args.val_lap and (args.no_prepare_training or plan.get("static_collection_plan_evidence")):
        parser.error("--val-lap requires training preparation and no pinned static split plan")
    rows = completed_rows(campaign / "runs.tsv")
    requested = int(plan["requested_laps"])
    static_plan_rows = {}
    static_plan_name = plan.get("static_collection_plan_evidence")
    if static_plan_name:
        static_plan_path = campaign / static_plan_name
        expected_static_plan_sha = plan.get("static_collection_plan_sha256")
        if not static_plan_path.is_file():
            raise ValueError("sealed static collection plan is missing")
        if not expected_static_plan_sha or digest(static_plan_path) != expected_static_plan_sha:
            raise ValueError("sealed static collection plan SHA-256 mismatch")
        static_plan = load(static_plan_path)
        if static_plan.get("scenario") != "full_loop_static":
            raise ValueError("invalid static collection plan scenario")
        static_plan_rows = {
            int(item["run"]): item for item in static_plan.get("runs", [])
        }
        if sorted(static_plan_rows)[:requested] != list(range(1, requested + 1)):
            raise ValueError("static collection plan does not cover every requested lap")
    scenario_name = plan.get("scenario_name", "full_loop")
    scenario_supported = scenario_name in COLLECTION_SCENARIOS
    laps = []
    all_attempts_terminal = True
    cleanup_evidence_ok = True
    validation_infrastructure_errors = False
    for index in range(1, requested + 1):
        row = rows.get(index)
        if row is None:
            all_attempts_terminal = False
            laps.append({
                "lap": index,
                "capture_status": "missing",
                "validation_status": "not_run",
                "training_usable": False,
                "valid": False,
                "errors": ["lap has no terminal record"],
                "checks": {},
            })
        elif row.get("status", "").startswith("failed_"):
            run = campaign_run(campaign, row["run_name"])
            marker = _failed_attempt_marker(run, "capture")
            if marker is None:
                cleanup_evidence_ok = False
            laps.append({
                "lap": index,
                "seed": int(row["seed"]),
                "run_name": row["run_name"],
                "run": str(run),
                "capture_status": "failed",
                "validation_status": "skipped",
                "training_usable": False,
                "valid": False,
                "errors": [row["status"]] + (
                    [] if marker is not None
                    else ["capture failure cleanup evidence is missing"]
                ),
                "checks": {},
                "payload_cleanup": (
                    _cleanup_summary(marker) if marker is not None
                    else {"status": "missing", "failure_kind": "capture"}
                ),
            })
        elif row.get("status") in ("captured", "completed"):
            run = campaign_run(campaign, row["run_name"])
            previous_invalid = _failed_attempt_marker(run, "validation")
            if previous_invalid is not None:
                restored = previous_invalid.get("evidence", {})
                if not isinstance(restored, dict) or restored.get("valid") is not False:
                    cleanup_evidence_ok = False
                    restored = {
                        "lap": index,
                        "run_name": row["run_name"],
                        "valid": False,
                        "errors": ["invalid validation cleanup evidence"],
                        "checks": {},
                    }
                restored.update({
                    "capture_status": "succeeded",
                    "validation_status": "failed",
                    "training_usable": False,
                    "payload_cleanup": _cleanup_summary(previous_invalid),
                })
                laps.append(restored)
                continue
            log_path = campaign / (row["run_name"] + "_offline_review.log")
            try:
                if args.rebuild_route_review:
                    archived = archive_route_review(run)
                    print("[REBUILD] Previous route-derived evidence retained at %s" % archived, flush=True)
                with log_path.open("a", encoding="utf-8") as log:
                    complete_offline_review(run, analysis_python=analysis_python,
                                            env=review_env, log=log)
                    admission_report = (
                        run / "episode_000" / "admission_v1" / "report.json"
                    )
                    if admission_report.is_file():
                        complete_bev_geometry_recovery(
                            run, analysis_python=analysis_python,
                            env=review_env, log=log)
                lap = validate_lap(
                    row, scenario_name, campaign,
                    planned_static_run=static_plan_rows.get(index),
                )
                lap.update({
                    "capture_status": "succeeded",
                    "validation_status": "passed" if lap["valid"] else "failed",
                    "training_usable": bool(lap["valid"]),
                })
                if not lap["valid"] and not args.delete_invalid_payload:
                    # A failed check is a judgement that may itself be wrong
                    # (policy, environment, a clock gate); never destroy the
                    # only copy of a captured lap without an explicit request.
                    lap["payload_cleanup"] = {
                        "status": "retained",
                        "failure_kind": "validation",
                        "reason": "validation failure keeps the payload; "
                                  "rerun with --delete-invalid-payload to reclaim space",
                    }
                elif not lap["valid"]:
                    try:
                        marker = cleanup_failed_attempt(
                            row["run_name"],
                            base=(campaign_root_from_metadata(campaign)
                                  if direct_layout else BASE),
                            run_root=(run if direct_layout else None),
                            failure_kind="validation",
                            failure_reason="; ".join(lap.get("errors", []))
                            or "validation checks failed",
                            validated_invalid=True,
                            evidence=lap,
                        )
                        lap["payload_cleanup"] = _cleanup_summary(marker)
                    except Exception as exc:
                        cleanup_evidence_ok = False
                        lap.setdefault("errors", []).append(
                            "failed_payload_cleanup:" + repr(exc))
                        lap["payload_cleanup"] = {
                            "status": "failed",
                            "failure_kind": "validation",
                            "error": repr(exc),
                        }
                laps.append(lap)
            except Exception as exc:
                validation_infrastructure_errors = True
                laps.append({
                    "lap": index,
                    "run_name": row["run_name"],
                    "run": str(run),
                    "capture_status": "succeeded",
                    "validation_status": "error",
                    "training_usable": False,
                    "valid": False,
                    "errors": ["offline_review:" + repr(exc)],
                    "checks": {},
                    "offline_review_log": str(log_path),
                    "payload_cleanup": {
                        "status": "not_attempted",
                        "reason": "validation infrastructure error; payload preserved",
                    },
                })
        else:
            all_attempts_terminal = False
            laps.append({
                "lap": index,
                "seed": int(row["seed"]),
                "run_name": row["run_name"],
                "capture_status": "interrupted",
                "validation_status": "not_run",
                "training_usable": False,
                "valid": False,
                "errors": ["nonterminal ledger status: " + row.get("status", "")],
                "checks": {},
            })

    valid_laps = [row for row in laps if row.get("valid") is True]
    adjacent_variation = []
    for before, after in zip(valid_laps, valid_laps[1:]):
        contract_required = int(after["lap"]) == int(before["lap"]) + 1
        adjacent_variation.append({
            "before_lap": before["lap"],
            "after_lap": after["lap"],
            "contract_required": contract_required,
            "weather_changed": before.get("weather") != after.get("weather"),
            "time_changed": before.get("time_hour") != after.get("time_hour"),
            "static_obstacle_changed": before.get("static_obstacle") != after.get("static_obstacle"),
            "dynamic_obstacle_changed": before.get("dynamic_obstacle") != after.get("dynamic_obstacle"),
        })
    # The team scenario contract alternates weather. Time and both obstacle
    # models are seeded random choices, so an adjacent repeat is valid and
    # must not make a sound capture fail.
    variation_valid = all(
        not row["contract_required"]
        or (
            row["weather_changed"]
            and (not static_plan_rows or row["time_changed"])
        )
        for row in adjacent_variation
    )
    variation_coverage = {
        "weather": sorted({row.get("weather") for row in valid_laps if row.get("weather")}),
        "time_hour": sorted({row.get("time_hour") for row in valid_laps
                             if row.get("time_hour") is not None}),
        "static_obstacle": sorted({row.get("static_obstacle") for row in valid_laps
                                   if row.get("static_obstacle")}),
        "dynamic_obstacle": sorted({row.get("dynamic_obstacle") for row in valid_laps
                                    if row.get("dynamic_obstacle")}),
    }
    completion = campaign / "COLLECTION_COMPLETE.json"
    completion_payload = load(completion) if completion.is_file() else {}
    completion_scenario = completion_payload.get("scenario_name", "full_loop")
    captured_laps = [
        row for row in laps if row.get("capture_status") == "succeeded"]
    capture_failures = [
        row for row in laps if row.get("capture_status") == "failed"]
    validation_failures = [
        row for row in captured_laps if row.get("valid") is not True]
    completed_attempts = len(captured_laps) + len(capture_failures)
    completion_counts_match = bool(
        completion.is_file()
        and int(completion_payload.get(
            "attempted_laps",
            completion_payload.get("captured_laps", -1))) == requested
        and int(completion_payload.get("captured_laps", -1)) == len(captured_laps)
        and int(completion_payload.get("failed_laps", 0)) == len(capture_failures)
    )
    invalid_laps = [row for row in laps if row.get("valid") is not True]
    minimum_valid_laps = 1 if requested == 1 else 2
    standard_valid = bool(
        completion.is_file()
        and scenario_supported
        and completion_scenario == scenario_name
        and len(laps) == requested
        and all_attempts_terminal
        and completion_counts_match
        and cleanup_evidence_ok
        and not validation_infrastructure_errors
        and len(valid_laps) >= minimum_valid_laps
        and variation_valid
    )
    full_variation_coverage = {
        "weather": sorted({row.get("weather") for row in laps if row.get("weather")}),
        "time_hour": sorted({row.get("time_hour") for row in laps
                             if row.get("time_hour") is not None}),
        "static_obstacle": sorted({row.get("static_obstacle") for row in laps
                                   if row.get("static_obstacle")}),
        "dynamic_obstacle": sorted({row.get("dynamic_obstacle") for row in laps
                                    if row.get("dynamic_obstacle")}),
    }
    subset_coverage_preserved = all(
        variation_coverage[key] == full_variation_coverage[key]
        for key in variation_coverage
    )
    subset_mode = bool(args.allow_valid_subset and invalid_laps)
    subset_valid = bool(
        standard_valid and subset_mode and len(valid_laps) >= 2
        and subset_coverage_preserved
    )
    published_laps = valid_laps if subset_mode else laps
    report_schema = ("morai_sensor_record_campaign_validation_subset_v1"
                     if subset_mode else "morai_sensor_record_campaign_validation_v1")
    report = {
        "schema": report_schema,
        "campaign": str(campaign),
        "scenario_name": scenario_name,
        "analysis_python": str(analysis_python),
        "analysis_environment": analysis_environment_report,
        "scenario_supported": scenario_supported,
        "campaign_plan_sha256": digest(campaign / "campaign_plan.json"),
        "static_collection_plan_sha256": plan.get(
            "static_collection_plan_sha256"
        ),
        "runs_ledger_sha256": digest(campaign / "runs.tsv"),
        "collection_completion_present": completion.is_file(),
        "collection_completion_scenario_matches": completion_scenario == scenario_name,
        "collection_completion_counts_match": completion_counts_match,
        "collection_completion_sha256": digest(completion) if completion.is_file() else None,
        "requested_laps": requested,
        "requested_attempts": requested,
        "completed_attempts": completed_attempts,
        "all_attempts_terminal": all_attempts_terminal,
        "capture_succeeded": len(captured_laps),
        "capture_failed": len(capture_failures),
        "capture_failed_laps": [int(row["lap"]) for row in capture_failures],
        "validation_passed": len(valid_laps),
        "validation_failed": len(validation_failures),
        "validation_failed_laps": [int(row["lap"]) for row in validation_failures],
        "completed_valid_laps": len(valid_laps),
        "selected_laps": len(valid_laps),
        "all_captured_laps_valid": not validation_failures,
        "cleanup_evidence_valid": cleanup_evidence_ok,
        "validation_infrastructure_errors": validation_infrastructure_errors,
        "total_frames": sum(int(row.get("frames", 0)) for row in valid_laps),
        "total_dataset_build_candidates": sum(
            int(row.get("dataset_build_candidates", 0)) for row in valid_laps),
        "total_excluded_frames": sum(
            int(row.get("excluded_frames", 0)) for row in valid_laps),
        "total_native_files": sum(int(row.get("native_files", 0)) for row in valid_laps),
        "total_native_bytes": sum(int(row.get("native_bytes", 0)) for row in valid_laps),
        "captured_total_frames": sum(int(row.get("frames", 0)) for row in laps),
        "captured_total_native_bytes": sum(int(row.get("native_bytes", 0)) for row in laps),
        "laps": published_laps,
        "excluded_laps": invalid_laps if subset_mode else [],
        "subset_policy": (
            "whole_lap_fail_closed_exclusion_with_full_variation_preservation_v1"
            if subset_mode else "terminal_failed_attempts_retained_as_evidence_v1"
        ),
        "subset_coverage_preserved": subset_coverage_preserved,
        "adjacent_environment_and_obstacle_variation": adjacent_variation,
        "variation_contract": (
            "sealed static plan: weather and time do not repeat on adjacent runs"
            if static_plan_rows else
            "weather alternates; time and obstacle choices may repeat"
        ),
        "variation_coverage": variation_coverage,
        "full_attempt_variation_coverage": full_variation_coverage,
        "variation_valid": variation_valid,
        "valid": subset_valid if subset_mode else standard_valid,
        "source_timestamps_modified": False,
        "physical_simultaneity_verified": False,
        "training_visible": False,
        "interpretation": (
            "PASS proves sealed native inputs, 10Hz cadence, ROS/native LiDAR identity, "
            "teacher continuity, actor/light transport, selected-scenario completion, and per-lap "
            "environment contract (including sealed static-plan weather/time when present). "
            "It does not create MORAI's missing official "
            "millisecond physical-exposure guarantee."
        ),
    }
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ((campaign / "validations" / ("validation_" + stamp))
              if direct_layout else campaign / ("validation_" + stamp))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    if direct_layout:
        report["campaign"] = relative_path(campaign, output)
        for lap in report["laps"]:
            if lap.get("run"):
                lap["run"] = relative_path(
                    campaign_run(campaign, lap["run_name"]), output)
        for lap in report.get("excluded_laps", []):
            if lap.get("run"):
                lap["run"] = relative_path(
                    campaign_run(campaign, lap["run_name"]), output)
    write_json(output / "report.json", report)
    write_json(output / "COMPLETE.json", {
        "schema": report_schema,
        "status": "passed" if report["valid"] else "failed",
        "files": {"report.json": digest(output / "report.json")},
        "source_timestamps_modified": False,
        "physical_simultaneity_verified": False,
        "training_visible": False,
    })
    training = None
    if report["valid"] and len(valid_laps) >= 2 and not args.no_prepare_training:
        training_output = output / "training_dataset_v1"
        print("[BUILD] preparing direct-reference visual training manifests", flush=True)
        build_command = [
            str(analysis_python),
            str(INTEGRATION / "tools" / "build_sensor_record_training_dataset.py"),
            "--validation-report", str(output / "report.json"),
            "--output", str(training_output),
        ]
        build_command.extend(admission_recovery_build_arguments([
            campaign_run(campaign,lap["run_name"]) for lap in valid_laps]))
        if args.val_lap:
            for lap_number in args.val_lap:
                build_command.extend(["--val-lap", str(lap_number)])
        split_plan_name = plan.get("static_collection_plan_evidence")
        if split_plan_name:
            split_plan_path = campaign / split_plan_name
            expected_split_sha = plan.get("static_collection_plan_sha256")
            if not expected_split_sha:
                raise ValueError(
                    "campaign plan is missing static collection plan SHA-256"
                )
            build_command.extend([
                "--split-plan", str(split_plan_path),
                "--split-plan-sha256", str(expected_split_sha),
            ])
        built = subprocess.run(build_command, env=review_env, text=True)
        training = {
            "return_code": int(built.returncode),
            "manifest_index": str(training_output / "manifest_index.json"),
            "training_ready": built.returncode == 0,
        }
    csv_path = _write_lap_results_csv(campaign, laps)
    print("[VALIDATION SUMMARY]")
    print("  total attempts      : %d/%d" % (completed_attempts, requested))
    print("  capture succeeded   : %d" % len(captured_laps))
    print("  capture failed      : %d [%s]" % (
        len(capture_failures), _lap_numbers(capture_failures)))
    print("  validation passed   : %d" % len(valid_laps))
    print("  validation failed   : %d [%s]" % (
        len(validation_failures), _lap_numbers(validation_failures)))
    print("  training usable     : %d" % len(valid_laps))
    print("[CSV] Windows: " + windows_path(csv_path))
    print("[CSV] WSL    : " + str(csv_path))
    print(json.dumps({
        "valid": report["valid"],
        "scenario_name": scenario_name,
        "requested_laps": requested,
        "requested_attempts": requested,
        "completed_attempts": completed_attempts,
        "capture_failed_laps": report["capture_failed_laps"],
        "validation_failed_laps": report["validation_failed_laps"],
        "completed_valid_laps": report["completed_valid_laps"],
        "total_frames": report["total_frames"],
        "total_dataset_build_candidates": report["total_dataset_build_candidates"],
        "total_native_files": report["total_native_files"],
        "total_native_bytes": report["total_native_bytes"],
        "variation_valid": variation_valid,
        "report": str(output / "report.json"),
        "lap_results_csv": str(csv_path),
        "physical_simultaneity_verified": False,
        "training_visible": bool(training and training["training_ready"]),
        "training": training,
    }, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(2)
    if training is not None and not training["training_ready"]:
        raise SystemExit(3)
    local_publication = None
    if direct_layout:
        local_publication = {
            "schema": "morai_local_dataset_campaign_v2",
            "scenario_folder": plan["dataset_scenario_folder"],
            "campaign_id": plan["campaign_id"],
            "dataset_campaign_id": plan.get("dataset_campaign_id", plan["campaign_id"]),
            "campaign": str(campaign_root_from_metadata(campaign)),
            "hardlinked": False,
            "direct_capture": True,
        }
        print("[LOCAL] direct dataset validated: "
              + local_publication["campaign"], flush=True)
    elif plan.get("campaign_id"):
        from nas_publish import materialize_validated
        try:
            local_publication = materialize_validated(output / "report.json")
            write_json(output / "local_dataset_publication.json", local_publication)
            print("[LOCAL] canonical dataset ready: " + local_publication["campaign"],
                  flush=True)
        except Exception as exc:
            print("[LOCAL] Validation passed but canonical dataset materialization failed: "
                  + str(exc), file=sys.stderr)
            raise SystemExit(4)
    else:
        print("[LOCAL] legacy campaign plan has no campaign_id; canonical materialization skipped",
              flush=True)

    # A direct campaign becomes visible only after every local validation and
    # publication artifact is sealed.  os.replace is an instant same-volume
    # directory rename; no sensor payload is copied.
    if direct_layout:
        campaign, output, local_publication, promoted = (
            finalize_direct_local_publication(
                campaign, output, SSD_ROOT, scenario_name, plan))
        print("[LOCAL] canonical dataset ready: "
              + local_publication["campaign"], flush=True)

    from runtime_paths import LOCAL_MACHINE
    auto_nas = LOCAL_MACHINE.get("nas_auto_publish", False)
    if not isinstance(auto_nas, bool):
        print("[NAS] local_machine.json nas_auto_publish must be true or false",
              file=sys.stderr)
        raise SystemExit(4)
    nas_requested = bool(
        args.nas_plan_only or args.nas_root
        or os.environ.get("MORAI_NAS_ROOT", "").strip()
        or auto_nas
    )
    if not args.no_nas_copy and nas_requested:
        if local_publication is None:
            print("[NAS] canonical local dataset was not materialized", file=sys.stderr)
            raise SystemExit(4)
        from nas_publish import publish_validated
        try:
            exports = publish_validated(
                output / "report.json", args.nas_root, args.nas_plan_only,
                local_publication=local_publication)
            write_json(output / "nas_publication.json", {"exports": exports})
        except Exception as exc:
            print("[NAS] Local validation PASSED; export incomplete, local data retained: "
                  + str(exc), file=sys.stderr)
            raise SystemExit(4)
    elif not args.no_nas_copy:
        print("[NAS] automatic publication is disabled on this PC; local dataset retained",
              flush=True)


def main():
    from process_guard import process_lock
    with process_lock("campaign"), process_lock("capture"), process_lock("training"):
        _main()


if __name__ == "__main__":
    main()
