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
from ssd_storage import SSD_ROOT, require_ssd
from scenario_runner.scenarios.segment_modes.modes import PRIMARY_SCENARIOS


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


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def effective_admission(run, episode, errors, seals):
    """Use an additive, sealed recalculation without changing admission_v1."""
    original_path = episode / "admission_v1" / "report.json"
    original = load(original_path)
    recovery_path = run / "ADMISSION_RECOVERY.json"
    if not recovery_path.is_file():
        return original
    try:
        recovery = load(recovery_path)
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


def resolve_campaign(value):
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.exists():
        path = BASE / "campaigns" / path
    path = path.resolve()
    require_ssd(path, reserve_bytes=0)
    return path


def completed_rows(path):
    latest = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            latest[int(row["lap"])] = row
    return latest


def validate_lap(row, expected_scenario="full_loop"):
    run = BASE / row["run_name"]
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

    frames = int(admission.get("frames", 0) or 0)
    counts = admission.get("counts", {})
    native_counts = native.get("counts", {})
    observed_scenario = setup.get("scenario_name", "full_loop")
    intervention_scenario = intervention.get("scenario_name", "full_loop")
    checks = {
        "supervisor_completed": result.get("error") is None,
        "scenario_name": (
            expected_scenario in PRIMARY_SCENARIOS
            and observed_scenario == expected_scenario
            and intervention_scenario == expected_scenario
        ),
        "campaign_seed": (
            int(setup.get("seed", -1)) == int(row["seed"])
            and int(profile_state.get("run_seed", -2)) == int(row["seed"])
        ),
        "campaign_environment": (
            row.get("weather") in ("SUNNY", "FOGGY")
            and int(row["time_hour"]) in (11, 13, 15)
            and setup.get("profile", {}).get("weather") == row.get("weather")
            and setup.get("time_hour") == int(row["time_hour"])
            and profile_state.get("weather") == row.get("weather")
            and profile_state.get("time_hour") == int(row["time_hour"])
            and profile_state.get("static_obstacle_model") == row.get("static_obstacle")
            and profile_state.get("dynamic_obstacle_model") == row.get("dynamic_obstacle")
        ),
        "full_lap_completed": outcome.get("full_mission_completed") is True,
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
    return {
        "lap": int(row["lap"]),
        "seed": int(row["seed"]),
        "run_name": row["run_name"],
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
        "checks": checks,
        "errors": errors,
        "input_seals": seals,
        "campaign_profile_state_sha256": (
            digest(episode / "campaign_profile_state.json")
            if (episode / "campaign_profile_state.json").is_file() else None
        ),
        "valid": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", help="campaign_id or local campaign directory")
    parser.add_argument("--no-prepare-training", action="store_true",
                        help="Only validate capture; do not build the direct-reference training manifests")
    args = parser.parse_args()
    campaign = resolve_campaign(args.campaign)
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
    dependency_check = subprocess.run(
        [str(analysis_python), "-c", "import cv2,numpy,pyproj"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if dependency_check.returncode:
        raise RuntimeError("Offline analysis dependencies missing: " + dependency_check.stderr[-500:])
    review_env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    from runtime_paths import CATKIN_ROOT, ROS_ROOT
    review_env["PYTHONPATH"] = ":".join([
        os.environ.get("MORAI_ASYNC_DEPS", ""),
        str(INTEGRATION),
        str(PROJECT_ROOT),
        str(CATKIN_ROOT / "devel/lib/python3/dist-packages"),
        str(ROS_ROOT / "lib/python3/dist-packages"),
        review_env.get("PYTHONPATH", ""),
    ])
    plan = load(campaign / "campaign_plan.json")
    rows = completed_rows(campaign / "runs.tsv")
    requested = int(plan["requested_laps"])
    scenario_name = plan.get("scenario_name", "full_loop")
    scenario_supported = scenario_name in PRIMARY_SCENARIOS
    laps = []
    for index in range(1, requested + 1):
        row = rows.get(index)
        if row is None or row.get("status") not in ("captured", "completed"):
            laps.append({
                "lap": index,
                "valid": False,
                "errors": ["lap has no completed terminal record"],
                "checks": {},
            })
        else:
            run = BASE / row["run_name"]
            log_path = campaign / (row["run_name"] + "_offline_review.log")
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    complete_offline_review(run, analysis_python=analysis_python,
                                            env=review_env, log=log)
                laps.append(validate_lap(row, scenario_name))
            except Exception as exc:
                laps.append({
                    "lap": index,
                    "run_name": row["run_name"],
                    "valid": False,
                    "errors": ["offline_review:" + repr(exc)],
                    "checks": {},
                    "offline_review_log": str(log_path),
                })

    adjacent_variation = []
    for before, after in zip(laps, laps[1:]):
        adjacent_variation.append({
            "before_lap": before["lap"],
            "after_lap": after["lap"],
            "weather_changed": before.get("weather") != after.get("weather"),
            "time_changed": before.get("time_hour") != after.get("time_hour"),
            "static_obstacle_changed": before.get("static_obstacle") != after.get("static_obstacle"),
            "dynamic_obstacle_changed": before.get("dynamic_obstacle") != after.get("dynamic_obstacle"),
        })
    # The team scenario contract alternates weather. Time and both obstacle
    # models are seeded random choices, so an adjacent repeat is valid and
    # must not make a sound capture fail.
    variation_valid = all(
        row["weather_changed"] for row in adjacent_variation
    )
    variation_coverage = {
        "weather": sorted({row.get("weather") for row in laps if row.get("weather")}),
        "time_hour": sorted({row.get("time_hour") for row in laps
                             if row.get("time_hour") is not None}),
        "static_obstacle": sorted({row.get("static_obstacle") for row in laps
                                   if row.get("static_obstacle")}),
        "dynamic_obstacle": sorted({row.get("dynamic_obstacle") for row in laps
                                    if row.get("dynamic_obstacle")}),
    }
    completion = campaign / "COLLECTION_COMPLETE.json"
    completion_payload = load(completion) if completion.is_file() else {}
    completion_scenario = completion_payload.get("scenario_name", "full_loop")
    report = {
        "schema": "morai_sensor_record_campaign_validation_v1",
        "campaign": str(campaign),
        "scenario_name": scenario_name,
        "scenario_supported": scenario_supported,
        "campaign_plan_sha256": digest(campaign / "campaign_plan.json"),
        "runs_ledger_sha256": digest(campaign / "runs.tsv"),
        "collection_completion_present": completion.is_file(),
        "collection_completion_scenario_matches": completion_scenario == scenario_name,
        "collection_completion_sha256": digest(completion) if completion.is_file() else None,
        "requested_laps": requested,
        "completed_valid_laps": sum(bool(row.get("valid")) for row in laps),
        "total_frames": sum(int(row.get("frames", 0)) for row in laps),
        "total_dataset_build_candidates": sum(int(row.get("dataset_build_candidates", 0)) for row in laps),
        "total_excluded_frames": sum(int(row.get("excluded_frames", 0)) for row in laps),
        "total_native_files": sum(int(row.get("native_files", 0)) for row in laps),
        "total_native_bytes": sum(int(row.get("native_bytes", 0)) for row in laps),
        "laps": laps,
        "adjacent_environment_and_obstacle_variation": adjacent_variation,
        "variation_contract": "weather alternates; time and obstacle choices may repeat",
        "variation_coverage": variation_coverage,
        "variation_valid": variation_valid,
        "valid": bool(completion.is_file() and scenario_supported
                      and completion_scenario == scenario_name
                      and len(laps) == requested
                      and all(row.get("valid") for row in laps)
                      and variation_valid),
        "source_timestamps_modified": False,
        "physical_simultaneity_verified": False,
        "training_visible": False,
        "interpretation": (
            "PASS proves sealed native inputs, 10Hz cadence, ROS/native LiDAR identity, "
            "teacher continuity, actor/light transport, selected-scenario completion, and per-lap "
            "team-scenario weather alternation. It does not create MORAI's missing official "
            "millisecond physical-exposure guarantee."
        ),
    }
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = campaign / ("validation_" + stamp)
    output.mkdir(exist_ok=False)
    write_json(output / "report.json", report)
    write_json(output / "COMPLETE.json", {
        "schema": "morai_sensor_record_campaign_validation_v1",
        "status": "passed" if report["valid"] else "failed",
        "files": {"report.json": digest(output / "report.json")},
        "source_timestamps_modified": False,
        "physical_simultaneity_verified": False,
        "training_visible": False,
    })
    training = None
    if report["valid"] and requested >= 2 and not args.no_prepare_training:
        training_output = output / "training_dataset_v1"
        print("[BUILD] preparing direct-reference visual training manifests", flush=True)
        built = subprocess.run([
            str(analysis_python),
            str(INTEGRATION / "tools" / "build_sensor_record_training_dataset.py"),
            "--validation-report", str(output / "report.json"),
            "--output", str(training_output),
        ], env=review_env, text=True)
        training = {
            "return_code": int(built.returncode),
            "manifest_index": str(training_output / "manifest_index.json"),
            "training_ready": built.returncode == 0,
        }
    print(json.dumps({
        "valid": report["valid"],
        "scenario_name": scenario_name,
        "requested_laps": requested,
        "completed_valid_laps": report["completed_valid_laps"],
        "total_frames": report["total_frames"],
        "total_dataset_build_candidates": report["total_dataset_build_candidates"],
        "total_native_files": report["total_native_files"],
        "total_native_bytes": report["total_native_bytes"],
        "variation_valid": variation_valid,
        "report": str(output / "report.json"),
        "physical_simultaneity_verified": False,
        "training_visible": bool(training and training["training_ready"]),
        "training": training,
    }, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(2)
    if training is not None and not training["training_ready"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
