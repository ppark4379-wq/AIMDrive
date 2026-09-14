#!/usr/bin/env python3
"""Seal an additive offline-review recovery without rewriting capture results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.journal import digest, write_json, verify_run
from async_capture.sensor_record_admission import verify_sealed_directory
from ssd_storage import require_ssd


SEALED_STAGES = {
    "quality": ("quality", "QUALITY_COMPLETE.json"),
    "native_audit": ("native_audit", "COMPLETE.json"),
    "native_associations": ("native_associations", "COMPLETE.json"),
    "gt_timeline": ("gt_timeline_v1", "COMPLETE.json"),
    "traffic_audit": ("traffic_audit_v1", "COMPLETE.json"),
    "actor_inventory_audit": ("actor_inventory_audit_v1", "COMPLETE.json"),
    "object_geometry_audit": ("object_geometry_audit_v1", "COMPLETE.json"),
    "route_signal_audit": ("route_signal_audit_v1", "COMPLETE.json"),
    "motion_time": ("motion_time_v1", "COMPLETE.json"),
    "exposure": ("exposure_v1", "COMPLETE.json"),
    "review_closeout": ("review_closeout", "COMPLETE.json"),
    "admission": ("admission_v1", "COMPLETE.json"),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--expected-error-substring",
        default="audit_scenario_interventions.py",
    )
    parser.add_argument(
        "--scenario-intervention-dir",
        default="scenario_intervention_audit_v3",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    require_ssd(run, reserve_bytes=0)
    output = run / "POSTPROCESS_RECOVERY.json"
    if output.exists():
        raise FileExistsError(output)

    original_path = run / "NATIVE_CAPTURE_RESULT.json"
    original = load(original_path)
    if not original.get("error") or original.get("admission") is not None:
        raise ValueError("Recovery requires an original failed postprocess result")
    supervisor_log = run.parent / (run.name + "_supervisor.log")
    supervisor_text = supervisor_log.read_text(encoding="utf-8", errors="replace")
    failure_text = str(original.get("error", "")) + "\n" + supervisor_text
    if args.expected_error_substring not in failure_text:
        raise ValueError("Expected original failure evidence was not found")

    probe = subprocess.run(
        [sys.executable, "-c", "import cv2,numpy,pyproj"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )
    if probe.returncode:
        raise RuntimeError("Recovery interpreter dependencies are incomplete")

    episode = run / "episode_000"
    verify_run(episode / "raw", payloads=False)
    seals = {"raw": digest(episode / "raw" / "RAW_COMPLETE.json")}
    for name, (relative, seal_name) in SEALED_STAGES.items():
        folder = episode / relative
        verify_sealed_directory(folder, seal_name)
        seals[name] = digest(folder / seal_name)
    intervention_folder = episode / args.scenario_intervention_dir
    verify_sealed_directory(intervention_folder, "COMPLETE.json")
    seals["scenario_interventions"] = digest(
        intervention_folder / "COMPLETE.json"
    )

    admission = load(episode / "admission_v1" / "report.json")
    intervention = load(intervention_folder / "report.json")
    if not (
        admission.get("capture_contract_valid") is True
        and admission.get("runtime_streams_valid") is True
        and admission.get("sensor_capture_status") == "admitted"
        and admission.get("teacher_transport_status") == "admitted"
        and admission.get("training_visible") is False
        and intervention.get("valid") is True
        and intervention.get("training_visible") is False
    ):
        raise ValueError("Recovered evidence does not pass collection admission")

    write_json(output, {
        "schema": "morai_sensor_record_postprocess_recovery_v1",
        "status": "completed",
        "recovery_is_additive": True,
        "original_result": {
            "path": original_path.name,
            "sha256": digest(original_path),
            "error": original["error"],
        },
        "failure_evidence": {
            "supervisor_log": supervisor_log.name,
            "sha256": digest(supervisor_log),
            "matched": args.expected_error_substring,
        },
        "analysis_python": sys.executable,
        "analysis_dependencies": ["cv2", "numpy", "pyproj"],
        "input_seals": seals,
        "admission_report_sha256": digest(
            episode / "admission_v1" / "report.json"
        ),
        "intervention_report_sha256": digest(
            intervention_folder / "report.json"
        ),
        "intervention_report_directory": args.scenario_intervention_dir,
        "source_timestamps_modified": False,
        "physical_simultaneity_verified": False,
        "training_visible": False,
        "script_sha256": digest(Path(__file__)),
    })
    print(json.dumps(load(output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
