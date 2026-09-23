#!/usr/bin/env python3
"""Add scale-aware object evidence and a new publication to a sealed dataset.

This never invokes the capture validator or its failed-attempt cleanup. Native
assets, existing reports, recoveries, manifests and run splits remain intact.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]

from async_capture.journal import digest
from async_capture.sensor_record_admission import verify_sealed_directory
from morai_pretraining.sensor_record_manifest import verify_episode_evidence
from ssd_storage import require_ssd


RECOVERY_NAME = "ADMISSION_RECOVERY_SCALE_V1.json"


def resolve_reference(value, parent):
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def reprocessing_plan(index_path, output):
    index_path = Path(index_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    seal = verify_sealed_directory(index_path.parent)
    if (seal.get("status") != "passed"
            or seal.get("files", {}).get(index_path.name) != digest(index_path)):
        raise ValueError("A passed sealed source publication is required")
    index = json.loads(index_path.read_text())
    if (index.get("training_ready") is not True
            or index.get("scenario_name") != "full_loop_static"
            or index.get("split_policy") != "static_collection_plan_run_holdout_v1"):
        raise ValueError("Requires the original static run-holdout publication")
    if output.exists():
        raise FileExistsError("New publication output already exists: " + str(output))
    report = resolve_reference(index["source_validation"], index_path.parent)
    split = resolve_reference(index["split_plan"], index_path.parent)
    if digest(report) != index["source_validation_sha256"]:
        raise ValueError("Original campaign validation changed")
    report_seal = verify_sealed_directory(report.parent)
    if (report_seal.get("status") != "passed"
            or report_seal.get("files", {}).get(report.name) != digest(report)):
        raise ValueError("Original campaign validation seal is invalid")
    if digest(split) != index["split_plan_sha256"]:
        raise ValueError("Original collection split plan changed")
    runs = []
    seen = set()
    for name in ("train", "val", "test"):
        if not index.get(name):
            raise ValueError("All original train/val/test splits are required")
        for row in index[name]:
            path = resolve_reference(row["path"], index_path.parent)
            if digest(path) != row["sha256"]:
                raise ValueError("Original episode manifest changed")
            doc = json.loads(path.read_text())
            if doc.get("split") != name:
                raise ValueError("Original split disagrees with its manifest")
            episode = resolve_reference(doc["episode"], path.parent)
            if episode in seen:
                raise ValueError("An episode appears more than once in source publication")
            seen.add(episode)
            verify_episode_evidence(
                episode, doc["evidence_seal_sha256"],
                doc.get("admission_recovery_path", "ADMISSION_RECOVERY.json"))
            runs.append({"lap": int(row["lap"]), "split": name,
                         "run": str(episode.parent)})
    return {"source_manifest_index": str(index_path),
            "source_manifest_index_sha256": digest(index_path),
            "validation_report": str(report), "split_plan": str(split),
            "split_plan_sha256": digest(split), "output": str(output),
            "admission_recovery_name": RECOVERY_NAME,
            "runs": sorted(runs, key=lambda row: row["lap"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-report", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = reprocessing_plan(args.manifest_index, args.output)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    require_ssd(Path(plan["validation_report"]), reserve_bytes=0)
    require_ssd(Path(plan["output"]).parent, reserve_bytes=0)
    for count, row in enumerate(plan["runs"], 1):
        print("[SCALE] %d/%d lap=%d split=%s" %
              (count, len(plan["runs"]), row["lap"], row["split"]), flush=True)
        command = [sys.executable, str(ROOT / "tools/recover_sensor_record_bev_geometry.py"),
                   "--run", row["run"], "--geometry-version", "3"]
        if args.catalog_report:
            command += ["--catalog-report", str(args.catalog_report.expanduser().resolve())]
        subprocess.check_call(command)
    # Check the original byte-level metadata contract once more before publishing.
    reprocessing_plan(args.manifest_index, args.output)
    subprocess.check_call([
        sys.executable, str(ROOT / "tools/build_sensor_record_training_dataset.py"),
        "--validation-report", plan["validation_report"],
        "--split-plan", plan["split_plan"],
        "--split-plan-sha256", plan["split_plan_sha256"],
        "--source-manifest-index", plan["source_manifest_index"],
        "--admission-recovery-name", RECOVERY_NAME,
        "--output", plan["output"],
    ])


if __name__ == "__main__":
    main()
