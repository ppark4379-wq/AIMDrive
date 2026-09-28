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
from morai_pretraining.sensor_record_manifest import coverage_failed_splits, verify_episode_evidence
from ssd_storage import require_ssd


RECOVERY_NAMES = {3: "ADMISSION_RECOVERY_SCALE_V1.json",
                  4: "ADMISSION_RECOVERY_CENTER_V1.json",
                  5: "ADMISSION_RECOVERY_FUSED_V1.json"}


def resolve_reference(value, parent):
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def reprocessing_plan(index_path, output, geometry_version=3, allow_coverage_failed=False,
                      geometry_registry=None):
    index_path = Path(index_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    seal = verify_sealed_directory(index_path.parent)
    index = json.loads(index_path.read_text())
    # A publication that failed only because a split lacked an object example
    # keeps sealed frames and splits; only an explicit request may revise it.
    coverage_failed = coverage_failed_splits(index) if allow_coverage_failed else []
    if (seal.get("files", {}).get(index_path.name) != digest(index_path)
            or not (seal.get("status") == "passed"
                    or (coverage_failed and seal.get("status") == "failed"))):
        raise ValueError("A passed sealed source publication is required")
    if not (index.get("training_ready") is True or coverage_failed):
        raise ValueError("Requires a training-ready or coverage-only-failed source publication")
    static_split = (index.get("scenario_name") == "full_loop_static"
                    and index.get("split_policy") == "static_collection_plan_run_holdout_v1")
    roundabout_split = (index.get("scenario_name") == "full_loop_roundabout"
                        and index.get("split_policy") == "complete_lap_holdout_v1"
                        and not index.get("test"))
    if not (static_split or roundabout_split):
        raise ValueError("Requires an approved static or roundabout run-holdout publication")
    if output.exists():
        raise FileExistsError("New publication output already exists: " + str(output))
    report = resolve_reference(index["source_validation"], index_path.parent)
    split = resolve_reference(index["split_plan"], index_path.parent) if static_split else None
    val_laps = sorted({int(number) for number in index.get("val_laps", [])}) if roundabout_split else []
    if roundabout_split and (not val_laps or val_laps != sorted(int(row["lap"]) for row in index.get("val", []))):
        raise ValueError("Roundabout validation runs disagree with the sealed source split")
    if digest(report) != index["source_validation_sha256"]:
        raise ValueError("Original campaign validation changed")
    report_seal = verify_sealed_directory(report.parent)
    if (report_seal.get("status") != "passed"
            or report_seal.get("files", {}).get(report.name) != digest(report)):
        raise ValueError("Original campaign validation seal is invalid")
    if static_split and digest(split) != index["split_plan_sha256"]:
        raise ValueError("Original collection split plan changed")
    runs = []
    seen = set()
    split_names = ("train", "val", "test") if static_split else ("train", "val")
    for name in split_names:
        if not index.get(name):
            raise ValueError("Every original run split must be nonempty")
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
            "validation_report": str(report),
            "split_plan": str(split) if split is not None else None,
            "split_plan_sha256": digest(split) if split is not None else None,
            "val_laps": val_laps, "output": str(output),
            "geometry_version": int(geometry_version),
            "source_coverage_failed_splits": coverage_failed,
            "admission_recovery_name": (
                "ADMISSION_RECOVERY_FUSED_%s.json" % digest(Path(geometry_registry).expanduser().resolve())[:12]
                if int(geometry_version) == 5 else RECOVERY_NAMES[int(geometry_version)]),
            "runs": sorted(runs, key=lambda row: row["lap"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-report", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--geometry-version", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--geometry-registry", type=Path, help="Required by --geometry-version 5")
    parser.add_argument("--allow-coverage-failed-source", action="store_true",
                        help="Revise a sealed publication that failed only required coverage")
    args = parser.parse_args()
    if args.geometry_version == 5 and args.geometry_registry is None:
        parser.error("--geometry-version 5 requires --geometry-registry")
    plan = reprocessing_plan(args.manifest_index, args.output, args.geometry_version,
                             args.allow_coverage_failed_source, args.geometry_registry)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    require_ssd(Path(plan["validation_report"]), reserve_bytes=0)
    require_ssd(Path(plan["output"]).parent, reserve_bytes=0)
    for count, row in enumerate(plan["runs"], 1):
        print("[SCALE] %d/%d lap=%d split=%s" %
              (count, len(plan["runs"]), row["lap"], row["split"]), flush=True)
        command = [sys.executable, str(ROOT / "tools/recover_sensor_record_bev_geometry.py"),
                   "--run", row["run"], "--geometry-version", str(plan["geometry_version"])]
        if args.catalog_report:
            command += ["--catalog-report", str(args.catalog_report.expanduser().resolve())]
        if args.geometry_registry:
            command += ["--geometry-registry", str(args.geometry_registry.expanduser().resolve())]
        subprocess.check_call(command)
    # Check the original byte-level metadata contract once more before publishing.
    reprocessing_plan(args.manifest_index, args.output, args.geometry_version,
                      args.allow_coverage_failed_source, args.geometry_registry)
    extra = ["--allow-coverage-failed-source"] if plan["source_coverage_failed_splits"] else []
    split_arguments = (["--split-plan", plan["split_plan"],
                        "--split-plan-sha256", plan["split_plan_sha256"]]
                       if plan["split_plan"] else
                       [part for lap in plan["val_laps"] for part in ("--val-lap", str(lap))])
    subprocess.check_call([
        sys.executable, str(ROOT / "tools/build_sensor_record_training_dataset.py"),
        "--validation-report", plan["validation_report"],
        *split_arguments,
        "--source-manifest-index", plan["source_manifest_index"],
        "--admission-recovery-name", plan["admission_recovery_name"],
        "--output", plan["output"],
    ] + extra)


if __name__ == "__main__":
    main()
