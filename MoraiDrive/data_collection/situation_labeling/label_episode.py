"""Run Sensor Record preprocessing and instruction labeling with one command.

Run in the collector's WSL/ROS analysis environment. Sensor bytes are referenced,
not copied. Capture-time raw/quality records must already be complete.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


INTERNAL = Path(__file__).resolve().parents[1] / "internal_sensor_record"
REPOSITORY = Path(__file__).resolve().parents[3]
GEOMETRY = "object_geometry_audit_v2"
REVIEW = "situation_review_v2"
ADMISSION = "situation_admission_v2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_stage(folder, seal_name="COMPLETE.json"):
    seal = read_json(folder / seal_name)
    files = seal.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty or invalid stage seal: " + str(folder))
    for name, expected in files.items():
        path = folder / name
        if (Path(name).name != name or path.is_symlink()
                or digest(path) != expected):
            raise ValueError("Changed or unsafe stage file: " + str(path))
    if "report.json" in files and read_json(folder / "report.json").get("structural_valid") is False:
        raise ValueError("Preprocessing reported invalid structure: " + str(folder))
    return seal


def resolve_episode(source):
    source = Path(source).expanduser().resolve()
    if (source / "raw").is_dir():
        episode = source
    else:
        candidates = sorted(p for p in source.glob("episode_*")
                            if (p / "raw").is_dir())
        if len(candidates) != 1:
            raise ValueError("Specify one episode directory or a run containing exactly one episode")
        episode = candidates[0]
    run = episode.parent
    for path in (episode / "raw/RAW_COMPLETE.json",
                 episode / "quality/QUALITY_COMPLETE.json",
                 run / "capture_contract.json", run / "sensor_contract.json",
                 run / "plan.json", episode / "episode.json"):
        if not path.is_file():
            raise ValueError("Required collector record missing: " + str(path))
    if read_json(episode / "raw/RAW_COMPLETE.json").get("status") != "completed":
        raise ValueError("Raw capture is not completed")
    verify_stage(episode / "quality", "QUALITY_COMPLETE.json")
    return episode, run


def preprocessing_stages(episode, run, catalog=None):
    """Ordered existing tools; V2 review/admission never overwrite V1 outputs."""
    stages = []

    def add(tool, folder, arguments):
        stages.append((tool, episode / folder, list(map(str, arguments))))

    add("audit_sensor_record_probe.py", "native_audit",
        ["--run", run, "--raw", episode / "raw", "--output", episode / "native_audit"])
    add("build_sensor_record_associations.py", "native_associations", ["--episode", episode])
    for tool, folder in (
        ("build_sensor_record_gt_timeline.py", "gt_timeline_v1"),
        ("audit_sensor_record_direct_traffic.py", "traffic_audit_v1"),
        ("audit_sensor_record_actor_inventory.py", "actor_inventory_audit_v1"),
        ("audit_sensor_record_object_geometry_v2.py", GEOMETRY),
        ("audit_sensor_record_route_signals.py", "route_signal_audit_v1"),
        ("audit_sensor_record_motion_time.py", "motion_time_v1"),
    ):
        args = ["--episode", episode, "--output", episode / folder]
        if folder == GEOMETRY and catalog is not None:
            args += ["--catalog-report", catalog]
        add(tool, folder, args)
    add("audit_sensor_record_exposure.py", "exposure_v1",
        ["--audit", episode / "native_audit", "--quality", episode / "quality",
         "--output", episode / "exposure_v1"])
    add("summarize_sensor_record_review.py", REVIEW,
        ["--episode", episode, "--object-geometry-dir", GEOMETRY,
         "--output", episode / REVIEW])
    add("admit_sensor_record_collection.py", ADMISSION,
        ["--episode", episode, "--capture-contract", run / "capture_contract.json",
         "--run-plan", run / "plan.json", "--review-dir", REVIEW,
         "--object-geometry-dir", GEOMETRY, "--output", episode / ADMISSION])
    return stages


def run_stages(stages, execute, verify=verify_stage):
    results = []
    for tool, folder, arguments in stages:
        if folder.exists():
            # Never hide an interrupted/changed audit by overwriting it.
            verify(folder)
            state = "reused"
            print("[REUSE] " + folder.name, flush=True)
        else:
            print("[PREPROCESS] " + tool, flush=True)
            execute([str(INTERNAL / "tools" / tool), *arguments])
            verify(folder)
            state = "created"
        results.append({"tool": tool, "output": str(folder), "status": state,
                        "seal_sha256": digest(folder / "COMPLETE.json")})
    return results


def indexed(rows, name):
    result = {}
    for row in rows:
        index = row["index"]
        if index in result:
            raise ValueError("Duplicate frame index in " + name)
        result[index] = row
    return result


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_admission_inputs(episode, run):
    """A valid seal alone does not prove a reused admission belongs to this run."""
    report = read_json(episode / ADMISSION / "report.json")
    dependencies = {
        "native_audit": "native_audit", "quality": "quality",
        "native_associations": "native_associations", "motion_time": "motion_time_v1",
        "review": REVIEW, "gt_timeline": "gt_timeline_v1",
        "traffic_audit": "traffic_audit_v1", "actor_inventory_audit": "actor_inventory_audit_v1",
        "object_geometry_audit": GEOMETRY, "route_signal_audit": "route_signal_audit_v1",
    }
    for key, folder in dependencies.items():
        seal = "QUALITY_COMPLETE.json" if key == "quality" else "COMPLETE.json"
        if report.get("input_seals", {}).get(key) != digest(episode / folder / seal):
            raise ValueError("Admission has different input evidence: " + key)
    for key, path in (("capture_contract_sha256", run / "capture_contract.json"),
                      ("run_plan_sha256", run / "plan.json"),
                      ("raw_manifest_sha256", episode / "raw/raw_manifest.json")):
        if report.get(key) != digest(path):
            raise ValueError("Admission has different capture identity: " + key)


def export_labeled_frames(episode, native_root, features, destination):
    """Join all frames, including unlabeled ones, to their original assets."""
    frames = indexed(read_jsonl(features / "frame_features.jsonl"), "features")
    rules = indexed(read_jsonl(features / "situation_rules.jsonl"), "rules")
    admissions = indexed(read_json(episode / ADMISSION / "frames.json"), "admission")
    if not frames or frames.keys() != rules.keys() or frames.keys() != admissions.keys():
        raise ValueError("Feature/rule/admission frame indices differ or are empty")
    counts = Counter()
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for index in sorted(frames):
            frame, rule, admission = frames[index], rules[index], admissions[index]
            stamp = frame["lidar_source_ns"]
            if stamp != admission["lidar_source_time_ns"]:
                raise ValueError("Frame timestamp differs at index " + str(index))
            assets = {}
            for role, meta in admission.get("assets", {}).items():
                relative = Path(meta["relative_path"])
                path = (native_root / relative).resolve()
                if (relative.is_absolute() or ".." in relative.parts
                        or native_root.resolve() not in path.parents or not path.is_file()):
                    raise ValueError("Missing or unsafe native asset: " + str(path))
                assets[role] = {**meta, "path": str(path)}
            instruction = rule.get("instruction")
            counts["labeled" if instruction else "unlabeled"] += 1
            row = {"episode_path": str(episode), "frame_id": index,
                   "sensor_time_ns": stamp, "assets": assets,
                   "zone": frame.get("zone_id"), "instruction": instruction,
                   "labeling": rule,
                   "dataset_build_candidate": frame.get("dataset_build_candidate"),
                   "object_training_label_valid": frame.get("object_training_label_valid"),
                   "frame_valid": frame.get("valid"),
                   "frame_invalid_reasons": frame.get("invalid_reasons", [])}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path,
                        help="collector episode directory, or its parent run")
    parser.add_argument("--output", type=Path, help="new result directory; default: EPISODE/situation_labels")
    parser.add_argument("--analysis-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--catalog-report", type=Path,
                        help="geometry catalog used when creating the V2 geometry audit")
    parser.add_argument("--static-supervisor-log", type=Path,
                        help="defaults to this run's supervisor.log, if present")
    args = parser.parse_args(argv)
    if args.catalog_report is not None:
        args.catalog_report = args.catalog_report.expanduser().resolve()
    if args.static_supervisor_log is not None:
        args.static_supervisor_log = args.static_supervisor_log.expanduser().resolve()
    episode, run = resolve_episode(args.episode)
    output = (args.output or episode / "situation_labels").expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    # Existing collector modules resolve recorded route/native identities and
    # supply ROS serialization dependencies; do not substitute today's route.
    sys.path.insert(0, str(INTERNAL))
    from python_environment import build_analysis_environment, probe_analysis_environment
    from async_capture.route_sources import resolve_route_sources
    from async_capture.native_session import recorded_native_root
    from ssd_storage import SSD_ROOT

    python = args.analysis_python.expanduser().resolve()
    env = build_analysis_environment(python)
    env["PYTHONPATH"] = str(REPOSITORY) + os.pathsep + env["PYTHONPATH"]
    probe_analysis_environment(python, env)
    # Also check dependencies used specifically by labeling before preprocessing.
    subprocess.run([str(python), "-c", "import yaml"], env=env, check=True)
    sources = resolve_route_sources(episode, read_json(run / "sensor_contract.json"))
    native_root = recorded_native_root(run, SSD_ROOT).resolve()
    static_log = args.static_supervisor_log or run / "supervisor.log"
    if args.static_supervisor_log is not None and not static_log.is_file():
        raise FileNotFoundError(static_log)
    if not static_log.is_file():
        static_log = None
        print("[INFO] No supervisor.log; static instructions will use rule-only evidence", flush=True)
    if args.catalog_report is not None and not args.catalog_report.is_file():
        raise FileNotFoundError(args.catalog_report)
    # An explicit new catalog must not be silently ignored by reusing old geometry.
    if args.catalog_report is not None and (episode / GEOMETRY).exists():
        raise ValueError("Geometry already exists; omit --catalog-report to reuse its sealed evidence")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "morai_labeled_episode_v1", "status": "running",
              "episode": str(episode), "native_root": str(native_root),
              "route_provenance": sources["provenance"],
              "static_supervisor_log": str(static_log) if static_log else None,
              "sensor_files_copied": False}
    write_json(output / "report.json", report)
    try:
        with (output / "pipeline.log").open("w", encoding="utf-8") as log:
            def execute(arguments):
                # Stage names appear on stdout; pipeline.log holds diagnostics.
                command = [str(python), *map(str, arguments)]
                log.write(json.dumps(command, ensure_ascii=False) + "\n")
                log.flush()
                subprocess.run(command, env=env, cwd=REPOSITORY,
                               stdout=log, stderr=subprocess.STDOUT, check=True)

            report["preprocessing"] = run_stages(
                preprocessing_stages(episode, run, args.catalog_report), execute)
            verify_admission_inputs(episode, run)
            features = output / "features"
            print("[LABEL] Generating frame instructions", flush=True)
            command = ["-m", "MoraiDrive.data_collection.situation_labeling.build_labels",
                       "--episode", episode, "--route-definition", sources["route_definition"],
                       "--global-path", sources["global_path"], "--route-key", sources["route_key"],
                       "--object-geometry-dir", GEOMETRY, "--admission-dir", ADMISSION,
                       "--output", features]
            if static_log is not None:
                command += ["--static-supervisor-log", static_log]
            execute(command)
        destination = output / "labeled_frames.jsonl"
        report["frame_counts"] = export_labeled_frames(episode, native_root, features, destination)
        report["labeled_frames"] = str(destination)
        report["instruction_report"] = read_json(features / "report.json")
        report["status"] = "completed"
        write_json(output / "report.json", report)
        write_json(output / "COMPLETE.json", {"files": {
            name: digest(output / name) for name in ("labeled_frames.jsonl", "report.json")}})
    except Exception as error:
        report.update(status="failed", error=str(error))
        write_json(output / "report.json", report)
        raise
    print(json.dumps({"status": "completed", "output": str(output),
                      "labeled_frames": str(destination),
                      "frame_counts": report["frame_counts"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
