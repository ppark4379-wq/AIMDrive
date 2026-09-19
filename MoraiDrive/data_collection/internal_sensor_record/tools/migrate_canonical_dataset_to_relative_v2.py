#!/usr/bin/env python3
"""Make legacy canonical campaigns self-contained and path-portable.

The command is a dry run unless ``--apply`` is supplied.  Sensor payloads are
already present in the canonical run directories and are never copied.  Only
small JSON/TSV publication records are atomically rewritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(target, parent):
    return Path(os.path.relpath(str(Path(target).resolve()),
                                str(Path(parent).resolve()))).as_posix()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_maps(campaign):
    by_source = {}
    by_lap = {}
    for run in campaign.iterdir():
        marker = run / ".LOCAL_DATASET_COMPLETE.json"
        if not run.is_dir() or not marker.is_file():
            continue
        row = json.loads(marker.read_text(encoding="utf-8"))
        source = row.get("source_run_name")
        if source:
            by_source[source] = run
        name = row.get("run_name", run.name)
        try:
            by_lap[int(name.rsplit("_lap_", 1)[1])] = run
        except (IndexError, ValueError):
            pass
    return by_source, by_lap


def migrate_campaign(campaign, apply):
    campaign = campaign.resolve()
    metadata = campaign / "_campaign"
    legacy_metadata = metadata / "metadata"
    by_source, by_lap = run_maps(campaign)
    if not by_lap:
        raise ValueError("campaign has no canonical runs: " + str(campaign))
    actions = []
    for name in ("campaign_plan.json", "runs.tsv", "COLLECTION_COMPLETE.json"):
        source, target = legacy_metadata / name, metadata / name
        if not target.exists():
            if not source.is_file():
                raise FileNotFoundError(source)
            actions.append("copy metadata/" + name)
            if apply:
                temporary = target.with_name(target.name + ".tmp.%d" % os.getpid())
                shutil.copyfile(str(source), str(temporary))
                os.replace(str(temporary), str(target))

    validations = sorted((metadata / "validations").glob("validation_*"))
    if not validations:
        raise ValueError("campaign has no validation: " + str(campaign))
    for validation in validations:
        report_path = validation / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["campaign"] = relative(metadata, validation)
        for group in (report.get("laps", []), report.get("excluded_laps", [])):
            for lap in group:
                if not lap.get("run_name"):
                    continue
                run = by_source.get(lap["run_name"]) or by_lap.get(int(lap["lap"]))
                if run is None:
                    raise ValueError("no canonical run for " + lap["run_name"])
                lap["run"] = relative(run, validation)
        actions.append("rewrite " + str(report_path.relative_to(campaign)))
        if apply:
            atomic_json(report_path, report)
            seal_path = validation / "COMPLETE.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal.setdefault("files", {})["report.json"] = sha256(report_path)
            atomic_json(seal_path, seal)
        validation_sha = sha256(report_path) if apply else None

        for training in sorted(validation.glob("training_dataset*")):
            index_path = training / "manifest_index.json"
            if not index_path.is_file():
                continue
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for split in ("train", "val"):
                for row in index.get(split, []):
                    manifest_path = training / Path(row["path"]).name
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    run = by_lap.get(int(row["lap"]))
                    if run is None:
                        raise ValueError("no canonical run for lap %s" % row["lap"])
                    manifest["episode"] = relative(run / "episode_000", training)
                    manifest["native_root"] = relative(run, training)
                    manifest["route"] = relative(run / "episode_000", training)
                    if apply:
                        manifest["campaign_validation_sha256"] = validation_sha
                        atomic_json(manifest_path, manifest)
                    row["path"] = manifest_path.name
                    if apply:
                        row["sha256"] = sha256(manifest_path)
            index["source_validation"] = relative(report_path, training)
            if apply:
                index["source_validation_sha256"] = validation_sha
                atomic_json(index_path, index)
                complete_path = training / "COMPLETE.json"
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                complete["source_validation_sha256"] = validation_sha
                complete["files"] = {
                    path.name: sha256(path)
                    for path in sorted(training.glob("*.json"))
                    if path.name != "COMPLETE.json"
                }
                atomic_json(complete_path, complete)
            actions.append("portable training publication " + training.name)

    for run in by_lap.values():
        marker_path = run / ".LOCAL_DATASET_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.update(schema="morai_local_dataset_materialization_v2",
                      hardlinked=False, self_contained=True,
                      relative_paths=True)
        marker.pop("validation_report", None)
        if apply:
            atomic_json(marker_path, marker)
    migration = {
        "schema": "morai_canonical_dataset_relative_migration_v1",
        "status": "completed" if apply else "planned",
        "campaign": campaign.name,
        "runs": len(by_lap),
        "sensor_payloads_copied": False,
        "relative_paths": True,
        "actions": actions,
    }
    if apply:
        atomic_json(metadata / "RELATIVE_LAYOUT_COMPLETE.json", migration)
    return migration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    campaigns = sorted(path for path in root.glob("*/*")
                       if (path / "_campaign").is_dir())
    results = [migrate_campaign(path, args.apply) for path in campaigns]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
