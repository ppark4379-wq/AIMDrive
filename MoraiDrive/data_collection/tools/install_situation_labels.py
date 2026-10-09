#!/usr/bin/env python3
"""Atomically install sealed, already filtered situation labels into native episodes.

Only situation_labels directories are replaced. Earlier labels remain in a
per-episode backup, and the dataset journal records every recoverable rename.
Preflight validates the entire cohort before creating any journal or staging.
Camera, LiDAR, raw, geometry, and quality records are never modified.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

SCHEMA = "situation_applied_labels_v1"
INSTALL_SCHEMA = "situation_native_label_install_v1"
FILES = ("labeled_frames.jsonl", "manifest.json", "COMPLETE.json")
MASKS = ("instruction_label_valid", "situation_target_valid", "handoff_target_valid", "instruction_target_valid")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value
            and not value.startswith("/") and all(part not in ("", ".", "..") for part in value.split("/")),
            "Unsafe relative path: " + repr(value))
    return PurePosixPath(value)


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def child(root, name):
    root = Path(root)
    path = root
    for part in relative(name).parts:
        path = path / part
        require(not path.is_symlink(), "Symlink path is not allowed: " + str(path))
    require(inside(path, root), "Path escapes root: " + str(path))
    return path


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink()
            raise
    os.replace(str(temporary), str(path))
    sync_directory(path.parent)


def sync_directory(path):
    # Directory fsync is unsupported by some mounted SSD filesystems.
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def verify_seal(folder):
    seal = read_json(child(folder, "COMPLETE.json"))
    files = seal.get("files")
    require(isinstance(files, dict) and bool(files), "Missing file seal: " + str(folder))
    for name, expected in files.items():
        require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                "Invalid file hash: " + str(name))
        path = child(folder, name)
        require(path.is_file() and sha256(path) == expected, "Sealed file changed: " + str(path))
    return seal


def file_pins(folder):
    return {name: sha256(child(folder, name)) for name in FILES}


def matches_installed(folder, pins):
    return folder.is_dir() and all(child(folder, name).is_file() for name in FILES) and file_pins(folder) == pins


def verify_installed(folder, pins):
    require(folder.is_dir() and not folder.is_symlink(), "Installed label folder missing/unsafe: " + str(folder))
    require(file_pins(folder) == pins, "Installed label file hashes differ: " + str(folder))
    verify_seal(folder)


def original_pins(folder):
    if not folder.exists():
        return None
    require(folder.is_dir() and not folder.is_symlink(), "Unsafe original label directory")
    require(child(folder, "labeled_frames.jsonl").is_file(), "Original native frame labels missing")
    verify_seal(folder)
    return {name: sha256(child(folder, name)) for name in ("labeled_frames.jsonl", "COMPLETE.json", "report.json", "manifest.json")
            if (folder / name).exists()}


def verify_original(folder, pins):
    require(pins is not None and folder.is_dir(), "Original label backup missing")
    for name, expected in pins.items():
        require(sha256(child(folder, name)) == expected, "Original labels changed: " + str(folder / name))
    verify_seal(folder)


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def assets_without_paths(assets):
    require(isinstance(assets, dict), "Malformed asset identity")
    return {role: ({key: value for key, value in ref.items() if key != "path"} if ref is not None else None)
            for role, ref in assets.items()}


def validate_rows(folder, record, dataset, old_folder):
    episode = child(dataset, record["native_episode_relative"])
    require(episode.is_dir() and child(episode, "episode.json").is_file()
            and child(episode, "raw").is_dir(), "Native episode metadata/raw missing: " + str(episode))
    rows = jsonl(folder / "labeled_frames.jsonl")
    require(bool(rows), "Empty applied label episode")
    originals = jsonl(old_folder / "labeled_frames.jsonl") if old_folder is not None else None
    original_by_frame = {}
    for original in originals or []:
        require(original.get("frame_id") not in original_by_frame, "Duplicate original native frame identity")
        original_by_frame[original.get("frame_id")] = original
    counts = Counter()
    for position, row in enumerate(rows):
        require(type(row.get("frame_id")) is int and type(row.get("sensor_time_ns")) is int
                and row["frame_id"] == row["labeling"].get("index"), "Frame/label index mismatch")
        if position:
            require(row["frame_id"] > rows[position - 1]["frame_id"]
                    and row["sensor_time_ns"] > rows[position - 1]["sensor_time_ns"], "Nonadvancing frame identity")
        require(row.get("episode_path") == str(episode) and row.get("episode_key") == record["episode_key"]
                and row.get("category") == record["category"], "Applied/native episode identity differs")
        portable = row.get("portable", {})
        native_root = str(PurePosixPath(record["native_episode_relative"]).parent)
        require(portable.get("native_episode_relative") == record["native_episode_relative"]
                and portable.get("native_root_relative") == native_root, "Native portable mapping differs")
        valid = row.get("instruction_label_valid")
        require(type(valid) is bool and all(type(row.get(name)) is bool and row[name] == valid for name in MASKS)
                and row["labeling"].get("action_valid") is valid, "Applied instruction masks differ")
        require(row["labeling"].get("instruction") == row.get("instruction") and
                ((isinstance(row.get("instruction"), str) and bool(row["instruction"].strip())) if valid
                 else row.get("instruction") is None), "Invalid filtered instruction")
        source = row.get("source_labeling")
        require(isinstance(source, dict), "Missing source class provenance")
        require({k: v for k, v in row["labeling"].items() if k not in ("action_valid", "instruction")} ==
                {k: v for k, v in source.items() if k not in ("action_valid", "instruction")},
                "Observation/source class fields changed")
        require(row.get("source_instruction") == source.get("instruction")
                and row.get("source_instruction_label_valid") == source.get("action_valid"), "Source instruction provenance differs")
        if row["frame_id"] in original_by_frame:
            old = original_by_frame[row["frame_id"]]
            require((old.get("frame_id"), old.get("sensor_time_ns")) == (row["frame_id"], row["sensor_time_ns"]),
                    "Native/source frame identity differs")
            require(assets_without_paths(old.get("assets")) == assets_without_paths(row.get("assets")),
                    "Native/source asset identity differs")
        mapped = {}
        for role, asset in row.get("assets", {}).items():
            if asset is None:
                continue
            name = str(PurePosixPath(native_root) / relative(asset["relative_path"]))
            path = child(dataset, name)
            require(asset.get("path") == str(path) and path.is_file() and path.stat().st_size == asset.get("size_bytes"),
                    "Native asset missing/size/path differs: " + str(path))
            mapped[role] = dict(dataset_relative_path=name, sha256=asset["sha256"], size_bytes=asset["size_bytes"])
        require(portable.get("assets") == mapped, "Portable asset identity differs")
        counts.update(rows=1, instruction_targets=int(valid), observation_valid=int(source.get("observation_valid") is True),
                      frame_valid=int(row.get("frame_valid") is True))
    for name, value in counts.items():
        require(record.get("counts", {}).get(name) == value, "Applied episode count differs: " + name)
    return counts


def load_package(labels_root, dataset_root):
    labels, dataset = (Path(p).expanduser().resolve() for p in (labels_root, dataset_root))
    require(labels.is_dir() and dataset.is_dir(), "Labels/dataset root missing")
    require(not inside(labels, dataset) and not inside(dataset, labels), "Labels package overlaps native dataset")
    seal = verify_seal(labels)
    require(seal.get("schema") == SCHEMA + "_complete" and seal.get("status") == "COMPLETE", "Incomplete applied package")
    index = read_json(child(labels, "applied_index.json"))
    require(index.get("schema") == SCHEMA + "_index" and index.get("status") == "COMPLETE", "Invalid applied index")
    require(index.get("input_identity") == seal.get("input_identity") and
            index["input_identity"].get("dataset_root") == str(dataset), "Package dataset/input identity differs")
    records = index.get("records")
    require(isinstance(records, list) and len(records) == index.get("episodes") and bool(records), "Invalid applied episode cohort")
    seen, plans = set(), []
    generation = sha256(labels / "COMPLETE.json")
    journal_root = child(dataset, ".situation_label_installs/" + generation)
    for record in records:
        key, native = record["episode_key"], record["native_episode_relative"]
        relative(key)
        require(native not in seen, "Duplicate native episode mapping")
        seen.add(native)
        category = relative(record["category"])
        require(len(category.parts) == 1, "Unsafe episode category")
        expected = "episodes/" + str(category) + "/" + key
        require(record["labeled_relative"] == expected + "/labeled_frames.jsonl"
                and record["manifest_relative"] == expected + "/manifest.json"
                and record["complete_relative"] == expected + "/COMPLETE.json", "Episode output mapping differs")
        source = child(labels, expected)
        episode_seal = verify_seal(source)
        require(episode_seal.get("schema") == SCHEMA + "_episode_complete" and episode_seal.get("status") == "COMPLETE"
                and set(episode_seal["files"]) == {"labeled_frames.jsonl", "manifest.json"}, "Invalid applied episode seal")
        pins = file_pins(source)
        for name, field in (("labeled_frames.jsonl", "labels_sha256"), ("manifest.json", "manifest_sha256"), ("COMPLETE.json", "complete_sha256")):
            require(seal["files"].get(expected + "/" + name) == pins[name], "Unpinned episode output file")
            if field in record:
                require(record[field] == pins[name], "Applied index hash differs: " + field)
        manifest = read_json(source / "manifest.json")
        require(manifest.get("schema") == SCHEMA and manifest.get("episode_key") == key
                and manifest.get("category") == record["category"] and manifest.get("counts") == record["counts"]
                and manifest.get("native_episode_relative") == native
                and manifest.get("source_labels_sha256") == record["source_labels_sha256"]
                and manifest.get("input_identity") == index["input_identity"], "Applied manifest/index differs")
        episode = child(dataset, native)
        slot = hashlib.sha256(native.encode("utf-8")).hexdigest()
        plans.append(dict(record=record, source=source, episode=episode, target=child(episode, "situation_labels"), pins=pins,
                          backup=child(episode, ".situation_label_backups/" + generation + "/previous"),
                          stage=child(episode, ".situation_label_stage_" + generation), journal=child(journal_root, slot + ".json")))
    return labels, dataset, index, generation, journal_root, plans


def state_for(plan, generation):
    if plan["journal"].exists():
        state = read_json(plan["journal"])
        require(state.get("schema") == INSTALL_SCHEMA and state.get("generation") == generation
                and state.get("native_episode_relative") == plan["record"]["native_episode_relative"]
                and state.get("installed_pins") == plan["pins"] and state.get("state") in ("prepared", "staged", "backed_up", "installed")
                and state.get("native_identity") == plan["native_identity"],
                "Resume episode journal differs")
        return state
    return None


def save_state(plan, state, value):
    state["state"] = value
    atomic_json(plan["journal"], state)


def check_stage(plan, state):
    stage = plan["stage"]
    require(stage.is_dir() and not stage.is_symlink(), "Unsafe interrupted staging directory")
    for path in stage.iterdir():
        require(path.name in FILES and path.is_file() and not path.is_symlink(), "Unexpected interrupted staging contents")
    if state["state"] != "prepared":
        verify_installed(stage, plan["pins"])


def recover(plan, state):
    target, backup, stage = (plan[name] for name in ("target", "backup", "stage"))
    if target.exists() and matches_installed(target, plan["pins"]):
        verify_installed(target, plan["pins"])
        if state["original_pins"] is not None:
            verify_original(backup, state["original_pins"])
        save_state(plan, state, "installed")
        return True
    if state["state"] == "installed":
        raise ValueError("Completed native labels changed")
    if backup.exists():
        verify_original(backup, state["original_pins"])
        require(not target.exists(), "Interrupted install has both old target and backup")
        # Restore first, including when interruption occurred before staging finished.
        os.rename(str(backup), str(target))
        sync_directory(target.parent)
        save_state(plan, state, "prepared")
    if state["original_pins"] is not None:
        verify_original(target, state["original_pins"])
    else:
        require(not target.exists(), "Unexpected native labels appeared")
    if stage.exists():
        check_stage(plan, state)
    return False


def install_episode(plan, generation, state):
    if state is None:
        require(not plan["backup"].exists() and not plan["stage"].exists(), "Unjournaled native backup/staging exists")
        state = dict(schema=INSTALL_SCHEMA, generation=generation, native_episode_relative=plan["record"]["native_episode_relative"],
                     installed_pins=plan["pins"], original_pins=plan["original_pins"], native_identity=plan["native_identity"], state="prepared",
                     backup_relative=plan["backup"].relative_to(plan["episode"]).as_posix())
        atomic_json(plan["journal"], state)
    elif recover(plan, state):
        return "reused"
    if state["state"] == "prepared":
        if not plan["stage"].exists():
            plan["stage"].mkdir()
        check_stage(plan, state)
        for name in FILES:
            shutil.copyfile(str(plan["source"] / name), str(plan["stage"] / name))
        verify_installed(plan["stage"], plan["pins"])
        for name in FILES:
            with (plan["stage"] / name).open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(plan["stage"])
        save_state(plan, state, "staged")
    require(plan["stage"].stat().st_dev == plan["episode"].stat().st_dev, "Native staging must share the episode filesystem")
    if state["original_pins"] is not None:
        verify_original(plan["target"], state["original_pins"])
        plan["backup"].parent.mkdir(parents=True, exist_ok=True)
        require(not plan["backup"].exists(), "Backup already exists")
        require(plan["backup"].parent.stat().st_dev == plan["target"].stat().st_dev == plan["episode"].stat().st_dev,
                "Native backup must share the episode filesystem")
        os.rename(str(plan["target"]), str(plan["backup"]))
        sync_directory(plan["episode"])
        sync_directory(plan["backup"].parent)
    else:
        require(not plan["target"].exists(), "Unexpected native labels appeared")
    save_state(plan, state, "backed_up")
    try:
        os.rename(str(plan["stage"]), str(plan["target"]))
        sync_directory(plan["episode"])
    except BaseException:
        if not plan["target"].exists() and plan["backup"].exists():
            os.rename(str(plan["backup"]), str(plan["target"]))
            sync_directory(plan["episode"])
            save_state(plan, state, "prepared")
        raise
    verify_installed(plan["target"], plan["pins"])
    save_state(plan, state, "installed")
    return "installed"


def install(labels_root, dataset_root, preflight_only=False, resume=False):
    labels, dataset, index, generation, journal_root, plans = load_package(labels_root, dataset_root)
    identity = dict(schema=INSTALL_SCHEMA, generation=generation, labels_root=str(labels), dataset_root=str(dataset),
                    applied_index_sha256=sha256(labels / "applied_index.json"))
    existing = journal_root.exists()
    if existing:
        if child(journal_root, "COMPLETE.json").exists():
            verify_seal(journal_root)
        require(resume or preflight_only, "Installation journal exists; use --resume")
        require(read_json(child(journal_root, "identity.json")) == identity, "Resume installation identity differs")
    states, counts = [], Counter()
    for plan in plans:
        plan["native_identity"] = {name: sha256(child(plan["episode"], name)) for name in
                                   ("episode.json", "raw/raw_manifest.json") if child(plan["episode"], name).is_file()}
        require("episode.json" in plan["native_identity"], "Native episode metadata missing")
        state = state_for(plan, generation) if existing else None
        states.append(state)
        old = plan["target"] if plan["target"].exists() else None
        if state is not None:
            if state["state"] == "installed" or (old is not None and matches_installed(old, plan["pins"])):
                verify_installed(plan["target"], plan["pins"])
                old = plan["backup"] if state["original_pins"] is not None else None
            elif plan["backup"].exists():
                require(not plan["target"].exists(), "Interrupted install has target and backup")
                old = plan["backup"]
            if state["original_pins"] is not None:
                verify_original(old, state["original_pins"])
            if plan["stage"].exists():
                check_stage(plan, state)
        elif old is not None:
            original_pins(old)
        plan["original_pins"] = state["original_pins"] if state is not None else original_pins(old) if old is not None else None
        counts.update(validate_rows(plan["source"], plan["record"], dataset, old))
    for name, value in counts.items():
        require(index.get("counts", {}).get(name) == value, "Applied global count differs: " + name)
    report = dict(schema=INSTALL_SCHEMA + "_report", status="PREFLIGHT_VERIFIED" if preflight_only else "COMPLETE",
                  generation=generation, dataset_root=str(dataset), labels_root=str(labels), episodes=len(plans), **dict(counts),
                  native_assets_modified=False, raw_geometry_quality_modified=False,
                  native_asset_verification="Sealed path/hash metadata, safe containment, file existence and size; sensor bytes are not rehashed",
                  backup_policy="Entire prior situation_labels directories preserved in per-episode .situation_label_backups",
                  backup_paths=[str(plan["backup"]) for plan in plans if plan["original_pins"] is not None],
                  journal_root=str(journal_root))
    if preflight_only:
        return report
    if not existing:
        journal_root.mkdir(parents=True)
        atomic_json(journal_root / "identity.json", identity)
    progress = Counter()
    for number, (plan, state) in enumerate(zip(plans, states), 1):
        progress[install_episode(plan, generation, state)] += 1
        if number % 25 == 0:
            print("[INSTALL] {}/{} episodes".format(number, len(plans)), flush=True)
    report["episode_results"] = dict(progress)
    atomic_json(journal_root / "install_report.json", report)
    atomic_json(journal_root / "COMPLETE.json", dict(schema=INSTALL_SCHEMA + "_complete", status="COMPLETE",
                generation=generation, files={name: sha256(journal_root / name) for name in
                    ["identity.json", "install_report.json"] + [plan["journal"].name for plan in plans]}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--preflight-only", action="store_true", help="Verify package and native records; write nothing")
    parser.add_argument("--resume", action="store_true", help="Recover/reuse only the same sealed generation")
    print(json.dumps(install(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
