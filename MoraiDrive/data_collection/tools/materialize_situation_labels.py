#!/usr/bin/env python3
"""Write standalone, filter-applied situation JSONL labels (stdlib only).

All source frames and full observation/evidence fields are retained. Excluded
instructions become null and every instruction mask agrees with the final mask.
Native data and the sealed source packages are read-only. Run --preflight-only
to verify every source label and exclusion before creating an output package.
"""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile


SCHEMA = "situation_applied_labels_v1"
TOP_FIELDS = ("episode_path", "frame_id", "sensor_time_ns", "assets", "zone",
              "zone_instruction", "frame_valid", "frame_invalid_reasons",
              "dataset_build_candidate", "object_training_label_valid",
              "traffic_training_label_valid", "instruction", "instruction_label_valid")
LABEL_FIELDS = ("observation_valid", "action_valid", "instruction", "label",
                "label_candidate", "instruction_label", "instruction_id", "instruction_state",
                "instruction_variables", "instruction_consistency", "instruction_quality", "unknown_reasons")
MASK_FIELDS = ("instruction_label_valid", "situation_target_valid", "handoff_target_valid",
               "instruction_target_valid")
PROFILE_METADATA = ("profile.json", "review_exclusions.json", "verification.json", "tools/loader.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
            "Unsafe/non-POSIX relative path: " + repr(value))
    require(not value.startswith("/") and all(p not in ("", ".", "..") for p in value.split("/")),
            "Unsafe relative path: " + value)
    return PurePosixPath(value)


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def join(root, relative):
    """Package paths must remain inside the package, including symlinks."""
    parts = safe_relative(relative).parts
    path = Path(root).joinpath(*parts)
    require(not path.is_symlink(), "Symlink package file: " + str(path))
    require(within(path.resolve(), Path(root).resolve()), "Package path escapes root: " + str(path))
    return path


def historical_relative(value, root):
    require(isinstance(value, str) and value.startswith("/") and "\\" not in value,
            "Unsafe historical native path")
    path = PurePosixPath(value)
    require(".." not in path.parts, "Unsafe historical native path")
    relative = path.relative_to(PurePosixPath(root)).as_posix()
    safe_relative(relative)
    return relative


def verify_seal(root, cache=None, selected=None):
    seal = read_json(join(root, "COMPLETE.json"))
    files = seal.get("files")
    require(isinstance(files, dict) and files, "Missing file seal: " + str(root))
    for name, expected in files.items():
        safe_relative(name)
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in "0123456789abcdef" for c in expected), "Malformed SHA256: " + name)
    names = files if selected is None else selected
    for name in names:
        require(name in files, "Unsealed required file: " + name)
        path = join(root, name)
        key = str(path.resolve())
        actual = cache.get(key) if cache is not None else None
        if actual is None:
            actual = sha256(path)
            if cache is not None:
                cache[key] = actual
        require(actual == files[name], "Sealed file changed: " + str(path))
    return seal


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink()
            raise
    os.replace(str(temporary), str(path))


def snapshot(source, destination):
    """An interrupted provenance copy must never look like a reusable file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256(source)
    if destination.exists():
        require(sha256(destination) == expected, "Existing provenance snapshot differs")
        return expected
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="." + destination.name, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        require(sha256(temporary) == expected, "Provenance changed during copy")
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return expected


def exclusions_by_identity(payload):
    entries = payload.get("exclusions")
    require(isinstance(entries, list), "Malformed exclusion list")
    result = {}
    for entry in entries:
        safe_relative(entry.get("episode_key"))
        require(type(entry.get("frame_id")) is int and entry["frame_id"] >= 0, "Invalid exclusion frame_id")
        identity = (entry["episode_key"], entry["frame_id"])
        require(identity not in result, "Duplicate exclusion: " + repr(identity))
        result[identity] = entry
    return result


def source_signature(row, key):
    signature = dict(episode_key=key, frame_id=row["frame_id"], sensor_time_ns=row["sensor_time_ns"],
                     instruction=row["instruction"])
    encoded = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_inputs(base_package, visual_profile, dataset_root):
    base, profile_root, dataset = (Path(p).expanduser().resolve() for p in
                                   (base_package, visual_profile, dataset_root))
    require(base.is_dir() and profile_root.is_dir() and dataset.is_dir(), "Missing input directory")
    cache = {}
    profile_seal = verify_seal(profile_root, cache, PROFILE_METADATA)
    require(profile_seal.get("schema") == "situation_visual_instruction_profile_complete_v1"
            and profile_seal.get("status") == "COMPLETE", "Incomplete/wrong visual profile")
    profile = read_json(profile_root / "profile.json")
    pins = profile.get("base_package_pins")
    require(isinstance(pins, dict) and {"COMPLETE.json", "portable_index.json", "portable_frames.jsonl.gz"} <= set(pins),
            "Missing base package pins")
    for name, expected in pins.items():
        path = join(base, name)
        actual = sha256(path)
        require(actual == expected, "Profile base pin mismatch: " + name)
        cache[str(path.resolve())] = actual
    base_seal = verify_seal(base, cache)
    require(base_seal.get("schema") == "situation_ssd_review_candidate_v1_complete"
            and base_seal.get("status") == "REVIEW_CANDIDATE_MECHANICAL_EXPORT", "Wrong base package contract")
    index = read_json(base / "portable_index.json")
    require(index.get("source_index_sha256") == base_seal.get("source_index_sha256")
            and index.get("source_complete_sha256") == base_seal.get("source_complete_sha256"), "Base source pins disagree")
    for name, key in (("source_bundle/version_index.json", "source_index_sha256"),
                      ("source_bundle/COMPLETE.json", "source_complete_sha256")):
        require(base_seal["files"].get(name) == index[key], "Copied source pin mismatch")
    review_spec = read_json(base / "review_exclusions.json")
    visual_spec = read_json(profile_root / "review_exclusions.json")
    require(review_spec.get("source_index_sha256") == visual_spec.get("source_index_sha256") == index["source_index_sha256"],
            "Exclusion source index mismatch")
    require(visual_spec.get("base_portable_index_sha256") == pins["portable_index.json"], "Visual index pin mismatch")
    require(index.get("review_exclusions_sha256") == base_seal["files"].get("review_exclusions.json"), "Review exclusion pin mismatch")
    reviews, visuals = exclusions_by_identity(review_spec), exclusions_by_identity(visual_spec)
    require(len(reviews) == index.get("review_exclusions_source_count"), "Review exclusion count mismatch")
    require(len(visuals) == profile.get("additional_instruction_exclusions"), "Visual exclusion count mismatch")
    require(not set(reviews) & set(visuals), "Visual profile re-excludes a base exclusion")
    records = index.get("records")
    require(isinstance(records, list) and records, "Empty episode index")
    require(len(records) == index.get("selected_episode_count") == profile.get("source_episodes"), "Episode count mismatch")
    by_key = {}
    for record in records:
        key = record["episode_key"]
        safe_relative(key)
        safe_relative(record["category"])
        require(len(safe_relative(record["category"]).parts) == 1, "Invalid episode category")
        require(key not in by_key, "Duplicate episode index: " + key)
        by_key[key] = record
        label_path = join(base, record["labeled_relative"])
        require(label_path.name == "labeled_frames.jsonl", "Unexpected label filename")
        seal = verify_seal(label_path.parent, cache)
        require(sha256(label_path.parent / "COMPLETE.json") == record["complete_sha256"], "Episode seal pin mismatch")
        require(seal["files"].get("labeled_frames.jsonl") == record["labels_sha256"]
                and seal["files"].get("report.json") == record["report_sha256"], "Episode source pins mismatch")
        require(seal.get("labeling_code_sha256") == record["labeling_code_sha256"], "Label generation pin mismatch")
        feature_seal_path = join(base, record["features_relative"] + "/COMPLETE.json")
        require(sha256(feature_seal_path) == record["features_complete_sha256"] == seal.get("features_seal_sha256"),
                "Episode feature seal pin mismatch")
        verify_seal(feature_seal_path.parent, cache)
        for name in ("native_episode_relative", "native_root_relative"):
            safe_relative(record[name])
        require(record["source_record"]["labels_sha256"] == record["labels_sha256"], "Source record label identity mismatch")
    identity = dict(schema=SCHEMA, base_complete_sha256=pins["COMPLETE.json"],
                    base_index_sha256=pins["portable_index.json"], base_frames_sha256=pins["portable_frames.jsonl.gz"],
                    profile_complete_sha256=sha256(profile_root / "COMPLETE.json"),
                    profile_json_sha256=sha256(profile_root / "profile.json"),
                    visual_exclusions_sha256=sha256(profile_root / "review_exclusions.json"),
                    review_exclusions_sha256=index["review_exclusions_sha256"],
                    materializer_sha256=sha256(__file__), dataset_root=str(dataset))
    return dict(base=base, profile_root=profile_root, dataset=dataset, profile=profile, index=index,
                records=records, by_key=by_key, reviews=reviews, visuals=visuals, identity=identity)


def compact_projection(full, record):
    value = {key: full.get(key) for key in TOP_FIELDS}
    labeling = full["labeling"]
    value.update(episode_key=record["episode_key"], category=record["category"],
                 labeling_code_sha256=record["labeling_code_sha256"],
                 situation_target_valid=labeling["action_valid"] and labeling["observation_valid"] and full.get("frame_valid") is True)
    value["labeling"] = {key: labeling.get(key) for key in LABEL_FIELDS}
    return value


def check_source_row(frame, full, record, inputs):
    key = record["episode_key"]
    identity = (key, full["frame_id"])
    require(type(full["frame_id"]) is int and full["frame_id"] >= 0 and type(full["sensor_time_ns"]) is int, "Invalid source frame identity")
    require(frame.get("schema") == "situation_ssd_review_candidate_v1_frame", "Wrong portable frame schema")
    require(frame.get("episode_key") == key and frame.get("category") == record["category"]
            and frame.get("source_label_relative") == record["labeled_relative"]
            and frame.get("labeling_code_sha256") == record["labeling_code_sha256"], "Portable/source record mismatch")
    compact = compact_projection(full, record)
    require(frame.get("original_row") == compact, "Full source/portable row mismatch: " + repr(identity))
    labeling = compact["labeling"]
    target = compact["situation_target_valid"]
    require(all(type(v) is bool for v in (target, compact["instruction_label_valid"], labeling["action_valid"],
                                           labeling["observation_valid"], compact["frame_valid"], frame.get("handoff_target_valid"))), "Malformed source mask")
    require(target == compact["instruction_label_valid"] == labeling["action_valid"], "Source masks disagree")
    require(full.get("instruction") == labeling.get("instruction"), "Source nested instruction disagreement")
    require((isinstance(full["instruction"], str) and bool(full["instruction"].strip())) if target
            else full["instruction"] is None, "Invalid source instruction")
    paths = frame["portable"]
    source_root = inputs["index"]["source_dataset_root_historical"]
    require(paths["native_episode_relative"] == record["native_episode_relative"] == historical_relative(full["episode_path"], source_root),
            "Native episode mapping changed")
    require(paths["native_root_relative"] == record["native_root_relative"] == str(PurePosixPath(record["native_episode_relative"]).parent),
            "Native parent mapping changed")
    expected_assets = {}
    for role, asset in full["assets"].items():
        safe_relative(role)
        if asset is None:
            continue
        relative = safe_relative(asset["relative_path"])
        expected = str(PurePosixPath(record["native_root_relative"]) / relative)
        require(historical_relative(asset["path"], source_root) == expected, "Source asset mapping changed")
        expected_assets[role] = dict(dataset_relative_path=expected, sha256=asset["sha256"], size_bytes=asset["size_bytes"])
    require(paths["assets"] == expected_assets, "Portable/source asset mapping mismatch")
    review = inputs["reviews"].get(identity)
    require(frame["handoff_target_valid"] == (target and review is None), "Base handoff mask disagrees")
    require(frame["review_exclusion_reasons"] == ([review["reason"]] if review else []), "Base review reason disagrees")
    if review:
        require(target, "Review excludes an invalid source target")
        for entry_key, source_key in (("sensor_time_ns", "sensor_time_ns"), ("original_instruction", "instruction")):
            if entry_key in review:
                require(review[entry_key] == full[source_key], "Review source identity changed")
    visual = inputs["visuals"].get(identity)
    if visual:
        require(frame["handoff_target_valid"], "Visual exclusion is not a base target")
        require(visual["source_signature_sha256"] == source_signature(full, key), "Visual source signature mismatch")
        require(visual["source_label_file_sha256"] == record["labels_sha256"]
                and visual["category"] == record["category"], "Visual source label pin mismatch")
    return target, review, visual


def apply_row(full, frame, record, inputs):
    target, review, visual = check_source_row(frame, full, record, inputs)
    valid = target and review is None and visual is None
    result = copy.deepcopy(full)
    result["episode_key"], result["category"] = record["episode_key"], record["category"]
    result["episode_path"] = str(inputs["dataset"].joinpath(*safe_relative(record["native_episode_relative"]).parts))
    result["portable"] = copy.deepcopy(frame["portable"])
    for role, asset in frame["portable"]["assets"].items():
        result["assets"][role]["path"] = str(inputs["dataset"].joinpath(*safe_relative(asset["dataset_relative_path"]).parts))
    result["source_instruction"] = full["instruction"]
    result["source_instruction_label_valid"] = full["instruction_label_valid"]
    result["source_situation_target_valid"] = target
    result["source_handoff_target_valid"] = frame["handoff_target_valid"]
    result["source_labeling"] = copy.deepcopy(full["labeling"])
    result["instruction"] = full["instruction"] if valid else None
    for name in MASK_FIELDS:
        result[name] = valid
    result["labeling"]["instruction"] = result["instruction"]
    result["labeling"]["action_valid"] = valid
    result["review_exclusion_reasons"] = frame["review_exclusion_reasons"]
    result["visual_profile_exclusion"] = visual
    result["instruction_filter_provenance"] = dict(schema=SCHEMA, source_label_relative=record["labeled_relative"],
                                                   source_labels_sha256=record["labels_sha256"],
                                                   source_frame_signature_sha256=source_signature(full, record["episode_key"]),
                                                   review_exclusion=review, visual_exclusion=visual)
    return result, dict(rows=1, source_target_valid=int(target), source_handoff_target_valid=int(frame["handoff_target_valid"]),
                        instruction_targets=int(valid), excluded=int(not valid), review_excluded=int(review is not None),
                        visual_excluded=int(visual is not None), observation_valid=int(full["labeling"]["observation_valid"]),
                        frame_valid=int(full["frame_valid"]))


def episode_rows(inputs):
    """Yield one complete episode at a time, preserving the exact source order."""
    seen_episodes, seen_reviews, seen_visuals = set(), set(), set()
    with gzip.open(inputs["base"] / "portable_frames.jsonl.gz", "rt", encoding="utf-8") as portable:
        pending = None
        while True:
            line = pending if pending is not None else next(portable, None)
            pending = None
            if line is None:
                break
            first = json.loads(line)
            key = first["episode_key"]
            require(key in inputs["by_key"] and key not in seen_episodes, "Unknown/interleaved/duplicate episode: " + key)
            seen_episodes.add(key)
            record, rows, counts, frame_ids = inputs["by_key"][key], [], Counter(), set()
            with join(inputs["base"], record["labeled_relative"]).open(encoding="utf-8") as full_stream:
                frame = first
                while True:
                    full_line = next(full_stream, None)
                    require(full_line is not None, "Portable episode has extra frames: " + key)
                    full = json.loads(full_line)
                    require(full["frame_id"] not in frame_ids, "Duplicate source frame_id: " + key)
                    frame_ids.add(full["frame_id"])
                    row, delta = apply_row(full, frame, record, inputs)
                    rows.append(row)
                    counts.update(delta)
                    identity = (key, full["frame_id"])
                    if identity in inputs["reviews"]:
                        seen_reviews.add(identity)
                    if identity in inputs["visuals"]:
                        seen_visuals.add(identity)
                    line = next(portable, None)
                    if line is None:
                        break
                    following = json.loads(line)
                    if following["episode_key"] != key:
                        pending = line
                        break
                    frame = following
                require(next(full_stream, None) is None, "Portable episode omits source frames: " + key)
            expected = record["counts"]
            require(all(counts[actual] == expected[original] for actual, original in
                        (("rows", "rows"), ("source_target_valid", "target_valid"),
                         ("source_handoff_target_valid", "handoff_target_valid"), ("review_excluded", "review_excluded"),
                         ("observation_valid", "observation_valid"), ("frame_valid", "frame_valid"))), "Episode source counts changed: " + key)
            yield record, rows, dict(counts)
    require(seen_episodes == set(inputs["by_key"]), "Missing source episodes")
    require(seen_reviews == set(inputs["reviews"]) and seen_visuals == set(inputs["visuals"]), "Exclusion refers to a missing source frame")


def verify_counts(inputs, counts, episodes):
    profile = inputs["profile"]
    require(episodes == profile["source_episodes"] and counts["rows"] == profile["source_frames"], "Global source cohort changed")
    require(counts["source_handoff_target_valid"] == profile["source_instruction_targets"], "Base target count changed")
    require(counts["instruction_targets"] == profile["expected_instruction_targets"]
            and counts["visual_excluded"] == profile["additional_instruction_exclusions"], "Final instruction target count changed")


def write_episode(output, record, rows, counts, identity, resume):
    relative = "episodes/" + record["category"] + "/" + record["episode_key"]
    folder = join(output, relative)
    manifest = dict(schema=SCHEMA, episode_key=record["episode_key"], category=record["category"], counts=counts,
                    source_labels_sha256=record["labels_sha256"], input_identity=identity,
                    native_episode_relative=record["native_episode_relative"])
    if folder.exists():
        require(resume, "Existing episode output; use --resume: " + str(folder))
        verify_seal(folder)
        require(read_json(folder / "manifest.json") == manifest, "Resume episode identity differs")
    else:
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = output / ".staging"
        staging.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="episode_", dir=staging))
        try:
            with (temporary / "labeled_frames.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(temporary / "manifest.json", manifest)
            atomic_json(temporary / "COMPLETE.json", dict(schema=SCHEMA + "_episode_complete", status="COMPLETE",
                        files={name: sha256(temporary / name) for name in ("labeled_frames.jsonl", "manifest.json")}))
            require(not folder.exists(), "Concurrent episode output exists")
            os.rename(str(temporary), str(folder))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return dict(episode_key=record["episode_key"], category=record["category"],
                labeled_relative=relative + "/labeled_frames.jsonl", manifest_relative=relative + "/manifest.json",
                complete_relative=relative + "/COMPLETE.json", labels_sha256=sha256(folder / "labeled_frames.jsonl"),
                manifest_sha256=sha256(folder / "manifest.json"),
                complete_sha256=sha256(folder / "COMPLETE.json"), counts=counts,
                native_episode_relative=record["native_episode_relative"], source_labels_sha256=record["labels_sha256"])


def materialize(base_package, visual_profile, dataset_root, output, preflight_only=False, resume=False):
    inputs = load_inputs(base_package, visual_profile, dataset_root)
    output = Path(output).expanduser().resolve()
    for source in (inputs["base"], inputs["profile_root"], inputs["dataset"]):
        require(not within(output, source) and not within(source, output), "Output overlaps a source directory")
    if not preflight_only:
        if output.exists():
            require(resume, "Output already exists; use --resume with identical inputs")
            require(read_json(join(output, "BUILDING.json")) == inputs["identity"], "Resume input identity differs")
            if (output / "COMPLETE.json").exists():
                verify_seal(output)
                return read_json(output / "report.json")
        else:
            output.mkdir(parents=True)
            atomic_json(output / "BUILDING.json", inputs["identity"])
    counts, records, episodes = Counter(), [], 0
    for record, rows, delta in episode_rows(inputs):
        episodes += 1
        counts.update(delta)
        if not preflight_only:
            records.append(write_episode(output, record, rows, delta, inputs["identity"], resume))
        if episodes % 25 == 0:
            print("[{}] {}/{} episodes, {} rows".format("PREFLIGHT" if preflight_only else "APPLY",
                  episodes, inputs["profile"]["source_episodes"], counts["rows"]), flush=True)
    verify_counts(inputs, counts, episodes)
    report = dict(schema=SCHEMA + "_report", status="PREFLIGHT_VERIFIED" if preflight_only else "COMPLETE",
                  episodes=episodes, **dict(counts), input_identity=inputs["identity"],
                  semantic_ready=False, manual_all_frame_camera_certification=False,
                  native_data_modified=False, native_assets_copied=False, source_labels_modified=False,
                  profile_verification_scope="sealed profile metadata and exact exclusion source signatures; camera artifacts not read",
                  mask_contract="instruction_label_valid = situation_target_valid = handoff_target_valid = instruction_target_valid = labeling.action_valid; excluded instruction and labeling.instruction are null")
    if preflight_only:
        return report
    snapshot_sources = {"source_provenance/base_COMPLETE.json": inputs["base"] / "COMPLETE.json",
                        "source_provenance/base_index.json": inputs["base"] / "portable_index.json",
                        "source_provenance/base_review_exclusions.json": inputs["base"] / "review_exclusions.json",
                        "source_provenance/profile_COMPLETE.json": inputs["profile_root"] / "COMPLETE.json",
                        "source_provenance/profile.json": inputs["profile_root"] / "profile.json",
                        "source_provenance/visual_review_exclusions.json": inputs["profile_root"] / "review_exclusions.json",
                        "source_provenance/profile_verification.json": inputs["profile_root"] / "verification.json",
                        "tools/materialize_situation_labels.py": Path(__file__)}
    files = {}
    for name, source in snapshot_sources.items():
        destination = join(output, name)
        files[name] = snapshot(source, destination)
    atomic_json(output / "applied_index.json", dict(schema=SCHEMA + "_index", status="COMPLETE",
                records=records, counts=dict(counts), episodes=episodes, input_identity=inputs["identity"],
                portable_contract="Relocate episode_path/assets.path using dataset_root and portable dataset-relative paths. JSONL instruction masks already contain the applied filter."))
    atomic_json(output / "report.json", report)
    for record in records:
        for name in ("labeled_relative", "manifest_relative", "complete_relative"):
            files[record[name]] = sha256(join(output, record[name]))
    for name in ("BUILDING.json", "applied_index.json", "report.json"):
        files[name] = sha256(output / name)
    atomic_json(output / "COMPLETE.json", dict(schema=SCHEMA + "_complete", status="COMPLETE", files=files,
                input_identity=inputs["identity"], semantic_ready=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-package", "visual-profile", "dataset-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-only", action="store_true", help="Verify all source rows; create no output")
    parser.add_argument("--resume", action="store_true", help="Reuse only sealed, identical completed episode outputs")
    args = parser.parse_args()
    print(json.dumps(materialize(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
