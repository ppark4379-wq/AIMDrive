"""Use an existing sealed full-pose source without promoting admission masks.

This derives instruction evidence only. It does not certify images, geometry
measurements, or visual-encoder training admission. Native files are read-only.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "morai_situation_instruction_source_resolution_v1"
ADMISSION_SCHEMA = "morai_situation_instruction_admission_derived_v1"
POSE_FIELDS = ("actor_pivot_world_xyz", "actor_rotation_rpy_deg", "rotation_rpy_deg")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _folder(episode, value):
    return (episode / value).resolve()


def _verify_stage(folder, pins, seal_name="COMPLETE.json"):
    seal_path = folder / seal_name
    if seal_path.is_symlink():
        raise ValueError("Symlinked source seal: " + str(seal_path))
    seal = _read(seal_path)
    files = seal.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty source seal: " + str(folder))
    pins[str(seal_path)] = _digest(seal_path)
    for name, expected in files.items():
        path = folder / name
        if Path(name).name != name or path.is_symlink() or _digest(path) != expected:
            raise ValueError("Changed or unsafe sealed source: " + str(path))
        pins[str(path)] = expected
    report = _read(folder / "report.json") if "report.json" in files else {}
    if report.get("structural_valid") is False:
        raise ValueError("Structurally invalid source: " + str(folder))
    return seal, report


def _indexed(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing frame rows: " + name)
    result = {}
    for row in rows:
        key = row.get("index")
        if not isinstance(key, int) or isinstance(key, bool) or key in result:
            raise ValueError("Invalid or duplicate frame index: " + name)
        result[key] = row
    return result


def _non_pose_frame(frame):
    result = deepcopy(frame)
    for actor in result.get("object_gt", {}).get("objects", []):
        for field in POSE_FIELDS:
            actor.pop(field, None)
    return result


def _verify_pose_only_parity(old_rows, new_rows):
    old = _indexed(old_rows, "baseline GT")
    new = _indexed(new_rows, "full-pose GT")
    if list(old) != list(new):
        raise ValueError("Full-pose frame identity differs")
    additions = 0
    for index, before in old.items():
        after = new[index]
        if before.get("lidar_source_ns") != after.get("lidar_source_ns"):
            raise ValueError("Full-pose source timestamp differs: " + str(index))
        if _non_pose_frame(before) != _non_pose_frame(after):
            raise ValueError("Full-pose changed non-pose GT evidence: " + str(index))
        old_actors = before.get("object_gt", {}).get("objects", [])
        new_actors = after.get("object_gt", {}).get("objects", [])
        for actor, updated in zip(old_actors, new_actors):
            for field in POSE_FIELDS:
                if field in actor and actor[field] != updated.get(field):
                    raise ValueError("Existing actor pose was modified: " + str(index))
                if field not in actor and field in updated:
                    value = updated[field]
                    if (not isinstance(value, list) or len(value) != 3
                            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                   or not math.isfinite(v) for v in value)):
                        raise ValueError("Invalid added full actor pose: " + str(index))
                    additions += 1
    return old, new, {"frames": len(old), "frame_identity_unchanged": True,
                      "source_timestamps_unchanged": True, "ego_xy_unchanged": True,
                      "all_non_pose_gt_fields_unchanged": True,
                      "pose_fields_added": additions,
                      "allowed_added_actor_fields": list(POSE_FIELDS)}


def inspect_fullpose_instruction_sources(
        episode, *, baseline_admission_dir, baseline_review_dir,
        baseline_geometry_dir, gt_timeline_dir, geometry_dir):
    """Verify the complete old chain and new pair, plus all-frame pose parity."""
    episode = Path(episode).resolve()
    run = episode.parent
    folders = {
        "native_audit": episode / "native_audit", "quality": episode / "quality",
        "native_associations": episode / "native_associations",
        "motion_time": episode / "motion_time_v1",
        "review": _folder(episode, baseline_review_dir),
        "gt_timeline": episode / "gt_timeline_v1",
        "traffic_audit": episode / "traffic_audit_v1",
        "actor_inventory_audit": episode / "actor_inventory_audit_v1",
        "object_geometry_audit": _folder(episode, baseline_geometry_dir),
        "route_signal_audit": episode / "route_signal_audit_v1",
    }
    pins, stages, reports = {}, {}, {}
    admission = _folder(episode, baseline_admission_dir)
    stages["baseline_admission"], base_report = _verify_stage(admission, pins)
    for key, folder in folders.items():
        seal_name = "QUALITY_COMPLETE.json" if key == "quality" else "COMPLETE.json"
        stages[key], reports[key] = _verify_stage(folder, pins, seal_name)
        if base_report.get("input_seals", {}).get(key) != pins[str(folder / seal_name)]:
            raise ValueError("Baseline admission has different input evidence: " + key)
    for key, path in (("capture_contract_sha256", run / "capture_contract.json"),
                      ("run_plan_sha256", run / "plan.json"),
                      ("raw_manifest_sha256", episode / "raw/raw_manifest.json")):
        pins[str(path)] = _digest(path)
        if base_report.get(key) != pins[str(path)]:
            raise ValueError("Baseline admission capture identity differs: " + key)
    full_gt = _folder(episode, gt_timeline_dir)
    full_geometry = _folder(episode, geometry_dir)
    if full_gt == folders["gt_timeline"] or full_geometry == folders["object_geometry_audit"]:
        raise ValueError("Expected distinct existing full-pose source stages")
    new_gt_seal, _ = _verify_stage(full_gt, pins)
    new_geometry_seal, _ = _verify_stage(full_geometry, pins)
    checks = {
        "full_gt_raw_manifest": new_gt_seal.get("raw_manifest_sha256")
            == pins[str(episode / "raw/raw_manifest.json")],
        # The producer names the native audit REPORT hash, not its seal hash.
        "full_gt_native_audit_report": new_gt_seal.get("native_audit_sha256")
            == pins[str(folders["native_audit"] / "report.json")],
        "full_geometry_gt_seal": new_geometry_seal.get("gt_timeline_seal_sha256")
            == pins[str(full_gt / "COMPLETE.json")],
        "full_geometry_quality_seal": new_geometry_seal.get("quality_seal_sha256")
            == pins[str(folders["quality"] / "QUALITY_COMPLETE.json")],
        "full_geometry_actor_inventory_seal": new_geometry_seal.get("actor_inventory_seal_sha256")
            == pins[str(folders["actor_inventory_audit"] / "COMPLETE.json")],
    }
    for key in ("geometry_registry_sha256", "catalog_report_sha256"):
        prior = stages["object_geometry_audit"].get(key)
        checks["unchanged_" + key] = bool(prior and new_geometry_seal.get(key) == prior)
    if not all(checks.values()):
        raise ValueError("Full-pose source dependency mismatch: " + ", ".join(
            key for key, valid in checks.items() if not valid))
    old, new, parity = _verify_pose_only_parity(
        _read(folders["gt_timeline"] / "frames.json"), _read(full_gt / "frames.json"))
    baseline_rows = _read(admission / "frames.json")
    admitted = _indexed(baseline_rows, "baseline admission")
    old_geometry = _indexed(_read(folders["object_geometry_audit"] / "frames.json"), "old geometry")
    geometry = _indexed(_read(full_geometry / "frames.json"), "full-pose geometry")
    if old.keys() != admitted.keys() or old.keys() != geometry.keys() or old.keys() != old_geometry.keys():
        raise ValueError("Admission/geometry frame identity differs")
    for index in old:
        stamp = old[index]["lidar_source_ns"]
        if admitted[index].get("lidar_source_time_ns") != stamp:
            raise ValueError("Baseline admission source timestamp differs")
        if geometry[index].get("lidar_source_ns") != stamp or old_geometry[index].get("lidar_source_ns") != stamp:
            raise ValueError("Selected geometry source timestamp differs")
        native_actors = {(a.get("id"), a.get("instance_key"))
                         for a in new[index].get("object_gt", {}).get("objects", [])}
        geometry_actors = [(a.get("actor_id"), a.get("instance_key")) for a in geometry[index].get("objects", [])]
        if len(geometry_actors) != len(set(geometry_actors)) or set(geometry_actors) != native_actors:
            raise ValueError("Full-pose geometry actor identity differs")
    return {"schema": SCHEMA, "episode": str(episode), "source_pins": pins,
            "source_checks": checks, "parity": parity,
            "baseline_admission_dir": str(admission),
            "baseline_review_dir": str(folders["review"]),
            "baseline_geometry_dir": str(folders["object_geometry_audit"]),
            "gt_timeline_dir": str(full_gt), "geometry_dir": str(full_geometry),
            "baseline_input_seals": deepcopy(base_report["input_seals"]),
            "baseline_admission_report": base_report,
            "baseline_admission_rows": baseline_rows,
            "observation_and_admission_masks_unchanged": True,
            "physical_front_safety_certified": False,
            "visual_encoder_admission_promoted": False}


def _compact_inspection(inspection):
    return {key: value for key, value in inspection.items()
            if key not in ("baseline_admission_rows", "baseline_admission_report")}


def verify_instruction_source_resolution(output_dir, episode=None):
    """Verify a completed derived receipt and every pinned source on resume."""
    root = Path(output_dir).resolve()
    complete = _read(root / "RESOLUTION_COMPLETE.json")
    if complete.get("schema") != SCHEMA or complete.get("status") != "COMPLETE":
        raise ValueError("Incomplete instruction source resolution")
    if episode is not None and complete.get("episode") != str(Path(episode).resolve()):
        raise ValueError("Derived source resolution belongs to another episode")
    for name, expected in complete.get("files", {}).items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or _digest(root / rel) != expected:
            raise ValueError("Changed derived instruction source resolution: " + name)
    receipt = _read(root / "source_resolution_receipt.json")
    for path, expected in receipt["source_pins"].items():
        if _digest(path) != expected:
            raise ValueError("Changed source after instruction derivation: " + path)
    for folder in ("instruction_source_review", "instruction_source_admission"):
        _verify_stage(root / folder, {})
    admission_rows = _read(root / "instruction_source_admission/frames.json")
    if admission_rows != _read(Path(receipt["baseline_admission_dir"]) / "frames.json"):
        raise ValueError("Derived admission promoted or changed baseline frame fields")
    return {"gt_timeline_dir": receipt["gt_timeline_dir"],
            "geometry_dir": receipt["geometry_dir"],
            "admission_dir": str(root / "instruction_source_admission"),
            "review_dir": str(root / "instruction_source_review"),
            "source_resolution_receipt": str(root / "source_resolution_receipt.json"),
            "source_resolution_complete_sha256": _digest(root / "RESOLUTION_COMPLETE.json")}


def derive_fullpose_instruction_admission(episode, output_dir, **source_options):
    """Create an exclusive, resumable instruction-only admission beside outputs."""
    episode = Path(episode).resolve()
    root = Path(output_dir).resolve()
    if root == episode.parent or episode.parent in root.parents:
        raise ValueError("Derived evidence must be outside the native run")
    inspection = inspect_fullpose_instruction_sources(episode, **source_options)
    if root.exists():
        resolved = verify_instruction_source_resolution(root, episode)
        receipt = _read(root / "source_resolution_receipt.json")
        for key in ("gt_timeline_dir", "geometry_dir", "baseline_admission_dir",
                    "baseline_review_dir", "baseline_geometry_dir", "source_pins"):
            if receipt.get(key) != inspection.get(key):
                raise ValueError("Resume requested different instruction evidence: " + key)
        return resolved
    root.mkdir(parents=True, exist_ok=False)
    review = root / "instruction_source_review"
    admission = root / "instruction_source_admission"
    review.mkdir()
    admission.mkdir()
    receipt = _compact_inspection(inspection)
    receipt["derivation_module_sha256"] = _digest(__file__)
    receipt["instruction_only"] = True
    _write(root / "source_resolution_receipt.json", receipt)
    full_seals = {
        "gt_timeline": _digest(Path(inspection["gt_timeline_dir"]) / "COMPLETE.json"),
        "object_geometry_audit": _digest(Path(inspection["geometry_dir"]) / "COMPLETE.json"),
    }
    _write(review / "report.json", {
        "schema": "morai_situation_instruction_review_derived_v1",
        "input_seals": {**inspection["baseline_input_seals"], **full_seals},
        "source_resolution_receipt_sha256": _digest(root / "source_resolution_receipt.json"),
        "baseline_review_seal_sha256": _digest(Path(inspection["baseline_review_dir"]) / "COMPLETE.json"),
        "frames": inspection["parity"]["frames"],
        "observation_and_admission_masks_unchanged": True,
        "physical_front_safety_certified": False,
        "visual_encoder_admission_promoted": False, "training_visible": False})
    _write(review / "COMPLETE.json", {"schema": SCHEMA,
        "files": {"report.json": _digest(review / "report.json")}, "training_visible": False})
    report = deepcopy(inspection["baseline_admission_report"])
    report["schema"] = ADMISSION_SCHEMA
    report["input_seals"] = {**inspection["baseline_input_seals"], **full_seals,
                             "review": _digest(review / "COMPLETE.json")}
    report["instruction_source_derivation"] = {
        "schema": SCHEMA, "instruction_only": True,
        "source_resolution_receipt_sha256": _digest(root / "source_resolution_receipt.json"),
        "baseline_admission_seal_sha256": _digest(Path(inspection["baseline_admission_dir"]) / "COMPLETE.json"),
        "all_admission_frame_fields_unchanged": True,
        "physical_front_safety_certified": False, "visual_encoder_admission_promoted": False}
    _write(admission / "frames.json", inspection["baseline_admission_rows"])
    _write(admission / "report.json", report)
    _write(admission / "COMPLETE.json", {"schema": SCHEMA,
        "files": {name: _digest(admission / name) for name in ("frames.json", "report.json")},
        "training_visible": False})
    files = ["source_resolution_receipt.json", "instruction_source_review/report.json",
             "instruction_source_review/COMPLETE.json", "instruction_source_admission/frames.json",
             "instruction_source_admission/report.json", "instruction_source_admission/COMPLETE.json"]
    _write(root / "RESOLUTION_COMPLETE.json", {"schema": SCHEMA, "status": "COMPLETE",
        "episode": str(episode), "files": {name: _digest(root / name) for name in files},
        "instruction_only": True, "training_visible": False})
    return verify_instruction_source_resolution(root, episode)


def verify_derived_instruction_sources(output_dir, episode=None):
    """Public wrapper used by the official and metadata-only entry points."""
    return verify_instruction_source_resolution(output_dir, episode)
