import copy
import hashlib
import json
from pathlib import Path

import pytest


from MoraiDrive.data_collection.situation_labeling import instruction_source_resolution as M


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(folder, values, extra=None, name="COMPLETE.json"):
    for key, value in values.items():
        put(folder / key, value)
    put(folder / name, {"files": {key: sha(folder / key) for key in values}, **(extra or {})})


@pytest.fixture
def case(tmp_path):
    ep = tmp_path / "native/run/episode_000"
    ep.mkdir(parents=True)
    old_actor = {"id": "21", "instance_key": "0|VEHICLE|21|0", "kind": "npc",
                 "position": [10.0, 0.0, 1.0], "size_wlh_m": [2.0, 4.0, 1.5],
                 "heading_deg": 0.0, "velocity_kmh": [1.0, 0.0, 0.0]}
    gt = [{"index": 0, "lidar_source_ns": 100, "ego_xy": [0.0, 0.0],
           "object_gt": {"objects": [old_actor], "transport_timing_candidate": False,
                         "model_identity_candidate": True, "complete_inventory": True}}]
    geom = [{"index": 0, "lidar_source_ns": 100,
             "objects": [{"actor_id": "21", "instance_key": "0|VEHICLE|21|0",
                          "api_box_geometry_candidate": False}]}]
    admission_rows = [{"index": 0, "lidar_source_time_ns": 100,
                       "dataset_build_candidate": False,
                       "object_gt_transport_timing_candidate": False,
                       "object_training_label_valid": False,
                       "sensor_bundle_valid": False, "assets": {}}]
    folders = {"native_audit": "native_audit", "quality": "quality",
               "native_associations": "native_associations", "motion_time": "motion_time_v1",
               "review": "review_old", "gt_timeline": "gt_timeline_v1",
               "traffic_audit": "traffic_audit_v1", "actor_inventory_audit": "actor_inventory_audit_v1",
               "object_geometry_audit": "geometry_old", "route_signal_audit": "route_signal_audit_v1"}
    for key, folder in folders.items():
        data = {"report.json": {"structural_valid": True}}
        extra = None
        if key == "gt_timeline":
            data["frames.json"] = gt
        if key == "object_geometry_audit":
            data["frames.json"] = geom
            extra = {"geometry_registry_sha256": "registry", "catalog_report_sha256": "catalog"}
        seal(ep / folder, data, extra, "QUALITY_COMPLETE.json" if key == "quality" else "COMPLETE.json")
    for p in (ep.parent / "capture_contract.json", ep.parent / "plan.json", ep / "raw/raw_manifest.json"):
        put(p, {"immutable": True})
    report = {"input_seals": {key: sha(ep / f / ("QUALITY_COMPLETE.json" if key == "quality" else "COMPLETE.json"))
                               for key, f in folders.items()},
              "capture_contract_sha256": sha(ep.parent / "capture_contract.json"),
              "run_plan_sha256": sha(ep.parent / "plan.json"),
              "raw_manifest_sha256": sha(ep / "raw/raw_manifest.json")}
    seal(ep / "admission_old", {"report.json": report, "frames.json": admission_rows})
    new_gt = copy.deepcopy(gt)
    new_gt[0]["object_gt"]["objects"][0].update(
        actor_pivot_world_xyz=[9.0, 0.0, 0.5], actor_rotation_rpy_deg=[0.2, 0.5, 0.0],
        rotation_rpy_deg=[0.2, 0.5, 0.0])
    seal(ep / "gt_timeline_fullpose_test", {"frames.json": new_gt, "report.json": {}}, {
        "raw_manifest_sha256": sha(ep / "raw/raw_manifest.json"),
        "native_audit_sha256": sha(ep / "native_audit/report.json")})
    new_geom = copy.deepcopy(geom)
    new_geom[0]["objects"][0]["api_box_geometry_candidate"] = True
    seal(ep / "object_geometry_fullpose_test", {"frames.json": new_geom, "report.json": {}}, {
        "gt_timeline_seal_sha256": sha(ep / "gt_timeline_fullpose_test/COMPLETE.json"),
        "quality_seal_sha256": sha(ep / "quality/QUALITY_COMPLETE.json"),
        "actor_inventory_seal_sha256": sha(ep / "actor_inventory_audit_v1/COMPLETE.json"),
        "geometry_registry_sha256": "registry", "catalog_report_sha256": "catalog"})
    options = dict(baseline_admission_dir="admission_old", baseline_review_dir="review_old",
                   baseline_geometry_dir="geometry_old", gt_timeline_dir="gt_timeline_fullpose_test",
                   geometry_dir="object_geometry_fullpose_test")
    return ep, tmp_path / "outputs/resolution", options


def inspect(case):
    ep, _, opts = case
    return M.inspect_fullpose_instruction_sources(ep, **opts)


def change_full_gt(case, transform):
    ep, _, _ = case
    p = ep / "gt_timeline_fullpose_test"
    rows = json.loads((p / "frames.json").read_text())
    transform(rows)
    s = json.loads((p / "COMPLETE.json").read_text())
    seal(p, {"frames.json": rows, "report.json": {}}, {k: v for k, v in s.items() if k != "files"})
    g = ep / "object_geometry_fullpose_test/COMPLETE.json"
    gs = json.loads(g.read_text()); gs["gt_timeline_seal_sha256"] = sha(p / "COMPLETE.json"); put(g, gs)


def test_all_baseline_masks_copied_and_source_pair_bound(case):
    ep, out, opts = case
    before = {str(p): sha(p) for p in ep.rglob("*.json")}
    result = M.derive_fullpose_instruction_admission(ep, out, **opts)
    assert json.loads((Path(result["admission_dir"]) / "frames.json").read_text()) == json.loads((ep / "admission_old/frames.json").read_text())
    assert not json.loads((Path(result["admission_dir"]) / "frames.json").read_text())[0]["dataset_build_candidate"]
    assert before == {str(p): sha(p) for p in ep.rglob("*.json")}
    report = json.loads((Path(result["admission_dir"]) / "report.json").read_text())
    assert report["input_seals"]["gt_timeline"] == sha(ep / "gt_timeline_fullpose_test/COMPLETE.json")
    assert report["input_seals"]["object_geometry_audit"] == sha(ep / "object_geometry_fullpose_test/COMPLETE.json")
    assert report["input_seals"]["review"] == sha(Path(result["review_dir"]) / "COMPLETE.json")
    assert report["instruction_source_derivation"]["visual_encoder_admission_promoted"] is False
    assert M.derive_fullpose_instruction_admission(ep, out, **opts) == result
    assert M.verify_derived_instruction_sources(out, ep) == result


@pytest.mark.parametrize("field,value", [("lidar_source_ns", 101), ("ego_xy", [2.0, 0.0]), ("index", 1)])
def test_non_pose_frame_changes_rejected(case, field, value):
    change_full_gt(case, lambda rows: rows[0].__setitem__(field, value))
    with pytest.raises(ValueError):
        inspect(case)


@pytest.mark.parametrize("field,value", [("velocity_kmh", [2.0, 0.0, 0.0]), ("id", "22"),
                                         ("position", [9.9, 0.0, 1.0]), ("size_wlh_m", [2.0, 5.0, 1.5])])
def test_non_pose_actor_changes_rejected(case, field, value):
    change_full_gt(case, lambda rows: rows[0]["object_gt"]["objects"][0].__setitem__(field, value))
    with pytest.raises(ValueError, match="non-pose"):
        inspect(case)


def test_duplicate_frames_rejected(case):
    change_full_gt(case, lambda rows: rows.append(copy.deepcopy(rows[0])))
    with pytest.raises(ValueError, match="duplicate"):
        inspect(case)


def test_frame_order_change_rejected():
    rows = [{"index": i, "lidar_source_ns": 100 + i, "ego_xy": [0, 0],
             "object_gt": {"objects": []}} for i in (0, 1)]
    with pytest.raises(ValueError, match="frame identity"):
        M._verify_pose_only_parity(rows, list(reversed(rows)))


def test_invalid_added_pose_rejected(case):
    change_full_gt(case, lambda rows: rows[0]["object_gt"]["objects"][0].__setitem__("actor_pivot_world_xyz", [None, 0, 0]))
    with pytest.raises(ValueError, match="Invalid added"):
        inspect(case)


@pytest.mark.parametrize("key", ["gt_timeline_seal_sha256", "quality_seal_sha256",
                                 "actor_inventory_seal_sha256", "geometry_registry_sha256",
                                 "catalog_report_sha256"])
def test_wrong_fullpose_dependency_rejected(case, key):
    ep, _, _ = case; p = ep / "object_geometry_fullpose_test/COMPLETE.json"
    value = json.loads(p.read_text()); value[key] = "different"; put(p, value)
    with pytest.raises(ValueError, match="dependency mismatch"):
        inspect(case)


def test_changed_baseline_seal_chain_rejected(case):
    ep, _, _ = case; p = ep / "admission_old/report.json"
    v = json.loads(p.read_text()); v["input_seals"]["gt_timeline"] = "wrong"
    rows = json.loads((p.parent / "frames.json").read_text()); seal(p.parent, {"frames.json": rows, "report.json": v})
    with pytest.raises(ValueError, match="Baseline admission"):
        inspect(case)


def test_geometry_actor_mismatch_rejected(case):
    ep, _, _ = case; p = ep / "object_geometry_fullpose_test"
    s = json.loads((p / "COMPLETE.json").read_text()); rows = json.loads((p / "frames.json").read_text())
    rows[0]["objects"][0]["actor_id"] = "wrong"
    seal(p, {"frames.json": rows, "report.json": {}}, {k: v for k, v in s.items() if k != "files"})
    with pytest.raises(ValueError, match="actor identity"):
        inspect(case)


def test_no_native_writes(case):
    ep, _, opts = case
    with pytest.raises(ValueError, match="outside the native"):
        M.derive_fullpose_instruction_admission(ep, ep.parent / "derived", **opts)


def test_incomplete_output_not_overwritten(case):
    ep, out, opts = case; out.mkdir(parents=True); put(out / "interrupted.json", {})
    with pytest.raises(FileNotFoundError):
        M.derive_fullpose_instruction_admission(ep, out, **opts)
    assert (out / "interrupted.json").exists()


def test_changed_output_or_source_rejected_on_resume(case):
    ep, out, opts = case; M.derive_fullpose_instruction_admission(ep, out, **opts)
    put(out / "instruction_source_admission/frames.json", [])
    with pytest.raises(ValueError, match="Changed derived"):
        M.verify_derived_instruction_sources(out, ep)


def test_resume_other_episode_rejected(case):
    ep, out, opts = case; M.derive_fullpose_instruction_admission(ep, out, **opts)
    with pytest.raises(ValueError, match="another episode"):
        M.verify_derived_instruction_sources(out, ep.parent / "episode_001")
