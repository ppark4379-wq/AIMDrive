from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT.parents[1]))
from morai_pretraining.scaled_object_geometry import (
    check_scaled_object_geometry, match_static_instances, resolve_pinned_preset,
    scaled_reference, load_scale_evidence,
)
from morai_pretraining import scaled_object_geometry as scaled
from async_capture.journal import RawWriter, MAGIC, PREFIX, digest, json_bytes
from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.diversity import STRICT_FLAGS, STRICT_CONDITIONS
from audit_sensor_record_object_geometry_v2 import check_object_geometry


def catalog(scale=(1., 1., 1.)):
    return {"kind": "static", "scale": list(scale),
            "visible_return_contradiction": False,
            "views": [{"scale": list(scale), "dimensions_lwh_m": [4., 2., 3.]}
                      for _ in range(4)]}


def observation(record=10, position=(1., 2., 3.), **kwargs):
    row = {"actor_id": "7", "model": "Box", "identity_valid": True,
           "record_id": record, "position": list(position)}
    row.update(kwargs)
    return row


def obstacle(position=(1., 2., 3.), scale=(1.2, 1.2, 1.2)):
    return {"model": "Box", "position": dict(zip("xyz", position)),
            "scale": list(scale)}


class ScaledGeometryTest(unittest.TestCase):
    def test_uniform_scale_changes_catalog_reference_not_observation(self):
        dimensions, evidence = scaled_reference(catalog(), [1.2, 1.2, 1.2])
        for actual, expected in zip(dimensions, [2.4, 4.8, 3.6]):
            self.assertAlmostEqual(actual, expected)
        self.assertIsNone(evidence["reason"])

    def test_nonuniform_measured_base_uses_proportional_ratio(self):
        dimensions, evidence = scaled_reference(catalog([2, 3, 2]), [1.8, 2.7, 1.8])
        self.assertEqual(dimensions[:2], [1.8, 3.6])
        self.assertAlmostEqual(dimensions[2], 2.7)
        self.assertAlmostEqual(evidence["relative_scale_multiplier"], .9)

    def test_anisotropic_delta_is_not_guessed_from_bbox(self):
        dimensions, evidence = scaled_reference(catalog([2, 3, 2]), [2., 2., 2.])
        self.assertIsNone(dimensions)
        self.assertEqual(evidence["reason"], "anisotropic_relative_scale_uncalibrated")

    def test_rounded_generator_vector_is_supported(self):
        dimensions, evidence = scaled_reference(catalog([2, 3, 2]), [2.173219, 3.259829, 2.173219])
        self.assertIsNotNone(dimensions)
        self.assertIsNone(evidence["reason"])

    def test_catalog_scale_missing_or_inconsistent_fails_closed(self):
        for case in [dict(catalog(), scale=None), catalog()]:
            if case["scale"] is not None:
                case["views"][1]["scale"] = [2, 2, 2]
            self.assertIsNone(scaled_reference(case, [1, 1, 1])[0])

    def test_match_uses_pivot_and_every_snapshot(self):
        rows = {"instance": [observation(), observation(11)]}
        self.assertTrue(match_static_instances([obstacle()], rows)["instance"]["valid"])
        rows["instance"][1]["position"][0] += .1
        self.assertFalse(match_static_instances([obstacle()], rows)["instance"]["valid"])

    def test_unknown_identity_and_ambiguous_placement_fail_closed(self):
        for placements, rows in [([obstacle(), obstacle()], [observation()]),
                ([obstacle()], [observation(identity_valid=False)]),
                ([obstacle()], [observation(model="Other")])]:
            self.assertFalse(match_static_instances(placements, {"k": rows})["k"]["valid"])

    def test_reused_numeric_actor_id_does_not_reuse_scale(self):
        result = match_static_instances([obstacle()], {
            "generation0": [observation()], "generation1": [observation(11)]})
        self.assertTrue(all(not row["valid"] for row in result.values()))

    def check(self, *, dimensions=(2.4, 4.8, 3.6), evidence=None, source_ids=(10,), case=None):
        if evidence is None:
            evidence = {"instances": match_static_instances([obstacle()], {
                "instance": [observation()]})}
        item = {"instance_key": "instance", "id": "7", "model_name": "Box",
                "kind": "obstacle", "size_wlh_m": list(dimensions)}
        return check_scaled_object_geometry(item, {"Box": case or catalog()}, "hash", .03, .02,
                                            evidence, list(source_ids), check_object_geometry)

    def test_valid_scaled_box_passes_but_wrong_dimensions_do_not(self):
        self.assertTrue(self.check()["api_box_geometry_candidate"])
        result = self.check(dimensions=(4.8, 2.4, 3.6))
        self.assertFalse(result["api_box_geometry_candidate"])
        self.assertIn("observed_dimensions_differ_from_scaled_catalog", result["reasons"])

    def test_missing_evidence_or_unverified_frame_cannot_recover(self):
        self.assertFalse(self.check(evidence={})["api_box_geometry_candidate"])
        self.assertFalse(self.check(source_ids=(10, 11))["api_box_geometry_candidate"])
        self.assertFalse(self.check(source_ids=())["api_box_geometry_candidate"])

    def test_existing_catalog_contradiction_is_not_removed(self):
        case = catalog()
        case["visible_return_contradiction"] = True
        self.assertIn("catalog_visible_return_contradiction", self.check(case=case)["reasons"])

    def test_null_static_preset_is_valid_for_nonstatic_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_pinned_preset(None, root, root), (None, None))
            with self.assertRaises(ValueError):
                resolve_pinned_preset("invalid", root, root)

    def test_preset_lookup_requires_exact_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode_000"
            episode.mkdir()
            preset = episode / "static_obstacle_preset.yaml"
            preset.write_text("schema: morai_static_obstacle_preset_v1\nobstacles: []\n")
            record = {"path": "/gone/scenario_runner/preset.yaml",
                      "sha256": hashlib.sha256(preset.read_bytes()).hexdigest()}
            self.assertIsNotNone(resolve_pinned_preset(record, episode, Path(directory))[0])
            preset.write_text(preset.read_text() + "tampered: true\n")
            self.assertIsNone(resolve_pinned_preset(record, episode, Path(directory))[0])


class TunnelScaleEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "campaign_lap_004" / "episode_000"
        self.episode.mkdir(parents=True)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json_bytes(value) + b"\n")

    def fixture(self, *, snapshots=None, scale=(1., 1., 1.), scenario="full_loop_tunnel_static", preset=None):
        runtime = {"/recorded/source_%d.py" % i: "%064x" % (i+1) for i in range(8)}
        position, rotation = [1., 2., 3.], [0., 0., 10.]
        box = {"center_xyz": position, "extent_xyz": [1., 2., 1.5], "rotation_xyz_deg": rotation}
        row = {"raw_actor_id": "7", "actor_kind": "obstacle", "model": "Box", "position_xyz": position,
            "rotation_xyz_deg": rotation, "world_bbox": box,
            "raw_bbox_normalized": {"center_xyz": position, "width_m": 2., "length_m": 4., "heading_deg": 10.},
            "global_velocity_raw": [0., 0., 0.], "angular_velocity_raw": [0., 0., 0.],
            "request_id": "creating_request_0", "handle_id": "creating_request_0",
            "stationary_controls_accepted": {"static_transform_lock": [
                {"state": "stable", "repair": {"physics_disabled": True, "transform_reapplied": True}},
                {"state": "stable", "repair": None}]}}
        candidate = {"count": 1, "obstacles": [dict(obstacle(scale=scale), actor_kind="obstacle")]}
        candidate_path = self.episode / "tunnel_static/candidate.json"
        self.write(candidate_path, candidate)
        boundary = {"body": {"rear_m": 1., "front_m": 3., "width_m": 2.}}
        boundary_sha = scaled._json_identity(boundary)
        metadata = {"scenario_name": scenario, "static_obstacle_preset": preset,
            "tunnel_static_trial": {"selected_candidate_sha256": digest(candidate_path),
                "runtime_source_sha256": runtime, "road_boundary_policy_sha256": boundary_sha}}
        self.write(self.episode / "episode.json", metadata)
        actor = {"actor_info": {"id": {"value": "7"}, "object_type": "OBJECT_TYPE_OBSTACLE", "client_key": ""},
            "transform": {"location": dict(zip("xyz", position)), "rotation": dict(zip("xyz", rotation))},
            "bounding_box": {"location": dict(zip("xyz", position)), "extent": dict(zip("xyz", box["extent_xyz"])),
                "rotation": dict(zip("xyz", rotation))},
            "global_velocity": dict(zip("xyz", [0., 0., 0.])), "angular_velocity": dict(zip("xyz", [0., 0., 0.]))}
        inventory = {"response": {"states": [actor]}, "actor_instances": {"7": {
            "instance_key": "generation0|7", "model_identity_valid": True, "model_name": "Box"}}}
        raw = self.episode / "raw"
        writer = RawWriter(raw, {"episode": metadata}, chunk_bytes=4096)
        for item in snapshots or [inventory, deepcopy(inventory)]:
            self.assertTrue(writer.submit({"role": "actor_inventory"}, json_bytes(item)))
        writer.close()
        self.active = []
        for path in sorted((raw / "chunks").iterdir()):
            with path.open("rb") as stream:
                self.assertEqual(stream.read(len(MAGIC)), MAGIC)
                while True:
                    prefix = stream.read(PREFIX.size)
                    if not prefix:
                        break
                    nm, np = PREFIX.unpack(prefix)
                    meta = json.loads(stream.read(nm))
                    offset = stream.tell()
                    payload = stream.read(np)
                    self.active.append(dict(meta, chunk_path="chunks/"+path.name, payload_offset=offset,
                        payload_bytes=np, payload_sha256=hashlib.sha256(payload).hexdigest()))
        self.write(self.episode / "quality/INDEX_COMPLETE.json", {"raw_path": str(raw)})
        self.raw_sha = digest(raw / "raw_manifest.json")
        screen = {"path_found": True, "geometry_checks": {"passed": True}, "road_boundary_policy": boundary,
                  "road_boundary_policy_sha256": boundary_sha}
        preview = {"status": "valid_driving_trial", "planned": candidate, "observed": [row],
            "hold_seconds": 5., "stationary_observed_seconds": 5., "measured_route_screen": screen,
            "cleanup": {"verified": True, "fixture_client_key": "fixture_owner", "owned_request_bindings": [{
                "raw_actor_id": "7", "request_id": "creating_request_0", "source_client_key": "fixture_owner", "verified": True}]}}
        drive = {"status": "passed", "runtime_source_sha256": runtime, "owned_actor_geometry": [deepcopy(row)],
            "observed_object_ids": ["7"], "collision_objects": [], "failed_completion_conditions": [],
            "road_boundary_policy": boundary, "road_boundary_policy_sha256": boundary_sha,
            "completion_conditions": {key: True for key in STRICT_CONDITIONS}, **{key: True for key in STRICT_FLAGS}}
        planner = {"validated_path_seen": True, "selected_routes": [{"verified": True}],
            "scene_geometry_sha256": scaled._json_identity(drive["owned_actor_geometry"]),
            "road_boundary_policy": boundary, "road_boundary_policy_sha256": boundary_sha}
        self.roles = {"candidate": "tunnel_static/candidate.json", "preview_report": "tunnel_static/preview.json",
            "drive_report": "tunnel_static/drive_trial.json", "planner_diagnostics": "tunnel_static/planner_diagnostics.json",
            "ego_trace": "tunnel_static/ego_trace.csv", "episode": "episode.json", "outcome": "outcome.json",
            "raw_seal": "raw/RAW_COMPLETE.json", "scene_restore": "tunnel_static/scene_restore.json"}
        for role, value in {"preview_report": preview, "drive_report": drive, "planner_diagnostics": planner,
                "outcome": {"status": "mission_completed", "full_mission_completed": True},
                "scene_restore": {"status": "restored"}}.items():
            self.write(self.episode / self.roles[role], value)
        (self.episode / self.roles["ego_trace"]).write_text("x,y,speed\n0,0,0\n1,1,1\n2,2,0\n")
        self.proof_path = self.episode / "tunnel_static/COLLECTION_TUNNEL_COMPLETE.json"
        self.proof = {"schema": "aimdrive_tunnel_static_collection_complete_v1", "scenario": "full_loop_tunnel_static",
            "status": "passed", "fixture_cleanup_verified": True, "scene_restored": True, "raw_record_complete": True,
            "selected_candidate_sha256": digest(candidate_path), "library_sha256": "a"*64,
            "runtime_source_sha256": runtime, "source_files": deepcopy(runtime)}
        self.reseal()
        return inventory

    def reseal(self):
        self.proof["files"] = {role: {"path": str(self.episode / rel), "sha256": digest(self.episode / rel)}
                               for role, rel in self.roles.items()}
        self.write(self.proof_path, self.proof)

    def change(self, role, update, *, reseal=True):
        path = self.episode / self.roles[role]
        value = json.loads(path.read_text())
        update(value)
        self.write(path, value)
        if reseal:
            self.reseal()

    def load(self):
        return load_scale_evidence(self.episode, self.active, self.raw_sha, self.root, self.root)

    def test_finalized_proof_links_explicit_scale_to_every_recorded_instance(self):
        self.fixture()
        before = {str(p): digest(p) for p in (self.episode / "raw").rglob("*") if p.is_file()}
        result = self.load()
        match = result["instances"]["generation0|7"]
        self.assertTrue(match["valid"])
        self.assertEqual(match["scale_xyz"], [1., 1., 1.])
        self.assertEqual(match["record_ids"], [0, 1])
        self.assertEqual(result["source"], "finalized_tunnel_fixture_scale_bridge_v1")
        self.assertIn("exact_fixture_scale_only", result["tunnel_proof"]["authorization_scope"])
        self.assertEqual(before, {str(p): digest(p) for p in (self.episode / "raw").rglob("*") if p.is_file()})
        # Capture-era source paths need not still exist on the analysis PC.
        self.assertFalse(Path("/recorded/source_0.py").exists())

    def test_missing_or_failed_final_proof_cannot_supply_scale(self):
        self.fixture()
        self.proof["status"] = "failed"
        self.write(self.proof_path, self.proof)
        with self.assertRaises(ValueError): self.load()
        self.proof_path.unlink()
        with self.assertRaises(OSError): self.load()

    def test_changed_artifact_and_resealed_foreign_candidate_are_rejected(self):
        self.fixture()
        self.change("candidate", lambda d: d["obstacles"][0].update(scale=[2., 2., 2.]), reseal=False)
        with self.assertRaises(ValueError): self.load()
        self.reseal()
        with self.assertRaises(ValueError): self.load()

    def test_raw_and_capture_runtime_bindings_cannot_be_substituted(self):
        self.fixture()
        with self.assertRaises(ValueError):
            load_scale_evidence(self.episode, self.active, "f"*64, self.root, self.root)
        self.change("drive_report", lambda d: d["runtime_source_sha256"].update({"/recorded/source_0.py": "f"*64}))
        self.proof["runtime_source_sha256"]["/recorded/source_0.py"] = "f"*64
        self.write(self.proof_path, self.proof)
        with self.assertRaises(ValueError): self.load()

    def test_captured_runtime_source_cannot_be_omitted_from_final_source_pins(self):
        self.fixture()
        del self.proof["source_files"]["/recorded/source_0.py"]
        self.write(self.proof_path, self.proof)
        with self.assertRaisesRegex(ValueError, "captured runtime/source identities"):
            self.load()

    def test_owner_model_motion_and_native_freeze_evidence_remain_required(self):
        self.fixture()
        original = (self.episode / self.roles["preview_report"]).read_bytes()
        edits = [lambda d: d["cleanup"]["owned_request_bindings"][0].update(source_client_key="foreign"),
                 lambda d: d["observed"][0].update(model="Other"),
                 lambda d: d["observed"][0].update(global_velocity_raw=[.1, 0., 0.]),
                 lambda d: d["observed"][0]["stationary_controls_accepted"].update(static_transform_lock=[])]
        for edit in edits:
            with self.subTest(edit=edit):
                (self.episode / self.roles["preview_report"]).write_bytes(original)
                self.change("preview_report", edit)
                with self.assertRaises(ValueError): self.load()

    def test_artifact_path_cannot_escape_or_substitute_another_episode(self):
        self.fixture()
        for value in ("../../candidate.json", "/different_run/episode_000/tunnel_static/candidate.json"):
            with self.subTest(path=value):
                self.proof["files"]["candidate"]["path"] = value
                self.write(self.proof_path, self.proof)
                with self.assertRaises(ValueError): self.load()

    def test_renamed_capture_uses_only_exact_local_artifact_hashes(self):
        self.fixture()
        old = self.episode
        destination = self.root / "promoted" / self.episode.parent.name
        destination.parent.mkdir()
        self.episode.parent.rename(destination)
        self.episode = destination / old.name
        # Supply a valid local INDEX: only paths move; journal/proof bytes stay sealed.
        self.write(self.episode / "quality/INDEX_COMPLETE.json", {"raw_path": str(self.episode / "raw")})
        self.assertTrue(self.load()["instances"]["generation0|7"]["valid"])

    def test_nonunit_scale_does_not_become_an_assumed_unit_spawn(self):
        self.fixture(scale=(2., 2., 2.))
        with self.assertRaisesRegex(ValueError, "explicit unit spawn scale"): self.load()

    def test_nonfinite_or_incomplete_hold_evidence_cannot_authorize_scale(self):
        self.fixture()
        path = self.episode / self.roles["preview_report"]
        original = path.read_bytes()
        for value in (None, "5", True, .5, float("nan")):
            with self.subTest(hold=value):
                data = json.loads(original)
                data["hold_seconds"] = value
                path.write_text(json.dumps(data))
                self.reseal()
                with self.assertRaises(ValueError): self.load()

    def test_failed_drive_scene_restore_or_changed_scene_hash_is_rejected(self):
        self.fixture()
        edits = (("drive_report", lambda d: d.update(fixture_stationary=False)),
                 ("scene_restore", lambda d: d.update(status="restore_failed")),
                 ("planner_diagnostics", lambda d: d.update(scene_geometry_sha256="f"*64)))
        originals = {role: (self.episode / self.roles[role]).read_bytes() for role, _ in edits}
        for role, edit in edits:
            with self.subTest(role=role):
                for old_role, content in originals.items():
                    (self.episode / self.roles[old_role]).write_bytes(content)
                self.change(role, edit)
                with self.assertRaises(ValueError): self.load()

    def test_legacy_non_tunnel_null_preset_never_uses_tunnel_proof(self):
        self.fixture(scenario="full_loop_dynamic")
        with patch.object(scaled, "_load_tunnel_scale_proof", side_effect=AssertionError("not a tunnel")):
            result = self.load()
        self.assertFalse(result["preset_resolved"])
        self.assertEqual(result["instances"], {})

    def test_legacy_hash_pinned_static_preset_keeps_its_original_join(self):
        preset = self.episode / "static_obstacle_preset.yaml"
        preset.write_text("schema: morai_static_obstacle_preset_v1\nobstacles:\n- model: Box\n  position: {x: 1, y: 2, z: 3}\n  scale: [1.2, 1.2, 1.2]\n")
        self.fixture(scenario="full_loop_static", preset={"path": str(preset), "sha256": digest(preset)})
        with patch.object(scaled, "_load_tunnel_scale_proof", side_effect=AssertionError("legacy preset")):
            result = self.load()
        self.assertTrue(result["instances"]["generation0|7"]["valid"])
        self.assertEqual(result["instances"]["generation0|7"]["scale_xyz"], [1.2, 1.2, 1.2])

    def test_native_body_drift_wrong_identity_and_reused_id_do_not_recover_scale(self):
        # Build each raw journal afresh so the altered observation is itself sealed.
        changes = ("extent", "moving", "model", "identity", "id", "reuse", "nonfinite", "foreign_owner")
        for case in changes:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                saved = self.episode
                self.episode = Path(folder) / "campaign_lap_004" / "episode_000"
                self.episode.mkdir(parents=True)
                template = {"response": {"states": [{"actor_info": {"id": {"value": "7"}, "object_type": "OBJECT_TYPE_OBSTACLE"},
                    "transform": {"location": {"x": 1., "y": 2., "z": 3.}, "rotation": {"x": 0., "y": 0., "z": 10.}},
                    "bounding_box": {"location": {"x": 1., "y": 2., "z": 3.}, "extent": {"x": 1., "y": 2., "z": 1.5},
                        "rotation": {"x": 0., "y": 0., "z": 10.}}, "global_velocity": {"x": 0., "y": 0., "z": 0.},
                    "angular_velocity": {"x": 0., "y": 0., "z": 0.}}]},
                    "actor_instances": {"7": {"instance_key": "generation0|7", "model_identity_valid": True, "model_name": "Box"}}}
                altered = deepcopy(template)
                if case == "extent": altered["response"]["states"][0]["bounding_box"]["extent"]["x"] += .02
                elif case == "moving": altered["response"]["states"][0]["global_velocity"]["x"] = .1
                elif case == "model": altered["actor_instances"]["7"]["model_name"] = "Other"
                elif case == "identity": altered["actor_instances"]["7"]["model_identity_valid"] = False
                elif case == "id":
                    altered["response"]["states"][0]["actor_info"]["id"]["value"] = "8"
                    altered["actor_instances"] = {"8": dict(altered["actor_instances"]["7"], instance_key="generation0|8")}
                elif case == "nonfinite": altered["response"]["states"][0]["bounding_box"]["extent"]["x"] = "nan"
                elif case == "foreign_owner": altered["response"]["states"][0]["actor_info"]["client_key"] = "foreign_owner"
                else: altered["actor_instances"]["7"]["instance_key"] = "generation1|7"
                self.fixture(snapshots=[template, altered])
                result = self.load()
                self.assertTrue(result["instances"])
                self.assertTrue(all(row["valid"] is False for row in result["instances"].values()))
                self.episode = saved


if __name__ == "__main__":
    unittest.main()
