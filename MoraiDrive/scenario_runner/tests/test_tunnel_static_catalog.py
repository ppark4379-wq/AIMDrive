"""Catalog evidence gates and pivot-aware conservative vehicle geometry."""

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from ..scenarios.full_loop_parts.tunnel_static_parts import catalog


class TunnelCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("registry.json", "profiles.json", "confirmation.json",
                     "camera_review.json", "runtime_api_contracts.json"):
            shutil.copyfile(catalog.DEFAULT_VEHICLE_REGISTRY.parent / name, self.root / name)
        self.registry = self.root / "registry.json"
        self.document = json.loads(self.registry.read_text())

    def write_registry(self):
        self.registry.write_text(json.dumps(self.document), encoding="utf-8")

    def load(self):
        return catalog.load_vehicle_models(self.registry, ["DEFAULT_SEDAN"])

    def test_only_explicit_native_reviewed_vehicle_profiles_are_selected(self):
        models = catalog.load_vehicle_models()
        self.assertEqual(len(models), 36)
        self.assertIn("DEFAULT_SEDAN", models)
        self.assertIn("2023_Hyundai_Ioniq5", models)
        # An older registry approval alone is insufficient.
        self.assertNotIn("2013_Hyundai_Porter2", models)
        self.assertNotIn("Man1", models)
        self.assertEqual(set(self.load()), {"DEFAULT_SEDAN"})

    def test_profile_hashes_and_revision_are_recorded(self):
        source = self.load()["DEFAULT_SEDAN"]["geometry_source"]
        self.assertEqual(source["registry_sha256"], hashlib.sha256(self.registry.read_bytes()).hexdigest())
        for label, name in (("profiles", "profiles.json"), ("api", "runtime_api_contracts.json"),
                            ("confirmation", "confirmation.json"), ("camera", "camera_review.json")):
            self.assertEqual(source[label + "_sha256"], hashlib.sha256((self.root / name).read_bytes()).hexdigest())

    def test_default_source_labels_remain_compatible_with_saved_candidates(self):
        # Saved candidate/acceptance documents compare these portable paths
        # exactly; moving the zone implementation must not invalidate them.
        static_source = catalog.load_static_models(allowed_models=["WoodBox"])["WoodBox"]["geometry_source"]
        vehicle_source = catalog.load_vehicle_models(allowed_models=["DEFAULT_SEDAN"])["DEFAULT_SEDAN"]["geometry_source"]
        self.assertEqual(static_source["catalog_path"],
                         "scenario_runner/config/static_obstacle_footprints_v1.json")
        self.assertEqual(vehicle_source["registry_path"],
                         "data_collection/internal_sensor_record/evidence/"
                         "object_geometry_registry_default3d_v2/registry.json")

    def test_conservative_box_encloses_visible_surface_and_api_collider(self):
        item = self.load()["DEFAULT_SEDAN"]
        profile = self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"]
        # Undo command-frame yaw rotation to inspect the union in actor FLU.
        angle = math.radians(item["heading_offset_deg"])
        x, y = item["center_offset_spawn_xy"]
        center = (x * math.cos(angle) + y * math.sin(angle),
                  -x * math.sin(angle) + y * math.cos(angle))
        sizes = (item["length_m"], item["width_m"])
        for prefix in ("", "raw_"):
            original = profile["center_offset_flu_m" if not prefix else "raw_offset_flu_m"]
            original_size = profile[prefix + "size_lwh_m"]
            for axis in (0, 1):
                self.assertLessEqual(center[axis] - sizes[axis] / 2,
                                     original[axis] - original_size[axis] / 2 + 1e-12)
                self.assertGreaterEqual(center[axis] + sizes[axis] / 2,
                                        original[axis] + original_size[axis] / 2 - 1e-12)

    def test_command_spawn_offset_is_rotated_once_for_arbitrary_world_yaw(self):
        item = self.load()["DEFAULT_SEDAN"]
        offset = item["center_offset_spawn_xy"]
        delta = math.radians(item["heading_offset_deg"])
        actor_offset = (offset[0] * math.cos(delta) + offset[1] * math.sin(delta),
                        -offset[0] * math.sin(delta) + offset[1] * math.cos(delta))
        for requested_degrees in (-153.5, 0, 64, 179):
            requested = math.radians(requested_degrees)
            actual = requested + delta
            world_spawn = (offset[0] * math.cos(requested) - offset[1] * math.sin(requested),
                           offset[0] * math.sin(requested) + offset[1] * math.cos(requested))
            world_actor = (actor_offset[0] * math.cos(actual) - actor_offset[1] * math.sin(actual),
                           actor_offset[0] * math.sin(actual) + actor_offset[1] * math.cos(actual))
            for expected, observed in zip(world_actor, world_spawn):
                self.assertAlmostEqual(expected, observed, places=12)

    def test_each_explicit_approval_gate_is_required(self):
        original = copy.deepcopy(self.document)
        for gate in ("approved", "evidence_sufficient", "camera_review_pass", "lidar_pass", "motion_pose_review_pass"):
            with self.subTest(gate=gate):
                self.document = copy.deepcopy(original)
                item = self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]
                target = item if gate in ("approved", "evidence_sufficient") else item["geometry_3d"]
                target[gate] = False
                self.write_registry()
                self.assertEqual(self.load(), {})

    def test_scaled_and_nonvehicle_entries_are_excluded(self):
        item = self.document["models"].pop("npc|DEFAULT_SEDAN|1,1,1")
        self.document["models"]["npc|DEFAULT_SEDAN|2,2,2"] = item
        self.document["models"]["obstacle|DEFAULT_SEDAN|1,1,1"] = item
        self.write_registry()
        self.assertEqual(self.load(), {})

    def test_profile_rejects_nonfinite_size_and_fabricated_center(self):
        original = copy.deepcopy(self.document)
        for key, value in (("size_lwh_m", [math.nan, 2, 1]),
                           ("center_offset_flu_m", [0, 0, 0]),
                           ("raw_offset_flu_m", [0, 0, 0]),
                           ("scale_xyz", [2, 2, 2])):
            with self.subTest(key=key):
                self.document = copy.deepcopy(original)
                self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"][key] = value
                self.write_registry()
                self.assertEqual(self.load(), {})

    def test_changed_camera_file_fails_before_using_catalog(self):
        path = self.root / "camera_review.json"
        review = json.loads(path.read_text())
        review["models"]["DEFAULT_SEDAN"]["pass"] = False
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_profile_evidence_hash_mismatch_excludes_model(self):
        self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"]["evidence_sha256"] = "0" * 64
        self.write_registry()
        self.assertEqual(self.load(), {})

    def test_live_raw_contract_is_available_for_preview(self):
        item = self.load()["DEFAULT_SEDAN"]
        profile = self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"]
        self.assertEqual(item["expected_api_raw"]["size_lwh_m"], profile["raw_size_lwh_m"])
        self.assertEqual(item["expected_api_raw"]["center_offset_actor_flu_m"], profile["raw_offset_flu_m"])
        self.assertIs(item["geometry_verified"], True)

    def test_static_catalog_preserves_measured_spawn_frame_offset(self):
        original = json.loads(catalog.DEFAULT_STATIC_CATALOG.read_text())
        self.assertEqual(len(catalog.load_static_models()), 25)
        item = catalog.load_static_models(allowed_models=["WoodBox"])["WoodBox"]
        expected = original["models"]["WoodBox"]
        self.assertEqual(item["center_offset_spawn_xy"], expected["center_offset_spawn_xy"])
        self.assertEqual(item["heading_offset_deg"], expected["heading_offset_deg"])
        self.assertIs(item["geometry_verified"], True)

    def test_static_catalog_rejects_unreviewed_and_invalid_dimensions(self):
        document = json.loads(catalog.DEFAULT_STATIC_CATALOG.read_text())
        document["models"]["WoodBox"]["observations"] = 0
        document["models"]["NCAP_GVT"]["width_m"] = 0
        path = self.root / "static.json"
        path.write_text(json.dumps(document))
        self.assertEqual(catalog.load_static_models(path, ["WoodBox", "NCAP_GVT"]), {})


if __name__ == "__main__":
    unittest.main()
