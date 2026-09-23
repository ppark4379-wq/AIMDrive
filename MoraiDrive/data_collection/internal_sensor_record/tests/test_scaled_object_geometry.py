from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from morai_pretraining.scaled_object_geometry import (
    check_scaled_object_geometry, match_static_instances, resolve_pinned_preset,
    scaled_reference,
)
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


if __name__ == "__main__":
    unittest.main()
