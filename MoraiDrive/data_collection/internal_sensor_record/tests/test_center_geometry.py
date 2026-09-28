from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from morai_pretraining.center_geometry import (
    catalog_return_support, check_center_object_geometry, horizontal_scale_reference,
    pivot_local_offset, resolve_training_center, summarize_pivot_offsets,
)
from morai_pretraining.sensor_record_manifest import coverage_failed_splits, objects_for_entry
from audit_sensor_record_object_geometry_v2 import check_object_geometry


def catalog(scale=(1., 1., 1.), offset=(.545, -.011, .587), returns=40, outside=.03):
    return {"kind": "static", "scale": list(scale), "visible_return_contradiction": False,
            "views": [{"view": index, "scale": list(scale), "dimensions_lwh_m": [1.7, .88, 1.19],
                       "local_center_offset_m": list(offset), "foreground_returns": returns,
                       "outside_box_fraction": outside} for index in range(4)]}


def scale_evidence(scale=(1., 1., 1.)):
    return {"instances": {"instance": {
        "valid": True, "actor_id": "7", "model_name": "Scooter", "record_ids": [10, 11],
        "scale_xyz": list(scale), "preset_obstacle_index": 0}}}


def pivot_evidence(offset=(0., 0., 0.), valid=True):
    return {"instances": {"instance": {
        "valid": valid, "record_ids": [10, 11], "offset_xyz_m": list(offset),
        "reason": None if valid else "observed_pivot_center_offset_unstable"}}}


def item(heading=90., size=(.88, 1.7, 1.19), position=(10., 20., 1.)):
    return {"instance_key": "instance", "id": "7", "model_name": "Scooter", "kind": "obstacle",
            "size_wlh_m": list(size), "heading_deg": heading, "position": list(position)}


class CenterGeometryTest(unittest.TestCase):
    def check(self, *, case=None, obj=None, scale=None, pivot=None, source_ids=(10,)):
        return check_center_object_geometry(
            obj or item(), {"Scooter": case or catalog()}, "hash", .03, .02,
            scale or scale_evidence(), pivot or pivot_evidence(), list(source_ids),
            check_object_geometry)

    def test_pivot_reported_box_is_recentred_from_catalog_offset(self):
        result = self.check()
        self.assertTrue(result["api_box_geometry_candidate"])
        self.assertEqual(result["center_evidence"]["center_source"],
                         "catalog_offset_applied_to_pivot_reported_box")
        # Heading 90 deg: local forward is world +Y and local left is world -X.
        x, y = result["training_center_world_xy"]
        self.assertAlmostEqual(x, 10. + .011)
        self.assertAlmostEqual(y, 20. + .545)

    def test_consistent_api_centre_is_kept(self):
        result = self.check(pivot=pivot_evidence((.53, -.01, 0.)))
        self.assertTrue(result["api_box_geometry_candidate"])
        self.assertEqual(result["training_center_world_xy"], [10., 20.])
        self.assertEqual(result["center_evidence"]["center_source"],
                         "api_box_center_consistent_with_catalog")

    def test_unexplained_centre_disagreement_is_masked(self):
        result = self.check(pivot=pivot_evidence((.2, 0., 0.)))
        self.assertFalse(result["api_box_geometry_candidate"])
        self.assertIn("observed_center_offset_differs_from_catalog", result["reasons"])
        self.assertNotIn("training_center_world_xy", result)

    def test_unstable_or_unlinked_pivot_evidence_is_masked(self):
        self.assertFalse(self.check(pivot=pivot_evidence(valid=False))["api_box_geometry_candidate"])
        self.assertFalse(self.check(source_ids=(10, 12))["api_box_geometry_candidate"])
        self.assertFalse(self.check(pivot={"instances": {}})["api_box_geometry_candidate"])

    def test_catalog_returns_must_support_the_box(self):
        few = self.check(case=catalog(returns=4, outside=.75))
        self.assertIn("catalog_visible_returns_insufficient", few["reasons"])
        outside = self.check(case=catalog(returns=180, outside=.87))
        self.assertIn("catalog_visible_returns_outside_box", outside["reasons"])
        missing = catalog()
        missing["views"][2]["outside_box_fraction"] = None
        self.assertIn("catalog_visible_returns_outside_box",
                      catalog_return_support(missing)["reason"])

    def test_unstable_catalog_centre_is_masked(self):
        case = catalog()
        case["views"][1]["local_center_offset_m"][0] += .1
        self.assertIn("catalog_center_offset_unstable", self.check(case=case)["reasons"])

    def test_unity_y_up_scale_matches_woodbox_measurement(self):
        case = catalog(scale=(2., 3., 2.))
        for view in case["views"]:
            view["dimensions_lwh_m"] = [2.4, 2.4, 3.6]
        reference, evidence = horizontal_scale_reference(case, [1., 1., 1.])
        for actual, expected in zip(reference, [1.2, 1.2, 1.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(evidence["scale_policy"], "unity_y_up_equal_horizontal_scale_v1")
        self.assertIsNone(horizontal_scale_reference(case, [1., 1., 2.])[0])

    def test_scaled_catalog_offset_uses_horizontal_multiplier(self):
        case = catalog(scale=(2., 3., 2.), offset=(.2, 0., 1.8))
        for view in case["views"]:
            view["dimensions_lwh_m"] = [3.4, 1.76, 3.57]
        result = self.check(case=case, pivot=pivot_evidence((.1, 0., .6)))
        self.assertTrue(result["api_box_geometry_candidate"], result["reasons"])
        self.assertEqual(result["training_center_world_xy"], [10., 20.])

    def test_wrong_dimensions_still_fail_before_centre(self):
        result = self.check(obj=item(size=(1.7, .88, 1.19)))
        self.assertIn("observed_dimensions_differ_from_scaled_catalog", result["reasons"])
        self.assertNotIn("center_evidence", result)

    def test_correction_beyond_bev_margin_is_masked(self):
        center, evidence = resolve_training_center(item(), [1.5, 0.], [0., 0.])
        self.assertIsNone(center)
        self.assertEqual(evidence["reason"], "center_correction_exceeds_bev_margin")

    def test_pivot_offset_matches_catalog_convention(self):
        yaw = math.radians(-81.87)
        actor = {"transform": {"location": {"x": 0., "y": 0., "z": 0.},
                               "rotation": {"x": 0., "y": 0., "z": -81.87}},
                 "bounding_box": {"location": {"x": .5 * math.cos(yaw), "y": .5 * math.sin(yaw),
                                               "z": .3}}}
        for actual, expected in zip(pivot_local_offset(actor), [.5, 0., .3]):
            self.assertAlmostEqual(actual, expected)

    def test_pivot_summary_requires_rigid_offset(self):
        rows = [{"record_id": 1, "offset_xyz_m": [0., 0., 0.], "box_pivot_yaw_difference_deg": 0.},
                {"record_id": 2, "offset_xyz_m": [.02, 0., 0.], "box_pivot_yaw_difference_deg": 0.}]
        self.assertFalse(summarize_pivot_offsets({"k": rows})["k"]["valid"])
        rows[1]["offset_xyz_m"] = [.005, 0., 0.]
        self.assertTrue(summarize_pivot_offsets({"k": rows})["k"]["valid"])

    def test_manifest_uses_training_centre_only_when_present(self):
        gt = {"object_gt": {"objects": [dict(item(), velocity_kmh=[0., 0., 0.])]}}
        geometry = {"objects": [{"instance_key": "instance", "api_box_geometry_candidate": True,
                                 "training_center_world_xy": [10.011, 20.545]}]}
        self.assertEqual(objects_for_entry(gt, geometry)[0]["y"], 20.545)
        del geometry["objects"][0]["training_center_world_xy"]
        self.assertEqual(objects_for_entry(gt, geometry)[0]["y"], 20.)

    def test_only_coverage_failures_are_revisable(self):
        index = {"training_ready": False, "required_coverage": ["waypoint_frames", "object_positive_frames"],
                 "coverage": {"train": {"waypoint_frames": 5, "object_positive_frames": 3},
                              "val": {"waypoint_frames": 5, "object_positive_frames": 0}}}
        self.assertEqual(coverage_failed_splits(index), ["val"])
        self.assertEqual(coverage_failed_splits(dict(index, training_ready=True)), [])
        full = deepcopy(index)
        full["coverage"]["val"]["object_positive_frames"] = 2
        self.assertEqual(coverage_failed_splits(full), [])


if __name__ == "__main__":
    unittest.main()
