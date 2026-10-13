"""Meaningful geometry and input isolation checks for the offline VE bridge."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "VLDrive"))
from tools.evaluation.ve_expert_planning_engine import (
    evaluate_planning_pair, signed_polygon_clearance, _vertices)


def box(x, y=0.0, length=1.0, width=1.0, source="gt", heading=0.0):
    return dict(center_world=[x, y], heading_deg=heading, length_m=length,
                width_m=width, speed_mps=None, track_id=None, kind="unknown",
                is_static=None, confidence=1.0, source=source)


def frame(gt_boxes=None, ve_boxes=None):
    return dict(ego_pose=dict(x=0.0, y=0.0, yaw=0.0), speed_mps=5.0,
                official_reference_path=[[float(x), 0.0] for x in range(81)],
                reference_path_source="official_global_path",
                gt_world_boxes=gt_boxes or [], ve_world_boxes=ve_boxes or [])


class LatticePairTest(unittest.TestCase):
    def test_gt_obstacle_missed_by_ve_exposes_collision(self):
        result = evaluate_planning_pair(frame([box(12.0)], []))
        self.assertTrue(result["generic_gt"]["active"])
        self.assertFalse(result["generic_ve"]["active"])
        self.assertFalse(result["comparison"]["decision_agreement"])
        self.assertFalse(result["comparison"]["gt_path_collision"])
        self.assertTrue(result["comparison"]["ve_path_collision"])
        self.assertGreater(result["comparison"]["path_mean_separation_m"], 0.3)

    def test_same_boxes_yield_same_path_without_mutating_inputs(self):
        value = frame([box(12.0)], [box(12.0, source="ve")])
        original = copy.deepcopy(value)
        result = evaluate_planning_pair(value)
        self.assertEqual(value, original)
        self.assertTrue(result["comparison"]["decision_agreement"])
        self.assertEqual(result["comparison"]["path_max_separation_m"], 0.0)
        self.assertEqual(result["generic_gt"]["path"], result["generic_ve"]["path"])

    def test_false_positive_changes_path_on_clear_road(self):
        result = evaluate_planning_pair(frame([], [box(12.0, source="ve")]))
        self.assertFalse(result["generic_gt"]["active"])
        self.assertTrue(result["generic_ve"]["active"])
        self.assertGreater(result["comparison"]["path_max_separation_m"], 1.0)
        self.assertFalse(result["generic_ve"]["gt_physical_footprint"]["collision"])
        self.assertIsNone(result["generic_ve"]["gt_physical_footprint"]["minimum_clearance_m"])

    def test_all_candidates_blocked_does_not_claim_control_stop(self):
        result = evaluate_planning_pair(frame([box(12.0, width=100.0)],
            [box(12.0, width=100.0, source="ve")]))
        self.assertEqual(result["generic_ve"]["decision"], "all_candidates_blocked")
        self.assertFalse(result["limitations"]["full_autonomous_driving_validated"])
        self.assertNotIn("target_velocity", result["generic_ve"])

    def test_unknown_classes_and_velocities_stay_geometry_only(self):
        result = evaluate_planning_pair(frame([box(12.0)], [box(12.0, source="ve")]))
        self.assertEqual(result["input_coverage"]["unknown_ve_categories"], 1)
        self.assertEqual(result["input_coverage"]["unknown_ve_velocities"], 1)
        self.assertFalse(result["limitations"]["class_and_velocity_used"])
        self.assertFalse(result["limitations"]["dynamic_time_prediction_validated"])

    def test_upstream_small_object_clamp_is_exposed(self):
        result = evaluate_planning_pair(frame([box(12.0, length=.3, width=.3)], []))
        self.assertEqual(result["input_coverage"]["clamped_gt_boxes"], 1)
        self.assertEqual(result["planner_provenance"]["obstacle_minimum_length_width_m"], 1.0)
        self.assertNotEqual(result["generic_ve"]["gt_physical_footprint"]["minimum_clearance_m"],
                            result["generic_ve"]["gt_planner_clamped_footprint"]["minimum_clearance_m"])

    def test_pose_translation_and_rotation_preserve_decision(self):
        base = frame([box(12.0)], [box(12.0, source="ve")])
        result = evaluate_planning_pair(base)
        translated = copy.deepcopy(base)
        translated["ego_pose"] = dict(x=302595.0, y=4124145.0, yaw=math.pi / 2)
        translated["official_reference_path"] = [[302595.0, 4124145.0 + x]
                                                 for x, _ in base["official_reference_path"]]
        for source in ("gt_world_boxes", "ve_world_boxes"):
            translated[source][0].update(center_world=[302595.0, 4124157.0], heading_deg=90.0)
        rotated = evaluate_planning_pair(translated)
        self.assertEqual(result["generic_ve"]["reason"], rotated["generic_ve"]["reason"])
        self.assertEqual(result["generic_ve"]["selected_offset_m"], rotated["generic_ve"]["selected_offset_m"])
        self.assertLess(rotated["route_ego_yaw_error_rad"], 1e-7)

    def test_official_gp_cropped_at_ego_not_future_gt(self):
        value = frame()
        value["ego_pose"]["x"] = 10.25
        result = evaluate_planning_pair(value)
        self.assertEqual(result["generic_ve"]["path"][0], [10.25, 0.0])
        self.assertAlmostEqual(result["reference_path_available_m"], 40.0)
        value["reference_path_source"] = "future_gt_waypoints"
        with self.assertRaises(ValueError):
            evaluate_planning_pair(value)

    def test_explicit_closed_gp_wraps_at_seam(self):
        value = frame()
        value.update(official_reference_path=[[0.0, 0.0], [40.0, 0.0],
                                              [40.0, 40.0], [0.0, 40.0]],
                     official_reference_closed=True)
        value["ego_pose"] = dict(x=0.0, y=1.0, yaw=-math.pi / 2)
        result = evaluate_planning_pair(value)
        self.assertAlmostEqual(result["reference_path_available_m"], 40.0)
        self.assertEqual(result["generic_ve"]["path"][0], [0.0, 1.0])
        self.assertTrue(result["reference_path_closed"])

    def test_route_index_disambiguates_repeated_location(self):
        value = frame()
        value["official_reference_path"] = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
            [30.0, 0.0], [40.0, 0.0], [50.0, 0.0], [60.0, 0.0], [60.0, 5.0],
            [50.0, 5.0], [40.0, 5.0], [30.0, 5.0], [20.0, 5.0], [10.0, 5.0],
            [0.0, 5.0], [0.0, 0.0], [0.0, -10.0], [0.0, -20.0], [0.0, -40.0]]
        value["reference_path_start_index"] = 15
        value["reference_path_index_source"] = "official_route_index"
        value["ego_pose"]["yaw"] = -math.pi / 2
        result = evaluate_planning_pair(value)
        self.assertGreaterEqual(result["route_closest_segment_index"], 12)
        self.assertEqual(result["generic_ve"]["path"][-1], [0.0, -40.0])
        self.assertEqual(result["reference_occurrence_selection"], "official_route_index_neighborhood")

    def test_route_index_requires_official_provenance(self):
        value = frame()
        value["reference_path_start_index"] = 12
        with self.assertRaises(ValueError):
            evaluate_planning_pair(value)

    def test_wrong_source_or_invalid_geometry_rejected(self):
        for changes in (dict(source="gt"), dict(length_m=-1.0),
                        dict(width_m=float("nan")), dict(speed_mps=-1.0),
                        dict(center_world=[1.0]), dict(confidence=1.2)):
            detected = box(12.0, source="ve")
            detected.update(changes)
            with self.assertRaises(ValueError):
                evaluate_planning_pair(frame([], [detected]))

    def test_import_and_plan_do_not_load_network_or_runtime_config(self):
        code = """import sys,json
from tools.evaluation.ve_expert_planning_engine import evaluate_planning_pair
evaluate_planning_pair(json.loads(sys.argv[1]))
bad=[key for key in sys.modules if key=='rospy' or key=='grpc' or
 key.endswith('.network.ros_manager') or key.endswith('.config.config')]
assert not bad,bad
"""
        process = subprocess.run([sys.executable, "-c", code, json.dumps(frame())],
                                 cwd=str(REPOSITORY / "VLDrive"), capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


class PolygonClearanceTest(unittest.TestCase):
    def test_axis_aligned_separation_and_overlap(self):
        first = _vertices((0, 0), 0, 2, 2)
        self.assertAlmostEqual(signed_polygon_clearance(first, _vertices((4, 0), 0, 2, 2)), 2.0)
        self.assertLess(signed_polygon_clearance(first, _vertices((1, 0), 0, 2, 2)), 0.0)
        self.assertAlmostEqual(signed_polygon_clearance(first, _vertices((2, 0), 0, 2, 2)), 0.0)

    def test_rotated_thin_obstacle_uses_polygon_not_center_distance(self):
        vehicle = _vertices((0, 0), 0, 4, 2)
        across = _vertices((0, 2.5), math.pi / 2, 4, .3)
        along = _vertices((0, 2.5), 0, 4, .3)
        self.assertLess(signed_polygon_clearance(vehicle, across), 0.0)
        self.assertGreater(signed_polygon_clearance(vehicle, along), 1.0)


if __name__ == "__main__":
    unittest.main()
