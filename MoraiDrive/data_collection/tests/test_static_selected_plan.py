import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from MoraiDrive.data_collection.situation_labeling.expert_static_phase import attach_static_selected_plan
from MoraiDrive.data_collection.situation_labeling.static_plan import (
    align_current_static_plan, compose_current_static_plan, validate_current_static_plan,
    attach_static_route_signals)
from MoraiDrive.data_collection.situation_labeling.mapper import attach_rule_instructions
from MoraiDrive.data_collection.situation_labeling.instruction_quality import finalize_instruction_quality


class StaticPlanRegression(unittest.TestCase):
    def setUp(self):
        self.frame = dict(index=414, lidar_source_ns=1_000_000_000, zone_id="static_obstacle",
            valid=True, dataset_build_candidate=True, object_inventory_usable=True,
            ego_speed_mps=0., ego_speed_source="backward_gt_ego_xy_difference",
            expert_static_selected_plan=dict(phase="follow", path_validated=True, selected_routes=["right_02.csv"],
                direction="right", anchor_interval_all_plans_agree=True, incoming_plan_agrees=True,
                source_sha256="plan", signal_zero_limit_all=True),
            expert_publications=dict(status="RECEIVE_BOUND_CURRENT_CONTEXT", sensor_ns=1_000_000_000,
                lidar_receive_monotonic_ns=1_000_000_000,
                control=dict(preceding_received_clock_ns=999_000_000, receive_monotonic_ns=999_000_000,
                    record_id=10, payload_sha256="c", longlCmdType=1, accel=0., brake=1.),
                target_velocity=dict(preceding_received_clock_ns=999_000_000, receive_monotonic_ns=999_000_001,
                    record_id=11, payload_sha256="t", value_mps=0.,
                    prior_zero_target=dict(preceding_received_clock_ns=900_000_000,
                        receive_monotonic_ns=900_000_000, payload_sha256="p", value_mps=0.))),
            static_plan_route_signal=dict(index=414, sensor_source_ns=1_000_000_000,
                valid=True, color=1, route_rpc_current_link_agreement=True, light_id="route-red",
                light_source_record_ids=[1, 2], light_source_bracket_ns=[990_000_000, 1_010_000_000],
                distance_to_stopline_m=2.4))
        self.actor = dict(index=414, actor_id="1", instance_key="fresh", kind="obstacle",
            rule_geometry_usable=True, model_identity_verified=True, semantic_class_valid=True)
        self.rule = dict(index=414, zone_id="static_obstacle", label="follow", status="matched",
            instruction_state="avoid_continue", action_valid=True, observation_valid=True, unknown_reasons=[],
            variables=dict(avoid_direction="right"), evidence=dict(target_actor_ref=dict(actor_id="1", instance_key="fresh")))

    def compose(self):
        return compose_current_static_plan(self.frame)

    def test_red_stop_retains_selected_right_plan(self):
        self.assertEqual(self.compose()[0]["state"], "static_signal_stop_plan")

    def test_route_red_without_signal_limiter_does_not_invent_stop_cause(self):
        self.frame["expert_static_selected_plan"]["signal_zero_limit_all"] = False
        result, _ = self.compose()
        self.assertEqual(result["state"], "static_signal_stop_plan")
        self.assertFalse(result["red_signal_cause_verified"])

    def test_invalid_route_does_not_invent_red_cause(self):
        self.frame["static_plan_route_signal"]["valid"] = False
        self.assertEqual(self.compose()[0]["state"], "static_stop_plan")

    def test_pending_positive_target_not_replaced_by_stopped_sentence(self):
        self.frame["expert_publications"]["target_velocity"]["value_mps"] = .12
        self.frame["expert_publications"]["control"]["brake"] = .6
        self.assertEqual(self.compose()[0]["state"], "static_prepare_resume_plan")

    def test_departure_plan_survives_below_point2_actual_speed(self):
        self.frame["expert_publications"]["target_velocity"]["value_mps"] = .44
        self.frame["expert_publications"]["control"].update(accel=.1, brake=0.)
        self.frame["ego_speed_mps"] = .008
        self.assertEqual(self.compose()[0]["state"], "static_resume_plan")

    def test_departure_without_recent_zero_target_preserves_original(self):
        self.frame["expert_publications"]["target_velocity"]["value_mps"] = 4.
        del self.frame["expert_publications"]["target_velocity"]["prior_zero_target"]
        self.assertEqual(self.compose()[1], "departure_requires_recent_zero_target")

    def test_future_receive_rejected(self):
        self.frame["expert_publications"]["control"]["receive_monotonic_ns"] = 1_000_000_001
        self.assertEqual(self.compose()[1], "current_expert_command_stale_or_missing")

    def test_stale_clock_rejected(self):
        self.frame["expert_publications"]["control"]["preceding_received_clock_ns"] = 1
        self.assertEqual(self.compose()[1], "current_expert_command_stale_or_missing")

    def test_other_longitudinal_command_mode_not_interpreted_as_throttle(self):
        self.frame["expert_publications"]["control"]["longlCmdType"] = 2
        self.assertEqual(self.compose()[1], "throttle_brake_command_mode_unverified")

    def test_wrong_frame_source_rejected(self):
        self.frame["expert_publications"]["sensor_ns"] = 1
        self.assertEqual(self.compose()[1], "current_expert_publications_unbound")

    def test_moving_with_zero_target_not_claimed_stopped(self):
        self.frame["ego_speed_mps"] = 1.
        self.assertIsNone(self.compose()[0])

    def test_no_upstream_mask_promotion(self):
        self.rule["action_valid"] = False
        before = copy.deepcopy(self.rule)
        align_current_static_plan([self.frame], [self.actor], [self.rule])
        self.assertEqual(self.rule, before)

    def test_no_actor_instance_copy(self):
        self.actor["instance_key"] = "different"
        align_current_static_plan([self.frame], [self.actor], [self.rule])
        self.assertEqual(self.rule["instruction_state"], "avoid_continue")

    def test_catalog_mapping_and_final_quality(self):
        align_current_static_plan([self.frame], [self.actor], [self.rule])
        attach_rule_instructions([self.rule])
        self.assertEqual(self.rule["instruction"], "Remain stopped at the red traffic light; the planned avoidance path goes to the right.")
        self.assertEqual(validate_current_static_plan([self.frame], [self.actor], self.rule), (True, []))
        self.rule["instruction_consistency"] = "consistent"
        report = finalize_instruction_quality([self.frame], [self.rule])
        self.assertEqual(report["counts"]["action_valid_frames"], 1)
        self.assertTrue(self.rule["action_valid"])

    def test_validator_rechecks_current_receipt_not_marker(self):
        align_current_static_plan([self.frame], [self.actor], [self.rule])
        attach_rule_instructions([self.rule])
        self.frame["expert_publications"]["control"]["brake"] = 0.
        self.assertFalse(validate_current_static_plan([self.frame], [self.actor], self.rule)[0])

    def test_plan_transition_inside_anchor_interval_not_current_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "supervisor.log"
            path.write_text("[INFO] [1.0, 0.0] start\n"
                "[STATIC] {'phase': 'follow', 'path_validated': True, 'selected_routes': ['right_02.csv']}\n"
                "[STATIC] {'phase': 'follow', 'path_validated': True, 'selected_routes': ['left_02.csv']}\n"
                "[INFO] [2.0, 2.0] end\n")
            self.frame.pop("expert_static_selected_plan")
            report = attach_static_selected_plan([self.frame], path)
            self.assertNotIn("expert_static_selected_plan", self.frame)
            self.assertEqual(report["selected_plan_changes_or_missing"], 1)

    def test_malformed_log_is_plan_discontinuity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "supervisor.log"
            path.write_text("[INFO] [1.0, 0.0] start\n"
                "[STATIC] {'phase': 'follow', 'path_validated': True, 'selected_routes': ['right_02.csv']}\n"
                "[STATIC] {bad\n[INFO] [2.0, 2.0] end\n")
            self.frame.pop("expert_static_selected_plan")
            report = attach_static_selected_plan([self.frame], path)
            self.assertEqual(report["invalid_dictionary_lines"], 1)
            self.assertNotIn("expert_static_selected_plan", self.frame)

    def _bind_interval(self, incoming):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "supervisor.log"
            path.write_text(incoming + "[INFO] [1.0, 0.0] start\n"
                "[STATIC] {'phase': 'follow', 'path_validated': True, 'selected_routes': ['right_02.csv']}\n"
                "[INFO] [2.0, 2.0] end\n")
            return attach_static_selected_plan([self.frame], path)

    def test_future_new_plan_cannot_cover_interval_before_its_first_dump(self):
        report = self._bind_interval("[STATIC] {'phase': 'follow', 'path_validated': True, 'selected_routes': ['left_02.csv']}\n")
        self.assertNotIn("expert_static_selected_plan", self.frame)
        self.assertEqual(report["incoming_selected_plan_unverified_or_changed"], 1)

    def test_missing_incoming_plan_is_not_assumed_unchanged(self):
        report = self._bind_interval("")
        self.assertNotIn("expert_static_selected_plan", self.frame)
        self.assertEqual(report["incoming_selected_plan_unverified_or_changed"], 1)

    def test_unchanged_incoming_and_interval_plans_are_bound(self):
        report = self._bind_interval("[STATIC] {'phase': 'follow', 'path_validated': True, 'selected_routes': ['right_02.csv']}\n")
        self.assertEqual(report["bound_frames"], 1)
        self.assertTrue(self.frame["expert_static_selected_plan"]["incoming_plan_agrees"])

    def test_malformed_incoming_plan_cannot_be_skipped(self):
        report = self._bind_interval("[STATIC] {bad\n")
        self.assertNotIn("expert_static_selected_plan", self.frame)
        self.assertEqual(report["incoming_selected_plan_unverified_or_changed"], 1)


if __name__ == "__main__":
    unittest.main()
