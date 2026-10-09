import copy
import unittest

from MoraiDrive.data_collection.situation_labeling.legacy_roundabout_guard import (
    ACTOR_REASON, CONFLICT_REASON, PHASE_REASON, legacy_roundabout_action_reasons,
)


class LegacyRoundaboutGuardTests(unittest.TestCase):
    def rule(self):
        return dict(index=141, zone_id="roundabout", instruction_source="expert_observation",
            instruction_id="collision_slowdown", status="matched", unknown_reasons=[],
            expert_raw_state=dict(source="existing_roundabout_text_log",
                legacy_transition_observation=dict(schema="roundabout_transition_observation_v1",
                    frame_index=141, phase_within_anchor_bounds=False,
                    numeric_target_current_verified=False)),
            evidence=dict(roundabout_expert_target=dict(valid=True,
                selected_actor_id="9", current_actor_id="3", geometric_target_matches=False)))

    def test_mismatched_actor_with_unbounded_phase_is_excluded_without_mutation(self):
        rule = self.rule()
        original = copy.deepcopy(rule)
        self.assertEqual(legacy_roundabout_action_reasons(rule), sorted([PHASE_REASON, ACTOR_REASON]))
        self.assertEqual(rule, original)

    def test_unknown_conflict_response_is_excluded(self):
        rule = self.rule()
        rule.update(instruction_id="yield_moving", status="unknown",
                    unknown_reasons=["conflict_response_unverified"])
        rule["evidence"]["roundabout_expert_target"].update(
            selected_actor_id="1", current_actor_id="1", geometric_target_matches=True)
        self.assertEqual(legacy_roundabout_action_reasons(rule), sorted([PHASE_REASON, CONFLICT_REASON]))

    def test_unbounded_phase_without_mismatch_or_unknown_response_is_preserved(self):
        rule = self.rule()
        rule["evidence"]["roundabout_expert_target"].update(
            selected_actor_id="9", current_actor_id=9, geometric_target_matches=True)
        self.assertEqual(legacy_roundabout_action_reasons(rule), [])

    def test_unknown_numeric_target_alone_is_preserved(self):
        rule = self.rule()
        rule["expert_raw_state"]["legacy_transition_observation"]["phase_within_anchor_bounds"] = True
        self.assertEqual(legacy_roundabout_action_reasons(rule), [])

    def test_current_path_fallback_does_not_depend_on_legacy_actor_phase(self):
        for identifier in ("current_plan_drive", "current_plan_slowdown", "current_plan_stop",
                           "current_departure_plan"):
            rule = self.rule()
            rule["instruction_id"] = identifier
            self.assertEqual(legacy_roundabout_action_reasons(rule), [])

    def test_current_response_requires_independently_recomputed_proof(self):
        rule = self.rule()
        rule["evidence"]["current_roundabout_cruise"] = dict(
            schema="roundabout_current_cruise_v1", valid=True, uses_future_ego=False,
            supervision_basis="recorded_current_roundabout_motion")
        self.assertTrue(legacy_roundabout_action_reasons(rule))
        rule["evidence"]["current_phase_consistency"] = dict(valid=True,
            recomputed_from_current_evidence=True, uses_future_ego=False)
        self.assertEqual(legacy_roundabout_action_reasons(rule), [])
        rule["evidence"]["current_phase_consistency"]["valid"] = False
        self.assertTrue(legacy_roundabout_action_reasons(rule))

    def test_marker_valid_boolean_without_provenance_does_not_exempt(self):
        rule = self.rule()
        rule["evidence"].update(current_roundabout_cruise={"valid": True},
            current_phase_consistency={"valid": True})
        self.assertTrue(legacy_roundabout_action_reasons(rule))

    def test_wrong_frame_proof_and_other_zones_are_preserved(self):
        rule = self.rule()
        rule["expert_raw_state"]["legacy_transition_observation"]["frame_index"] = 140
        self.assertEqual(legacy_roundabout_action_reasons(rule), [])
        rule = self.rule()
        rule["zone_id"] = "static_obstacle"
        self.assertEqual(legacy_roundabout_action_reasons(rule), [])


if __name__ == "__main__":
    unittest.main()
