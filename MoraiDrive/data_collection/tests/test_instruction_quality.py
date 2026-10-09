import copy

from MoraiDrive.data_collection.situation_labeling.instruction_quality import finalize_instruction_quality
from MoraiDrive.data_collection.situation_labeling.mapper import attach_rule_instructions


def sample(state="normal", *, inventory=True):
    frame = dict(index=1, zone_id="dynamic_obstacle", valid=True,
                 dataset_build_candidate=True, object_inventory_usable=inventory)
    row = dict(index=1, zone_id="dynamic_obstacle", status="matched",
               instruction_state=state, variables={}, unknown_reasons=[],
               instruction_consistency="consistent")
    attach_rule_instructions([row])
    return frame, row


def test_valid_normal_retains_typed_target():
    frame, row = sample()
    finalize_instruction_quality([frame], [row])
    assert row["instruction"] == "Continue driving."
    assert row["action_valid"] and row["observation_valid"]


def test_copied_normal_over_unusable_inventory_is_quarantined():
    frame, row = sample(inventory=False)
    row.update(status="unknown", instruction_state=None,
               instruction_gap_filled_from=0, unknown_reasons=["object_inventory_unusable"])
    finalize_instruction_quality([frame], [row])
    assert row["instruction"] is None
    assert row["instruction_rejected"]["instruction"] == "Continue driving."
    assert not row["action_valid"] and not row["observation_valid"]


def test_smoothing_cannot_leave_waiting_state_with_yielding_target():
    frame, row = sample("yielding")
    row.update(instruction_state="waiting", instruction_id="dynamic_yielding",
               instruction="There is a pedestrian to the left of the ego vehicle. Yield to a pedestrian.",
               instruction_variables=dict(object="a pedestrian", position="to the left of"))
    finalize_instruction_quality([frame], [row])
    assert not row["action_valid"] and row["instruction"] is None
    assert "instruction_state_or_source_disagrees" in row["instruction_quality"]["action_exclusion_reasons"]


def test_same_template_with_incorrect_slot_text_is_not_accepted():
    frame, row = sample()
    row["instruction"] = "Resume driving."
    finalize_instruction_quality([frame], [row])
    assert row["instruction"] is None and not row["observation_valid"]


def test_upstream_semantic_mask_is_preserved():
    frame, row = sample()
    row.update(action_valid=False, observation_valid=False)
    row["variables"]["target_semantic_valid"] = False
    finalize_instruction_quality([frame], [row])
    assert row["instruction"] is None
    assert not row["action_valid"] and not row["observation_valid"]


def test_future_missing_or_borderline_cannot_be_promoted_by_manual_override():
    for consistency in ("no_future", "consistent_borderline", "no_ego_track"):
        frame, row = sample()
        row.update(instruction_consistency=consistency)
        finalize_instruction_quality([frame], [row])
        assert not row["action_valid"] and row["instruction"] is None


def test_roundabout_expert_target_is_valid_independent_of_unknown_rule_state():
    frame, row = sample()
    frame["zone_id"] = "roundabout"
    row.update(zone_id="roundabout", instruction_state=None, status="unknown",
               expert_state="CRUISE_AFTER_MERGE", label="cruise", instruction_id="cruise_normal",
               instruction="Continue driving.", instruction_variables={},
               instruction_source="expert_observation", expert_alignment={"status": "matched"})
    finalize_instruction_quality([frame], [row])
    assert row["action_valid"] and row["instruction"] == "Continue driving."


def test_stale_held_expert_target_is_not_an_independent_current_observation():
    frame, row = sample()
    row["evidence"] = {"instruction_held_from_index": 0}
    finalize_instruction_quality([frame], [row])
    assert not row["action_valid"] and row["instruction"] is None


def test_duplicate_frame_indices_are_rejected():
    import pytest
    frame, row = sample()
    with pytest.raises(ValueError, match="Duplicate"):
        finalize_instruction_quality([frame, copy.deepcopy(frame)], [row])


def test_numeric_distance_slot_validated_without_rejecting_valid_numbers():
    from MoraiDrive.data_collection.situation_labeling.instruction_quality import _valid_slot
    assert _valid_slot("distance", 12)
    assert _valid_slot("distance", 0.5)
    assert not _valid_slot("distance", float("nan"))
    assert not _valid_slot("distance", True)
    assert not _valid_slot("direction", " ")
