"""Rendered roundabout positions retain the identity of the selected actor."""
from copy import deepcopy
import math

import pytest

from MoraiDrive.data_collection.situation_labeling.expert_instructions import (
    apply_expert_instructions,
)
from MoraiDrive.data_collection.situation_labeling.legacy_roundabout_guard import (
    ACTOR_REASON, PHASE_REASON, legacy_roundabout_action_reasons,
)


def example(phase="yield", source="explicit_ros"):
    frame = dict(index=91, zone_id="roundabout", lidar_source_ns=10_000_000_000,
        valid=True, dataset_build_candidate=True, object_inventory_usable=True,
        ego_world_x_m=0.0, ego_world_y_m=0.0, ego_heading_deg=90.0,
        ego_speed_mps=3.0, ego_speed_source="backward_gt_ego_xy_difference",
        roundabout_phase="entry", lead_actor_id="3", conflict_actor_id="3")
    rule = dict(index=91, zone_id="roundabout", status="matched",
        instruction_state="yielding", unknown_reasons=[], related_actors=["3"],
        variables=dict(target_actor_id="3", target_kind="vehicle",
            target_distance_m=math.hypot(30, 15), target_side="front",
            roundabout_phase="entry"),
        evidence=dict(target_actor_ref=dict(actor_id="3", instance_key="vehicle:3")))
    def vehicle(actor_id, x, y):
        return dict(index=91, lidar_source_ns=frame["lidar_source_ns"],
            actor_id=actor_id, instance_key="vehicle:" + actor_id, kind="npc",
            rule_geometry_usable=True, model_identity_verified=True,
            semantic_class_valid=True, semantic_class="vehicle",
            world_x_m=x, world_y_m=y)
    # A 90-degree Ego heading puts actor 3 to the right and actor 9 to the left.
    objects = [vehicle("3", 15.0, 30.0), vehicle("9", -18.0, 16.0)]
    raw = dict(mode="roundabout_entry", roundabout_phase=phase,
        roundabout_front_id="9", roundabout_conflict_id="9", source=source,
        ego_speed_mps=3.0, planner_target_speed_mps=2.0)
    if source == "existing_roundabout_text_log":
        raw["legacy_transition_observation"] = dict(
            schema="roundabout_transition_observation_v1", frame_index=91,
            phase_within_anchor_bounds=False, numeric_target_current_verified=False)
    return frame, rule, objects, raw


@pytest.mark.parametrize("phase", ["yield", "follow", "collision_slowdown"])
def test_verified_selected_actor_binds_all_final_target_fields(phase):
    frame, rule, objects, raw = example(phase)
    original = deepcopy(rule)
    source_objects = deepcopy(objects)
    apply_expert_instructions([frame], [rule], None, [],
        index_observations={91: raw}, objects=objects)
    assert "to the left of" in rule["instruction"]
    assert rule["instruction_variables"]["position"] == "to the left of"
    assert rule["variables"] == dict(original["variables"],
        target_actor_id="9", target_kind="vehicle",
        target_distance_m=math.hypot(18, 16), target_side="left",
        target_instance_key="vehicle:9")
    assert rule["related_actors"] == ["9"]
    evidence = rule["evidence"]
    assert evidence["target_actor_ref"] == dict(actor_id="9", instance_key="vehicle:9")
    assert evidence["roundabout_rule_target_provenance"] == dict(
        variables=original["variables"], related_actors=original["related_actors"],
        target_actor_ref=original["evidence"]["target_actor_ref"])
    assert evidence["roundabout_expert_target"]["current_actor_id"] == "3"
    assert evidence["roundabout_expert_target"]["geometric_target_matches"] is False
    assert evidence["roundabout_selected_target_binding"]["uses_future_ego"] is False
    assert objects == source_objects
    assert rule["instruction_state"] == original["instruction_state"]
    assert rule["status"] == original["status"]
    rule["variables"]["roundabout_phase"] = "changed"
    assert evidence["roundabout_rule_target_provenance"]["variables"]["roundabout_phase"] == "entry"


@pytest.mark.parametrize("invalid", ["stale_index", "stale_stamp", "identity",
                                    "geometry", "instance", "duplicate", "missing"])
def test_unverified_selected_actor_does_not_rebind_or_render(invalid):
    frame, rule, objects, raw = example()
    if invalid == "stale_index":
        objects[1]["index"] -= 1
    elif invalid == "stale_stamp":
        objects[1]["lidar_source_ns"] -= 100_000_000
    elif invalid == "identity":
        objects[1]["model_identity_verified"] = False
    elif invalid == "geometry":
        objects[1]["rule_geometry_usable"] = False
    elif invalid == "instance":
        objects[1]["instance_key"] = None
    elif invalid == "duplicate":
        objects.append(deepcopy(objects[1]))
    else:
        objects.pop()
    original = deepcopy(rule)
    apply_expert_instructions([frame], [rule], None, [],
        index_observations={91: raw}, objects=objects)
    assert rule["instruction"] is None
    assert rule["variables"] == original["variables"]
    assert rule["related_actors"] == original["related_actors"]
    assert rule["evidence"]["target_actor_ref"] == original["evidence"]["target_actor_ref"]
    assert "roundabout_selected_target_binding" not in rule["evidence"]
    assert "roundabout_rule_target_provenance" not in rule["evidence"]


def test_descriptive_rebinding_does_not_certify_legacy_phase_or_erase_cause_mismatch():
    frame, rule, objects, raw = example(source="existing_roundabout_text_log")
    original_transition = deepcopy(raw["legacy_transition_observation"])
    apply_expert_instructions([frame], [rule], None, [],
        index_observations={91: raw}, objects=objects)
    assert rule["variables"]["target_actor_id"] == "9"
    assert rule["expert_raw_state"]["legacy_transition_observation"] == original_transition
    assert rule["evidence"]["roundabout_selected_target_binding"]["verifies_legacy_phase_time"] is False
    assert legacy_roundabout_action_reasons(rule) == sorted([PHASE_REASON, ACTOR_REASON])


def test_verified_selected_actor_populates_a_missing_original_binding():
    frame, rule, objects, raw = example()
    rule["variables"] = {"roundabout_phase": "entry"}
    rule["related_actors"] = []
    rule["evidence"].pop("target_actor_ref")
    apply_expert_instructions([frame], [rule], None, [],
        index_observations={91: raw}, objects=objects)
    assert rule["variables"]["target_actor_id"] == "9"
    assert rule["related_actors"] == ["9"]
    assert rule["evidence"]["target_actor_ref"] == dict(actor_id="9", instance_key="vehicle:9")
    assert rule["evidence"]["roundabout_rule_target_provenance"] == dict(
        variables={"roundabout_phase": "entry"}, related_actors=[], target_actor_ref=None)
