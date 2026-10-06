from MoraiDrive.data_collection.situation_labeling.expert_instructions import (
    apply_expert_instructions, canonical_state, load_config,
)
from MoraiDrive.data_collection.situation_labeling.mapper import attach_rule_instructions


def test_restored_speed_conditions_and_expert_priority():
    mapping, _ = load_config()
    for phase, speed, target, expected in (
        ("follow", 5.0, 4.0, "follow_decelerate"),
        ("follow", 0.0, 0.0, "follow_stopped"),
        ("yield", 0.0, 0.0, "yield_stopped"),
        ("collision_slowdown", 0.0, 0.0, "collision_stopped"),
        ("emergency", 0.0, 0.0, "emergency_stopped"),
    ):
        raw = dict(mode="roundabout_entry", roundabout_phase=phase,
                   ego_speed_mps=speed, planner_target_speed_mps=target)
        entry, reason = canonical_state("roundabout", raw, mapping)
        assert reason is None and entry["instruction_id"] == expected
        row = dict(zone_id="roundabout", status="matched", instruction_state="follow",
                   expert_state=entry["state"], label=entry["label"],
                   instruction_id=expected, instruction="Expert-selected sentence")
        attach_rule_instructions([row])
        assert row["instruction_id"] == expected
        assert row["instruction"] == "Expert-selected sentence"


def test_rear_pressure_preserves_current_frame_position():
    frame = dict(index=1, zone_id="roundabout", lidar_source_ns=1)
    row = dict(index=1, zone_id="roundabout", status="matched",
               instruction_state="follow", variables={"target_side": "front"})
    raw = dict(mode="roundabout_entry", roundabout_phase="rear_pressure",
               roundabout_conflict_id=42)
    objects = [dict(index=1, actor_id=42, rule_geometry_usable=True,
                    forward_m=-8.0, left_m=0.0)]
    apply_expert_instructions([frame], [row], None, [],
                             index_observations={1: raw}, objects=objects)
    attach_rule_instructions([row])
    assert row["instruction_id"] == "rear_pressure"
    assert row["instruction"] == (
        "There is a vehicle behind the ego vehicle. Maintain the current driving behavior.")
