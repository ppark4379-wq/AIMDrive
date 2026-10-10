"""Current-command checks must survive the final consistency and quality passes."""
import pytest

from MoraiDrive.data_collection.tests.test_departure_plan import case
from MoraiDrive.data_collection.situation_labeling.consistency import drop_contradicted_instructions
from MoraiDrive.data_collection.situation_labeling.departure_plan import align_current_departure_plans
from MoraiDrive.data_collection.situation_labeling.instruction_quality import finalize_instruction_quality


def final_quality(frames, rules, *, objects=None):
    # Final quality receives the complete past track in the official pipeline.
    # These extra rows carry no fabricated instruction targets.
    earlier = [dict(index=f["index"], zone_id=f["zone_id"], instruction=None)
               for f in frames if f["index"] not in {r["index"] for r in rules}]
    return finalize_instruction_quality(frames, earlier + rules, objects=objects)


@pytest.mark.parametrize("kind", ["response", "normal", "roundabout"])
def test_fresh_departure_reaches_final_training_target_without_future_motion(kind):
    frames, objects, rules = case(kind)
    align_current_departure_plans(frames, objects, rules)
    drop_contradicted_instructions(frames, rules, objects=objects)
    final_quality(frames, rules, objects=objects)
    rule = rules[0]
    assert rule["instruction"] == "Resume driving."
    assert rule["action_valid"] and rule["observation_valid"]
    assert rule["evidence"]["current_phase_consistency"]["recomputed_from_current_expert_commands"]


def test_changed_receipt_is_quarantined_by_consistency_and_final_quality():
    frames, objects, rules = case()
    align_current_departure_plans(frames, objects, rules)
    frames[-1]["expert_publications"]["control"]["brake"] = 1.
    drop_contradicted_instructions(frames, rules, objects=objects)
    final_quality(frames, rules, objects=objects)
    assert rules[0]["instruction"] is None and not rules[0]["action_valid"]


def test_missing_target_does_not_exempt_fresh_hard_brake_resume_conflict():
    frames, objects, rules = case()
    rules[0].update(instruction_state="resume", instruction_id="dynamic_resume",
                    instruction="Resume driving.")
    pub = frames[-1]["expert_publications"]
    pub["control"].update(accel=0., brake=1.)
    pub["target_velocity"] = None
    drop_contradicted_instructions(frames, rules, objects=objects)
    final_quality(frames, rules, objects=objects)
    assert rules[0]["instruction"] is None and not rules[0]["action_valid"]
    assert "current_dynamic_resume_hard_hold_command_conflict" in rules[0]["unknown_reasons"]


def test_final_quality_cannot_accept_departure_marker_without_inventory_revalidation():
    frames, objects, rules = case()
    align_current_departure_plans(frames, objects, rules)
    final_quality(frames, rules)
    assert rules[0]["instruction"] is None and not rules[0]["action_valid"]


@pytest.mark.parametrize("speed", [.525, 1., 1.953, 2., 4.])
def test_current_stop_command_closes_continue_motion_threshold_gap(speed):
    """A brake/zero-target transition is already a stop intent before v==0."""
    from MoraiDrive.data_collection.situation_labeling.consistency import _contradiction
    frames, objects, rules = case("normal")
    frame, rule = frames[-1], rules[0]
    frame["ego_speed_mps"] = speed
    rule.update(instruction_state="normal", instruction_id="dynamic_normal",
                instruction="Continue driving.", evidence={})
    pub = frame["expert_publications"]
    pub["control"].update(accel=0., brake=1.)
    pub["target_velocity"]["value_mps"] = 0.
    if 1. <= speed <= 2.:
        # Reproduces the previous blind interval in the motion-only check.
        assert _contradiction(rule["instruction"], 0., 0., speed, 0.) is None
    drop_contradicted_instructions(frames, rules, objects=objects)
    final_quality(frames, rules, objects=objects)
    assert rule["instruction"] is None and rule["action_valid"] is False
    assert rule["observation_valid"] is True
    assert "current_dynamic_obstacle_continue_hard_hold_command_conflict" in rule["unknown_reasons"]


@pytest.mark.parametrize("zone,text", [
    ("dynamic_obstacle", "Continue driving."),
    ("static_obstacle", "Continue driving."),
    ("roundabout", "Continue driving along the current path.")])
def test_explicit_continue_stop_conflict_is_independent_of_collection_category(zone, text):
    from MoraiDrive.data_collection.situation_labeling.departure_plan import current_instruction_command_reasons
    frames, _, rules = case("normal")
    frame, rule = frames[-1], rules[0]
    frame["zone_id"] = rule["zone_id"] = zone
    rule.update(instruction_id="normal", instruction=text)
    pub = frame["expert_publications"]
    pub["control"].update(accel=0., brake=1.)
    pub["target_velocity"]["value_mps"] = 0.
    assert current_instruction_command_reasons(frame, rule) == [
        "current_{}_continue_hard_hold_command_conflict".format(zone)]


@pytest.mark.parametrize("mutation", ["missing_target", "stale_target", "future_target",
    "positive_target", "negative_target", "light_brake", "throttle", "maintain", "avoidance"])
def test_continue_guard_does_not_invent_stop_intent_or_override_lateral_intent(mutation):
    from MoraiDrive.data_collection.situation_labeling.departure_plan import current_instruction_command_reasons
    frames, _, rules = case("normal")
    frame, rule = frames[-1], rules[0]
    rule.update(instruction_id="dynamic_normal", instruction="Continue driving.")
    pub = frame["expert_publications"]
    pub["control"].update(accel=0., brake=1.)
    target = pub["target_velocity"]
    target["value_mps"] = 0.
    if mutation == "missing_target": pub["target_velocity"] = None
    elif mutation == "stale_target":
        target.update(receive_age_ms=101., receive_monotonic_ns=pub["lidar_receive_monotonic_ns"]-101_000_000)
    elif mutation == "future_target":
        target.update(receive_age_ms=-1., receive_monotonic_ns=pub["lidar_receive_monotonic_ns"]+1_000_000)
    elif mutation == "positive_target": target["value_mps"] = 4.
    elif mutation == "negative_target": target["value_mps"] = -1.
    elif mutation == "light_brake": pub["control"]["brake"] = .2
    elif mutation == "throttle": pub["control"]["accel"] = .3
    elif mutation == "maintain": rule["instruction"] = "Maintain the current driving behavior."
    elif mutation == "avoidance": rule["instruction"] = "Continue the avoidance maneuver to the right."
    assert current_instruction_command_reasons(frame, rule) == []
