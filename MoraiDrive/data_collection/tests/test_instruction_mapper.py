"""Rule instruction mapping keeps Expert labels separate from GT candidates."""
import json

from MoraiDrive.data_collection.situation_labeling.attach_expert_instructions import main
from MoraiDrive.data_collection.situation_labeling.mapper import (
    attach_rule_instructions, load_label_instruction_map,
)
from MoraiDrive.data_collection.situation_labeling.renderer import load_catalog


def rule(zone, state, *, label=None, candidate=None, variables=None, status="matched"):
    return {"index": 0, "zone_id": zone, "status": status,
            "instruction_state": state, "label": label,
            "label_candidate": candidate, "variables": variables or {},
            "instruction_id": None, "instruction": None}


def test_rule_only_static_mapping_and_template_variables():
    catalog = load_catalog()
    mapping = load_label_instruction_map(catalog=catalog)
    row = rule("static_obstacle", "approach_maintain", variables={
        "target_kind": "obstacle", "target_side": "center", "target_distance_m": 24.8})
    assert attach_rule_instructions([row]) == {"rendered": 1}
    assert row["instruction_id"] == "static_approach_maintain"
    assert row["instruction"] == catalog["static_obstacle"]["static_approach_maintain"][
        "instruction"].format(position="전방")
    assert row["instruction_source"] == "rule_state"
    assert row["label"] is None
    assert row["instruction_variables"] == {"position": "전방"}
    returning = rule("static_obstacle", "return_to_path")
    attach_rule_instructions([returning])
    assert returning["instruction_id"] == "static_return_to_path"
    assert returning["instruction"] == catalog["static_obstacle"]["static_return_to_path"]["instruction"]


def test_same_highway_label_maps_to_distinct_instructions_without_promoting_candidate():
    rows = [rule("highway", state, candidate="ACC_FOLLOW", variables={
        "target_kind": "vehicle", "target_position": "front", "target_distance_m": 12.6})
        for state in ("follow", "decelerating", "stopped")]
    assert attach_rule_instructions(rows) == {"rendered": 3}
    assert [row["instruction_id"] for row in rows] == [
        "acc_follow", "acc_decelerate", "acc_stop"]
    assert len({row["instruction"] for row in rows}) == 3
    assert all(row["instruction_source"] == "rule_label_candidate" for row in rows)
    assert all(row["label"] is None for row in rows)


def test_expert_label_and_rule_state_select_exact_variant():
    row = rule("highway", "stopped", label="ACC_FOLLOW", variables={
        "target_kind": "vehicle", "target_position": "front", "target_distance_m": 8.0})
    attach_rule_instructions([row])
    assert row["instruction_id"] == "acc_stop"
    assert row["instruction_source"] == "expert_label_and_rule"
    assert row["instruction_label"] == "ACC_FOLLOW"
    assert row["instruction"] is not None

def test_missing_variables_and_unknown_states_do_not_render():
    incomplete = rule("static_obstacle", "approach_maintain")
    unknown = rule("static_obstacle", None, status="unknown")
    unsafe_candidate = rule("roundabout", "resume_candidate")
    assert attach_rule_instructions([incomplete, unknown, unsafe_candidate]) == {
        "label_rule_state_unmapped": 1, "missing_variables": 1,
        "rule_state_unavailable": 1}
    assert incomplete["instruction_id"] == "static_approach_maintain"
    assert incomplete["instruction"] is None
    assert incomplete["instruction_missing_variables"] == ["position"]
    assert unknown["instruction_id"] is None
    assert unsafe_candidate["instruction_id"] is None


def test_conflicting_expert_label_and_rule_state_fail_closed():
    row = rule("highway", "normal", label="ACC_FOLLOW")
    row.update(instruction_id="acc_follow", instruction="stale sentence")
    attach_rule_instructions([row])
    assert row["label"] == "ACC_FOLLOW"
    assert row["instruction_id"] is None
    assert row["instruction"] is None
    assert row["instruction_mapping_status"] == "expert_label_rule_state_unmapped"


def test_dynamic_zone_has_no_expert_label_but_still_maps_rule_state():
    # Dynamic obstacle has no Expert-sourced label; its instruction_state is
    # the only label dimension and renders through label: null like static
    # obstacle's rule-only states.
    row = rule("dynamic_obstacle", "yielding", variables={
        "target_kind": "pedestrian", "target_side": "left", "target_distance_m": 12.4})
    assert attach_rule_instructions([row]) == {"rendered": 1}
    assert row["instruction_id"] == "dynamic_yielding"
    assert row["label"] is None


def test_standalone_postprocess_works_without_expert_log(tmp_path):
    source, destination = tmp_path / "features", tmp_path / "instructions"
    source.mkdir()
    (source / "frame_features.jsonl").write_text(
        json.dumps({"index": 0, "zone_id": "static_obstacle"}) + "\n", encoding="utf8")
    original = json.dumps(rule("static_obstacle", "return_to_path")) + "\n"
    (source / "situation_rules.jsonl").write_text(original, encoding="utf8")
    report = main(["--features-dir", str(source), "--output", str(destination)])
    result = json.loads((destination / "situation_instructions.jsonl").read_text(encoding="utf8"))
    assert report["source_expert_logs"] == []
    assert report["instruction_mapping_counts"] == {"rendered": 1}
    assert result["instruction_id"] == "static_return_to_path"
    assert result["instruction"] is not None
    assert (source / "situation_rules.jsonl").read_text(encoding="utf8") == original