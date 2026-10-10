"""Rule instruction mapping keeps Expert labels separate from GT candidates."""
import json

from MoraiDrive.data_collection.situation_labeling.attach_expert_instructions import main
from MoraiDrive.data_collection.situation_labeling.mapper import (
    attach_rule_instructions, load_label_instruction_map,
)
from MoraiDrive.data_collection.situation_labeling.renderer import (
    actor_semantic_class, load_catalog, render_instruction,
)
import pytest


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
        "instruction"].format(position="in front of")
    assert row["instruction_source"] == "rule_state"
    assert row["label"] is None
    assert row["instruction_variables"] == {"position": "in front of"}
    returning = rule("static_obstacle", "return_to_path")
    attach_rule_instructions([returning])
    assert returning["instruction_id"] == "static_return_to_path"
    assert returning["instruction"] == catalog["static_obstacle"]["static_return_to_path"]["instruction"]


def test_follow_clear_obstacle_renders_keep_path_instruction():
    catalog = load_catalog()
    row = rule("static_obstacle", "keep_path", label="follow", variables={
        "target_kind": "obstacle", "target_side": "right",
        "target_distance_m": 16.35})
    # A directly mapped Expert FOLLOW fallback must be replaced by the
    # more specific label + current-frame geometry decision.
    row.update(instruction_id="follow_avoid", instruction="stale fallback",
               instruction_source="expert_observation")
    assert attach_rule_instructions([row]) == {"rendered": 1}
    assert row["instruction_id"] == "follow_keep_path"
    assert row["instruction_source"] == "expert_label_and_rule"
    assert row["instruction"] == catalog["static_obstacle"]["follow_keep_path"][
        "instruction"].format(**row["instruction_variables"])


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


@pytest.mark.parametrize("model,semantic,noun", [
    ("Woman8", "pedestrian", "a pedestrian"),
    ("Deer1", "deer", "a deer"),
    ("Dog1", "dog", "a dog"),
    ("Dog2", "dog", "a dog"),
    ("Dog10", "dog", "a dog"),
    ("Ped_Pet1", "unknown", "an unidentified moving object"),
])
def test_dynamic_yield_noun_and_action_describe_same_semantic_target(model, semantic, noun):
    assert actor_semantic_class("pedestrian", model, identity_verified=True)[0] == semantic
    row = rule("dynamic_obstacle", "yielding", variables={
        "target_kind": "pedestrian", "target_model_name": model,
        "target_semantic_class": semantic, "target_side": "left"})
    attach_rule_instructions([row])
    assert row["instruction"] == (
        f"There is {noun} to the left of the ego vehicle. Yield to {noun}.")
    assert row["instruction_variables"] == {"position": "to the left of", "object": noun}
    if semantic != "pedestrian":
        assert "pedestrian" not in row["instruction"]


@pytest.mark.parametrize("model", ["Man101", "Dog1_variant", "Ped_Pet1", "NCAP_EBT", "", None, {}])
def test_model_semantics_never_guess_from_names_or_api_pedestrian_kind(model):
    assert actor_semantic_class("pedestrian", model, identity_verified=True)[0] == "unknown"
    assert actor_semantic_class("pedestrian", "Woman8", identity_verified=False) == (
        "unknown", "model_identity_unverified")


def test_legacy_renderer_without_model_metadata_keeps_text_compatibility_only():
    text, values, missing = render_instruction(load_catalog(), "dynamic_obstacle", "dynamic_yielding", {
        "target_kind": "pedestrian", "target_side": "left"})
    assert text == "There is a pedestrian to the left of the ego vehicle. Yield to a pedestrian."
    assert missing == []
    assert values["object"] == "a pedestrian"


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
