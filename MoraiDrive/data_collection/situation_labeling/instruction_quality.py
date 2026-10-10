"""Final, fail-closed eligibility for SituationHead targets after every label pass.

This checks recorded evidence and typed instruction consistency. It neither
certifies physical sensor simultaneity nor fills missing behavior from neighbors.
Rejected text is retained for review, outside the training instruction field.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import yaml

from .mapper import load_label_instruction_map
from .renderer import load_catalog, render_template

SCHEMA = "situation_instruction_quality_v1"
CONFIG = Path(__file__).resolve().parent / "config"
UNVERIFIED_ACTION = frozenset((
    "object_inventory_unusable", "ego_response_cause_unverified",
    "lane_change_motion_unverified", "resume_front_safety_unverified",
    "avoidance_corridor_safety_unverified", "dynamic_behavior_unverified",
    "current_highway_selected_plan_unverified",
    "current_highway_longitudinal_command_unverified",
))
STOPPED_VARIANTS = {"yield_moving": "yield_stopped", "follow": "follow_stopped",
                    "collision_slowdown": "collision_stopped"}


def _valid_slot(name, value):
    if name == "distance":
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)
    return isinstance(value, str) and bool(value.strip())


def generation_contract():
    """Pin executable labeling sources and active catalogs, including dirty edits."""
    root = Path(__file__).resolve().parent
    paths = sorted(list(root.glob("*.py")) + list((root / "config").glob("*.yaml")))
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return dict(schema="situation_label_generation_contract_v1", quality_schema=SCHEMA,
                code_sha256=hashlib.sha256(encoded).hexdigest(), files=files)


def _expert_instruction_valid(rule, catalog, expert):
    """An independently aligned Expert phase need not equal a GT rule state."""
    alignment = rule.get("expert_alignment") or {}
    if alignment.get("status") != "matched":
        return False
    zone, identifier = rule.get("zone_id"), rule.get("instruction_id")
    state = rule.get("expert_state")
    matches = [entry for entry in expert.get(zone, []) if entry["state"] == state]
    if len(matches) != 1:
        return False
    expected = matches[0]["instruction_id"]
    if identifier != expected and not (
            rule.get("instruction_moving_variant") == expected
            and STOPPED_VARIANTS.get(expected) == identifier):
        return False
    return catalog[zone][identifier].get("label") == rule.get("label")


def finalize_instruction_quality(frames, rules, *, objects=None, catalog=None, mapping=None):
    """Mask unsupported targets and quarantine conflicting text; never repair it.

Observation validity is independent of the action mask. The documented
dynamic_behavior_unverified template can retain observation-only targets;
unknown inventory cannot establish a normal-driving action.
"""
    catalog = load_catalog() if catalog is None else catalog
    mapping = load_label_instruction_map(catalog=catalog) if mapping is None else mapping
    expert = yaml.safe_load((CONFIG / "expert_state_mapping.yaml").read_text(encoding="utf-8"))
    indexed = {f["index"]: f for f in frames}
    if len(indexed) != len(frames) or len({r["index"] for r in rules}) != len(rules):
        raise ValueError("Duplicate frame in instruction quality pass")
    if {r["index"] for r in rules} != set(indexed):
        raise ValueError("Instruction quality frame indices differ")
    counts, observation_reasons, action_reasons = Counter(), Counter(), Counter()
    for rule in rules:
        frame = indexed[rule["index"]]
        rule.setdefault("instruction", None)
        observation, action = [], []
        zone, identifier = rule.get("zone_id"), rule.get("instruction_id")
        text = rule.get("instruction")
        prior_observation, prior_action = rule.get("observation_valid"), rule.get("action_valid")
        from .legacy_roundabout_guard import legacy_roundabout_action_reasons
        action.extend(legacy_roundabout_action_reasons(rule))
        from .departure_plan import current_instruction_command_reasons
        action.extend(current_instruction_command_reasons(frame, rule))
        if identifier == "current_departure_plan":
            from .departure_plan import validate_current_departure_plan
            verified, reasons = validate_current_departure_plan(frames, objects, rule)
            if not verified:
                action.extend(reasons)
        if frame.get("valid") is not True or frame.get("dataset_build_candidate") is not True:
            observation.append("frame_not_admitted")
        if zone != frame.get("zone_id") or zone not in catalog:
            observation.append("unknown_or_conflicting_zone")
        if frame.get("object_inventory_usable") is not True:
            observation.append("object_inventory_unusable")
        variables = rule.get("variables") or {}
        if prior_observation is False:
            observation.append("upstream_observation_mask")
        if variables.get("target_semantic_valid") is False:
            observation.append("target_semantics_unverified")
        template = catalog.get(zone, {}).get(identifier)
        if text:
            if template is None:
                observation.append("unknown_instruction_template")
            else:
                slots = rule.get("instruction_variables") or {}
                if any(not _valid_slot(name, slots.get(name))
                       for name in template["required_variables"]):
                    observation.append("missing_or_invalid_instruction_slots")
                else:
                    rendered, _, missing = render_template(template, slots)
                    if missing or " ".join(rendered.split()) != " ".join(text.split()):
                        observation.append("sentence_template_slots_disagree")
        else:
            action.append("instruction_unavailable")
        if prior_action is False:
            action.append("upstream_action_mask")
        if identifier == "dynamic_behavior_unverified" or rule.get("label") == "unknown":
            action.append("observation_only_action_unverified")
        if text and template is not None:
            label = rule.get("label")
            selected_plan = (rule.get("evidence") or {}).get("current_highway_selected_plan") or {}
            if zone == "highway" and (label is None or selected_plan.get("valid") is True):
                label = rule.get("label_candidate")
            rule_match = (rule.get("status") == "matched" and
                          mapping.get((zone, label, rule.get("instruction_state"))) == identifier)
            expert_match = (rule.get("instruction_source") == "expert_observation"
                            and _expert_instruction_valid(rule, catalog, expert))
            if not rule_match and not expert_match:
                action.append("instruction_state_or_source_disagrees")
        if rule.get("instruction_consistency") != "consistent":
            action.append("motion_consistency_not_verified")
        action.extend(reason for reason in rule.get("unknown_reasons", []) if reason in UNVERIFIED_ACTION)
        # Older hold paths can copy an actor's slots without a fresh target join.
        # Keep this distinct from a harmless repeated sentence supported now.
        evidence = rule.get("evidence") or {}
        if evidence.get("instruction_held_from_index") is not None:
            action.append("held_instruction_requires_current_frame_revalidation")
        observation = sorted(set(observation))
        action = sorted(set(action + observation))
        rule["observation_valid"] = not observation
        rule["action_valid"] = not action
        rule["instruction_quality"] = dict(
            schema=SCHEMA, observation_exclusion_reasons=observation,
            action_exclusion_reasons=action, physical_sensor_synchrony_certified=False)
        if text and action and not (identifier == "dynamic_behavior_unverified" and not observation):
            rule["instruction_rejected"] = dict(
                instruction=text, instruction_id=identifier,
                instruction_state=rule.get("instruction_state"),
                instruction_variables=rule.get("instruction_variables"),
                instruction_source=rule.get("instruction_source"),
                instruction_consistency=rule.get("instruction_consistency"))
            rule["instruction"] = None
            counts["quarantined_text_frames"] += 1
        counts["frames"] += 1
        counts["action_valid_frames"] += int(rule["action_valid"])
        counts["observation_valid_frames"] += int(rule["observation_valid"])
        observation_reasons.update(observation)
        action_reasons.update(action)
    return dict(schema=SCHEMA, counts=dict(counts),
                observation_exclusion_reasons=dict(observation_reasons),
                action_exclusion_reasons=dict(action_reasons),
                missing_instruction_policy="null_with_explicit_masks_no_neighbor_normal_fallback")
