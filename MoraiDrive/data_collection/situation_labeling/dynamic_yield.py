"""Causal yield episodes; a momentary creep is not an obstacle release.

This pass never authorizes Resume, changes upstream masks, or uses future ego
poses. Generic Yield describes a continuing conflict even during a brief stop;
Remain stopped requires a stable recorded stop, not one threshold crossing.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math

from .dynamic_resume import _frame_usable, _masked, _number
from .mapper import load_label_instruction_map
from .renderer import load_catalog, render_instruction
from .rules import dynamic_obstacle_rule

SCHEMA = "current_dynamic_yield_v1"
BASIS = "recorded_current_yield_episode"
YIELD_STATES = {"yield_start", "yielding", "waiting"}
REVALIDATABLE_REASON = "future_pedestrian_motion_unverified"


@lru_cache(maxsize=1)
def _render_config():
    catalog = load_catalog()
    return catalog, load_label_instruction_map(catalog=catalog)


def _usable(frame):
    return (_frame_usable(frame) and frame.get("zone_id") == "dynamic_obstacle"
            and _number(frame.get("ego_speed_mps")) is not None)


def _connected(a, b):
    if not _usable(a) or not _usable(b):
        return False
    dt = (b["lidar_source_ns"] - a["lidar_source_ns"]) / 1e9
    distance = math.hypot(b["ego_world_x_m"] - a["ego_world_x_m"],
                          b["ego_world_y_m"] - a["ego_world_y_m"])
    return (b["index"] == a["index"] + 1 and 0 < dt <= .25
            and distance <= 50 * dt + .05
            and math.isclose(distance / dt, b["ego_speed_mps"], rel_tol=1e-4, abs_tol=1e-4))


def _context(frame, actors):
    if not _usable(frame):
        return None
    raw = dynamic_obstacle_rule(frame, actors)
    if (raw.get("status") != "matched" or _masked(raw)
            or set(raw.get("unknown_reasons") or ()) - {REVALIDATABLE_REASON}
            or raw.get("instruction_state") not in YIELD_STATES | {"approach"}
            or raw.get("evidence", {}).get("pedestrian_on_conflict_path") not in (True, False)):
        return None
    ref = raw["evidence"].get("target_actor_ref") or {}
    # Reused actor IDs must have exactly one current inventory row. A second
    # instance under the same API ID is ambiguous even if one ref matches.
    if sum(a.get("actor_id") == ref.get("actor_id") for a in actors) != 1:
        return None
    selected = [a for a in actors if a.get("actor_id") == ref.get("actor_id")
                and a.get("instance_key") == ref.get("instance_key")]
    if len(selected) != 1 or not ref.get("instance_key"):
        return None
    actor = selected[0]
    if (actor.get("index") != frame["index"]
            or actor.get("lidar_source_ns") != frame["lidar_source_ns"]
            or actor.get("rule_geometry_usable") is not True
            or not all(_number(actor.get(k)) is not None for k in
                       ("world_x_m", "world_y_m", "path_d_m", "delta_s_m",
                        "actor_width_m", "actor_length_m"))
            or min(actor["actor_width_m"], actor["actor_length_m"]) <= 0
            or set(actor.get("invalid_reasons") or ()) - {
                "outside_training_bev_or_unknown", "object_geometry_not_candidate"}):
        return None
    expected_delta = actor.get("path_s_m")
    if expected_delta is not None and not math.isclose(
            expected_delta - frame["ego_path_s_m"], actor["delta_s_m"], abs_tol=1e-3):
        return None
    return raw, actor


def _past_window(history):
    current = history[-1]
    earlier = [f for f in history[:-1]
               if .3 <= (current["lidar_source_ns"] - f["lidar_source_ns"]) / 1e9 <= .6]
    if not earlier:
        return None
    first = max(earlier, key=lambda f: f["index"])
    return history[history.index(first):]


def _response(window):
    speeds = [f["ego_speed_mps"] for f in window]
    low_speed = max(speeds) <= 2.0
    drop = speeds[0] - speeds[-1]
    slowing = drop >= .3 and sum(b < a - .02 for a, b in zip(speeds, speeds[1:])) >= 2
    return low_speed or slowing


def _decisions(frames, objects, excluded_indices=()):
    grouped = defaultdict(list)
    for actor in objects:
        grouped[actor.get("index")].append(actor)
    decisions, history = {}, []
    previous, previous_actor, active_ref = None, None, None
    last_supported_ns = None
    for frame in sorted(frames, key=lambda f: f["index"]):
        context = _context(frame, grouped[frame["index"]])
        if frame["index"] in excluded_indices or context is None:
            history, active_ref, previous, previous_actor = [], None, None, None
            last_supported_ns = None
            continue
        raw, actor = context
        ref = (actor["actor_id"], actor["instance_key"])
        continuous = _connected(previous, frame)
        if continuous and previous_actor is not None:
            dt = (frame["lidar_source_ns"] - previous["lidar_source_ns"]) / 1e9
            continuous = (ref == (previous_actor["actor_id"], previous_actor["instance_key"])
                          and actor.get("model_name") == previous_actor.get("model_name")
                          and math.hypot(actor["world_x_m"] - previous_actor["world_x_m"],
                                         actor["world_y_m"] - previous_actor["world_y_m"]) <= 20 * dt + .5)
        if not continuous:
            history, active_ref = [], None
            last_supported_ns = None
        history.append(frame)
        history = [f for f in history if
                   frame["lidar_source_ns"] - f["lidar_source_ns"] <= 600_000_000]
        previous, previous_actor = frame, actor
        if raw["evidence"]["pedestrian_on_conflict_path"] is not True:
            # A verified approaching observation belongs to this actor's
            # causal history, but a non-conflicting current frame has no
            # authority to issue Yield. A later conflict must start a fresh
            # episode using independently observed past motion.
            active_ref, last_supported_ns = None, None
            continue
        window = _past_window(history)
        if window is None:
            continue
        if active_ref is None:
            if not _response(window):
                continue
            active_ref = ref
            onset = frame["index"]
            last_supported_ns = frame["lidar_source_ns"]
        supported_now = _response(window)
        speeds = [f["ego_speed_mps"] for f in window]
        meaningful_reacceleration = (speeds[-1] > 2.0 and speeds[-1] - speeds[0] >= .3
                                     and sum(b > a + .02 for a, b in zip(speeds, speeds[1:])) >= 2)
        if supported_now:
            last_supported_ns = frame["lidar_source_ns"]
        if meaningful_reacceleration or (frame["lidar_source_ns"] - last_supported_ns > 300_000_000):
            active_ref = None
            continue
        stable_stop = all(f["ego_speed_mps"] <= .2 for f in window)
        state = "waiting" if stable_stop else "yielding"
        receipt = {"schema": SCHEMA, "supervision_basis": BASIS,
                   "uses_future_ego": False, "physical_front_safety_certified": False,
                   "target_actor_ref": {"actor_id": ref[0], "instance_key": ref[1]},
                   "episode_onset_index": onset, "past_start_index": window[0]["index"],
                   "past_window_seconds": (frame["lidar_source_ns"] - window[0]["lidar_source_ns"]) / 1e9,
                   "past_max_speed_mps": max(f["ego_speed_mps"] for f in window),
                   "current_speed_mps": frame["ego_speed_mps"], "state": state}
        decisions[frame["index"]] = (state, raw, receipt)
    return decisions


def align_current_dynamic_yield(frames, objects, rules):
    """Stabilize verified current yields without promoting any exclusion."""
    if objects is None:
        raise ValueError("An explicit inventory is required")
    if len({f["index"] for f in frames}) != len(frames):
        raise ValueError("Duplicate frame index")
    excluded = {r["index"] for r in rules if _masked(r)
                or set(r.get("unknown_reasons") or ()) - {REVALIDATABLE_REASON}
                or r.get("status") != "matched"
                or r.get("instruction_state") not in YIELD_STATES | {"approach"}}
    decisions = _decisions(frames, objects, excluded)
    counts = Counter()
    for rule in rules:
        choice = decisions.get(rule["index"])
        if choice is None:
            if (rule["index"] not in excluded and rule.get("instruction_state") in YIELD_STATES):
                rule["action_valid"] = False
                rule["unknown_reasons"] = sorted(set(rule.get("unknown_reasons") or ()) |
                                                 {"current_yield_response_unverified"})
                counts["unsupported_yield_excluded"] += 1
            counts["unchanged_without_current_yield_proof"] += 1
            continue
        state, raw, receipt = choice
        # Compare fresh actor/semantic/position fields before retaining a row.
        if any(rule.get("variables", {}).get(k) != raw["variables"].get(k) for k in
               ("target_actor_id", "target_kind", "target_model_name", "target_semantic_class", "target_side")):
            rule["action_valid"] = False
            rule["unknown_reasons"] = sorted(set(rule.get("unknown_reasons") or ()) |
                                             {"current_yield_target_disagrees"})
            counts["target_disagreement_excluded"] += 1
            continue
        before = rule["instruction_state"]
        rule["instruction_state"] = state
        if REVALIDATABLE_REASON in (rule.get("unknown_reasons") or ()):
            rule["unknown_reasons"] = [r for r in rule["unknown_reasons"] if r != REVALIDATABLE_REASON]
            rule.setdefault("evidence", {})["current_yield_reason_resolution"] = {
                "removed_reason": REVALIDATABLE_REASON,
                "replacement_basis": BASIS, "uses_future_ego": False}
            counts["future_motion_reason_revalidated"] += 1
        rule.setdefault("evidence", {})["current_dynamic_yield"] = receipt
        rule["evidence"].setdefault("postprocessing_history", []).append({
            "operation": "current_dynamic_yield", "state_before": before, "state_after": state})
        counts["verified_current_yield"] += 1
        if state != before:
            counts["to_" + state] += 1
    return dict(sorted(counts.items()))


def validate_current_dynamic_yield(frames, objects, rule):
    """Recompute continuous same-instance conflict and motion from source rows."""
    if objects is None or len({f["index"] for f in frames}) != len(frames):
        return False, ["current_dynamic_yield_inventory_or_frames_missing"]
    marker = rule.get("evidence", {}).get("current_dynamic_yield") or {}
    if (marker.get("schema") != SCHEMA or marker.get("supervision_basis") != BASIS
            or marker.get("uses_future_ego") is not False
            or marker.get("physical_front_safety_certified") is not False
            or rule.get("status") != "matched" or _masked(rule) or rule.get("unknown_reasons")):
        return False, ["current_dynamic_yield_marker_or_eligibility_invalid"]
    past = [f for f in frames if f["index"] <= rule["index"]]
    choice = _decisions(past, objects).get(rule["index"])
    if choice is None:
        return False, ["current_dynamic_yield_evidence_unverified"]
    state, raw, expected_marker = choice
    if marker != expected_marker or rule.get("instruction_state") != state:
        return False, ["current_dynamic_yield_motion_or_instance_disagrees"]
    catalog, mapping = _render_config()
    identifier = mapping.get(("dynamic_obstacle", None, state))
    text, slots, missing = render_instruction(catalog, "dynamic_obstacle", identifier, raw["variables"])
    if (missing or rule.get("instruction_id") != identifier
            or rule.get("instruction") != text or rule.get("instruction_variables") != slots):
        return False, ["current_dynamic_yield_rendered_instruction_disagrees"]
    if any(rule.get("variables", {}).get(k) != raw["variables"].get(k) for k in
           ("target_actor_id", "target_kind", "target_model_name", "target_semantic_class", "target_side")):
        return False, ["current_dynamic_yield_target_disagrees"]
    return True, []
