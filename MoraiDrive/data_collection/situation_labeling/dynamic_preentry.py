"""Recorded responses to a currently observed, approaching crossing actor.

The two-second linear prediction is a bounded GT labeling policy, not an
observed future trajectory, an Expert selected-target receipt, or a safety
certificate. Inventory, actor motion and ego response are rechecked causally.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math

from .dynamic_approach import _context, TARGET_KEYS
from .dynamic_resume import _masked
from .dynamic_yield import _connected, _past_window, _response
from .mapper import load_label_instruction_map
from .renderer import load_catalog, render_instruction
from .rules import RuleConfig

SCHEMA = "current_dynamic_preentry_yield_v1"
BASIS = "recorded_current_preentry_yield"
HORIZON_S = 2.0
YIELD_STATES = {"yielding", "waiting"}
REVALIDATABLE_REASON = "future_pedestrian_motion_unverified"


@lru_cache(maxsize=1)
def _render_config():
    catalog = load_catalog()
    return catalog, load_label_instruction_map(catalog=catalog)


def _eligible(rule, final=False):
    state_ok = rule.get("instruction_state") in YIELD_STATES if final else rule.get("instruction_state") == "approach"
    evidence = rule.get("evidence") or {}
    return (rule.get("zone_id") == "dynamic_obstacle" and rule.get("status") == "matched"
            and state_ok and not _masked(rule)
            and not (set(rule.get("unknown_reasons") or ()) - {REVALIDATABLE_REASON})
            and not evidence.get("current_dynamic_yield")
            and not evidence.get("current_dynamic_response"))


def _risk(window, actors, raw):
    frame, current = window[-1], actors[-1]
    if raw.get("evidence", {}).get("pedestrian_on_conflict_path") is not False:
        return None
    if current["delta_s_m"] <= 0:
        return None
    duration = (frame["lidar_source_ns"] - window[0]["lidar_source_ns"]) / 1e9
    lateral = [a["path_d_m"] for a in actors]
    sign = 1. if lateral[-1] > 0 else -1.
    if any(sign * d <= 0 for d in lateral):
        return None
    changes = [sign * (a - b) for a, b in zip(lateral, lateral[1:])]
    toward_distance = sign * (lateral[0] - lateral[-1])
    if (toward_distance < .10 or toward_distance / duration < .3
            or min(changes) < -.01 or sum(d > .02 for d in changes) < 2):
        return None
    # An actual actor translation must support the projected path-d change.
    # This rejects stationary actors and discontinuous route projections.
    world_dx = current["world_x_m"] - actors[0]["world_x_m"]
    world_dy = current["world_y_m"] - actors[0]["world_y_m"]
    world_distance = math.hypot(world_dx, world_dy)
    if world_distance < .10 or toward_distance > world_distance + .05:
        return None
    ego_dx = frame["ego_world_x_m"] - window[0]["ego_world_x_m"]
    ego_dy = frame["ego_world_y_m"] - window[0]["ego_world_y_m"]
    ego_distance = math.hypot(ego_dx, ego_dy)
    projection_error = None
    if ego_distance > .10:
        left_x, left_y = -ego_dy / ego_distance, ego_dx / ego_distance
        projected_d = world_dx * left_x + world_dy * left_y
        projection_error = abs(projected_d - (lateral[-1] - lateral[0]))
        if projection_error > .08 + .25 * toward_distance:
            return None
    for a, b in zip(actors, actors[1:]):
        world_step = math.hypot(b["world_x_m"] - a["world_x_m"], b["world_y_m"] - a["world_y_m"])
        dt = (b["lidar_source_ns"] - a["lidar_source_ns"]) / 1e9
        if (world_step > 20 * dt + .5 or abs(b["path_d_m"] - a["path_d_m"]) > world_step + .05
                or abs(b["actor_width_m"] - a["actor_width_m"]) > .05
                or abs(b["actor_length_m"] - a["actor_length_m"]) > .05):
            return None
    radius = math.hypot(current["actor_width_m"], current["actor_length_m"]) / 2
    clearance = abs(lateral[-1]) - radius - RuleConfig().ego_width_m / 2
    toward_speed = toward_distance / duration
    time_to_entry = clearance / toward_speed
    if not 0 < clearance or not 0 < time_to_entry <= HORIZON_S:
        return None
    return {"prediction_policy": "constant_observed_lateral_velocity",
            "prediction_horizon_seconds": HORIZON_S,
            "predicted_time_to_entry_s": time_to_entry,
            "current_clearance_m": clearance, "current_delta_s_m": current["delta_s_m"],
            "actor_toward_speed_mps": toward_speed, "actor_observed_toward_distance_m": toward_distance,
            "actor_observed_world_distance_m": world_distance,
            "actor_world_path_projection_error_m": projection_error}


def _decisions(frames, objects, excluded_indices=()):
    grouped = defaultdict(list)
    for actor in objects:
        grouped[actor.get("index")].append(actor)
    history, actor_history, choices = [], [], {}
    previous, previous_actor, active_ref, last_supported_ns = None, None, None, None
    for frame in sorted(frames, key=lambda f:f["index"]):
        context = _context(frame, grouped[frame["index"]])
        if context is None or frame["index"] in excluded_indices:
            history, actor_history, previous, previous_actor, active_ref = [], [], None, None, None
            last_supported_ns = None
            continue
        raw, actor = context
        ref = actor["actor_id"], actor["instance_key"]
        continuous = (_connected(previous, frame) and previous_actor is not None
                      and ref == (previous_actor["actor_id"], previous_actor["instance_key"])
                      and actor.get("model_name") == previous_actor.get("model_name"))
        if not continuous:
            history, actor_history, active_ref, last_supported_ns = [], [], None, None
        history.append(frame); actor_history.append(actor)
        keep = [i for i, f in enumerate(history) if frame["lidar_source_ns"]-f["lidar_source_ns"] <= 600_000_000]
        history = [history[i] for i in keep]; actor_history = [actor_history[i] for i in keep]
        previous, previous_actor = frame, actor
        window = _past_window(history)
        if window is None:
            continue
        selected_actors = actor_history[len(history)-len(window):]
        risk = _risk(window, selected_actors, raw)
        if risk is None:
            active_ref, last_supported_ns = None, None
            continue
        response = _response(window)
        if active_ref is None:
            if not response:
                continue
            active_ref, onset, last_supported_ns = ref, frame["index"], frame["lidar_source_ns"]
        speeds = [f["ego_speed_mps"] for f in window]
        reaccelerating = (speeds[-1] > 2 and speeds[-1]-speeds[0] >= .3
                          and sum(b>a+.02 for a,b in zip(speeds,speeds[1:])) >= 2)
        if response:
            last_supported_ns = frame["lidar_source_ns"]
        if reaccelerating or frame["lidar_source_ns"]-last_supported_ns > 300_000_000:
            active_ref, last_supported_ns = None, None
            continue
        state = "waiting" if all(v <= .2 for v in speeds) else "yielding"
        receipt = {"schema":SCHEMA, "supervision_basis":BASIS,
                   "uses_future_ego":False, "uses_future_actor":False,
                   "physical_front_safety_certified":False, "expert_cause_verified":False,
                   "target_actor_ref":{"actor_id":ref[0],"instance_key":ref[1]},
                   "episode_onset_index":onset,"past_start_index":window[0]["index"],
                   "past_window_seconds":(frame["lidar_source_ns"]-window[0]["lidar_source_ns"])/1e9,
                   "current_speed_mps":speeds[-1],"ego_speed_drop_mps":speeds[0]-speeds[-1],
                   "ego_low_response":max(speeds)<=2,"response_currently_supported":response,
                   "last_supported_source_ns":last_supported_ns,"state":state}
        receipt.update(risk)
        choices[frame["index"]] = state, raw, receipt
    return choices


def align_current_dynamic_preentry_yield(frames, objects, rules):
    if objects is None or len({f["index"] for f in frames}) != len(frames):
        raise ValueError("Explicit inventory and unique frames are required")
    choices = _decisions(frames, objects, {r["index"] for r in rules if not _eligible(r)})
    counts = Counter()
    for rule in rules:
        choice = choices.get(rule["index"])
        if choice is None:
            continue
        state, raw, receipt = choice
        if any(rule.get("variables",{}).get(k)!=raw["variables"].get(k) for k in TARGET_KEYS):
            rule["action_valid"] = False
            rule["unknown_reasons"] = sorted(set(rule.get("unknown_reasons")or())|{"current_preentry_target_disagrees"})
            counts["target_disagreement_excluded"] += 1
            continue
        before = rule["instruction_state"]
        rule["instruction_state"] = state
        rule.setdefault("evidence",{})["current_dynamic_preentry_yield"] = receipt
        if REVALIDATABLE_REASON in (rule.get("unknown_reasons")or()):
            rule["unknown_reasons"] = [r for r in rule["unknown_reasons"] if r!=REVALIDATABLE_REASON]
            rule["evidence"]["current_preentry_reason_resolution"] = {
                "removed_reason":REVALIDATABLE_REASON,"replacement_basis":BASIS,
                "uses_future_ego":False,"uses_future_actor":False}
        rule["evidence"].setdefault("postprocessing_history",[]).append({
            "operation":"current_dynamic_preentry_yield","state_before":before,"state_after":state})
        counts["verified_current_preentry_response"] += 1
        counts["to_"+state] += 1
    return dict(sorted(counts.items()))


def validate_current_dynamic_preentry_yield(frames, objects, rule):
    if objects is None or len({f["index"] for f in frames}) != len(frames):
        return False,["current_preentry_inventory_or_frames_missing"]
    marker = rule.get("evidence",{}).get("current_dynamic_preentry_yield") or {}
    if (not _eligible(rule,final=True) or rule.get("unknown_reasons")
            or marker.get("schema")!=SCHEMA or marker.get("supervision_basis")!=BASIS
            or any(marker.get(k) is not False for k in
                   ("uses_future_ego","uses_future_actor","physical_front_safety_certified","expert_cause_verified"))):
        return False,["current_preentry_marker_or_eligibility_invalid"]
    choice = _decisions([f for f in frames if f["index"]<=rule["index"]],objects).get(rule["index"])
    if choice is None:
        return False,["current_preentry_observed_risk_or_response_unverified"]
    state, raw, expected_marker = choice
    if marker!=expected_marker or rule["instruction_state"]!=state:
        return False,["current_preentry_motion_or_instance_disagrees"]
    catalog,mapping = _render_config()
    identifier = mapping.get(("dynamic_obstacle",None,state))
    text,slots,missing = render_instruction(catalog,"dynamic_obstacle",identifier,raw["variables"])
    if (missing or rule.get("instruction_id")!=identifier or rule.get("instruction")!=text
            or rule.get("instruction_variables")!=slots):
        return False,["current_preentry_rendered_instruction_disagrees"]
    if any(rule.get("variables",{}).get(k)!=raw["variables"].get(k) for k in TARGET_KEYS):
        return False,["current_preentry_target_disagrees"]
    return True,[]
