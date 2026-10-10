"""Observed constant-speed dynamic approach, without future-motion inference.

Maintain describes the recorded current behavior. It is neither permission to
proceed through an obstacle nor a physical front-safety certificate.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math

from .dynamic_resume import _frame_usable, _masked, _number
from .mapper import load_label_instruction_map
from .renderer import load_catalog, render_instruction
from .rules import dynamic_obstacle_rule

SCHEMA = "current_dynamic_approach_v1"
BASIS = "recorded_current_approach"
REVALIDATABLE_REASON = "future_pedestrian_motion_unverified"
TARGET_KEYS = ("target_actor_id", "target_kind", "target_model_name",
               "target_semantic_class", "target_side")


@lru_cache(maxsize=1)
def _render_config():
    catalog = load_catalog()
    return catalog, load_label_instruction_map(catalog=catalog)


def _eligible(row):
    evidence = row.get("evidence") or {}
    return (row.get("zone_id") == "dynamic_obstacle" and row.get("status") == "matched"
            and row.get("instruction_state") == "approach" and not _masked(row)
            and not (set(row.get("unknown_reasons") or ()) - {REVALIDATABLE_REASON})
            and not evidence.get("current_dynamic_yield")
            and not evidence.get("current_dynamic_response"))


def _context(frame, actors):
    if (not _frame_usable(frame) or frame.get("zone_id") != "dynamic_obstacle"
            or _number(frame.get("ego_speed_mps")) is None):
        return None
    raw = dynamic_obstacle_rule(frame, actors)
    if not _eligible(raw):
        return None
    ref = raw.get("evidence", {}).get("target_actor_ref") or {}
    selected = [a for a in actors if a.get("actor_id") == ref.get("actor_id")]
    if len(selected) != 1 or not ref.get("instance_key"):
        return None
    actor = selected[0]
    if (actor.get("instance_key") != ref["instance_key"]
            or actor.get("index") != frame["index"]
            or actor.get("lidar_source_ns") != frame["lidar_source_ns"]
            or actor.get("rule_geometry_usable") is not True
            or not all(_number(actor.get(k)) is not None for k in
                       ("world_x_m", "world_y_m", "path_d_m", "delta_s_m",
                        "actor_width_m", "actor_length_m"))
            or min(actor["actor_width_m"], actor["actor_length_m"]) <= 0
            or set(actor.get("invalid_reasons") or ()) - {
                "outside_training_bev_or_unknown", "object_geometry_not_candidate"}):
        return None
    actual_distance = math.hypot(actor["world_x_m"]-frame["ego_world_x_m"],
                                 actor["world_y_m"]-frame["ego_world_y_m"])
    if actual_distance > 35.0:
        return None
    actor_s = _number(actor.get("path_s_m"))
    actor_ego_s = _number(actor.get("ego_path_s_m"))
    if actor_s is not None and not math.isclose(
            actor_s-frame["ego_path_s_m"], actor["delta_s_m"], abs_tol=1e-3):
        return None
    if actor_ego_s is not None and not math.isclose(actor_ego_s,frame["ego_path_s_m"],abs_tol=1e-3):
        return None
    return raw, actor


def _connected(a, b):
    dt = (b["lidar_source_ns"]-a["lidar_source_ns"]) / 1e9
    if b["index"] != a["index"]+1 or not 0 < dt <= .25:
        return False
    distance = math.hypot(b["ego_world_x_m"]-a["ego_world_x_m"],
                          b["ego_world_y_m"]-a["ego_world_y_m"])
    ds = b["ego_path_s_m"]-a["ego_path_s_m"]
    return (distance <= 50*dt+.05 and ds > 0
            and abs(b["ego_path_d_m"]-a["ego_path_d_m"]) <= distance+.05
            and math.isclose(distance/dt,b["ego_speed_mps"],rel_tol=1e-4,abs_tol=1e-4))


def _decisions(frames, objects, excluded_indices=()):
    grouped = defaultdict(list)
    for actor in objects:
        grouped[actor.get("index")].append(actor)
    history, decisions = [], {}
    previous, previous_actor = None, None
    for frame in sorted(frames,key=lambda f:f["index"]):
        context = _context(frame,grouped[frame["index"]])
        if context is None or frame["index"] in excluded_indices:
            history, previous, previous_actor = [], None, None
            continue
        raw, actor = context
        continuous = (previous is not None and _connected(previous,frame)
                      and actor["actor_id"] == previous_actor["actor_id"]
                      and actor["instance_key"] == previous_actor["instance_key"]
                      and actor.get("model_name") == previous_actor.get("model_name"))
        if continuous:
            dt = (frame["lidar_source_ns"]-previous["lidar_source_ns"])/1e9
            continuous = math.hypot(actor["world_x_m"]-previous_actor["world_x_m"],
                                    actor["world_y_m"]-previous_actor["world_y_m"]) <= 20*dt+.5
        if not continuous:
            history = []
        history.append(frame)
        history = [f for f in history if frame["lidar_source_ns"]-f["lidar_source_ns"]<=600_000_000]
        previous, previous_actor = frame, actor
        starts = [f for f in history[:-1] if
                  .3 <= (frame["lidar_source_ns"]-f["lidar_source_ns"])/1e9 <= .6]
        if not starts:
            continue
        first = max(starts,key=lambda f:f["index"])
        window = history[history.index(first):]
        speeds = [f["ego_speed_mps"] for f in window]
        if min(speeds) <= 2.0 or max(speeds)-min(speeds) > .15:
            continue
        receipt = {"schema":SCHEMA,"supervision_basis":BASIS,
                   "uses_future_ego":False,"physical_front_safety_certified":False,
                   "target_actor_ref":{"actor_id":actor["actor_id"],"instance_key":actor["instance_key"]},
                   "past_start_index":first["index"],
                   "past_window_seconds":(frame["lidar_source_ns"]-first["lidar_source_ns"])/1e9,
                   "past_min_speed_mps":min(speeds),"past_max_speed_mps":max(speeds),
                   "current_speed_mps":frame["ego_speed_mps"],"state":"approach"}
        decisions[frame["index"]]=(raw,receipt)
    return decisions


def align_current_dynamic_approach(frames, objects, rules):
    """Certify current approach only; never promote an explicit False mask."""
    if objects is None or len({f["index"] for f in frames}) != len(frames):
        raise ValueError("Unique frames and explicit inventory are required")
    excluded={r["index"] for r in rules if not _eligible(r)}
    choices=_decisions(frames,objects,excluded)
    counts=Counter()
    for rule in rules:
        choice=choices.get(rule["index"])
        if choice is None:
            continue
        raw,receipt=choice
        if any(rule.get("variables",{}).get(k)!=raw["variables"].get(k) for k in TARGET_KEYS):
            rule["action_valid"]=False
            rule["unknown_reasons"]=sorted(set(rule.get("unknown_reasons")or())|{"current_approach_target_disagrees"})
            counts["target_disagreement_excluded"]+=1
            continue
        rule.setdefault("evidence",{})["current_dynamic_approach"]=receipt
        if REVALIDATABLE_REASON in (rule.get("unknown_reasons") or ()):
            rule["unknown_reasons"]=[r for r in rule["unknown_reasons"] if r!=REVALIDATABLE_REASON]
            rule["evidence"]["current_approach_reason_resolution"]={
                "removed_reason":REVALIDATABLE_REASON,"replacement_basis":BASIS,"uses_future_ego":False}
            counts["future_motion_reason_revalidated"]+=1
        counts["verified_current_approach"]+=1
    return dict(sorted(counts.items()))


def validate_current_dynamic_approach(frames, objects, rule):
    """Recompute past/current actor, motion and final rendering independently."""
    if objects is None or len({f["index"] for f in frames}) != len(frames):
        return False,["current_dynamic_approach_inventory_or_frames_missing"]
    marker=rule.get("evidence",{}).get("current_dynamic_approach")or{}
    if (not _eligible(rule) or rule.get("unknown_reasons")
            or marker.get("schema")!=SCHEMA or marker.get("supervision_basis")!=BASIS
            or marker.get("uses_future_ego") is not False
            or marker.get("physical_front_safety_certified") is not False):
        return False,["current_dynamic_approach_marker_or_eligibility_invalid"]
    past=[f for f in frames if f["index"]<=rule["index"]]
    choice=_decisions(past,objects).get(rule["index"])
    if choice is None:
        return False,["current_dynamic_approach_evidence_unverified"]
    raw,expected_marker=choice
    if marker!=expected_marker:
        return False,["current_dynamic_approach_motion_or_instance_disagrees"]
    catalog,mapping=_render_config()
    identifier=mapping.get(("dynamic_obstacle",None,"approach"))
    text,slots,missing=render_instruction(catalog,"dynamic_obstacle",identifier,raw["variables"])
    if (missing or rule.get("instruction_id")!=identifier or rule.get("instruction")!=text
            or rule.get("instruction_variables")!=slots):
        return False,["current_dynamic_approach_rendering_disagrees"]
    if any(rule.get("variables",{}).get(k)!=raw["variables"].get(k) for k in TARGET_KEYS):
        return False,["current_dynamic_approach_target_disagrees"]
    return True,[]
