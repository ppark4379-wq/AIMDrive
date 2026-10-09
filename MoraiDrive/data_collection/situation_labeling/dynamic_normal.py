"""Describe a verified current stop without inferring its traffic cause.

An empty, usable current inventory does not authorize Continue while stopped.
This narrowly replaces raw dynamic normal by the existing generic Remain
stopped sentence after a contiguous recorded stop. It does not infer signal
state, front safety, future departure, or an unobserved obstacle release.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math

from .dynamic_resume import _frame_usable, _masked, _number
from .dynamic_yield import _past_window
from .mapper import load_label_instruction_map
from .renderer import load_catalog, render_instruction
from .rules import dynamic_obstacle_rule

SCHEMA = "current_dynamic_normal_v1"
BASIS = "recorded_current_normal_stop"
MARKER = "current_dynamic_normal"
STOP_MPS = .2
OTHER_MARKERS = ("current_dynamic_response", "current_dynamic_yield",
                 "current_dynamic_preentry_yield", "current_dynamic_approach")


@lru_cache(maxsize=1)
def _render_config():
    catalog = load_catalog()
    return catalog, load_label_instruction_map(catalog=catalog)


def _eligible(rule, state="normal"):
    evidence = rule.get("evidence") or {}
    return (rule.get("zone_id") == "dynamic_obstacle"
            and rule.get("status") == "matched"
            and rule.get("instruction_state") == state
            and not _masked(rule) and not rule.get("unknown_reasons")
            and not any(evidence.get(k) for k in OTHER_MARKERS))


def _context(frame, actors):
    # Empty means the sealed inventory is explicitly usable and currently has
    # no rows; an absent/unusable inventory can never be treated as empty.
    if (not _frame_usable(frame) or _masked(frame) or frame.get("invalid_reasons")
            or frame.get("zone_id") != "dynamic_obstacle"
            or _number(frame.get("ego_speed_mps")) is None or actors):
        return None
    raw = dynamic_obstacle_rule(frame, actors)
    return raw if _eligible(raw) and not raw.get("related_actors") else None


def _connected(previous, current):
    if previous is None:
        return False
    dt = (current["lidar_source_ns"] - previous["lidar_source_ns"]) / 1e9
    distance = math.hypot(current["ego_world_x_m"] - previous["ego_world_x_m"],
                          current["ego_world_y_m"] - previous["ego_world_y_m"])
    ds = current["ego_path_s_m"] - previous["ego_path_s_m"]
    return (current["index"] == previous["index"] + 1 and 0 < dt <= .25
            and distance <= 50 * dt + .05
            and abs(ds) <= distance + .05
            and abs(current["ego_path_d_m"] - previous["ego_path_d_m"]) <= distance + .05
            and math.isclose(distance / dt, current["ego_speed_mps"],
                             rel_tol=1e-4, abs_tol=1e-4))


def _decisions(frames, objects, excluded_indices=()):
    grouped = defaultdict(list)
    for actor in objects:
        grouped[actor.get("index")].append(actor)
    history, previous, decisions = [], None, {}
    for frame in sorted(frames, key=lambda f: f["index"]):
        raw = _context(frame, grouped[frame["index"]])
        if raw is None or frame["index"] in excluded_indices:
            history, previous = [], None
            continue
        if not _connected(previous, frame):
            history = []
        history.append(frame)
        history = [f for f in history if
                   frame["lidar_source_ns"] - f["lidar_source_ns"] <= 600_000_000]
        previous = frame
        window = _past_window(history)
        if window is None or max(f["ego_speed_mps"] for f in window) > STOP_MPS:
            continue
        receipt = {
            "schema": SCHEMA, "supervision_basis": BASIS,
            "state": "wait_clear", "uses_future_ego": False,
            "physical_front_safety_certified": False,
            "traffic_signal_cause_verified": False,
            "current_inventory_actor_count": 0,
            "past_start_index": window[0]["index"],
            "past_start_source_ns": window[0]["lidar_source_ns"],
            "current_source_ns": frame["lidar_source_ns"],
            "past_window_seconds": (frame["lidar_source_ns"] - window[0]["lidar_source_ns"]) / 1e9,
            "current_speed_mps": frame["ego_speed_mps"],
            "past_max_speed_mps": max(f["ego_speed_mps"] for f in window),
        }
        decisions[frame["index"]] = (raw, receipt)
    return decisions


def align_current_dynamic_normal(frames, objects, rules):
    """Replace only independently verified raw normal stops; exclude edges."""
    if objects is None or len({f["index"] for f in frames}) != len(frames):
        raise ValueError("Unique frames and explicit current inventory are required")
    excluded = {r["index"] for r in rules if not _eligible(r)}
    choices = _decisions(frames, objects, excluded)
    by_index = {f["index"]: f for f in frames}
    counts = Counter()
    for rule in rules:
        if not _eligible(rule):
            continue
        choice = choices.get(rule["index"])
        if choice:
            raw, receipt = choice
            if (rule.get("variables") != raw.get("variables")
                    or rule.get("related_actors") != raw.get("related_actors")):
                rule["action_valid"] = False
                rule["unknown_reasons"] = ["current_normal_source_disagrees"]
                counts["source_disagreement_excluded"] += 1
                continue
            rule.setdefault("instruction_postprocessing_history", []).append({
                "operation": "recorded_current_normal_stop",
                "before_state": rule["instruction_state"], "after_state": "wait_clear",
                "uses_future_ego": False, "traffic_signal_cause_verified": False})
            rule["instruction_state"] = "wait_clear"
            rule.setdefault("evidence", {})[MARKER] = receipt
            counts["verified_current_stop"] += 1
        else:
            frame = by_index.get(rule["index"], {})
            speed = _number(frame.get("ego_speed_mps"))
            if speed is not None and speed <= STOP_MPS:
                # The current frame already contradicts Continue. Without a
                # full stable stop window no alternative action is invented.
                rule["action_valid"] = False
                rule["unknown_reasons"] = ["current_normal_stop_phase_unverified"]
                counts["current_stop_boundary_excluded"] += 1
    return dict(sorted(counts.items()))


def validate_current_dynamic_normal(frames, objects, rule):
    """Recompute current/past stops and final text, never trust a marker alone."""
    if objects is None or len({f["index"] for f in frames}) != len(frames):
        return False, ["current_dynamic_normal_inventory_or_frames_missing"]
    marker = rule.get("evidence", {}).get(MARKER) or {}
    if (not _eligible(rule, "wait_clear") or marker.get("schema") != SCHEMA
            or marker.get("supervision_basis") != BASIS):
        return False, ["current_dynamic_normal_marker_or_eligibility_invalid"]
    past = [f for f in frames if f["index"] <= rule["index"]]
    choice = _decisions(past, objects).get(rule["index"])
    if choice is None:
        return False, ["current_dynamic_normal_stop_unverified"]
    raw, expected = choice
    if (marker != expected or rule.get("variables") != raw.get("variables")
            or rule.get("related_actors") != raw.get("related_actors")):
        return False, ["current_dynamic_normal_source_or_motion_disagrees"]
    catalog, mapping = _render_config()
    identifier = mapping.get(("dynamic_obstacle", None, "wait_clear"))
    text, slots, missing = render_instruction(catalog, "dynamic_obstacle", identifier, raw["variables"])
    if (missing or rule.get("instruction_id") != identifier or rule.get("instruction") != text
            or rule.get("instruction_variables") != slots):
        return False, ["current_dynamic_normal_rendering_disagrees"]
    return True, []
