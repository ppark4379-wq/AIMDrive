"""Postprocess existing Expert observations into verified frame instructions.

No collector or Expert runtime writes are required. Sparse logs yield sparse
labels; absent planner states are never inferred from GT rule candidates.
"""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
import json
import math
import re
from pathlib import Path
from string import Formatter

import yaml

from .renderer import DIRECTIONS, OBJECTS, POSITIONS, render_template

CONFIG = Path(__file__).resolve().parent / "config"
PREFIX = "[EXPERT_STATE] "
MATCH_OPERATORS = {"planner_target_speed_mps_lte", "planner_below_ego_mps",
                   "acc_target_speed_mps_lte", "acc_below_ego_mps",
                   "acc_object_type_in", "ego_speed_mps_lte", "ego_speed_mps_gt"}
RAW_FIELDS = {"mode", "reason", "highway_state", "roundabout_phase",
              "roundabout_entering", "roundabout_committed",
              "pedestrian_acquired", "pedestrian_released", "acc_object_type",
              "roundabout_resume_detected", "roundabout_before_merge"}
# roundabout_before_merge is never present in a real Expert observation either --
# it is computed from the frame's own route-geometric roundabout_phase (approach/
# entry/merge/circulating/exit; a distinct field from the Expert planner's raw
# roundabout_phase of follow/yield/cruise/...), because the planner's own
# entering/committed bits flip together on the same tick (see roundabout_entry.py's
# _plan), so they cannot distinguish "still approaching the merge point" from
# "already merged" the way route geometry can.
# roundabout_resume_detected is never present in a real Expert observation --
# apply_expert_instructions injects it on a copy of `raw` only when continuous
# admitted past GT motion and the same current actor prove actual departure
# after a yield/decel/stop response returns to following,
# so FOLLOW_RESUME is selected through the same declarative canonical_state
# matching as every other state, not a separate code path.
ROUNDABOUT_RESPONDING_STATES = {
    "YIELD_STOPPED", "YIELD_MOVING", "FOLLOW_STOPPED", "FOLLOW_DECEL",
    "COLLISION_STOPPED", "COLLISION_SLOWDOWN", "EMERGENCY_STOPPED", "EMERGENCY",
}
VARIABLE_FIELDS = {"position", "distance", "object", "direction", "pedestrian_state"}


def _roundabout_admitted(frame):
    return (frame.get("valid") is True and frame.get("dataset_build_candidate") is True
            and frame.get("object_inventory_usable") is True)


def _roundabout_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _roundabout_connected(previous, current):
    return (previous is not None and _roundabout_admitted(previous) and _roundabout_admitted(current)
            and current["index"] == previous["index"] + 1
            and all(_roundabout_finite(f.get("lidar_source_ns")) for f in (previous, current))
            and 0 < (current["lidar_source_ns"] - previous["lidar_source_ns"]) / 1e9 <= .25)


def enrich_roundabout_current_observation(frame, raw):
    """Supplement old text-log observations from admitted causal current GT.

    Original observations, including exact ROS observations, remain intact.
    The returned match copy and provenance make the supplemental source clear.
    """
    match = dict(raw)
    speed = frame.get("ego_speed_mps")
    if (raw.get("source") == "existing_roundabout_text_log"
            and "ego_speed_mps" not in raw and _roundabout_admitted(frame)
            and frame.get("ego_speed_source") == "backward_gt_ego_xy_difference"
            and _roundabout_finite(speed) and speed >= 0):
        match["ego_speed_mps"] = speed
        return match, {"field": "ego_speed_mps", "value": speed,
                       "source": "backward_gt_ego_xy_difference", "frame_index": frame["index"],
                       "lidar_source_ns": frame.get("lidar_source_ns"), "uses_future_ego": False}
    return match, None


def _roundabout_actor(frame, objects, actor_id):
    matches = [obj for obj in objects if actor_id is not None and obj.get("actor_id") is not None
               and str(obj["actor_id"]) == str(actor_id)]
    if not _roundabout_admitted(frame) or len(matches) != 1:
        return None
    actor = matches[0]
    if (actor.get("index") != frame["index"] or actor.get("lidar_source_ns") != frame.get("lidar_source_ns")
            or actor.get("rule_geometry_usable") is not True or actor.get("model_identity_verified") is not True
            or actor.get("semantic_class_valid") is not True or actor.get("semantic_class") != "vehicle"
            or actor.get("kind") not in ("npc", "vehicle") or not actor.get("instance_key")):
        return None
    return actor


def _roundabout_target_proof(frame, raw, objects):
    phase = raw.get("roundabout_phase")
    if phase == "follow":
        selected, current = raw.get("roundabout_front_id"), frame.get("lead_actor_id")
    elif phase in ("yield", "collision_slowdown", "emergency"):
        selected, current = raw.get("roundabout_conflict_id"), frame.get("conflict_actor_id")
    else:
        return None
    actor = _roundabout_actor(frame, objects, selected)
    if actor is None:
        return {"valid": False, "reason": "expert_selected_actor_current_identity_or_geometry_unverified",
                "selected_actor_id": selected, "current_actor_id": current}
    forward, left = actor.get("forward_m"), actor.get("left_m")
    if not all(_roundabout_finite(v) for v in (forward, left)):
        x, y, heading = frame.get("ego_world_x_m"), frame.get("ego_world_y_m"), frame.get("ego_heading_deg")
        ox, oy = actor.get("world_x_m"), actor.get("world_y_m")
        if all(_roundabout_finite(v) for v in (x, y, ox, oy, heading)):
            angle = math.radians(heading)
            forward = (ox-x)*math.cos(angle)+(oy-y)*math.sin(angle)
            left = -(ox-x)*math.sin(angle)+(oy-y)*math.cos(angle)
        else:
            return {"valid": False, "reason": "expert_selected_actor_current_position_unverified",
                    "selected_actor_id": selected, "current_actor_id": current}
    side = (("front" if forward >= 0 else "rear") if abs(forward) >= abs(left)
            else "left" if left > 0 else "right")
    return {"valid": True, "selected_actor_id": selected, "current_actor_id": current,
            "geometric_target_matches": current is not None and str(selected) == str(current),
            "instance_key": actor["instance_key"], "lidar_source_ns": frame["lidar_source_ns"],
            "forward_m": forward, "left_m": left, "side": side, "position": _position(side),
            "distance_m": math.hypot(forward, left)}


def _bind_roundabout_selected_target(rule, frame, objects, proof):
    """Keep the rendered actor description and its final identity together.

    The original geometric rule target remains separate evidence. In
    particular, a descriptive binding does not verify a legacy phase or erase
    the selected/current actor mismatch used by the causal action guard.
    """
    if not proof or proof.get("valid") is not True:
        return False
    actor = _roundabout_actor(frame, objects, proof.get("selected_actor_id"))
    if (actor is None or actor["instance_key"] != proof.get("instance_key")
            or proof.get("lidar_source_ns") != frame.get("lidar_source_ns")
            or not _roundabout_finite(proof.get("distance_m"))
            or proof.get("side") not in ("front", "rear", "left", "right")):
        return False
    evidence = rule.setdefault("evidence", {})
    variables = rule.get("variables") or {}
    evidence.setdefault("roundabout_rule_target_provenance", {
        "variables": deepcopy(variables),
        "related_actors": deepcopy(rule.get("related_actors")),
        "target_actor_ref": deepcopy(evidence.get("target_actor_ref")),
    })
    rule["variables"] = dict(variables,
        target_actor_id=actor["actor_id"], target_kind="vehicle",
        target_distance_m=proof["distance_m"], target_side=proof["side"],
        target_instance_key=actor["instance_key"])
    rule["related_actors"] = [actor["actor_id"]]
    evidence["target_actor_ref"] = {
        "actor_id": actor["actor_id"], "instance_key": actor["instance_key"]}
    evidence["roundabout_selected_target_binding"] = {
        "schema": "roundabout_selected_target_binding_v1", "valid": True,
        "frame_index": frame["index"], "lidar_source_ns": frame["lidar_source_ns"],
        "actor_id": actor["actor_id"], "instance_key": actor["instance_key"],
        "source": "current_sealed_gt_selected_actor_identity_and_geometry",
        "uses_future_ego": False, "verifies_legacy_phase_time": False,
    }
    return True


def roundabout_departure_proof(history, frame, raw, objects, response):
    """A phase flip is not a departure: recheck past motion and current actor."""
    proof = {"valid": False, "uses_future_ego": False}
    target = _roundabout_target_proof(frame, raw, objects)
    if response is None or raw.get("roundabout_phase") != "follow" or not target or not target["valid"]:
        return dict(proof, reason="current_follow_response_or_target_unverified")
    if (str(response["actor_id"]) != str(target["selected_actor_id"])
            or response["instance_key"] != target["instance_key"]):
        return dict(proof, reason="response_actor_changed_or_respawned")
    track = history + [frame]
    if len(track) < 2 or any(not _roundabout_connected(a, b) for a, b in zip(track, track[1:])):
        return dict(proof, reason="current_departure_track_unverified")
    first = next((f for f in reversed(track[:-1])
                  if .3 <= (frame["lidar_source_ns"] - f["lidar_source_ns"]) / 1e9 <= .6), None)
    if first is None or not all(_roundabout_finite(f.get(k)) for f in track
                                for k in ("ego_world_x_m", "ego_world_y_m", "ego_speed_mps")):
        return dict(proof, reason="current_departure_motion_unavailable")
    previous = track[-2]
    dt = (frame["lidar_source_ns"] - previous["lidar_source_ns"]) / 1e9
    instant = math.hypot(frame["ego_world_x_m"]-previous["ego_world_x_m"],
                         frame["ego_world_y_m"]-previous["ego_world_y_m"]) / dt
    window = (frame["lidar_source_ns"] - first["lidar_source_ns"]) / 1e9
    mean = math.hypot(frame["ego_world_x_m"]-first["ego_world_x_m"],
                      frame["ego_world_y_m"]-first["ego_world_y_m"]) / window
    speed = frame["ego_speed_mps"]
    accelerating = speed >= response["speed_mps"] + .3 and speed > previous["ego_speed_mps"]
    valid = (instant > .2 and mean > .2 and instant < 50
             and math.isclose(speed, instant, rel_tol=1e-4, abs_tol=1e-4) and accelerating)
    return dict(proof, valid=valid, reason=None if valid else "actual_departure_not_observed",
                response_index=response["index"], past_start_index=first["index"],
                current_speed_mps=instant, past_mean_speed_mps=mean,
                response_speed_mps=response["speed_mps"], target=target)


def load_config(config_dir=CONFIG):
    config_dir = Path(config_dir)
    catalog = yaml.safe_load((config_dir / "instruction_catalog.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((config_dir / "expert_state_mapping.yaml").read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not isinstance(mapping, dict):
        raise ValueError("instruction config must contain zone mappings")
    for zone, entries in mapping.items():
        if zone not in catalog or not isinstance(entries, list):
            raise ValueError("invalid expert mapping zone: " + str(zone))
        seen = set()
        for entry in entries:
            state, instruction_id = entry["state"], entry["instruction_id"]
            if state in seen:
                raise ValueError("duplicate canonical Expert state: " + zone + ":" + state)
            seen.add(state)
            instruction = catalog[zone].get(instruction_id)
            if instruction is None or instruction["label"] != entry["label"]:
                raise ValueError("Expert mapping has missing or inconsistent instruction: " + zone + ":" + state)
            if not isinstance(entry.get("match"), dict) or not entry["match"]:
                raise ValueError("Expert mapping requires nonempty match: " + state)
            if set(entry["match"]) - MATCH_OPERATORS - RAW_FIELDS:
                raise ValueError("unsupported Expert match field: " + state)
    for zone, entries in catalog.items():
        for key, item in entries.items():
            fields = {name for _, name, _, _ in Formatter().parse(item["instruction"]) if name}
            if fields != set(item["required_variables"]) or fields - VARIABLE_FIELDS:
                raise ValueError("instruction variables differ from placeholders: " + zone + ":" + key)
    return mapping, catalog


def _matches(spec, raw):
    for key, expected in spec.items():
        if key in ("planner_target_speed_mps_lte", "acc_target_speed_mps_lte"):
            if (key == "planner_target_speed_mps_lte"
                    and raw.get("source") == "existing_roundabout_text_log"):
                # A retained phase does not freshly observe its numerical
                # target, including a historical zero. A current stop must
                # instead have an independently observed current response.
                return False
            field = key[:-4]
            value = raw.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value > expected:
                return False
        elif key in ("planner_below_ego_mps", "acc_below_ego_mps"):
            if key == "planner_below_ego_mps" and raw.get("source") == "existing_roundabout_text_log":
                # Sparse ROUNDABOUT transitions persist phase/actor identity,
                # but their target is a historical value: target changes are
                # absent from the logger's dedup key. Comparing that target to
                # current Ego creates false FOLLOW_DECEL labels seconds later.
                # Exact per-tick observations keep their original comparison.
                return False
            field = "planner_target_speed_mps" if key.startswith("planner") else "acc_target_speed_mps"
            target, ego = raw.get(field), raw.get("ego_speed_mps")
            if (not isinstance(target, (int, float)) or not isinstance(ego, (int, float))
                    or not math.isfinite(target) or not math.isfinite(ego)
                    or ego-target < expected):
                return False
        elif key in ("ego_speed_mps_lte", "ego_speed_mps_gt"):
            speed = raw.get("ego_speed_mps")
            if (not isinstance(speed, (int, float)) or not math.isfinite(speed) or
                    (speed > expected if key.endswith("lte") else speed <= expected)):
                return False
        elif key == "acc_object_type_in":
            if raw.get("acc_object_type") not in expected:
                return False
        elif raw.get(key) != expected:
            return False
    return True


def canonical_state(zone, raw, mapping):
    matches = [entry for entry in mapping.get(zone, ()) if _matches(entry["match"], raw)]
    if not matches:
        return None, "expert_state_unmapped"
    # Ordered entries put stop and deceleration before their general fallback.
    return matches[0], None


_HIGHWAY_DECISION = re.compile(
    r"\[HighwayDecision\] pos=\(([-+\d.]+),([-+\d.]+)\).*?"
    r"speed=([-+\d.]+) target=([-+\d.]+) reason=([^\s]+)")
_HIGHWAY_ACC_REASONS = {"highway_lane_emergency_stop", "highway_lane_merge_acc"}


def _observation_lines(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                yield str(path), line_no, line


def load_expert_log(path):
    """Read existing sparse diagnostics or an independently supplied state journal."""
    exact, spatial = {}, []
    for source_path, line_no, line in _observation_lines(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(PREFIX):
            line = line[len(PREFIX):]
        elif not line.startswith("{"):
            match = _HIGHWAY_DECISION.search(line)
            if match and match.group(5) in _HIGHWAY_ACC_REASONS:
                x, y, speed, target = map(float, match.group(1, 2, 3, 4))
                spatial.append({"mode":"highway_lane", "highway_state":"ACC_FOLLOW",
                                "reason":match.group(5), "ego_xy":[x,y],
                                "ego_speed_mps":speed, "planner_target_speed_mps":target,
                                "source":"existing_highway_decision_log"})
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Expert observation JSON at line {line_no}") from error
        if not isinstance(row, dict):
            continue
        if row.get("kind") == "plan" and isinstance(row.get("decision"), dict):
            decision, ego = row["decision"], row.get("ego") or {}
            if isinstance(ego.get("xy"), list) and decision.get("phase"):
                spatial.append({"mode":"roundabout_entry",
                                "roundabout_phase":decision.get("phase"),
                                "roundabout_entering":decision.get("entering"),
                                "roundabout_committed":decision.get("committed"),
                                "roundabout_front_id":decision.get("front_id"),
                                "roundabout_conflict_id":decision.get("blocker_unique_id"),
                                "ego_xy":ego["xy"], "ego_speed_mps":ego.get("speed_mps"),
                                "planner_target_speed_mps":decision.get("target_velocity"),
                                "source":"existing_roundabout_debug"})
            continue
        if "ego_header_seq" not in row or "mode" not in row:
            continue
        seq, stamp = row["ego_header_seq"], row.get("ego_source_ns")
        if (not isinstance(seq, int) or isinstance(seq, bool) or
                not isinstance(stamp, int) or isinstance(stamp, bool)):
            raise ValueError(f"invalid Expert header seq/source timestamp at line {line_no}")
        values = exact.setdefault((seq, stamp), [])
        if row not in values:
            values.append(row)
    return exact, spatial


def _spatial_index(frames, observations, tolerance_m=0.5):
    """Accept only one-to-one position matches; stationary ties remain unknown."""
    grouped = {}
    for raw in observations:
        xy = raw.get("ego_xy")
        if (not isinstance(xy, list) or len(xy) < 2 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy[:2])):
            continue
        zone = "roundabout" if raw.get("source") == "existing_roundabout_debug" else None
        candidates = []
        for frame in frames:
            if zone is not None and frame.get("zone_id") != zone:
                continue
            if zone is None and frame.get("zone_id") not in ("highway", "tunnel"):
                continue
            x, y = frame.get("ego_world_x_m"), frame.get("ego_world_y_m")
            if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and
                    math.isfinite(x) and math.isfinite(y) and
                    math.hypot(x-xy[0], y-xy[1]) <= tolerance_m):
                candidates.append(frame["index"])
        if len(candidates) == 1:
            grouped.setdefault(candidates[0], []).append(raw)
    return grouped


def _ego_index(episode):
    path = Path(episode) / "quality" / "active_signals.json"
    signals = json.loads(path.read_text(encoding="utf-8"))
    by_stamp, by_seq = {}, {}
    for row in signals:
        if row.get("role") != "ego":
            continue
        stamp, seq = row.get("source_timestamp_ns"), row.get("seq")
        if not isinstance(stamp, int) or not isinstance(seq, int):
            continue
        by_stamp.setdefault(stamp, set()).add(seq)
        by_seq.setdefault(seq, set()).add(stamp)
    ordered = sorted(by_stamp)
    return ordered, by_stamp, by_seq


def _position(value):
    return POSITIONS.get(value, value if value in POSITIONS.values() else None)


def _variables(frame, rule, raw, objects=()):
    if (frame.get("zone_id") == "roundabout"
            and raw.get("roundabout_phase") == "rear_pressure"):
        selected_id = raw.get("roundabout_conflict_id")
        matches = [obj for obj in objects if selected_id is not None
                   and obj.get("actor_id") is not None
                   and str(obj["actor_id"]) == str(selected_id)]
        position = None
        if len(matches) == 1 and matches[0].get("rule_geometry_usable") is True:
            forward, left = matches[0].get("forward_m"), matches[0].get("left_m")
            if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in (forward, left)):
                if abs(forward) >= abs(left):
                    position = _position("front" if forward >= 0 else "rear")
                else:
                    position = _position("left" if left > 0 else "right")
        return dict(position=position)
    source = rule.get("variables") or {}
    target_kind = source.get("target_kind")
    target_distance = source.get("target_distance_m")
    zone = frame.get("zone_id")
    if zone == "roundabout":
        proof = _roundabout_target_proof(frame, raw, objects)
        if proof is not None and not proof["valid"]:
            # A position from another GT target cannot describe the Expert's
            # selected actor, even when the sentence needs only a position.
            return dict(position=None, distance=None, object=None,
                        direction=None, pedestrian_state=None)
        if proof is not None:
            # Raw selected identity is authoritative for this independently
            # aligned Expert observation. Missing route anchors may prevent a
            # geometric lead/conflict join; they do not change its actor.
            return dict(position=proof["position"], distance=round(proof["distance_m"]),
                        object="a vehicle", direction=None, pedestrian_state=None)
        phase = raw.get("roundabout_phase")
        if phase == "follow":
            selected_id = raw.get("roundabout_front_id")
            frame_id, frame_distance = frame.get("lead_actor_id"), frame.get("lead_distance_m")
        elif phase in ("yield", "collision_slowdown", "emergency"):
            selected_id = raw.get("roundabout_conflict_id")
            frame_id, frame_distance = frame.get("conflict_actor_id"), frame.get("conflict_distance_m")
        else:
            selected_id, frame_id, frame_distance = None, None, None
        if selected_id is not None and frame_id is not None and str(selected_id) == str(frame_id):
            target_distance = frame_distance
            target_kind = "vehicle"
        else:
            target_distance = None
            target_kind = None
    elif target_distance is None and zone == "static_obstacle":
        if frame.get("nearest_obstacle_id") is not None:
            target_distance = frame.get("nearest_obstacle_distance_m")
            target_kind = "obstacle"
    elif target_distance is None and zone in ("highway", "tunnel"):
        if frame.get("lead_actor_id") is not None:
            target_distance = frame.get("lead_distance_m")
            target_kind = "vehicle" if target_distance is not None else None
    if (not isinstance(target_distance, (int, float)) or isinstance(target_distance, bool)
            or not math.isfinite(target_distance) or target_distance < 0):
        target_distance = None
    position = _position(source.get("target_position") or source.get("target_position_text")
                         or source.get("target_side"))
    if position is None and target_distance is not None:
        position = "in front of"
    object_name = OBJECTS.get("pedestrian" if target_kind == "person" else target_kind)
    direction = DIRECTIONS.get(source.get("avoid_direction"))
    pedestrian_state = None
    if raw.get("pedestrian_released") is True:
        pedestrian_state = "The pedestrian situation has been resolved."
    elif target_kind in ("pedestrian", "person") and target_distance is not None:
        pedestrian_state = (f"There is a pedestrian approximately {round(target_distance)} m away "
                            f"{position or 'in front of'} the ego vehicle.")
    return dict(position=position,
                distance=None if target_distance is None else round(target_distance),
                object=object_name, direction=direction, pedestrian_state=pedestrian_state)

def apply_expert_instructions(frames, rules, episode, expert_log, *, max_gap_ns=25_000_000,
                              config_dir=CONFIG, index_observations=None, objects=()):
    """Add Expert fields and bind descriptions to their verified current actor.

    Original GT rule target provenance and rule decisions remain recorded.

    `index_observations`, if given, is a {frame_index: raw_dict} mapping
    already resolved to an exact frame (e.g. from
    expert_roundabout_phase.build_roundabout_observations's sim-time
    interpolation for episodes collected before Expert emitted a per-tick
    header-seq observation). It is tried after the header-seq exact match
    and before the weaker unique-position spatial match.
    """
    mapping, catalog = load_config(config_dir)
    objects_by_index = {}
    for obj in objects:
        objects_by_index.setdefault(obj["index"], []).append(obj)
    exact, observations = load_expert_log(expert_log)
    if exact:
        if episode is None:
            raise ValueError("--episode is required for header-seq Expert observations")
        stamps, stamp_seqs, _ = _ego_index(episode)
    else:
        stamps, stamp_seqs = [], {}
    spatial = _spatial_index(frames, observations)
    indexed = {row["index"]: row for row in frames}
    if len(indexed) != len(frames):
        raise ValueError("duplicate feature frame index")
    counts = Counter()
    prev_roundabout_responding = False
    prior_roundabout_index = None
    roundabout_history, roundabout_response = [], None
    previous_roundabout_frame = None
    for row in rules:
        frame = indexed[row["index"]]
        row.update(expert_state=None, expert_raw_state=None, label=None,
                   instruction_id=None, instruction_variables=None, instruction=None,
                   instruction_source=None, expert_alignment=None)
        zone = frame.get("zone_id")
        if (zone != "roundabout" or not _roundabout_connected(previous_roundabout_frame, frame)):
            roundabout_history, roundabout_response = [], None
        previous_roundabout_frame = frame if zone == "roundabout" and _roundabout_admitted(frame) else None
        if zone != "roundabout":
            prev_roundabout_responding = False
            prior_roundabout_index = None
        elif prior_roundabout_index is not None and row["index"] != prior_roundabout_index + 1:
            prev_roundabout_responding = False
        if zone == "roundabout":
            prior_roundabout_index = row["index"]
        if zone == "dynamic_obstacle":
            row["expert_alignment"] = {"status":"excluded", "reason":"dynamic_expert_state_unavailable"}
            counts["dynamic_expert_state_unavailable"] += 1
            continue
        if zone not in mapping:
            row["expert_alignment"] = {"status":"unknown", "reason":"expert_zone_unmapped"}
            counts["expert_zone_unmapped"] += 1
            continue
        raw, alignment = None, None
        reason = "expert_observation_missing"
        stamp = frame.get("lidar_source_ns")
        if exact:
            if not isinstance(stamp, int) or not stamps:
                reason = "ego_source_time_unavailable"
            else:
                pos = bisect_right(stamps, stamp)-1
                if pos < 0 or stamp-stamps[pos] > max_gap_ns:
                    reason = "ego_source_time_outside_tolerance"
                else:
                    candidates = stamp_seqs[stamps[pos]]
                    if len(candidates) != 1:
                        reason = "ego_seq_ambiguous"
                    else:
                        seq = next(iter(candidates))
                        observations_at_seq = exact.get((seq, stamps[pos]), ())
                        if len(observations_at_seq) != 1:
                            reason = "expert_state_missing_or_ambiguous"
                        else:
                            raw = observations_at_seq[0]
                            alignment = {"status":"matched", "method":"header_seq_and_source_time",
                                         "ego_header_seq":seq, "ego_source_ns":stamps[pos],
                                         "lidar_source_ns":stamp, "gap_ns":stamp-stamps[pos],
                                         "physical_simultaneity_verified":False}
                            ego_xy = raw.get("ego_xy")
                            x, y = frame.get("ego_world_x_m"), frame.get("ego_world_y_m")
                            if (isinstance(ego_xy, list) and len(ego_xy) >= 2 and
                                    isinstance(x, (int, float)) and isinstance(y, (int, float)) and
                                    all(isinstance(v, (int, float)) and math.isfinite(v)
                                        for v in (x, y, ego_xy[0], ego_xy[1]))):
                                error = math.hypot(x-ego_xy[0], y-ego_xy[1])
                                if error > 3.0:
                                    raw, alignment = None, None
                                    reason = "expert_ego_position_mismatch"
        if raw is None and index_observations is not None and row["index"] in index_observations:
            raw = index_observations[row["index"]]
            alignment = {"status":"matched", "method":"sim_time_interpolated",
                         "physical_simultaneity_verified":False}
        if raw is None and spatial.get(row["index"]):
            matches = spatial[row["index"]]
            if len(matches) == 1:
                raw = matches[0]
                alignment = {"status":"matched", "method":"unique_ego_position",
                             "position_tolerance_m":0.5,
                             "physical_simultaneity_verified":False}
            else:
                reason = "expert_position_ambiguous"
        if raw is None:
            if zone == "roundabout":
                prev_roundabout_responding = False
                roundabout_history, roundabout_response = [], None
            row["expert_alignment"] = {"status":"unknown", "reason":reason}
            counts[reason] += 1
            continue
        row["expert_raw_state"] = raw
        row["expert_alignment"] = alignment
        if zone == "roundabout" and raw.get("source") == "existing_roundabout_text_log":
            row.setdefault("evidence", {})["roundabout_legacy_numeric_target"] = {
                "historical_target_speed_mps": raw.get("planner_target_speed_mps"),
                "numeric_target_current_verified": False,
                "current_speed_delta_comparison_enabled": False,
                "reason": "phase_log_dedup_does_not_track_target_changes",
                "transition_observation": raw.get("legacy_transition_observation"),
            }
        match_raw = raw
        if zone == "roundabout":
            match_raw, speed_proof = enrich_roundabout_current_observation(frame, raw)
            if speed_proof is not None:
                row.setdefault("evidence", {})["roundabout_current_observation"] = speed_proof
            phase = frame.get("roundabout_phase")
            match_raw["roundabout_before_merge"] = (
                True if phase in ("approach", "entry", "merge") else
                False if phase in ("circulating", "exit") else None)
            row.setdefault("evidence", {})["roundabout_geometry_phase"] = {
                "phase": phase, "before_merge": match_raw["roundabout_before_merge"]}
            target_proof = _roundabout_target_proof(frame, raw, objects_by_index.get(row["index"], ()))
            if target_proof is not None:
                row.setdefault("evidence", {})["roundabout_expert_target"] = target_proof
            departure = roundabout_departure_proof(
                roundabout_history, frame, raw, objects_by_index.get(row["index"], ()), roundabout_response)
            match_raw["roundabout_resume_detected"] = departure["valid"]
            if roundabout_response is not None and raw.get("roundabout_phase") == "follow":
                row.setdefault("evidence", {})["roundabout_current_departure"] = departure
        entry, reason = canonical_state(frame.get("zone_id"), match_raw, mapping)
        if (zone == "roundabout" and raw.get("source") == "existing_roundabout_text_log"
                and raw.get("roundabout_phase") == "emergency"
                and (raw.get("legacy_transition_observation") or {}).get(
                    "phase_within_anchor_bounds") is not True):
            # The timestamp-less emergency transition may lie on either side
            # of this sensor frame. Neither a guessed emergency nor an
            # opposite departure command is a current plan observation.
            entry, reason = None, "expert_roundabout_emergency_transition_time_unverified"
        if zone == "roundabout" and raw.get("roundabout_phase") == "cruise":
            # The old before-merge bool collapses approach, entry and merge.
            # Use the current topology phase when available: approaching a
            # merge does not mean the ego is already merging. Unknown topology
            # keeps the intermediate legacy mapping; the final current-motion
            # guard rejects unsupported entry/merge claims separately.
            phase_state = {"approach": "CRUISE_APPROACH", "entry": "CRUISE_APPROACH",
                           "merge": "CRUISE_MERGE", "circulating": "CRUISE_AFTER_MERGE",
                           "exit": "CRUISE_AFTER_MERGE"}.get(frame.get("roundabout_phase"))
            if phase_state is not None:
                exact_phase_entries = [item for item in mapping[zone] if item["state"] == phase_state]
                if len(exact_phase_entries) == 1:
                    previous_state = entry.get("state") if entry is not None else None
                    entry, reason = exact_phase_entries[0], None
                    row.setdefault("evidence", {})["roundabout_current_phase_mapping"] = {
                        "phase": frame["roundabout_phase"], "frame_index": frame["index"],
                        "lidar_source_ns": frame.get("lidar_source_ns"),
                        "previous_mapped_state": previous_state, "selected_state": phase_state,
                        "source": "current_route_geometry_phase", "uses_future_ego": False}
        if zone == "roundabout":
            prev_roundabout_responding = (
                reason is None and entry["state"] in ROUNDABOUT_RESPONDING_STATES)
            front = _roundabout_actor(frame, objects_by_index.get(row["index"], ()), raw.get("roundabout_front_id"))
            if prev_roundabout_responding and front is not None and _roundabout_finite(frame.get("ego_speed_mps")):
                roundabout_response = dict(index=frame["index"], actor_id=front["actor_id"],
                                           instance_key=front["instance_key"], speed_mps=frame["ego_speed_mps"])
            elif (reason is not None or entry["state"] == "FOLLOW_RESUME" or front is None
                  or (roundabout_response is not None and
                      (str(front["actor_id"]) != str(roundabout_response["actor_id"])
                       or front["instance_key"] != roundabout_response["instance_key"]))):
                roundabout_response = None
            if _roundabout_admitted(frame):
                roundabout_history.append(frame)
                roundabout_history = [f for f in roundabout_history
                                      if (frame["lidar_source_ns"]-f["lidar_source_ns"])/1e9 <= .6]
        if reason is not None:
            row["expert_alignment"]["status"] = "unmapped"
            row["expert_alignment"]["reason"] = reason
            counts[reason] += 1
            continue
        template = catalog[frame["zone_id"]][entry["instruction_id"]]
        row["expert_state"] = entry["state"]
        row["expert_raw_state"] = raw
        row["label"] = entry["label"]
        row["instruction_id"] = entry["instruction_id"]
        values = _variables(frame, row, raw, objects_by_index.get(row["index"], ()))
        rendered, selected, missing = render_template(template, values)
        row["instruction_variables"] = selected
        if missing:
            row["expert_alignment"]["instruction_status"] = "missing_variables"
            row["expert_alignment"]["missing_variables"] = missing
            counts["instruction_variables_missing"] += 1
            if zone == "roundabout" and target_proof is not None and not target_proof["valid"]:
                row["expert_alignment"]["current_actor_reason"] = target_proof["reason"]
        else:
            row["instruction"] = rendered
            row["instruction_source"] = "expert_observation"
            if (zone == "roundabout" and raw.get("roundabout_phase") == "follow"
                    and roundabout_response is not None and not departure["valid"]
                    and _roundabout_finite(frame.get("ego_speed_mps"))
                    and frame["ego_speed_mps"] <= .2 and entry["state"] != "FOLLOW_STOPPED"):
                row["instruction"] = None
                row["expert_alignment"]["instruction_status"] = "current_departure_unverified"
                counts["current_departure_unverified"] += 1
                continue
            if zone == "roundabout":
                _bind_roundabout_selected_target(
                    row, frame, objects_by_index.get(row["index"], ()), target_proof)
            counts["instruction_rendered"] += 1
        counts["expert_state_mapped"] += 1
    return dict(sorted(counts.items()))


def hold_roundabout_instruction_across_brief_tracking_gaps(rules, max_gap=3):
    """Record a blocked stale hold without copying an action or actor.

    A short association gap is still missing current-frame evidence. A later
    gap pass may rerender a verified current tuple, but cannot borrow a prior
    Expert state. Keep this API for callers and auditable legacy comparisons.
    """
    previous = None
    for row in rules:
        if (max_gap > 0 and previous is not None
                and previous.get("zone_id") == row.get("zone_id") == "roundabout"
                and row["index"] == previous["index"] + 1
                and previous.get("instruction") and row.get("instruction") is None):
            evidence = row.get("evidence")
            row["evidence"] = dict(evidence) if isinstance(evidence, dict) else {}
            row["evidence"]["instruction_hold_blocked"] = {
                "from_index": previous["index"], "reason": "current_frame_evidence_required"}
        previous = row
