"""Postprocess existing Expert observations into verified frame instructions.

No collector or Expert runtime writes are required. Sparse logs yield sparse
labels; absent planner states are never inferred from GT rule candidates.
"""
from bisect import bisect_right
from collections import Counter
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
              "roundabout_resume_detected"}
# roundabout_resume_detected is never present in a real Expert observation --
# apply_expert_instructions injects it on a copy of `raw` only on the single
# frame after a yield/decel/stop response resolves back to plain following,
# so FOLLOW_RESUME is selected through the same declarative canonical_state
# matching as every other state, not a separate code path.
ROUNDABOUT_RESPONDING_STATES = {
    "YIELD_STOPPED", "YIELD_MOVING", "FOLLOW_STOPPED", "FOLLOW_DECEL",
    "COLLISION_STOPPED", "COLLISION_SLOWDOWN", "EMERGENCY_STOPPED", "EMERGENCY",
}
VARIABLE_FIELDS = {"position", "distance", "object", "direction", "pedestrian_state"}


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
            field = key[:-4]
            value = raw.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value > expected:
                return False
        elif key in ("planner_below_ego_mps", "acc_below_ego_mps"):
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


def _variables(frame, rule, raw):
    source = rule.get("variables") or {}
    target_kind = source.get("target_kind")
    target_distance = source.get("target_distance_m")
    zone = frame.get("zone_id")
    if zone == "roundabout":
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
                              config_dir=CONFIG, index_observations=None):
    """Add Expert fields to rule rows without changing GT rule decisions.

    `index_observations`, if given, is a {frame_index: raw_dict} mapping
    already resolved to an exact frame (e.g. from
    expert_roundabout_phase.build_roundabout_observations's sim-time
    interpolation for episodes collected before Expert emitted a per-tick
    header-seq observation). It is tried after the header-seq exact match
    and before the weaker unique-position spatial match.
    """
    mapping, catalog = load_config(config_dir)
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
    for row in rules:
        frame = indexed[row["index"]]
        row.update(expert_state=None, expert_raw_state=None, label=None,
                   instruction_id=None, instruction_variables=None, instruction=None,
                   instruction_source=None, expert_alignment=None)
        zone = frame.get("zone_id")
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
            row["expert_alignment"] = {"status":"unknown", "reason":reason}
            counts[reason] += 1
            continue
        row["expert_raw_state"] = raw
        row["expert_alignment"] = alignment
        match_raw = raw
        if zone == "roundabout" and prev_roundabout_responding:
            # A yield/decel/stop response just resolved back to plain
            # following -- match as FOLLOW_RESUME instead of FOLLOW through
            # the same declarative canonical_state lookup, not a separate
            # code path (see ROUNDABOUT_RESPONDING_STATES above).
            match_raw = dict(raw, roundabout_resume_detected=True)
        entry, reason = canonical_state(frame.get("zone_id"), match_raw, mapping)
        if zone == "roundabout":
            prev_roundabout_responding = (
                reason is None and entry["state"] in ROUNDABOUT_RESPONDING_STATES)
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
        values = _variables(frame, row, raw)
        rendered, selected, missing = render_template(template, values)
        row["instruction_variables"] = selected
        if missing:
            row["expert_alignment"]["instruction_status"] = "missing_variables"
            row["expert_alignment"]["missing_variables"] = missing
            counts["instruction_variables_missing"] += 1
        else:
            row["instruction"] = rendered
            row["instruction_source"] = "expert_observation"
            counts["instruction_rendered"] += 1
        counts["expert_state_mapped"] += 1
    return dict(sorted(counts.items()))
