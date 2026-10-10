"""Align situation instructions with the ego's own recorded motion.

Training-label hygiene for Stage 2, which learns the future trajectory: an
instruction that disagrees with that trajectory teaches the model to ignore
instructions. Legacy checks use recorded future poses. The isolated current
response policy revalidates marked static/dynamic rows using current inventory
and past GT; legacy future-window checks cannot overwrite those decisions.

Before sentences are attached (states only):
  align_highway_lane_changes   lane change spans and sides from the ego's lateral motion
  align_static_avoidance       avoid starts when the swing starts; return once Ego moves back
  hold_dynamic_wait_until_moving  after yielding, "remain stopped" until Ego actually moves
After sentences are attached:
  fill_short_instruction_gaps  restore a missing rendering only from that frame's own verified state
  smooth_short_instruction_islands  repair rendering noise without changing an observed action
  use_stopped_roundabout_sentences  roundabout yield/follow/slow-down while standing -> "remain stopped"
  drop_contradicted_instructions  null a sentence the next 2.5 s contradicts (>= 0.3 s runs)
"""

from bisect import bisect_left
from collections import Counter, defaultdict
import copy
import re

from .renderer import load_catalog, render_instruction, render_template

HORIZON_S = 2.5
HORIZON_TOLERANCE_S = 0.3
MAX_TRACK_INTERVAL_S = 0.25
SPEED_BACK_S = 0.5
SIDE_MOTION_M = 0.3        # lateral motion that starts a swing / contradicts a direction
STATIONARY_PROGRESS_M = 1.0
KEEP_GOING_MIN_SPEED_MPS = 2.0
KEEP_GOING_SPEED_RATIO = 0.4
STOPPED_SPEED_MPS = 1.0
MIN_CONTRADICTION_FRAMES = 3
MIN_KEPT_ISLAND_FRAMES = 5   # a shorter kept stretch inside one dropped episode is dropped too
BACKWARD_JITTER_M = 0.5      # a standing Ego's position noise; not a reversal
MAX_FILLED_GAP_FRAMES = 3
AVOID_START_FRAMES = 5     # "perform an avoidance maneuver" before "continue"
RESUME_SPEED_MPS = 3.0
WAIT_PROGRESS_M = 1.5
MAX_ISLAND_FRAMES = 2
LANE_CHANGE_MIN_SHIFT_M = 2.0      # a lateral move this large crosses a lane boundary
LATERAL_SPEED_MPS = 0.25
LATERAL_SPEED_HALF_WINDOW = 5      # frames each side for the lateral speed
HIGHWAY_KEEP_FOLLOW_IDS = ("keep_lane_normal", "keep_lane_object", "acc_follow")
HIGHWAY_MAX_ISLAND_FRAMES = 9      # keep/follow alternation shorter than 1 s
APPROACH_RANGE_M = 35.0
RESPONSE_STATES = ("hard_braking", "stopped", "yielding", "decelerating")
STANDING_SPEED_MPS = 0.5
ROUNDABOUT_STOPPED = {"yield_moving": "yield_stopped", "follow": "follow_stopped",
                      "collision_slowdown": "collision_stopped"}
ROUNDABOUT_STOPPED_STATES = {"yield_stopped": "wait_conflict", "follow_stopped": "wait_lead",
                            "collision_stopped": "wait_conflict"}
MANEUVER_WORDS = ("avoid", "between", "Return to", "Resume", "Move into")
KEEP_WORDS = ("Continue driving", "Maintain the current")
YIELD_STATES = ("yield_start", "yielding", "waiting")
# The side a maneuver goes to, not the obstacle's position ("an obstacle to
# the left of the ego vehicle. ... maneuver to the right").
MANEUVER_SIDE = re.compile(r"(?:maneuver to the|into the) (left|right)")


class _Track:
    """Ego route distance s and lateral offset d (+ left) by source time."""

    def __init__(self, frames):
        # A future position beyond an invalid sample or a collection gap is
        # not a continuous trajectory observation for this frame.
        eligible = [f for f in frames if f.get("valid") is not False
                    and f.get("dataset_build_candidate") is not False
                    and all(_finite(f.get(key)) for key in
                            ("lidar_source_ns", "ego_path_s_m", "ego_path_d_m", "index"))]
        indices = Counter(f["index"] for f in eligible)
        stamps = Counter(f["lidar_source_ns"] for f in eligible)
        rows = sorted((f["lidar_source_ns"] / 1e9, f["ego_path_s_m"], f["ego_path_d_m"], f["index"])
                      for f in eligible if indices[f["index"]] == 1
                      and stamps[f["lidar_source_ns"]] == 1)
        self.t = [row[0] for row in rows]
        self.s = [row[1] for row in rows]
        self.d = [row[2] for row in rows]
        self.indices = [row[3] for row in rows]
        self.position = {row[3]: n for n, row in enumerate(rows)}
        self.segment = []
        segment = 0
        for n, row in enumerate(rows):
            if n and (row[3] != rows[n - 1][3] + 1
                      or not 0 < row[0] - rows[n - 1][0] <= MAX_TRACK_INTERVAL_S):
                segment += 1
            self.segment.append(segment)

    def connected(self, first, last):
        return (first is not None and last is not None
                and 0 <= first <= last < len(self.t)
                and self.segment[first] == self.segment[last])

    def at(self, index, offset_s):
        """Position n of the frame offset_s after frame `index`, or None."""
        n = self.position.get(index)
        if n is None:
            return None
        j = bisect_left(self.t, self.t[n] + offset_s)
        if (j >= len(self.t) or not self.connected(n, j)
                or abs(self.t[j] - self.t[n] - offset_s) > HORIZON_TOLERANCE_S):
            return None
        return j

    def speed(self, n):
        if n is None or not 0 <= n < len(self.t):
            return None
        p = bisect_left(self.t, self.t[n] - SPEED_BACK_S)
        return ((self.s[n] - self.s[p]) / (self.t[n] - self.t[p])
                if self.connected(p, n) and self.t[n] - self.t[p] > 0.3 else None)

    def future(self, index):
        """(route progress, lateral change) over the horizon, or None."""
        n, j = self.position.get(index), self.at(index, HORIZON_S)
        if n is None or j is None or self.s[j] < self.s[n] - BACKWARD_JITTER_M:
            return None
        return max(0.0, self.s[j] - self.s[n]), self.d[j] - self.d[n]


def _spans(rules, keep, track=None):
    span = []
    for row in rules:
        if keep(row):
            if span and (row["index"] != span[-1]["index"] + 1
                         or (track is not None and not track.connected(
                             track.position.get(span[-1]["index"]), track.position.get(row["index"])))):
                yield span
                span = []
            span.append(row)
        elif span:
            yield span
            span = []
    if span:
        yield span


def _motion_row_usable(row, frame):
    if frame is None or row.get("status", "matched") != "matched":
        return False
    for scope in (row, frame, row.get("variables") or {}, row.get("evidence") or {}):
        if any(scope.get(key) is False for key in
               ("valid", "dataset_build_candidate", "object_inventory_usable", "target_semantic_valid",
                "semantic_valid", "observation_valid", "action_valid")):
            return False
    return True


def align_static_avoidance(frames, rules):
    """Avoid sentences follow the swing: maintain -> avoid -> continue -> return."""
    track, counts = _Track(frames), Counter()
    indexed = {frame["index"]: frame for frame in frames}
    avoid_span = lambda row: (row.get("zone_id") == "static_obstacle"
                              and not row.get("evidence", {}).get("current_static_maneuver_policy")
                              and _motion_row_usable(row, indexed.get(row["index"]))
                              and row.get("instruction_state") in ("avoid", "avoid_continue")
                              and (row.get("variables") or {}).get("avoid_direction") in ("left", "right"))
    for span in _spans(rules, avoid_span, track):
        sign = 1.0 if span[0]["variables"]["avoid_direction"] == "left" else -1.0
        n0 = track.position.get(span[0]["index"])
        if n0 is None:
            continue
        base = track.d[n0]
        displaced = lambda row: (row["index"] in track.position
                                 and sign * (track.d[track.position[row["index"]]] - base) >= SIDE_MOTION_M)
        swinging = lambda row: (track.future(row["index"]) is not None
                                and sign * track.future(row["index"])[1] >= SIDE_MOTION_M)
        start = next((k for k, row in enumerate(span) if displaced(row) or swinging(row)), None)
        if start is None:
            for row in span:
                row["instruction_state"] = "approach_maintain"
            counts["span_without_swing"] += 1
            continue
        offsets = [sign * (track.d[track.position[row["index"]]] - base)
                   if row["index"] in track.position else float("-inf") for row in span]
        # Return once the next 2.5 s bring Ego back toward the path while it is
        # still displaced -- the same horizon the trajectory target uses.
        back = next((k for k in range(start + AVOID_START_FRAMES, len(span))
                     if track.future(span[k]["index"]) is not None
                     and -sign * track.future(span[k]["index"])[1] >= SIDE_MOTION_M
                     and offsets[k] >= SIDE_MOTION_M), None)
        for k, row in enumerate(span):
            state = ("approach_maintain" if k < start else
                     "return_to_path" if back is not None and k >= back else
                     "avoid" if k < start + AVOID_START_FRAMES else "avoid_continue")
            if state != row["instruction_state"]:
                row.setdefault("evidence", {})["rule_state_before_motion_alignment"] = row["instruction_state"]
                row["instruction_state"] = state
                counts["to_" + state] += 1
    return dict(sorted(counts.items()))


def _braking(track, index):
    """Ego above 2 m/s now and below 40 % of that speed 2.5 s later."""
    n, j = track.position.get(index), track.at(index, HORIZON_S)
    speed = track.speed(n) if n is not None else None
    if j is None or speed is None or speed <= KEEP_GOING_MIN_SPEED_MPS:
        return False
    v_future = track.speed(j)
    return v_future is not None and v_future < KEEP_GOING_SPEED_RATIO * speed


def _lateral_bursts(track, rows):
    """(start, end, sign) runs of rows whose lateral speed exceeds 0.25 m/s one way."""
    signs = []
    for row in rows:
        n = track.position.get(row["index"])
        sign = 0
        if n is not None:
            a = max(0, n - LATERAL_SPEED_HALF_WINDOW)
            b = min(len(track.t) - 1, n + LATERAL_SPEED_HALF_WINDOW)
            if track.connected(a, b) and track.t[b] - track.t[a] > 0.3:
                v = (track.d[b] - track.d[a]) / (track.t[b] - track.t[a])
                sign = 1 if v >= LATERAL_SPEED_MPS else -1 if v <= -LATERAL_SPEED_MPS else 0
        signs.append(sign)
    bursts, k = [], 0
    while k < len(rows):
        if signs[k] == 0:
            k += 1
            continue
        start, end, gap = k, k, 0
        m = k + 1
        while m < len(rows) and rows[m]["index"] == rows[m - 1]["index"] + 1:
            if not track.connected(track.position.get(rows[m - 1]["index"]),
                                   track.position.get(rows[m]["index"])):
                break
            if signs[m] == signs[k]:
                end, gap = m, 0
            elif signs[m] == 0 and gap < 3:
                gap += 1
            else:
                break
            m += 1
        bursts.append((start, end, signs[k]))
        k = end + 1
    return bursts


def align_highway_lane_changes(frames, rules, objects):
    """Lane change spans and sides from the lateral motion; stale changes become keep/follow.

    The map-link state machine can stay "changing left" for tens of seconds
    after the move (or through the return to the original lane). A span is a
    lateral move of >= 2 m on the route frame; it starts when the next 2.5 s
    bring >= 0.3 m of that move. Moving back toward the route lane is the
    return to the original lane.
    """
    track, counts = _Track(frames), Counter()
    indexed = {frame["index"]: frame for frame in frames}
    rows = [row for row in rules if row.get("zone_id") == "highway"
            and _motion_row_usable(row, indexed.get(row["index"]))]
    ahead = defaultdict(list)
    current_inventory = defaultdict(list)
    for o in objects:
        current_inventory[o.get("index")].append(o)
        if (o.get("kind") == "npc" and o.get("rule_geometry_usable") is True
                and _finite(o.get("distance_to_ego_m"))
                and 0 < o["distance_to_ego_m"] <= APPROACH_RANGE_M
                and _finite(o.get("forward_m")) and o["forward_m"] > 0):
            ahead[o["index"]].append(o)
    in_span = set()
    for start, end, sign in _lateral_bursts(track, rows):
        n0, n1 = track.position.get(rows[start]["index"]), track.position.get(rows[end]["index"])
        if n0 is None or n1 is None or abs(track.d[n1] - track.d[n0]) < LANE_CHANGE_MIN_SHIFT_M:
            continue
        first = start
        for k in range(start - 1, max(-1, start - 26), -1):
            if rows[k]["index"] != rows[k + 1]["index"] - 1:
                break
            future = track.future(rows[k]["index"])
            if future is None or sign * future[1] < SIDE_MOTION_M:
                break
            first = k
        returning = abs(track.d[n1]) < abs(track.d[n0]) - 1.5
        side = "left" if sign > 0 else "right"
        counts["lane_change_spans"] += 1
        counts["return_spans" if returning else "change_spans"] += 1
        for row in rows[first:end + 1]:
            if row.get("status") == "invalid_frame":
                continue
            in_span.add(row["index"])
            keep_state = row.get("instruction_state") in RESPONSE_STATES[:3]
            before = (row.get("label_candidate"), row.get("instruction_state"))
            row["label_candidate"] = "CHANGE_" + side.upper()
            if not keep_state:
                row["instruction_state"] = "return_to_source_lane" if returning else "lane_change"
            row["status"], row["unknown_reasons"] = "matched", []
            row.setdefault("variables", {})["lane_change_direction"] = side
            if before != (row["label_candidate"], row["instruction_state"]):
                row.setdefault("evidence", {})["rule_state_before_motion_alignment"] = "/".join(map(str, before))
                counts["frames_to_lane_change"] += 1
    for row in rows:
        label = row.get("label_candidate") or ""
        if row["index"] in in_span or not label.startswith("CHANGE_"):
            continue
        variables = row.setdefault("variables", {})
        reason, state = variables.get("response_reason"), row.get("instruction_state")
        before = label + "/" + str(state)
        if reason in ("lead_vehicle", "merging_vehicle") and state in RESPONSE_STATES:
            row["label_candidate"] = "ACC_FOLLOW"
        elif reason == "lead_vehicle":
            row["label_candidate"], row["instruction_state"] = "ACC_FOLLOW", "follow"
        elif state in RESPONSE_STATES:
            row.update(label_candidate=None, instruction_state=None, status="unknown",
                       unknown_reasons=["ego_response_cause_unverified"])
        else:
            frame = indexed[row["index"]]
            inventory = current_inventory.get(row["index"], [])
            from .highway_target_selection import select_current_highway_target
            actor, role, selection_reasons, selection_receipt = select_current_highway_target(frame, inventory)
            row.setdefault("evidence", {})["current_highway_target_selection"] = selection_receipt
            is_lead = role == "lead_vehicle"
            if selection_reasons:
                row.update(label_candidate=None, instruction_state=None, status="unknown")
                row.setdefault("unknown_reasons", []).extend(
                    reason for reason in selection_reasons if reason not in row.get("unknown_reasons", []))
            if actor is not None:
                # A forward actor in an adjacent lane is not automatically
                # the current-lane lead. Preserve its actual current side.
                from .highway_position import body_position
                position = body_position(actor)
                variables.update(target_actor_id=actor.get("actor_id"), target_kind="vehicle",
                                 target_position=position, target_position_text=position,
                                 target_distance_m=actor["distance_to_ego_m"],
                                 response_reason="lead_vehicle" if is_lead else "approaching_vehicle")
                row["related_actors"] = [{"actor_id": actor["actor_id"],
                                          "role": "lead_vehicle" if is_lead else "approaching_vehicle"}]
                speed = frame.get("ego_speed_mps")
                following = is_lead and _finite(speed) and actor["distance_to_ego_m"] <= max(20.0, 2.0*speed)
                responding = (any(frame.get(key) is True for key in
                                  ("ego_stopped", "ego_decelerating", "ego_hard_braking"))
                              or _finite(speed) and speed <= .2)
                if responding:
                    # Seeing a distant lead does not establish why Ego has
                    # stopped/braked. Existing cause-supported response rows
                    # are preserved above; never turn an unexplained stop
                    # into either Maintain or a lead-caused stopping sentence.
                    row.update(label_candidate=None, instruction_state=None, status="unknown")
                    row.setdefault("unknown_reasons", []).append("ego_response_cause_unverified")
                else:
                    row["label_candidate"], row["instruction_state"] = (
                        ("ACC_FOLLOW", "follow") if following else ("KEEP_LANE", "approach_keep"))
                row.setdefault("evidence", {})["stale_highway_current_target"] = {
                    "actor_id": actor["actor_id"], "instance_key": actor["instance_key"],
                    "lidar_source_ns": actor["lidar_source_ns"], "position": position,
                    "forward_m": actor["forward_m"], "left_m": actor["left_m"],
                    "current_lane_lead": is_lead, "uses_neighbor_sentence": False}
            elif not selection_reasons:
                row["label_candidate"], row["instruction_state"] = "KEEP_LANE", "normal"
                for key in ("target_actor_id", "target_kind", "target_position", "target_position_text", "target_distance_m", "response_reason"):
                    variables[key] = None
        variables["lane_change_direction"] = None
        row.setdefault("evidence", {})["rule_state_before_motion_alignment"] = before
        counts["stale_change_to_" + str(row.get("instruction_state"))] += 1
    return dict(sorted(counts.items()))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and abs(value) != float("inf")


def align_dynamic_yield_onset(frames, rules):
    """Preserve the existing approach-to-yield alignment independently of resume.

    This legacy onset check still uses future motion. Separating it prevents
    the targeted current resume fix from deleting previously supported yields.
    """
    track, counts = _Track(frames), Counter()
    indexed = {frame["index"]: frame for frame in frames}
    # 1. An approach span that ends in a yield: the yield began when Ego
    #    started braking, before the conflict test fired. Converted once per
    #    span from the first braking frame, so the state cannot alternate.
    by_index = {row["index"]: k for k, row in enumerate(rules)}
    approach = lambda row: (row.get("zone_id") == "dynamic_obstacle"
                            and _motion_row_usable(row, indexed.get(row["index"]))
                            and row.get("instruction_state") == "approach")
    for span in _spans(rules, approach, track):
        after = by_index.get(span[-1]["index"] + 1)
        if (after is None or rules[after].get("instruction_state") not in YIELD_STATES
                or not _motion_row_usable(rules[after], indexed.get(rules[after]["index"]))
                or not track.connected(track.position.get(span[-1]["index"]),
                                       track.position.get(rules[after]["index"]))):
            continue
        start = next((k for k, row in enumerate(span) if _braking(track, row["index"])), None)
        for row in span[start:] if start is not None else ():
            row.setdefault("evidence", {})["rule_state_before_motion_alignment"] = "approach"
            row["instruction_state"] = "yielding"
            counts["to_yielding"] += 1
    return dict(sorted(counts.items()))


def hold_dynamic_wait_until_moving(frames, rules):
    """Legacy yield + post-yield alignment, retained for legacy callers/tests."""
    track = _Track(frames)
    indexed = {frame["index"]: frame for frame in frames}
    counts = Counter(align_dynamic_yield_onset(frames, rules))
    # 2. After a yield: remain stopped until Ego moves, resume while slow, then normal.
    after_yield, previous = False, None
    for row in rules:
        state = row.get("instruction_state")
        n = track.position.get(row["index"])
        if (previous is not None and (row["index"] != previous + 1
                or not track.connected(track.position.get(previous), n))):
            after_yield = False
        previous = row["index"]
        if (row.get("zone_id") != "dynamic_obstacle"
                or not _motion_row_usable(row, indexed.get(row["index"]))):
            after_yield = False
            continue
        if state in YIELD_STATES:
            after_yield = True
            continue
        if not after_yield or state not in ("normal", "approach", "resume"):
            continue
        future, n = track.future(row["index"]), track.position.get(row["index"])
        speed = track.speed(n) if n is not None else None
        if future is None or speed is None:
            continue
        new = ("wait_clear" if future[0] < WAIT_PROGRESS_M else
               "resume" if speed < RESUME_SPEED_MPS else None)
        if new is None:
            after_yield = False
            continue
        if new != state:
            row.setdefault("evidence", {})["rule_state_before_motion_alignment"] = state
            row["instruction_state"] = new
            row["status"] = "matched"
            counts["to_" + new] += 1
    return dict(sorted(counts.items()))


_INSTRUCTION_HISTORY_FIELDS = (
    "instruction_state", "instruction_id", "instruction", "instruction_variables",
    "instruction_source", "instruction_label", "instruction_mapping_status",
    "instruction_consistency", "status", "label", "label_candidate", "behavior_state",
)


def _history(row, operation, **details):
    """Keep typed targets together in the audit trail, not just the old text."""
    row.setdefault("instruction_postprocessing_history", []).append({
        "operation": operation,
        "before": {key: copy.deepcopy(row.get(key)) for key in _INSTRUCTION_HISTORY_FIELDS},
        **copy.deepcopy(details),
    })


def _frames_by_index(frames):
    if frames is None:
        return None
    indexed = {row["index"]: row for row in frames}
    if len(indexed) != len(frames):
        raise ValueError("duplicate frame index in instruction postprocessing")
    return indexed


def _trusted_rendering(row, frame=None):
    """A neighbor cannot supply a state, actor, or an exclusion override."""
    if (row.get("status") != "matched" or not row.get("instruction_state")
            or not row.get("instruction_id") or not row.get("instruction_source")
            or row.get("instruction_dropped") is not None
            or str(row.get("instruction_consistency", "")).startswith("dropped_")
            or row.get("instruction_mapping_status") in
            ("rule_state_unavailable", "label_rule_state_unmapped", "expert_label_rule_state_unmapped")):
        return False
    for scope in (row, row.get("variables") or {}, row.get("evidence") or {}, frame or {}):
        if any(scope.get(key) is False for key in
               ("valid", "frame_valid", "dataset_build_candidate", "object_inventory_usable",
                "semantic_valid", "target_semantic_valid", "observation_valid", "action_valid")):
            return False
    # Descriptive approach labels can explicitly lack a future safety certificate.
    # That is different from missing actor identity, geometry, or sensor evidence.
    permitted = {"future_avoidance_trajectory_unverified", "future_pedestrian_motion_unverified"}
    if set(row.get("unknown_reasons") or ()) - permitted:
        return False
    return True


def _own_rendering(row, catalog, mapping):
    """Render from this frame's slots; never inherit another actor's location."""
    zone, instruction_id = row.get("zone_id"), row.get("instruction_id")
    template = catalog.get(zone, {}).get(instruction_id)
    if template is None:
        return None
    label = row.get("label")
    if label is None and zone == "highway":
        label = row.get("label_candidate")
    if template.get("label") != label:
        return None
    if mapping.get((zone, label, row.get("instruction_state"))) != instruction_id:
        return None
    if "variables" in row:
        text, values, missing = render_instruction(catalog, zone, instruction_id, row["variables"])
    else:
        text, values, missing = render_template(template, row.get("instruction_variables") or {})
    return None if missing else (text, values)


def _target_identity(row, required_variables):
    """Even equal nouns/positions must refer to the same actor incarnation."""
    evidence = row.get("evidence") or {}
    refs = evidence.get("corridor_pair_actor_refs")
    if not refs:
        ref = evidence.get("target_actor_ref")
        refs = [ref] if ref else []
    if refs:
        if any(not ref.get("instance_key") or ref.get("actor_id") is None for ref in refs):
            return None
        return tuple(sorted((str(ref["actor_id"]), ref["instance_key"]) for ref in refs))
    if required_variables or row.get("related_actors"):
        return None
    return ()


def _same_verified_rendering(rows, frames, catalog, mapping):
    signatures, renderings = [], []
    for row in rows:
        frame = None if frames is None else frames.get(row["index"])
        if frames is not None and frame is None:
            return None
        if not _trusted_rendering(row, frame):
            return None
        rendered = _own_rendering(row, catalog, mapping)
        if rendered is None:
            return None
        text, values = rendered
        target = _target_identity(row, catalog[row["zone_id"]][row["instruction_id"]]["required_variables"])
        if target is None:
            return None
        signatures.append((row.get("zone_id"), row.get("instruction_state"), row.get("instruction_id"),
                           row.get("label"), row.get("label_candidate"), row.get("behavior_state"),
                           row.get("instruction_source"), row.get("instruction_label"), target,
                           tuple(sorted(values.items()))))
        renderings.append((text, values))
    if any(signature != signatures[0] for signature in signatures[1:]):
        return None
    for before, after in zip(rows, rows[1:]):
        if after["index"] != before["index"] + 1:
            return None
        if frames is not None:
            dt = (frames[after["index"]]["lidar_source_ns"]
                  - frames[before["index"]]["lidar_source_ns"]) / 1e9
            if not 0 < dt <= MAX_TRACK_INTERVAL_S:
                return None
    return renderings


def fill_short_instruction_gaps(rules, frames=None):
    """Restore only a missing rendering of an already resolved, identical state.

    Unknown inventories, unknown states and deliberate exclusions remain null.
    Agreement between neighbors never certifies an unobserved frame as normal.
    """
    from .mapper import load_label_instruction_map
    catalog, indexed = load_catalog(), _frames_by_index(frames)
    mapping = load_label_instruction_map(catalog=catalog)
    filled, previous = 0, None
    for k, row in enumerate(rules):
        if row.get("instruction"):
            previous = k
            continue
        if previous is None:
            continue
        nxt = next((m for m in range(k, min(len(rules), k + MAX_FILLED_GAP_FRAMES + 1))
                    if rules[m].get("instruction")), None)
        if (nxt is None or nxt - previous - 1 > MAX_FILLED_GAP_FRAMES
                or rules[nxt].get("instruction") != rules[previous].get("instruction")):
            continue
        span = rules[previous:nxt + 1]
        rendered = _same_verified_rendering(span, indexed, catalog, mapping)
        if rendered is None or rendered[0][0] != rules[previous]["instruction"]:
            continue
        for gap, (text, values) in zip(span[1:-1], rendered[1:-1]):
            _history(gap, "restore_own_rendering", agreeing_indices=[span[0]["index"], span[-1]["index"]])
            gap.update(instruction=text, instruction_variables=values, instruction_mapping_status="rendered")
            gap["instruction_gap_filled_from"] = span[0]["index"]
            filled += 1
    return filled


def smooth_short_instruction_islands(rules, max_frames=MAX_ISLAND_FRAMES, zone=None, ids=None,
                                     frames=None):
    """Repair text noise only when state, ID, slots and actor identity already agree.

    A real stop, resume, direction, target, or object-position change is preserved
    even if it lasts one frame. States and actions are not voted on by neighbors.
    """
    from .mapper import load_label_instruction_map
    catalog, indexed = load_catalog(), _frames_by_index(frames)
    mapping = load_label_instruction_map(catalog=catalog)
    changed, k = 0, 1
    while k < len(rules) - 1:
        before = rules[k - 1]
        if (not before.get("instruction") or (zone and before.get("zone_id") != zone)
                or (ids and before.get("instruction_id") not in ids)):
            k += 1
            continue
        end = next((m for m in range(k, min(len(rules), k + max_frames + 1))
                    if rules[m].get("instruction") == before.get("instruction")), None)
        if end is None or end == k or any(not row.get("instruction") for row in rules[k:end]):
            k += 1
            continue
        span = rules[k - 1:end + 1]
        rendered = _same_verified_rendering(span, indexed, catalog, mapping)
        if rendered is None or rendered[0][0] != before["instruction"]:
            k += 1
            continue
        for row, (text, values) in zip(span[1:-1], rendered[1:-1]):
            _history(row, "repair_own_rendering", agreeing_indices=[span[0]["index"], span[-1]["index"]])
            row["instruction_smoothed_from"] = row["instruction"]
            row.update(instruction=text, instruction_variables=values, instruction_mapping_status="rendered")
            changed += 1
        k = end + 1
    return changed


def use_stopped_roundabout_sentences(frames, rules, catalog=None):
    """Roundabout moving sentences while Ego stands (>= 0.3 s) become their stopped variant."""
    catalog = load_catalog() if catalog is None else catalog
    track = _Track(frames)
    indexed = _frames_by_index(frames)
    standing = []
    for row in rules:
        n = track.position.get(row["index"])
        future = track.future(row["index"])
        speed = track.speed(n) if n is not None else None
        standing.append(row.get("zone_id") == "roundabout"
                        and _trusted_rendering(row, indexed.get(row["index"]))
                        and row.get("instruction") is not None
                        and row.get("instruction_id") in ROUNDABOUT_STOPPED
                        and speed is not None and speed < STANDING_SPEED_MPS
                        and future is not None and future[0] < STATIONARY_PROGRESS_M)
    changed, k = 0, 0
    while k < len(rules):
        if not standing[k]:
            k += 1
            continue
        end = k
        while (end + 1 < len(rules) and standing[end + 1]
               and rules[end + 1]["index"] == rules[end]["index"] + 1
               and track.connected(track.position.get(rules[end]["index"]),
                                   track.position.get(rules[end + 1]["index"]))):
            end += 1
        if end - k + 1 >= MIN_CONTRADICTION_FRAMES:
            for row in rules[k:end + 1]:
                target = ROUNDABOUT_STOPPED[row["instruction_id"]]
                text, values, missing = render_template(catalog["roundabout"][target],
                                                        row.get("instruction_variables") or {})
                if missing:
                    continue
                _history(row, "roundabout_stopped_variant", motion_horizon_s=HORIZON_S)
                row["instruction_moving_variant"] = row["instruction_id"]
                row.update(instruction_state=ROUNDABOUT_STOPPED_STATES[target], instruction_id=target,
                           instruction=text, instruction_variables=values,
                           instruction_label=catalog["roundabout"][target].get("label"),
                           instruction_mapping_status="rendered")
                changed += 1
        k = end + 1
    return changed


def _contradiction(text, ds, dd, v_now, v_future, *, obstacle_only=False):
    maneuver = any(word in text for word in MANEUVER_WORDS)
    side = MANEUVER_SIDE.search(text)
    if side and side.group(1) == "right" and dd >= SIDE_MOTION_M:
        return "direction"
    if side and side.group(1) == "left" and dd <= -SIDE_MOTION_M:
        return "direction"
    # Static obstacle instructions do not override longitudinal constraints.
    if obstacle_only:
        return None
    if maneuver and ds < STATIONARY_PROGRESS_M:
        return "stationary"
    keep = any(word in text for word in KEEP_WORDS)
    if keep and v_now is not None and v_now < STOPPED_SPEED_MPS and ds < STATIONARY_PROGRESS_M:
        return "keep_going_while_stopped"
    if (keep and v_now is not None and v_future is not None and v_now > KEEP_GOING_MIN_SPEED_MPS
            and v_future < KEEP_GOING_SPEED_RATIO * v_now):
        return "keep_going_while_stopping"
    return None


def drop_contradicted_instructions(frames, rules, *, objects=None):
    """Check supported current responses causally; keep other legacy checks.

    Current-policy markers are not certificates. Recompute their motion and
    current actor evidence before accepting the final rendered sentence.
    """
    track = _Track(frames)
    current_frames = _frames_by_index(frames)
    counts = Counter()
    found = {}
    for k, rule in enumerate(rules):
        text = rule.get("instruction")
        if not text:
            continue
        from .legacy_roundabout_guard import legacy_roundabout_action_reasons
        current_action_reasons = legacy_roundabout_action_reasons(rule)
        from .departure_plan import current_instruction_command_reasons
        current_action_reasons.extend(current_instruction_command_reasons(current_frames.get(rule["index"]), rule))
        if current_action_reasons:
            rule["action_valid"] = False
            rule.setdefault("unknown_reasons", []).extend(
                reason for reason in current_action_reasons if reason not in rule.get("unknown_reasons", []))
            _history(rule, "drop_current_action_evidence_conflict", reasons=current_action_reasons)
            rule["instruction_dropped"] = text
            rule["instruction"] = None
            rule["instruction_consistency"] = "dropped_current_action_evidence_conflict"
            counts["dropped_current_action_evidence_conflict"] += 1
            continue
        if rule.get("zone_id") == "highway":
            if "current_highway_maneuver_motion" in (rule.get("evidence") or {}):
                from .highway_maneuver import validate_current_highway_maneuver
                moving, movement_reasons = validate_current_highway_maneuver(frames, objects, rule)
                if not moving:
                    rule["action_valid"] = False
                    rule.setdefault("unknown_reasons", []).extend(
                        r for r in movement_reasons if r not in rule.get("unknown_reasons", []))
                    _history(rule, "drop_current_highway_maneuver_motion", reasons=movement_reasons)
                    rule["instruction_dropped"] = text
                    rule["instruction"] = None
                    rule["instruction_consistency"] = "dropped_current_highway_maneuver_motion"
                    continue
            # Every highway policy path shares the final current-target check.
            # Motion receipts do not certify position words or stale slots.
            from .highway_position import validate_current_highway_position
            position_valid, position_reasons = validate_current_highway_position(frames, objects, rule)
            if not position_valid:
                rule["action_valid"] = False
                rule["observation_valid"] = False
                rule.setdefault("unknown_reasons", []).extend(
                    r for r in position_reasons if r not in rule.get("unknown_reasons", []))
                _history(rule, "drop_current_highway_position_inconsistency", reasons=position_reasons)
                rule["instruction_dropped"] = text
                rule["instruction"] = None
                rule["instruction_consistency"] = "dropped_current_highway_position"
                continue
        evidence = rule.get("evidence") or {}
        static_plan = evidence.get("current_static_selected_plan") or {}
        highway_plan = evidence.get("current_highway_selected_plan") or {}
        static = evidence.get("current_static_maneuver_policy") or {}
        dynamic = evidence.get("current_dynamic_response") or {}
        dynamic_yield = evidence.get("current_dynamic_yield") or {}
        dynamic_approach = evidence.get("current_dynamic_approach") or {}
        dynamic_preentry = evidence.get("current_dynamic_preentry_yield") or {}
        dynamic_normal = evidence.get("current_dynamic_normal") or {}
        highway = evidence.get("current_highway_response") or {}
        roundabout_cruise = evidence.get("current_roundabout_cruise") or {}
        departure_plan = evidence.get("current_departure_plan") or {}
        validator = None
        if departure_plan.get("schema") == "current_departure_plan_v1":
            from .departure_plan import validate_current_departure_plan
            validator = validate_current_departure_plan
        elif static_plan.get("schema") == "current_static_selected_plan_v1":
            from .static_plan import validate_current_static_plan
            validator = validate_current_static_plan
        elif highway_plan.get("valid") is True:
            from .highway_plan import validate_current_highway_plan
            validator = validate_current_highway_plan
        elif static.get("decision") in ("fresh_current_outward_motion", "current_past_window_converging"):
            from .static_maneuver import validate_current_static_maneuver
            validator = validate_current_static_maneuver
        elif dynamic_normal.get("supervision_basis") == "recorded_current_normal_stop":
            from .dynamic_normal import validate_current_dynamic_normal
            validator = validate_current_dynamic_normal
        elif dynamic.get("supervision_basis") == "recorded_ego_motion":
            from .dynamic_resume import validate_current_dynamic_response
            validator = validate_current_dynamic_response
        elif dynamic_yield.get("supervision_basis") == "recorded_current_yield_episode":
            from .dynamic_yield import validate_current_dynamic_yield
            validator = validate_current_dynamic_yield
        elif dynamic_preentry.get("supervision_basis") == "recorded_current_preentry_yield":
            from .dynamic_preentry import validate_current_dynamic_preentry_yield
            validator = validate_current_dynamic_preentry_yield
        elif dynamic_approach.get("supervision_basis") == "recorded_current_approach":
            from .dynamic_approach import validate_current_dynamic_approach
            validator = validate_current_dynamic_approach
        elif highway.get("supervision_basis") == "recorded_current_motion":
            from .highway_current import validate_current_highway_response
            validator = validate_current_highway_response
        elif roundabout_cruise.get("supervision_basis") == "recorded_current_roundabout_motion":
            from .roundabout_current import validate_current_roundabout_cruise
            validator = validate_current_roundabout_cruise
        if validator is not None:
            valid, reasons = (validator(frames, objects, rule) if objects is not None
                              else (False, ["current_inventory_objects_missing"]))
            evidence["current_phase_consistency"] = {
                "recomputed_from_current_evidence": True,
                "recomputed_from_current_inventory_and_past_gt": validator.__name__ not in
                    ("validate_current_static_plan", "validate_current_highway_plan"),
                "recomputed_from_current_selected_plan": validator.__name__ in
                    ("validate_current_static_plan", "validate_current_highway_plan"),
                "recomputed_from_current_expert_commands": validator.__name__ == "validate_current_departure_plan",
                "uses_future_ego": False, "valid": valid, "reasons": reasons}
            if valid:
                rule["instruction_consistency"] = "consistent"
            else:
                _history(rule, "drop_current_phase_inconsistency", reasons=reasons)
                rule["instruction_dropped"] = text
                rule["instruction"] = None
                rule["instruction_consistency"] = "dropped_current_phase"
            continue
        n = track.position.get(rule["index"])
        if n is None:
            rule["instruction_consistency"] = "no_ego_track"
            continue
        j = track.at(rule["index"], HORIZON_S)
        if j is None or track.s[j] < track.s[n] - BACKWARD_JITTER_M:
            rule["instruction_consistency"] = "no_future"
            continue
        v_future = track.speed(j)
        kind = _contradiction(text, max(0.0, track.s[j] - track.s[n]), track.d[j] - track.d[n],
                              track.speed(n), v_future,
                              obstacle_only=rule.get("zone_id") == "static_obstacle")
        rule["instruction_consistency"] = "consistent"
        if kind:
            found[k] = kind
    # A contradiction shorter than 0.3 s sits on a threshold edge; dropping it
    # would only make the sentence flicker.
    keys = sorted(found)
    run = []
    for k in keys + [None]:
        if run and (k is None or k != run[-1] + 1 or found[k] != found[run[-1]]
                    or rules[k]["index"] != rules[run[-1]]["index"] + 1
                    or rules[k].get("zone_id") != rules[run[-1]].get("zone_id")
                    or not track.connected(track.position.get(rules[run[-1]]["index"]),
                                           track.position.get(rules[k]["index"]))):
            if len(run) >= MIN_CONTRADICTION_FRAMES:
                for m in run:
                    _history(rules[m], "drop_motion_contradiction", reason=found[m], horizon_s=HORIZON_S)
                    rules[m]["instruction_dropped"] = rules[m]["instruction"]
                    rules[m]["instruction"] = None
                    rules[m]["instruction_consistency"] = "dropped_" + found[m]
            else:
                for m in run:
                    rules[m]["instruction_consistency"] = "consistent_borderline"
            run = []
        if k is not None:
            run.append(k)
    # The other edge: a few kept frames inside one long contradiction (speed
    # crossing a threshold while creeping at a red light) would flicker the
    # sentence back for under 0.5 s; they share the episode's verdict.
    dropped = [k for k, rule in enumerate(rules) if rule.get("instruction_dropped") is not None
               and str(rule.get("instruction_consistency", "")).startswith("dropped_")]
    for a, b in zip(dropped, dropped[1:]):
        island = rules[a + 1:b]
        if (0 < len(island) < MIN_KEPT_ISLAND_FRAMES
                and rules[b]["index"] - rules[a]["index"] == b - a
                and track.connected(track.position.get(rules[a]["index"]),
                                    track.position.get(rules[b]["index"]))
                and all(row.get("zone_id") == rules[a].get("zone_id") for row in island + [rules[b]])
                and all(not (row.get("evidence") or {}).get("current_phase_consistency")
                        for row in island)
                and rules[a]["instruction_dropped"] == rules[b]["instruction_dropped"]
                and all(row.get("instruction") == rules[a]["instruction_dropped"] for row in island)):
            for row in island:
                _history(row, "drop_short_motion_contradiction_island",
                         between_indices=[rules[a]["index"], rules[b]["index"]])
                row["instruction_dropped"] = row["instruction"]
                row["instruction"] = None
                row["instruction_consistency"] = rules[a]["instruction_consistency"]
    for rule in rules:
        if rule.get("instruction_consistency"):
            counts[rule["instruction_consistency"]] += 1
    return dict(sorted(counts.items()))
