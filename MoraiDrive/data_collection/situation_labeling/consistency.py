"""Align situation instructions with the ego's own recorded motion.

Training-label hygiene for Stage 2, which learns the future trajectory: an
instruction that disagrees with that trajectory teaches the model to ignore
instructions. These passes use the recorded future, so they only shape
training labels and are not inference-time rules.

Before sentences are attached (states only):
  align_highway_lane_changes   lane change spans and sides from the ego's lateral motion
  align_static_avoidance       avoid starts when the swing starts; return once Ego moves back
  hold_dynamic_wait_until_moving  after yielding, "remain stopped" until Ego actually moves
After sentences are attached:
  fill_short_instruction_gaps  bridge 1-3 frame rule gaps between identical sentences
  smooth_short_instruction_islands  a 1-2 frame different sentence between identical ones takes theirs
  use_stopped_roundabout_sentences  roundabout yield/follow/slow-down while standing -> "remain stopped"
  drop_contradicted_instructions  null a sentence the next 2.5 s contradicts (>= 0.3 s runs)
"""

from bisect import bisect_left
from collections import Counter, defaultdict
import re

from .renderer import load_catalog, render_template

HORIZON_S = 2.5
HORIZON_TOLERANCE_S = 0.3
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
MANEUVER_WORDS = ("avoid", "between", "Return to", "Resume", "Move into")
KEEP_WORDS = ("Continue driving", "Maintain the current")
YIELD_STATES = ("yield_start", "yielding", "waiting")
# The side a maneuver goes to, not the obstacle's position ("an obstacle to
# the left of the ego vehicle. ... maneuver to the right").
MANEUVER_SIDE = re.compile(r"(?:maneuver to the|into the) (left|right)")


class _Track:
    """Ego route distance s and lateral offset d (+ left) by source time."""

    def __init__(self, frames):
        rows = sorted((f["lidar_source_ns"] / 1e9, f["ego_path_s_m"], f["ego_path_d_m"], f["index"])
                      for f in frames if f.get("ego_path_s_m") is not None
                      and f.get("ego_path_d_m") is not None)
        self.t = [row[0] for row in rows]
        self.s = [row[1] for row in rows]
        self.d = [row[2] for row in rows]
        self.position = {row[3]: n for n, row in enumerate(rows)}

    def at(self, index, offset_s):
        """Position n of the frame offset_s after frame `index`, or None."""
        n = self.position.get(index)
        if n is None:
            return None
        j = bisect_left(self.t, self.t[n] + offset_s)
        if j >= len(self.t) or abs(self.t[j] - self.t[n] - offset_s) > HORIZON_TOLERANCE_S:
            return None
        return j

    def speed(self, n):
        p = bisect_left(self.t, self.t[n] - SPEED_BACK_S)
        return (self.s[n] - self.s[p]) / (self.t[n] - self.t[p]) if self.t[n] - self.t[p] > 0.3 else None

    def future(self, index):
        """(route progress, lateral change) over the horizon, or None."""
        n, j = self.position.get(index), self.at(index, HORIZON_S)
        if n is None or j is None or self.s[j] < self.s[n] - BACKWARD_JITTER_M:
            return None
        return max(0.0, self.s[j] - self.s[n]), self.d[j] - self.d[n]


def _spans(rules, keep):
    span = []
    for row in rules:
        if keep(row):
            if span and row["index"] != span[-1]["index"] + 1:
                yield span
                span = []
            span.append(row)
        elif span:
            yield span
            span = []
    if span:
        yield span


def align_static_avoidance(frames, rules):
    """Avoid sentences follow the swing: maintain -> avoid -> continue -> return."""
    track, counts = _Track(frames), Counter()
    avoid_span = lambda row: (row.get("zone_id") == "static_obstacle"
                              and row.get("instruction_state") in ("avoid", "avoid_continue")
                              and (row.get("variables") or {}).get("avoid_direction") in ("left", "right"))
    for span in _spans(rules, avoid_span):
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
    q = bisect_left(track.t, track.t[j] - SPEED_BACK_S)
    v_future = (track.s[j] - track.s[q]) / max(track.t[j] - track.t[q], 1e-3)
    return v_future < KEEP_GOING_SPEED_RATIO * speed


def _lateral_bursts(track, rows):
    """(start, end, sign) runs of rows whose lateral speed exceeds 0.25 m/s one way."""
    signs = []
    for row in rows:
        n = track.position.get(row["index"])
        sign = 0
        if n is not None:
            a = max(0, n - LATERAL_SPEED_HALF_WINDOW)
            b = min(len(track.t) - 1, n + LATERAL_SPEED_HALF_WINDOW)
            if track.t[b] - track.t[a] > 0.3:
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
    rows = [row for row in rules if row.get("zone_id") == "highway"]
    ahead = defaultdict(list)
    for o in objects:
        if (o.get("kind") == "npc" and _finite(o.get("distance_to_ego_m"))
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
            near = ahead.get(row["index"])
            if near:
                actor = min(near, key=lambda o: o["distance_to_ego_m"])
                variables.update(target_actor_id=actor.get("actor_id"), target_kind="vehicle",
                                 target_position="front", target_position_text="front",
                                 target_distance_m=actor["distance_to_ego_m"])
                row["label_candidate"], row["instruction_state"] = "KEEP_LANE", "approach_keep"
            else:
                row["label_candidate"], row["instruction_state"] = "KEEP_LANE", "normal"
        variables["lane_change_direction"] = None
        row.setdefault("evidence", {})["rule_state_before_motion_alignment"] = before
        counts["stale_change_to_" + str(row.get("instruction_state"))] += 1
    return dict(sorted(counts.items()))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and abs(value) != float("inf")


def hold_dynamic_wait_until_moving(frames, rules):
    """Yield from the braking onset; after yielding remain stopped until Ego moves, then resume."""
    track, counts = _Track(frames), Counter()
    # 1. An approach span that ends in a yield: the yield began when Ego
    #    started braking, before the conflict test fired. Converted once per
    #    span from the first braking frame, so the state cannot alternate.
    by_index = {row["index"]: k for k, row in enumerate(rules)}
    approach = lambda row: (row.get("zone_id") == "dynamic_obstacle"
                            and row.get("instruction_state") == "approach")
    for span in _spans(rules, approach):
        after = by_index.get(span[-1]["index"] + 1)
        if after is None or rules[after].get("instruction_state") not in YIELD_STATES:
            continue
        start = next((k for k, row in enumerate(span) if _braking(track, row["index"])), None)
        for row in span[start:] if start is not None else ():
            row.setdefault("evidence", {})["rule_state_before_motion_alignment"] = "approach"
            row["instruction_state"] = "yielding"
            counts["to_yielding"] += 1
    # 2. After a yield: remain stopped until Ego moves, resume while slow, then normal.
    after_yield = False
    for row in rules:
        state = row.get("instruction_state")
        if row.get("zone_id") != "dynamic_obstacle":
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


def fill_short_instruction_gaps(rules):
    """Bridge a <= 3 frame gap of undetermined rule state between identical sentences."""
    filled, previous = 0, None
    for k, row in enumerate(rules):
        if row.get("instruction"):
            previous = k
            continue
        if previous is None or row.get("status") == "invalid_frame":
            continue
        nxt = next((m for m in range(k, min(len(rules), k + MAX_FILLED_GAP_FRAMES + 1))
                    if rules[m].get("instruction")), None)
        before = rules[previous]
        if (nxt is None or nxt - previous - 1 > MAX_FILLED_GAP_FRAMES
                or rules[nxt]["index"] - before["index"] != nxt - previous
                or rules[nxt].get("instruction") != before.get("instruction")
                or rules[nxt].get("zone_id") != before.get("zone_id")):
            continue
        for gap in rules[previous + 1:nxt]:
            if gap.get("instruction") or gap.get("status") == "invalid_frame":
                break
            for key in ("instruction_id", "instruction", "instruction_variables", "instruction_source"):
                gap[key] = before.get(key)
            gap["instruction_gap_filled_from"] = before["index"]
            filled += 1
    return filled


def smooth_short_instruction_islands(rules, max_frames=MAX_ISLAND_FRAMES, zone=None, ids=None):
    """A short (default 1-2 frame) different sentence between two identical ones takes theirs.

    With zone/ids the pass is limited to those sentences on both sides and inside.
    """
    keys = ("instruction_id", "instruction", "instruction_variables", "instruction_source")
    changed = 0
    k = 1
    while k < len(rules) - 1:
        before = rules[k - 1]
        if (not before.get("instruction") or (zone and before.get("zone_id") != zone)
                or (ids and before.get("instruction_id") not in ids)):
            k += 1
            continue
        end = next((m for m in range(k, min(len(rules), k + max_frames + 1))
                    if rules[m].get("instruction") == before.get("instruction")), None)
        if (end is None or end == k
                or rules[end]["index"] - before["index"] != end - (k - 1)
                or any(row.get("zone_id") != before.get("zone_id") or row.get("status") == "invalid_frame"
                       or (ids and row.get("instruction") and row.get("instruction_id") not in ids)
                       for row in rules[k:end + 1])):
            k += 1
            continue
        for row in rules[k:end]:
            row["instruction_smoothed_from"] = row.get("instruction")
            for key in keys:
                row[key] = before.get(key)
            changed += 1
        k = end + 1
    return changed


def use_stopped_roundabout_sentences(frames, rules, catalog=None):
    """Roundabout moving sentences while Ego stands (>= 0.3 s) become their stopped variant."""
    catalog = load_catalog() if catalog is None else catalog
    track = _Track(frames)
    standing = []
    for row in rules:
        n = track.position.get(row["index"])
        future = track.future(row["index"])
        speed = track.speed(n) if n is not None else None
        standing.append(row.get("zone_id") == "roundabout"
                        and row.get("instruction_id") in ROUNDABOUT_STOPPED
                        and speed is not None and speed < STANDING_SPEED_MPS
                        and future is not None and future[0] < STATIONARY_PROGRESS_M)
    changed, k = 0, 0
    while k < len(rules):
        if not standing[k]:
            k += 1
            continue
        end = k
        while end + 1 < len(rules) and standing[end + 1] and rules[end + 1]["index"] == rules[end]["index"] + 1:
            end += 1
        if end - k + 1 >= MIN_CONTRADICTION_FRAMES:
            for row in rules[k:end + 1]:
                target = ROUNDABOUT_STOPPED[row["instruction_id"]]
                text, values, missing = render_template(catalog["roundabout"][target],
                                                        row.get("instruction_variables") or {})
                if missing:
                    continue
                row["instruction_moving_variant"] = row["instruction_id"]
                row.update(instruction_id=target, instruction=text, instruction_variables=values)
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
    if (keep and v_now is not None and v_now > KEEP_GOING_MIN_SPEED_MPS
            and v_future < KEEP_GOING_SPEED_RATIO * v_now):
        return "keep_going_while_stopping"
    return None


def drop_contradicted_instructions(frames, rules):
    """Null a sentence the next 2.5 s contradicts for >= 3 consecutive frames."""
    track = _Track(frames)
    counts = Counter()
    found = {}
    for k, rule in enumerate(rules):
        text = rule.get("instruction")
        if not text:
            continue
        n = track.position.get(rule["index"])
        if n is None:
            rule["instruction_consistency"] = "no_ego_track"
            continue
        j = track.at(rule["index"], HORIZON_S)
        if j is None or track.s[j] < track.s[n] - BACKWARD_JITTER_M:
            rule["instruction_consistency"] = "no_future"
            continue
        q = bisect_left(track.t, track.t[j] - SPEED_BACK_S)
        v_future = (track.s[j] - track.s[q]) / max(track.t[j] - track.t[q], 1e-3)
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
        if run and (k is None or k != run[-1] + 1 or found[k] != found[run[-1]]):
            if len(run) >= MIN_CONTRADICTION_FRAMES:
                for m in run:
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
                and rules[a]["instruction_dropped"] == rules[b]["instruction_dropped"]
                and all(row.get("instruction") == rules[a]["instruction_dropped"] for row in island)):
            for row in island:
                row["instruction_dropped"] = row["instruction"]
                row["instruction"] = None
                row["instruction_consistency"] = rules[a]["instruction_consistency"]
    for rule in rules:
        if rule.get("instruction_consistency"):
            counts[rule["instruction_consistency"]] += 1
    return dict(sorted(counts.items()))
