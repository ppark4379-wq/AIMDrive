"""Forward-fill Expert's recorded static-obstacle phase onto feature frames.

`_update_log()` (expert/.../planning_process.py) prints a one-line
"[STATIC] <reason> offset=... target=..." transition exactly once per change,
every control tick, so no transition is missed. That line carries no
timestamp of its own; sim-time is interpolated from the sparse periodic
"[INFO] [epoch, sim_time]: [ExpertInput] ..." anchor lines, using the count
of periodic "[CTRL]" debug dumps between anchors (isochronous ticks) as a
proportional clock between anchors.
"""
import re
import ast
import bisect
from collections import Counter
import hashlib
from pathlib import Path

ANCHOR_RE = re.compile(r"^\[INFO\] \[(\d+\.\d+),\s*([\d.]+)\]")
TRANSITION_RE = re.compile(r"^\[STATIC\] (static_recorded_\w+) offset=([^ ]+) target=([^ ]+)")

# Explicit Expert raw-state -> instruction label translation. Kept separate
# from rules.py so a renamed or merged raw Expert state only needs an edit
# here, not a change to rule logic that branches on the translated label.
RAW_STATE_TO_LABEL = {
    "static_recorded_approach": "approach",
    "static_recorded_follow": "follow",
    "static_recorded_rejoined": "rejoined",
}


def _parse_log(log_path):
    anchors = []  # (line_no, sim_time)
    ctrl_lines = []  # line_no
    transitions = []  # (line_no, reason)
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), 1):
        m = ANCHOR_RE.match(line)
        if m:
            anchors.append((line_no, float(m.group(2))))
            continue
        if line.startswith("[CTRL]"):
            ctrl_lines.append(line_no)
            continue
        m = TRANSITION_RE.match(line)
        if m:
            transitions.append((line_no, m.group(1)))
    return anchors, ctrl_lines, transitions


def _interpolate_time(line_no, anchors, ctrl_lines):
    """Sim time at `line_no`, proportional to CTRL-dump count between the
    bracketing anchors (isochronous ticks -> proportional-in-time clock)."""
    before = max((a for a in anchors if a[0] <= line_no), default=None)
    after = min((a for a in anchors if a[0] >= line_no), default=None)
    if before is None or after is None or before == after:
        nearest = before or after
        return nearest[1] if nearest else None
    b_line, b_time = before
    a_line, a_time = after
    ctrl_total = sum(1 for c in ctrl_lines if b_line < c <= a_line)
    if ctrl_total == 0:
        return (b_time + a_time) / 2
    ctrl_before = sum(1 for c in ctrl_lines if b_line < c <= line_no)
    return b_time + (a_time - b_time) * (ctrl_before / ctrl_total)


def _frame_index_for_time(frame_times, sim_time):
    for index, t in enumerate(frame_times):
        if t >= sim_time:
            return index
    # A transition after capture end has no receiving frame. Clamping it to
    # the last frame would insert a future planner state into past GT.
    return None


def attach_static_phase(frames, log_path):
    """Set frame["expert_static_phase"] (the raw Expert string, for display)
    and frame["expert_label"] (translated via RAW_STATE_TO_LABEL, for rule
    logic) via forward-fill of Expert's static_recorded_{approach,follow,
    rejoined} transitions. `frames` must be sorted ascending by index with
    contiguous positions (frames[i]["index"] increases with i); each frame
    needs a "lidar_source_ns" field. A no-op (leaves frames untouched) when
    the log has no STATIC transitions.
    """
    anchors, ctrl_lines, transitions = _parse_log(log_path)
    if not transitions:
        return frames
    frame_times = [frame["lidar_source_ns"] / 1e9 for frame in frames]
    events = []
    for line_no, reason in transitions:
        sim_time = _interpolate_time(line_no, anchors, ctrl_lines)
        if sim_time is None:
            continue
        position = _frame_index_for_time(frame_times, sim_time)
        if position is not None:
            events.append((position, reason))
    events.sort(key=lambda e: e[0])
    current = None
    event_iter = iter(events)
    next_event = next(event_iter, None)
    for position, frame in enumerate(frames):
        while next_event is not None and next_event[0] == position:
            current = next_event[1]
            next_event = next(event_iter, None)
        frame["expert_static_phase"] = current
        frame["expert_label"] = RAW_STATE_TO_LABEL.get(current)
    return frames


def attach_static_selected_plan(frames, log_path):
    """Bind a validated selected path only across an unchanged log interval.

    Legacy STATIC dictionary dumps have no message timestamp. Sparse sim-time
    anchors therefore establish only an interval: every recorded plan in that
    interval must agree. An interpolated phase alone is not current path proof.
    This never infers a path from future Ego movement or changes admission masks.
    """
    path = Path(log_path)
    data = path.read_bytes()
    anchors, plans = [], []
    invalid_dictionary_lines = 0
    for line_number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        match = ANCHOR_RE.match(line)
        if match:
            anchors.append((line_number, float(match.group(2))))
        if not line.startswith("[STATIC] {"):
            continue
        try:
            entry = ast.literal_eval(re.sub(r"\b(?:inf|nan)\b", "None", line[len("[STATIC] "):]))
            if not isinstance(entry, dict):
                raise ValueError("STATIC dictionary expected")
            names = entry.get("selected_routes")
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                names = []
            signature = (entry.get("phase"), entry.get("path_validated"), tuple(names))
            plans.append((line_number, signature, entry))
        except (ValueError, SyntaxError):
            # A malformed dump is a plan discontinuity, not an absent event.
            plans.append((line_number, ("unparsed", False, ()), {}))
            invalid_dictionary_lines += 1
    counts = Counter()
    counts["invalid_dictionary_lines"] = invalid_dictionary_lines
    if (len(anchors) < 2 or any(b[1] <= a[1] for a, b in zip(anchors, anchors[1:]))):
        return {"bound_frames": 0, "reason": "missing_or_nonmonotonic_sim_time_anchors"}
    times = [anchor[1] for anchor in anchors]
    source_sha = hashlib.sha256(data).hexdigest()
    plan_lines = [item[0] for item in plans]
    for frame in frames:
        frame.pop("expert_static_selected_plan", None)
        if frame.get("zone_id") != "static_obstacle":
            continue
        stamp = frame.get("lidar_source_ns")
        if not isinstance(stamp, int):
            counts["missing_source_stamp"] += 1
            continue
        sim_time = stamp / 1e9
        after = bisect.bisect_left(times, sim_time)
        before = after - 1
        if after < len(times) and times[after] == sim_time:
            # An anchor is not itself a STATIC observation. Use the interval
            # ending at it; never silently use a future plan after this time.
            before = after - 1
        if before < 0 or after >= len(anchors):
            counts["unbracketed_frame"] += 1
            continue
        lo, hi = anchors[before], anchors[after]
        if not 0 < hi[1] - lo[1] <= 6.0:
            counts["anchor_interval_too_wide"] += 1
            continue
        first_plan = bisect.bisect_left(plan_lines, lo[0])
        interval = plans[first_plan:bisect.bisect_right(plan_lines, hi[0])]
        signatures = {item[1] for item in interval}
        if len(signatures) != 1:
            counts["selected_plan_changes_or_missing"] += 1
            continue
        phase, validated, names = next(iter(signatures))
        directions = {re.match(r"^(left|right)_", name).group(1)
                      for name in names if re.match(r"^(left|right)_", name)}
        if (phase != "follow" or validated is not True or not names
                or len(directions) != 1 or any(not re.match(r"^(left|right)_", name) for name in names)):
            counts["not_validated_single_direction_follow_plan"] += 1
            continue
        # A uniform interval alone can still start with a *new* plan after
        # this frame. Its incoming captured plan must agree too; otherwise
        # the unstamped transition cannot be placed on either side of it.
        incoming = plans[first_plan - 1] if first_plan else None
        if incoming is None or incoming[1] != (phase, validated, names):
            counts["incoming_selected_plan_unverified_or_changed"] += 1
            continue
        frame["expert_static_selected_plan"] = {
            "schema": "static_selected_path_interval_v1", "phase": phase,
            "path_validated": validated, "selected_routes": list(names),
            "direction": next(iter(directions)), "anchor_interval_all_plans_agree": True,
            "incoming_plan_agrees": True, "incoming_plan_log_line": incoming[0],
            "source_path": str(path), "source_sha256": source_sha,
            "anchor_before": {"line": lo[0], "sim_seconds": lo[1]},
            "anchor_after": {"line": hi[0], "sim_seconds": hi[1]},
            "plan_log_lines": [item[0] for item in interval],
            "speed_source_counts": dict(Counter(str(item[2].get("speed_source")) for item in interval)),
            "signal_zero_limit_all": all(
                item[2].get("speed_source") == "signal"
                and isinstance(item[2].get("speed_limits"), dict)
                and (item[2].get("speed_limits") or {}).get("signal") == 0
                for item in interval),
            "timing_basis": "unchanged recorded plan inside bracketing sim-time anchors",
            "physical_sensor_synchrony_certified": False}
        counts["bound_frames"] += 1
    return dict(sorted(counts.items()))
