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
    return len(frame_times) - 1 if frame_times else None


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
