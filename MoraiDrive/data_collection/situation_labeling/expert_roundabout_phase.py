"""Recover Expert's recorded roundabout_entry phase for already-collected
episodes, as frame-indexed observations for expert_instructions.py.

`_update_log()` (expert/.../planning_process.py) prints a one-line
"[ROUNDABOUT] <phase> target=... conflict_id=... ttc=... front_id=... gap=...
hold_gap=... entering=... merged=... ceiling=... adjusted=..." transition
exactly once per change -- the same mechanism and the same lack of a
timestamp as static's "[STATIC]" line. Sim-time is recovered the same way
static's phase is (expert_static_phase.py): interpolating between the sparse
periodic "[INFO] [epoch, sim_time]: [ExpertInput] ..." anchor lines using the
count of periodic "[CTRL]" debug dumps between anchors (isochronous ticks) as
a proportional clock between anchors.

Episodes collected after Expert started emitting a per-tick "[EXPERT_STATE]"
JSON line (see planning_process.py's _update_log) carry a true ROS header
seq and source timestamp and should use expert_instructions.py's exact join
instead -- build_roundabout_observations here is an approximate, time-
interpolated stand-in for episodes collected before that change, meant to be
passed as apply_expert_instructions(..., index_observations=...).
"""
import re
import hashlib
from pathlib import Path

ANCHOR_RE = re.compile(r"^\[INFO\] \[(\d+\.\d+),\s*([\d.]+)\]")
TRANSITION_RE = re.compile(
    r"^\[ROUNDABOUT\] (\w+) target=([-+\d.]+)km/h conflict_id=(\S+) "
    r"ttc=\S+s front_id=(\S+) gap=\S+ hold_gap=[-+\d.]+m "
    r"entering=(True|False) merged=(True|False)")
_BOOL = {"True": True, "False": False}


def _parse_id(token):
    return None if token == "None" else token


def _parse_log(log_path):
    anchors = []  # (line_no, sim_time)
    ctrl_lines = []  # line_no
    transitions = []  # (line_no, raw_dict)
    source_bytes = Path(log_path).read_bytes()
    text = source_bytes.decode("utf-8", errors="replace")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
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
            phase, target_kmh, conflict_id, front_id, entering, merged = m.groups()
            transitions.append((line_no, {
                "mode": "roundabout_entry",
                "roundabout_phase": phase,
                "roundabout_entering": _BOOL[entering],
                "roundabout_committed": _BOOL[merged],
                "roundabout_front_id": _parse_id(front_id),
                "roundabout_conflict_id": _parse_id(conflict_id),
                "planner_target_speed_mps": float(target_kmh) / 3.6,
                "source": "existing_roundabout_text_log",
                "transition_source": {"path": str(log_path), "sha256": source_sha256,
                                      "line_no": line_no},
            }))
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


def build_roundabout_observations(frames, log_path):
    """Return {frame_index: raw_dict} via forward-fill of Expert's recorded
    roundabout_entry phase transitions, for roundabout-zone frames only.
    `frames` must be sorted ascending by index with contiguous positions
    (frames[i]["index"] increases with i); each frame needs a
    "lidar_source_ns" field. Empty when the log has no ROUNDABOUT
    transitions. Pass the result as
    apply_expert_instructions(..., index_observations=...).
    """
    anchors, ctrl_lines, transitions = _parse_log(log_path)
    if not transitions:
        return {}
    frame_times = [frame["lidar_source_ns"] / 1e9 for frame in frames]
    events = []
    for line_no, raw in transitions:
        sim_time = _interpolate_time(line_no, anchors, ctrl_lines)
        if sim_time is None:
            continue
        position = _frame_index_for_time(frame_times, sim_time)
        if position is not None:
            before = max((a for a in anchors if a[0] <= line_no), default=None)
            after = min((a for a in anchors if a[0] >= line_no), default=None)
            events.append((position, raw, sim_time,
                           before[1] if before else None,
                           after[1] if after else None))
    events.sort(key=lambda e: e[0])
    observations = {}
    current = None
    event_iter = iter(events)
    next_event = next(event_iter, None)
    for position, frame in enumerate(frames):
        while next_event is not None and next_event[0] == position:
            current = next_event
            next_event = next(event_iter, None)
        if current is not None and frame.get("zone_id") == "roundabout":
            # A transition log is deduplicated on phase/actor, NOT target speed.
            # Retain the historical numerical target for audit; never imply
            # that forward filling the phase freshly observes its target.
            raw = dict(current[1])
            next_lower = next_event[3] if next_event is not None else None
            t = frame["lidar_source_ns"] / 1e9
            raw["legacy_transition_observation"] = {
                "schema": "roundabout_transition_observation_v1",
                "transition_source": raw["transition_source"],
                "transition_estimated_sim_time_s": current[2],
                "transition_anchor_lower_s": current[3],
                "transition_anchor_upper_s": current[4],
                "next_transition_anchor_lower_s": next_lower,
                "frame_index": frame["index"],
                "lidar_source_ns": frame["lidar_source_ns"],
                "transition_estimated_age_seconds": t - current[2],
                "phase_within_anchor_bounds": (
                    current[4] is not None and t >= current[4]
                    and (next_lower is None or t < next_lower)),
                "numeric_target_current_verified": False,
                "numeric_target_reason": "phase_log_dedup_does_not_track_target_changes",
                "uses_future_ego": False,
                "physical_simultaneity_verified": False,
            }
            observations[frame["index"]] = raw
    return observations
