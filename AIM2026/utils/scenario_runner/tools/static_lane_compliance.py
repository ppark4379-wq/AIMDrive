#!/usr/bin/env python3
"""Check a driven Ego trace against the painted lane markings of the static segment.

Completion without collision does not show that the Expert stayed on the road
paint rules.  This module sweeps the Ego body along the recorded trace and
rejects contact with any no-cross marking, and bounds how far the body centre
left the official route.

No-cross markings are every solid MGeo boundary except stop lines, plus the
intersection guide lines that continue the yellow centre line.
"""

# Resolve this relocated package before importing its shared dependencies.
import sys as _runner_sys
from pathlib import Path as _RunnerPath
for _runner_ancestor in _RunnerPath(__file__).resolve().parents:
    if (_runner_ancestor / "project_paths.py").is_file():
        _runner_sys.path.insert(0, str(_runner_ancestor.parent))
        break
from scenario_runner.project_paths import AIM_ROOT, morai_path, activate_project_paths
activate_project_paths()


import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = AIM_ROOT
MAP_DIR = morai_path("scenario_runner") / "data" / "R_KR_PR_K-city_2025"
DEFAULT_MARKINGS = morai_path("scenario_runner") / "config" / "static_segment_lane_markings.json"
DEFAULT_ROUTE = MAP_DIR / "2026_molit_comp_global_path.txt"
SCHEMA = "morai_static_lane_compliance_v1"
SEGMENT_INDEX = (483, 753)
# The trace must span the intersection that holds the obstacle zone.
REQUIRED_ROUTE_INDEX = (640, 715)
STOP_LINE = 530
CENTER_LINE = 501
GUIDE_LINE = 525

# MORAI gRPC Ego box (2023_Hyundai_Ioniq5): 4.635 m long, 2.161 m wide with
# mirrors, centre 1.53 m ahead of and 0.04 m right of the reported position.
# Paint contact is decided by the 1.890 m body (IONIQ 5 width without mirrors),
# which contains the tyres and is symmetric about the vehicle centre line
# through the reported position; the 0.04 m box offset comes from the mirrors.
EGO_BODY = {"length_m": 4.635, "width_m": 1.890,
            "center_forward_m": 1.53, "center_left_m": 0.0}
DEFAULT_POLICY = {
    "min_marking_clearance_m": 0.0,
    "max_right_deviation_m": 4.0,
    "max_sample_gap_m": 3.0,
    "densify_step_m": 0.2,
}
# Upper bound of the body-centre right deviation for each class.
AVOIDANCE_CLASSES = (("small", 2.0), ("medium", 4.0), ("large", float("inf")))
# Trajectory shape: no swing below STRAIGHT_MAX_M; a swing whose body centre
# first reaches SWING_ONSET_M right of the route at or after LATE_ONSET_INDEX
# is late.  Recorded right_02/right_03 swings start at index 647-654; a
# right_01 -> right_03 connection behind a right-side obstacle started at 669.
STRAIGHT_MAX_M = 0.6
SWING_ONSET_M = 0.5
LATE_ONSET_INDEX = 660
TRAJECTORY_CLASSES = ("straight", "small", "medium", "late", "large")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def avoidance_class(max_right_m):
    for name, upper in AVOIDANCE_CLASSES:
        if max_right_m < upper:
            return name
    return AVOIDANCE_CLASSES[-1][0]


def trajectory_class(max_right_m, onset_index):
    if max_right_m < STRAIGHT_MAX_M:
        return "straight"
    magnitude = avoidance_class(max_right_m)
    if magnitude != "large" and onset_index is not None and onset_index >= LATE_ONSET_INDEX:
        return "late"
    return magnitude


def _solid(line):
    return any(str(shape).startswith("solid") for shape in line["lane_shape"])


def no_cross_segments(document):
    """Return (N, 4) segment endpoints, half paint widths and line labels."""
    lines = document["lines"]
    centre_ends = [np.array(point) for line in lines
                   if CENTER_LINE in line["lane_type"]
                   for point in (line["points_xy"][0], line["points_xy"][-1])]

    def continues_centre(line):
        ends = (np.array(line["points_xy"][0]), np.array(line["points_xy"][-1]))
        return all(any(np.hypot(*(end - other)) <= 0.5 for other in centre_ends)
                   for end in ends)

    rows, half, labels = [], [], []
    for line in lines:
        if STOP_LINE in line["lane_type"]:
            continue
        if _solid(line):
            kind = "solid"
        elif GUIDE_LINE in line["lane_type"] and continues_centre(line):
            kind = "centre_guide"
        else:
            continue
        points = line["points_xy"]
        for a, b in zip(points[:-1], points[1:]):
            rows.append([a[0], a[1], b[0], b[1]])
            half.append(0.5 * float(line["lane_width_m"]))
            labels.append("{}:{}:{}".format(line["idx"], line["lane_type"][0], kind))
    return (np.asarray(rows, dtype=float).reshape(-1, 4),
            np.asarray(half, dtype=float), labels)


def load_markings(path=DEFAULT_MARKINGS):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "morai_static_segment_lane_markings_v1":
        raise ValueError("Unsupported lane marking file: {}".format(path))
    return document


def load_route(path=DEFAULT_ROUTE, segment=SEGMENT_INDEX):
    return np.loadtxt(path)[segment[0]:segment[1] + 1, :2], int(segment[0])


def read_trace_csv(path):
    """Rows of (x, y, yaw_rad) from a Scenario Runner --ego-trace file."""
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            values = (float(row["x"]), float(row["y"]), math.radians(float(row["yaw_deg"])))
            if all(math.isfinite(value) for value in values):
                rows.append(values)
    return rows


def read_active_signals(path):
    """Rows of (x, y, yaw_rad) from a Sensor Record quality/active_signals.json."""
    return ego_trace_from_active_signals(json.loads(Path(path).read_text(encoding="utf-8")))


def ego_trace_from_active_signals(rows):
    """One (x, y, yaw_rad) per Ego source timestamp, in source-time order."""
    unique = {}
    for row in rows:
        if row.get("role") != "ego" or row.get("source_timestamp_ns") is None:
            continue
        signal = row.get("signal") or {}
        position = signal.get("position") or []
        if len(position) >= 2 and signal.get("yaw") is not None:
            unique[int(row["source_timestamp_ns"])] = (
                float(position[0]), float(position[1]), float(signal["yaw"]))
    return [unique[key] for key in sorted(unique)]


def densify(trace, step_m):
    trace = np.asarray(trace, dtype=float)
    if len(trace) < 2:
        return trace, 0.0
    trace[:, 2] = np.unwrap(trace[:, 2])
    gaps = np.hypot(np.diff(trace[:, 0]), np.diff(trace[:, 1]))
    rows = [trace[:1]]
    for index, gap in enumerate(gaps):
        count = max(1, int(math.ceil(gap / step_m)))
        t = np.linspace(0.0, 1.0, count + 1)[1:, None]
        rows.append(trace[index] + t * (trace[index + 1] - trace[index]))
    return np.vstack(rows), float(gaps.max())


def body_centres(trace, body=EGO_BODY):
    x, y, yaw = trace[:, 0], trace[:, 1], trace[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    forward, left = body["center_forward_m"], body["center_left_m"]
    return np.column_stack((x + forward * c - left * s, y + forward * s + left * c))


def _point_box_distance(px, py, hl, hw):
    return np.hypot(np.maximum(np.abs(px) - hl, 0.0), np.maximum(np.abs(py) - hw, 0.0))


def _point_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length2 = np.maximum(dx * dx + dy * dy, 1e-12)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
    return np.hypot(ax + t * dx - px, ay + t * dy - py)


def _segment_hits_box(ax, ay, bx, by, hl, hw):
    """Liang-Barsky clip of segments against the centred box."""
    dx, dy = bx - ax, by - ay
    low = np.zeros_like(ax)
    high = np.ones_like(ax)
    hit = np.ones_like(ax, dtype=bool)
    for p, q in ((-dx, ax + hl), (dx, hl - ax), (-dy, ay + hw), (dy, hw - ay)):
        parallel = np.abs(p) < 1e-12
        hit &= ~(parallel & (q < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(parallel, 0.0, q / np.where(parallel, 1.0, p))
        low = np.where(~parallel & (p < 0), np.maximum(low, r), low)
        high = np.where(~parallel & (p > 0), np.minimum(high, r), high)
    return hit & (low <= high)


def box_segment_clearance(centre, yaw, segments, body=EGO_BODY):
    """Distance from one body box to every segment (0 when they overlap)."""
    hl, hw = 0.5 * body["length_m"], 0.5 * body["width_m"]
    c, s = math.cos(yaw), math.sin(yaw)
    local = segments - np.array([centre[0], centre[1], centre[0], centre[1]])
    ax = local[:, 0] * c + local[:, 1] * s
    ay = -local[:, 0] * s + local[:, 1] * c
    bx = local[:, 2] * c + local[:, 3] * s
    by = -local[:, 2] * s + local[:, 3] * c
    distance = np.minimum(_point_box_distance(ax, ay, hl, hw),
                          _point_box_distance(bx, by, hl, hw))
    for cx, cy in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
        distance = np.minimum(distance, _point_segment_distance(cx, cy, ax, ay, bx, by))
    return np.where(_segment_hits_box(ax, ay, bx, by, hl, hw), 0.0, distance)


def route_frame(points, route, first_index):
    """Continuous route index and signed lateral offset (+left) of each point."""
    a, b = route[:-1], route[1:]
    d = b - a
    length2 = np.maximum((d * d).sum(axis=1), 1e-12)
    # The official path repeats a point at some link joints; such a zero-length
    # segment has no direction and never defines the lateral offset.
    degenerate = (d * d).sum(axis=1) < 1e-6
    index, lateral = [], []
    for x, y in points:
        t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / length2, 0.0, 1.0)
        px, py = a[:, 0] + t * d[:, 0], a[:, 1] + t * d[:, 1]
        k = int(np.argmin(np.where(degenerate, np.inf, np.hypot(px - x, py - y))))
        index.append(first_index + k + float(t[k]))
        lateral.append(float((d[k, 0] * (y - a[k, 1]) - d[k, 1] * (x - a[k, 0]))
                             / math.sqrt(length2[k])))
    return np.asarray(index), np.asarray(lateral)


def route_pose(route, first_index, index):
    """Point, unit tangent and left normal of the official route at a fractional index."""
    k = int(np.clip(math.floor(index) - first_index, 0, len(route) - 2))
    t = float(index) - first_index - k
    a, b = route[k], route[k + 1]
    # Borrow the direction of the nearest non-degenerate segment at a repeated point.
    for j in sorted(range(len(route) - 1), key=lambda j: abs(j - k)):
        step = route[j + 1] - route[j]
        if np.hypot(*step) > 1e-3:
            tangent = step / np.hypot(*step)
            break
    return a + t * (b - a), tangent, np.array([-tangent[1], tangent[0]])


def centre_line_lateral(markings, route, first_index, index, reach_m=5.0):
    """Lateral offset (+left) of the nearest centre marking left of the route."""
    point, _, normal = route_pose(route, first_index, index)
    segments, _, labels = no_cross_segments(markings)
    best = None
    for (ax, ay, bx, by), label in zip(segments, labels):
        if not (label.endswith(":centre_guide") or ":{}:".format(CENTER_LINE) in label):
            continue
        d = np.array([bx - ax, by - ay])
        denominator = normal[0] * (-d[1]) - normal[1] * (-d[0])
        if abs(denominator) < 1e-9:
            continue
        rhs = np.array([ax, ay]) - point
        lateral = (rhs[0] * (-d[1]) - rhs[1] * (-d[0])) / denominator
        u = (normal[0] * rhs[1] - normal[1] * rhs[0]) / denominator
        if 0.0 <= u <= 1.0 and 0.0 < lateral <= reach_m:
            best = lateral if best is None else min(best, lateral)
    if best is None:
        raise ValueError("No centre marking left of route index {}".format(index))
    return float(best)


def evaluate_trace(trace, markings=None, route=None, policy=None, body=EGO_BODY):
    policy = dict(DEFAULT_POLICY, **(policy or {}))
    markings = load_markings() if markings is None else markings
    route_points, first_index = load_route() if route is None else route
    reasons = []
    result = {"schema": SCHEMA, "policy": policy, "ego_body": dict(body),
              "raw_samples": len(trace)}
    if len(trace) < 2:
        result.update(passed=False, reasons=["ego_trace_missing"])
        return result
    dense, max_gap = densify(trace, policy["densify_step_m"])
    centres = body_centres(dense, body)
    route_index, lateral = route_frame(centres, route_points, first_index)
    # Samples before the start or after the goal project onto a route end
    # point; their lateral offset is not a deviation from the segment.
    inside = ((route_index > SEGMENT_INDEX[0] + 0.5) & (route_index < SEGMENT_INDEX[1] - 0.5))
    covered = (route_index.min() <= REQUIRED_ROUTE_INDEX[0]
               and route_index.max() >= REQUIRED_ROUTE_INDEX[1])
    if not covered:
        reasons.append("ego_trace_incomplete")
    if max_gap > policy["max_sample_gap_m"]:
        reasons.append("ego_trace_gap")

    segments, half_width, labels = no_cross_segments(markings)
    best = (float("inf"), None, None)
    touched = {}
    for centre, yaw, keep in zip(centres, dense[:, 2], inside):
        if not keep:
            continue
        near = ((np.minimum(segments[:, 0], segments[:, 2]) < centre[0] + 5.0)
                & (np.maximum(segments[:, 0], segments[:, 2]) > centre[0] - 5.0)
                & (np.minimum(segments[:, 1], segments[:, 3]) < centre[1] + 5.0)
                & (np.maximum(segments[:, 1], segments[:, 3]) > centre[1] - 5.0))
        if not near.any():
            continue
        clearance = box_segment_clearance(centre, yaw, segments[near], body) - half_width[near]
        k = int(np.argmin(clearance))
        label = labels[int(np.flatnonzero(near)[k])]
        if clearance[k] < best[0]:
            best = (float(clearance[k]), label, [round(float(v), 3) for v in centre])
        for j in np.flatnonzero(clearance <= policy["min_marking_clearance_m"]):
            touched.setdefault(labels[int(np.flatnonzero(near)[j])],
                               [round(float(v), 3) for v in centre])
    if touched:
        reasons.append("solid_lane_marking_touched")
    zone_lateral = lateral[inside] if inside.any() else lateral
    zone_index = route_index[inside] if inside.any() else route_index
    max_right = max(0.0, float(-zone_lateral.min()))
    swinging = -zone_lateral >= SWING_ONSET_M
    onset = float(zone_index[np.argmax(swinging)]) if swinging.any() else None
    peak = float(zone_index[np.argmin(zone_lateral)])
    max_left = max(0.0, float(zone_lateral.max()))
    if max_right > policy["max_right_deviation_m"]:
        reasons.append("lateral_deviation_limit_exceeded")
    result.update({
        "dense_samples": int(len(dense)),
        "max_sample_gap_m": round(max_gap, 3),
        "route_index_span": [round(float(route_index.min()), 1),
                             round(float(route_index.max()), 1)],
        "markings_source_sha256": markings.get("source", {}).get("sha256"),
        "min_marking_clearance_m": round(best[0], 3) if best[1] else None,
        "closest_marking": best[1],
        "closest_at_body_centre_xy": best[2],
        "touched_markings": [{"marking": key, "first_body_centre_xy": value}
                             for key, value in sorted(touched.items())],
        "max_right_deviation_m": round(max_right, 3),
        "max_left_deviation_m": round(max_left, 3),
        "avoidance_class": avoidance_class(max_right),
        "swing_onset_route_index": round(onset, 1) if onset is not None else None,
        "peak_route_index": round(peak, 1),
        "trajectory_class": trajectory_class(max_right, onset),
        "passed": not reasons,
        "reasons": reasons,
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path, help="Scenario Runner --ego-trace CSV")
    source.add_argument("--active-signals", type=Path,
                        help="Sensor Record episode quality/active_signals.json")
    parser.add_argument("--markings", type=Path, default=DEFAULT_MARKINGS)
    parser.add_argument("--max-right-deviation-m", type=float,
                        default=DEFAULT_POLICY["max_right_deviation_m"])
    parser.add_argument("--min-marking-clearance-m", type=float,
                        default=DEFAULT_POLICY["min_marking_clearance_m"])
    args = parser.parse_args(argv)
    trace = (read_trace_csv(args.trace) if args.trace
             else read_active_signals(args.active_signals))
    result = evaluate_trace(trace, load_markings(args.markings), policy={
        "max_right_deviation_m": args.max_right_deviation_m,
        "min_marking_clearance_m": args.min_marking_clearance_m,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
