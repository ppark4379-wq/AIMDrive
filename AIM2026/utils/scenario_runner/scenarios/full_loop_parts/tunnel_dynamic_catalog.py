"""Offline tunnel pedestrian path candidates derived from the official route.

Candidates are geometry only. MORAI surface height and collision clearance
must be reviewed before a path is marked approved for later runtime use.
"""

import math

from ...utils.transform_utils import interpolate_on_polyline, polyline_length


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def route_offset_point(route_points, s_m, d_m):
    """Return XYZ at route distance s and left-positive lateral offset d."""
    s_m = _finite(s_m, "s_m")
    d_m = _finite(d_m, "d_m")
    length = polyline_length(route_points)
    if length <= 0.0 or not 0.0 <= s_m <= length:
        raise ValueError("s_m is outside the official route")
    x, y, z, yaw_deg = interpolate_on_polyline(route_points, s_m)
    yaw = math.radians(yaw_deg)
    return {"x": x - d_m * math.sin(yaw),
            "y": y + d_m * math.cos(yaw), "z": z}


def tunnel_section_bounds(entry_s, exit_s, division_count, margin):
    """Shared longitudinal sections for pedestrian and vehicle candidates."""
    entry_s = _finite(entry_s, "entry_s")
    exit_s = _finite(exit_s, "exit_s")
    margin = _finite(margin, "endpoint_margin_m")
    divisions = _finite(division_count, "division_count")
    if divisions <= 0 or not divisions.is_integer():
        raise ValueError("division_count must be a positive integer")
    if margin < 0:
        raise ValueError("endpoint margin must not be negative")
    start, end = entry_s + margin, exit_s - margin
    length = (end - start) / int(divisions)
    if length <= 0:
        raise ValueError("endpoint margin leaves no usable tunnel length")
    return [(start + length * i, start + length * (i + 1))
            for i in range(int(divisions))]


def generate_tunnel_candidates(route_points, entry_s, exit_s, spec):
    """Expand compact route-relative templates into reviewable straight paths."""
    if not isinstance(spec, dict):
        raise ValueError("tunnel candidate_catalog must be a mapping")
    entry_s = _finite(entry_s, "entry_s")
    exit_s = _finite(exit_s, "exit_s")
    margin = _finite(spec.get("endpoint_margin_m"), "endpoint_margin_m")
    offset = _finite(spec.get("side_offset_m"), "side_offset_m")
    opposite_lane_1 = _finite(spec.get("opposite_lane_1_offset_m"),
                              "opposite_lane_1_offset_m")
    opposite_lane_2 = _finite(spec.get("opposite_lane_2_offset_m"),
                              "opposite_lane_2_offset_m")
    if (max(abs(opposite_lane_1), abs(opposite_lane_2)) > 5.0
            or opposite_lane_1 <= opposite_lane_2):
        raise ValueError("opposite lane offsets must be bounded and lane 1 left of lane 2")
    straight = _finite(spec.get("straight_length_m"), "straight_length_m")
    cross_delta = _finite(spec.get("cross_delta_s_m"), "cross_delta_s_m")
    diagonal_delta = _finite(spec.get("diagonal_delta_s_m"), "diagonal_delta_s_m")
    route_length = polyline_length(route_points)
    if not -1e-6 <= entry_s < exit_s <= route_length + 1e-6:
        raise ValueError("tunnel entry/exit is outside the official route")
    # Full-route subtraction and a sliced route can differ by a few ulps.
    entry_s = max(0.0, entry_s)
    exit_s = min(route_length, exit_s)
    if margin < 0 or not 0 < offset <= 5 or min(straight, cross_delta, diagonal_delta) <= 0:
        raise ValueError("tunnel candidate dimensions must be positive and bounded")
    sections = tunnel_section_bounds(entry_s, exit_s, spec.get("division_count"), margin)
    section_length = sections[0][1] - sections[0][0]
    if max(straight, cross_delta, diagonal_delta) > section_length + 1e-6:
        raise ValueError("candidate length exceeds a tunnel section")
    approved = set(spec.get("approved_ids") or [])
    disabled = set(spec.get("disabled_ids") or [])
    if approved & disabled:
        raise ValueError("candidate IDs cannot be both approved and disabled")

    candidates = []
    for number, (section_start, section_end) in enumerate(sections, 1):
        s = (section_start + section_end) / 2.0
        opposite_lane = 1 if number % 2 else 2
        opposite_d = opposite_lane_1 if opposite_lane == 1 else opposite_lane_2
        templates = {
            "straight_same_direction": ((s - straight / 2, offset),
                                        (s + straight / 2, offset)),
            "straight_opposite": ((s + straight / 2, opposite_d),
                                  (s - straight / 2, opposite_d)),
            "cross_left_to_right": ((s - cross_delta / 2, offset),
                                    (s + cross_delta / 2, -offset)),
            "cross_right_to_left": ((s - cross_delta / 2, -offset),
                                    (s + cross_delta / 2, offset)),
            "diagonal_cross": ((s - diagonal_delta / 2, offset),
                               (s + diagonal_delta / 2, -offset)),
        }
        for behavior, (start, end) in templates.items():
            candidate_id = f"{behavior}_{number:02d}"
            if not all(section_start - 1e-6 <= point[0] <= section_end + 1e-6
                       for point in (start, end)):
                raise ValueError(f"{candidate_id} extends outside its tunnel section")
            start_xyz = route_offset_point(route_points, *start)
            end_xyz = route_offset_point(route_points, *end)
            if math.hypot(end_xyz["x"] - start_xyz["x"],
                          end_xyz["y"] - start_xyz["y"]) < 1.0:
                raise ValueError(f"{candidate_id} is shorter than 1 m")
            candidates.append({
                "id": candidate_id,
                "behavior": behavior,
                "section_index": number,
                "section_s_range_m": {"start": section_start, "end": section_end},
                "anchor_s_m": s,
                "anchor_from_entry_m": s - entry_s,
                "status": ("approved" if candidate_id in approved else
                           "disabled" if candidate_id in disabled else "candidate"),
                "start_sd": {"s_m": start[0], "d_m": start[1]},
                "end_sd": {"s_m": end[0], "d_m": end[1]},
                "start_xyz": start_xyz,
                "end_xyz": end_xyz,
            })
            if behavior == "straight_opposite":
                candidates[-1]["lane_index"] = opposite_lane
    unknown = (approved | disabled) - {item["id"] for item in candidates}
    if unknown:
        raise ValueError(f"Unknown tunnel candidate IDs: {sorted(unknown)}")
    return candidates


def select_tunnel_mix(candidates, rng, behaviors, allow_repeat=False, approved_only=True):
    """Choose one eligible behavior per section, with seeded backtracking.

    Backtracking avoids rejecting a feasible mix when a later section has
    only one approved behavior left. No disabled candidate is ever selected.
    """
    known = {item["behavior"] for item in candidates}
    if (not isinstance(behaviors, list) or not behaviors
            or any(not isinstance(name, str) for name in behaviors)
            or len(set(behaviors)) != len(behaviors) or set(behaviors) - known):
        raise ValueError("random_mix.behaviors must list distinct known behaviors")
    sections = sorted({item["section_index"] for item in candidates})
    if not sections or (not allow_repeat and len(behaviors) < len(sections)):
        raise ValueError("Not enough distinct behaviors for tunnel sections")
    options = {}
    for section in sections:
        options[section] = sorted(
            (item for item in candidates
             if item["section_index"] == section and item["behavior"] in behaviors
             and item["status"] != "disabled"
             and (not approved_only or item["status"] == "approved")),
            key=lambda item: item["id"],
        )
        rng.shuffle(options[section])

    def choose(index, selected, used):
        if index == len(sections):
            return selected
        for item in options[sections[index]]:
            if allow_repeat or item["behavior"] not in used:
                result = choose(index + 1, selected + [item], used | {item["behavior"]})
                if result is not None:
                    return result
        return None

    selected = choose(0, [], set())
    if selected is None:
        raise ValueError("No eligible tunnel mix covers every section; check review statuses")
    return selected
