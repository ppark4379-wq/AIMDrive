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


def generate_tunnel_candidates(route_points, entry_s, exit_s, spec):
    """Expand compact route-relative templates into reviewable straight paths."""
    if not isinstance(spec, dict):
        raise ValueError("tunnel candidate_catalog must be a mapping")
    entry_s = _finite(entry_s, "entry_s")
    exit_s = _finite(exit_s, "exit_s")
    margin = _finite(spec.get("endpoint_margin_m"), "endpoint_margin_m")
    offset = _finite(spec.get("side_offset_m"), "side_offset_m")
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
    positions = spec.get("positions_from_entry_m")
    if not isinstance(positions, list) or not positions:
        raise ValueError("positions_from_entry_m must be a nonempty list")
    approved = set(spec.get("approved_ids") or [])
    disabled = set(spec.get("disabled_ids") or [])
    if approved & disabled:
        raise ValueError("candidate IDs cannot be both approved and disabled")

    candidates = []
    for number, raw_position in enumerate(positions, 1):
        s = entry_s + _finite(raw_position, "positions_from_entry_m")
        templates = {
            "straight_same_direction": ((s, offset), (s + straight, offset)),
            "straight_opposite": ((s + straight, -offset), (s, -offset)),
            "cross_left_to_right": ((s, offset), (s + cross_delta, -offset)),
            "cross_right_to_left": ((s, -offset), (s + cross_delta, offset)),
            "diagonal_cross": ((s, offset), (s + diagonal_delta, -offset)),
        }
        for behavior, (start, end) in templates.items():
            candidate_id = f"{behavior}_{number:02d}"
            if not all(entry_s + margin <= point[0] <= exit_s - margin
                       for point in (start, end)):
                raise ValueError(f"{candidate_id} extends outside the tunnel margin")
            start_xyz = route_offset_point(route_points, *start)
            end_xyz = route_offset_point(route_points, *end)
            if math.hypot(end_xyz["x"] - start_xyz["x"],
                          end_xyz["y"] - start_xyz["y"]) < 1.0:
                raise ValueError(f"{candidate_id} is shorter than 1 m")
            candidates.append({
                "id": candidate_id,
                "behavior": behavior,
                "status": ("approved" if candidate_id in approved else
                           "disabled" if candidate_id in disabled else "candidate"),
                "start_sd": {"s_m": start[0], "d_m": start[1]},
                "end_sd": {"s_m": end[0], "d_m": end[1]},
                "start_xyz": start_xyz,
                "end_xyz": end_xyz,
            })
    unknown = (approved | disabled) - {item["id"] for item in candidates}
    if unknown:
        raise ValueError(f"Unknown tunnel candidate IDs: {sorted(unknown)}")
    return candidates
