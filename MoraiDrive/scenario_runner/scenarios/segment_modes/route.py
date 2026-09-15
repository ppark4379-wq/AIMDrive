"""Route-window operations for new modes only; source paths are never edited."""
from dataclasses import dataclass
from typing import Tuple

from ...utils.transform_utils import polyline_length
from ..full_loop_parts import route_profile


# The proven collection expert can request up to 30 m of forward lookahead.
# Keep a margin beyond that so a bounded mission reaches its unchanged END
# before the expert exhausts its private steering path.
EXPERT_LOOKAHEAD_TAIL_M = 35.0


@dataclass(frozen=True)
class RouteWindow:
    full_points: Tuple
    prefix_s: float
    first_index: int
    last_index: int


def validate_boundary(points, boundary):
    if not 0 <= boundary.index < len(points):
        raise ValueError("Mission boundary index outside original route")
    actual = points[boundary.index]
    if any(abs(float(a) - b) > 1e-6 for a, b in zip(actual[:3], boundary.xyz)):
        raise ValueError("Mission boundary differs from approved XYZ coordinates")


def validate_window(points, mode):
    if mode.start is None and mode.end is None:
        return
    if mode.start is None or mode.end is None:
        raise ValueError("A mission requires both START and END")
    validate_boundary(points, mode.start)
    validate_boundary(points, mode.end)
    if mode.start.index >= mode.end.index:
        raise ValueError("Mission START must precede END")


def path_s_at_index(context, index):
    window = getattr(context, "mission_route", None)
    if window is None:
        return route_profile.path_s_at_index(context, index)
    index = max(0, min(int(index), len(window.full_points) - 1))
    return polyline_length(window.full_points[:index + 1]) - window.prefix_s


def remap_event_distances(context, prefix_s):
    for field in (
        "highway_activation_s", "highway_exit_s",
        "tunnel_entry_s", "tunnel_exit_s",
    ):
        setattr(context, field, getattr(context, field) - prefix_s)


def apply_route_window(context, mode):
    full_points = tuple(context.route_points)
    validate_window(full_points, mode)
    if mode.start is None:
        return
    first, last = mode.start.index, mode.end.index
    prefix = polyline_length(full_points[:first + 1])
    context.mission_route = RouteWindow(full_points, prefix, first, last)
    context.route_points = list(full_points[first:last + 1])
    context.route_length_m = polyline_length(context.route_points)
    context.pure_pursuit_last_s = 0.0
    context.expert_last_s = 0.0

    start, end = mode.start, mode.end
    context.start_tf = context.grpc.make_transform(*start.xyz, start.yaw_deg)
    context.goal_x, context.goal_y, context.goal_z = end.xyz
    context.goal_yaw = end.yaw_deg
    first_link = context.route_links.index(start.link_id)
    last_link = context.route_links.index(end.link_id)
    if first_link > last_link:
        raise ValueError("Mission link order differs from route order")
    context.route_links = context.route_links[first_link:last_link + 1]
    context.zone_allowed_links = set(context.route_links)
    context.start_link, context.end_link = start.link_id, end.link_id
    context.route_waypoint_indices = context.build_route_waypoint_indices()
    remap_event_distances(context, prefix)
    context.cfg["expert_is_closed_path"] = False


def build_expert_route_points(context):
    """Append an expert-only tail while preserving the mission START/END path."""
    window = getattr(context, "mission_route", None)
    if window is None:
        return context.route_points

    route_points = list(context.route_points)
    previous = window.full_points[window.last_index]
    tail_length_m = 0.0
    for point in window.full_points[window.last_index + 1:]:
        route_points.append(point)
        tail_length_m += polyline_length((previous, point))
        previous = point
        if tail_length_m >= EXPERT_LOOKAHEAD_TAIL_M:
            break

    if tail_length_m < EXPERT_LOOKAHEAD_TAIL_M:
        raise ValueError(
            "Mission END lacks enough source-route points for expert lookahead: "
            f"available={tail_length_m:.3f}m, "
            f"required={EXPERT_LOOKAHEAD_TAIL_M:.3f}m"
        )
    return route_points


def prepare_full_loop_route(context):
    context.mission_route = None
    debug = bool(context.cfg.get("route_debug_enabled", False))
    context.cfg["route_debug_enabled"] = False
    try:
        route_profile.prepare_full_loop_route(context)
        apply_route_window(context, context.mission_mode)
    finally:
        context.cfg["route_debug_enabled"] = debug
    mode = context.mission_mode
    first = mode.start.index if mode.start else 0
    last = mode.end.index if mode.end else len(context.route_points) - 1
    print(
        f"[Mission] route={mode.name} global_index={first}..{last} "
        f"start={context.route_points[0]} end={context.route_points[-1]} "
        f"length={context.route_length_m:.3f}m"
    )
    if debug:
        context.dump_route_debug()
