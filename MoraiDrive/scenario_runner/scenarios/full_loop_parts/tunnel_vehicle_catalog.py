"""Whole-tunnel vehicle plans constrained by actual MGeo topology."""

import math
from types import SimpleNamespace
from .highway import _route_geometry
from .tunnel_vehicle_brake import validate_sudden_brake
from .tunnel_dynamic_catalog import _finite, tunnel_section_bounds
from ...utils.transform_utils import (
    interpolate_on_polyline,
    project_distance_on_polyline,
)

BEHAVIORS = (
    "vehicle_slow",
    "vehicle_sudden_brake",
    "vehicle_oncoming",
    "vehicle_cut_in",
)


class UnsupportedVehicleTopology(ValueError):
    """A recognized behavior cannot be driven on this map."""


def lane_change_edge(loader, source, target):
    link = loader.link_set[source]
    return any(
        link.get("can_move_" + side + "_lane") is True
        and link.get(side + "_lane_change_dst_link_idx") == target
        for side in ("left", "right")
    )


def validate_vehicle_route(loader, links, allow_lane_change=False):
    if not links or len(set(links)) != len(links):
        raise ValueError("Vehicle route must contain distinct links")
    for link in links:
        if link not in loader.link_set or len(loader.get_link_points(link)) < 2:
            raise ValueError(f"Vehicle route link unavailable: {link}")
        if any(
            not all(math.isfinite(float(v)) for v in p[:3])
            for p in loader.get_link_points(link)
        ):
            raise ValueError(f"Non-finite MGeo point: {link}")
    for source, target in zip(links, links[1:]):
        a, b = loader.link_set[source], loader.link_set[target]
        if allow_lane_change and lane_change_edge(loader, source, target):
            continue
        pa, pb = loader.get_link_points(source)[-1], loader.get_link_points(target)[0]
        if (
            not a.get("to_node_idx")
            or a["to_node_idx"] != b.get("from_node_idx")
            or math.dist(pa[:3], pb[:3]) > 1
        ):
            raise ValueError(f"Disconnected vehicle route: {source} -> {target}")


def nearest_route_pose(loader, links, x, y):
    best = None
    for index, link in enumerate(links):
        points = loader.get_link_points(link)
        waypoint = min(
            range(len(points)),
            key=lambda i: math.hypot(points[i][0] - x, points[i][1] - y),
        )
        p = points[waypoint]
        error = math.hypot(p[0] - x, p[1] - y)
        if best is None or error < best[0]:
            a, b = (
                points[max(0, waypoint - 1)],
                points[min(len(points) - 1, waypoint + 1)],
            )
            best = (
                error,
                {
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "yaw_deg": math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
                    "link": link,
                    "waypoint": waypoint,
                    "route_index": index,
                },
            )
    return best


def direction_evidence(loader, links, points, entry_s, exit_s):
    evidence = []
    for link in links:
        lane = loader.get_link_points(link)
        for index in (0, len(lane) // 2, len(lane) - 1):
            p = lane[index]
            s = project_distance_on_polyline(points, p[0], p[1])
            if not entry_s + 0.1 < s < exit_s - 0.1:
                continue
            x, y, _, yaw = interpolate_on_polyline(points, s)
            if math.hypot(p[0] - x, p[1] - y) > 8:
                continue
            a, b = lane[max(0, index - 1)], lane[min(len(lane) - 1, index + 1)]
            norm = math.hypot(b[0] - a[0], b[1] - a[1])
            if norm < 1e-6:
                raise ValueError(f"Degenerate MGeo tangent: {link}")
            tx, ty = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            evidence.append(
                {
                    "link": link,
                    "s_m": s,
                    "tangent_dot": ((b[0] - a[0]) * tx + (b[1] - a[1]) * ty) / norm,
                    "lateral_m": -(p[0] - x) * ty + (p[1] - y) * tx,
                }
            )
    return evidence


def require_direction(evidence, opposite=False, separated=False):
    if not evidence or any(
        (e["tangent_dot"] >= -0.5 if opposite else e["tangent_dot"] <= 0.5)
        or (separated and abs(e["lateral_m"]) < 2)
        for e in evidence
    ):
        raise ValueError("Vehicle route direction/lane separation validation failed")


def cut_in_source_links(loader, zone):
    # Oncoming links supply drivable geometry only; they are never NPC routes.
    links = list(zone.get("oncoming_route_links") or [])
    if not links:
        raise UnsupportedVehicleTopology(
            "vehicle_cut_in unavailable: opposite source corridor missing"
        )
    validate_vehicle_route(loader, links)
    return links


def build_vehicle_plan(
    loader, points, entry_s, exit_s, zone, cfg, behavior, event_section=None
):
    if behavior not in BEHAVIORS:
        raise ValueError(f"Unknown tunnel vehicle behavior: {behavior}")
    bounds = tunnel_section_bounds(
        entry_s,
        exit_s,
        zone["candidate_catalog"]["division_count"],
        zone["candidate_catalog"]["endpoint_margin_m"],
    )
    sections = [
        {
            "section_index": i,
            "start_s_m": lo,
            "end_s_m": hi,
            "anchor_s_m": (lo + hi) / 2,
            "anchor_from_entry_m": (lo + hi) / 2 - entry_s,
        }
        for i, (lo, hi) in enumerate(bounds, 1)
    ]
    gap, minimum = _finite(cfg["initial_spawn_gap_m"], "initial_spawn_gap_m"), _finite(
        cfg["minimum_initial_gap_m"], "minimum_initial_gap_m"
    )
    if not 0 < minimum <= gap < exit_s - entry_s - cfg["pass_clearance_m"]:
        raise ValueError("Unsafe vehicle initial spawn gap / tunnel length")
    event = behavior == "vehicle_cut_in"
    safe = [
        s["section_index"]
        for s in sections
        if s["section_index"] in cfg["event_sections"]
        and s["anchor_s_m"] - entry_s >= minimum
        and exit_s - s["anchor_s_m"] >= cfg["pass_clearance_m"]
        and (s["anchor_s_m"] - entry_s - gap >= cfg["minimum_event_run_in_m"])
    ]
    if event and (
        not safe or (event_section is not None and event_section not in safe)
    ):
        raise ValueError("No safe tunnel event section for vehicle behavior")
    chosen = (
        (event_section if event_section is not None else safe[0]) if event else None
    )
    region = next((s for s in sections if s["section_index"] == chosen), None)
    forward = list(zone.get("npc_route_links") or zone["route_links"])
    validate_vehicle_route(loader, forward)
    links, spawn_s = forward, entry_s + gap
    speeds = {
        "vehicle_slow": "slow_speed_mps",
        "vehicle_sudden_brake": "cruise_speed_mps",
        "vehicle_oncoming": "oncoming_speed_mps",
        "vehicle_cut_in": "cut_in_speed_mps",
    }
    sudden = (
        validate_sudden_brake(cfg["sudden_brake"])
        if behavior == "vehicle_sudden_brake"
        else None
    )
    speed = (
        sudden["cruise_speed_mps"]
        if sudden
        else _finite(cfg[speeds[behavior]], speeds[behavior])
    )
    if speed <= 0:
        raise ValueError("Vehicle moving speed must be positive")
    if behavior == "vehicle_oncoming":
        links = list(zone.get("oncoming_route_links") or [])
        if not links:
            raise UnsupportedVehicleTopology(
                "vehicle_oncoming unavailable: actual opposite MGeo route missing"
            )
        validate_vehicle_route(loader, links)
        spawn_s = exit_s - zone["candidate_catalog"]["endpoint_margin_m"]
    elif behavior == "vehicle_cut_in":
        links = cut_in_source_links(loader, zone)
    evidence = direction_evidence(
        loader,
        links,
        points,
        entry_s,
        exit_s,
    )
    require_direction(
        evidence,
        opposite=behavior in ("vehicle_oncoming", "vehicle_cut_in"),
        separated=behavior in ("vehicle_oncoming", "vehicle_cut_in"),
    )
    x, y, _, _ = interpolate_on_polyline(points, spawn_s)
    _, pose = nearest_route_pose(loader, links, x, y)
    if behavior == "vehicle_cut_in":
        _, _, _, pose["yaw_deg"] = interpolate_on_polyline(points, spawn_s)
    actual_s = project_distance_on_polyline(points, pose["x"], pose["y"])
    if (
        actual_s - entry_s < minimum
        or abs(actual_s - spawn_s) > cfg["spawn_position_tolerance_m"]
    ):
        raise ValueError("MGeo vehicle spawn cannot provide the planned initial gap")
    # Follow the NPC's own connected route beyond the Ego segment boundary.
    cleanup_links = forward if behavior == "vehicle_cut_in" else links
    route_points, _, route_length = _route_geometry(
        SimpleNamespace(map_loader=loader), cleanup_links
    )
    boundary_s = entry_s if behavior == "vehicle_oncoming" else exit_s
    bx, by, _, _ = interpolate_on_polyline(points, boundary_s)
    boundary_route_s = project_distance_on_polyline(route_points, bx, by)
    cleanup_route_s = boundary_route_s + _finite(
        cfg["cleanup_after_boundary_m"], "cleanup_after_boundary_m"
    )
    if cleanup_route_s >= route_length:
        raise ValueError("Vehicle route has insufficient continuation for cleanup")
    cx, cy, cz, _ = interpolate_on_polyline(route_points, cleanup_route_s)
    return {
        "cleanup_route_links": cleanup_links,
        "cleanup_boundary_route_s_m": boundary_route_s,
        "cleanup_route_s_m": cleanup_route_s,
        "cleanup_xyz": {"x": cx, "y": cy, "z": cz},
        "cleanup_policy": "outside_tunnel_and_ego_clearance",
        "behavior": behavior,
        "event_section": chosen,
        "safe_event_sections": list(cfg["event_sections"]) if sudden else safe,
        "event_region": region,
        **(
            {
                "sudden_brake_settings": sudden,
                "brake_regions": [
                    s for s in sections if s["section_index"] in cfg["event_sections"]
                ],
            }
            if sudden
            else {}
        ),
        "sections": sections,
        "entry_s_m": entry_s,
        "exit_s_m": exit_s,
        "spawn_policy": "runtime_ego_gap_or_oncoming_eta",
        "spawn_preview_only": True,
        "spawn_s_m": actual_s,
        "spawn_target": pose,
        "route_links": [] if behavior == "vehicle_cut_in" else links,
        **(
            {
                "source_corridor_links": links,
                "source_geometry_only": True,
                "control_strategy": "direct_steering_then_forward_route",
            }
            if behavior == "vehicle_cut_in"
            else {}
        ),
        "target_route_links": forward,
        "speed_mps": speed,
        "direction_evidence": evidence,
        "lifecycle": "each_actor_through_continuation_or_teardown",
    }


def build_behavior_catalog(loader, points, entry_s, exit_s, zone, cfg):
    plans, unavailable = {}, {}
    for behavior in BEHAVIORS:
        try:
            plans[behavior] = build_vehicle_plan(
                loader, points, entry_s, exit_s, zone, cfg, behavior
            )
        except UnsupportedVehicleTopology as error:
            unavailable[behavior] = str(error)
    return {
        "behaviors": plans,
        "unavailable_behaviors": unavailable,
        "sections": plans["vehicle_slow"]["sections"],
    }
