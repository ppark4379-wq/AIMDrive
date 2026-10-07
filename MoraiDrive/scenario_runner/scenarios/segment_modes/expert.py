"""Adapt scenario-owned runtime inputs before launching the Expert.

Tunnel vehicle trials request official-path ACC instead of recorded Jamming.
Full-loop settings remain intact. Segment modes retain only speed zones wholly
contained by their exported route, for both link-ID and ENU-coordinate zones.
"""
import json
from pathlib import Path


FILTERED_MISSIONS = frozenset((
    "full_loop_static", "full_loop_dynamic", "full_loop_roundabout",
    "full_loop_roundabout_nonuniform", "full_loop_roundabout_speed_random",
    "full_loop_roundabout_brake",
    "full_loop_highway",
    "full_loop_tunnel_dynamic",
    "full_loop_tunnel_vehicle",
))


def nearest_index(points, target):
    return min(
        range(len(points)),
        key=lambda index: (
            (points[index][0] - target[0]) ** 2
            + (points[index][1] - target[1]) ** 2
        ),
    )


def zone_endpoints(zone, map_loader):
    """Resolve either an ENU-coordinate or legacy link-ID speed zone."""
    if "start_point" in zone or "end_point" in zone:
        if "start_point" not in zone or "end_point" not in zone:
            raise ValueError(
                "Speed zone must define both start_point and end_point: "
                + str(zone.get("name", "unnamed"))
            )
        return zone["start_point"], zone["end_point"]
    if "start_link_id" not in zone or "end_link_id" not in zone:
        raise ValueError(
            "Speed zone needs coordinate points or link IDs: "
            + str(zone.get("name", "unnamed"))
        )
    return (
        map_loader.get_link_points(zone["start_link_id"])[0],
        map_loader.get_link_points(zone["end_link_id"])[-1],
    )


def select_speed_zones(zones, window, map_loader):
    """Discard disjoint zones; reject partial overlaps instead of misprojecting.

    Compare on the original full route, NOT on the shortened route: nearest
    points on a short route can incorrectly reverse a distant zone's endpoints.
    """
    kept, skipped = [], []
    for zone in zones:
        start, end = zone_endpoints(zone, map_loader)
        first = nearest_index(window.full_points, start)
        last = nearest_index(window.full_points, end)
        spans = ((first, last),) if first <= last else (
            (first, len(window.full_points) - 1), (0, last),
        )
        overlaps = any(
            lower <= window.last_index and upper >= window.first_index
            for lower, upper in spans
        )
        if not overlaps:
            skipped.append(zone.get("name", "unnamed"))
        elif window.first_index <= first <= last <= window.last_index:
            kept.append(zone)
        else:
            raise ValueError(
                "Mission speed zone partially overlaps route; explicit clipping "
                f"is required: {zone.get('name', 'unnamed')} "
                f"zone={first}..{last}, "
                f"route={window.first_index}..{window.last_index}"
            )
    return kept, skipped


def prepare_speed_zones(context):
    """Run after runtime export and before the external expert is launched."""
    if context.mission_mode.name not in FILTERED_MISSIONS:
        return
    path = Path(context.expert_controller.runtime_config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    profile = config["planning"]["velocity_profile"]
    zones = profile.get("speed_limit_zones") or []
    kept, skipped = select_speed_zones(
        zones,
        context.mission_route,
        context.map_loader,
    )
    changed = bool(skipped)
    if skipped:
        profile["speed_limit_zones"] = kept
    if context.mission_mode.name == "full_loop_tunnel_vehicle":
        # Lead-vehicle trials use the exported global path and ordinary ACC.
        # A temporarily stopped NPC must not commit a recorded static detour.
        config["planning"].setdefault("jamming", {})["enabled"] = False
        changed = True
        print("[Mission] tunnel vehicle Expert: global path + ACC; recorded Jamming disabled")
    if changed:
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"[Mission] speed zones mode={context.mission_mode.name} "
        f"kept={len(kept)} excluded_outside_route={skipped}"
    )
