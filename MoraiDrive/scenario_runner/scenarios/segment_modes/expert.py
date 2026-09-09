"""Adapt only the three short missions' scenario-owned expert inputs.

The external expert, full-loop modes, and highway settings are not changed.
"""
import json
from pathlib import Path


SHORT_MISSIONS = frozenset((
    "full_loop_static", "full_loop_dynamic", "full_loop_roundabout",
))


def nearest_index(points, target):
    return min(
        range(len(points)),
        key=lambda index: (
            (points[index][0] - target[0]) ** 2
            + (points[index][1] - target[1]) ** 2
        ),
    )


def select_speed_zones(zones, window, map_loader):
    """Discard disjoint zones; reject partial overlaps instead of misprojecting.

    Compare on the original full route, NOT on the shortened route: nearest
    points on a short route can incorrectly reverse a distant zone's endpoints.
    """
    kept, skipped = [], []
    for zone in zones:
        start = map_loader.get_link_points(zone["start_link_id"])[0]
        end = map_loader.get_link_points(zone["end_link_id"])[-1]
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
    if context.mission_mode.name not in SHORT_MISSIONS:
        return
    path = Path(context.expert_controller.runtime_config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    profile = config["planning"]["velocity_profile"]
    kept, skipped = select_speed_zones(
        profile.get("speed_limit_zones") or [],
        context.mission_route,
        context.map_loader,
    )
    if skipped:
        profile["speed_limit_zones"] = kept
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"[Mission] speed zones mode={context.mission_mode.name} "
        f"kept={len(kept)} excluded_outside_route={skipped}"
    )
