"""Explicit experimental obstacle arrangements, never prescribed Ego paths.

Templates describe where measured native actors go. The production Jamming
planner must still find and verify the actual path through each arrangement.
No simulator, ROS, Expert, NumPy or Shapely imports occur when listing them.
"""

import copy
import math


DOMAIN = "two_lane_physical_road"
SCHEMA = "aimdrive_tunnel_advanced_pattern_v1"
ROAD_MARGIN_M = 0.15
REFERENCE_EGO = {"rear_m": .86050, "front_m": 3.91422, "width_m": 2.36623,
                 "planning_road_margin_m": .35, "runtime_road_margin_m": .15}
REGIONS = {
    "own_lane": {"center_lateral_m": 0., "jitter_m": .25, "lane": "own"},
    "opposing_lane": {"center_lateral_m": 3.5, "jitter_m": .25, "lane": "opposing"},
    "near_divider": {"center_lateral_m": 1.75, "jitter_m": .20, "lane": "nearest"},
    "outer_own": {"center_lateral_m": -.85, "jitter_m": .05, "lane": "own"},
    "outer_opposing": {"center_lateral_m": 4.35, "jitter_m": .05, "lane": "opposing"},
}
PATTERNS = {
    "short_detour": {"counts": [1], "default_count": 1,
        "slots": {1: [(40., "own_lane")]},
        "design": {"partial_lateral_shift_m": 2.7, "entry_length_m": 10., "return_length_m": 28.,
                   "suitable_parallel_obstacle_width_m": 1.59377}},
    "asymmetric_return": {"counts": [1], "default_count": 1,
        "slots": {1: [(35., "own_lane")]},
        "design": {"partial_lateral_shift_m": 2.7, "entry_length_m": 12., "return_length_m": 40.}},
    "long_group_bypass": {"counts": [1, 2, 3], "default_count": 2,
        "slots": {1: [(60., "own_lane")], 2: [(45., "own_lane"), (75., "own_lane")],
                  3: [(35., "own_lane"), (55., "own_lane"), (75., "own_lane")]},
        "design": {"full_lateral_shift_m": 3.5, "entry_length_m": 30., "return_length_m": 30.}},
    "independent_detours": {"counts": [2], "default_count": 2,
        "slots": {2: [(25., "own_lane"), (75., "own_lane")]},
        "minimum_center_gap_m": 50.,
        "design": {"partial_lateral_shift_m": 2.7, "entry_length_m": 10., "return_length_m": 28.,
                   "minimum_surface_gap_reference_m": 39.24}},
    "alternating_three": {"counts": [3], "default_count": 3,
        "slots": {3: [(25., "own_lane"), (50., "opposing_lane"), (70., "own_lane")]},
        "design": {"partial_lateral_shift_m": 2.7, "interior_return_lateral_m": .35,
                   "interior_shift_length_m": 10., "interior_return_length_m": 15., "return_length_m": 28.}},
    "window_bypass": {"counts": [2, 3], "default_count": 2,
        "slots": {2: [(50., "outer_own"), (50., "outer_opposing")],
                  3: [(45., "outer_own"), (45., "outer_opposing"), (85., "own_lane")]},
        "enforce_adjacent_center_gap": False,
        "design": {"window_reference_lateral_m": 1.75,
                   "side_by_side_obstacles_require_full_body_planner_gate": True}},
}
PATTERN_NAMES = tuple(PATTERNS)
PATTERN_COUNT_CAPS = {name: tuple(value["counts"]) for name, value in PATTERNS.items()}


def describe_patterns():
    return [{"name": name, "actor_count_caps": list(value["counts"]),
             "default_count": value["default_count"], "design_reference": copy.deepcopy(value["design"]),
             "actual_jamming_path_required": True} for name, value in PATTERNS.items()]


def resolve_slots(pattern, count, window, rng, *, min_pair_gap_m, lateral_regions=None):
    """Keep template spacing; translate its group within a requested window."""
    if pattern is not None and pattern not in PATTERNS:
        raise ValueError("Unknown expanded tunnel pattern: " + str(pattern))
    specification = PATTERNS.get(pattern)
    if count is None:
        count = specification["default_count"] if specification else rng.choice((1, 2))
    if specification and count not in specification["counts"]:
        raise ValueError("Pattern {} supports actor counts {}".format(pattern, specification["counts"]))
    low, high = window
    if specification:
        anchors = list(specification["slots"][count])
        station_low, station_high = min(s for s, _ in anchors), max(s for s, _ in anchors)
        if station_high-station_low > high-low+1e-9:
            raise ValueError("Station window is too short for this pattern's body/gap design")
        shift_low, shift_high = low-station_low, high-station_high
        # Preserve the documented anchors when possible, with one shared
        # deterministic jitter instead of compressing independent maneuvers.
        shift = rng.uniform(max(shift_low, -.5), min(shift_high, .5)) if shift_low <= .5 and shift_high >= -.5 else min(max(0., shift_low), shift_high)
        slots = [{"station_m": station+shift, "region": region} for station, region in anchors]
        gap = max(min_pair_gap_m, specification.get("minimum_center_gap_m", 0.))
        enforce = specification.get("enforce_adjacent_center_gap", True)
        if enforce and any(b["station_m"]-a["station_m"] < gap-1e-9 for a, b in zip(slots, slots[1:])):
            raise ValueError("Pattern spacing is below the requested center gap")
    else:
        gap, enforce = min_pair_gap_m, True
        needed = (count-1)*gap
        if high-low < needed:
            raise ValueError("Expanded station window cannot satisfy requested actor gaps")
        samples = sorted(rng.uniform(low, high-needed) for _ in range(count))
        slots = [{"station_m": station+i*gap, "region": "own_lane"}
                 for i, station in enumerate(samples)]
    if lateral_regions is not None:
        regions = [lateral_regions] if isinstance(lateral_regions, str) else list(lateral_regions)
        if len(regions) == 1:
            regions *= count
        if len(regions) != count or any(region not in REGIONS for region in regions):
            raise ValueError("Lateral regions must supply one region or one known region per actor")
        for slot, region in zip(slots, regions):
            slot["region"] = region
    return count, slots, gap, enforce, copy.deepcopy(specification["design"] if specification else {})


def yaw_options(mapping, model):
    values = mapping.get(model, [0.]) if mapping is not None else [0.]
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise ValueError("Static yaw offsets must be numbers or a list per model")
    offsets = []
    for value in values:
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("Static yaw offsets must be finite degrees")
        value = (float(value)+180.) % 360.-180.
        if value not in offsets:
            offsets.append(value)
    return offsets
