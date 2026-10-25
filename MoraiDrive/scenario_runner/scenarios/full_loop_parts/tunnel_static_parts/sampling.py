"""Opt-in sampling of native actors throughout the mapped two-lane tunnel.

Placement intents describe actor arrangements, never an Ego maneuver. The
production planner and each live trial must still verify the actual path.
Legacy generation and its random draw sequence are deliberately separate.
"""

import copy
import math
import random
from collections.abc import Mapping

from .advanced_patterns import DOMAIN, REFERENCE_EGO, VEHICLE_YAW_KEY, yaw_options, yaw_options_for
from .layout import (
    MAP_NAME, OPPOSING_LINK, SCHEMA as CANDIDATE_SCHEMA, TUNNEL_LINK,
    _finite, _hash, _integer, expanded_road_boundary, normalize_footprint,
    projected_footprint_dimensions, route_pose, tunnel_geometry, validate_candidate,
)


SCHEMA = "aimdrive_tunnel_static_sampling_v1"
PLACEMENT_INTENTS = (
    "random_scatter", "own_lane_scatter", "opposing_lane_scatter",
    "alternating_lanes", "same_station_window",
)
NATIVE_STATIC_YAW_POLICY = {
    "Steel_Barricade": [90.], "concrete-barrier": [90.], "Rockslide": [],
}


class SamplingExhausted(ValueError):
    """A bounded sampling search found no placement under the configured gates."""

    def __init__(self, attempts, reason):
        self.attempts = attempts
        self.last_rejection = reason
        super().__init__("No sampled native layout passed geometry gates after {} attempts: {}".format(
            attempts, reason))


def _yaw_policy(requested):
    if requested is not None and not isinstance(requested, Mapping):
        raise ValueError("Static yaw offsets must map model names to degrees")
    policy = copy.deepcopy(NATIVE_STATIC_YAW_POLICY)
    for name in sorted(requested or {}):
        values = yaw_options(requested, name)
        if name == VEHICLE_YAW_KEY:
            policy[name] = values   # opt-in oblique parked cars; every listed angle is screened like any other layout
            continue
        permitted = NATIVE_STATIC_YAW_POLICY.get(name, [0.])
        if any(value not in permitted for value in values):
            raise ValueError("Sampling yaw policy exceeds native permitted rotations: " + str(name))
        policy[name] = values
    return policy


def _model_pool(static_catalog, vehicles, actor_kind, width_cap, yaw_policy):
    statics = static_catalog.get("models", static_catalog)
    pool, excluded = [], []
    for kind, models in (("obstacle", statics), ("vehicle", vehicles)):
        if actor_kind != "mixed" and actor_kind != kind:
            continue
        if not isinstance(models, Mapping):
            raise ValueError("Sampling catalogs must map names to measured native footprints")
        for name in sorted(models):
            raw = models[name]
            try:
                if (not isinstance(name, str) or not name or not isinstance(raw, Mapping)
                        or raw.get("geometry_verified") is not True
                        or not isinstance(raw.get("geometry_source"), Mapping)
                        or not raw["geometry_source"]):
                    raise ValueError("verified geometry evidence is missing")
                scale = raw.get("scale_xyz", raw.get("scale", [1., 1., 1.]))
                if (not isinstance(scale, (list, tuple)) or len(scale) != 3
                        or any(isinstance(value, bool) or float(value) != 1. for value in scale)):
                    raise ValueError("measured geometry is not at native unit scale")
                footprint = normalize_footprint(raw)
                offsets = yaw_options_for(yaw_policy, kind, name)
                for offset in offsets:
                    projected = projected_footprint_dimensions(footprint, offset)
                    if projected["width_m"] <= width_cap+1e-9:
                        pool.append((kind, name, footprint, raw, offset, projected["width_m"]))
                if not offsets or not any(projected_footprint_dimensions(footprint, angle)["width_m"]
                                          <= width_cap+1e-9 for angle in offsets):
                    raise ValueError("no permitted native yaw fits the projected road width")
            except (KeyError, TypeError, ValueError) as error:
                excluded.append({"actor_kind": kind, "model": name, "reason": str(error)})
    if not pool:
        raise ValueError("No verified native model/yaw combination fits sampled placement")
    return pool, excluded


def _sample_slots(rng, count, intent, window, gap_range):
    low, high = window
    gap_low, gap_high = gap_range
    gaps = [rng.uniform(gap_low, gap_high) for _ in range(count-1)]
    if intent == "same_station_window":
        gaps[0] = 0.
    span = sum(gaps)
    if span > high-low+1e-9:
        raise ValueError("sampled independent gaps exceed the station window")
    start = rng.uniform(low, high-span)
    stations = [start]
    for gap in gaps:
        stations.append(stations[-1]+gap)
    if intent in ("own_lane_scatter", "opposing_lane_scatter"):
        lanes = ["own" if intent == "own_lane_scatter" else "opposing"]*count
    elif intent == "alternating_lanes":
        first = rng.randrange(2)
        lanes = ["own" if (first+index) % 2 == 0 else "opposing" for index in range(count)]
    elif intent == "same_station_window":
        lanes = ["own", "opposing"]
        if rng.randrange(2):
            lanes.reverse()
        lanes.extend(rng.choice(("own", "opposing")) for _ in range(count-2))
    else:
        lanes = [None]*count
    return list(zip(stations, lanes))


def _place_actor(rng, slot, choice, geometry):
    station, requested_lane = slot
    kind, model, footprint, raw, offset, projected_width = choice
    # This necessary nominal width bound speeds rejection. Complete mapped
    # polygons, not this interval, establish the final placement gate.
    half_width = .5*projected_width+.15
    low = -.5*geometry["own_lane_width_m"]+half_width
    high = geometry["own_lane_width_m"]*.5+geometry["opposing_lane_width_m"]-half_width
    divider = geometry["own_lane_width_m"]*.5
    if requested_lane == "own":
        high = min(high, divider)
    elif requested_lane == "opposing":
        low = max(low, divider)
    if low > high:
        raise ValueError("selected native footprint cannot occupy its requested lane side")
    lateral = rng.uniform(low, high)
    lane = TUNNEL_LINK if lateral <= divider else OPPOSING_LINK
    position, route_yaw = route_pose(geometry["own_points"], station)
    native_yaw = math.degrees(route_yaw)+(180. if lane == OPPOSING_LINK else 0.)
    spawn_yaw = math.radians(native_yaw+offset-footprint["heading_offset_deg"])
    centre = (position[0]-math.sin(route_yaw)*lateral, position[1]+math.cos(route_yaw)*lateral)
    cx, cy = footprint["center_offset_spawn_xy"]
    row = {
        "actor_kind": kind, "model": model, "scale": [1., 1., 1.],
        "station_m": station, "lateral_m": lateral, "yaw_deg": math.degrees(spawn_yaw),
        "planned_bbox": copy.deepcopy(footprint),
        "position": {"x": centre[0]-cx*math.cos(spawn_yaw)+cy*math.sin(spawn_yaw),
                     "y": centre[1]-cx*math.sin(spawn_yaw)-cy*math.cos(spawn_yaw), "z": position[2]},
        "placement_lane_id": lane, "placement_region": "continuous_two_lane",
        "native_forward_yaw_deg": native_yaw, "static_yaw_offset_deg": offset,
        "traffic_direction": "own_forward" if lane == TUNNEL_LINK else "opposing_forward",
    }
    for key in ("geometry_source", "expected_api_raw"):
        if key in raw:
            row[key] = copy.deepcopy(raw[key])
    return row


def generate_sampled_candidate(map_loader, static_catalog, vehicle_models, seed, *, count=None,
                               candidate_index=0, allow_three=False,
                               station_window_m=(20., 105.), gap_range_m=(0., 55.),
                               placement_intent="random_scatter", actor_kind="mixed",
                               static_yaw_offsets_deg=None, max_attempts=200, max_width_m=6.7,
                               _road_boundary=None):
    """Sample one deterministic, geometrically valid candidate without live I/O.

    ``count=3`` and ``allow_three=True`` are both required for three actors.
    Each ordinary adjacent gap is sampled independently; same-station windows
    deliberately use a zero first gap and must pass every pair's inflated OBB
    check. ``gap_range_m`` applies to the remaining gaps in that intent.
    Model pools must already be restricted to the campaign's reviewed allowlist.
    ``_road_boundary`` permits reuse of the same production boundary object.
    """
    _integer(seed, "seed")
    _integer(candidate_index, "candidate_index", True)
    if not isinstance(allow_three, bool):
        raise ValueError("allow_three must be a boolean")
    if count is not None and _integer(count, "count") not in ((1, 2, 3) if allow_three else (1, 2)):
        raise ValueError("Count three requires explicit count=3 and allow_three=True")
    _integer(max_attempts, "max_attempts")
    if not 1 <= max_attempts <= 10000:
        raise ValueError("max_attempts must be in 1..10000")
    if placement_intent not in PLACEMENT_INTENTS:
        raise ValueError("Unknown tunnel placement intent: " + str(placement_intent))
    if placement_intent == "same_station_window" and count not in (2, 3):
        raise ValueError("Same-station windows require explicit count=2 or count=3")
    if actor_kind not in ("mixed", "obstacle", "vehicle"):
        raise ValueError("actor_kind must be mixed, obstacle, or vehicle")
    geometry = tunnel_geometry(map_loader)
    if not isinstance(station_window_m, (list, tuple)) or len(station_window_m) != 2:
        raise ValueError("Station window must contain two finite values")
    low, high = [_finite(value, "sampled station window") for value in station_window_m]
    if not .01 < low <= high < geometry["length_m"]-.01:
        raise ValueError("Sampled station window is outside mapped tunnel portals")
    if not isinstance(gap_range_m, (list, tuple)) or len(gap_range_m) != 2:
        raise ValueError("Gap range must contain two finite values")
    gap_low, gap_high = [_finite(value, "sampled gap range") for value in gap_range_m]
    if not 0. <= gap_low <= gap_high:
        raise ValueError("Sampled gap range must satisfy 0 <= minimum <= maximum")
    width_cap = min(_finite(max_width_m, "max_width_m", True),
                    geometry["own_lane_width_m"]+geometry["opposing_lane_width_m"]-.30)
    yaw_policy = _yaw_policy(static_yaw_offsets_deg)
    pool, excluded = _model_pool(static_catalog, vehicle_models, actor_kind, width_cap, yaw_policy)
    boundary = _road_boundary if _road_boundary is not None else expanded_road_boundary(map_loader)
    rng = random.Random(int(_hash([seed, candidate_index, SCHEMA]), 16))
    actual_count = rng.choice((1, 2)) if count is None else count
    ordinary_gaps = actual_count-1-(1 if placement_intent == "same_station_window" else 0)
    if ordinary_gaps*gap_low > high-low+1e-9:
        raise ValueError("Requested minimum gaps exceed the sampled station window")
    catalog_hash = _hash({"static_models": static_catalog.get("models", static_catalog),
                          "vehicle_models": vehicle_models})
    last_rejection = "no attempted placement"
    for attempt in range(1, max_attempts+1):
        try:
            slots = _sample_slots(rng, actual_count, placement_intent, (low, high), (gap_low, gap_high))
            obstacles = [_place_actor(rng, slot, rng.choice(pool), geometry) for slot in slots]
            candidate = {
                "schema": CANDIDATE_SCHEMA, "map": MAP_NAME,
                "seed": seed, "candidate_index": candidate_index, "count": actual_count,
                "obstacles": obstacles, "expert_verified": False, "live_avoidance_verified": False,
                "physical_walls_verified": False, "catalog_sha256": catalog_hash,
                "experimental_placement_domain": DOMAIN,
                "source_geometry": {"ego_link": TUNNEL_LINK, "opposing_link": OPPOSING_LINK,
                                    "sha256": geometry["source_sha256"]},
                "placement": {
                    "sampling_schema": SCHEMA, "station_min_m": low, "station_max_m": high,
                    "sampled_gap_range_m": [gap_low, gap_high], "footprint_domain": DOMAIN,
                    "outer_road_margin_m": .15, "enforce_adjacent_center_gap": True,
                    "center_gap_policy": "independent sampled gaps; all-pair inflated OBBs govern clearance",
                    "requested_pattern": None, "realized_layout_pattern": placement_intent,
                    "placement_intent": placement_intent, "realized_pattern": None,
                    "reference_ego_body": copy.deepcopy(REFERENCE_EGO),
                    "actual_jamming_path_required": True, "static_yaw_offsets_deg": yaw_policy,
                    "verified_geometry_exclusions": excluded, "sampling_attempts": attempt,
                    "road_boundary_source_sha256": boundary.source_sha256,
                    "geometry_basis": "whole native OBB plus 15 cm in mapped two-lane road; physical walls unverified",
                },
            }
            if actual_count == 3:
                candidate["experimental_three_actor_trial"] = True
            checked = validate_candidate(candidate, map_loader, _road_boundary=boundary)
            candidate["placement"]["geometry"] = checked["placement_geometry"]
            candidate["placement"]["layout_signature"] = layout_signature(candidate)
            return candidate
        except ValueError as error:
            last_rejection = str(error)
    raise SamplingExhausted(max_attempts, last_rejection)


def layout_signature(candidate, coordinate_precision=2):
    """Hash model/centre/yaw tuples; ignore seeds, indexes, and pattern labels.

    Rounded centimetre coordinates treat negligible jitter as the same layout.
    Actor order does not affect duplicate detection. This also accepts legacy
    candidate documents for comparison with historical placements.
    """
    _integer(coordinate_precision, "coordinate_precision", True)
    if coordinate_precision > 8:
        raise ValueError("coordinate_precision must be in 0..8")
    rows = []
    for actor in candidate["obstacles"]:
        yaw = (_finite(actor["yaw_deg"], "signature yaw")+180.) % 360.-180.
        rounded_yaw = round(yaw, coordinate_precision)
        if rounded_yaw == 180.:
            rounded_yaw = -180.
        coordinates = [round(_finite(actor[key], "signature " + key), coordinate_precision)
                       for key in ("station_m", "lateral_m")]
        # JSON distinguishes -0.0 from 0.0, although those are one pose.
        coordinates = [0. if value == 0. else value for value in coordinates]
        rows.append((actor["actor_kind"], actor["model"], *coordinates,
                     0. if rounded_yaw == 0. else rounded_yaw))
    return _hash(sorted(rows))
