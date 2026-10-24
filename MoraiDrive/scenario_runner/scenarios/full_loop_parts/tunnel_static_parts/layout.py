"""Deterministic tunnel candidates using measured model footprints and nominal MGeo.

Station/lateral coordinates describe the box centre, not the object's spawn
pivot. Model offsets use the legacy catalog's spawn frame: x forward, y left.
These candidates are preparation artifacts, never validated Expert scenarios.
"""

import bisect
import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping


SCHEMA = "aimdrive_tunnel_static_candidate_v1"
MAP_NAME = "R_KR_PR_K-city_2025"
TUNNEL_LINK = "A2256W000126"
OPPOSING_LINK = "A2256W000131"


def _finite(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(name + " must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(name + " must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, int) or (nonnegative and value < 0):
        raise ValueError(name + " must be an integer" + (" >= 0" if nonnegative else ""))
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _points(values):
    points = []
    for row in values:
        if len(row) < 3:
            raise ValueError("Tunnel points must include x, y, z")
        point = tuple(_finite(value, "point") for value in row[:3])
        if not points or math.dist(point[:2], points[-1][:2]) > 1e-9:
            points.append(point)
    if len(points) < 2:
        raise ValueError("Tunnel link needs at least two distinct points")
    return points


def polyline_stations(points):
    result = [0.0]
    for first, last in zip(points, points[1:]):
        result.append(result[-1] + math.dist(first[:2], last[:2]))
    return result


def route_pose(points, station_m):
    stations = polyline_stations(points)
    station = _finite(station_m, "station_m")
    if not 0 <= station <= stations[-1]:
        raise ValueError("station_m lies outside tunnel link")
    index = max(0, min(len(points)-2, bisect.bisect_right(stations, station)-1))
    first, last = points[index:index+2]
    span = stations[index+1]-stations[index]
    ratio = (station-stations[index])/span
    position = tuple(a+(b-a)*ratio for a, b in zip(first, last))
    yaw = math.atan2(last[1]-first[1], last[0]-first[0])
    return position, yaw


def project_polyline(points, x, y):
    """Return nearest (station, signed lateral left, distance) on a polyline."""
    x, y = _finite(x, "x"), _finite(y, "y")
    best = (float("inf"), 0.0, 0.0)
    progress = 0.0
    for first, last in zip(points, points[1:]):
        dx, dy = last[0]-first[0], last[1]-first[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        fraction = min(1., max(0., ((x-first[0])*dx+(y-first[1])*dy)/(length*length)))
        rx, ry = x-first[0]-fraction*dx, y-first[1]-fraction*dy
        distance = math.hypot(rx, ry)
        if distance < best[0]:
            best = distance, progress+fraction*length, (-dy*rx+dx*ry)/length
        progress += length
    return best[1], best[2], best[0]


def tunnel_geometry(map_loader):
    raw = {identity: map_loader.get_link(identity) for identity in (TUNNEL_LINK, OPPOSING_LINK)}
    lanes, widths = [], []
    source = {}
    for identity, link in raw.items():
        if str(link.get("road_type")) != "2":
            raise ValueError("Expected mapped tunnel road_type=2: " + identity)
        points = _points(link.get("points", []))
        width = min(_finite(link.get(field), field, True) for field in ("width_start", "width_end"))
        # The shared nominal road contract pairs two mapped 3.5 m lanes.
        if abs(width-3.5) > 1e-6:
            raise ValueError("Tunnel screening requires nominal 3.5 m lanes")
        source[identity] = {"points": points, "width_start": link["width_start"],
                            "width_end": link["width_end"], "road_type": link["road_type"]}
        lanes.append(points)
        widths.append(width)
    own, opposing = lanes[0], list(reversed(lanes[1]))
    for first, last in ((own[0], opposing[0]), (own[-1], opposing[-1])):
        if not 2.5 <= math.dist(first[:2], last[:2]) <= 4.5:
            raise ValueError("Opposing tunnel link does not match the Ego lane")
    return {"own_points": own, "opposing_points": opposing,
            "own_lane_width_m": widths[0], "opposing_lane_width_m": widths[1],
            "length_m": polyline_stations(own)[-1], "source_sha256": _hash(source)}


def normalize_footprint(value):
    if not isinstance(value, Mapping):
        raise ValueError("Model footprint must be a mapping")
    offset = value.get("center_offset_spawn_xy", [0.0, 0.0])
    if len(offset) != 2:
        raise ValueError("Footprint spawn offset must have two values")
    return {"width_m": _finite(value.get("width_m"), "width_m", True),
            "length_m": _finite(value.get("length_m"), "length_m", True),
            "center_offset_spawn_xy": [_finite(item, "spawn offset") for item in offset],
            "heading_offset_deg": _finite(value.get("heading_offset_deg", 0.), "heading_offset_deg")}


def bbox_world(obstacle):
    footprint = normalize_footprint(obstacle["planned_bbox"])
    yaw = math.radians(_finite(obstacle["yaw_deg"], "yaw_deg"))
    x, y, z = (_finite(obstacle["position"][axis], "position " + axis) for axis in ("x", "y", "z"))
    cx, cy = footprint["center_offset_spawn_xy"]
    return {"center_xyz": [x+cx*math.cos(yaw)-cy*math.sin(yaw),
                           y+cx*math.sin(yaw)+cy*math.cos(yaw), z],
            "heading_deg": math.degrees(yaw)+footprint["heading_offset_deg"],
            "width_m": footprint["width_m"], "length_m": footprint["length_m"]}


def footprint_corners(center_xy, width_m, length_m, heading_deg):
    angle = math.radians(_finite(heading_deg, "heading_deg"))
    width, length = _finite(width_m, "width_m", True), _finite(length_m, "length_m", True)
    x, y = (_finite(item, "box centre") for item in center_xy[:2])
    forward, left = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
    return [(x+a*length*.5*forward[0]+b*width*.5*left[0],
             y+a*length*.5*forward[1]+b*width*.5*left[1]) for a, b in ((-1,-1),(-1,1),(1,1),(1,-1))]


def boxes_overlap(first, second, margin_m=0.):
    boxes = [footprint_corners(item["center_xyz"], item["width_m"]+2*margin_m,
                              item["length_m"]+2*margin_m, item["heading_deg"]) for item in (first, second)]
    for box in boxes:
        for a, b in ((box[0],box[1]), (box[1],box[2])):
            axis = (-(b[1]-a[1]), b[0]-a[0])
            bounds = [(min(x*axis[0]+y*axis[1] for x,y in corners),
                       max(x*axis[0]+y*axis[1] for x,y in corners)) for corners in boxes]
            if bounds[0][1] < bounds[1][0] or bounds[1][1] < bounds[0][0]:
                return False
    return True


def normalized_geometry(candidate, measured_geometry=None):
    if measured_geometry is None:
        return [bbox_world(item) for item in candidate["obstacles"]]
    if len(measured_geometry) != len(candidate["obstacles"]):
        raise ValueError("Measured geometry count differs from candidate")
    result = []
    for obstacle,item in zip(candidate["obstacles"],measured_geometry):
        center = item["center_xyz"]
        if len(center) != 3:
            raise ValueError("Measured box centre requires x, y, z")
        observed = {"center_xyz": [_finite(v,"measured centre") for v in center],
                    "heading_deg": _finite(item["heading_deg"],"measured heading"),
                    "width_m": _finite(item["width_m"],"measured width",True),
                    "length_m": _finite(item["length_m"],"measured length",True)}
        result.append(conservative_box_union(bbox_world(obstacle),observed))
    return result


def conservative_box_union(planned, observed):
    """Contain approved visible geometry and actual collider in planned axes."""
    angle = math.radians(planned["heading_deg"])
    forward,left = (math.cos(angle),math.sin(angle)),(-math.sin(angle),math.cos(angle))
    corners = [point for item in (planned,observed) for point in footprint_corners(
        item["center_xyz"],item["width_m"],item["length_m"],item["heading_deg"])]
    along = [x*forward[0]+y*forward[1] for x,y in corners]
    across = [x*left[0]+y*left[1] for x,y in corners]
    c_along,c_across = .5*(max(along)+min(along)),.5*(max(across)+min(across))
    return {"center_xyz":[c_along*forward[0]+c_across*left[0],
                           c_along*forward[1]+c_across*left[1],observed["center_xyz"][2]],
            "heading_deg":planned["heading_deg"],"width_m":max(across)-min(across),
            "length_m":max(along)-min(along)}


def placement_geometry(candidate, geometry):
    """Describe nominal spacing from complete footprint corners, not pivots.

The report does not establish a feasible Expert maneuver. Station extents are
corner projections on the mapped Ego lane, so curved-road measurements retain
the same frame as candidate placement and the portal length budget.
"""
    ordered = sorted(enumerate(candidate["obstacles"]), key=lambda row: float(row[1]["station_m"]))
    actors = []
    for index, obstacle in ordered:
        box = bbox_world(obstacle)
        corners = footprint_corners(box["center_xyz"], box["width_m"], box["length_m"], box["heading_deg"])
        stations = [project_polyline(geometry["own_points"], *point)[0] for point in corners]
        actors.append({"index": index, "actor_kind": obstacle["actor_kind"], "model": obstacle["model"],
                       "center_station_m": float(obstacle["station_m"]),
                       "length_m": box["length_m"], "width_m": box["width_m"],
                       "corner_station_min_m": min(stations), "corner_station_max_m": max(stations),
                       "corners_world_xy": [list(point) for point in corners]})
    adjacent = [{"first_index": first["index"], "second_index": last["index"],
                 "center_gap_m": last["center_station_m"] - first["center_station_m"],
                 "longitudinal_surface_gap_m": last["corner_station_min_m"] - first["corner_station_max_m"]}
                for first, last in zip(actors, actors[1:])]
    placement = candidate.get("placement", {})
    low = _finite(placement.get("station_min_m", actors[0]["center_station_m"]), "station_min_m")
    high = _finite(placement.get("station_max_m", actors[-1]["center_station_m"]), "station_max_m")
    gap = (_finite(placement["min_pair_gap_m"], "min_pair_gap_m", True)
           if "min_pair_gap_m" in placement else None)
    side_by_side = (candidate.get("experimental_placement_domain") == "two_lane_physical_road"
                    and placement.get("enforce_adjacent_center_gap") is False)
    required = (len(actors)-1)*gap if gap is not None and not side_by_side else None
    first_corner = min(actor["corner_station_min_m"] for actor in actors)
    last_corner = max(actor["corner_station_max_m"] for actor in actors)
    return {"schema": "aimdrive_tunnel_placement_geometry_v1",
            "basis": "nominal MGeo Ego-lane stations and complete planned OBB corners",
            "actor_count": len(actors), "actors_in_station_order": actors, "adjacent_pairs": adjacent,
            "minimum_center_gap_m": min((row["center_gap_m"] for row in adjacent), default=None),
            "minimum_longitudinal_surface_gap_m": min((row["longitudinal_surface_gap_m"] for row in adjacent), default=None),
            "length_budget": {"tunnel_length_m": geometry["length_m"],
                "center_window_min_m": low, "center_window_max_m": high, "center_window_span_m": high-low,
                "required_center_span_m": required,
                "center_window_slack_m": high-low-required if required is not None else None,
                "sum_actor_length_m": sum(actor["length_m"] for actor in actors),
                "sum_projected_actor_span_m": sum(actor["corner_station_max_m"]-actor["corner_station_min_m"] for actor in actors),
                "occupied_longitudinal_span_m": last_corner-first_corner,
                "before_first_corner_m": first_corner,
                "after_last_corner_m": geometry["length_m"]-last_corner},
            "all_pair_obb_check_count": len(actors)*(len(actors)-1)//2,
            "obb_margin_each_actor_m": 0.5, "expert_path_verified": False,
            "physical_walls_verified": False}


def expanded_road_boundary(map_loader):
    """Load the exact production road polygon only for an explicit experiment."""
    links = getattr(map_loader, "link_set", getattr(map_loader, "links", None))
    if not isinstance(links, (dict, list)):
        raise ValueError("Expanded placement requires the connected MGeo link set")
    try:
        import sys
        module = sys.modules.get("AIM2026.Expert.src.planning.jamming.road_boundary")
        if module is None:
            from .screening import _expert_search
            _, boundary_type, _, _, _ = _expert_search()
        else:
            boundary_type = module.RoadBoundary
        return boundary_type.from_links(links, planning_margin_m=.35, runtime_margin_m=.15)
    except (ImportError, ModuleNotFoundError) as error:
        raise ValueError("Expanded placement road dependencies are unavailable: " + str(error)) from error


def projected_footprint_dimensions(profile, yaw_offset_deg=0.):
    """Return road-normal width and road-longitudinal span without resizing."""
    footprint = normalize_footprint(profile)
    angle = math.radians(_finite(yaw_offset_deg, "yaw offset"))
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    return {"width_m": footprint["width_m"]*cosine+footprint["length_m"]*sine,
            "length_m": footprint["length_m"]*cosine+footprint["width_m"]*sine}


def expanded_region_width_limit(region_name, lateral_limit_m=.25):
    """Necessary straight-road width limits; full mapped polygons still decide.

    Campaigns use these limits to avoid assigning a normal car to an outer
    wall window slot. Passing them is not an actual road/path guarantee.
    """
    from .advanced_patterns import REGIONS
    if region_name not in REGIONS:
        raise ValueError("Unknown expanded placement region")
    jitter = _finite(lateral_limit_m, "lateral_limit_m")
    if not 0. <= jitter <= .25:
        raise ValueError("Expanded region jitter is limited to 25 cm")
    region = REGIONS[region_name]
    centre = region["center_lateral_m"]
    reach = min(jitter, region["jitter_m"])
    best = min(max(1.75, centre-reach), centre+reach)
    capacity = lambda lateral: max(0., 2.*min(lateral+1.75-.15, 5.25-lateral-.15))
    return {"region": region_name, "nominal_center_lateral_m": centre,
            "nominal_projected_width_limit_m": capacity(centre),
            "best_case_projected_width_limit_m": capacity(best),
            "full_polygon_validation_required": True}


def _expanded_pose_check(obstacle, yaw, *, continuous_sampling=False):
    from .advanced_patterns import REGIONS
    region = obstacle.get("placement_region")
    lane = obstacle.get("placement_lane_id")
    if continuous_sampling and region == "continuous_two_lane":
        # This opt-in region covers the two nominal lanes continuously.
        # The complete inflated polygon below still decides whether it fits.
        definition = {"center_lateral_m": 1.75, "jitter_m": 3.35, "lane": "nearest"}
    elif region in REGIONS:
        definition = REGIONS[region]
    else:
        raise ValueError("Expanded actor requires an explicit native lane and placement region")
    if lane not in (TUNNEL_LINK, OPPOSING_LINK):
        raise ValueError("Expanded actor requires an explicit native lane and placement region")
    if abs(float(obstacle["lateral_m"])-definition["center_lateral_m"]) > definition["jitter_m"]+1e-9:
        raise ValueError("Expanded actor lateral coordinate is outside its declared region")
    expected_lane = (OPPOSING_LINK if definition["lane"] == "opposing" else
                     TUNNEL_LINK if definition["lane"] == "own" else
                     TUNNEL_LINK if float(obstacle["lateral_m"]) <= 1.75 else OPPOSING_LINK)
    if lane != expected_lane:
        raise ValueError("Expanded actor lane identity disagrees with its placement region")
    expected_direction = "opposing_forward" if lane == OPPOSING_LINK else "own_forward"
    if obstacle.get("traffic_direction") != expected_direction:
        raise ValueError("Expanded actor traffic direction disagrees with its native lane")
    native = math.degrees(yaw) + (180. if lane == OPPOSING_LINK else 0.)
    declared = _finite(obstacle.get("native_forward_yaw_deg"), "native forward heading")
    difference = lambda first, last: abs((first-last+180.) % 360.-180.)
    if difference(native, declared) > 1e-6:
        raise ValueError("Expanded actor native traffic direction disagrees with its lane")
    offset = _finite(obstacle.get("static_yaw_offset_deg", 0.), "static yaw offset")
    if obstacle["actor_kind"] == "vehicle" and abs(offset) > 1e-9:
        raise ValueError("Stationary vehicles must face their native lane traffic direction")
    footprint = normalize_footprint(obstacle["planned_bbox"])
    expected = native+offset-footprint["heading_offset_deg"]
    if difference(float(obstacle["yaw_deg"]), expected) > 1e-6:
        raise ValueError("Expanded actor yaw disagrees with its native lane/profile rotation")


def validate_candidate(candidate, map_loader, *, _road_boundary=None):
    if candidate.get("schema") != SCHEMA or candidate.get("map") != MAP_NAME:
        raise ValueError("Unsupported tunnel candidate schema or map")
    _integer(candidate.get("seed"), "seed")
    _integer(candidate.get("candidate_index"), "candidate_index", True)
    count = _integer(candidate.get("count"), "count")
    obstacles = candidate.get("obstacles", [])
    if count not in (1, 2, 3) or len(obstacles) != count:
        raise ValueError("Tunnel candidate supports one or two actors, or an explicitly marked three-actor trial")
    if count == 3 and candidate.get("experimental_three_actor_trial") is not True:
        raise ValueError("Three actors require experimental_three_actor_trial=True")
    if candidate.get("expert_verified") is not False or candidate.get("live_avoidance_verified") is not False:
        raise ValueError("Tunnel candidates cannot claim Expert/live avoidance validation")
    geometry = tunnel_geometry(map_loader)
    if candidate.get("source_geometry", {}).get("sha256") != geometry["source_sha256"]:
        raise ValueError("Candidate tunnel source geometry differs from current map")
    domain = candidate.get("experimental_placement_domain")
    if domain not in (None, "two_lane_physical_road"):
        raise ValueError("Unsupported experimental placement domain")
    if "sampling_schema" in candidate.get("placement", {}) and domain != "two_lane_physical_road":
        raise ValueError("Continuous sampling requires the explicit two-lane physical-road domain")
    boundary = (_road_boundary if _road_boundary is not None else expanded_road_boundary(map_loader)) if domain else None
    continuous_sampling = False
    if domain:
        from .advanced_patterns import PATTERNS, yaw_options
        placement = candidate.get("placement", {})
        if "sampling_schema" in placement:
            if placement["sampling_schema"] != "aimdrive_tunnel_static_sampling_v1":
                raise ValueError("Unsupported continuous tunnel sampling schema")
            continuous_sampling = True
            if placement.get("requested_pattern") is not None:
                raise ValueError("Continuous sampling cannot claim a fixed placement template")
        if (placement.get("footprint_domain") != domain
                or placement.get("outer_road_margin_m") != .15):
            raise ValueError("Expanded placement requires the fixed physical-road footprint contract")
        if placement.get("road_boundary_source_sha256") != boundary.source_sha256:
            raise ValueError("Expanded placement road geometry differs from the current common boundary")
        name = placement.get("requested_pattern")
        if name is not None and name not in PATTERNS:
            raise ValueError("Unknown expanded placement pattern")
        if name is not None and count not in PATTERNS[name]["counts"]:
            raise ValueError("Expanded actor count differs from its declared pattern")
        expected_gap_policy = PATTERNS.get(name, {}).get("enforce_adjacent_center_gap", True)
        if placement.get("enforce_adjacent_center_gap") is not expected_gap_policy:
            raise ValueError("Expanded gap policy differs from its declared pattern")
        if name == "independent_detours" and float(placement.get("min_pair_gap_m", 0.)) < 50.:
            raise ValueError("Independent detours require the full-body 50 m center-spacing template")
    boxes = []
    for obstacle in obstacles:
        if obstacle.get("actor_kind") not in ("obstacle", "vehicle"):
            raise ValueError("Invalid candidate actor kind")
        if not isinstance(obstacle.get("model"), str) or not obstacle["model"]:
            raise ValueError("Candidate model name is missing")
        if obstacle.get("scale") != [1., 1., 1.]:
            raise ValueError("Tunnel candidates require measured scale [1,1,1]")
        station = _finite(obstacle["station_m"], "station_m")
        lateral = _finite(obstacle["lateral_m"], "lateral_m")
        position, yaw = route_pose(geometry["own_points"], station)
        if boundary is not None:
            _expanded_pose_check(obstacle, yaw, continuous_sampling=continuous_sampling)
            if continuous_sampling:
                if obstacle.get("placement_region") != "continuous_two_lane":
                    raise ValueError("Continuous sampled actors require the continuous two-lane region")
                low = _finite(placement.get("station_min_m"), "sampled station minimum")
                high = _finite(placement.get("station_max_m"), "sampled station maximum")
                if not .01 < low <= station <= high < geometry["length_m"]-.01:
                    raise ValueError("Sampled actor lies outside its declared tunnel station window")
                if obstacle["actor_kind"] == "obstacle":
                    permitted = {"Steel_Barricade": (90.,), "concrete-barrier": (90.,),
                                 "Rockslide": ()}.get(obstacle["model"], (0.,))
                    offset = _finite(obstacle.get("static_yaw_offset_deg", 0.), "sampled static yaw")
                    if not any(abs((offset-angle+180.) % 360.-180.) <= 1e-6 for angle in permitted):
                        raise ValueError("Sampled static rotation is outside the native model yaw policy")
            if obstacle["actor_kind"] == "obstacle":
                allowed = yaw_options(candidate["placement"].get("static_yaw_offsets_deg"), obstacle["model"])
                offset = float(obstacle.get("static_yaw_offset_deg", 0.))
                if not any(abs((offset-angle+180.) % 360.-180.) <= 1e-6 for angle in allowed):
                    raise ValueError("Expanded static rotation is outside its explicit per-model yaw policy")
        expected = (position[0]-math.sin(yaw)*lateral, position[1]+math.cos(yaw)*lateral)
        box = bbox_world(obstacle)
        if math.dist(expected, box["center_xyz"][:2]) > .01:
            raise ValueError("Candidate station/lateral disagrees with footprint centre")
        for corner in footprint_corners(box["center_xyz"],box["width_m"],box["length_m"],box["heading_deg"]):
            s, lat, distance = project_polyline(geometry["own_points"],*corner)
            if (not .01 < s < geometry["length_m"]-.01
                    or boundary is None and (abs(lat) > geometry["own_lane_width_m"]*.5-.15
                                              or distance > geometry["own_lane_width_m"]*.5-.15)):
                raise ValueError("Candidate whole footprint does not fit nominal Ego lane")
        if boundary is not None:
            from shapely.geometry import Polygon
            inflated = footprint_corners(box["center_xyz"], box["width_m"]+.30,
                                         box["length_m"]+.30, box["heading_deg"])
            portal = boundary.to_local(inflated)[:, 1]
            terminal = float(boundary.to_local([geometry["own_points"][-1][:2]])[0, 1])
            if (not all(.01 < float(s) < terminal-.01 for s in portal)
                    or not boundary.covers_geometry(Polygon(inflated))):
                raise ValueError("Expanded full OBB plus 15 cm leaves mapped tunnel road or portals")
        boxes.append(box)
    for first in range(len(boxes)):
        for last in range(first+1, len(boxes)):
            if boxes_overlap(boxes[first], boxes[last], margin_m=.5):
                raise ValueError("Candidate actor boxes overlap: {} and {}".format(first, last))
    spacing = placement_geometry(candidate, geometry)
    placement = candidate.get("placement", {})
    minimum_gap = placement.get("min_pair_gap_m")
    enforce_gap = not domain or placement.get("enforce_adjacent_center_gap") is not False
    if enforce_gap and minimum_gap is not None and any(row["center_gap_m"] < float(minimum_gap)-1e-9
                                       for row in spacing["adjacent_pairs"]):
        raise ValueError("Candidate center gap is below the placement policy")
    if "geometry" in placement and placement["geometry"] != spacing:
        raise ValueError("Candidate placement geometry metadata disagrees with its actor corners")
    geometry["placement_geometry"] = spacing
    if boundary is not None:
        geometry.update(placement_domain=domain, road_boundary=boundary,
                        road_boundary_source_sha256=boundary.source_sha256)
    return geometry


def generate_candidate(map_loader, static_catalog, vehicle_models, seed, *, count=None,
                       candidate_index=0, actor_kind="mixed", station_min_m=20., station_max_m=75.,
                       min_pair_gap_m=18., lateral_limit_m=.25, max_width_m=2.8, allow_three=False,
                       pattern=None, footprint_domain="own_lane", static_yaw_offsets_deg=None,
                       station_window_m=None, lateral_regions=None):
    """Generate one reproducible nominal-lane candidate, with no simulator writes.

``vehicle_models`` maps model names to GetVehicleSpec/observed footprints and
must set ``geometry_verified: True`` for each eligible actual vehicle.
"""
    if (pattern is not None or footprint_domain != "own_lane" or static_yaw_offsets_deg is not None
            or station_window_m is not None or lateral_regions is not None):
        return _generate_expanded_candidate(map_loader, static_catalog, vehicle_models, seed,
            count=count, candidate_index=candidate_index, actor_kind=actor_kind,
            station_min_m=station_min_m, station_max_m=station_max_m,
            min_pair_gap_m=min_pair_gap_m, lateral_limit_m=lateral_limit_m, max_width_m=max_width_m,
            allow_three=allow_three, pattern=pattern, footprint_domain=footprint_domain,
            static_yaw_offsets_deg=static_yaw_offsets_deg, station_window_m=station_window_m,
            lateral_regions=lateral_regions)
    _integer(seed,"seed")
    _integer(candidate_index,"candidate_index",True)
    if actor_kind not in ("mixed","obstacle","vehicle"):
        raise ValueError("actor_kind must be mixed, obstacle, or vehicle")
    if not isinstance(allow_three, bool):
        raise ValueError("allow_three must be a boolean")
    allowed_counts = (1, 2, 3) if allow_three else (1, 2)
    if count is not None and _integer(count,"count") not in allowed_counts:
        raise ValueError("count must be one or two; count three requires allow_three=True")
    geometry = tunnel_geometry(map_loader)
    low = _finite(station_min_m,"station_min_m")
    high = min(_finite(station_max_m,"station_max_m"),geometry["length_m"]-20.)
    gap = _finite(min_pair_gap_m,"min_pair_gap_m",True)
    lateral_limit = _finite(lateral_limit_m,"lateral_limit_m")
    max_width = _finite(max_width_m,"max_width_m",True)
    if low < 20. or high <= low or lateral_limit < 0 or lateral_limit > .25:
        raise ValueError("Invalid tunnel station/lateral placement range")
    static_models = static_catalog.get("models",static_catalog)
    pool = []
    for kind, catalog in (("obstacle",static_models),("vehicle",vehicle_models)):
        if actor_kind != "mixed" and kind != actor_kind:
            continue
        if not isinstance(catalog, Mapping):
            raise ValueError(kind + " catalog must map names to footprints")
        for model in sorted(catalog):
            raw = catalog[model]
            if kind == "vehicle" and raw.get("geometry_verified") is not True:
                continue
            footprint = normalize_footprint(raw)
            if footprint["width_m"] <= max_width:
                pool.append((kind,model,footprint,raw))
    if not pool:
        raise ValueError("No eligible measured models for tunnel placement")
    rng = random.Random(int(_hash([seed,candidate_index]),16))
    actual_count = rng.choice((1,2)) if count is None else count
    required_span = (actual_count-1)*gap
    if high-low < required_span:
        raise ValueError("Tunnel placement range cannot satisfy {} center gaps".format(actual_count-1))
    if actual_count == 1:
        stations = [rng.uniform(low,high)]
    else:
        # Order N samples in the residual window, then add i*gap. This keeps
        # every adjacent gap >= policy while retaining the existing two-actor
        # random draw sequence and never silently enabling a third actor.
        sampled = sorted(rng.uniform(low, high-required_span) for _ in range(actual_count))
        stations = [station+index*gap for index, station in enumerate(sampled)]
    catalog_hash = _hash({"static_models": static_models,"vehicle_models": vehicle_models})
    for attempt in range(100):
        obstacles = []
        for index, station in enumerate(stations):
            kind,model,footprint,raw = rng.choice(pool)
            lateral = rng.uniform(-lateral_limit,lateral_limit)
            position,yaw = route_pose(geometry["own_points"],station)
            # Compensate measured heading offset so every full box follows
            # the lane tangent; compensate the pivot in the spawn frame.
            spawn_yaw = yaw-math.radians(footprint["heading_offset_deg"])
            cx,cy = footprint["center_offset_spawn_xy"]
            x = position[0]-math.sin(yaw)*lateral-cx*math.cos(spawn_yaw)+cy*math.sin(spawn_yaw)
            y = position[1]+math.cos(yaw)*lateral-cx*math.sin(spawn_yaw)-cy*math.cos(spawn_yaw)
            obstacle = {"actor_kind":kind,"model":model,"scale":[1.,1.,1.],
                              "station_m":station,"lateral_m":lateral,"yaw_deg":math.degrees(spawn_yaw),
                              "position":{"x":x,"y":y,"z":position[2]},"planned_bbox":copy.deepcopy(footprint)}
            for metadata in ("geometry_source","expected_api_raw"):
                if metadata in raw:
                    obstacle[metadata] = copy.deepcopy(raw[metadata])
            obstacles.append(obstacle)
        candidate = {"schema":SCHEMA,"map":MAP_NAME,"seed":seed,"candidate_index":candidate_index,
                     "count":actual_count,"obstacles":obstacles,"expert_verified":False,
                     "live_avoidance_verified":False,"physical_walls_verified":False,
                     "source_geometry":{"ego_link":TUNNEL_LINK,"opposing_link":OPPOSING_LINK,
                                        "sha256":geometry["source_sha256"]},
                     "catalog_sha256":catalog_hash,"placement":{"station_min_m":low,
                         "station_max_m":high,"min_pair_gap_m":gap,"lateral_limit_m":lateral_limit,
                         "geometry_basis":"nominal MGeo lanes; physical walls unverified"}}
        if actual_count == 3:
            candidate["experimental_three_actor_trial"] = True
        try:
            verified_geometry = validate_candidate(candidate,map_loader)
            candidate["placement"]["geometry"] = verified_geometry["placement_geometry"]
            return candidate
        except ValueError:
            if attempt == 99:
                raise ValueError("No whole-footprint placement fits configured nominal lane")
    raise AssertionError("Unreachable candidate generation state")


def _generate_expanded_candidate(map_loader, static_catalog, vehicle_models, seed, *, count,
                                candidate_index, actor_kind, station_min_m, station_max_m,
                                min_pair_gap_m, lateral_limit_m, max_width_m, allow_three,
                                pattern, footprint_domain, static_yaw_offsets_deg,
                                station_window_m, lateral_regions):
    from .advanced_patterns import (DOMAIN, REGIONS, REFERENCE_EGO, SCHEMA as PATTERN_SCHEMA,
                                    resolve_slots, yaw_options)
    if footprint_domain != DOMAIN:
        raise ValueError("Expanded patterns require explicit footprint_domain='two_lane_physical_road'")
    _integer(seed, "seed")
    _integer(candidate_index, "candidate_index", True)
    if not isinstance(allow_three, bool):
        raise ValueError("allow_three must be a boolean")
    if count is not None and _integer(count, "count") not in ((1, 2, 3) if allow_three else (1, 2)):
        raise ValueError("Expanded count three requires explicit allow_three=True")
    if pattern == "alternating_three" and count is None:
        raise ValueError("Alternating three requires explicit count=3 and allow_three=True")
    if actor_kind not in ("mixed", "obstacle", "vehicle"):
        raise ValueError("actor_kind must be mixed, obstacle, or vehicle")
    if static_yaw_offsets_deg is not None and not isinstance(static_yaw_offsets_deg, Mapping):
        raise ValueError("Static yaw offsets must map model names to degrees")
    geometry = tunnel_geometry(map_loader)
    boundary = expanded_road_boundary(map_loader)
    if station_window_m is None:
        station_window_m = (station_min_m, station_max_m)
    if not isinstance(station_window_m, (list, tuple)) or len(station_window_m) != 2:
        raise ValueError("Station window must contain its minimum and maximum metres")
    low, high = [_finite(value, "expanded station window") for value in station_window_m]
    if low < .01 or high < low or high >= geometry["length_m"]-.01:
        raise ValueError("Expanded station window is outside mapped tunnel portals")
    gap = _finite(min_pair_gap_m, "min_pair_gap_m", True)
    jitter = _finite(lateral_limit_m, "lateral_limit_m")
    width_cap = min(_finite(max_width_m, "max_width_m", True),
                    geometry["own_lane_width_m"]+geometry["opposing_lane_width_m"]-.30)
    if not 0. <= jitter <= .25:
        raise ValueError("Expanded region jitter is limited to 25 cm")
    static_models = static_catalog.get("models", static_catalog)
    pools, excluded = [], []
    for kind, models in (("obstacle", static_models), ("vehicle", vehicle_models)):
        if actor_kind != "mixed" and kind != actor_kind:
            continue
        if not isinstance(models, Mapping):
            raise ValueError("Expanded catalog must map names to measured native footprints")
        for name in sorted(models):
            raw = models[name]
            try:
                if (not isinstance(raw, Mapping) or raw.get("geometry_verified") is not True
                        or not isinstance(raw.get("geometry_source"), Mapping) or not raw["geometry_source"]):
                    raise ValueError("verified geometry evidence is missing")
                scale = raw.get("scale_xyz", raw.get("scale", [1., 1., 1.]))
                if (not isinstance(scale, (list, tuple)) or len(scale) != 3
                        or any(isinstance(value, bool) or float(value) != 1. for value in scale)):
                    raise ValueError("measured geometry is not at native unit scale")
                footprint = normalize_footprint(raw)
                offsets = yaw_options(static_yaw_offsets_deg, name) if kind == "obstacle" else [0.]
                eligible = []
                for angle in offsets:
                    if projected_footprint_dimensions(footprint, angle)["width_m"] <= width_cap+1e-9:
                        eligible.append(angle)
                if not eligible:
                    raise ValueError("no explicitly permitted yaw fits projected width policy")
                pools.append((kind, name, footprint, raw, eligible))
            except (KeyError, ValueError, TypeError) as error:
                excluded.append({"actor_kind": kind, "model": name, "reason": str(error)})
    if not pools:
        raise ValueError("No verified native model/yaw combination fits expanded placement")
    rng = random.Random(int(_hash([seed, candidate_index, "expanded", pattern]), 16))
    actual_count, slots, effective_gap, enforce_gap, design = resolve_slots(
        pattern, count, (low, high), rng, min_pair_gap_m=gap, lateral_regions=lateral_regions)
    if actual_count == 3 and (not allow_three or count != 3):
        raise ValueError("Expanded three-actor trials require explicit count=3 and allow_three=True")
    slot_pools = []
    for slot in slots:
        limit = min(width_cap, expanded_region_width_limit(slot["region"], jitter)["best_case_projected_width_limit_m"])
        options = []
        for kind, name, footprint, raw, offsets in pools:
            permitted = [angle for angle in offsets
                         if projected_footprint_dimensions(footprint, angle)["width_m"] <= limit+1e-9]
            if permitted:
                options.append((kind, name, footprint, raw, permitted))
        if not options:
            raise ValueError("No verified native model/yaw fits pattern region: " + slot["region"])
        slot_pools.append(options)
    catalog_hash = _hash({"static_models": static_models, "vehicle_models": vehicle_models})
    for attempt in range(100):
        obstacles = []
        for slot, pool in zip(slots, slot_pools):
            kind, name, footprint, raw, offsets = rng.choice(pool)
            region = REGIONS[slot["region"]]
            lateral = region["center_lateral_m"]+rng.uniform(-min(jitter, region["jitter_m"]),
                                                            min(jitter, region["jitter_m"]))
            lane = (OPPOSING_LINK if region["lane"] == "opposing" else
                    TUNNEL_LINK if region["lane"] == "own" else
                    TUNNEL_LINK if lateral <= 1.75 else OPPOSING_LINK)
            position, yaw = route_pose(geometry["own_points"], slot["station_m"])
            native_heading = math.degrees(yaw)+(180. if lane == OPPOSING_LINK else 0.)
            offset = rng.choice(offsets)
            spawn_yaw = math.radians(native_heading+offset-footprint["heading_offset_deg"])
            cx, cy = footprint["center_offset_spawn_xy"]
            centre = (position[0]-math.sin(yaw)*lateral, position[1]+math.cos(yaw)*lateral)
            row = {"actor_kind": kind, "model": name, "scale": [1., 1., 1.],
                   "station_m": slot["station_m"], "lateral_m": lateral,
                   "yaw_deg": math.degrees(spawn_yaw), "planned_bbox": copy.deepcopy(footprint),
                   "position": {"x": centre[0]-cx*math.cos(spawn_yaw)+cy*math.sin(spawn_yaw),
                                "y": centre[1]-cx*math.sin(spawn_yaw)-cy*math.cos(spawn_yaw), "z": position[2]},
                   "placement_lane_id": lane, "placement_region": slot["region"],
                   "native_forward_yaw_deg": native_heading, "static_yaw_offset_deg": offset,
                   "traffic_direction": "opposing_forward" if lane == OPPOSING_LINK else "own_forward"}
            for key in ("geometry_source", "expected_api_raw"):
                if key in raw:
                    row[key] = copy.deepcopy(raw[key])
            obstacles.append(row)
        candidate = {"schema": SCHEMA,
            "map": MAP_NAME, "seed": seed, "candidate_index": candidate_index, "count": actual_count,
            "obstacles": obstacles, "expert_verified": False, "live_avoidance_verified": False,
            "physical_walls_verified": False, "catalog_sha256": catalog_hash,
            "experimental_placement_domain": DOMAIN,
            "source_geometry": {"ego_link": TUNNEL_LINK, "opposing_link": OPPOSING_LINK,
                                "sha256": geometry["source_sha256"]},
            "placement": {"station_min_m": low, "station_max_m": high,
                "min_pair_gap_m": effective_gap, "lateral_limit_m": jitter,
                "footprint_domain": DOMAIN, "outer_road_margin_m": .15,
                "enforce_adjacent_center_gap": enforce_gap, "requested_pattern": pattern,
                "realized_layout_pattern": pattern or "custom_two_lane", "realized_pattern": None,
                "pattern_design_reference": design, "reference_ego_body": copy.deepcopy(REFERENCE_EGO),
                "actual_jamming_path_required": True,
                "static_yaw_offsets_deg": copy.deepcopy(static_yaw_offsets_deg or {}),
                "verified_geometry_exclusions": excluded,
                "road_boundary_source_sha256": boundary.source_sha256,
                "geometry_basis": "whole native OBB plus 15 cm in mapped two-lane road; physical walls unverified"}}
        # Use the established candidate schema; pattern schema is separate
        # metadata, so ordinary replay and native actors keep their contract.
        candidate["placement"]["pattern_schema"] = PATTERN_SCHEMA
        if actual_count == 3:
            candidate["experimental_three_actor_trial"] = True
        try:
            checked = validate_candidate(candidate, map_loader, _road_boundary=boundary)
            candidate["placement"]["geometry"] = checked["placement_geometry"]
            return candidate
        except ValueError:
            if attempt == 99:
                raise ValueError("No whole native OBB layout fits expanded road, portal and spacing policies")
    raise AssertionError("Unreachable expanded generation state")
