"""Explicit, hash-bound instruction topology derived from captured Expert code.

This adds an instruction feature anchor, never rewrites the recorded route,
waypoints, map, sensor metadata, or dataset admission.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import yaml

from .features import RouteProjector


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _constant(source, name):
    assignments = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    seen = set()
    while True:
        if name in seen or name not in assignments:
            raise ValueError("Captured topology constant is missing or circular")
        seen.add(name)
        node = assignments[name]
        if isinstance(node, ast.Name):
            name = node.id
        else:
            return ast.literal_eval(node)


def apply_derived_roundabout_topology(route, config_path):
    """Return a copied route and an auditable receipt; reject source drift.

    The caller must opt in with an explicit configuration path. Exact captured
    route, global path and MGeo bytes are required, as well as the archived
    source of the constants. No code in that archived source is executed.
    """
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "captured_roundabout_topology_derivation_v1":
        raise ValueError("Unsupported derived roundabout topology schema")
    if config.get("route_key") != route.get("route_id"):
        raise ValueError("Derived topology route key differs")
    map_root = Path(route.get("map_root", ""))
    paths = {"route_definition": route["route_definition_source"],
             "global_path": route["global_path_source"],
             "link_set": map_root / "link_set.json",
             "node_set": map_root / "node_set.json"}
    pins = config.get("source_sha256") or {}
    if set(pins) != set(paths):
        raise ValueError("Derived topology source pin set differs")
    for key, path in paths.items():
        if _sha(path) != pins[key]:
            raise ValueError("Derived topology source changed: " + key)
    source = config["parameter_source"]
    if hashlib.sha256(source["text"].encode()).hexdigest() != source["sha256"]:
        raise ValueError("Captured Expert parameter snapshot changed")
    xy = _constant(source["text"], config["merge_point_constant"])
    ring_id = _constant(source["text"], config["ring_link_constant"])
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        raise ValueError("Invalid captured merge point")
    result = deepcopy(route)
    zone = result.get("zones", {}).get("roundabout")
    if not zone or ring_id not in zone.get("internal_loop_links", []):
        raise ValueError("Captured ring link is outside the captured roundabout")
    if zone.get("merge_point_xy") is not None and list(zone["merge_point_xy"]) != list(xy):
        raise ValueError("Captured route already has a conflicting merge anchor")
    path_projection = RouteProjector(result["nav_waypoints"]).project(*xy)
    # Validate Euclidean distance to the actual closest segment, including its
    # endpoints; a signed perpendicular distance alone can hide extrapolation.
    import json
    links = json.loads(Path(paths["link_set"]).read_text(encoding="utf-8-sig"))
    if isinstance(links, dict):
        links = links.get("links", links)
        if isinstance(links, dict):
            links = list(links.values())
    matches = [r for r in links if str(r.get("idx", r.get("id"))) == ring_id]
    if len(matches) != 1:
        raise ValueError("Captured merge ring link cannot be resolved uniquely")
    ring_points = matches[0]["points"]
    ring_projection = RouteProjector(ring_points).project(*xy)
    def distance(points, projection):
        i = projection[2]
        a, b = points[i], points[i + 1]
        vx, vy = b[0] - a[0], b[1] - a[1]
        t = max(0.0, min(1.0, ((xy[0]-a[0])*vx+(xy[1]-a[1])*vy)/(vx*vx+vy*vy)))
        return ((xy[0]-a[0]-t*vx)**2+(xy[1]-a[1]-t*vy)**2)**0.5
    path_distance = distance(result["nav_waypoints"], path_projection)
    ring_distance = distance(ring_points, ring_projection)
    if path_distance > config["max_anchor_path_distance_m"] or ring_distance > config["max_anchor_ring_distance_m"]:
        raise ValueError("Captured merge anchor is outside the verified map/path")
    zone["merge_point_xy"] = list(xy)
    receipt = {"schema": config["schema"], "configuration_sha256": _sha(config_path),
               "source_sha256": pins, "parameter_source": {k: v for k, v in source.items() if k != "text"},
               "merge_point_xy": list(xy), "ring_link_id": ring_id,
               "anchor_path_s_m": path_projection[0], "anchor_path_distance_m": path_distance,
               "anchor_ring_distance_m": ring_distance,
               "capture_route_bytes_modified": False, "nav_waypoints_modified": False,
               "physical_semantic_certified": False}
    result["instruction_topology_derivation"] = receipt
    return result, receipt
