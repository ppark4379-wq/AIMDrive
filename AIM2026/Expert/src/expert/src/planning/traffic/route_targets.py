"""Bind signal permissions to an exact ordered MGeo route and its runtime slice.

The Expert path may begin/end inside a link. A unique contiguous match to the
ordered link points proves which stop-line occurrence it approaches; heading
or an unordered nearest-link match must not choose the intended movement.
"""
import hashlib
import json
import math
import os


SCHEMA = "aimdrive_route_signal_targets_v1"


def signal_permission(status, movement):
    """Canonical protected-movement phases; transitional/unknown masks stay unknown."""
    if movement not in ("straight", "left") or int(status) not in (1, 4, 16, 32, 33, 48):
        return None
    return int(status) in ((16, 48) if movement == "straight" else (32, 33, 48))


def _read_map(path):
    with open(path, "rb") as source:
        raw = source.read()
    document = json.loads(raw.decode("utf-8-sig"))
    return {str(row["idx"]): row for row in document}, hashlib.sha256(raw).hexdigest()


def _xy(point):
    result = (float(point.x), float(point.y)) if hasattr(point, "x") else tuple(map(float, point[:2]))
    if len(result) != 2 or not all(math.isfinite(value) for value in result):
        raise ValueError("Invalid route signal path coordinate")
    return result


def _close(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= 1e-3


def build_route_signal_targets(path, map_dir, route_links, is_closed_path=False):
    """Return audited local stop indices and movements from the *following* link.

    ``route_links`` is the original ordered route, including links beyond an
    active mission window. No route is inferred when the caller omits it.
    """
    route_links = list(route_links)
    if len(route_links) < 2 or any(not isinstance(link, str) or not link for link in route_links):
        raise ValueError("Route signal binding requires explicit ordered link IDs")
    links, link_hash = _read_map(os.path.join(map_dir, "link_set.json"))
    nodes, node_hash = _read_map(os.path.join(map_dir, "node_set.json"))
    runtime = [_xy(point) for point in path]
    if len(runtime) < 2:
        raise ValueError("Route signal binding requires at least two path points")
    if is_closed_path and not _close(runtime[0], runtime[-1]):
        raise ValueError("Closed Expert signal path geometry does not close")
    reference = []
    link_ends = []
    for index, link_id in enumerate(route_links):
        if link_id not in links:
            raise ValueError("Route signal link absent from map: " + link_id)
        link = links[link_id]
        if index and str(links[route_links[index - 1]]["to_node_idx"]) != str(link["from_node_idx"]):
            raise ValueError("Disconnected ordered route signal links")
        points = [_xy(point) for point in link.get("points", [])]
        if not points:
            raise ValueError("Route signal link has no geometry: " + link_id)
        reference.extend(points)
        link_ends.append(len(reference) - 1)
    if is_closed_path and str(links[route_links[-1]]["to_node_idx"]) != str(links[route_links[0]]["from_node_idx"]):
        raise ValueError("Closed route signal links do not close")
    starts = [index for index, point in enumerate(reference)
              if index + len(runtime) <= len(reference) and _close(point, runtime[0])]
    matches = [index for index in starts
               if all(_close(a, b) for a, b in zip(runtime, reference[index:index + len(runtime)]))]
    if len(matches) != 1:
        raise ValueError("Expert path is not a unique exact slice of the ordered MGeo signal route")
    first = matches[0]
    last = first + len(runtime) - 1
    targets = []
    for index, (link_id, end_index) in enumerate(zip(route_links, link_ends)):
        node_id = str(links[link_id]["to_node_idx"])
        if node_id not in nodes:
            raise ValueError("Route signal node absent from map: " + node_id)
        node = nodes[node_id]
        signal_id = node.get("traffic_light_id")
        if not (first <= end_index <= last and node.get("on_stop_line")
                and signal_id is not None and str(signal_id).strip()):
            continue
        if not _close(reference[end_index], _xy(node.get("point", []))):
            raise ValueError("Route signal stop node differs from exact link endpoint")
        following = route_links[index + 1] if index + 1 < len(route_links) else (
            route_links[0] if is_closed_path else None)
        related = str(links[following].get("related_signal") or "").lower() if following else ""
        movement = related if related in ("straight", "left") else None
        targets.append({"signal_id": str(signal_id), "node_id": node_id,
                        "path_index": end_index - first, "route_link_index": index,
                        "route_link_id": link_id, "signal_route_link_id": following,
                        "signal_movement": movement, "related_signal_raw": related or None})
    return {"schema": SCHEMA, "targets": targets, "route_links": route_links,
            "route_slice_start_index": first, "path_points": len(runtime),
            "source_sha256": {"link_set": link_hash, "node_set": node_hash}}
