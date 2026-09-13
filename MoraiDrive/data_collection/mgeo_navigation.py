"""Lightweight MGeo link matcher used by collection and route planning."""
import json
import heapq
import bisect
import zlib
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class LinkMatch:
    link_id: str
    distance_m: float
    heading_rad: float
    projected_x: float
    projected_y: float


@dataclass(frozen=True)
class RouteProgress:
    route_index: int
    route_link_index: int
    current_link_id: str
    distance_to_route_m: float
    near_x: float
    near_y: float
    near_route_index: int
    target_x: float
    target_y: float
    target_route_index: int
    command: int
    gt_command: int
    is_junction: bool
    distance_to_stopline_m: Optional[float]
    affected_light_id: Optional[str]
    future_waypoints: List[List[float]]


def load_official_route(route_definition_path: str, global_path_path: str,
                        route_key: str = "full_loop") -> dict:
    """Load the exact route links and dense path used by Scenario Runner.

    Unlike a checkpoint route, this function never performs link matching or a
    shortest-path search.  The two Scenario Runner files are the source of
    truth, so collector labels cannot silently take a different branch.
    """
    with open(route_definition_path, encoding="utf-8-sig") as f:
        definitions = yaml.safe_load(f) or {}
    definition = definitions.get(route_key)
    if not isinstance(definition, dict):
        raise ValueError(
            f"route definition '{route_key}' not found: {route_definition_path}"
        )

    link_ids = [str(link_id) for link_id in definition.get("route_links", [])]
    if len(link_ids) < 2:
        raise ValueError(f"official route needs at least two links: {route_key}")

    waypoints = []
    with open(global_path_path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if len(fields) < 2:
                raise ValueError(
                    f"{global_path_path}:{line_no}: expected x y [z]"
                )
            waypoints.append([
                float(fields[0]), float(fields[1]),
                float(fields[2]) if len(fields) >= 3 else 0.0,
            ])
    if len(waypoints) < 2:
        raise ValueError(f"official global path is empty: {global_path_path}")

    for checkpoint in definition.get("checkpoints", []):
        index = int(checkpoint["path_index"])
        if index < 0 or index >= len(waypoints):
            raise ValueError(
                f"checkpoint {checkpoint.get('id')} has invalid path_index={index}"
            )

    return {
        "route_id": route_key,
        "map_id": definition.get("map_name"),
        "route_source": "scenario_runner_official",
        "route_definition_source": os.path.abspath(route_definition_path),
        "global_path_source": os.path.abspath(global_path_path),
        "nav_link_ids": link_ids,
        "nav_waypoints": waypoints,
        "checkpoints": definition.get("checkpoints", []),
    }


def stable_mgeo_id(value: Optional[str]) -> int:
    """Stable positive int32 ID for LMDrive fields that require actor integers."""
    if not value:
        return -1
    return zlib.crc32(str(value).encode("utf-8")) & 0x7fffffff


@dataclass(frozen=True)
class LocalizationState:
    x: float
    y: float
    heading_rad: float
    speed_mps: float
    heading_source: str


class GNSSMotionEstimator:
    """Derive MGeo pose/course/speed when MORAI Ego status is stale."""
    def __init__(self, min_course_distance_m: float = 0.05,
                 min_course_speed_mps: float = 0.3):
        self.min_course_distance_m = min_course_distance_m
        self.min_course_speed_mps = min_course_speed_mps
        self._last = None
        self._last_motion_heading = None

    def update(self, x: float, y: float, timestamp_ns: int,
               fallback_heading_rad: float, fallback_speed_mps: float) -> LocalizationState:
        heading = self._last_motion_heading
        speed = float(fallback_speed_mps)
        source = "ego_fallback"
        if self._last is not None:
            last_x, last_y, last_timestamp_ns = self._last
            dt = (timestamp_ns - last_timestamp_ns) / 1_000_000_000.0
            dx, dy = x - last_x, y - last_y
            distance = math.hypot(dx, dy)
            if dt > 1e-4:
                speed = distance / dt
                if distance >= self.min_course_distance_m and speed >= self.min_course_speed_mps:
                    heading = math.atan2(dy, dx)
                    self._last_motion_heading = heading
                    source = "gnss_course"
                elif heading is not None:
                    source = "gnss_course_held"
        self._last = (x, y, timestamp_ns)
        if heading is None:
            heading = fallback_heading_rad
        return LocalizationState(x, y, heading, speed, source)


class MGeoMap:
    def __init__(self, link_set_path: str, grid_size_m: float = 20.0,
                 node_set_path: Optional[str] = None):
        with open(link_set_path, encoding="utf-8") as f:
            links = json.load(f)
        self.links = {str(link["idx"]): link for link in links}
        self.nodes = {}
        if node_set_path:
            with open(node_set_path, encoding="utf-8") as f:
                self.nodes = {str(node["idx"]): node for node in json.load(f)}
        self.grid_size_m = grid_size_m
        self.incoming_by_node: Dict[str, List[str]] = {}
        self.outgoing_by_node: Dict[str, List[str]] = {}
        self._segments: List[Tuple[str, float, float, float, float, float]] = []
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        for link in links:
            self.outgoing_by_node.setdefault(str(link.get("from_node_idx")), []).append(
                str(link["idx"])
            )
            self.incoming_by_node.setdefault(str(link.get("to_node_idx")), []).append(
                str(link["idx"])
            )
            for a, b in zip(link.get("points", []), link.get("points", [])[1:]):
                ax, ay = float(a[0]), float(a[1])
                dx, dy = float(b[0]) - ax, float(b[1]) - ay
                length_sq = dx * dx + dy * dy
                if length_sq <= 1e-9:
                    continue
                index = len(self._segments)
                self._segments.append((str(link["idx"]), ax, ay, dx, dy, length_sq))
                min_cx, max_cx = sorted((self._cell(ax), self._cell(ax + dx)))
                min_cy, max_cy = sorted((self._cell(ay), self._cell(ay + dy)))
                for cx in range(min_cx, max_cx + 1):
                    for cy in range(min_cy, max_cy + 1):
                        self._grid.setdefault((cx, cy), []).append(index)

    def _cell(self, value: float) -> int:
        return math.floor(value / self.grid_size_m)

    @staticmethod
    def _angle_error(a: float, b: float) -> float:
        return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)

    def match(self, x: float, y: float, heading_rad: Optional[float] = None,
              heading_weight_m: float = 5.0) -> Optional[LinkMatch]:
        cx, cy = self._cell(x), self._cell(y)
        candidates = set()
        for radius in (1, 2, 4):
            for ix in range(cx - radius, cx + radius + 1):
                for iy in range(cy - radius, cy + radius + 1):
                    candidates.update(self._grid.get((ix, iy), ()))
            if candidates:
                break
        if not candidates:
            candidates = range(len(self._segments))

        best = None
        for index in candidates:
            link_id, ax, ay, dx, dy, length_sq = self._segments[index]
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
            px, py = ax + t * dx, ay + t * dy
            distance_sq = (x - px) ** 2 + (y - py) ** 2
            segment_heading = math.atan2(dy, dx)
            score = math.sqrt(distance_sq)
            if heading_rad is not None:
                score += heading_weight_m * self._angle_error(segment_heading, heading_rad)
                # A checkpoint commonly lies exactly at both an incoming link's
                # end and an outgoing link's start. Its yaw describes departure,
                # so break otherwise identical ties in favor of the outgoing one.
                score += t * 1e-3
            if best is None or score < best[0]:
                best = (score, distance_sq, link_id, segment_heading, px, py)
        if best is None:
            return None
        return LinkMatch(best[2], math.sqrt(best[1]), best[3], best[4], best[5])

    def shortest_link_path(self, start_link_id: str, goal_link_id: str) -> List[str]:
        """Return a directed MGeo link path, including both endpoint links."""
        if start_link_id not in self.links or goal_link_id not in self.links:
            raise ValueError("unknown start or goal link")
        if start_link_id == goal_link_id:
            return [start_link_id]

        outgoing: Dict[str, List[str]] = {}
        for link_id, link in self.links.items():
            outgoing.setdefault(str(link.get("from_node_idx")), []).append(link_id)

        start_node = str(self.links[start_link_id].get("to_node_idx"))
        goal_node = str(self.links[goal_link_id].get("from_node_idx"))
        distances = {start_node: 0.0}
        previous: Dict[str, Tuple[str, str]] = {}
        queue = [(0.0, start_node)]
        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance != distances.get(node_id):
                continue
            if node_id == goal_node:
                break
            for link_id in outgoing.get(node_id, []):
                link = self.links[link_id]
                next_node = str(link.get("to_node_idx"))
                next_distance = distance + float(link.get("link_length") or 1.0)
                if next_distance < distances.get(next_node, math.inf):
                    distances[next_node] = next_distance
                    previous[next_node] = (node_id, link_id)
                    heapq.heappush(queue, (next_distance, next_node))
        if goal_node not in distances:
            raise ValueError(f"no directed path: {start_link_id} -> {goal_link_id}")

        middle = []
        node_id = goal_node
        while node_id != start_node:
            node_id, link_id = previous[node_id]
            middle.append(link_id)
        return [start_link_id] + list(reversed(middle)) + [goal_link_id]

    def build_checkpoint_route(self, checkpoints: List[dict], close_loop: bool = False) -> dict:
        """Map ordered checkpoints to links and connect them on the directed graph."""
        matched = []
        for checkpoint in checkpoints:
            match = self.match(
                float(checkpoint["x"]), float(checkpoint["y"]),
                math.radians(float(checkpoint["yaw_deg"])),
            )
            if match is None:
                raise ValueError(f"checkpoint {checkpoint['name']} did not match a link")
            matched.append(match)

        pairs = list(zip(matched, matched[1:]))
        if close_loop:
            pairs.append((matched[-1], matched[0]))
        route_links: List[str] = []
        for start, goal in pairs:
            segment = self.shortest_link_path(start.link_id, goal.link_id)
            if route_links and segment[0] == route_links[-1]:
                segment = segment[1:]
            route_links.extend(segment)

        route_points = []
        for link_id in route_links:
            points = self.links[link_id].get("points", [])
            if route_points and points and route_points[-1][:2] == points[0][:2]:
                points = points[1:]
            route_points.extend(points)
        return {
            "checkpoint_matches": [
                {
                    "name": checkpoint["name"], "link_id": match.link_id,
                    "distance_m": match.distance_m,
                    "heading_error_deg": math.degrees(self._angle_error(
                        match.heading_rad, math.radians(float(checkpoint["yaw_deg"]))
                    )),
                }
                for checkpoint, match in zip(checkpoints, matched)
            ],
            "nav_link_ids": route_links,
            "nav_node_ids": [
                str(self.links[route_links[0]].get("from_node_idx"))
            ] + [str(self.links[link_id].get("to_node_idx")) for link_id in route_links],
            "nav_waypoints": route_points,
        }


class MGeoRouteTracker:
    """Forward route localization with START re-entry for closed routes."""
    COMMAND_LEFT = 1
    COMMAND_RIGHT = 2
    COMMAND_STRAIGHT = 3
    COMMAND_LANEFOLLOW = 4

    def __init__(self, mgeo_map: MGeoMap, route_metadata: dict,
                 target_distance_m: float = 25.0,
                 near_distance_m: float = 4.0,
                 traffic_light_lookahead_m: float = 45.0,
                 command_lookahead_m: float = 30.0,
                 initial_route_index: Optional[int] = None):
        self.map = mgeo_map
        self.link_ids = list(route_metadata["nav_link_ids"])
        if not self.link_ids:
            raise ValueError("route has no nav_link_ids")
        self.target_distance_m = target_distance_m
        # LMDrive RoutePlanner(4.0, 50)는 지나간 route point 제거 threshold에
        # 가깝다. MORAI route는 0.5 m dense point라서 현재 index 기준 약 4 m
        # 앞 tracking point를 near_node로 쓰는 의도적 compatibility approximation이다.
        self.near_distance_m = near_distance_m
        self.traffic_light_lookahead_m = traffic_light_lookahead_m
        # Kept for callers that still pass the legacy argument. Commands are
        # now pre-annotated per link and no longer searched 30 m ahead.
        self.command_lookahead_m = command_lookahead_m
        missing = [link_id for link_id in self.link_ids if link_id not in mgeo_map.links]
        if missing:
            raise ValueError(f"route contains unknown MGeo links: {missing}")

        self.points: List[Tuple[float, float]] = [
            (float(point[0]), float(point[1]))
            for point in route_metadata.get("nav_waypoints", [])
        ]
        reference_points: List[Tuple[float, float]] = []
        reference_link_indices: List[int] = []
        reference_link_end_distances: List[float] = []
        reference_distance = 0.0
        for link_index, link_id in enumerate(self.link_ids):
            points = mgeo_map.links[link_id].get("points", [])
            for point in points:
                xy = (float(point[0]), float(point[1]))
                if reference_points:
                    reference_distance += math.hypot(
                        xy[0] - reference_points[-1][0],
                        xy[1] - reference_points[-1][1],
                    )
                reference_points.append(xy)
                reference_link_indices.append(link_index)
            reference_link_end_distances.append(reference_distance)

        if not self.points:
            # Checkpoint-route compatibility fallback. Official Scenario Runner
            # routes always provide nav_waypoints and therefore never enter here.
            self.points = reference_points
        if not self.points:
            raise ValueError("route has no points")
        self.cumulative_m = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.cumulative_m.append(
                self.cumulative_m[-1] + math.hypot(b[0] - a[0], b[1] - a[1])
            )

        # The current K-City official global path is the exact concatenation of
        # route link points. Keep that exact per-point association when possible.
        exact_official_alignment = len(self.points) == len(reference_points) and all(
            math.hypot(point[0] - reference[0], point[1] - reference[1]) <= 1e-3
            for point, reference in zip(self.points, reference_points)
        )
        if (route_metadata.get("route_source") == "scenario_runner_official" and
                not exact_official_alignment):
            raise ValueError(
                "official global path does not match the selected MGeo link_set; "
                "use the same map data as Scenario Runner"
            )
        if exact_official_alignment:
            self.point_link_indices = reference_link_indices
        else:
            # Generic precomputed dense routes may resample the same MGeo path.
            # Associate them monotonically by route distance instead of doing a
            # new spatial link search at crossings.
            self.point_link_indices = [
                min(
                    bisect.bisect_left(reference_link_end_distances, distance),
                    len(self.link_ids) - 1,
                )
                for distance in self.cumulative_m
            ]
        self.link_end_point_indices = []
        for link_index in range(len(self.link_ids)):
            indices = [
                index for index, value in enumerate(self.point_link_indices)
                if value == link_index
            ]
            self.link_end_point_indices.append(
                indices[-1] if indices else (
                    self.link_end_point_indices[-1]
                    if self.link_end_point_indices else 0
                )
            )
        self.point_road_options = [
            self._link_command(self.link_ids[link_index])
            for link_index in self.point_link_indices
        ]
        if initial_route_index is not None and initial_route_index < 0:
            raise ValueError("initial_route_index must be >= 0")
        self._route_index: Optional[int] = initial_route_index
        # Only an explicitly closed geometry can wrap; open routes still stop at END.
        self._closed_route = (
            len(self.points) > 2 and self.cumulative_m[-1] > 0.0
            and math.hypot(self.points[0][0] - self.points[-1][0],
                           self.points[0][1] - self.points[-1][1]) <= 1e-3
        )

    def _point_heading(self, index: int) -> float:
        before = self.points[max(0, index - 1)]
        after = self.points[min(len(self.points) - 1, index + 1)]
        return math.atan2(after[1] - before[1], after[0] - before[0])

    def _nearest_index(self, x: float, y: float, heading_rad: float) -> int:
        if self._route_index is None:
            candidates = range(len(self.points))
        else:
            # Never move backward. About 250 m forward is enough to survive
            # sparse collection while avoiding a wrong branch at crossings.
            end_distance = self.cumulative_m[self._route_index] + 250.0
            end = bisect.bisect_right(self.cumulative_m, end_distance)
            candidates = range(self._route_index, max(self._route_index + 1, end))
        def score(i):
            return (math.hypot(self.points[i][0] - x, self.points[i][1] - y)
                    + 2.0 * self.map._angle_error(self._point_heading(i), heading_rad))

        best = min(candidates, key=score)
        if (self._closed_route and self._route_index is not None
                and self.cumulative_m[-1] - self.cumulative_m[self._route_index] <= 25.0):
            # Re-enter only the first 50 m, close to the route and facing forward.
            # A strict score margin keeps a stopped vehicle at the shared seam
            # from oscillating between the first and last point.
            start_end = bisect.bisect_right(self.cumulative_m, 50.0)
            start_best = min(range(min(start_end, len(self.points) - 1)), key=score)
            start_distance = math.hypot(self.points[start_best][0] - x,
                                        self.points[start_best][1] - y)
            if (start_distance <= 3.0
                    and self.map._angle_error(self._point_heading(start_best), heading_rad)
                    <= math.pi / 3.0
                    and score(start_best) + 0.25 < score(best)):
                return start_best
        return best

    def _link_command(self, link_id: str) -> int:
        """Annotate a route link once with LMDrive RoadOption semantics."""
        signal = str(self.map.links[link_id].get("related_signal") or "").lower()
        if "left" in signal:
            return self.COMMAND_LEFT
        if "right" in signal:
            return self.COMMAND_RIGHT
        if signal == "straight":
            return self.COMMAND_STRAIGHT
        return self.COMMAND_LANEFOLLOW

    def get_future_waypoints(self, route_index: int,
                             count: int = 50) -> List[List[float]]:
        """Return the planned dense route ahead, never future Ego samples."""
        if self._closed_route:
            # The final point duplicates START; do not repeat it at every seam.
            indices = [(route_index + offset) % (len(self.points) - 1)
                       for offset in range(count)]
        else:
            indices = range(route_index, min(route_index + count, len(self.points)))
        return [
            [
                float(self.points[index][0]),
                float(self.points[index][1]),
                int(self.point_road_options[index]),
            ]
            for index in indices
        ]

    def _index_at_distance_ahead(self, route_index: int, distance_m: float) -> int:
        target_distance = self.cumulative_m[route_index] + distance_m
        if self._closed_route:
            target_distance %= self.cumulative_m[-1]
        return min(
            bisect.bisect_left(self.cumulative_m, target_distance),
            len(self.points) - 1,
        )

    def _nearest_route_index_from(self, x: float, y: float,
                                  start_index: int, max_ahead_m: float = 80.0) -> int:
        end_distance = self.cumulative_m[start_index] + max_ahead_m
        end = bisect.bisect_right(self.cumulative_m, end_distance)
        candidates = range(start_index, max(start_index + 1, end))
        return min(
            candidates,
            key=lambda i: math.hypot(self.points[i][0] - x, self.points[i][1] - y),
        )

    def hazard_actor_ids(self, actors: List[object], ego_x: float, ego_y: float,
                         ego_heading_rad: float, route_index: int,
                         is_junction: bool) -> Dict[str, List[int]]:
        """Approximate LMDrive hazard lists with MORAI object poses and route topology."""
        result = {
            "vehicle": [],
            "lane_vehicle": [],
            "junction_vehicle": [],
            "pedestrian": [],
            "bike": [],
        }
        ego_forward = (math.cos(ego_heading_rad), math.sin(ego_heading_rad))
        ego_link_index = self.point_link_indices[route_index]
        ego_speed_gate_m = 30.0
        for actor in actors:
            dx = float(actor.x) - ego_x
            dy = float(actor.y) - ego_y
            distance = math.hypot(dx, dy)
            if distance > 50.0:
                continue
            longitudinal = dx * ego_forward[0] + dy * ego_forward[1]
            if longitudinal < -5.0:
                continue

            actor_index = self._nearest_route_index_from(
                float(actor.x), float(actor.y), route_index
            )
            actor_link_index = self.point_link_indices[actor_index]
            actor_route_distance = (
                self.cumulative_m[actor_index] - self.cumulative_m[route_index]
            )
            lateral_error = math.hypot(
                self.points[actor_index][0] - float(actor.x),
                self.points[actor_index][1] - float(actor.y),
            )

            if actor.obj_type == "vehicle":
                same_link = actor_link_index == ego_link_index
                ahead_on_route = 0.0 <= actor_route_distance <= ego_speed_gate_m
                same_lane_like = ahead_on_route and lateral_error <= 3.5
                crossing_in_junction = (
                    is_junction and distance <= 20.0 and longitudinal > -2.0
                    and lateral_error <= 8.0
                )
                if same_lane_like:
                    result["vehicle"].append(int(actor.obj_id))
                    if same_link:
                        result["lane_vehicle"].append(int(actor.obj_id))
                elif crossing_in_junction:
                    result["vehicle"].append(int(actor.obj_id))
                    result["junction_vehicle"].append(int(actor.obj_id))
            elif actor.obj_type == "pedestrian":
                if 0.0 <= longitudinal <= 15.0 and lateral_error <= 6.0:
                    result["pedestrian"].append(int(actor.obj_id))
        return result

    def _is_topology_junction(self, link_index: int) -> bool:
        link = self.map.links[self.link_ids[link_index]]
        from_node = str(link.get("from_node_idx"))
        to_node = str(link.get("to_node_idx"))
        return (
            bool(link.get("related_signal")) or
            len(self.map.incoming_by_node.get(from_node, [])) > 1 or
            len(self.map.outgoing_by_node.get(from_node, [])) > 1 or
            len(self.map.incoming_by_node.get(to_node, [])) > 1 or
            len(self.map.outgoing_by_node.get(to_node, [])) > 1
        )

    def _road_semantics(self, route_index: int):
        current_link_index = self.point_link_indices[route_index]
        is_junction = self._is_topology_junction(current_link_index)
        distance_to_stopline = None
        affected_light_id = None
        for link_index in range(current_link_index, len(self.link_ids)):
            link = self.map.links[self.link_ids[link_index]]
            node = self.map.nodes.get(str(link.get("to_node_idx")))
            if node and node.get("on_stop_line"):
                end_index = self.link_end_point_indices[link_index]
                distance_to_stopline = max(
                    0.0, self.cumulative_m[end_index] - self.cumulative_m[route_index]
                )
                affected_light_id = node.get("traffic_light_id") or None
                if affected_light_id:
                    break
            if self.cumulative_m[self.link_end_point_indices[link_index]] - \
                    self.cumulative_m[route_index] > self.traffic_light_lookahead_m:
                break
        return is_junction, distance_to_stopline, affected_light_id

    def update(self, x: float, y: float, heading_rad: float) -> RouteProgress:
        self._route_index = self._nearest_index(x, y, heading_rad)
        near_index = self._index_at_distance_ahead(self._route_index, self.near_distance_m)
        target_index = self._index_at_distance_ahead(self._route_index, self.target_distance_m)
        px, py = self.points[self._route_index]
        near_x, near_y = self.points[near_index]
        target_x, target_y = self.points[target_index]
        link_index = self.point_link_indices[self._route_index]
        is_junction, distance_to_stopline, affected_light_id = \
            self._road_semantics(self._route_index)
        return RouteProgress(
            route_index=self._route_index,
            route_link_index=link_index,
            current_link_id=self.link_ids[link_index],
            distance_to_route_m=math.hypot(px - x, py - y),
            near_x=near_x,
            near_y=near_y,
            near_route_index=near_index,
            target_x=target_x,
            target_y=target_y,
            target_route_index=target_index,
            command=self.point_road_options[near_index],
            gt_command=self.point_road_options[self._route_index],
            is_junction=is_junction,
            distance_to_stopline_m=distance_to_stopline,
            affected_light_id=affected_light_id,
            future_waypoints=self.get_future_waypoints(self._route_index),
        )
