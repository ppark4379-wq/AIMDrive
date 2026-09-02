"""Lightweight MGeo link matcher used by collection and route planning."""
import json
import heapq
import bisect
import zlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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
    target_x: float
    target_y: float
    target_route_index: int
    command: int
    is_junction: bool
    distance_to_stopline_m: Optional[float]
    affected_light_id: Optional[str]


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
        self._segments: List[Tuple[str, float, float, float, float, float]] = []
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        for link in links:
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
    """Stateful, forward-only route localization and LMDrive command provider."""
    COMMAND_LEFT = 1
    COMMAND_RIGHT = 2
    COMMAND_STRAIGHT = 3
    COMMAND_LANEFOLLOW = 4

    def __init__(self, mgeo_map: MGeoMap, route_metadata: dict,
                 target_distance_m: float = 25.0,
                 command_lookahead_m: float = 30.0,
                 initial_route_index: Optional[int] = None):
        self.map = mgeo_map
        self.link_ids = list(route_metadata["nav_link_ids"])
        if not self.link_ids:
            raise ValueError("route has no nav_link_ids")
        self.target_distance_m = target_distance_m
        self.command_lookahead_m = command_lookahead_m
        self.points: List[Tuple[float, float]] = []
        self.point_link_indices: List[int] = []
        self.link_end_point_indices: List[int] = []
        for link_index, link_id in enumerate(self.link_ids):
            points = mgeo_map.links[link_id].get("points", [])
            if self.points and points and self.points[-1] == tuple(points[0][:2]):
                points = points[1:]
            for point in points:
                self.points.append((float(point[0]), float(point[1])))
                self.point_link_indices.append(link_index)
            self.link_end_point_indices.append(len(self.points) - 1)
        if not self.points:
            raise ValueError("route has no points")
        self.cumulative_m = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.cumulative_m.append(
                self.cumulative_m[-1] + math.hypot(b[0] - a[0], b[1] - a[1])
            )
        if initial_route_index is not None and initial_route_index < 0:
            raise ValueError("initial_route_index must be >= 0")
        self._route_index: Optional[int] = initial_route_index

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
        return min(candidates, key=lambda i: (
            math.hypot(self.points[i][0] - x, self.points[i][1] - y)
            + 2.0 * self.map._angle_error(self._point_heading(i), heading_rad)
        ))

    def _command(self, route_index: int) -> int:
        current_distance = self.cumulative_m[route_index]
        end = bisect.bisect_right(
            self.cumulative_m, current_distance + self.command_lookahead_m
        )
        seen = set()
        for point_index in range(route_index, min(end, len(self.points))):
            link_index = self.point_link_indices[point_index]
            if link_index in seen:
                continue
            seen.add(link_index)
            signal = str(self.map.links[self.link_ids[link_index]].get("related_signal") or "")
            if "left" in signal:
                return self.COMMAND_LEFT
            if "right" in signal:
                return self.COMMAND_RIGHT
            if signal == "straight":
                return self.COMMAND_STRAIGHT
        return self.COMMAND_LANEFOLLOW

    def _road_semantics(self, route_index: int):
        current_link_index = self.point_link_indices[route_index]
        current_link = self.map.links[self.link_ids[current_link_index]]
        is_junction = bool(current_link.get("related_signal"))
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
                affected_light_id = node.get("traffic_light_id")
                break
            if self.cumulative_m[self.link_end_point_indices[link_index]] - \
                    self.cumulative_m[route_index] > 100.0:
                break
        return is_junction, distance_to_stopline, affected_light_id

    def update(self, x: float, y: float, heading_rad: float) -> RouteProgress:
        self._route_index = self._nearest_index(x, y, heading_rad)
        target_distance = self.cumulative_m[self._route_index] + self.target_distance_m
        target_index = min(
            bisect.bisect_left(self.cumulative_m, target_distance), len(self.points) - 1
        )
        px, py = self.points[self._route_index]
        target_x, target_y = self.points[target_index]
        link_index = self.point_link_indices[self._route_index]
        is_junction, distance_to_stopline, affected_light_id = \
            self._road_semantics(self._route_index)
        return RouteProgress(
            route_index=self._route_index,
            route_link_index=link_index,
            current_link_id=self.link_ids[link_index],
            distance_to_route_m=math.hypot(px - x, py - y),
            target_x=target_x,
            target_y=target_y,
            target_route_index=target_index,
            command=self._command(self._route_index),
            is_junction=is_junction,
            distance_to_stopline_m=distance_to_stopline,
            affected_light_id=affected_light_id,
        )
