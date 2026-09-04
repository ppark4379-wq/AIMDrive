#!/usr/bin/env python3
import math
import time


class TrafficLightStopController(object):
    """Compute a red-light speed cap from stop nodes projected on the route."""

    def __init__(
        self,
        path,
        stopline_mapper,
        is_closed_path=False,
        stop_statuses=None,
        route_lateral_threshold=2.0,
        max_lookahead=120.0,
        stop_margin=4.0,
        comfortable_deceleration=2.5,
        signal_timeout=2.0,
        hold_on_stale=True,
    ):
        self.path = list(path)
        self.stopline_mapper = stopline_mapper
        self.is_closed_path = bool(is_closed_path)
        self.stop_statuses = set(int(x) for x in (stop_statuses or [1]))
        self.route_lateral_threshold = float(route_lateral_threshold)
        self.max_lookahead = float(max_lookahead)
        self.stop_margin = float(stop_margin)
        self.comfortable_deceleration = max(0.1, float(comfortable_deceleration))
        self.signal_timeout = max(0.0, float(signal_timeout))
        self.hold_on_stale = bool(hold_on_stale)

        self._path_s = self._build_path_distances()
        self._loop_length = self._calculate_loop_length()
        self._route_targets = self._build_route_targets()
        self._active_red_signal_id = None
        self._last_log_key = None
        self.last_decision = None

        target_count = sum(len(targets) for targets in self._route_targets.values())
        print(
            "[TrafficSignal] direct node mapping ready: "
            "map_signal_ids={}, stop_nodes={}, route_signal_ids={}, route_stop_nodes={}".format(
                len(stopline_mapper),
                getattr(stopline_mapper, "stop_node_count", "?"),
                len(self._route_targets),
                target_count,
            )
        )

    @staticmethod
    def _distance_xy(a, b):
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    def _build_path_distances(self):
        if not self.path:
            return []
        distances = [0.0]
        for start, end in zip(self.path[:-1], self.path[1:]):
            distances.append(distances[-1] + self._distance_xy(start, end))
        return distances

    def _calculate_loop_length(self):
        if not self._path_s:
            return 0.0
        length = self._path_s[-1]
        if self.is_closed_path and len(self.path) > 1:
            length += self._distance_xy(self.path[-1], self.path[0])
        return length

    def _nearest_path_index(self, point):
        px, py = float(point[0]), float(point[1])
        best_index = None
        best_distance_sq = float("inf")
        for index, path_point in enumerate(self.path):
            dx = float(path_point.x) - px
            dy = float(path_point.y) - py
            distance_sq = dx * dx + dy * dy
            if distance_sq < best_distance_sq:
                best_index = index
                best_distance_sq = distance_sq
        return best_index, math.sqrt(best_distance_sq)

    def _build_route_targets(self):
        targets_by_signal = {}
        if not self.path:
            return targets_by_signal

        for signal_id in self.stopline_mapper.mapped_signal_ids():
            for stop_point in self.stopline_mapper.get_stop_points(signal_id):
                point = stop_point.get("point", [])
                if len(point) < 2:
                    continue
                path_index, lateral_distance = self._nearest_path_index(point)
                if path_index is None or lateral_distance > self.route_lateral_threshold:
                    continue
                targets_by_signal.setdefault(str(signal_id), []).append({
                    "signal_id": str(signal_id),
                    "node_id": str(stop_point.get("idx", "")),
                    "point": point,
                    "path_index": path_index,
                    "path_s": self._path_s[path_index],
                    "lateral_distance": lateral_distance,
                })

        for targets in targets_by_signal.values():
            targets.sort(key=lambda target: target["path_s"])
        return targets_by_signal

    @staticmethod
    def _parse_signal(current_traffic_light):
        if not current_traffic_light or len(current_traffic_light) < 2:
            return None
        return {
            "id": str(current_traffic_light[0]),
            "status": int(current_traffic_light[1]),
            "type": int(current_traffic_light[2]) if len(current_traffic_light) >= 3 else None,
            "received_at": (
                float(current_traffic_light[3])
                if len(current_traffic_light) >= 4
                else None
            ),
        }

    def _forward_distance(self, current_waypoint, target):
        current_waypoint = max(0, min(int(current_waypoint), len(self._path_s) - 1))
        distance = target["path_s"] - self._path_s[current_waypoint]
        if self.is_closed_path:
            if distance < -0.75 and self._loop_length > 0.0:
                distance += self._loop_length
            elif distance < 0.0:
                distance = 0.0
        return distance

    def _find_upcoming_target(self, signal_id, current_waypoint):
        candidates = []
        for target in self._route_targets.get(str(signal_id), []):
            distance = self._forward_distance(current_waypoint, target)
            if 0.0 <= distance <= self.max_lookahead:
                candidates.append((distance, target))
        if not candidates:
            return None, None
        return min(candidates, key=lambda item: item[0])

    def _log_once(self, key, message):
        if key == self._last_log_key:
            return
        self._last_log_key = key
        print(message)

    def get_target_velocity(
        self,
        current_traffic_light,
        current_waypoint,
        planned_velocity,
    ):
        """Return a speed cap in m/s; planned speed is returned when unrestricted."""
        planned_velocity = float(planned_velocity)
        signal = self._parse_signal(current_traffic_light)
        if signal is None:
            self.last_decision = None
            return planned_velocity

        is_fresh = True
        if signal["received_at"] is not None and self.signal_timeout > 0.0:
            is_fresh = time.monotonic() - signal["received_at"] <= self.signal_timeout

        is_red = signal["status"] in self.stop_statuses
        if is_fresh:
            if is_red:
                self._active_red_signal_id = signal["id"]
            else:
                if self._active_red_signal_id == signal["id"]:
                    self._active_red_signal_id = None
                self.last_decision = None
                self._log_once(
                    ("go", signal["id"], signal["status"]),
                    "[TrafficSignal] GO: id={} status={}".format(
                        signal["id"], signal["status"]
                    ),
                )
                return planned_velocity
        elif not (self.hold_on_stale and self._active_red_signal_id == signal["id"]):
            self.last_decision = None
            return planned_velocity

        signal_id = self._active_red_signal_id or signal["id"]
        distance, target = self._find_upcoming_target(signal_id, current_waypoint)
        if target is None:
            self.last_decision = None
            self._log_once(
                ("unmapped", signal_id),
                "[TrafficSignal] red signal has no upcoming route stop node: id={}".format(
                    signal_id
                ),
            )
            return planned_velocity

        remaining_distance = max(0.0, distance - self.stop_margin)
        allowed_velocity = math.sqrt(
            2.0 * self.comfortable_deceleration * remaining_distance
        )
        target_velocity = min(planned_velocity, allowed_velocity)
        self.last_decision = {
            "signal_id": signal_id,
            "status": signal["status"],
            "node_id": target["node_id"],
            "path_index": target["path_index"],
            "distance": distance,
            "target_velocity": target_velocity,
        }

        state = "STOP" if target_velocity <= 0.01 else "APPROACH"
        self._log_once(
            (state, signal_id, target["node_id"]),
            "[TrafficSignal] {}: id={} node={} path_index={} distance={:.1f}m".format(
                state,
                signal_id,
                target["node_id"],
                target["path_index"],
                distance,
            ),
        )
        return target_velocity

    def get_route_targets(self, signal_id):
        return list(self._route_targets.get(str(signal_id), []))

    @property
    def route_signal_count(self):
        return len(self._route_targets)
