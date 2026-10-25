#!/usr/bin/env python3
import math
import time

from .route_targets import SCHEMA as ROUTE_SIGNAL_SCHEMA, signal_permission

LEFT_GREEN_BIT = 32


class TrafficLightStopController(object):
    """Compute a route-signal speed cap for the exact upcoming stop occurrence."""

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
        route_signal_targets=None,
        ignored_signal_ids=None,
        left_arrow_only_signal_ids=None,
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
        self.route_permission_bound = route_signal_targets is not None
        self._route_signal_document = route_signal_targets
        self.ignored_signal_ids = set(str(x) for x in (ignored_signal_ids or []))
        self.left_arrow_only_signal_ids = set(
            str(x) for x in (left_arrow_only_signal_ids or [])
        )
        if self.ignored_signal_ids & self.left_arrow_only_signal_ids:
            raise ValueError("A signal cannot be both ignored and left-arrow-only")

        self._path_s = self._build_path_distances()
        self._loop_length = self._calculate_loop_length()
        self._route_targets = self._build_route_targets()
        self._active_red_signal_id = None
        self._active_stop_target_key = None
        self._active_stop_status = None
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

        if self._route_signal_document is not None:
            document = self._route_signal_document
            if document.get("schema") != ROUTE_SIGNAL_SCHEMA or document.get("path_points") != len(self.path):
                raise ValueError("Invalid exact-route signal target contract")
            for row in document["targets"]:
                target = dict(row)
                index = int(target["path_index"])
                if not 0 <= index < len(self.path):
                    raise ValueError("Exact-route signal target outside Expert path")
                signal_id = str(target["signal_id"])
                points = [point for point in self.stopline_mapper.get_stop_points(signal_id)
                          if str(point.get("idx")) == str(target["node_id"])]
                if len(points) != 1:
                    raise ValueError("Exact-route signal node is absent or duplicated in map")
                point = points[0]["point"]
                lateral = math.hypot(float(self.path[index].x) - float(point[0]),
                                     float(self.path[index].y) - float(point[1]))
                if lateral > 1e-3:
                    raise ValueError("Exact-route signal node differs from Expert path")
                target.update(point=point, path_s=self._path_s[index], lateral_distance=lateral)
                targets_by_signal.setdefault(signal_id, []).append(target)
            for targets in targets_by_signal.values():
                targets.sort(key=lambda target: target["path_s"])
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
                    "signal_movement": None,
                    "signal_route_link_id": None,
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
            "source_time": (
                float(current_traffic_light[4])
                if len(current_traffic_light) >= 5
                and current_traffic_light[4] is not None
                else None
            ),
            "reference_time": (
                float(current_traffic_light[5])
                if len(current_traffic_light) >= 6
                and current_traffic_light[5] is not None
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

    def _next_route_target(self, current_waypoint):
        candidates = [self._find_upcoming_target(signal_id, current_waypoint)
                      for signal_id in self._route_targets
                      if signal_id not in self.ignored_signal_ids]
        candidates = [row for row in candidates if row[1] is not None]
        return min(candidates, key=lambda row: row[0]) if candidates else (None, None)

    def _log_once(self, key, message):
        if key == self._last_log_key:
            return
        self._last_log_key = key
        print(message)

    def _signal_is_fresh(self, signal):
        if signal is None:
            return False
        if (
            signal["source_time"] is not None
            and signal["reference_time"] is not None
            and self.signal_timeout > 0.0
        ):
            return (
                0.0
                <= signal["reference_time"] - signal["source_time"]
                <= self.signal_timeout
            )
        elif signal["received_at"] is not None and self.signal_timeout > 0.0:
            return time.monotonic() - signal["received_at"] <= self.signal_timeout
        return True


    def get_target_velocity(self, current_traffic_light, current_waypoint, planned_velocity):
        """Apply explicit signal policies while retaining exact route binding."""
        if self.route_permission_bound:
            return self._get_route_target_velocity(current_traffic_light, current_waypoint, planned_velocity)
        return self._get_policy_target_velocity(current_traffic_light, current_waypoint, planned_velocity)


    def _get_route_target_velocity(
        self,
        current_traffic_light,
        current_waypoint,
        planned_velocity,
    ):
        """Return a speed cap in m/s; planned speed is returned when unrestricted."""
        planned_velocity = float(planned_velocity)
        signal = self._parse_signal(current_traffic_light)
        forced_stale = False
        if self.route_permission_bound:
            _, upcoming = self._next_route_target(current_waypoint)
            if upcoming is None:
                self.last_decision = None
                return planned_velocity
            key = (upcoming["signal_id"], upcoming["node_id"], upcoming["path_index"])
            if signal is None or signal["id"] != upcoming["signal_id"]:
                if not (upcoming["signal_id"] in self.left_arrow_only_signal_ids
                        or (self.hold_on_stale and self._active_stop_target_key == key)):
                    self.last_decision = None
                    return planned_velocity
                # A future/previous/unrelated signal packet cannot release the
                # route stop held here. This is stale evidence, never a fresh GT.
                held_status = self._active_stop_status if self._active_stop_target_key == key else None
                signal = {"id": upcoming["signal_id"], "status": held_status,
                          "received_at": None, "source_time": None, "reference_time": None}
                forced_stale = True
        if signal is None:
            self.last_decision = None
            return planned_velocity

        is_fresh = self._signal_is_fresh(signal) and not forced_stale
        # Bind the movement to this upcoming stop occurrence *before* deciding
        # whether the phase permits it. One light ID can govern several nodes.
        distance, target = self._find_upcoming_target(signal["id"], current_waypoint)
        if target is None:
            self.last_decision = None
            if self._active_red_signal_id == signal["id"]:
                self._active_red_signal_id = None
                self._active_stop_target_key = None
            self._log_once(
                ("unmapped", signal["id"]),
                "[TrafficSignal] signal has no upcoming route stop node: id={}".format(signal["id"]),
            )
            return planned_velocity
        target_key = (signal["id"], target["node_id"], target["path_index"])
        left_only = signal["id"] in self.left_arrow_only_signal_ids
        if left_only:
            permission = (signal["status"] is not None and signal["status"] >= 0
                          and bool(signal["status"] & LEFT_GREEN_BIT)) if is_fresh else None
            is_red = permission is not True
        else:
            permission = signal_permission(signal["status"], target.get("signal_movement"))
            is_red = (not permission if permission is not None
                      else signal["status"] in self.stop_statuses)
        # An unsupported fresh phase is not evidence to release a protected
        # movement already stopped at this exact route occurrence.
        if (self.route_permission_bound and target.get("signal_movement") in ("straight", "left")
                and permission is None and self._active_stop_target_key == target_key):
            is_red = True
        if is_fresh:
            if is_red:
                self._active_red_signal_id = signal["id"]
                self._active_stop_target_key = target_key
                self._active_stop_status = signal["status"]
            else:
                if self._active_red_signal_id == signal["id"]:
                    self._active_red_signal_id = None
                    self._active_stop_target_key = None
                self.last_decision = None
                self._log_once(
                    ("go", signal["id"], signal["status"]),
                    "[TrafficSignal] GO: id={} status={}".format(
                        signal["id"], signal["status"]
                    ),
                )
                return planned_velocity
        elif not (left_only or (self.hold_on_stale and self._active_red_signal_id == signal["id"]
                                    and self._active_stop_target_key == target_key)):
            self.last_decision = None
            return planned_velocity
        signal_id = signal["id"]

        return self._apply_stop_target(distance, target, signal["status"],
                                       planned_velocity, permission, is_fresh)


    def _get_policy_target_velocity(
        self,
        current_traffic_light,
        current_waypoint,
        planned_velocity,
    ):
        """Return a speed cap in m/s; planned speed is returned when unrestricted."""
        planned_velocity = float(planned_velocity)
        signal = self._parse_signal(current_traffic_light)
        is_fresh = self._signal_is_fresh(signal)
        stop_candidates = []
        ordinary_stop = False

        # All other lights retain their configured red/status and stale rules.
        if (signal is not None
                and signal["id"] not in self.ignored_signal_ids
                and signal["id"] not in self.left_arrow_only_signal_ids):
            distance, target = self._find_upcoming_target(signal["id"], current_waypoint)
            target_key = ((signal["id"], target["node_id"], target["path_index"])
                          if target is not None else None)
            if is_fresh:
                ordinary_stop = signal["status"] in self.stop_statuses
                if ordinary_stop and target is not None:
                    self._active_red_signal_id = signal["id"]
                    self._active_stop_target_key = target_key
                    self._active_stop_status = signal["status"]
                elif self._active_red_signal_id == signal["id"]:
                    self._active_red_signal_id = None
                    self._active_stop_target_key = None
            else:
                ordinary_stop = (self.hold_on_stale and target_key is not None
                                 and self._active_red_signal_id == signal["id"]
                                 and self._active_stop_target_key == target_key)
            if ordinary_stop and target is not None:
                stop_candidates.append((distance, target, signal["status"]))

        # A left-only approach needs positive permission for this exact ID.
        # Missing/stale samples (including [] from SyncMode) cannot release it.
        for signal_id in sorted(self.left_arrow_only_signal_ids):
            left_is_green = (
                is_fresh and signal["id"] == signal_id
                and signal["status"] >= 0
                and bool(signal["status"] & LEFT_GREEN_BIT)
            )
            if left_is_green:
                continue
            distance, target = self._find_upcoming_target(signal_id, current_waypoint)
            if target is not None:
                status = signal["status"] if signal and signal["id"] == signal_id else None
                stop_candidates.append((distance, target, status))

        if not stop_candidates:
            self.last_decision = None
            if signal is not None:
                if ordinary_stop:
                    self._log_once(
                        ("unmapped", signal["id"]),
                        "[TrafficSignal] red signal has no upcoming route stop node: id={}".format(
                            signal["id"]
                        ),
                    )
                elif signal["id"] in self.ignored_signal_ids or is_fresh:
                    state = "IGNORE" if signal["id"] in self.ignored_signal_ids else "GO"
                    self._log_once(
                        (state, signal["id"], signal["status"]),
                        "[TrafficSignal] {}: id={} status={}".format(
                            state, signal["id"], signal["status"]
                        ),
                    )
            return planned_velocity

        distance, target, status = min(stop_candidates, key=lambda item: item[0])
        signal_id = target["signal_id"]
        permission = False if (is_fresh and signal["id"] == signal_id
                               and signal_id in self.left_arrow_only_signal_ids) else None
        return self._apply_stop_target(distance, target, status, planned_velocity,
                                       permission, is_fresh)


    def _apply_stop_target(self, distance, target, status, planned_velocity,
                           permission=None, is_fresh=False):
        signal_id = target["signal_id"]
        remaining_distance = max(0.0, distance - self.stop_margin)
        allowed_velocity = math.sqrt(
            2.0 * self.comfortable_deceleration * remaining_distance
        )
        target_velocity = min(planned_velocity, allowed_velocity)
        self.last_decision = {
            "signal_id": signal_id,
            "status": status,
            "node_id": target["node_id"],
            "path_index": target["path_index"],
            "distance": distance,
            "target_velocity": target_velocity,
            "signal_movement": target.get("signal_movement"),
            "signal_route_link_id": target.get("signal_route_link_id"),
            "signal_permission": permission if is_fresh else None,
            "route_permission_bound": self.route_permission_bound,
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
