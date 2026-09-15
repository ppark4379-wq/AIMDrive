"""Time-aligned roundabout entry prediction and bumper-gap following."""

import json
import math
import os

from ...localization.point import Point
from ..highway_lane_planning.lane_map import HighwayLaneMap
from ..highway_lane_planning.planner import HighwayLanePlanner
from ..highway_lane_planning.structures import HighwayLaneLink
from . import parameters as p
from .structures import LatticePlanningResult


class RoundaboutLaneMap(HighwayLaneMap):
    """Use the existing MGeo projection/advance APIs with local road links."""

    def __init__(self, map_data_dir=None):
        self.links = {}
        self.successors = {}
        self.bounds = {}
        if map_data_dir is None:
            return
        with open(os.path.join(map_data_dir, "link_set.json"), encoding="utf-8-sig") as stream:
            raw_links = json.load(stream)
        stop = Point(*p.ROUNDABOUT_ENTRY_STOP_POINT)
        for raw in raw_links:
            coordinates = raw.get("points", [])
            if len(coordinates) < 2 or not any(
                math.hypot(x[0] - stop.x, x[1] - stop.y) < 80.0
                for x in coordinates
            ):
                continue
            points = [Point(float(x[0]), float(x[1])) for x in coordinates]
            cumulative = HighwayLanePlanner._cumulative_distance(points)
            if cumulative[-1] > 0.01:
                link = HighwayLaneLink(raw, 0, points, cumulative, False)
                self.links[link.id] = link
                self.bounds[link.id] = (
                    min(x[0] for x in coordinates), max(x[0] for x in coordinates),
                    min(x[1] for x in coordinates), max(x[1] for x in coordinates),
                )
        for link in self.links.values():
            self.successors[link.id] = tuple(
                other.id for other in self.links.values()
                if link.to_node_id and other.from_node_id == link.to_node_id
            )

    def match(self, position, heading=None, preferred_ids=()):
        best, score = None, float("inf")
        for link in self.links.values():
            bounds = self.bounds.get(link.id)
            if bounds is not None and not (
                bounds[0] - 2.5 <= position.x <= bounds[1] + 2.5
                and bounds[2] - 2.5 <= position.y <= bounds[3] + 2.5
            ):
                continue
            match = self.project(position, link)
            if match is None or match.distance > 2.5:
                continue
            angle = abs(self._angle_difference(heading, match.yaw)) if heading is not None else 0.0
            if angle > math.pi / 3.0:
                continue
            cost = match.distance + 2.0 * angle
            if cost < score:
                best, score = match, cost
        return best


class RoundaboutEntryPlanner:
    def __init__(self, map_data_dir=None):
        self.lane_map = RoundaboutLaneMap(map_data_dir)
        self.committed = False
        self.previous_position = None
        self.active = False

    @staticmethod
    def _route(points):
        return HighwayLaneLink(
            {"idx": "ego_route"}, 0, points,
            HighwayLanePlanner._cumulative_distance(points), False,
        )

    @staticmethod
    def _distance(speed, target, time_value, acceleration_scale=1.0):
        acceleration = (
            p.ROUNDABOUT_PREDICTION_ACCELERATION * acceleration_scale if target >= speed
            else -p.ROUNDABOUT_COMFORT_BRAKING
        )
        ramp = min(time_value, abs((target - speed) / acceleration))
        return max(0.0, speed * ramp + 0.5 * acceleration * ramp * ramp
                   + target * (time_value - ramp))

    def _object_poses(self, obj, match, time_value):
        yaw = math.radians(obj.heading_deg)
        speed = max(0.0, float(obj.velocity))
        if match is None or speed < 0.05:
            return [(Point(obj.position.x + speed * time_value * math.cos(yaw),
                           obj.position.y + speed * time_value * math.sin(yaw)), yaw)]
        # Preserve the measured lane offset instead of snapping the box to its centerline.
        offset = (-(obj.position.x - match.x) * math.sin(match.yaw)
                  + (obj.position.y - match.y) * math.cos(match.yaw))
        heading_error = self.lane_map._angle_difference(yaw, match.yaw)
        return [
            (Point(point.x - offset * math.sin(angle), point.y + offset * math.cos(angle)),
             angle + heading_error * math.exp(-time_value))
            for point, angle, _ in self.lane_map.advance_options(match, speed * time_value)
        ]

    def _tracks(self, ego, entries):
        tracks = []
        for entry in entries:
            obj = entry["object_info"]
            if obj.is_static or obj.type not in (1, 2):
                continue
            if obj.position.distance(ego.position) > p.ROUNDABOUT_TRACK_DISTANCE:
                continue
            match = self.lane_map.match(obj.position, math.radians(obj.heading_deg))
            tracks.append((obj, match))
        return tracks

    def _front(self, ego, route, ego_s, tracks):
        lead = None
        for obj, _ in tracks:
            match = self.lane_map.project(obj.position, route)
            difference = abs(self.lane_map._angle_difference(
                math.radians(obj.heading_deg), match.yaw))
            if match.s <= ego_s or match.distance > 2.0 or difference > math.pi / 4:
                continue
            gap = match.s - ego_s - p.VEHICLE_LENGTH - 0.5 * max(1.0, obj.size_x)
            if lead is None or gap < lead["gap"]:
                speed = max(0.0, obj.velocity * math.cos(difference))
                desired_gap = p.ROUNDABOUT_FOLLOW_MIN_GAP + p.ROUNDABOUT_FOLLOW_TIME_GAP * ego.velocity
                limit = max(0.0, speed + 0.5 * (gap - desired_gap))
                lead = {"gap": gap, "speed": speed, "limit": limit,
                        "id": obj.unique_id, "object": obj}
        return lead

    def _collision(self, ego, route, ego_s, tracks, target, predictions, acceleration_scale=1.0):
        speed = max(0.0, ego.velocity)
        rear_collision = None
        rear_impacts = set()
        for index, poses in enumerate(predictions):
            time_value = index * p.ROUNDABOUT_PREDICTION_STEP
            distance = self._distance(speed, target, time_value, acceleration_scale)
            if ego_s + distance > route.length:
                break
            point, yaw, _ = self.lane_map.point_at_s(route, ego_s + distance)
            if index == 0:
                point, yaw = ego.position, ego.yaw
            for track_index, ((obj, _), alternatives) in enumerate(zip(tracks, poses)):
                if track_index in rear_impacts:
                    continue
                for center, heading in alternatives:
                    if HighwayLanePlanner._obb_collision(point, yaw, center, heading, obj):
                        behind = ((center.x - point.x) * math.cos(yaw)
                                  + (center.y - point.y) * math.sin(yaw)) < 0.0
                        rear = behind and math.cos(heading - yaw) > 0.5
                        collision = {"collision_time": time_value, "collision_distance": distance,
                                     "blocker_unique_id": obj.unique_id, "blocker": obj.name,
                                     "rear_conflict": rear,
                                     "conflict_point": [float(center.x), float(center.y)]}
                        if not rear:
                            return collision
                        rear_impacts.add(track_index)
                        if rear_collision is None:
                            rear_collision = collision
        return rear_collision

    def plan(self, ego, path, entries, state, desired_velocity=None):
        stop = Point(*p.ROUNDABOUT_ENTRY_STOP_POINT)
        if self.previous_position is not None and ego.position.distance(self.previous_position) > 12.0:
            self.committed = False
        self.previous_position = Point(ego.position.x, ego.position.y)
        self.active = state["in_approach"] or (
            self.committed and ego.position.distance(stop) < p.ROUNDABOUT_COMMITTED_DISTANCE
        )
        if not self.active:
            self.committed = False
            return None
        if len(path) < 2:
            return self._result(path, 0.0, "invalid_path", {})
        route = self._route(path)
        ego_match = self.lane_map.project(ego.position, route)
        stop_match = self.lane_map.project(stop, route)
        if ego_match is None or stop_match is None or ego_match.distance > 5.0:
            return self._result(path, 0.0, "invalid_path", {})
        # Vehicle position follows the existing rear-to-front OBB convention.
        stop_gap = stop_match.s - ego_match.s - p.VEHICLE_LENGTH - p.ROUNDABOUT_STOP_BUFFER
        if state["past_go_line"]:
            self.committed = True
        cruise = min(p.ROUNDABOUT_ENTRY_CRUISE_VELOCITY,
                     p.ROUNDABOUT_ENTRY_CRUISE_VELOCITY if desired_velocity is None
                     else max(0.0, desired_velocity))
        tracks = self._tracks(ego, entries)
        front = self._front(ego, route, ego_match.s, tracks)
        target = min(cruise, front["limit"]) if front else cruise
        count = int(round(p.ROUNDABOUT_PREDICTION_HORIZON / p.ROUNDABOUT_PREDICTION_STEP)) + 1
        predictions = [
            [self._object_poses(obj, match, i * p.ROUNDABOUT_PREDICTION_STEP)
             for obj, match in tracks]
            for i in range(count)
        ]
        collision = self._collision(ego, route, ego_match.s, tracks, target, predictions)
        if collision is None and not self.committed:
            # A launch must remain clear with slower-than-requested acceleration too.
            collision = self._collision(
                ego, route, ego_match.s, tracks, target, predictions, acceleration_scale=0.65
            )
        details = dict(state, committed=self.committed, stop_gap=stop_gap,
                       front_id=front["id"] if front else None,
                       front_gap=front["gap"] if front else None)
        phase = "follow" if front else "cruise"
        if collision is not None:
            details.update(collision)
            if not self.committed:
                phase = "yield"
                target = min(target, math.sqrt(2.0 * p.ROUNDABOUT_COMFORT_BRAKING * max(0.0, stop_gap)))
                if stop_gap <= 0.25 or collision["collision_time"] <= p.ROUNDABOUT_EMERGENCY_TTC:
                    target = 0.0
            elif collision["rear_conflict"]:
                # Braking for a following car makes its closing speed worse.
                phase = "rear_pressure"
            else:
                # Entry permission removes occupancy stops, never collision protection.
                phase = "collision_slowdown"
                safe_target = None
                for factor in (0.75, 0.5, 0.25, 0.0):
                    candidate = target * factor
                    if self._collision(ego, route, ego_match.s, tracks, candidate, predictions) is None:
                        safe_target = candidate
                        break
                target = 0.0 if safe_target is None else safe_target
                if collision["collision_time"] <= p.ROUNDABOUT_EMERGENCY_TTC:
                    target = 0.0
                    phase = "emergency"
        elif not self.committed and stop_gap <= 1.5 and cruise > 0.05:
            self.committed = True
            details["committed"] = True
        details["target_velocity"] = target
        result = self._result(path, target, phase, details)
        result.predicted_object_paths = [
            [row[j][0][0] for row in predictions] for j in range(len(tracks))
        ]
        return result

    @staticmethod
    def _result(path, target, phase, details):
        details["phase"] = phase
        if phase in ("yield", "emergency", "invalid_path", "collision_slowdown"):
            reason = "roundabout_entry_occupied_stop" if target <= 0.05 else "roundabout_entry_yield"
        else:
            reason = "roundabout_entry_follow_lead" if phase == "follow" else "roundabout_entry_cruise"
        return LatticePlanningResult(
            path, mode="roundabout_entry", acc_override=True,
            obstacle_detected=phase not in ("cruise", "invalid_path"),
            acc_target_velocity=target,
            ttc=details.get("collision_time", float("inf")),
            nearest_vehicle_distance=(details["front_gap"]
                                      if details.get("front_gap") is not None else float("inf")),
            reason=reason, debug_details={"roundabout_entry": details},
        )
