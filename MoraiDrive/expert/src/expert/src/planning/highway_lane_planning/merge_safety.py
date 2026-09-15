"""Shared, time-aligned merge safety for both highway convergence sections."""

import math

from ...localization.point import Point
from . import parameters as p


class MergeSafety:
    def __init__(self, planner):
        self.planner = planner
        self.lane_map = planner.lane_map
        self.tracks = {}
        self.zone = None
        self.details = {}
        self.following_ids = set()

    def reset(self):
        self.tracks.clear()
        self.zone = None
        self.details = {}

    def zone_for(self, ego_match):
        link_id = ego_match.lane_link.id
        successors = set(self.lane_map.successors.get(link_id, ()))
        for name, shared, center in (
            ("FIRST_PREDICTIVE", p.PREDICTIVE_MERGE_SUCCESSORS, p.PREDICTIVE_MERGE_POINT),
            ("SECOND_ENTRY", p.CONSERVATIVE_MERGE_SUCCESSORS, p.CONSERVATIVE_MERGE_POINT),
        ):
            if (link_id in shared or successors.intersection(shared)) and math.hypot(
                ego_match.x - center[0], ego_match.y - center[1]
            ) < 130.0:
                return name
        return None

    def _pose(self, obj, match, time_value):
        yaw = math.radians(obj.heading_deg)
        speed = max(0.0, obj.velocity)
        if match is None or speed < 0.05:
            return [(Point(obj.position.x + speed * time_value * math.cos(yaw),
                           obj.position.y + speed * time_value * math.sin(yaw)), yaw)]
        offset = (-(obj.position.x - match.x) * math.sin(match.yaw)
                  + (obj.position.y - match.y) * math.cos(match.yaw))
        angle_error = self.lane_map._angle_difference(yaw, match.yaw)
        return [
            (Point(point.x - offset * math.sin(angle), point.y + offset * math.cos(angle)),
             angle + angle_error * math.exp(-time_value))
            for point, angle, _ in self.lane_map.advance_options(match, speed * time_value)
        ]

    def _update_tracks(self, ego, entries, matches):
        match_by_id = {id(entry["object_info"]): match for entry, match in matches}
        for record in self.tracks.values():
            record[2] += self.planner.control_dt
        for entry in entries:
            obj = entry["object_info"]
            if obj.position.distance(ego.position) > 160.0:
                continue
            match = match_by_id.get(id(obj))
            key = self.planner._merge_zone_object_key(entry, match)
            if key.startswith("link:"):
                key = "{}:{:.1f}:{:.1f}".format(key, obj.position.x, obj.position.y)
            self.tracks[key] = [obj, match, 0.0]
        self.tracks = {key: value for key, value in self.tracks.items()
                       if value[2] <= p.MERGE_TRACK_HOLD_TIME}
        return list(self.tracks.items())

    @staticmethod
    def travel(speed, target, time_value):
        delay = min(p.MERGE_CONTROL_DELAY, time_value)
        remaining = time_value - delay
        accel = p.MERGE_PREDICT_ACCEL if target >= speed else -p.MERGE_PREDICT_DECEL
        ramp = min(remaining, abs((target - speed) / accel))
        return speed * delay + speed * ramp + 0.5 * accel * ramp * ramp + target * (remaining - ramp)

    def _route(self, ego, ego_match, reference_path, speed):
        # The control path can be shorter than the safety horizon; extend it on
        # its actual successor, preserving its first measured points.
        points = list(reference_path or [])
        if len(points) < 2:
            points = [item[0] for item in self.lane_map.route_points(
                ego_match, max(80.0, speed * p.MERGE_SAFETY_HORIZON + 20.0), 0.5)]
        nearest = min(range(len(points)), key=lambda i: points[i].distance(ego.position))
        points = [Point(ego.position.x, ego.position.y)] + points[nearest + 1:]
        if len(points) < 2:
            return [], []
        cumulative = self.planner._cumulative_distance(points)
        needed = max(80.0, speed * p.MERGE_SAFETY_HORIZON + 20.0)
        if cumulative[-1] < needed:
            yaw = math.atan2(points[-1].y - points[-2].y, points[-1].x - points[-2].x)
            end_match = self.lane_map.match(points[-1], heading=yaw)
            if end_match is not None:
                extension = self.lane_map.route_points(end_match, needed - cumulative[-1], 0.5)
                points.extend(item[0] for item in extension[1:])
                cumulative = self.planner._cumulative_distance(points)
        return points, cumulative

    def _conflict(self, ego, points, cumulative, target, tracks, predictions):
        initial_rear = set()
        for index, row in enumerate(predictions):
            time_value = index * p.MERGE_SAFETY_STEP
            distance = self.travel(max(0.0, ego.velocity), target, time_value)
            if distance > cumulative[-1]:
                return {"time": time_value, "id": "path_end", "distance": distance}
            point, yaw = self.planner._pose_at_distance(points, cumulative, distance)
            if index == 0:
                point, yaw = ego.position, ego.yaw
            for track_index, ((key, (obj, _, _)), options) in enumerate(zip(tracks, row)):
                for center, angle in options:
                    # Rear vehicles on a converging lane must stay in this check.
                    # Only an already-following, same-path car is exempt from a
                    # new braking demand that would increase its closing speed.
                    dx, dy = center.x - point.x, center.y - point.y
                    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
                    lateral = abs(-dx * math.sin(yaw) + dy * math.cos(yaw))
                    if (index == 0 and key in self.following_ids and longitudinal < -3.0
                            and lateral < 1.0 and math.cos(angle - yaw) > 0.95):
                        initial_rear.add(track_index)
                    if track_index in initial_rear:
                        continue
                    if self.planner._obb_collision(point, yaw, center, angle, obj):
                        return {"time": time_value, "id": key, "distance": distance,
                                "rear": longitudinal < 0.0, "kind": "obb"}
                    if math.cos(angle - yaw) > 0.8 and lateral < 0.5 * (p.VEHICLE_WIDTH + max(1.0, obj.size_y)):
                        gap = (longitudinal - p.VEHICLE_LENGTH - 0.5 * max(1.0, obj.size_x)
                               if longitudinal >= 0 else -longitudinal - 0.5 * max(1.0, obj.size_x))
                        if gap < p.MERGE_BUMPER_CLEARANCE:
                            return {"time": time_value, "id": key, "distance": distance,
                                    "rear": longitudinal < 0.0, "kind": "gap"}
        return None

    def limit(self, ego, ego_match, matches, entries, reference_path=None, desired_velocity=None):
        zone = self.zone_for(ego_match)
        if zone is None:
            self.reset()
            return None
        if zone != self.zone:
            self.reset()
            self.zone = zone
        tracks = self._update_tracks(ego, entries, matches)
        self.following_ids = {
            key for key, (_, match, _) in tracks if match is not None
            and match.lane_link.id == ego_match.lane_link.id and match.s < ego_match.s
        }
        requested = max(0.0, ego.velocity) if desired_velocity is None else max(0.0, desired_velocity)
        if desired_velocity is None and requested < 1.0:
            requested = p.MERGE_MIN_YIELD_SPEED
        points, cumulative = self._route(ego, ego_match, reference_path, max(ego.velocity, requested))
        if len(points) < 2:
            self.details = {"zone": zone, "decision": "STOP", "cause": "invalid_path"}
            return 0.0
        predictions = [
            [self._pose(obj, match, i * p.MERGE_SAFETY_STEP + age)
             for _, (obj, match, age) in tracks]
            for i in range(int(round(p.MERGE_SAFETY_HORIZON / p.MERGE_SAFETY_STEP)) + 1)
        ]
        collision = self._conflict(ego, points, cumulative, requested, tracks, predictions)
        self.details = {"zone": zone, "decision": "GO", "requested": requested,
                        "conflict": collision, "tracks": len(tracks),
                        "predicted_paths": [[row[j][0][0] for row in predictions] for j in range(len(tracks))]}
        if collision is None:
            return float("inf") if desired_velocity is None else requested
        # Search from the fastest candidate down. Every candidate includes
        # actuator delay and finite braking, not an instantaneous speed change.
        target = requested - p.MERGE_SPEED_SEARCH_STEP
        while target > 0.0:
            if self._conflict(ego, points, cumulative, target, tracks, predictions) is None:
                self.details.update(decision="YIELD", target=target)
                return target
            target -= p.MERGE_SPEED_SEARCH_STEP
        self.details.update(decision="STOP", target=0.0)
        return 0.0
