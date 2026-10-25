"""Time-aligned roundabout entry prediction and bumper-gap following."""

import json
import math
import os
from types import SimpleNamespace

from ...interfaces.geometry import Point
from ...obstacle.object_info import is_ve_tracked_unknown
from ..highway.common.lane_map import HighwayLaneMap
from ..highway.planner import HighwayLanePlanner
from ..highway.common.types import HighwayLaneLink
from ...config.parameters import lattice as p
from ..lattice.structures import LatticePlanningResult
from .diagnostics import RoundaboutDiagnostics


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
        self.entering = False
        self.previous_position = None
        self.active = False
        self.diagnostics = RoundaboutDiagnostics()

    @staticmethod
    def _route(points):
        return HighwayLaneLink(
            {"idx": "ego_route"}, 0, points,
            HighwayLanePlanner._cumulative_distance(points), False,
        )

    @staticmethod
    def _distance(speed, target, time_value, acceleration_scale=1.0, launch_delay=0.0):
        delay = min(time_value, launch_delay) if target > speed else 0.0
        time_value -= delay
        acceleration = (
            p.ROUNDABOUT_PREDICTION_ACCELERATION * acceleration_scale if target >= speed
            else -p.ROUNDABOUT_COMFORT_BRAKING
        )
        ramp = min(time_value, abs((target - speed) / acceleration))
        return max(0.0, speed * delay + speed * ramp + 0.5 * acceleration * ramp * ramp
                   + target * (time_value - ramp))

    def _signed_route_gap(self, point, route, ego_match):
        match = self.lane_map.project(point, route)
        if match is None:
            return None
        progress = match.s
        # A sliding preview no longer contains a passed hold/merge point.
        if match.s <= 1e-6 or match.s >= route.length - 1e-6:
            progress += ((point.x - match.x) * math.cos(match.yaw)
                         + (point.y - match.y) * math.sin(match.yaw))
        return progress - ego_match.s

    @staticmethod
    def _following_distances(speed, ceiling, count, lead, acceleration_scale, launch_delay):
        distance = 0.0
        distances = [distance]
        step = p.ROUNDABOUT_PREDICTION_STEP
        for index in range(1, count):
            elapsed = (index - 1) * step
            gap = lead["gap"] + lead["speed"] * elapsed - distance
            desired_gap = p.ROUNDABOUT_FOLLOW_MIN_GAP + p.ROUNDABOUT_FOLLOW_TIME_GAP * speed
            target = min(ceiling, max(0.0, lead["speed"] + 0.5 * (gap - desired_gap)))
            accel = p.ROUNDABOUT_PREDICTION_ACCELERATION * acceleration_scale
            if elapsed < launch_delay and target > speed:
                accel = 0.0
            next_speed = max(0.0, speed + max(-p.ROUNDABOUT_COMFORT_BRAKING * step,
                                             min(accel * step, target - speed)))
            distance += 0.5 * (speed + next_speed) * step
            distances.append(distance)
            speed = next_speed
        return distances

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

    def _is_rear_entry_vehicle(self, match):
        """Classify upstream traffic along the circulating lane, not Ego's yaw."""
        circulating = self.lane_map.links.get(p.ROUNDABOUT_ENTRY_CIRCULATING_LINK_ID)
        if match is None or circulating is None:
            return False
        merge = self.lane_map.project(Point(*p.ROUNDABOUT_ENTRY_MERGE_POINT), circulating)
        if merge is None:
            return False
        rear_distance = self.lane_map.forward_distance(match, merge)
        if rear_distance is None or rear_distance <= 0.0:
            return False
        # On a loop, a vehicle ahead can eventually reach the merge again.
        # Use the nearer directed arc so that it retains the front distance cap.
        front_distance = self.lane_map.forward_distance(merge, match)
        if front_distance is not None and front_distance <= rear_distance:
            return False
        approach = self.lane_map.match(Point(*p.ROUNDABOUT_ENTRY_HOLD_POINT))
        if approach is not None and (
                match.lane_link.id == approach.lane_link.id
                or self.lane_map.forward_distance(match, approach) is not None):
            # Cars queued behind Ego on its entry road are not circulating traffic.
            return False
        return True

    def _tracks(self, ego, entries):
        tracks = []
        audit = self.diagnostics.tracks if self.diagnostics.checks is not None else None
        for entry in entries:
            obj = entry["object_info"]
            if not is_ve_tracked_unknown(obj) and (obj.is_static or obj.type not in (1, 2)):
                if audit is not None:
                    audit.append((obj, 'static_or_non_vehicle', None))
                continue
            distance = obj.position.distance(ego.position)
            if distance > max(p.ROUNDABOUT_TRACK_DISTANCE, p.ROUNDABOUT_REAR_TRACK_DISTANCE):
                if audit is not None:
                    audit.append((obj, 'outside_max_distance', None))
                continue
            match = self.lane_map.match(obj.position, math.radians(obj.heading_deg))
            if distance > p.ROUNDABOUT_TRACK_DISTANCE and (
                    distance > p.ROUNDABOUT_REAR_TRACK_DISTANCE
                    or not self._is_rear_entry_vehicle(match)):
                if audit is not None:
                    audit.append((obj, 'rear_direction_not_confirmed', match))
                continue
            tracks.append((obj, match))
            if audit is not None:
                audit.append((obj, 'base_range' if distance <= p.ROUNDABOUT_TRACK_DISTANCE else 'rear_extension', match))
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

    def _collision(self, ego, route, ego_s, tracks, target, predictions,
                   acceleration_scale=1.0, launch_delay=0.0, lead=None, diagnostic=None):
        speed = max(0.0, ego.velocity)
        rear_collision = None
        rear_impacts = set()
        route_origin, route_yaw, _ = self.lane_map.point_at_s(route, ego_s)
        yaw_error = self.lane_map._angle_difference(ego.yaw, route_yaw)
        # Enlarge only the incoming rear vehicle's front envelope before entry.
        # The measured body stays entirely inside it, so no original collision
        # can disappear. Tracking, lead following and prediction timing stay intact.
        extra_margin = max(0.0, p.ROUNDABOUT_ENTRY_REAR_FRONT_MARGIN) if not (self.entering or self.committed) else 0.0
        collision_shapes = []
        for obj, match in tracks:
            margin = extra_margin if extra_margin and self._is_rear_entry_vehicle(match) else 0.0
            shape = SimpleNamespace(size_x=max(1.0, obj.size_x) + margin, size_y=obj.size_y) if margin else obj
            collision_shapes.append((margin, shape))
        if diagnostic is not None:
            diagnostic['rear_front_margin_m_by_track'] = [margin for margin, _ in collision_shapes]
        # Follow a moving lead throughout the prediction, not a frozen launch limit.
        distances = (self._following_distances(speed, target, len(predictions), lead,
                                              acceleration_scale, launch_delay)
                     if lead is not None else None)
        for index, poses in enumerate(predictions):
            time_value = index * p.ROUNDABOUT_PREDICTION_STEP
            distance = (distances[index] if distances is not None else
                        self._distance(speed, target, time_value, acceleration_scale, launch_delay))
            if ego_s + distance > route.length:
                return {"collision_time": time_value, "collision_distance": distance,
                        "blocker_unique_id": None, "blocker": "prediction_path_end",
                        "rear_conflict": False, "prediction_incomplete": True}
            point, yaw, _ = self.lane_map.point_at_s(route, ego_s + distance)
            # Preserve the measured pose at rest; a stopped car cannot snap to path yaw.
            tracking_weight = math.exp(-distance / p.VEHICLE_LENGTH)
            point = Point(point.x + (ego.position.x - route_origin.x) * tracking_weight,
                          point.y + (ego.position.y - route_origin.y) * tracking_weight)
            yaw += yaw_error * tracking_weight
            if diagnostic is not None:
                diagnostic['checked_until_s'] = time_value
            if diagnostic is not None and (index % max(1, round(2. / p.ROUNDABOUT_PREDICTION_STEP)) == 0
                                           or index == len(predictions) - 1):
                diagnostic['ego_samples'].append([time_value, point.x, point.y, math.degrees(yaw)])
            for track_index, ((obj, _), alternatives) in enumerate(zip(tracks, poses)):
                if track_index in rear_impacts:
                    continue
                for center, heading in alternatives:
                    if diagnostic is not None:
                        # Reference-point distance only; this is not bumper clearance.
                        distance_sq = (center.x - point.x) ** 2 + (center.y - point.y) ** 2
                        previous = diagnostic['closest_reference_distance'].get(str(track_index))
                        if previous is None or distance_sq < previous['distance_sq_m2']:
                            diagnostic['closest_reference_distance'][str(track_index)] = dict(
                                id=obj.unique_id, t_s=time_value, distance_sq_m2=distance_sq,
                                ego_reference_xy=[point.x, point.y], object_center_xy=[center.x, center.y])
                    margin, shape = collision_shapes[track_index]
                    collision_center = (Point(center.x + .5 * margin * math.cos(heading),
                                              center.y + .5 * margin * math.sin(heading)) if margin else center)
                    if HighwayLanePlanner._obb_collision(point, yaw, collision_center, heading, shape):
                        behind = ((center.x - point.x) * math.cos(yaw)
                                  + (center.y - point.y) * math.sin(yaw)) < 0.0
                        rear = behind and math.cos(heading - yaw) > 0.5
                        collision = {"collision_time": time_value, "collision_distance": distance,
                                     "blocker_unique_id": obj.unique_id, "blocker": obj.name,
                                     "rear_conflict": rear,
                                     "conflict_point": [float(center.x), float(center.y)]}
                        if margin:
                            collision['rear_front_margin_m'] = margin
                        if not rear:
                            return collision
                        rear_impacts.add(track_index)
                        if rear_collision is None:
                            rear_collision = collision
        return rear_collision

    def _checked_collision(self, ego, route, ego_s, tracks, target, predictions, lead=None):
        checks = [(1.0, 0.0)]
        if not self.committed:
            checks.append((p.ROUNDABOUT_SLOW_ACCELERATION_SCALE, 0.0))
            if ego.velocity <= 0.1:
                checks.append((p.ROUNDABOUT_SLOW_ACCELERATION_SCALE, p.ROUNDABOUT_LAUNCH_DELAY))
        conflicts = []
        for scale, delay in checks:
            diagnostic = None
            kwargs = {'lead': lead}
            if self.diagnostics.checks is not None:
                diagnostic = dict(target_mps=target, acceleration_scale=scale, launch_delay_s=delay,
                                  lead_id=lead['id'] if lead else None, checked_until_s=0.,
                                  ego_samples=[], closest_reference_distance={})
                kwargs['diagnostic'] = diagnostic
            collision = self._collision(ego, route, ego_s, tracks, target, predictions, scale, delay,
                                        **kwargs)
            if diagnostic is not None:
                diagnostic['collision'] = dict(collision) if collision is not None else None
                self.diagnostics.checks.append(diagnostic)
            if collision is not None:
                collision["acceleration_scale"] = scale
                collision["launch_delay"] = delay
                conflicts.append(collision)
                if collision["collision_time"] <= 0.0:
                    break
        return min(conflicts, key=lambda c: (c["rear_conflict"], c["collision_time"])) if conflicts else None

    def _entry_follow_speed(self, ego, route, ego_s, tracks, ceiling, predictions, lead):
        ceilings = [ceiling * factor for factor in (0.85, 0.7, 0.55)]
        if lead is not None:
            ceilings.append(min(ceiling, lead["speed"]))
        for candidate in sorted(set(ceilings), reverse=True):
            target = min(candidate, lead["limit"]) if lead else candidate
            if target <= 0.1:
                continue
            if self._checked_collision(ego, route, ego_s, tracks, candidate, predictions, lead) is None:
                return candidate, target
        return None

    def plan(self, ego, path, entries, state, desired_velocity=None):
        self.diagnostics.begin()
        before = dict(entering=self.entering, committed=self.committed, active=self.active)
        result = self._plan(ego, path, entries, state, desired_velocity)
        try:
            self.diagnostics.finish(self, ego, path, entries, state, desired_velocity, before, result)
        except Exception as error:
            self.diagnostics.disabled = True
            print('[ROUNDABOUT_DEBUG] capture disabled: {}'.format(error))
        return result

    def _plan(self, ego, path, entries, state, desired_velocity=None):
        stop = Point(*p.ROUNDABOUT_ENTRY_STOP_POINT)
        if self.previous_position is not None and ego.position.distance(self.previous_position) > 12.0:
            self.committed = False
            self.entering = False
        self.previous_position = Point(ego.position.x, ego.position.y)
        self.active = state["in_approach"] or (
            (self.committed or self.entering)
            and ego.position.distance(stop) < p.ROUNDABOUT_COMMITTED_DISTANCE
        )
        if not self.active:
            self.committed = False
            self.entering = False
            return None
        if len(path) < 2:
            return self._result(path, 0.0, "invalid_path", {})
        route = self._route(path)
        ego_match = self.lane_map.project(ego.position, route)
        if ego_match is None or ego_match.distance > 5.0:
            return self._result(path, 0.0, "invalid_path", {})
        # HOLD_POINT already specifies the Ego reference pose shown in MORAI.
        stop_gap = self._signed_route_gap(Point(*p.ROUNDABOUT_ENTRY_HOLD_POINT), route, ego_match)
        merge_gap = self._signed_route_gap(Point(*p.ROUNDABOUT_ENTRY_MERGE_POINT), route, ego_match)
        if stop_gap is None or merge_gap is None:
            return self._result(path, 0.0, "invalid_path", {})
        if stop_gap < -p.ROUNDABOUT_HOLD_TOLERANCE:
            self.entering = True
        heading_error = abs(self.lane_map._angle_difference(ego.yaw, ego_match.yaw))
        if self.entering and merge_gap <= 0.0 and heading_error <= math.pi / 6.0:
            self.committed = True
        cruise = min(p.ROUNDABOUT_ENTRY_CRUISE_VELOCITY,
                     p.ROUNDABOUT_ENTRY_CRUISE_VELOCITY if desired_velocity is None
                     else max(0.0, desired_velocity))
        tracks = self._tracks(ego, entries)
        front = self._front(ego, route, ego_match.s, tracks)
        ceiling = cruise
        before_hold = not self.entering and not self.committed
        if before_hold and stop_gap > p.ROUNDABOUT_HOLD_TOLERANCE:
            ceiling = min(ceiling, p.ROUNDABOUT_APPROACH_VELOCITY)
        target = min(ceiling, front["limit"]) if front else ceiling
        count = int(round(p.ROUNDABOUT_PREDICTION_HORIZON / p.ROUNDABOUT_PREDICTION_STEP)) + 1
        predictions = [
            [self._object_poses(obj, match, i * p.ROUNDABOUT_PREDICTION_STEP)
             for obj, match in tracks]
            for i in range(count)
        ]
        if self.diagnostics.checks is not None:
            self.diagnostics.predictions = (tracks, predictions)
        collision = self._checked_collision(ego, route, ego_match.s, tracks, ceiling, predictions, front)
        original_collision = collision
        adjusted = False
        if (collision is not None and before_hold and stop_gap <= 1.5
                and collision["collision_time"] > p.ROUNDABOUT_EMERGENCY_TTC):
            selected = self._entry_follow_speed(
                ego, route, ego_match.s, tracks, ceiling, predictions, front)
            if selected is not None:
                ceiling, target = selected
                collision = None
                adjusted = True
        details = dict(state, committed=self.committed, stop_gap=stop_gap,
                       hold_gap=stop_gap,
                       hold_point=list(p.ROUNDABOUT_ENTRY_HOLD_POINT),
                       entering=self.entering, merge_gap=merge_gap,
                       prediction_horizon=p.ROUNDABOUT_PREDICTION_HORIZON,
                       speed_ceiling=ceiling, following_speed_adjusted=adjusted,
                       front_id=front["id"] if front else None,
                       front_gap=front["gap"] if front else None)
        phase = "follow" if front else "cruise"
        if adjusted:
            details["avoided_conflict_id"] = original_collision.get("blocker_unique_id")
        if collision is not None:
            details.update(collision)
            if before_hold:
                phase = "yield"
                target = min(target, math.sqrt(2.0 * p.ROUNDABOUT_COMFORT_BRAKING * max(0.0, stop_gap)))
                if stop_gap <= p.ROUNDABOUT_HOLD_TOLERANCE or collision["collision_time"] <= p.ROUNDABOUT_EMERGENCY_TTC:
                    target = 0.0
            elif self.committed and collision["rear_conflict"]:
                # Braking for a following car makes its closing speed worse.
                phase = "rear_pressure"
            else:
                # Entry permission removes occupancy stops, never collision protection.
                phase = "collision_slowdown"
                safe_target = None
                for factor in (0.75, 0.5, 0.25, 0.0):
                    candidate = ceiling * factor
                    if self._checked_collision(ego, route, ego_match.s, tracks, candidate, predictions, front) is None:
                        safe_target = min(candidate, front["limit"]) if front else candidate
                        break
                if safe_target is None and collision["rear_conflict"]:
                    # Do not turn an unavoidable closing-rear conflict into hard braking.
                    # Entry admission must prevent this; keep the front-safe speed ceiling.
                    phase = "rear_pressure"
                    details["no_safe_velocity"] = True
                else:
                    target = 0.0 if safe_target is None else safe_target
                if not collision["rear_conflict"] and collision["collision_time"] <= p.ROUNDABOUT_EMERGENCY_TTC:
                    target = 0.0
                    phase = "emergency"
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
