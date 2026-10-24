"""Shared, time-aligned merge safety for both highway convergence sections."""

import math

from ...interfaces.geometry import Point
from . import parameters as p


class MergeSafety:
    def __init__(self, planner):
        self.planner = planner
        self.lane_map = planner.lane_map
        self.tracks = {}
        self.zone = None
        self.details = {}
        self.following_ids = set()
        self.spacing_ids = None
        self.second_entry_speed = None
        self.second_launch_active = False

    def reset(self):
        self.tracks.clear()
        self.spacing_ids = None
        self.zone = None
        self.details = {}
        self.second_entry_speed = None
        self.second_launch_active = False

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
        # Integrate the same decaying heading error used below. Otherwise a
        # matched car pointing into Ego's lane would keep its old lateral
        # position forever, hiding a real cut-in.
        offset += speed * math.sin(angle_error) * (1.0 - math.exp(-time_value))
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
    def travel(speed, target, time_value, acceleration=None):
        delay = min(p.MERGE_CONTROL_DELAY, time_value)
        remaining = time_value - delay
        acceleration = p.MERGE_PREDICT_ACCEL if acceleration is None else acceleration
        accel = acceleration if target >= speed else -p.MERGE_PREDICT_DECEL
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

    def _shares_corridor(self, ego_match, obj, match):
        if match is None or match.lane_link.id == ego_match.lane_link.id:
            return True
        if (self.lane_map.forward_distance(ego_match, match) is not None
                or self.lane_map.forward_distance(match, ego_match) is not None):
            return True
        if (set(self.lane_map.successors.get(ego_match.lane_link.id, ()))
                & set(self.lane_map.successors.get(match.lane_link.id, ()))):
            return True
        # A matched car can still be actively cutting across its mapped lane.
        error = self.lane_map._angle_difference(math.radians(obj.heading_deg), match.yaw)
        return abs(error) > math.radians(5.0)

    @staticmethod
    def body_gap(point, yaw, center, angle, obj, lateral_margin=p.COLLISION_MARGIN):
        # Project the entire Ego body onto the other lane's axes. Parallel
        # neighbors have no lateral overlap; converging angled bodies still do.
        ego_x = point.x + 0.5 * p.VEHICLE_LENGTH * math.cos(yaw)
        ego_y = point.y + 0.5 * p.VEHICLE_LENGTH * math.sin(yaw)
        dx, dy = center.x - ego_x, center.y - ego_y
        along = dx * math.cos(angle) + dy * math.sin(angle)
        lateral = abs(-dx * math.sin(angle) + dy * math.cos(angle))
        cosine, sine = abs(math.cos(yaw-angle)), abs(math.sin(yaw-angle))
        half_along = 0.5 * (p.VEHICLE_LENGTH*cosine + p.VEHICLE_WIDTH*sine)
        half_lateral = 0.5 * (p.VEHICLE_LENGTH*sine + p.VEHICLE_WIDTH*cosine)
        gap = abs(along) - half_along - 0.5 * max(1.0, obj.size_x)
        overlap = lateral < half_lateral + 0.5 * max(1.0, obj.size_y) + lateral_margin
        return gap, along < 0.0, lateral, overlap

    def ego_pose(self, ego, points, cumulative, distance):
        """Keep measured heading until enough travel permits a physical turn.

        The short connector from Ego to a nearby lane center may point sharply
        sideways. It must not rotate a stationary OBB during actuator delay.
        """
        if distance <= 1e-9:
            return ego.position, ego.yaw
        point, path_yaw = self.planner._pose_at_distance(points, cumulative, distance)
        delta = math.atan2(math.sin(path_yaw - ego.yaw), math.cos(path_yaw - ego.yaw))
        maximum_turn = distance * p.PREDICTION_MAX_YAW_PER_METER
        yaw = ego.yaw + max(-maximum_turn, min(maximum_turn, delta))
        return point, yaw

    def _conflict(self, ego, points, cumulative, target, tracks, predictions, ignore_ids=(), acceleration=None):
        initial_rear = set()
        for index, row in enumerate(predictions):
            time_value = index * p.MERGE_SAFETY_STEP
            distance = self.travel(max(0.0, ego.velocity), target, time_value, acceleration)
            if distance > cumulative[-1]:
                return {"time": time_value, "id": "path_end", "distance": distance}
            point, yaw = self.ego_pose(ego, points, cumulative, distance)
            for track_index, ((key, (obj, _, _)), options) in enumerate(zip(tracks, row)):
                if key in ignore_ids:
                    continue
                for center, angle in options:
                    # Rear vehicles on a converging lane must stay in this check.
                    # Only an already-following, same-path car is exempt from a
                    # new braking demand that would increase its closing speed.
                    dx, dy = center.x - point.x, center.y - point.y
                    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
                    lateral = abs(-dx * math.sin(yaw) + dy * math.cos(yaw))
                    rear_along = dx * math.cos(angle) + dy * math.sin(angle)
                    rear_lateral = abs(-dx * math.sin(angle) + dy * math.cos(angle))
                    if (index == 0 and key in self.following_ids and rear_along < -3.0
                            and rear_lateral < 1.0 and math.cos(angle - yaw) > 0.95):
                        initial_rear.add(track_index)
                    if track_index in initial_rear:
                        continue
                    if self.planner._obb_collision(point, yaw, center, angle, obj):
                        return {"time": time_value, "id": key, "distance": distance,
                                "rear": longitudinal < 0.0, "kind": "obb",
                                "ego_xy": (point.x, point.y), "ego_yaw_deg": math.degrees(yaw),
                                "object_xy": (center.x, center.y), "object_yaw_deg": math.degrees(angle)}
                    # Parallel neighbors must not become in-lane leads just
                    # because Ego is still angled after a lane change. Evaluate
                    # spacing along the other car's lane; OBB tests above still
                    # check all vehicles, including crossing/cutting-in bodies.
                    if self.spacing_ids is not None and key not in self.spacing_ids:
                        continue
                    # In the second convergence, apply the longitudinal following
                    # clearance only where the bodies share a lateral corridor.
                    # Inflating that corridor again can classify a passing
                    # neighbor as a lead (logged npc:25 at 2.51 m lateral).
                    # The full, margin-inflated SAT collision check above is
                    # unchanged and still checks every predicted pose.
                    gap_margin = 0.0 if self.zone == "SECOND_ENTRY" else p.COLLISION_MARGIN
                    gap, gap_rear, gap_lateral, lateral_overlap = self.body_gap(
                        point, yaw, center, angle, obj, lateral_margin=gap_margin)
                    if math.cos(angle - yaw) > 0.8 and lateral_overlap:
                        if gap < p.MERGE_BUMPER_CLEARANCE:
                            return {"time": time_value, "id": key, "distance": distance,
                                    "rear": gap_rear, "kind": "gap",
                                    "lateral": gap_lateral, "bumper_gap": gap,
                                    "ego_xy": (point.x, point.y), "ego_yaw_deg": math.degrees(yaw),
                                    "object_xy": (center.x, center.y), "object_yaw_deg": math.degrees(angle)}
        return None

    def _second_merge_entered(self, ego_match):
        """Identify entry into the second conflict area using route progress."""
        if ego_match.lane_link.id in p.CONSERVATIVE_MERGE_SUCCESSORS:
            return True
        center = self.lane_map.project(Point(*p.CONSERVATIVE_MERGE_POINT), ego_match.lane_link)
        return ego_match.s >= center.s - p.CONSERVATIVE_MERGE_ENTRY_RADIUS

    def _second_merge_follower(self, ego, ego_match, obj, match):
        # The two MGeo links remain distinct after their physical lanes have
        # converged. Recognize an already-established following relationship
        # from measured geometry, never from a future predicted merge alone.
        if (self.zone != "SECOND_ENTRY" or match is None
                or not self._second_merge_entered(ego_match)):
            return False
        shared = (set(self.lane_map.successors.get(ego_match.lane_link.id, ()))
                  & set(self.lane_map.successors.get(match.lane_link.id, ()))
                  & set(p.CONSERVATIVE_MERGE_SUCCESSORS))
        if not shared:
            return False
        center = Point(ego.position.x + .5*p.VEHICLE_LENGTH*math.cos(ego.yaw),
                       ego.position.y + .5*p.VEHICLE_LENGTH*math.sin(ego.yaw))
        projection = self.lane_map.project(center, match.lane_link)
        if (projection.distance > .5*p.VEHICLE_WIDTH
                or abs(self.lane_map._angle_difference(ego.yaw, projection.yaw)) > math.radians(5.)
                or abs(self.lane_map._angle_difference(math.radians(obj.heading_deg), match.yaw)) > math.radians(5.)):
            return False
        gap, rear, _, _ = self.body_gap(ego.position, ego.yaw, obj.position,
                                       math.radians(obj.heading_deg), obj)
        return rear and gap >= p.MERGE_BUMPER_CLEARANCE

    def limit(self, ego, ego_match, matches, entries, reference_path=None, desired_velocity=None,
              forward_speed_limit=float("inf")):
        zone = self.zone_for(ego_match)
        if zone is None:
            self.reset()
            return None
        if zone != self.zone:
            self.reset()
            self.zone = zone
        tracks = self._update_tracks(ego, entries, matches)
        self.spacing_ids = {key for key, (obj, match, _) in tracks
                            if self._shares_corridor(ego_match, obj, match)}
        self.following_ids = {
            key for key, (obj, match, _) in tracks
            if (match is None or self.lane_map.forward_distance(match, ego_match) is not None
                or self._second_merge_follower(ego, ego_match, obj, match))
        }
        # Link boundaries / unmatched upstream cars must not turn an actual
        # same-lane follower into a request to brake in front of it. _conflict
        # still requires measured rear position, <1 m lateral offset and
        # aligned headings; vehicles on a converging lane remain checked.
        requested = max(0.0, ego.velocity) if desired_velocity is None else max(0.0, desired_velocity)
        if desired_velocity is None and requested < 1.0:
            requested = p.MERGE_MIN_YIELD_SPEED
        profile_requested = requested
        second_entered = zone == "SECOND_ENTRY" and self._second_merge_entered(ego_match)
        if second_entered:
            if self.second_entry_speed is None:
                self.second_entry_speed = min(max(0.0, ego.velocity, requested), p.SECOND_MERGE_SLOWDOWN_MAX_SPEED)
            # Keep the entry speed through the merge, but validate it anew each
            # cycle. A lower route profile must not repeatedly undo a safe GO.
            if requested > 0.05:
                requested = min(max(requested, self.second_entry_speed), forward_speed_limit)
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
        self.details["profile_requested"] = profile_requested
        launch_eligible = (
            zone == "SECOND_ENTRY"
            and ego_match.lane_link.id in p.SECOND_MERGE_LAUNCH_LINK_IDS
            and ego.position.distance(Point(*p.CONSERVATIVE_MERGE_POINT)) <= p.SECOND_MERGE_LAUNCH_RADIUS
            and (self.second_launch_active or ego.velocity <= p.SECOND_MERGE_LAUNCH_START_SPEED)
            and requested <= forward_speed_limit
            and requested >= p.SECOND_MERGE_LAUNCH_MIN_TARGET
            and requested > max(0.0, ego.velocity) + 0.5)
        self.second_launch_active = False
        if launch_eligible and (collision is None or collision.get("rear")):
            # The stronger controller and the prediction must agree. Check both
            # a conservative launch and full acceleration against ALL traffic;
            # distance alone must never exempt a fast closing rear vehicle.
            launch_conflict = None
            for acceleration in (p.SECOND_MERGE_LAUNCH_ACCEL_MIN,
                                 0.5 * (p.SECOND_MERGE_LAUNCH_ACCEL_MIN + p.SECOND_MERGE_LAUNCH_ACCEL_MAX),
                                 p.SECOND_MERGE_LAUNCH_ACCEL_MAX):
                launch_conflict = self._conflict(
                    ego, points, cumulative, requested, tracks, predictions, acceleration=acceleration)
                if launch_conflict is not None:
                    break
            self.details["launch_conflict"] = launch_conflict
            if launch_conflict is not None:
                self.details["launch_conflict_acceleration"] = acceleration
            if launch_conflict is None:
                self.second_launch_active = True
                self.details.update(decision="GO", target=requested, conflict=None,
                                    requested_conflict=collision, cause="second_merge_verified_launch",
                                    launch_verified=True, launch_target=requested,
                                    launch_acceleration=(p.SECOND_MERGE_LAUNCH_ACCEL_MIN,
                                                         p.SECOND_MERGE_LAUNCH_ACCEL_MAX))
                if requested > profile_requested:
                    self.details["speed_floor"] = requested
                return requested
        if collision is None:
            if requested > profile_requested + 0.05:
                self.details.update(cause="second_merge_committed_speed", target=requested,
                                    speed_floor=requested)
            return float("inf") if desired_velocity is None else requested
        checked = [(requested, collision)]
        # Once past the second merge area, reducing speed for a closing rear
        # car can create the collision. Validate holding the measured speed
        # before searching slower candidates. Never accelerate above it, exceed
        # front ACC, or exempt the rear car from the trajectory check.
        if (zone == "SECOND_ENTRY" and collision.get("rear")
                and self._second_merge_entered(ego_match)
                and ego.velocity > requested + 0.05
                and ego.velocity <= forward_speed_limit):
            hold_conflict = self._conflict(
                ego, points, cumulative, ego.velocity, tracks, predictions)
            self.details["hold_conflict"] = hold_conflict
            checked.append((float(ego.velocity), hold_conflict))
            if hold_conflict is None:
                self.details.update(decision="GO", target=float(ego.velocity),
                                    cause="second_merge_rear_safe_hold",
                                    requested_conflict=collision, conflict=None,
                                    speed_floor=float(ego.velocity))
                return float(ego.velocity)
        # Search from the fastest candidate down. Every candidate includes
        # actuator delay and finite braking, not an instantaneous speed change.
        target = requested - p.MERGE_SPEED_SEARCH_STEP
        while target > 0.0:
            conflict = self._conflict(ego, points, cumulative, target, tracks, predictions)
            if conflict is None:
                self.details.update(decision="YIELD", target=target)
                if second_entered and target > profile_requested:
                    self.details["speed_floor"] = target
                return target
            checked.append((target, conflict))
            target -= p.MERGE_SPEED_SEARCH_STEP
        if zone == "SECOND_ENTRY":
            # Exhausting the moving candidates is not proof that stopping is
            # safe. Expose an unavoidable/fallback stop separately for diagnosis.
            stop_conflict = self._conflict(ego, points, cumulative, 0.0, tracks, predictions)
            self.details.update(stop_conflict=stop_conflict,
                                cause="no_safe_speed" if stop_conflict else "validated_stop")
            if second_entered and stop_conflict and stop_conflict.get("rear"):
                # No collision-free solution: choose a checked non-accelerating
                # rear-risk mitigation only if stopping predicts earlier impact.
                # Do not call it GO or weaken a front/crossing emergency stop.
                current = min(max(0.0, ego.velocity), forward_speed_limit)
                if current > 0.05:
                    current_conflict = self._conflict(ego, points, cumulative, current, tracks, predictions)
                    if current_conflict is None:
                        self.details.update(decision="GO", target=current, speed_floor=current,
                                            cause="second_merge_rear_safe_hold", conflict=None)
                        return current
                    checked.append((current, current_conflict))
                alternatives = [(speed, hazard) for speed, hazard in checked
                                if 0.05 < speed <= min(max(0.0, ego.velocity), forward_speed_limit)
                                and hazard and hazard.get("rear")
                                and hazard["time"] > stop_conflict["time"] + p.MERGE_SAFETY_STEP
                                and self._conflict(ego, points, cumulative, speed, tracks, predictions,
                                                   ignore_ids={hazard["id"]}) is None]
                if alternatives:
                    speed, hazard = max(alternatives, key=lambda item: (item[1]["time"], item[0]))
                    self.details.update(decision="MITIGATE", target=speed, speed_floor=speed,
                                        cause="second_merge_rear_risk_mitigation",
                                        requested_conflict=collision, conflict=hazard)
                    return speed
        self.details.update(decision="STOP", target=0.0)
        return 0.0
