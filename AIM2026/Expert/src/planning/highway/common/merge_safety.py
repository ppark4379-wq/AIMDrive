"""Shared, time-aligned merge safety for both highway convergence sections."""

import math

from ....interfaces.geometry import Point
from ....config.parameters import highway as p


class MergeSafety:
    def __init__(self, planner):
        self.planner = planner
        self.lane_map = planner.lane_map
        self.tracks = {}
        self.zone = None
        self.details = {}
        self.following_ids = set()
        self.second_rear_gap_ids = set()
        self.spacing_ids = None
        self.second_entry_speed = None
        self.second_launch_active = False
        self.second_follow_key = None
        self.second_follow_details = {}
        self.yield_speed = None
        self.ego_lane_match = None

    def reset(self):
        self._cleared_rows = None
        self.tracks.clear()
        self.second_rear_gap_ids.clear()
        self.spacing_ids = None
        self.zone = None
        self.details = {}
        self.second_entry_speed = None
        self.second_launch_active = False
        self.second_follow_key = None
        self.second_follow_details = {}
        self.yield_speed = None
        self.ego_lane_match = None

    def zone_for(self, ego_match):
        link_id = ego_match.lane_link.id
        successors = set(self.lane_map.successors.get(link_id, ()))
        next_successors = {n for s in successors for n in self.lane_map.successors.get(s, ())}
        for name, shared, center in (
            ("FIRST_PREDICTIVE", p.PREDICTIVE_MERGE_SUCCESSORS, p.PREDICTIVE_MERGE_POINT),
            ("SECOND_ENTRY", p.CONSERVATIVE_MERGE_SUCCESSORS, p.CONSERVATIVE_MERGE_POINT),
        ):
            # The first merge ends on its common downstream link. Once the
            # entire car has cleared the junction, ordinary ACC/cut-in and
            # lane-change checks own traffic; a 130 m circle is not a yield zone.
            if (name == 'FIRST_PREDICTIVE' and link_id in shared
                    and ego_match.s > p.VEHICLE_LENGTH + p.MERGE_BUMPER_CLEARANCE):
                continue
            upstream_second = name == 'SECOND_ENTRY' and bool(next_successors.intersection(shared))
            if (link_id in shared or successors.intersection(shared) or upstream_second) and math.hypot(
                ego_match.x - center[0], ego_match.y - center[1]
            ) < 130.0:
                return name
        return None

    def _pose(self, obj, match, time_value):
        yaw = math.radians(obj.heading_deg)
        speed = max(0.0, obj.velocity)
        if getattr(obj, 'speed_unknown', False):
            # Missing speed is neither a measured zero nor a reason to brake
            # for an unrelated object. Sample its reachable longitudinal band
            # densely enough for overlapping body checks on this actual route.
            count = max(1, int(math.ceil(speed * max(0., time_value) / max(1., obj.size_x / 2.))))
            return [pose for i in range(count+1)
                    for pose in self._pose_at_speed(obj, match, time_value, speed*i/count)]
        # Detection confidence and motion-history maturity are different.
        # A provisional but available speed is still a measurement: do not
        # invent an additional stationary vehicle at every new track's origin.
        # Measured zero speed and unavailable speed retain their own checks.
        return self._pose_at_speed(obj, match, time_value, speed)

    def _pose_at_speed(self, obj, match, time_value, speed):
        yaw = math.radians(obj.heading_deg)
        if match is None or speed < 0.05:
            return [(Point(obj.position.x + speed * time_value * math.cos(yaw),
                           obj.position.y + speed * time_value * math.sin(yaw)), yaw)]
        offset = (-(obj.position.x - match.x) * math.sin(match.yaw)
                  + (obj.position.y - match.y) * math.cos(match.yaw))
        angle_error = self.lane_map._angle_difference(yaw, match.yaw)
        # On the configured no-cut-in highway, preserve the measured lateral
        # offset along the mapped road instead of extrapolating heading noise
        # into a lane change. Keep measured geometry and road successors:
        # current overlaps, Ego lane changes and road convergence still collide.
        # Other roads retain the existing cut-in prediction.
        lane_keeping = (
            p.ASSUME_HIGHWAY_VEHICLES_KEEP_LANE
            and getattr(match.lane_link, 'road_id', None) in {road for _, road in p.SECTION_ROADS}
        )
        if not lane_keeping:
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

    def _conflict(self, ego, points, cumulative, target, tracks, predictions, ignore_ids=(), acceleration=None,
                  body_only=False):
        side_margins = [self.planner._parallel_lane_margin(ego, self.ego_lane_match, obj, match)
                        for _, (obj, match, _) in tracks]
        # Enclosing circles include BOTH padded bodies and the gap clearance.
        # Outside this bound neither exact OBB nor spacing can conflict.
        radii_squared = [(
            .5*p.VEHICLE_LENGTH
            + math.hypot(.5*p.VEHICLE_LENGTH+p.COLLISION_MARGIN,
                         .5*p.VEHICLE_WIDTH+max(p.COLLISION_MARGIN, margin))
            + .5*math.hypot(max(1., obj.size_x), max(1., obj.size_y))
            + p.MERGE_BUMPER_CLEARANCE + max(0., margin))**2
            for (_, (obj, _, _)), margin in zip(tracks, side_margins)]
        # Many speed candidates share exactly the same acceleration/braking
        # trajectory before reaching their target. Reuse only fully cleared
        # rows, within this limit() call and identical prediction/ignore sets.
        cache = getattr(self, '_cleared_rows', None)
        context = (id(predictions), id(points), id(tracks), tuple(sorted(ignore_ids)), body_only)
        for index, row in enumerate(predictions):
            time_value = index * p.MERGE_SAFETY_STEP
            distance = self.travel(max(0.0, ego.velocity), target, time_value, acceleration)
            if distance > cumulative[-1]:
                return {"time": time_value, "id": "path_end", "distance": distance}
            cache_key = (context, index, distance)
            if cache is not None and cache_key in cache:
                continue
            point, yaw = self.ego_pose(ego, points, cumulative, distance)
            for track_index, ((key, (obj, obj_match, _)), options) in enumerate(zip(tracks, row)):
                if key in ignore_ids:
                    continue
                if (getattr(obj, 'keep_lane_side_only', False)
                        and self.ego_lane_match is not None and self.ego_lane_match.distance <= .5
                        and self.planner._parallel_lane_margin(ego, self.ego_lane_match, obj, obj_match) == 0.):
                    continue
                for center, angle in options:
                    # Every rear body stays in collision prediction. Only its
                    # spacing margin may be exempt below once following is
                    # established; a converging car retains both checks.
                    dx, dy = center.x - point.x, center.y - point.y
                    if dx*dx + dy*dy > radii_squared[track_index]:
                        continue
                    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
                    lateral = abs(-dx * math.sin(yaw) + dy * math.cos(yaw))
                    side_margin = side_margins[track_index]
                    if self.planner._obb_collision(point, yaw, center, angle, obj, lateral_margin=side_margin):
                        return {"time": time_value, "id": key, "distance": distance,
                                "rear": longitudinal < 0.0, "kind": "obb",
                                "ego_xy": (point.x, point.y), "ego_yaw_deg": math.degrees(yaw),
                                "object_xy": (center.x, center.y), "object_yaw_deg": math.degrees(angle)}
                    if body_only:
                        continue
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
                    # Do not reintroduce a parallel neighbor's removed side
                    # padding through the longitudinal spacing check.
                    gap_margin = 0.0 if self.zone == "SECOND_ENTRY" else side_margin
                    gap, gap_rear, gap_lateral, lateral_overlap = self.body_gap(
                        point, yaw, center, angle, obj, lateral_margin=gap_margin)
                    # Once a car is measurably following us, its rear spacing
                    # is not a demand to brake in front of it. This exempts ONLY
                    # rear clearance, not the OBB check above or a future lead.
                    if gap_rear and key in self.second_rear_gap_ids:
                        continue
                    if math.cos(angle - yaw) > 0.8 and lateral_overlap:
                        if gap < p.MERGE_BUMPER_CLEARANCE:
                            return {"time": time_value, "id": key, "distance": distance,
                                    "rear": gap_rear, "kind": "gap",
                                    "lateral": gap_lateral, "bumper_gap": gap,
                                    "ego_xy": (point.x, point.y), "ego_yaw_deg": math.degrees(yaw),
                                    "object_xy": (center.x, center.y), "object_yaw_deg": math.degrees(angle)}
            if cache is not None:
                cache.add(cache_key)
        return None

    def _second_merge_entered(self, ego_match):
        """Identify entry into the second conflict area using route progress."""
        if ego_match.lane_link.id in p.CONSERVATIVE_MERGE_SUCCESSORS:
            return True
        if not set(self.lane_map.successors.get(ego_match.lane_link.id, ())).intersection(p.CONSERVATIVE_MERGE_SUCCESSORS):
            return False
        center = self.lane_map.project(Point(*p.CONSERVATIVE_MERGE_POINT), ego_match.lane_link)
        return ego_match.s >= center.s - p.CONSERVATIVE_MERGE_ENTRY_RADIUS

    def _rear_mitigation(self, ego, points, cumulative, requested, forward_limit, tracks, predictions,
                         stop_conflict):
        """Compare actual rear impact, without trading it for a front hazard.

        A violated spacing margin is not evidence that stopping is safer.
        This fallback may still predict rear impact, so report MITIGATE, never
        GO. Only a later worst-case body impact (or none) justifies movement.
        """
        stopped = self._conflict(ego, points, cumulative, 0., tracks, predictions, body_only=True)
        if not stopped or not stopped.get('rear'):
            return None
        key = stopped['id']
        best = None
        speed = min(requested, forward_limit)
        while speed > .05:
            accelerations = ((p.SECOND_MERGE_LAUNCH_ACCEL_MIN,
                              .5*(p.SECOND_MERGE_LAUNCH_ACCEL_MIN+p.SECOND_MERGE_LAUNCH_ACCEL_MAX),
                              p.SECOND_MERGE_LAUNCH_ACCEL_MAX) if speed > ego.velocity+.05
                             else (p.MERGE_PREDICT_ACCEL,))
            worst_time = float('inf')
            allowed = True
            for acceleration in accelerations:
                # Every OTHER vehicle keeps both gap and body constraints.
                if self._conflict(ego, points, cumulative, speed, tracks, predictions,
                                  ignore_ids={key}, acceleration=acceleration) is not None:
                    allowed = False
                    break
                impact = self._conflict(ego, points, cumulative, speed, tracks, predictions,
                                        acceleration=acceleration, body_only=True)
                if impact and (not impact.get('rear') or impact['id'] != key):
                    allowed = False
                    break
                worst_time = min(worst_time, impact['time'] if impact else float('inf'))
            if allowed and worst_time > stopped['time'] + p.MERGE_SAFETY_STEP:
                if best is None or worst_time > best[0]:
                    best = (worst_time, speed)
                if math.isinf(worst_time):
                    break
            speed -= p.MERGE_SPEED_SEARCH_STEP
        if best is None:
            return None
        impact_time, speed = best
        self.yield_speed = speed
        self.details.update(decision='MITIGATE', target=speed, speed_floor=speed,
            cause='merge_rear_impact_mitigation', stop_conflict=stop_conflict,
            stop_body_conflict=stopped, mitigated_body_time=impact_time,
            rear_mitigation_verified=True, launch_target=speed,
            brake_release_verified=True, brake_release_target=speed)
        return speed

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
        connected = (match.lane_link.id == ego_match.lane_link.id
                     or self.lane_map.forward_distance(match, ego_match) is not None)
        if not shared and not connected:
            return False
        angle = math.radians(obj.heading_deg)
        gap, rear, lateral, _ = self.body_gap(ego.position, ego.yaw, obj.position, angle, obj)
        # Compare measured vehicles, not two still-distinct map centerlines.
        # Requiring the full rear clearance here revoked follower status just
        # as the rear car approached, causing a harmful fallback STOP.
        return (rear and gap >= 0.0 and lateral < .5*p.VEHICLE_WIDTH
                and math.cos(angle - ego.yaw) > .95)

    def _first_merge_follower(self, ego, ego_match, obj, match):
        """Exempt established following distance, never a rear body collision."""
        if self.zone != 'FIRST_PREDICTIVE' or match is None:
            return False
        connected = (match.lane_link.id == ego_match.lane_link.id
                     or self.lane_map.forward_distance(match, ego_match) is not None)
        gap, rear, lateral, _ = self.body_gap(
            ego.position, ego.yaw, obj.position, math.radians(obj.heading_deg), obj)
        return (connected and rear and gap >= 0. and lateral < .5*p.VEHICLE_WIDTH
                and math.cos(math.radians(obj.heading_deg)-ego.yaw) > .95)

    def _remember_second_yield(self, collision, tracks, ego_match):
        """Latch the actual vehicle yielded to, never replace it with a later car."""
        if self.zone != 'SECOND_ENTRY' or self.second_follow_key is not None or not collision:
            return
        if collision.get('rear'):
            return
        for key, (obj, match, age) in tracks:
            if key != collision.get('id') or match is None or age > 0.0:
                continue
            if match.lane_link.id == ego_match.lane_link.id:
                return
            successors = set(self.lane_map.successors.get(match.lane_link.id, ()))
            if (match.lane_link.id in p.CONSERVATIVE_MERGE_SUCCESSORS
                    or successors.intersection(p.CONSERVATIVE_MERGE_SUCCESSORS)):
                self.second_follow_key = key
                return

    def _second_follow_speed(self, ego, tracks, requested):
        """After the yielded vehicle passes, follow its gap; all collision checks still run."""
        self.second_follow_details = {}
        if self.zone != 'SECOND_ENTRY' or self.second_follow_key is None:
            return requested
        target = next((record for key, record in tracks if key == self.second_follow_key), None)
        if target is None:
            self.second_follow_details = dict(target=self.second_follow_key, phase='lost')
            self.second_follow_key = None
            return requested
        obj, match, age = target
        self.second_follow_details = dict(target=self.second_follow_key, phase='waiting_for_pass', age=age)
        # A held or unmatched observation does not authorize a launch.
        if age > 0.0 or match is None or getattr(obj, 'speed_unknown', False):
            return requested
        projection = self.lane_map.project(ego.position, match.lane_link)
        # Compare along the target lane, including the complete Ego body in front
        # of its rear-axle reference. Do not assume a side-by-side car has passed.
        gap = match.s - projection.s - p.VEHICLE_LENGTH - .5*max(1., obj.size_x)
        if gap < p.MERGE_BUMPER_CLEARANCE:
            return requested
        desired_gap = p.MERGE_BUMPER_CLEARANCE + max(0., ego.velocity)*p.CONSERVATIVE_MERGE_FOLLOW_TIME_GAP
        speed = min(requested, max(0., obj.velocity + p.ACC_DISTANCE_GAIN*(gap-desired_gap)))
        self.second_follow_details.update(phase='following_yielded_vehicle', gap=gap,
                                          desired_gap=desired_gap, target_speed=speed)
        return speed

    def limit(self, ego, ego_match, matches, entries, reference_path=None, desired_velocity=None,
              forward_speed_limit=float("inf")):
        zone = self.zone_for(ego_match)
        if zone is None:
            self.reset()
            return None
        if zone != self.zone:
            self.reset()
            self.zone = zone
        self.ego_lane_match = ego_match
        self._cleared_rows = set()
        tracks = self._update_tracks(ego, entries, matches)
        self.second_rear_gap_ids = {
            key for key, (obj, match, _) in tracks
            if (self._second_merge_follower(ego, ego_match, obj, match)
                or self._first_merge_follower(ego, ego_match, obj, match))
        }
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
        if zone == 'SECOND_ENTRY' and not self._second_merge_entered(ego_match):
            # The old speed profile slowed only after entering 422/445. Begin
            # the same approach limit on their predecessors while stopping room
            # remains; a completed upstream lane change must not release it.
            requested = min(requested, p.SECOND_MERGE_SLOWDOWN_MAX_SPEED)
        profile_requested = requested
        second_entered = zone == "SECOND_ENTRY" and self._second_merge_entered(ego_match)
        if second_entered:
            if self.second_entry_speed is None:
                self.second_entry_speed = min(max(0.0, ego.velocity, requested), p.SECOND_MERGE_SLOWDOWN_MAX_SPEED)
            # Keep the entry speed through the merge, but validate it anew each
            # cycle. A lower route profile must not repeatedly undo a safe GO.
            if requested > 0.05:
                requested = min(max(requested, self.second_entry_speed), forward_speed_limit)
        requested = self._second_follow_speed(ego, tracks, requested)
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
        # Restore a released yield progressively, but only if that slower
        # trajectory is ALSO safe against every front/rear/crossing object.
        # Never retain a braking command that creates a rear collision.
        releasing_yield = collision is None and self.yield_speed is not None and requested > self.yield_speed
        if releasing_yield:
            recovered = min(requested, max(self.yield_speed, max(0., ego.velocity))
                            + p.MERGE_ACC_RECOVERY_RATE * self.planner.control_dt)
            if self._conflict(ego, points, cumulative, recovered, tracks, predictions) is None:
                requested = recovered
            self.yield_speed = requested if requested < profile_requested else None
        self.details = {"zone": zone, "decision": "GO", "requested": requested,
                        "conflict": collision, "tracks": len(tracks),
                        "predicted_paths": [[row[j][0][0] for row in predictions] for j in range(len(tracks))]}
        self.details["profile_requested"] = profile_requested
        self.details['second_follow'] = dict(self.second_follow_details)
        self.details['rear_spacing_only_exempt'] = sorted(self.second_rear_gap_ids)
        self.details['parallel_side_margin_only'] = [key for key, (obj, match, _) in tracks
            if self.planner._parallel_lane_margin(ego, ego_match, obj, match) == 0.0]
        if releasing_yield and requested >= max(.05, ego.velocity):
            self.details['brake_release_verified'] = True
            self.details['brake_release_target'] = requested
        launch_eligible = (
            ((zone == "SECOND_ENTRY"
              and ego_match.lane_link.id in p.SECOND_MERGE_LAUNCH_LINK_IDS
              and ego.position.distance(Point(*p.CONSERVATIVE_MERGE_POINT)) <= p.SECOND_MERGE_LAUNCH_RADIUS)
             or (zone == 'FIRST_PREDICTIVE' and collision is not None and collision.get('rear')))
            # A moving vehicle also needs a verified acceleration option when
            # braking would put it in a converging rear vehicle's path.
            and (self.second_launch_active or ego.velocity <= p.SECOND_MERGE_LAUNCH_START_SPEED
                 or collision is not None and collision.get('rear'))
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
                                    requested_conflict=collision, cause=("second_merge_verified_launch"
                                        if zone == 'SECOND_ENTRY' else 'first_merge_rear_verified_launch'),
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
        # In either merge area, reducing speed for a closing rear
        # car can create the collision. Validate holding the measured speed
        # before searching slower candidates. Never accelerate above it, exceed
        # front ACC, or exempt the rear car from the trajectory check.
        if (collision.get("rear")
                and ego.velocity > requested + 0.05
                and ego.velocity <= forward_speed_limit):
            hold_conflict = self._conflict(
                ego, points, cumulative, ego.velocity, tracks, predictions)
            self.details["hold_conflict"] = hold_conflict
            checked.append((float(ego.velocity), hold_conflict))
            if hold_conflict is None:
                self.details.update(decision="GO", target=float(ego.velocity),
                                    cause=("second_merge_rear_safe_hold" if zone == 'SECOND_ENTRY'
                                           else 'first_merge_rear_safe_hold'),
                                    requested_conflict=collision, conflict=None,
                                    speed_floor=float(ego.velocity))
                return float(ego.velocity)
        # Search from the fastest candidate down. Every candidate includes
        # actuator delay and finite braking, not an instantaneous speed change.
        target = requested - p.MERGE_SPEED_SEARCH_STEP
        while target > 0.0:
            conflict = self._conflict(ego, points, cumulative, target, tracks, predictions)
            if conflict is None:
                self._remember_second_yield(collision, tracks, ego_match)
                releasing_yield = self.yield_speed is not None and target > self.yield_speed
                if releasing_yield:
                    recovered = min(target, max(self.yield_speed, max(0., ego.velocity))
                                    + p.MERGE_ACC_RECOVERY_RATE * self.planner.control_dt)
                    if self._conflict(ego, points, cumulative, recovered, tracks, predictions) is None:
                        target = recovered
                self.yield_speed = target
                self.details.update(decision="YIELD", target=target)
                if releasing_yield and target >= max(.05, ego.velocity):
                    self.details.update(brake_release_verified=True, brake_release_target=target)
                if second_entered and target > profile_requested:
                    self.details["speed_floor"] = target
                return target
            checked.append((target, conflict))
            target -= p.MERGE_SPEED_SEARCH_STEP
        if zone in ("FIRST_PREDICTIVE", "SECOND_ENTRY"):
            # Exhausting the moving candidates is not proof that stopping is
            # safe. Expose an unavoidable/fallback stop separately for diagnosis.
            stop_conflict = self._conflict(ego, points, cumulative, 0.0, tracks, predictions)
            self.details.update(stop_conflict=stop_conflict,
                                cause="no_safe_speed" if stop_conflict else "validated_stop")
            if stop_conflict and stop_conflict.get("rear"):
                mitigation = self._rear_mitigation(ego, points, cumulative, requested,
                    forward_speed_limit, tracks, predictions, stop_conflict)
                if mitigation is not None:
                    return mitigation
                # No collision-free solution: choose a checked non-accelerating
                # rear-risk mitigation only if stopping predicts earlier impact.
                # Apply on the approach too: the old centre-radius boundary
                # excluded Ego 4.5 m before it and forced a worse rear STOP.
                # Do not call it GO or weaken a front/crossing emergency stop.
                current = min(max(0.0, ego.velocity), forward_speed_limit)
                if current > 0.05:
                    current_conflict = self._conflict(ego, points, cumulative, current, tracks, predictions)
                    if current_conflict is None:
                        self.details.update(decision="GO", target=current, speed_floor=current,
                                            cause=("second_merge_rear_safe_hold" if zone == 'SECOND_ENTRY'
                                                   else 'first_merge_rear_safe_hold'), conflict=None)
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
                                        cause=("second_merge_rear_risk_mitigation" if zone == 'SECOND_ENTRY'
                                               else 'first_merge_rear_risk_mitigation'),
                                        requested_conflict=collision, conflict=hazard)
                    return speed
        self._remember_second_yield(collision, tracks, ego_match)
        self.yield_speed = 0.0
        self.details.update(decision="STOP", target=0.0)
        return 0.0
