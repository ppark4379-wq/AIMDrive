"""차선 변경 후보의 시작·여유·복귀·진행 상태 처리.

HighwayLanePlanner의 동일 인스턴스 상태를 사용하는 메서드 묶음이다.
생성자나 별도 상태를 추가하지 않으며, 기존 메서드 본문과 호출 순서를 유지한다.
"""
import math
from . import parameters
from .structures import HighwayPlanningResult


class LaneChangeMixin:
    @staticmethod
    def _ready_for_lane_change(ego_speed, gap, ttc):
        if ego_speed <= parameters.ACC_STOPPED_SPEED:
            return True
        gap_safe = gap >= parameters.ACC_LANE_CHANGE_MIN_DISTANCE
        ttc_safe = math.isinf(ttc) or ttc >= parameters.ACC_LANE_CHANGE_MIN_TTC
        return gap_safe and ttc_safe

    def _target_lane_has_clearance(
        self, target_link, candidate, object_matches, ego_speed
    ):
        if not candidate.points:
            return False
        ego_projection = self.lane_map.project(candidate.points[0], target_link)
        for entry, object_match in object_matches:
            if self.obstacle_motion.is_static(entry["object_info"]):
                # The swept path already checks its actual body. Do not turn
                # a harmless shoulder object into a centered lead vehicle.
                continue
            same_target_link = object_match.lane_link.id == target_link.id
            if same_target_link:
                signed_distance = object_match.s - ego_projection.s
            else:
                # A lane can end shortly after the lateral maneuver. Vehicles
                # on its connected successor are still directly in front of
                # the completed lane change and must block that candidate.
                signed_distance = self.lane_map.forward_distance(
                    ego_projection, object_match
                )
                if signed_distance is None:
                    behind = self.lane_map.forward_distance(object_match, ego_projection)
                    if behind is None:
                        continue
                    signed_distance = -behind
            object_speed = max(0.0, float(entry["object_info"].velocity))
            front_clearance = (
                parameters.TARGET_LANE_FRONT_CLEARANCE
                + max(0.0, ego_speed - object_speed)
                * parameters.TARGET_LANE_FRONT_TIME_GAP
            )
            rear_clearance = (
                parameters.TARGET_LANE_REAR_CLEARANCE
                + max(0.0, object_speed - ego_speed)
                * parameters.TARGET_LANE_REAR_TIME_GAP
            )
            if (self.lane_change_safety.first_merge_static_escape(candidate)
                    and getattr(candidate, "lane_change_verified", False)):
                front_clearance = (
                    parameters.FIRST_MERGE_STATIC_FRONT_BUMPER_GAP
                    + parameters.VEHICLE_LENGTH
                    + 0.5 * max(1.0, entry["object_info"].size_x)
                    + max(0.0, ego_speed - object_speed) * parameters.TARGET_LANE_FRONT_TIME_GAP
                )
                rear_clearance = (
                    parameters.FIRST_MERGE_STATIC_REAR_BUMPER_GAP
                    + 0.5 * max(1.0, entry["object_info"].size_x)
                    + max(0.0, object_speed - ego_speed)
                    * parameters.FIRST_MERGE_STATIC_REAR_TIME_GAP
                )
                if signed_distance < 0.0:
                    self._last_lane_checks.append(
                        "first-static rear-gap={:.1f} required={:.1f} closing={:.1f}".format(
                            -signed_distance, rear_clearance, max(0.0, object_speed-ego_speed)))
            if 0.0 <= signed_distance < front_clearance:
                self._last_lane_checks.append(
                    "{} blocked-front link={} gap={:.1f} required={:.1f} speed={:.1f}".format(
                        candidate.direction,
                        target_link.id,
                        signed_distance,
                        front_clearance,
                        object_speed,
                    )
                )
                return False
            if -rear_clearance < signed_distance < 0.0:
                self._last_lane_checks.append(
                    "{} blocked-rear link={} gap={:.1f} required={:.1f} speed={:.1f}".format(
                        candidate.direction,
                        target_link.id,
                        -signed_distance,
                        rear_clearance,
                        object_speed,
                    )
                )
                return False
        self._last_lane_checks.append(
            "{} clear link={}".format(candidate.direction, target_link.id)
        )
        return True

    def _remaining_active_candidate(self, vehicle_state):
        candidate = self.active_candidate
        if candidate is None or len(candidate.points) < 2:
            return None
        nearest_index = min(
            range(len(candidate.points)),
            key=lambda index: ((candidate.points[index].x - vehicle_state.position.x)**2
                               + (candidate.points[index].y - vehicle_state.position.y)**2),
        )
        start_index = max(0, nearest_index - 1)
        candidate.points = candidate.points[start_index:]
        # Preserve the verified lateral transition, but replenish its lane-follow
        # tail before safety prediction or Pure Pursuit can run out of points.
        needed = max(parameters.PATH_LOOKAHEAD_MIN,
                     max(vehicle_state.velocity, self._requested_speed)
                     * max(parameters.MERGE_SAFETY_HORIZON, parameters.LANE_CHANGE_SETTLE_TIME) + 20.0)
        remaining = self._cumulative_distance(candidate.points)[-1]
        if remaining < needed:
            endpoint = self._project_maneuver_route(candidate.points[-1], candidate.target_link_id)
            if endpoint is not None and endpoint.distance <= parameters.LANE_MATCH_MAX_DISTANCE:
                extension = self.lane_map.route_points(
                    endpoint, needed - remaining, parameters.PATH_SAMPLE_SPACING)
                for point, _, _ in extension[1:]:
                    if math.hypot(point.x-candidate.points[-1].x,
                                  point.y-candidate.points[-1].y) > .1:
                        candidate.points.append(point)
        if getattr(candidate, "compact_static", False):
            # Release the cap from bends already passed. Remaining bends still
            # limit speed and the full traffic prediction runs again below.
            candidate.speed_limit = self._static_change_speed_limit(self._max_curvature(candidate.points))
        candidate.valid = len(candidate.points) >= 2
        candidate.collision_time = float("inf")
        return candidate

    def _lane_change_traffic(self, entries, matches):
        # Preserve short dropouts handled by the merge guard even when a static
        # escape is evaluated independently of its source-lane speed ceiling.
        entries = list(entries)
        matches = list(matches)
        ids = {id(entry["object_info"]) for entry in entries}
        for obj, match, age in self.merge_safety.tracks.values():
            if id(obj) in ids:
                continue
            held = {"object_info": obj, "prediction_age": age}
            entries.append(held)
            if match is not None:
                matches.append((held, match))
        return entries, matches

    @staticmethod
    def _static_change_speed_limit(curvature):
        return min(parameters.STATIC_CHANGE_MAX_SPEED, math.sqrt(
            parameters.STATIC_CHANGE_LATERAL_ACCEL / max(curvature, 1e-6)))

    @staticmethod
    def _describe_static_change(candidate, details):
        if getattr(candidate, "static_escape", False):
            details["highway_static_escape"] = dict(
                compact=getattr(candidate, "compact_static", False),
                distance=getattr(candidate, "transition_distance", parameters.LANE_CHANGE_DISTANCE),
                speed_limit=getattr(candidate, "speed_limit", float("inf")),
                verified=getattr(candidate, "lane_change_verified", False),
                planner=('static_zone_corridor' if getattr(candidate, 'static_corridor', False)
                         else 'highway_lane'),
            )
            details["lane_checks"].append("static escape length={:.0f}m speed_limit={:.2f}m/s".format(
                details["highway_static_escape"]["distance"], details["highway_static_escape"]["speed_limit"]))

    def _safe_return_to_source(self, vehicle_state, entries, matches):
        source = self.lane_map.links.get(self.source_link_id)
        target = self.lane_map.links.get(self.target_link_id)
        if source is None or target is None:
            return None
        # An abort is also a lane change: do not cross a one-way boundary
        # back into the entrance lane, or jump over an intermediate lane.
        if source.id not in {link.id for _, link in self.lane_map.legal_neighbors(target)}:
            return None
        projection = self.lane_map.project(vehicle_state.position, source)
        if projection.distance > parameters.LANE_CHANGE_ABORT_MAX_LATERAL:
            return None
        direction = "right" if self.state == self.CHANGE_LEFT else "left"
        candidate = self._build_candidate(vehicle_state, source, projection, direction,
                                          self.lane_map.links.get(self.target_link_id),
                                          transition_distance=parameters.LANE_CHANGE_RETURN_DISTANCE)
        self._evaluate_lane_change(candidate, vehicle_state, entries, matches, committed=True)
        if not candidate.valid:
            return None
        old_target = self.target_link_id
        self.target_link_id = source.id
        self.source_link_id = old_target
        self.active_candidate = candidate
        self.state = self.CHANGE_LEFT if direction == "left" else self.CHANGE_RIGHT
        return candidate

    def _evaluate_lane_change(self, candidate, vehicle_state, entries, matches, committed=False):
        if not candidate.valid or len(candidate.points) < 2:
            return
        return self._evaluate_speed_hold(candidate, vehicle_state, entries, matches, committed)

    def _evaluate_speed_hold(self, candidate, ego, entries, matches, committed):
        """Do not slow to fit a gap; allow separately validated acceleration.

        The dedicated entry-merge follower keeps its separate policy. Geometry limits are
        admission checks, not a request to brake halfway through a lane change.
        """
        candidate.hold_change_speed = True
        if not hasattr(candidate, 'change_cruise_speed'):
            candidate.change_cruise_speed = (
                max(0., ego.velocity) if ego.velocity > parameters.ACC_STOPPED_SPEED
                else min(self._requested_speed, getattr(candidate, 'speed_limit', float('inf'))))
            candidate.change_entry_speed = candidate.change_cruise_speed
        speed = candidate.change_cruise_speed
        end = min(range(len(candidate.points)),
                  key=lambda i: candidate.points[i].distance(candidate.transition_end))
        curvature = self._max_curvature(candidate.points[:end+1])
        curve_limit = math.sqrt(parameters.STATIC_CHANGE_LATERAL_ACCEL / max(curvature, 1e-6))
        if speed <= .05 or speed > getattr(candidate, 'speed_limit', float('inf')) + 1e-6:
            conflict = dict(time=0., rear=False, kind='change_speed_not_supported', speed=speed)
        else:
            conflict = self.lane_change_safety.check(candidate, ego, entries, matches, speed,
                                                    committed=committed)
        if conflict is None and not committed:
            if speed**2 * curvature > parameters.STATIC_CHANGE_LATERAL_ACCEL:
                conflict = dict(time=0., rear=False, kind='change_speed_not_supported', speed=speed)
        candidate.acceleration_conflict = None
        if committed:
            # Entry speed is a lower reference, not a permanent low-speed cap.
            # Test acceleration on the SAME lateral path, against all front/rear
            # traffic and both acceleration envelopes, before raising its target.
            faster = min(self._requested_speed, curve_limit,
                         getattr(candidate, 'speed_limit', float('inf')))
            if faster > speed + .05:
                acceleration_conflict = self.lane_change_safety.check(
                    candidate, ego, entries, matches, faster, committed=True)
                candidate.acceleration_conflict = acceleration_conflict
                if acceleration_conflict is None:
                    speed = candidate.change_cruise_speed = faster
                    conflict = None
        # A rear-only conflict must not mask a simultaneous new front hazard.
        rear_conflict = conflict
        remaining = list(entries)
        while committed and conflict and conflict.get('rear'):
            identity = conflict.get('object_id')
            reduced = [e for e in remaining
                       if getattr(e['object_info'], 'unique_id', None) != identity]
            if identity is None or len(reduced) == len(remaining):
                break
            remaining = reduced
            conflict = self.lane_change_safety.check(candidate, ego, remaining, matches, speed,
                                                    committed=True)
        conflict = conflict or rear_conflict
        candidate.prediction_conflict = candidate.requested_conflict = conflict
        candidate.lane_change_verified = conflict is None
        candidate.safety_speed = speed if conflict is None else float('inf')
        candidate.collision_time = conflict['time'] if conflict else float('inf')
        if conflict:
            candidate.valid = False
            candidate.cost = float('inf')
            self._last_lane_checks.append('{} blocked-{} at held speed={:.2f}m/s'.format(
                candidate.direction, conflict['kind'], speed))

    @staticmethod
    def _describe_change_speed(candidate, details):
        if getattr(candidate, 'hold_change_speed', False):
            details['lane_change_cruise'] = dict(
                speed=candidate.change_cruise_speed,
                entry_speed=getattr(candidate, 'change_entry_speed', candidate.change_cruise_speed),
                acceleration_conflict=getattr(candidate, 'acceleration_conflict', None),
                verified=getattr(candidate, 'lane_change_verified', False),
                maintain_speed=(candidate.valid or bool(
                    (getattr(candidate, 'prediction_conflict', None) or {}).get('rear'))),
                conflict=getattr(candidate, 'prediction_conflict', None))

    def _result_during_change(
        self, candidate, nearest_distance, ttc, acc_target_velocity, common
    ):
        self._describe_change_speed(candidate, common['debug_details'])
        if getattr(candidate, 'hold_change_speed', False):
            conflict = getattr(candidate, 'prediction_conflict', None)
            if candidate.valid or (conflict and conflict.get('rear')):
                # The actual selected path has its own complete traffic check.
                # Old-lane ACC and generic merge pacing cannot lower this speed.
                return HighwayPlanningResult(
                    candidate.points, active=True, acc_override=True,
                    acc_target_velocity=candidate.change_cruise_speed,
                    highway_state=self.state, selected_offset=candidate.offset,
                    candidates=[candidate],
                    reason=('highway_lane_change_rear_risk' if conflict
                            else 'highway_lane_change_in_progress'), **common)
        collision_during_change = not candidate.valid
        urgent_front_acc = (
            not getattr(candidate, "lane_change_verified", False)
            and not math.isinf(acc_target_velocity)
            and (
                nearest_distance <= parameters.TARGET_LANE_FRONT_CLEARANCE
                or ttc <= parameters.ACC_LANE_CHANGE_MIN_TTC
            )
        )
        merge_acc = not math.isinf(
            common.get("debug_details", {}).get(
                "filtered_merge_limit", float("inf")
            )
        )
        checked_limit = getattr(candidate, "safety_speed", float("inf"))
        rear_conflict = (getattr(candidate, "prediction_conflict", None) or {}).get("rear", False)
        safety_acc = collision_during_change or urgent_front_acc or merge_acc or math.isfinite(checked_limit)
        collision_time = getattr(candidate, "collision_time", float("inf"))
        emergency_stop = (
            collision_during_change and not rear_conflict
            and collision_time <= parameters.MERGE_EMERGENCY_COLLISION_TIME
        )
        if rear_conflict and not math.isfinite(checked_limit) and not merge_acc and not urgent_front_acc:
            # Slowing on the committed path would increase this rear conflict;
            # retain the maneuver speed, including its curvature ceiling.
            lane_change_speed_limit = min(
                getattr(candidate, "speed_limit", float("inf")),
                getattr(candidate, "maneuver_speed_limit", float("inf")))
            safety_acc = math.isfinite(lane_change_speed_limit)
        elif emergency_stop:
            lane_change_speed_limit = 0.0
        elif math.isfinite(checked_limit):
            lane_change_speed_limit = min(checked_limit, acc_target_velocity) if merge_acc else checked_limit
        elif safety_acc:
            lane_change_speed_limit = min(getattr(candidate, "speed_limit", float("inf")), (
                acc_target_velocity
                if not math.isinf(acc_target_velocity)
                else parameters.ACC_UNMATCHED_HAZARD_SPEED
            ))
        else:
            # Once the selected path remains collision-free, do not retain the
            # source-lane lead vehicle's ACC limit during lateral motion.
            lane_change_speed_limit = float("inf")
        return HighwayPlanningResult(
            candidate.points,
            active=True,
            obstacle_detected=safety_acc,
            acc_override=safety_acc,
            acc_target_velocity=lane_change_speed_limit,
            # Keep the lateral maneuver state after commitment. ACC may cap
            # speed at the same time, but a newly observed vehicle must not
            # discard the active path or disable lane-change steering limits.
            highway_state=self.state,
            selected_offset=candidate.offset,
            candidates=[candidate],
            reason=(
                "highway_lane_change_emergency_stop"
                if emergency_stop
                else "highway_lane_change_rear_risk"
                if rear_conflict and not math.isfinite(checked_limit)
                else "highway_lane_change_acc"
                if safety_acc
                else "highway_lane_change_in_progress"
            ),
            **common
        )
