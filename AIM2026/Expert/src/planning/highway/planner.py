#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lane-link based highway behavior and trajectory planner."""

import math
import time

import numpy as np

from ...interfaces.geometry import Point
from . import parameters
from .lane_map import HighwayLaneMap
from .merge_safety import MergeSafety
from .lane_change_safety import LaneChangeSafety
from .obstacle_motion import ObstacleMotion
from .entry_merge import EntryMerge
from .static_corridor import HighwayStaticCorridor
from .structures import HighwayCandidatePath, HighwayPlanningResult


from .merge_speed import MergeSpeedMixin
from .predictive_merge import PredictiveMergeMixin
from .lane_change import LaneChangeMixin
from .lane_following import LaneFollowingMixin
from .trajectory import TrajectoryMixin
from .collision_prediction import CollisionPredictionMixin


class HighwayLanePlanner(
    MergeSpeedMixin,
    PredictiveMergeMixin,
    LaneChangeMixin,
    LaneFollowingMixin,
    TrajectoryMixin,
    CollisionPredictionMixin,
):
    KEEP = "KEEP_LANE"
    CHANGE_LEFT = "CHANGE_LEFT"
    CHANGE_RIGHT = "CHANGE_RIGHT"
    ACC = "ACC_FOLLOW"

    def __init__(self, map_data_dir=None, lane_map=None):
        self.lane_map = lane_map or HighwayLaneMap(map_data_dir)
        self.control_dt = 1.0 / parameters.PLANNER_UPDATE_RATE
        self.state = self.KEEP
        self.target_link_id = None
        self.source_link_id = None
        self.active_candidate = None
        self.changed_lane_hold = False
        self._lane_change_cooldown_origin = None
        self._last_lane_change_direction = None
        self._merge_speed_limit_state = float("inf")
        self._last_merge_debug = []
        self._last_lane_checks = []
        self._merge_decisions = {}
        self._merge_go_object_ids = set()
        self._merge_zone_priority = None
        self._merge_zone_other_key = None
        self._merge_zone_clear_cycles = 0
        self._predictive_merge_priority = None
        self._predictive_merge_clear_cycles = 0
        self._predictive_merge_other_key = None
        self._predictive_merge_last_speed = 0.0
        self._predictive_merge_last_clear_time = float("inf")
        self._predictive_merge_dropout_hold_time = 0.0
        self.merge_safety = MergeSafety(self)
        self._merge_guard_active = False
        self._front_lead_speed = None
        self.obstacle_motion = ObstacleMotion()
        self.lane_change_safety = LaneChangeSafety(self)
        self.entry_merge = EntryMerge(self)
        self.static_corridor = None
        self._requested_speed = 0.0
        self.search_time_budget = None
        self._search_deadline = None

    def set_control_dt(self, control_dt):
        control_dt = float(control_dt)
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self.control_dt = control_dt

    def search_timed_out(self):
        return self._search_deadline is not None and time.monotonic() >= self._search_deadline

    def _cycles_for_reference_count(self, reference_cycles):
        duration = float(reference_cycles) / parameters.PLANNER_UPDATE_RATE
        return max(1, int(math.ceil(duration / self.control_dt - 1e-9)))

    def plan(self, vehicle_state, reference_path, vehicle_entries, desired_velocity=None,
             prediction_path=None, allow_lane_change=True):
        # Reuse NPC trajectories across candidate speeds/lanes within this
        # observation only; never retain stale predictions into the next tick.
        self.lane_change_safety.predictions.clear()
        self._search_deadline = None
        ego_match = self._resolve_ego_lane(vehicle_state)
        if ego_match is None:
            self._reset()
            return HighwayPlanningResult(
                reference_path,
                mode="highway_lane",
                reason="highway_lane_match_failed",
            )

        current_link = ego_match.lane_link
        self.target_link_id = current_link.id
        self._requested_speed = max(0.0, float(desired_velocity)) if desired_velocity is not None else max(
            vehicle_state.velocity, parameters.LANE_CHANGE_LAUNCH_SPEED
        )
        remaining_active_candidate = None
        if self.state in (self.CHANGE_LEFT, self.CHANGE_RIGHT) and self.active_candidate is not None:
            remaining_active_candidate = self._remaining_active_candidate(vehicle_state)
        self.obstacle_motion.update(vehicle_entries, self.control_dt)
        object_matches = self._match_objects(vehicle_entries)
        predicted_paths = self._prediction_paths(object_matches)
        entry_result = self.entry_merge.plan(vehicle_state, ego_match, vehicle_entries,
                                              object_matches, allow_lane_change)
        if entry_result is not None:
            return entry_result
        keep_candidate = self._build_candidate(
            vehicle_state, current_link, ego_match, direction="keep"
        )
        static_path = keep_candidate.points if allow_lane_change else (prediction_path or reference_path)
        static_distance = self.lane_change_safety.static_distance(static_path, vehicle_entries)
        static_avoidance = math.isfinite(static_distance)
        nearest_distance, ttc = self._front_risk(
            vehicle_state, ego_match, object_matches, vehicle_entries
        )
        if static_distance < nearest_distance:
            nearest_distance = static_distance
            self._front_lead_speed = 0.0
            ttc = min(ttc, static_distance / max(0.1, vehicle_state.velocity))
        acc_target_velocity = self._acc_speed_limit(
            vehicle_state.velocity, nearest_distance, ttc, self._front_lead_speed
        )
        if static_avoidance:
            # Slow early while room remains for a passing vehicle to clear.
            # This is a ceiling, never a positive speed floor near the obstacle.
            available = max(0.0, static_distance - parameters.ACC_STANDSTILL_DISTANCE)
            comfortable_stop_speed = math.sqrt(2.0 * parameters.STATIC_WAIT_DECEL * available)
            acc_target_velocity = min(acc_target_velocity, comfortable_stop_speed)
        front_acc_target_velocity = acc_target_velocity
        slower_lead = (
            self._front_lead_speed is not None
            and self._requested_speed - self._front_lead_speed >= parameters.LANE_CHANGE_SPEED_ADVANTAGE
        )
        proactive_overtake = slower_lead and nearest_distance <= parameters.LANE_CHANGE_TRIGGER_DISTANCE
        merge_reference = prediction_path if prediction_path is not None else reference_path
        if self.state in (self.CHANGE_LEFT, self.CHANGE_RIGHT) and self.active_candidate is not None:
            merge_reference = self.active_candidate.points
        elif self.changed_lane_hold or static_avoidance:
            # Keep predicting the lane we actually command after completion.
            # The global route still follows the source lane; reconnecting to
            # it would falsely predict a turn back into the passed obstacle.
            merge_reference = keep_candidate.points
        raw_merge_acc_target_velocity = self._merge_acc_speed_limit(
            vehicle_state, ego_match, object_matches, vehicle_entries,
            reference_path=merge_reference,
            desired_velocity=(min(desired_velocity, front_acc_target_velocity)
                              if desired_velocity is not None else None),
            forward_speed_limit=front_acc_target_velocity,
        )
        merge_acc_target_velocity = self._stabilize_merge_speed_limit(
            raw_merge_acc_target_velocity, vehicle_state.velocity
        )
        merge_conflict = not math.isinf(merge_acc_target_velocity)
        # A source-path ACC/merge ceiling is not a veto on a faster lane.
        # Overtakes in merge zones must pass the full merge prediction horizon.
        merge_source_lead = self._merge_conflict_is_source_lead(ego_match, object_matches)
        merge_allows_overtake = (math.isinf(merge_acc_target_velocity) or merge_source_lead
                                or (self._merge_guard_active and self.merge_safety.details.get("decision") == "GO"))
        second_merge_distance = vehicle_state.position.distance(
            Point(*parameters.CONSERVATIVE_MERGE_POINT)
        )
        second_merge_lane_preference = (
            parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS
            < second_merge_distance
            <= parameters.SECOND_MERGE_LANE_PREFERENCE_RADIUS
            and current_link.section_index in parameters.SECOND_MERGE_LANE_PREFERENCE_SECTIONS
            and current_link.lane_number in (2, 3, 4)
        )
        second_merge_middle_lane = (
            second_merge_lane_preference
            and current_link.lane_number == 3
        )

        second_merge_leave_right = (
            second_merge_lane_preference
            and current_link.lane_number
            not in parameters.SECOND_MERGE_PREFERRED_LANE_NUMBERS
        )
        preference_merge_go = (second_merge_leave_right and self._merge_guard_active
                               and self.merge_safety.details.get("decision") == "GO"
                               and self.merge_safety.details.get("conflict") is None)
        source_path_escape = (static_avoidance or (proactive_overtake and merge_allows_overtake)
                              or preference_merge_go) and allow_lane_change
        acc_target_velocity = min(
            acc_target_velocity, merge_acc_target_velocity
        )

        self._evaluate_dynamic_collision(
            keep_candidate, vehicle_state, object_matches
        )
        self._last_lane_checks = []
        if proactive_overtake and not static_avoidance:
            self._last_lane_checks.append("overtake lead={:.1f}m/s desired={:.1f}m/s gap={:.1f}m".format(
                self._front_lead_speed, self._requested_speed, nearest_distance))
            if merge_source_lead:
                self._last_lane_checks.append("source-lane following limit; checking adjacent paths")
        debug_details = {
            "ego_speed": float(vehicle_state.velocity),
            "ego_lane": current_link.lane_number,
            "ego_s": float(ego_match.s),
            "total_vehicle_objects": len(vehicle_entries),
            "matched_vehicle_objects": len(object_matches),
            "front_acc_limit": front_acc_target_velocity,
            "raw_merge_limit": raw_merge_acc_target_velocity,
            "filtered_merge_limit": merge_acc_target_velocity,
            "merge_conflict_is_source_lead": merge_source_lead,
            "keep_valid": keep_candidate.valid,
            "keep_collision_time": getattr(
                keep_candidate, "collision_time", float("inf")
            ),
            "merge_checks": list(getattr(self, "_last_merge_debug", [])),
            "lane_checks": self._last_lane_checks,
            "static_obstacle_distance": static_distance,
            "obstacle_motion": [dict(
                object_id=getattr(entry["object_info"], "unique_id", None),
                state=self.obstacle_motion.states[id(entry["object_info"])]["state"],
                stopped_seconds=self.obstacle_motion.states[id(entry["object_info"])]["seconds"],
            ) for entry in vehicle_entries],
            "merge_zone_priority": self._merge_zone_priority or "NONE",
            "merge_zone_priorities": {
                "FIRST_PREDICTIVE": self._predictive_merge_priority or "NONE",
                "SECOND_ENTRY": self._merge_zone_priority or "NONE",
            },
            "merge_zone_center": (
                parameters.CONSERVATIVE_MERGE_POINT[0],
                parameters.CONSERVATIVE_MERGE_POINT[1],
                parameters.CONSERVATIVE_MERGE_POINT_Z,
            ),
            "merge_zone_radius": parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
        }

        common = dict(
            mode="highway_lane",
            nearest_vehicle_distance=nearest_distance,
            ttc=ttc,
            current_link_id=current_link.id,
            target_link_id=self.target_link_id or current_link.id,
            highway_lane_links=self.lane_map.links.values(),
            object_lane_matches=object_matches,
            predicted_object_paths=predicted_paths,
            debug_details=debug_details,
        )

        if self._merge_guard_active:
            safety = self.merge_safety.details
            debug_details["merge_safety"] = {key: value for key, value in safety.items()
                                             if key != "predicted_paths"}
            common["predicted_object_paths"] = safety.get("predicted_paths", [])
            # These paths also include unmatched/held tracks, so they must not
            # be zipped against the ordinary lane-match list by the renderer.
            common["object_lane_matches"] = []
            if (self.state not in (self.CHANGE_LEFT, self.CHANGE_RIGHT)
                    and not current_link.excluded and not source_path_escape):
                limited = not math.isinf(acc_target_velocity)
                self.state = self.ACC if limited else self.KEEP
                return HighwayPlanningResult(
                    keep_candidate.points if self.changed_lane_hold else reference_path,
                    active=self.changed_lane_hold,
                    obstacle_detected=limited, acc_override=limited,
                    acc_target_velocity=acc_target_velocity, highway_state=self.state,
                    candidates=[keep_candidate],
                    reason=("highway_lane_emergency_stop" if acc_target_velocity <= 0.05
                            else "highway_lane_merge_acc" if limited
                            else "highway_lane_merge_clear"),
                    **common
                )

        if not allow_lane_change:
            limited = math.isfinite(acc_target_velocity)
            self.state = self.ACC if limited else self.KEEP
            return HighwayPlanningResult(
                reference_path, obstacle_detected=limited, acc_override=limited,
                acc_target_velocity=acc_target_velocity, highway_state=self.state,
                candidates=[keep_candidate], reason="highway_lane_follow_only", **common
            )

        if self.state in (self.CHANGE_LEFT, self.CHANGE_RIGHT):
            heading_aligned = abs(self.lane_map._angle_difference(vehicle_state.yaw, ego_match.yaw)) < math.radians(5.0)
            corridor_complete = (not getattr(self.active_candidate, 'static_corridor', False)
                                 or vehicle_state.position.distance(self.active_candidate.transition_end) < 1.0)
            if ego_match.distance <= parameters.MANEUVER_COMPLETE_DISTANCE and heading_aligned and corridor_complete:
                completed_direction = self.state
                self.state = self.KEEP
                self.source_link_id = None
                self.active_candidate = None
                self.changed_lane_hold = True
                self._last_lane_change_direction = completed_direction
                self._lane_change_cooldown_origin = Point(
                    vehicle_state.position.x, vehicle_state.position.y
                )
            else:
                active_candidate = remaining_active_candidate
                if active_candidate is None:
                    active_candidate = keep_candidate
                else:
                    change_entries, change_matches = self._lane_change_traffic(vehicle_entries, object_matches)
                    if getattr(active_candidate, 'static_corridor', False):
                        self.static_corridor.validate_geometry(active_candidate, change_entries)
                    self._evaluate_lane_change(active_candidate, vehicle_state,
                                               change_entries, change_matches, committed=True)
                    conflict = getattr(active_candidate, "prediction_conflict", None)
                    if getattr(active_candidate, "prediction_conflict", None) and conflict and conflict.get("rear"):
                        aborted = self._safe_return_to_source(vehicle_state, change_entries, change_matches)
                        if aborted is not None:
                            active_candidate = aborted
                            common["target_link_id"] = self.target_link_id
                            debug_details["returned_to_source_for_rear_vehicle"] = True
                debug_details["lane_change_conflict"] = getattr(active_candidate, "prediction_conflict", None)
                self._describe_static_change(active_candidate, debug_details)
                if getattr(active_candidate, "lane_change_verified", False) and (
                        getattr(active_candidate, "static_escape", False)
                        or getattr(active_candidate, "minimum_prediction_time", 0.0) >= parameters.MERGE_SAFETY_HORIZON):
                    # The active escape was checked against all current and
                    # held merge traffic at its own speed; discard keep-path ACC.
                    debug_details["filtered_merge_limit"] = float("inf")
                return self._result_during_change(
                    active_candidate,
                    nearest_distance,
                    ttc,
                    acc_target_velocity,
                    common,
                )

        lane_change_cooldown_distance = 0.0
        lane_change_cooldown_active = False
        if self._lane_change_cooldown_origin is not None:
            lane_change_cooldown_distance = vehicle_state.position.distance(
                self._lane_change_cooldown_origin
            )
            if (
                lane_change_cooldown_distance
                < parameters.LANE_CHANGE_COOLDOWN_DISTANCE
            ):
                lane_change_cooldown_active = True
            else:
                self._lane_change_cooldown_origin = None
                self._last_lane_change_direction = None
        debug_details["lane_change_cooldown_remaining"] = (
            max(
                0.0,
                parameters.LANE_CHANGE_COOLDOWN_DISTANCE
                - lane_change_cooldown_distance,
            )
            if lane_change_cooldown_active
            else 0.0
        )

        keep_collision_time = getattr(
            keep_candidate, "collision_time", float("inf")
        )
        if (
            (
                (merge_conflict and not source_path_escape)
                or self._merge_zone_priority == "OTHER"
                or second_merge_middle_lane
            )
            and
            not keep_candidate.valid
            and keep_collision_time
            <= parameters.MERGE_EMERGENCY_COLLISION_TIME
        ):
            self.state = self.ACC
            return HighwayPlanningResult(
                (
                    keep_candidate.points
                    if self.changed_lane_hold
                    else reference_path
                ),
                obstacle_detected=True,
                acc_override=True,
                acc_target_velocity=0.0,
                highway_state=self.ACC,
                candidates=[keep_candidate],
                reason="highway_lane_emergency_stop",
                **common
            )

        force_leave_excluded = current_link.excluded
        front_braking_needed = (
            not math.isinf(front_acc_target_velocity)
            and front_acc_target_velocity
            < vehicle_state.velocity - 0.5
        )
        proactive_lane_change = (
            static_avoidance
            or ttc <= parameters.LANE_CHANGE_TRIGGER_TTC
            or (
                nearest_distance <= parameters.LANE_CHANGE_TRIGGER_DISTANCE
                and (front_braking_needed or slower_lead)
            )
            or (
                vehicle_state.velocity <= parameters.ACC_STOPPED_SPEED
                and nearest_distance
                <= parameters.ACC_LANE_CHANGE_MIN_DISTANCE
            )
            or (
                self.state == self.ACC
                and nearest_distance
                <= parameters.LANE_CHANGE_TRIGGER_DISTANCE
            )
        )
        safety_avoidance_needed = (
            proactive_lane_change or merge_conflict or not keep_candidate.valid
        )
        avoidance_needed = (
            safety_avoidance_needed or second_merge_leave_right
        )
        lane_follow_path = (
            keep_candidate.points if self.changed_lane_hold or static_avoidance else reference_path
        )
        if (
            keep_candidate.valid
            and not force_leave_excluded
            and not avoidance_needed
        ):
            self.state = self.KEEP
            return HighwayPlanningResult(
                lane_follow_path,
                active=self.changed_lane_hold,
                highway_state=self.KEEP,
                candidates=[keep_candidate] if self.changed_lane_hold else [],
                reason=(
                    "highway_lane_hold_changed_lane"
                    if self.changed_lane_hold
                    else "highway_lane_no_obstacle_global_path"
                ),
                **common
            )

        # Preserve merge priority. Ordinary overtakes use the preparation gate;
        # confirmed static obstructions use the full swept-path safety check.
        if avoidance_needed and not force_leave_excluded:
            # Merge speed is computed from arrival order. Do not apply the
            # lane-change preparation cap here: MERGE_GO must clear the merge
            # at normal planned speed, while MERGE_YIELD uses its own limit.
            if merge_conflict and not source_path_escape:
                self.state = self.ACC
                return HighwayPlanningResult(
                    lane_follow_path,
                    obstacle_detected=True,
                    acc_override=True,
                    acc_target_velocity=acc_target_velocity,
                    highway_state=self.ACC,
                    candidates=[keep_candidate],
                    reason="highway_lane_merge_acc",
                    **common
                )
            if lane_change_cooldown_active and not static_avoidance and not proactive_overtake:
                if math.isinf(acc_target_velocity) and not keep_candidate.valid:
                    acc_target_velocity = min(
                        vehicle_state.velocity,
                        parameters.ACC_UNMATCHED_HAZARD_SPEED,
                    )
                has_speed_limit = not math.isinf(acc_target_velocity)
                self.state = self.ACC if has_speed_limit else self.KEEP
                return HighwayPlanningResult(
                    lane_follow_path,
                    obstacle_detected=safety_avoidance_needed,
                    acc_override=has_speed_limit,
                    acc_target_velocity=acc_target_velocity,
                    highway_state=self.state,
                    candidates=[keep_candidate],
                    reason="highway_lane_change_cooldown",
                    **common
                )
            if math.isinf(acc_target_velocity) and not keep_candidate.valid:
                acc_target_velocity = min(
                    vehicle_state.velocity,
                    parameters.ACC_UNMATCHED_HAZARD_SPEED,
                )
            # Do not slow down merely to enter a lane change. If the current
            # gap/TTC is safe, evaluate the adjacent lanes immediately. ACC is
            # retained only when the maneuver cannot safely start.
            acc_ready = self._ready_for_lane_change(
                vehicle_state.velocity, nearest_distance, ttc
            )
            if not acc_ready and not static_avoidance:
                self.state = self.ACC
                return HighwayPlanningResult(
                    lane_follow_path,
                    obstacle_detected=True,
                    acc_override=True,
                    acc_target_velocity=acc_target_velocity,
                    highway_state=self.ACC,
                    candidates=[keep_candidate],
                    reason="highway_lane_acc_prepare",
                    **common
                )

        # Optional new-maneuver search must not hold an old driving command
        # for seconds. Mandatory current-path/committed-maneuver checks above
        # stay complete; an unverified timed-out candidate is never selected.
        self._search_deadline = (None if self.search_time_budget is None else
                                 time.monotonic() + self.search_time_budget)
        candidates = [keep_candidate]
        valid_changes = []
        cooldown_blocked = False
        change_entries, change_matches = self._lane_change_traffic(vehicle_entries, object_matches)
        neighbors = self.lane_map.legal_neighbors(current_link)
        if not neighbors:
            self._last_lane_checks.append("no legal adjacent lane on link={}".format(current_link.id))
        for direction, target_link in neighbors:
            change_state = self.CHANGE_LEFT if direction == "left" else self.CHANGE_RIGHT
            if (lane_change_cooldown_active and not static_avoidance
                    and change_state != self._last_lane_change_direction):
                cooldown_blocked = True
                self._last_lane_checks.append("{} blocked-return-cooldown".format(direction))
                continue
            distances = (parameters.LANE_CHANGE_DISTANCE,)
            if static_avoidance:
                distances += parameters.STATIC_CHANGE_DISTANCES
            for distance in distances:
                if self.search_timed_out():
                    break
                candidate = self._build_candidate(
                    vehicle_state, target_link, self.lane_map.project(vehicle_state.position, target_link),
                    direction=direction, source_link=current_link, transition_distance=distance,
                    prediction_horizon=(parameters.MERGE_SAFETY_HORIZON
                                        if self._merge_guard_active and not static_avoidance else 0.0),
                )
                candidate.static_escape = static_avoidance
                candidate.compact_static = static_avoidance and distance < parameters.LANE_CHANGE_DISTANCE
                candidate.speed_limit = float("inf")
                if candidate.compact_static:
                    candidate.speed_limit = self._static_change_speed_limit(candidate.curvature_cost)
                    if candidate.curvature_cost > parameters.STATIC_CHANGE_MAX_CURVATURE:
                        candidate.valid = False
                        candidate.cost = float("inf")
                        self._last_lane_checks.append("{} blocked-curvature length={:.0f}m".format(direction, distance))
                self._evaluate_lane_change(candidate, vehicle_state, change_entries, change_matches)
                if not self._target_lane_has_clearance(
                    target_link, candidate, change_matches,
                    min(vehicle_state.velocity, self._requested_speed, candidate.speed_limit),
                ):
                    candidate.valid = False
                    candidate.cost = float("inf")
                candidates.append(candidate)
                if candidate.valid:
                    valid_changes.append(candidate)
                    # Longer, faster paths are preferred. Only shorten when
                    # the source obstruction makes the normal path infeasible.
                    break

        if not valid_changes and static_avoidance and not self.search_timed_out():
            # The ordinary highway choices above retain priority. Only a
            # confirmed static obstruction enables the static-zone search.
            if self.static_corridor is None:
                self.static_corridor = HighwayStaticCorridor(self)
            extensions = self.static_corridor.candidates(
                vehicle_state, current_link, change_entries, change_matches)
            candidates.extend(extensions)
            valid_changes.extend(candidate for candidate in extensions if candidate.valid)
            debug_details['static_corridor_search'] = self.static_corridor.status

        debug_details['lane_search_budget_exhausted'] = self.search_timed_out()
        if not valid_changes:
            if self.search_timed_out() and static_avoidance:
                acc_target_velocity = min(acc_target_velocity, max(0.0, vehicle_state.velocity))
                self._last_lane_checks.append("search budget exhausted; keep current-lane braking/ACC")
            if second_merge_leave_right and not safety_avoidance_needed:
                self.state = self.KEEP
                return HighwayPlanningResult(
                    lane_follow_path,
                    active=self.changed_lane_hold,
                    highway_state=self.KEEP,
                    candidates=candidates,
                    reason="highway_lane_no_obstacle_global_path",
                    **common
                )
            self.state = self.ACC
            has_speed_limit = not math.isinf(acc_target_velocity)
            return HighwayPlanningResult(
                lane_follow_path,
                active=self.changed_lane_hold or static_avoidance,
                obstacle_detected=not keep_candidate.valid,
                acc_override=has_speed_limit,
                acc_target_velocity=acc_target_velocity,
                highway_state=self.ACC,
                candidates=candidates,
                reason=("highway_static_wait_for_gap" if static_avoidance
                        else "highway_lane_change_cooldown" if cooldown_blocked
                        else "highway_lane_no_safe_change"),
                **common
            )

        if second_merge_lane_preference:
            # Safety/legality are checked above. Rank all remaining options,
            # including a safe right-lane escape if the preferred lanes are busy.
            # Store the actual selection cost so candidate logs show the bias.
            for candidate in candidates:
                candidate.lane_preference_cost = (
                    0.0 if candidate.target_lane in parameters.SECOND_MERGE_PREFERRED_LANE_NUMBERS
                    else parameters.SECOND_MERGE_RIGHT_LANE_COST_PENALTY)
                candidate.cost += candidate.lane_preference_cost
            debug_details["lane_preference"] = {
                "preferred_lanes": parameters.SECOND_MERGE_PREFERRED_LANE_NUMBERS,
                "right_lane_penalty": parameters.SECOND_MERGE_RIGHT_LANE_COST_PENALTY,
            }
        best = min(valid_changes, key=lambda candidate: candidate.cost)
        self.source_link_id = current_link.id
        self.target_link_id = best.target_link_id
        self.active_candidate = best
        self.state = self.CHANGE_LEFT if best.direction == "left" else self.CHANGE_RIGHT
        common["target_link_id"] = self.target_link_id
        self._describe_static_change(best, debug_details)
        self._describe_change_speed(best, debug_details)
        speed_limit = getattr(best, "safety_speed", float("inf"))
        return HighwayPlanningResult(
            best.points,
            active=True,
            obstacle_detected=safety_avoidance_needed,
            # Apply the verified gap speed throughout the lateral maneuver.
            acc_override=math.isfinite(speed_limit),
            acc_target_velocity=speed_limit,
            highway_state=self.state,
            selected_offset=best.offset,
            candidates=candidates,
            reason=(
                "highway_lane_second_merge_prefer_left"
                if (
                    second_merge_leave_right
                    and not safety_avoidance_needed
                    and best.direction == "left"
                )
                else "highway_lane_safe_change"
            ),
            **common
        )


    def reset_lateral_state(self):
        self.entry_merge.reset()
        if self.static_corridor is not None:
            self.static_corridor.reset()
        self.state = self.KEEP
        self.target_link_id = None
        self.source_link_id = None
        self.active_candidate = None
        self.changed_lane_hold = False
        self._lane_change_cooldown_origin = None
        self._last_lane_change_direction = None

    def _reset(self):
        self.reset_lateral_state()
        self.obstacle_motion.reset()
        self.merge_safety.reset()
        self._merge_guard_active = False
        self._merge_speed_limit_state = float("inf")
        self._merge_decisions = {}
        self._merge_go_object_ids = set()
        self._merge_zone_priority = None
        self._merge_zone_other_key = None
        self._merge_zone_clear_cycles = 0
        self._predictive_merge_priority = None
        self._predictive_merge_clear_cycles = 0
        self._predictive_merge_other_key = None
        self._predictive_merge_last_speed = 0.0
        self._predictive_merge_last_clear_time = float("inf")
        self._predictive_merge_dropout_hold_time = 0.0

    def reset(self):
        self._reset()
