#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lane-link based highway behavior and trajectory planner."""

import math

import numpy as np

from ...localization.point import Point
from . import parameters
from .lane_map import HighwayLaneMap
from .structures import HighwayCandidatePath, HighwayPlanningResult


class HighwayLanePlanner:
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

    def set_control_dt(self, control_dt):
        control_dt = float(control_dt)
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self.control_dt = control_dt

    def _cycles_for_reference_count(self, reference_cycles):
        duration = float(reference_cycles) / parameters.PLANNER_UPDATE_RATE
        return max(1, int(math.ceil(duration / self.control_dt - 1e-9)))

    def plan(self, vehicle_state, reference_path, vehicle_entries):
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
        object_matches = self._match_objects(vehicle_entries)
        predicted_paths = self._prediction_paths(object_matches)
        nearest_distance, ttc = self._front_risk(
            vehicle_state, ego_match, object_matches, vehicle_entries
        )
        acc_target_velocity = self._acc_speed_limit(
            vehicle_state.velocity, nearest_distance, ttc
        )
        front_acc_target_velocity = acc_target_velocity
        raw_merge_acc_target_velocity = self._merge_acc_speed_limit(
            vehicle_state, ego_match, object_matches, vehicle_entries
        )
        merge_acc_target_velocity = self._stabilize_merge_speed_limit(
            raw_merge_acc_target_velocity, vehicle_state.velocity
        )
        merge_conflict = not math.isinf(merge_acc_target_velocity)
        acc_target_velocity = min(
            acc_target_velocity, merge_acc_target_velocity
        )

        keep_candidate = self._build_candidate(
            vehicle_state,
            current_link,
            ego_match,
            direction="keep",
        )
        self._evaluate_dynamic_collision(
            keep_candidate, vehicle_state, object_matches
        )
        self._last_lane_checks = []
        debug_details = {
            "ego_speed": float(vehicle_state.velocity),
            "ego_lane": current_link.lane_number,
            "ego_s": float(ego_match.s),
            "total_vehicle_objects": len(vehicle_entries),
            "matched_vehicle_objects": len(object_matches),
            "front_acc_limit": front_acc_target_velocity,
            "raw_merge_limit": raw_merge_acc_target_velocity,
            "filtered_merge_limit": merge_acc_target_velocity,
            "keep_valid": keep_candidate.valid,
            "keep_collision_time": getattr(
                keep_candidate, "collision_time", float("inf")
            ),
            "merge_checks": list(getattr(self, "_last_merge_debug", [])),
            "lane_checks": self._last_lane_checks,
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

        if self.state in (self.CHANGE_LEFT, self.CHANGE_RIGHT):
            if ego_match.distance <= parameters.MANEUVER_COMPLETE_DISTANCE:
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
                active_candidate = self._remaining_active_candidate(vehicle_state)
                if active_candidate is None:
                    active_candidate = keep_candidate
                else:
                    self._evaluate_dynamic_collision(
                        active_candidate, vehicle_state, object_matches
                    )
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

        second_merge_distance = vehicle_state.position.distance(
            Point(*parameters.CONSERVATIVE_MERGE_POINT)
        )
        second_merge_lane_preference = (
            parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS
            < second_merge_distance
            <= parameters.SECOND_MERGE_LANE_PREFERENCE_RADIUS
            and current_link.lane_number in (2, 3, 4)
        )
        second_merge_protect_middle = (
            second_merge_lane_preference
            and current_link.lane_number == 3
        )

        keep_collision_time = getattr(
            keep_candidate, "collision_time", float("inf")
        )
        if (
            (
                merge_conflict
                or self._merge_zone_priority == "OTHER"
                or second_merge_protect_middle
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
            ttc <= parameters.LANE_CHANGE_TRIGGER_TTC
            or (
                nearest_distance <= parameters.LANE_CHANGE_TRIGGER_DISTANCE
                and front_braking_needed
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
        second_merge_leave_right = (
            second_merge_lane_preference
            and current_link.lane_number
            not in parameters.SECOND_MERGE_PREFERRED_LANE_NUMBERS
        )
        avoidance_needed = (
            safety_avoidance_needed or second_merge_leave_right
        )
        lane_follow_path = (
            keep_candidate.points if self.changed_lane_hold else reference_path
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

        # On a normal highway lane, always stabilize longitudinal motion first.
        # A lane change is considered only on a later cycle after speed/gap/TTC
        # become safe. The excluded leftmost lane is left immediately instead.
        if avoidance_needed and not force_leave_excluded:
            # Merge speed is computed from arrival order. Do not apply the
            # lane-change preparation cap here: MERGE_GO must clear the merge
            # at normal planned speed, while MERGE_YIELD uses its own limit.
            if merge_conflict:
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
            if lane_change_cooldown_active:
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
            if not acc_ready:
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

        candidates = [keep_candidate]
        valid_changes = []
        for direction, target_link in self.lane_map.legal_neighbors(current_link):
            candidate = self._build_candidate(
                vehicle_state,
                target_link,
                self.lane_map.project(vehicle_state.position, target_link),
                direction=direction,
                source_link=current_link,
            )
            self._evaluate_dynamic_collision(
                candidate, vehicle_state, object_matches
            )
            if not self._target_lane_has_clearance(
                target_link,
                candidate,
                object_matches,
                vehicle_state.velocity,
            ):
                candidate.valid = False
                candidate.cost = float("inf")
            candidates.append(candidate)
            if candidate.valid:
                valid_changes.append(candidate)

        if second_merge_protect_middle:
            # In this section lane 3 is already a preferred lane. Never escape
            # from it into lane 4. If lane 2 is unavailable, stay in lane 3 and
            # let longitudinal ACC handle the obstacle instead.
            valid_changes = [
                candidate
                for candidate in valid_changes
                if candidate.target_lane
                in parameters.SECOND_MERGE_PREFERRED_LANE_NUMBERS
            ]

        if not valid_changes:
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
                obstacle_detected=not keep_candidate.valid,
                acc_override=has_speed_limit,
                acc_target_velocity=acc_target_velocity,
                highway_state=self.ACC,
                candidates=candidates,
                reason="highway_lane_no_safe_change",
                **common
            )

        if second_merge_lane_preference:
            # This is a soft lane ranking, not a prohibited-lane rule. Prefer
            # lanes 2/3 when safe, while retaining lane 4 as an escape option.
            best = min(
                valid_changes,
                key=lambda candidate: (
                    candidate.cost
                    + (
                        0.0
                        if candidate.target_lane
                        in parameters.SECOND_MERGE_PREFERRED_LANE_NUMBERS
                        else parameters.SECOND_MERGE_RIGHT_LANE_COST_PENALTY
                    )
                ),
            )
        else:
            best = min(valid_changes, key=lambda candidate: candidate.cost)
        self.source_link_id = current_link.id
        self.target_link_id = best.target_link_id
        self.active_candidate = best
        self.state = self.CHANGE_LEFT if best.direction == "left" else self.CHANGE_RIGHT
        common["target_link_id"] = self.target_link_id
        return HighwayPlanningResult(
            best.points,
            active=True,
            obstacle_detected=safety_avoidance_needed,
            # A safe lane change keeps the route planner's requested speed.
            # Dynamic collision/clearance checks above remain the safety gate.
            acc_target_velocity=float("inf"),
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
                    continue
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
            if same_target_link and -rear_clearance < signed_distance < 0.0:
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

    def _merge_acc_speed_limit(
        self, vehicle_state, ego_match, object_matches, vehicle_entries=()
    ):
        """Return a speed ceiling when two different links reach one merge."""
        ego_link = ego_match.lane_link
        self._last_merge_debug = []
        self._merge_go_object_ids = set()
        ego_distance = max(0.0, ego_link.length - ego_match.s)
        ego_speed = max(0.0, float(vehicle_state.velocity))

        ego_eta = ego_distance / max(0.1, ego_speed)
        speed_limit = float("inf")
        ego_successors = set(self.lane_map.successors.get(ego_link.id, ()))
        predictive_successors = ego_successors.intersection(
            parameters.PREDICTIVE_MERGE_SUCCESSORS
        )
        predictive_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        predictive_zone = (
            bool(predictive_successors)
            and vehicle_state.position.distance(predictive_point)
            <= parameters.PREDICTIVE_MERGE_EGO_RADIUS
        )
        if predictive_zone:
            return self._predictive_merge_zone_speed_limit(
                vehicle_state,
                ego_match,
                object_matches,
                vehicle_entries,
                predictive_successors,
                predictive_point,
            )
        if self._predictive_merge_priority is not None:
            self._predictive_merge_priority = None
            self._predictive_merge_clear_cycles = 0
            self._predictive_merge_other_key = None
            self._predictive_merge_last_speed = 0.0
            self._predictive_merge_last_clear_time = float("inf")
            self._predictive_merge_dropout_hold_time = 0.0

        conservative_successors = ego_successors.intersection(
            parameters.CONSERVATIVE_MERGE_SUCCESSORS
        )
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        conservative_zone = (
            bool(conservative_successors)
            and vehicle_state.position.distance(merge_point)
            <= parameters.CONSERVATIVE_MERGE_EGO_RADIUS
        )
        if conservative_zone:
            priority_limit = self._merge_zone_priority_speed_limit(
                vehicle_state,
                ego_link,
                object_matches,
                vehicle_entries,
                conservative_successors,
                merge_point,
            )
            if self._merge_zone_priority == "EGO":
                # Ego entered first, so merge-opponent OBB predictions must not
                # brake it into a side/rear collision. Same-lane front ACC and
                # non-merge safety logic remain active outside this override.
                self._merge_speed_limit_state = float("inf")
                return float("inf")
            if math.isinf(priority_limit):
                return float("inf")
            return priority_limit

        if self._merge_zone_priority is not None and (
            not conservative_successors
            or vehicle_state.position.distance(merge_point)
            > parameters.CONSERVATIVE_MERGE_EGO_RADIUS
        ):
            self._reset_merge_zone_priority()

        self._merge_decisions = {
            key: decision
            for key, decision in self._merge_decisions.items()
            if key[0] in ego_successors
        }
        if (
            parameters.MERGE_MIN_REMAINING_DISTANCE
            <= ego_distance
            <= parameters.MERGE_LOOKAHEAD_DISTANCE
        ):
            for entry, object_match in object_matches:
                object_link = object_match.lane_link
                if object_link.id == ego_link.id:
                    continue
                shared_successors = ego_successors.intersection(
                    self.lane_map.successors.get(object_link.id, ())
                )
                if not shared_successors:
                    continue

                object_distance = max(0.0, object_link.length - object_match.s)
                if object_distance > parameters.MERGE_LOOKAHEAD_DISTANCE:
                    self._last_merge_debug.append(
                        "graph object={} link={} outside distance={:.1f}m".format(
                            getattr(entry["object_info"], "name", "") or "unnamed",
                            object_link.id,
                            object_distance,
                        )
                    )
                    continue
                successor_id = sorted(shared_successors)[0]
                conservative_yield = (
                    conservative_zone
                    and successor_id in conservative_successors
                    and entry["object_info"].position.distance(merge_point)
                    <= parameters.CONSERVATIVE_MERGE_OBJECT_RADIUS
                )
                object_speed = max(0.0, float(entry["object_info"].velocity))
                if object_speed < 0.1 and not conservative_yield:
                    continue
                object_eta = object_distance / max(0.1, object_speed)
                object_key = (
                    successor_id,
                    getattr(entry["object_info"], "name", "")
                    or object_link.id,
                )
                decision = self._merge_decisions.get(object_key)
                if conservative_yield:
                    # This merge is intentionally safety-first: even if Ego's
                    # ETA is smaller, let the neighboring approach vehicle pass.
                    decision = "YIELD"
                    self._merge_decisions[object_key] = decision
                elif decision is None:
                    decision = (
                        "GO"
                        if ego_eta + parameters.MERGE_EGO_PRIORITY_MARGIN
                        < object_eta
                        else "YIELD"
                    )
                    self._merge_decisions[object_key] = decision

                if decision == "GO":
                    self._merge_go_object_ids.add(id(entry["object_info"]))
                    self._last_merge_debug.append(
                        "graph object={} link={} successor={} ego_eta={:.2f}s obj_eta={:.2f}s -> MERGE_GO locked".format(
                            getattr(entry["object_info"], "name", "") or "unnamed",
                            object_link.id,
                            successor_id,
                            ego_eta,
                            object_eta,
                        )
                    )
                    continue
                if (
                    not conservative_yield
                    and
                    abs(ego_eta - object_eta)
                    > parameters.MERGE_ARRIVAL_TIME_WINDOW
                ):
                    self._last_merge_debug.append(
                        "graph object={} link={} ego_eta={:.2f}s obj_eta={:.2f}s -> time-clear".format(
                            getattr(entry["object_info"], "name", "") or "unnamed",
                            object_link.id,
                            ego_eta,
                            object_eta,
                        )
                    )
                    continue

                yield_time = object_eta + parameters.MERGE_YIELD_TIME_GAP
                speed_limit = min(
                    speed_limit,
                    ego_distance / max(0.1, yield_time),
                )
                if conservative_yield:
                    speed_limit = min(
                        speed_limit,
                        parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                    )
                self._last_merge_debug.append(
                    "graph object={} link={} successor={} ego_eta={:.2f}s obj_eta={:.2f}s -> {} limit={:.2f}m/s".format(
                        getattr(entry["object_info"], "name", "") or "unnamed",
                        object_link.id,
                        successor_id,
                        ego_eta,
                        object_eta,
                        (
                            "SAFE_ZONE_YIELD"
                            if conservative_yield
                            else "MERGE_YIELD locked"
                        ),
                        speed_limit,
                    )
                )

        if conservative_zone:
            # Keep yielding while the other vehicle is crossing the beginning
            # of the shared successor. This closes the one-frame gap where its
            # lane match changes from an approach link to the merged link.
            matched_ids = {id(entry["object_info"]) for entry, _ in object_matches}
            for entry, object_match in object_matches:
                object_info = entry["object_info"]
                if object_match.lane_link.id not in conservative_successors:
                    continue
                if object_info.position.distance(merge_point) > parameters.CONSERVATIVE_MERGE_OBJECT_RADIUS:
                    continue
                if object_match.s > parameters.CONSERVATIVE_MERGE_PASS_DISTANCE:
                    continue
                object_speed = max(0.0, float(object_info.velocity))
                clear_distance = max(
                    0.0,
                    parameters.CONSERVATIVE_MERGE_PASS_DISTANCE
                    - object_match.s,
                )
                clear_time = (
                    clear_distance / max(0.1, object_speed)
                    + parameters.MERGE_YIELD_TIME_GAP
                )
                speed_limit = min(
                    speed_limit,
                    ego_distance / max(0.1, clear_time),
                    parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                )
                self._last_merge_debug.append(
                    "successor object={} link={} s={:.1f}m -> SAFE_ZONE_YIELD limit={:.2f}m/s".format(
                        getattr(object_info, "name", "") or "unnamed",
                        object_match.lane_link.id,
                        object_match.s,
                        speed_limit,
                    )
                )

            # A vehicle between lane centerlines can temporarily fail map
            # matching. In this designated zone, proximity is enough to keep a
            # conservative speed until its lane association becomes stable.
            for entry in vehicle_entries:
                object_info = entry["object_info"]
                if id(object_info) in matched_ids:
                    continue
                if object_info.position.distance(merge_point) > parameters.CONSERVATIVE_MERGE_OBJECT_RADIUS:
                    continue
                speed_limit = min(
                    speed_limit,
                    parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                )
                self._last_merge_debug.append(
                    "unmatched object={} near merge -> SAFE_ZONE_YIELD limit={:.2f}m/s".format(
                        getattr(object_info, "name", "") or "unnamed",
                        speed_limit,
                    )
                )
        trajectory_limit = self._trajectory_merge_speed_limit(
            vehicle_state, ego_match, object_matches, vehicle_entries
        )
        speed_limit = min(speed_limit, trajectory_limit)
        if math.isinf(speed_limit):
            return float("inf")
        return min(ego_speed, speed_limit)

    def _predictive_merge_zone_speed_limit(
        self,
        vehicle_state,
        ego_match,
        object_matches,
        vehicle_entries,
        merge_successors,
        merge_point,
    ):
        """Give the merge circle to the OBB expected to touch it first."""
        ego_link = ego_match.lane_link
        relevant = []
        for entry, object_match in object_matches:
            object_link = object_match.lane_link
            if object_link.id == ego_link.id:
                continue
            if object_link.id not in parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS:
                continue
            relevant.append((entry, object_match))

        touching = []
        for entry, object_match in relevant:
            object_info = entry["object_info"]
            object_speed = max(0.0, float(object_info.velocity))
            object_half_length = 0.5 * max(
                1.0, float(getattr(object_info, "size_x", 0.0))
            )
            object_half_width = 0.5 * max(
                1.0, float(getattr(object_info, "size_y", 0.0))
            )
            object_touch_distance = self._first_circle_obb_touch_distance(
                object_match,
                merge_point,
                parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
                object_half_length,
                object_half_width,
                max_distance=(
                    object_speed * parameters.MERGE_PRIORITY_HORIZON
                ),
            )
            object_touch_time = self._touch_time(
                object_touch_distance,
                object_speed,
                parameters.MERGE_PRIORITY_HORIZON,
            )
            if not math.isinf(object_touch_time):
                touching.append((entry, object_match, object_touch_time))

        ego_speed = max(0.0, float(vehicle_state.velocity))
        ego_entry_distance = self._first_circle_obb_touch_distance(
            ego_match,
            merge_point,
            parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
            0.5 * parameters.VEHICLE_LENGTH,
            0.5 * parameters.VEHICLE_WIDTH,
            center_offset=0.5 * parameters.VEHICLE_LENGTH,
            max_distance=max(
                parameters.PREDICTIVE_MERGE_EGO_RADIUS,
                ego_speed * parameters.MERGE_PRIORITY_HORIZON,
            ),
        )
        ego_inside = ego_entry_distance <= 1e-6
        ego_touch_time = self._touch_time(
            ego_entry_distance,
            ego_speed,
            parameters.MERGE_PRIORITY_HORIZON,
        )
        earliest_touch = (
            min(time_value for _, _, time_value in touching)
            if touching
            else float("inf")
        )
        other_inside = earliest_touch <= 1e-9
        # Actual occupancy wins over an earlier noisy ETA decision. Once Ego
        # touches the conflict core first, merge pacing must never brake it into
        # a side/rear impact.
        if self._predictive_merge_priority is None:
            # 현재 원 안에 먼저 들어온 차량을 우선한다. 둘 다 원 밖이면
            # 미래 경로가 단순히 원에 닿는지만 보지 않고 예상 진입시간을
            # 비교하여 상대 차량이 먼저 도착할 때만 선제 감속한다.
            if other_inside:
                self._predictive_merge_priority = "OTHER"
                self._predictive_merge_clear_cycles = 0
            elif ego_inside:
                self._predictive_merge_priority = "EGO"
            elif touching and self._merge_arrivals_overlap(
                ego_touch_time,
                ego_speed,
                touching,
                parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
            ):
                self._predictive_merge_priority = "OTHER"
                self._predictive_merge_clear_cycles = 0
            elif not math.isinf(ego_touch_time):
                # Use the same future path/OBB criterion for Ego, and lock GO
                # before its current rear-axle position reaches the circle.
                self._predictive_merge_priority = "EGO"

            if self._predictive_merge_priority == "OTHER":
                winner_entry, winner_match, _ = min(
                    touching, key=lambda item: item[2]
                )
                self._predictive_merge_other_key = self._merge_zone_object_key(
                    winner_entry, winner_match
                )

        if self._predictive_merge_priority == "OTHER" and ego_inside:
            # A future ETA is only a provisional right-of-way decision. Sensor
            # and velocity noise can swap two arrivals separated by a fraction
            # of a second. If Ego actually touches the circle before the
            # selected vehicle has ever reached it, Ego is the real winner and
            # must not keep braking into the merge.
            tracked_match = next(
                (
                    object_match
                    for entry, object_match in object_matches
                    if self._merge_zone_object_key(entry, object_match)
                    == self._predictive_merge_other_key
                ),
                None,
            )
            tracked_has_entered = False
            if tracked_match is not None:
                tracked_link = tracked_match.lane_link
                if tracked_link.id in merge_successors:
                    tracked_has_entered = True
                elif tracked_link.id in parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS:
                    tracked_circle_s = self.lane_map.project(
                        merge_point, tracked_link
                    ).s
                    tracked_has_entered = tracked_match.s >= tracked_circle_s
            if (
                tracked_match is not None
                and not tracked_has_entered
                and not other_inside
            ):
                previous_key = self._predictive_merge_other_key
                self._predictive_merge_priority = "EGO"
                self._predictive_merge_other_key = None
                self._predictive_merge_clear_cycles = 0
                self._last_merge_debug.append(
                    "predictive-zone actual entry override: EGO entered before {}".format(
                        previous_key or "predicted vehicle"
                    )
                )

        if self._predictive_merge_priority == "EGO":
            self._merge_speed_limit_state = float("inf")
            self._predictive_merge_other_key = None
            for entry, _ in relevant:
                self._merge_go_object_ids.add(id(entry["object_info"]))
            self._last_merge_debug.append(
                "predictive-zone priority=EGO ego_touch={}s (Ego OBB first), late_vehicles_go={}".format(
                    "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                    len(self._merge_go_object_ids),
                )
            )
            return float("inf")

        if self._predictive_merge_priority != "OTHER":
            self._last_merge_debug.append(
                "predictive-zone priority=PENDING monitored={} (no predicted entry)".format(
                    ",".join(parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS)
                )
            )
            return float("inf")

        # Decide and lock the winner early, but do not disturb normal driving
        # before the user-designated first-merge braking point.
        brake_start = Point(*parameters.PREDICTIVE_MERGE_BRAKE_START_POINT)
        brake_start_s = self.lane_map.project(brake_start, ego_link).s
        if ego_match.s + 1e-6 < brake_start_s:
            self._merge_speed_limit_state = float("inf")
            self._last_merge_debug.append(
                "predictive-zone priority=OTHER waiting brake-start "
                "remaining={:.1f}m -> no decel".format(
                    brake_start_s - ego_match.s
                )
            )
            return float("inf")

        # Follow only the vehicle which actually won priority. A stream of
        # later approach vehicles must not extend OTHER forever after the
        # winner has cleared the overlap.
        tracked_entry = None
        tracked_match = None
        for entry, object_match in object_matches:
            if (
                self._merge_zone_object_key(entry, object_match)
                == self._predictive_merge_other_key
            ):
                tracked_entry = entry
                tracked_match = object_match
                break

        if tracked_entry is None and str(
            self._predictive_merge_other_key or ""
        ).startswith("link:"):
            # MORAI normally supplies stable NPC names. For unnamed objects,
            # bridge the one frame in which its link changes to the successor.
            successor_entries = [
                (entry, object_match)
                for entry, object_match in object_matches
                if object_match.lane_link.id in merge_successors
            ]
            if len(successor_entries) == 1:
                tracked_entry, tracked_match = successor_entries[0]
                self._predictive_merge_other_key = self._merge_zone_object_key(
                    tracked_entry, tracked_match
                )

        tracked_cleared = (
            tracked_match is not None
            and tracked_match.lane_link.id in merge_successors
            and tracked_match.s > parameters.PREDICTIVE_MERGE_PASS_DISTANCE
        )
        cycle_time = self.control_dt
        if tracked_entry is not None:
            tracked_speed = max(
                0.0, float(tracked_entry["object_info"].velocity)
            )
            clear_time = self._predictive_merge_clear_time(
                tracked_match, tracked_speed, merge_successors
            )
            self._predictive_merge_last_speed = tracked_speed
            self._predictive_merge_last_clear_time = clear_time
            predicted_hold = (
                clear_time + parameters.PREDICTIVE_MERGE_FOLLOW_TIME_GAP
                if not math.isinf(clear_time)
                else parameters.PREDICTIVE_MERGE_DROPOUT_MAX_HOLD_TIME
            )
            self._predictive_merge_dropout_hold_time = max(
                parameters.PREDICTIVE_MERGE_DROPOUT_MIN_HOLD_TIME,
                min(
                    parameters.PREDICTIVE_MERGE_DROPOUT_MAX_HOLD_TIME,
                    predicted_hold,
                ),
            )
        else:
            self._predictive_merge_dropout_hold_time = max(
                0.0,
                self._predictive_merge_dropout_hold_time - cycle_time,
            )
            if not math.isinf(self._predictive_merge_last_clear_time):
                self._predictive_merge_last_clear_time = max(
                    0.0,
                    self._predictive_merge_last_clear_time - cycle_time,
                )
            tracked_speed = self._predictive_merge_last_speed
            clear_time = self._predictive_merge_last_clear_time

        if tracked_entry is not None and not tracked_cleared:
            self._predictive_merge_clear_cycles = 0
        else:
            self._predictive_merge_clear_cycles += 1
        confirmed_clear = (
            tracked_cleared
            and self._predictive_merge_clear_cycles
            >= self._cycles_for_reference_count(
                parameters.PREDICTIVE_MERGE_CLEAR_CYCLES
            )
        )
        predicted_clear_after_dropout = (
            tracked_entry is None
            and self._predictive_merge_clear_cycles
            >= self._cycles_for_reference_count(
                parameters.PREDICTIVE_MERGE_CLEAR_CYCLES
            )
            and self._predictive_merge_dropout_hold_time <= 1e-9
        )
        if confirmed_clear or predicted_clear_after_dropout:
            self._predictive_merge_priority = "EGO"
            self._predictive_merge_other_key = None
            self._predictive_merge_clear_cycles = 0
            self._predictive_merge_last_speed = 0.0
            self._predictive_merge_last_clear_time = float("inf")
            self._predictive_merge_dropout_hold_time = 0.0
            self._merge_speed_limit_state = float("inf")
            for entry, _ in relevant:
                self._merge_go_object_ids.add(id(entry["object_info"]))
            self._last_merge_debug.append(
                "predictive-zone other cleared -> priority={}".format(
                    self._predictive_merge_priority or "PENDING"
                )
            )
            return float("inf")

        if tracked_entry is None:
            self._last_merge_debug.append(
                "predictive-zone winner dropout key={} hold={:.2f}s using_last_speed={:.2f}m/s".format(
                    self._predictive_merge_other_key or "unknown",
                    self._predictive_merge_dropout_hold_time,
                    tracked_speed,
                )
            )
        if tracked_speed < parameters.PREDICTIVE_MERGE_MOVING_SPEED:
            # A genuinely stopped priority vehicle cannot be followed safely.
            speed_limit = 0.0
        else:
            rolling_floor = parameters.MERGE_MIN_YIELD_SPEED
            desired_arrival_time = (
                clear_time + parameters.PREDICTIVE_MERGE_FOLLOW_TIME_GAP
            )
            pacing_limit = (
                ego_entry_distance / max(0.1, desired_arrival_time)
                if not math.isinf(ego_entry_distance)
                else rolling_floor
            )
            braking_limit = math.sqrt(
                2.0
                * parameters.PREDICTIVE_MERGE_COMFORT_DECEL
                * max(0.0, ego_entry_distance)
            )
            speed_limit = max(
                rolling_floor,
                min(
                    parameters.PREDICTIVE_MERGE_MAX_SPEED,
                    pacing_limit,
                    braking_limit if braking_limit > 1e-6 else 0.0,
                ),
            )
            speed_limit = min(
                speed_limit,
                parameters.PREDICTIVE_MERGE_MAX_SPEED,
            )
        self._last_merge_debug.append(
            "predictive-zone priority=OTHER key={} touch={}s ego_touch={}s clear_time={}s entry_distance={:.1f}m limit={:.2f}m/s".format(
                self._predictive_merge_other_key or "unknown",
                "inf" if math.isinf(earliest_touch) else "{:.2f}".format(earliest_touch),
                "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                "inf" if math.isinf(clear_time) else "{:.2f}".format(clear_time),
                0.0 if math.isinf(ego_entry_distance) else ego_entry_distance,
                speed_limit,
            )
        )
        # Moving priority traffic is paced, not stopped for. The generic OBB
        # collision gate can still command zero for a real imminent collision.
        return speed_limit

    @staticmethod
    def _predictive_merge_clear_time(
        object_match, object_speed, merge_successors
    ):
        if object_match is None or object_speed <= 0.1:
            return float("inf")
        object_link = object_match.lane_link
        if object_link.id in merge_successors:
            remaining = max(
                0.0,
                parameters.PREDICTIVE_MERGE_PASS_DISTANCE - object_match.s,
            )
        elif object_link.id in parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS:
            remaining = (
                max(0.0, object_link.length - object_match.s)
                + parameters.PREDICTIVE_MERGE_PASS_DISTANCE
            )
        else:
            return float("inf")
        return remaining / object_speed

    def _first_circle_obb_touch_distance(
        self,
        lane_match,
        circle_center,
        circle_radius,
        half_length,
        half_width,
        center_offset=0.0,
        max_distance=0.0,
    ):
        """Return route distance until an OBB first intersects a circle."""
        max_distance = max(0.0, float(max_distance))

        def touches(distance):
            options = self.lane_map.advance_options(lane_match, distance)
            return any(
                self._obb_touches_circle(
                    point,
                    yaw,
                    center_offset,
                    half_length,
                    half_width,
                    circle_center,
                    circle_radius,
                )
                for point, yaw, _ in options
            )

        if touches(0.0):
            return 0.0
        if max_distance <= 1e-9:
            return float("inf")

        step = max(0.25, parameters.PATH_SAMPLE_SPACING)
        previous = 0.0
        distance = min(step, max_distance)
        while distance <= max_distance + 1e-9:
            if touches(distance):
                low = previous
                high = distance
                for _ in range(8):
                    middle = 0.5 * (low + high)
                    if touches(middle):
                        high = middle
                    else:
                        low = middle
                return high
            if distance >= max_distance - 1e-9:
                break
            previous = distance
            distance = min(distance + step, max_distance)
        return float("inf")

    @staticmethod
    def _touch_time(distance, speed, horizon=float("inf")):
        if distance <= 1e-6:
            return 0.0
        if math.isinf(distance) or speed <= 0.1:
            return float("inf")
        time_value = distance / speed
        return time_value if time_value <= horizon + 1e-9 else float("inf")

    @staticmethod
    def _merge_arrivals_overlap(ego_entry_time, ego_speed, touching, radius):
        """Yield unless Ego can fully clear with both time and distance margin."""
        if not touching:
            return False
        winner_entry, _, other_entry_time = min(touching, key=lambda item: item[2])
        if math.isinf(ego_entry_time):
            return True
        other_speed = max(
            0.0, float(winner_entry["object_info"].velocity)
        )
        # Conservative OBB traversal estimate from first touch until Ego's rear
        # leaves the conflict circle.
        occupancy_length = 2.0 * float(radius) + parameters.VEHICLE_LENGTH
        ego_exit_time = ego_entry_time + occupancy_length / max(0.1, ego_speed)
        clear_time = other_entry_time - ego_exit_time
        predicted_bumper_gap = max(0.0, clear_time) * other_speed
        return (
            clear_time < parameters.MERGE_GO_CLEAR_TIME_MARGIN
            or predicted_bumper_gap
            < parameters.MERGE_CLOSE_GAP_YIELD_DISTANCE
        )

    @staticmethod
    def _obb_touches_circle(
        path_point,
        yaw,
        center_offset,
        half_length,
        half_width,
        circle_center,
        circle_radius,
    ):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        center_x = path_point.x + center_offset * cos_yaw
        center_y = path_point.y + center_offset * sin_yaw
        delta_x = circle_center.x - center_x
        delta_y = circle_center.y - center_y
        local_x = delta_x * cos_yaw + delta_y * sin_yaw
        local_y = -delta_x * sin_yaw + delta_y * cos_yaw
        closest_x = max(-half_length, min(half_length, local_x))
        closest_y = max(-half_width, min(half_width, local_y))
        error_x = local_x - closest_x
        error_y = local_y - closest_y
        return error_x * error_x + error_y * error_y <= circle_radius ** 2

    def _merge_zone_priority_speed_limit(
        self,
        vehicle_state,
        ego_link,
        object_matches,
        vehicle_entries,
        conservative_successors,
        merge_point,
    ):
        """Use the same future-OBB first-arrival rule as the first merge."""
        ego_match = self.lane_map.project(vehicle_state.position, ego_link)
        relevant = []
        for entry, object_match in object_matches:
            object_link = object_match.lane_link
            if object_link.id == ego_link.id:
                continue
            shared_successors = set(
                self.lane_map.successors.get(object_link.id, ())
            ).intersection(conservative_successors)
            if shared_successors:
                relevant.append((entry, object_match))

        touching = []
        for entry, object_match in relevant:
            object_info = entry["object_info"]
            object_speed = max(0.0, float(object_info.velocity))
            touch_distance = self._first_circle_obb_touch_distance(
                object_match,
                merge_point,
                parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
                0.5 * max(1.0, float(getattr(object_info, "size_x", 0.0))),
                0.5 * max(1.0, float(getattr(object_info, "size_y", 0.0))),
                max_distance=object_speed * parameters.MERGE_PRIORITY_HORIZON,
            )
            touch_time = self._touch_time(
                touch_distance,
                object_speed,
                parameters.MERGE_PRIORITY_HORIZON,
            )
            if not math.isinf(touch_time):
                touching.append((entry, object_match, touch_time))

        ego_speed = max(0.0, float(vehicle_state.velocity))
        ego_entry_distance = self._first_circle_obb_touch_distance(
            ego_match,
            merge_point,
            parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
            0.5 * parameters.VEHICLE_LENGTH,
            0.5 * parameters.VEHICLE_WIDTH,
            center_offset=0.5 * parameters.VEHICLE_LENGTH,
            max_distance=max(
                parameters.CONSERVATIVE_MERGE_EGO_RADIUS,
                ego_speed * parameters.MERGE_PRIORITY_HORIZON,
            ),
        )
        ego_inside = ego_entry_distance <= 1e-6
        ego_touch_time = self._touch_time(
            ego_entry_distance,
            ego_speed,
            parameters.MERGE_PRIORITY_HORIZON,
        )
        earliest_touch = (
            min(item[2] for item in touching) if touching else float("inf")
        )
        other_inside = earliest_touch <= 1e-9

        if self._merge_zone_priority is None:
            if other_inside:
                self._merge_zone_priority = "OTHER"
            elif ego_inside:
                self._merge_zone_priority = "EGO"
            elif touching and self._merge_arrivals_overlap(
                ego_touch_time,
                ego_speed,
                touching,
                parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
            ):
                self._merge_zone_priority = "OTHER"
            elif not math.isinf(ego_touch_time):
                self._merge_zone_priority = "EGO"

            if self._merge_zone_priority == "OTHER":
                winner_entry, winner_match, _ = min(
                    touching, key=lambda item: item[2]
                )
                self._merge_zone_other_key = self._merge_zone_object_key(
                    winner_entry, winner_match
                )
                self._merge_zone_clear_cycles = 0

        tracked_entry = None
        tracked_match = None
        for entry, object_match in object_matches:
            if (
                self._merge_zone_object_key(entry, object_match)
                == self._merge_zone_other_key
            ):
                tracked_entry = entry
                tracked_match = object_match
                break

        if self._merge_zone_priority == "OTHER" and ego_inside:
            tracked_has_entered = False
            if tracked_match is not None:
                tracked_link = tracked_match.lane_link
                if tracked_link.id in conservative_successors:
                    tracked_has_entered = True
                else:
                    tracked_circle_s = self.lane_map.project(
                        merge_point, tracked_link
                    ).s
                    tracked_has_entered = tracked_match.s >= tracked_circle_s
            if not tracked_has_entered and not other_inside:
                previous_key = self._merge_zone_other_key
                self._merge_zone_priority = "EGO"
                self._merge_zone_other_key = None
                self._merge_zone_clear_cycles = 0
                self._last_merge_debug.append(
                    "second predictive-zone actual entry override: EGO entered before {}".format(
                        previous_key or "predicted vehicle"
                    )
                )

        if self._merge_zone_priority == "EGO":
            self._merge_speed_limit_state = float("inf")
            for entry, object_match in object_matches:
                object_link = object_match.lane_link
                shared_successors = set(
                    self.lane_map.successors.get(object_link.id, ())
                ).intersection(conservative_successors)
                if object_link.id in conservative_successors or shared_successors:
                    self._merge_go_object_ids.add(id(entry["object_info"]))
            self._last_merge_debug.append(
                "second predictive-zone priority=EGO ego_touch={}s, late_vehicles_go={}".format(
                    "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                    len(self._merge_go_object_ids),
                )
            )
            return float("inf")

        if self._merge_zone_priority != "OTHER":
            self._last_merge_debug.append(
                "second predictive-zone priority=PENDING (no predicted entry)"
            )
            return float("inf")

        tracked_cleared = (
            tracked_match is not None
            and tracked_match.lane_link.id in conservative_successors
            and tracked_match.s > parameters.CONSERVATIVE_MERGE_PASS_DISTANCE
        )
        if tracked_entry is not None and not tracked_cleared:
            self._merge_zone_clear_cycles = 0
        else:
            self._merge_zone_clear_cycles += 1
        if (
            self._merge_zone_clear_cycles
            >= self._cycles_for_reference_count(
                parameters.CONSERVATIVE_MERGE_CLEAR_CYCLES
            )
        ):
            self._merge_zone_priority = "EGO"
            self._merge_zone_other_key = None
            self._merge_zone_clear_cycles = 0
            self._merge_speed_limit_state = float("inf")
            self._last_merge_debug.append(
                "second predictive-zone other cleared -> priority=EGO"
            )
            return float("inf")

        tracked_speed = (
            max(0.0, float(tracked_entry["object_info"].velocity))
            if tracked_entry is not None
            else 0.0
        )
        if tracked_match is None or tracked_speed <= 0.1:
            clear_time = float("inf")
        elif tracked_match.lane_link.id in conservative_successors:
            clear_time = max(
                0.0,
                parameters.CONSERVATIVE_MERGE_PASS_DISTANCE - tracked_match.s,
            ) / tracked_speed
        else:
            clear_time = (
                max(0.0, tracked_match.lane_link.length - tracked_match.s)
                + parameters.CONSERVATIVE_MERGE_PASS_DISTANCE
            ) / tracked_speed

        if tracked_speed < parameters.CONSERVATIVE_MERGE_MOVING_SPEED:
            speed_limit = 0.0
        else:
            rolling_floor = parameters.MERGE_MIN_YIELD_SPEED
            desired_arrival_time = (
                clear_time + parameters.CONSERVATIVE_MERGE_FOLLOW_TIME_GAP
            )
            pacing_limit = (
                ego_entry_distance / max(0.1, desired_arrival_time)
                if not math.isinf(ego_entry_distance)
                else rolling_floor
            )
            braking_limit = math.sqrt(
                2.0
                * parameters.CONSERVATIVE_MERGE_COMFORT_DECEL
                * max(0.0, ego_entry_distance)
            )
            speed_limit = max(
                rolling_floor,
                min(
                    parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                    pacing_limit,
                    braking_limit if braking_limit > 1e-6 else 0.0,
                ),
            )
        self._last_merge_debug.append(
            "second predictive-zone priority=OTHER key={} touch={}s ego_touch={}s clear_time={}s entry_distance={:.1f}m limit={:.2f}m/s".format(
                self._merge_zone_other_key or "unknown",
                "inf" if math.isinf(earliest_touch) else "{:.2f}".format(earliest_touch),
                "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                "inf" if math.isinf(clear_time) else "{:.2f}".format(clear_time),
                0.0 if math.isinf(ego_entry_distance) else ego_entry_distance,
                speed_limit,
            )
        )
        return speed_limit

    @staticmethod
    def _merge_zone_object_key(entry, lane_match):
        object_info = entry["object_info"]
        unique_id = getattr(object_info, "unique_id", None)
        if unique_id is not None:
            return "npc:{}".format(unique_id)
        name = str(getattr(object_info, "name", "") or "").strip()
        if name:
            return name
        if lane_match is not None:
            return "link:{}".format(lane_match.lane_link.id)
        return "anonymous:{:.1f}:{:.1f}".format(
            object_info.position.x, object_info.position.y
        )

    def _reset_merge_zone_priority(self):
        self._merge_zone_priority = None
        self._merge_zone_other_key = None
        self._merge_zone_clear_cycles = 0
        self._predictive_merge_priority = None
        self._predictive_merge_clear_cycles = 0
        self._predictive_merge_other_key = None
        self._predictive_merge_last_speed = 0.0
        self._predictive_merge_last_clear_time = float("inf")
        self._predictive_merge_dropout_hold_time = 0.0

    def _trajectory_merge_speed_limit(
        self, vehicle_state, ego_match, object_matches, vehicle_entries
    ):
        """Backstop merge ACC when an incoming vehicle misses lane matching."""
        ego_speed = max(0.0, float(vehicle_state.velocity))
        horizon = parameters.MERGE_TRAJECTORY_HORIZON
        step = parameters.MERGE_TRAJECTORY_TIME_STEP
        matched_object_ids = {
            id(entry["object_info"]) for entry, _ in object_matches
        }
        time_value = step
        earliest_collision = float("inf")
        unmatched_count = sum(
            1
            for entry in vehicle_entries
            if id(entry["object_info"]) not in matched_object_ids
        )
        while time_value <= horizon + 1e-9:
            ego_point, ego_yaw, _ = self.lane_map.advance_default(
                ego_match, ego_speed * time_value
            )
            for entry in vehicle_entries:
                object_info = entry["object_info"]
                if id(object_info) in self._merge_go_object_ids:
                    continue
                if id(object_info) in matched_object_ids:
                    continue
                if (
                    vehicle_state.position.distance(object_info.position)
                    > parameters.MERGE_TRAJECTORY_MAX_DISTANCE
                ):
                    continue
                local_position = entry.get("local_position")
                if (
                    local_position is not None
                    and local_position.x < 0.0
                    and abs(local_position.y)
                    <= parameters.ACC_FALLBACK_LATERAL_THRESHOLD
                ):
                    # Braking for a same-lane rear vehicle only worsens risk.
                    continue
                object_speed = max(0.0, float(object_info.velocity))
                object_yaw = math.radians(
                    float(getattr(object_info, "heading_deg", 0.0))
                )
                object_point = Point(
                    object_info.position.x
                    + math.cos(object_yaw) * object_speed * time_value,
                    object_info.position.y
                    + math.sin(object_yaw) * object_speed * time_value,
                )
                if self._obb_collision(
                    ego_point,
                    ego_yaw,
                    object_point,
                    object_yaw,
                    object_info,
                ):
                    earliest_collision = min(earliest_collision, time_value)
            time_value += step

        if math.isinf(earliest_collision):
            if unmatched_count:
                self._last_merge_debug.append(
                    "trajectory unmatched={} collision=none horizon={:.1f}s".format(
                        unmatched_count, horizon
                    )
                )
            return float("inf")
        if earliest_collision <= 1.0:
            self._last_merge_debug.append(
                "trajectory unmatched={} collision={:.2f}s -> STOP".format(
                    unmatched_count, earliest_collision
                )
            )
            return 0.0
        delay_ratio = max(
            0.0,
            (earliest_collision - 0.5)
            / (earliest_collision + parameters.MERGE_YIELD_TIME_GAP),
        )
        limit = ego_speed * delay_ratio
        self._last_merge_debug.append(
            "trajectory unmatched={} collision={:.2f}s -> limit={:.2f}m/s".format(
                unmatched_count, earliest_collision, limit
            )
        )
        return limit

    def _stabilize_merge_speed_limit(self, raw_limit, ego_speed):
        """Apply emergency reductions now and release transient ACC smoothly."""
        previous = self._merge_speed_limit_state
        recovery_step = parameters.MERGE_ACC_RECOVERY_RATE * self.control_dt
        if not math.isinf(raw_limit):
            if math.isinf(previous):
                filtered = max(0.0, raw_limit)
            else:
                filtered = min(
                    max(0.0, raw_limit),
                    previous + recovery_step,
                )
            self._merge_speed_limit_state = filtered
            return filtered

        if math.isinf(previous):
            return float("inf")
        recovered = previous + recovery_step
        if recovered >= max(0.0, float(ego_speed)):
            self._merge_speed_limit_state = float("inf")
            return float("inf")
        self._merge_speed_limit_state = recovered
        return recovered

    def _remaining_active_candidate(self, vehicle_state):
        candidate = self.active_candidate
        if candidate is None or len(candidate.points) < 2:
            return None
        nearest_index = min(
            range(len(candidate.points)),
            key=lambda index: candidate.points[index].distance(vehicle_state.position),
        )
        start_index = max(0, nearest_index - 1)
        candidate.points = candidate.points[start_index:]
        candidate.valid = len(candidate.points) >= 2
        candidate.collision_time = float("inf")
        return candidate

    def _resolve_ego_lane(self, vehicle_state):
        global_match = self.lane_map.match(
            vehicle_state.position,
            heading=vehicle_state.yaw,
            preferred_ids=(self.target_link_id,) if self.target_link_id else (),
        )
        if self.target_link_id is None:
            return global_match

        target = self.lane_map.links.get(self.target_link_id)
        if target is not None:
            target_match = self.lane_map.project(vehicle_state.position, target)
            if target_match.distance <= parameters.MANEUVER_TARGET_MAX_DISTANCE:
                return target_match
            for successor_id in self.lane_map.successors.get(target.id, ()):
                successor = self.lane_map.links[successor_id]
                successor_match = self.lane_map.project(vehicle_state.position, successor)
                if successor_match.distance <= parameters.LANE_MATCH_MAX_DISTANCE:
                    self.target_link_id = successor_id
                    return successor_match
        return global_match

    def _match_objects(self, vehicle_entries):
        matches = []
        for entry in vehicle_entries:
            object_info = entry["object_info"]
            heading = math.radians(float(getattr(object_info, "heading_deg", 0.0)))
            lane_match = self.lane_map.match(object_info.position, heading=heading)
            if lane_match is not None:
                matches.append((entry, lane_match))
        return matches

    def _front_risk(
        self, vehicle_state, ego_match, object_matches, vehicle_entries=()
    ):
        nearest = float("inf")
        minimum_ttc = float("inf")
        matched_object_ids = set()
        for entry, object_match in object_matches:
            matched_object_ids.add(id(entry["object_info"]))
            center_distance = self.lane_map.forward_distance(ego_match, object_match)
            if center_distance is None or center_distance > parameters.FRONT_RELEVANCE_DISTANCE:
                continue
            object_info = entry["object_info"]
            object_length = max(1.0, float(getattr(object_info, "size_x", 0.0)))
            gap = max(
                0.0,
                center_distance
                - 0.5 * parameters.VEHICLE_LENGTH
                - 0.5 * object_length,
            )
            closing_speed = max(
                0.0, vehicle_state.velocity - float(object_info.velocity)
            )
            object_ttc = gap / closing_speed if closing_speed > 0.1 else float("inf")
            nearest = min(nearest, gap)
            minimum_ttc = min(minimum_ttc, object_ttc)

        # Link matching can briefly fail at merges/splits. Keep ACC available
        # through an ego-local same-lane check so a real lead vehicle is not
        # ignored merely because its MGeo link ID differs for a few frames.
        for entry in vehicle_entries:
            if id(entry["object_info"]) in matched_object_ids:
                continue
            local_position = entry.get("local_position")
            if local_position is None or local_position.x <= 0.0:
                continue
            if abs(local_position.y) > parameters.ACC_FALLBACK_LATERAL_THRESHOLD:
                continue
            object_info = entry["object_info"]
            object_length = max(1.0, float(getattr(object_info, "size_x", 0.0)))
            gap = max(
                0.0,
                local_position.x
                - 0.5 * parameters.VEHICLE_LENGTH
                - 0.5 * object_length,
            )
            if gap > parameters.FRONT_RELEVANCE_DISTANCE:
                continue
            closing_speed = max(
                0.0, vehicle_state.velocity - float(object_info.velocity)
            )
            object_ttc = gap / closing_speed if closing_speed > 0.1 else float("inf")
            nearest = min(nearest, gap)
            minimum_ttc = min(minimum_ttc, object_ttc)
        return nearest, minimum_ttc

    @staticmethod
    def _acc_speed_limit(ego_speed, gap, ttc):
        """Return an explicit longitudinal speed ceiling for ACC_FOLLOW."""
        if math.isinf(gap):
            return float("inf")
        closing_speed = 0.0 if math.isinf(ttc) else gap / max(0.1, ttc)
        lead_speed = max(0.0, ego_speed - closing_speed)
        desired_gap = (
            parameters.ACC_STANDSTILL_DISTANCE
            + max(0.0, ego_speed) * parameters.ACC_TIME_GAP
        )
        if math.isinf(ttc) and gap >= desired_gap:
            # Once the lead vehicle has opened a safe gap, remove the ceiling.
            # Capping by current Ego speed here would lock a stopped car at 0.
            return float("inf")
        speed_limit = (
            lead_speed
            + parameters.ACC_DISTANCE_GAIN * (gap - desired_gap)
        )
        return max(0.0, speed_limit)

    def _build_candidate(
        self,
        vehicle_state,
        target_link,
        target_projection,
        direction,
        source_link=None,
    ):
        candidate = HighwayCandidatePath(layer="lane_{}".format(direction))
        candidate.direction = direction
        candidate.target_link_id = target_link.id
        candidate.target_lane = target_link.lane_number
        source_lane = source_link.lane_number if source_link else target_link.lane_number
        candidate.offset = (
            source_lane - target_link.lane_number
        ) * parameters.LANE_WIDTH

        path_distance = max(
            parameters.PATH_LOOKAHEAD_MIN,
            vehicle_state.velocity * parameters.PREDICTION_HORIZON + 12.0,
        )
        if direction == "keep":
            # A short polynomial recentering segment can produce a steering spike
            # at highway speed. Follow the measured link centerline directly.
            candidate.points = [
                item[0]
                for item in self.lane_map.route_points(
                    target_projection,
                    path_distance,
                    parameters.PATH_SAMPLE_SPACING,
                )
            ]
            candidate.curvature_cost = self._max_curvature(candidate.points)
            candidate.cost = 4.0 * candidate.curvature_cost
            return candidate

        transition_distance = parameters.LANE_CHANGE_DISTANCE
        goal, goal_yaw, _ = self.lane_map.advance_default(
            target_projection, transition_distance
        )
        coefficients, goal_x = self._solve_goal(vehicle_state, goal, goal_yaw)
        if coefficients is None:
            candidate.valid = False
            candidate.cost = float("inf")
            return candidate

        candidate.points = self._sample_polynomial(
            vehicle_state, goal_x, coefficients
        )
        travelled = transition_distance + parameters.PATH_SAMPLE_SPACING
        while travelled <= path_distance:
            point, _, _ = self.lane_map.advance_default(target_projection, travelled)
            if not candidate.points or point.distance(candidate.points[-1]) > 0.1:
                candidate.points.append(point)
            travelled += parameters.PATH_SAMPLE_SPACING

        candidate.curvature_cost = self._max_curvature(candidate.points)
        lane_change_penalty = 0.0 if direction == "keep" else 0.25
        no_successor_penalty = (
            2.0
            if direction != "keep"
            and not self.lane_map.successors.get(target_link.id, ())
            else 0.0
        )
        candidate.cost = (
            4.0 * candidate.curvature_cost
            + lane_change_penalty
            + no_successor_penalty
        )
        return candidate

    @staticmethod
    def _solve_goal(vehicle_state, goal, goal_yaw):
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        dx = goal.x - vehicle_state.position.x
        dy = goal.y - vehicle_state.position.y
        goal_x = cos_yaw * dx + sin_yaw * dy
        goal_y = -sin_yaw * dx + cos_yaw * dy
        relative_yaw = math.atan2(
            math.sin(goal_yaw - vehicle_state.yaw),
            math.cos(goal_yaw - vehicle_state.yaw),
        )
        if goal_x < 2.0:
            return None, goal_x
        relative_yaw = max(math.radians(-70.0), min(math.radians(70.0), relative_yaw))
        slope = math.tan(relative_yaw)
        a3 = (10.0 * goal_y - 4.0 * slope * goal_x) / goal_x ** 3
        a4 = (-15.0 * goal_y + 7.0 * slope * goal_x) / goal_x ** 4
        a5 = (6.0 * goal_y - 3.0 * slope * goal_x) / goal_x ** 5
        return (a3, a4, a5), goal_x

    @staticmethod
    def _sample_polynomial(vehicle_state, goal_x, coefficients):
        a3, a4, a5 = coefficients
        count = max(2, int(math.ceil(goal_x / parameters.PATH_SAMPLE_SPACING)))
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        points = []
        for local_x in np.linspace(0.0, goal_x, count + 1):
            local_y = a3 * local_x ** 3 + a4 * local_x ** 4 + a5 * local_x ** 5
            points.append(
                Point(
                    vehicle_state.position.x + cos_yaw * local_x - sin_yaw * local_y,
                    vehicle_state.position.y + sin_yaw * local_x + cos_yaw * local_y,
                )
            )
        return points

    def _evaluate_dynamic_collision(self, candidate, vehicle_state, object_matches):
        if not candidate.valid or len(candidate.points) < 2:
            return
        cumulative = self._cumulative_distance(candidate.points)
        ego_speed = max(parameters.PREDICTION_MIN_SPEED, vehicle_state.velocity)
        minimum_clearance = float("inf")
        collision_time = float("inf")

        for time_value in self._prediction_times(parameters.PREDICTION_TIME_STEP):
            ego_point, ego_yaw = self._pose_at_distance(
                candidate.points, cumulative, ego_speed * time_value
            )
            for entry, object_match in object_matches:
                object_info = entry["object_info"]
                if id(object_info) in self._merge_go_object_ids:
                    continue
                local_position = entry.get("local_position")
                if (
                    candidate.direction == "keep"
                    and local_position is not None
                    and local_position.x <= 0.0
                ):
                    # A rear vehicle must never make Ego brake on a keep-lane
                    # path; braking increases the rear-end collision risk.
                    continue
                predicted_options = self.lane_map.advance_options(
                    object_match,
                    max(0.0, float(object_info.velocity)) * time_value,
                )
                for object_point, object_yaw, _ in predicted_options:
                    clearance = ego_point.distance(object_point)
                    minimum_clearance = min(minimum_clearance, clearance)
                    if self._obb_collision(
                        ego_point, ego_yaw, object_point, object_yaw, object_info
                    ):
                        candidate.valid = False
                        candidate.cost = float("inf")
                        collision_time = time_value
                        break
                if not candidate.valid:
                    break
            if not candidate.valid:
                break
        candidate.collision_time = collision_time
        candidate.obstacle_cost = (
            0.0 if math.isinf(minimum_clearance) else 1.0 / max(0.1, minimum_clearance)
        )
        if candidate.valid:
            candidate.cost += 0.4 * candidate.obstacle_cost

    def _result_during_change(
        self, candidate, nearest_distance, ttc, acc_target_velocity, common
    ):
        collision_during_change = not candidate.valid
        urgent_front_acc = (
            not math.isinf(acc_target_velocity)
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
        safety_acc = collision_during_change or urgent_front_acc or merge_acc
        collision_time = getattr(candidate, "collision_time", float("inf"))
        emergency_stop = (
            collision_during_change
            and collision_time <= parameters.MERGE_EMERGENCY_COLLISION_TIME
        )
        if emergency_stop:
            lane_change_speed_limit = 0.0
        elif safety_acc:
            lane_change_speed_limit = (
                acc_target_velocity
                if not math.isinf(acc_target_velocity)
                else parameters.ACC_UNMATCHED_HAZARD_SPEED
            )
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
                else "highway_lane_change_acc"
                if safety_acc
                else "highway_lane_change_in_progress"
            ),
            **common
        )

    def _prediction_paths(self, object_matches):
        paths = []
        for entry, lane_match in object_matches:
            points = []
            velocity = max(0.0, float(entry["object_info"].velocity))
            # RViz shows one future OBB per vehicle at the configured horizon.
            # SAT collision checking still uses the finer fixed time step above.
            for time_value in (0.0, parameters.PREDICTION_HORIZON):
                point, _, _ = self.lane_map.advance_default(
                    lane_match, velocity * time_value
                )
                points.append(point)
            paths.append(points)
        return paths

    @staticmethod
    def _prediction_times(step):
        horizon = max(0.0, float(parameters.PREDICTION_HORIZON))
        step = max(1e-3, float(step))
        values = [0.0]
        time_value = step
        while time_value < horizon - 1e-9:
            values.append(time_value)
            time_value += step
        if horizon > 1e-9:
            values.append(horizon)
        return values

    @staticmethod
    def _cumulative_distance(points):
        values = [0.0]
        for start, end in zip(points[:-1], points[1:]):
            values.append(values[-1] + start.distance(end))
        return values

    @staticmethod
    def _pose_at_distance(points, cumulative, target):
        target = max(0.0, min(target, cumulative[-1]))
        for index in range(1, len(cumulative)):
            if cumulative[index] < target:
                continue
            start = points[index - 1]
            end = points[index]
            segment = cumulative[index] - cumulative[index - 1]
            ratio = 0.0 if segment < 1e-9 else (target - cumulative[index - 1]) / segment
            return (
                Point(
                    start.x + ratio * (end.x - start.x),
                    start.y + ratio * (end.y - start.y),
                ),
                math.atan2(end.y - start.y, end.x - start.x),
            )
        start, end = points[-2], points[-1]
        return end, math.atan2(end.y - start.y, end.x - start.x)

    @staticmethod
    def _obb_collision(ego_rear, ego_yaw, object_center, object_yaw, object_info):
        ego_forward = (math.cos(ego_yaw), math.sin(ego_yaw))
        ego_left = (-ego_forward[1], ego_forward[0])
        ego_center = (
            ego_rear.x + 0.5 * parameters.VEHICLE_LENGTH * ego_forward[0],
            ego_rear.y + 0.5 * parameters.VEHICLE_LENGTH * ego_forward[1],
        )
        ego_extents = (
            0.5 * parameters.VEHICLE_LENGTH + parameters.COLLISION_MARGIN,
            0.5 * parameters.VEHICLE_WIDTH + parameters.COLLISION_MARGIN,
        )
        object_forward = (math.cos(object_yaw), math.sin(object_yaw))
        object_left = (-object_forward[1], object_forward[0])
        object_extents = (
            0.5 * max(1.0, float(getattr(object_info, "size_x", 0.0))),
            0.5 * max(1.0, float(getattr(object_info, "size_y", 0.0))),
        )
        delta = (object_center.x - ego_center[0], object_center.y - ego_center[1])
        ego_axes = (ego_forward, ego_left)
        object_axes = (object_forward, object_left)
        for axis in ego_axes + object_axes:
            center_projection = abs(delta[0] * axis[0] + delta[1] * axis[1])
            ego_projection = sum(
                ego_extents[i]
                * abs(ego_axes[i][0] * axis[0] + ego_axes[i][1] * axis[1])
                for i in range(2)
            )
            object_projection = sum(
                object_extents[i]
                * abs(object_axes[i][0] * axis[0] + object_axes[i][1] * axis[1])
                for i in range(2)
            )
            if center_projection > ego_projection + object_projection:
                return False
        return True

    @staticmethod
    def _max_curvature(points):
        maximum = 0.0
        for index in range(1, len(points) - 1):
            first, middle, last = points[index - 1], points[index], points[index + 1]
            a = middle.distance(first)
            b = last.distance(middle)
            c = last.distance(first)
            denominator = a * b * c
            if denominator < 1e-9:
                continue
            twice_area = abs(
                (middle.x - first.x) * (last.y - first.y)
                - (middle.y - first.y) * (last.x - first.x)
            )
            maximum = max(maximum, 2.0 * twice_area / denominator)
        return maximum

    def reset_lateral_state(self):
        self.state = self.KEEP
        self.target_link_id = None
        self.source_link_id = None
        self.active_candidate = None
        self.changed_lane_hold = False
        self._lane_change_cooldown_origin = None
        self._last_lane_change_direction = None

    def _reset(self):
        self.reset_lateral_state()
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
