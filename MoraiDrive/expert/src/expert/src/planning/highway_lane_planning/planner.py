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
        self.state = self.KEEP
        self.target_link_id = None
        self.source_link_id = None
        self.active_candidate = None
        self.changed_lane_hold = False
        self._merge_speed_limit_state = float("inf")
        self._last_merge_debug = []
        self._last_lane_checks = []
        self._merge_decisions = {}
        self._merge_go_object_ids = set()

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
                self.state = self.KEEP
                self.source_link_id = None
                self.active_candidate = None
                self.changed_lane_hold = True
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
        avoidance_needed = (
            proactive_lane_change or merge_conflict or not keep_candidate.valid
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
            if math.isinf(acc_target_velocity):
                acc_target_velocity = min(
                    vehicle_state.velocity,
                    parameters.ACC_UNMATCHED_HAZARD_SPEED,
                )
            acc_target_velocity = min(
                acc_target_velocity,
                parameters.ACC_PREPARE_TARGET_SPEED,
            )
            acc_ready = self.state == self.ACC and self._ready_for_lane_change(
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

        if not valid_changes:
            self.state = self.ACC
            return HighwayPlanningResult(
                lane_follow_path,
                obstacle_detected=not keep_candidate.valid,
                acc_override=True,
                acc_target_velocity=acc_target_velocity,
                highway_state=self.ACC,
                candidates=candidates,
                reason="highway_lane_no_safe_change",
                **common
            )

        best = min(valid_changes, key=lambda candidate: candidate.cost)
        self.source_link_id = current_link.id
        self.target_link_id = best.target_link_id
        self.active_candidate = best
        self.state = self.CHANGE_LEFT if best.direction == "left" else self.CHANGE_RIGHT
        common["target_link_id"] = self.target_link_id
        return HighwayPlanningResult(
            best.points,
            active=True,
            obstacle_detected=True,
            acc_target_velocity=parameters.ACC_LANE_CHANGE_MAX_SPEED,
            highway_state=self.state,
            selected_offset=best.offset,
            candidates=candidates,
            reason="highway_lane_safe_change",
            **common
        )

    @staticmethod
    def _ready_for_lane_change(ego_speed, gap, ttc):
        if ego_speed > parameters.ACC_LANE_CHANGE_MAX_SPEED:
            return False
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
            if object_match.lane_link.id != target_link.id:
                continue
            signed_distance = object_match.s - ego_projection.s
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

    def _merge_acc_speed_limit(
        self, vehicle_state, ego_match, object_matches, vehicle_entries=()
    ):
        """Return a speed ceiling when two different links reach one merge."""
        ego_link = ego_match.lane_link
        self._last_merge_debug = []
        self._merge_go_object_ids = set()
        ego_distance = max(0.0, ego_link.length - ego_match.s)
        ego_speed = max(0.0, float(vehicle_state.velocity))
        if ego_speed < 0.1:
            return float("inf")

        ego_eta = ego_distance / max(0.1, ego_speed)
        speed_limit = float("inf")
        ego_successors = set(self.lane_map.successors.get(ego_link.id, ()))
        conservative_successors = ego_successors.intersection(
            parameters.CONSERVATIVE_MERGE_SUCCESSORS
        )
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        conservative_zone = (
            bool(conservative_successors)
            and vehicle_state.position.distance(merge_point)
            <= parameters.CONSERVATIVE_MERGE_EGO_RADIUS
        )
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
        recovery_step = (
            parameters.MERGE_ACC_RECOVERY_RATE
            / parameters.PLANNER_UPDATE_RATE
        )
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
        speed_limit = (
            lead_speed
            + parameters.ACC_DISTANCE_GAIN * (gap - desired_gap)
        )
        return max(0.0, min(float(ego_speed), speed_limit))

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
        return HighwayPlanningResult(
            candidate.points,
            active=True,
            obstacle_detected=collision_during_change,
            acc_override=collision_during_change,
            acc_target_velocity=(
                acc_target_velocity
                if collision_during_change
                else parameters.ACC_LANE_CHANGE_MAX_SPEED
            ),
            # Keep the lateral maneuver state after commitment. ACC may cap
            # speed at the same time, but a newly observed vehicle must not
            # discard the active path or disable lane-change steering limits.
            highway_state=self.state,
            selected_offset=candidate.offset,
            candidates=[candidate],
            reason=(
                "highway_lane_change_acc"
                if collision_during_change
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

    def _reset(self):
        self.reset_lateral_state()
        self._merge_speed_limit_state = float("inf")
        self._merge_decisions = {}
        self._merge_go_object_ids = set()

    def reset(self):
        self._reset()
