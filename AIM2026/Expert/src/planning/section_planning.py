#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zone-aware process wrapper for static and highway lattice planning."""

import json
import math
from ..obstacle.object_info import is_ve_tracked_unknown

from ..config.parameters import lattice as parameters
from .lattice.lattice_planner import LatticePlanner
from .roundabout.planner import RoundaboutEntryPlanner
from .static_obstacle.avoidance import StaticAvoidance
from .lattice.centerline_lattice import CenterlineLattice
from .lattice.structures import LatticePlanningResult
from .highway import HighwayLanePlanner


# 구간 활성 영역은 config/parameters/sections.py에서 관리한다.
from ..config.parameters.sections import (
    STATIC_OBSTACLE_ZONE_START,
    STATIC_OBSTACLE_ZONE_END,
    STATIC_OBSTACLE_ZONE_HALF_WIDTH,
    HIGHWAY_ZONE_CENTERLINE,
    HIGHWAY_RIGHT_TURN_CENTERLINE,
    HIGHWAY_RIGHT_TURN_HALF_WIDTH,
)

class LatticePlanningProcess:
    """Select the static-zone policy or highway Lattice/ACC policy."""

    def __init__(
        self,
        zone_bounds=None,
        zone_start=STATIC_OBSTACLE_ZONE_START,
        zone_end=STATIC_OBSTACLE_ZONE_END,
        zone_half_width=STATIC_OBSTACLE_ZONE_HALF_WIDTH,
        planner=None,
        highway_zone_points=HIGHWAY_ZONE_CENTERLINE,
        highway_zone_half_width=parameters.HIGHWAY_ZONE_HALF_WIDTH,
        highway_planner=None,
        map_data_dir=None,
        highway_lane_planner=None,
        jamming_road_boundary=None,
        jamming_enabled=True,
    ):
        # zone_* 인자는 기존 정적장애물 구간 테스트/설정을 위해 유지합니다.
        self.zone_bounds = zone_bounds
        self.zone_start = zone_start
        self.zone_end = zone_end
        self.zone_half_width = float(zone_half_width)
        self.planner = planner or LatticePlanner(sample_spacing=parameters.STATIC_SAMPLE_SPACING)
        self.static_avoidance = StaticAvoidance(self.planner)
        from .static_obstacle.recorded_planner import RecordedStaticPlanner
        self.recorded_static = (RecordedStaticPlanner(self.planner, background=True).use_static_zone_policy()
                                if zone_bounds is None and zone_start == STATIC_OBSTACLE_ZONE_START
                                else None)
        from .jamming.recorded_planner import JammingPlanner
        from .jamming.road_boundary import RoadBoundary
        if (jamming_enabled and jamming_road_boundary is None and map_data_dir is not None
                and zone_bounds is None and zone_start == STATIC_OBSTACLE_ZONE_START):
            jamming_road_boundary = RoadBoundary.from_map_dir(map_data_dir)
        self.recorded_jamming = (JammingPlanner(background=True, road_boundary=jamming_road_boundary)
                                 if jamming_enabled and zone_bounds is None
                                 and zone_start == STATIC_OBSTACLE_ZONE_START
                                 else None)
        self.centerline_lattice = CenterlineLattice(map_data_dir)
        self.static_corridor = None   # set by attach_static_map(): receding-horizon static section planning
        self.map_data_dir = map_data_dir

        self.highway_zone_points = tuple(highway_zone_points or ())
        self.highway_zone_half_width = float(highway_zone_half_width)
        # 정적 우측 선호의 이전 offset이 고속도로 비용에 섞이지 않도록 별도 인스턴스 사용
        self.highway_planner = highway_planner or LatticePlanner()
        self.highway_lane_planner = highway_lane_planner or (
            HighwayLanePlanner(map_data_dir) if map_data_dir else None
        )
        self._highway_acc_override = False
        self._roundabout_entry_committed = False
        self.roundabout_planner = RoundaboutEntryPlanner(map_data_dir)

        self.last_result = None
        self._last_log_state = None

    def attach_static_map(self, occupancy, tunnel_map=None, global_path=None):
        """Plan standing obstacles every tick on accumulated occupancy maps: the static-obstacle section on
        ``occupancy`` (static_map.py), the Jamming section on ``tunnel_map`` (tunnel_map.py). Elsewhere nothing
        changes. With the global route both planners are set up now, not on their first tick in the section."""
        from .static_obstacle.recorded_planner import RecordedStaticPlanner
        from .static_obstacle.corridor_optimizer import StaticCorridorPlanner
        if self.recorded_static is not None and occupancy is not None:
            self.static_corridor = StaticCorridorPlanner(RecordedStaticPlanner(self.planner), occupancy,
                                                         map_data_dir=self.map_data_dir)
            if global_path is not None:
                self.static_corridor.prepare(list(global_path))
        if self.recorded_jamming is not None and tunnel_map is not None:
            self.recorded_jamming.attach_map(tunnel_map, global_path)
        return self.static_corridor

    def set_static_editor_mode(self, enabled):
        """Enable scene-change replanning only for the interactive editor."""
        if self.recorded_static is not None:
            self.recorded_static.set_live_preview(enabled)
        if self.recorded_jamming is not None:
            self.recorded_jamming.set_live_preview(enabled)

    def set_static_editor_driving(self, enabled):
        """Commit the current editor route to normal Expert tracking."""
        if self.recorded_static is not None:
            self.recorded_static.set_live_drive(enabled)
        if self.recorded_jamming is not None:
            self.recorded_jamming.set_live_drive(enabled)

    def run(self, vehicle_state, reference_path, object_info_list, desired_velocity=None,
            prediction_path=None, global_path=None, ego_header_seq=None, ego_source_ns=None):
        static_objects = [item['object_info'] for item in object_info_list
                          if getattr(item['object_info'], 'is_static', False)
                          and -75 < item['object_info'].position.x < -40
                          and -180 < item['object_info'].position.y < -90]
        corridor = (self.static_corridor is not None and global_path is not None
                    and self.static_corridor.contains(vehicle_state))
        if self.recorded_static is not None and not corridor:
            self.recorded_static.refresh_live_preview(vehicle_state, static_objects)
            self.recorded_static.preplan(vehicle_state,
                global_path if global_path is not None else (prediction_path or reference_path), static_objects)
        approach_result = (self.recorded_static.approach_result(vehicle_state, reference_path)
                           if self.recorded_static is not None and not corridor else None)
        roundabout_result = self._run_roundabout_entry_guard(
            vehicle_state, reference_path, object_info_list, desired_velocity,
            prediction_path,
        )
        if roundabout_result is not None:
            self.static_avoidance.reset()
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset()
            result = roundabout_result
        elif self.recorded_jamming is not None and (jamming_result := self.recorded_jamming.run(
                vehicle_state, reference_path, object_info_list, prediction_path, global_path,
                desired_velocity=desired_velocity)) is not None:
            self.static_avoidance.reset()
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset()
            result = jamming_result
        elif corridor and (corridor_result := self.static_corridor.run(
                vehicle_state, reference_path, global_path)) is not None:
            self.static_avoidance.reset()
            result = corridor_result
        elif self.recorded_static is not None and (
                (self.recorded_static.interactive_preview and (
                    self.recorded_static.active
                    or self.recorded_static.pending is not None
                    or bool(static_objects)))
                or self.recorded_static.active or (
                self.recorded_static.contains(vehicle_state) and
                (self.recorded_static.pending is not None or static_objects))):
            result = self.recorded_static.plan(vehicle_state, reference_path, static_objects, prediction_path)
        elif approach_result is not None:
            result = approach_result
        elif (not self.static_avoidance.active and self.zone_bounds is None
              and self.centerline_lattice.contains(vehicle_state)):
            result = self.centerline_lattice.run(vehicle_state, reference_path, object_info_list)
        elif (self.recorded_static is not None and self._is_inside_static_zone(vehicle_state)
              and self.recorded_static.past_rejoin(vehicle_state)):
            self.static_avoidance.reset()
            result = LatticePlanningResult(reference_path, mode="static_obstacle", reason="static_recorded_rejoined")
        elif self._is_inside_static_zone(vehicle_state) or self.static_avoidance.active:
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset()
            result = self._run_static_zone(
                vehicle_state, reference_path, object_info_list, prediction_path
            )
        elif self._is_inside_highway_zone(vehicle_state):
            result = self._run_highway_zone(
                vehicle_state, reference_path, object_info_list, desired_velocity,
                prediction_path,
            )
        else:
            self._highway_acc_override = False
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset()
            reason = "zone_not_configured" if not self._has_any_zone() else "outside_zone"
            result = LatticePlanningResult(reference_path, reason=reason)

        self._update_log(result, vehicle_state, ego_header_seq, ego_source_ns)
        self.last_result = result
        return result

    def _run_static_zone(self, vehicle_state, reference_path, object_info_list, prediction_path=None):
        if (not self.static_avoidance.active and self.zone_bounds is None
                and self.zone_start == STATIC_OBSTACLE_ZONE_START):
            stop_x, stop_y = parameters.STATIC_PLANNING_STOP_POINT
            dx, dy = self.zone_end[0] - self.zone_start[0], self.zone_end[1] - self.zone_start[1]
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            progress = ((vehicle_state.position.x - stop_x) * ux
                        + (vehicle_state.position.y - stop_y) * uy)
            forward = math.cos(vehicle_state.yaw) * ux + math.sin(vehicle_state.yaw) * uy
            side = -math.sin(vehicle_state.yaw) * ux + math.cos(vehicle_state.yaw) * uy
            # First contact of the physical front bumper, including heading.
            # Use the same reference origin as the ego collision box, without its safety margin.
            bumper_progress = (progress + self.planner.vehicle_length * forward
                               + self.planner.vehicle_half_width * abs(side))
            before_line = bumper_progress < -parameters.STATIC_PLANNING_STOP_TOLERANCE
            if before_line or forward < 0.5:
                # Keep the original path and normal ACC/signal limits before the
                # stop line. Only new lateral avoidance is gated here.
                return LatticePlanningResult(reference_path, mode='static_obstacle',
                    reason='static_wrong_direction' if forward < 0.5 else 'static_before_stop_line',
                    debug_details={'static_avoidance': {
                        'phase': 'wait_stop_line', 'path_validated': False,
                        'stop_line_node': 'A1256W000807', 'stop_line_progress': bumper_progress,
                        'stop_line_reference': 'front_bumper'}})
        static_obstacles = [
            item["object_info"]
            for item in object_info_list
            if getattr(item["object_info"], "is_static", False)
        ]
        return self.static_avoidance.plan(
            vehicle_state, reference_path, static_obstacles, prediction_path
        )

    def _run_roundabout_entry_guard(
        self, vehicle_state, reference_path, object_info_list, desired_velocity=None,
        prediction_path=None,
    ):
        state = self._roundabout_entry_state(vehicle_state)
        result = self.roundabout_planner.plan(
            vehicle_state, prediction_path if prediction_path is not None else reference_path,
            object_info_list, state, desired_velocity
        )
        if result is not None:
            result.path = reference_path
        self._roundabout_entry_committed = self.roundabout_planner.committed
        return result

    def _run_highway_zone(self, vehicle_state, reference_path, object_info_list,
                          desired_velocity=None, prediction_path=None):
        vehicle_entries = [
            item
            for item in object_info_list
            if getattr(item["object_info"], "is_static", False)
            or getattr(item["object_info"], "type", -1) in (1, 2)
            or is_ve_tracked_unknown(item["object_info"])
        ]
        if self._is_inside_highway_right_turn_zone(vehicle_state):
            self._highway_acc_override = False
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset_lateral_state()
                safety_result = self.highway_lane_planner.plan(
                    vehicle_state, reference_path, vehicle_entries,
                    desired_velocity=desired_velocity, prediction_path=prediction_path,
                    allow_lane_change=False,
                )
                self.highway_lane_planner.reset_lateral_state()
                if safety_result.acc_override:
                    safety_result.path = reference_path
                    safety_result.active = False
                    safety_result.candidates = []
                    safety_result.reason = (
                        "highway_right_turn_" + safety_result.reason
                    )
                    return safety_result
            return LatticePlanningResult(
                reference_path,
                mode="highway_lane",
                reason="highway_right_turn_global_path",
            )
        if self.highway_lane_planner is not None:
            return self.highway_lane_planner.plan(
                vehicle_state, reference_path, vehicle_entries,
                desired_velocity=desired_velocity, prediction_path=prediction_path,
            )

        # map_data_dir를 넣지 않는 단위 테스트/호환 실행에서만 기존 offset 방식 사용
        nearest_distance, ttc = self._assess_highway_risk(
            vehicle_state, vehicle_entries
        )
        self._update_highway_acc_override(nearest_distance, ttc)

        if self._highway_acc_override:
            # 기준 경로를 넘기면 기존 ACC가 동일 차로의 급위험 차량을 우선 처리합니다.
            return LatticePlanningResult(
                reference_path,
                obstacle_detected=True,
                mode="highway",
                acc_override=True,
                nearest_vehicle_distance=nearest_distance,
                ttc=ttc,
                reason="highway_acc_override",
            )

        result = self.highway_planner.plan(
            vehicle_state,
            reference_path,
            [item["object_info"] for item in vehicle_entries],
            prefer_right=False,
            longitudinal_speed_scale=parameters.HIGHWAY_LATTICE_SPEED_SCALE,
            inactive_reason="no_relevant_highway_vehicle",
            active_reason="highway_vehicle_lattice",
            mode="highway",
        )
        result.nearest_vehicle_distance = nearest_distance
        result.ttc = ttc
        return result

    def roundabout_entry_should_ignore_legacy_acc(self, vehicle_state, object_info):
        return (
            self.roundabout_planner.active
            and not getattr(object_info, "is_static", False)
            and (getattr(object_info, "type", -1) in (1, 2) or is_ve_tracked_unknown(object_info))
        )

    def _roundabout_entry_active(self, vehicle_state, reference_path):
        return self._roundabout_entry_state(vehicle_state)["in_decision_zone"]

    def _roundabout_entry_state(self, vehicle_state):
        approach_start = parameters.ROUNDABOUT_ENTRY_APPROACH_START
        approach_end = parameters.ROUNDABOUT_ENTRY_APPROACH_END
        stop_point = parameters.ROUNDABOUT_ENTRY_STOP_POINT
        along, lateral, segment_length = self._segment_coordinates(
            vehicle_state.position, approach_start, approach_end
        )
        stop_along, _, _ = self._segment_coordinates(
            stop_point, approach_start, approach_end
        )
        hold_along, _, _ = self._segment_coordinates(
            parameters.ROUNDABOUT_ENTRY_HOLD_POINT, approach_start, approach_end
        )
        in_approach = (
            0.0 <= along <= segment_length
            and lateral <= parameters.ROUNDABOUT_ENTRY_APPROACH_HALF_WIDTH
        )
        decision_start = hold_along - parameters.ROUNDABOUT_ENTRY_STOP_BACK_DISTANCE
        decision_end = hold_along + parameters.ROUNDABOUT_ENTRY_STOP_FORWARD_DISTANCE
        in_decision_zone = in_approach and decision_start <= along <= decision_end
        past_go_line = in_approach and along > decision_end
        return {
            "active": in_decision_zone,
            "ego_along": along,
            "ego_lateral": lateral,
            "approach_length": segment_length,
            "stop_along": stop_along,
            "hold_along": hold_along,
            "decision_start": decision_start,
            "decision_end": decision_end,
            "in_approach": in_approach,
            "in_decision_zone": in_decision_zone,
            "past_go_line": past_go_line,
            "committed": self._roundabout_entry_committed,
            "ego_distance": abs(hold_along - along),
        }

    @classmethod
    def _point_in_segment_band(cls, point, start, end, half_width):
        along, lateral, segment_length = cls._segment_coordinates(point, start, end)
        return 0.0 <= along <= segment_length and lateral <= half_width

    @staticmethod
    def _segment_coordinates(point, start, end):
        point_x = float(point[0]) if isinstance(point, tuple) else point.x
        point_y = float(point[1]) if isinstance(point, tuple) else point.y
        start_x, start_y = start
        end_x, end_y = end
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return 0.0, math.hypot(point_x - start_x, point_y - start_y), 0.0
        relative_x = point_x - start_x
        relative_y = point_y - start_y
        along = (relative_x * dx + relative_y * dy) / length
        lateral = abs(relative_x * dy - relative_y * dx) / length
        return along, lateral, length

    @classmethod
    def _polyline_coordinates(cls, point, polyline):
        if len(polyline) < 2:
            return None, float("inf"), 0.0
        best_progress = None
        best_lateral = float("inf")
        progress_before = 0.0
        total_length = 0.0
        for start, end in zip(polyline[:-1], polyline[1:]):
            _, _, segment_length = cls._segment_coordinates(start, start, end)
            total_length += segment_length

        for start, end in zip(polyline[:-1], polyline[1:]):
            along, lateral, segment_length = cls._segment_coordinates(point, start, end)
            clamped_along = max(0.0, min(segment_length, along))
            distance_sq = lateral ** 2 + (along - clamped_along) ** 2
            if distance_sq < best_lateral ** 2:
                best_lateral = math.sqrt(distance_sq)
                best_progress = progress_before + clamped_along
            progress_before += segment_length
        return best_progress, best_lateral, total_length

    @staticmethod
    def _assess_highway_risk(vehicle_state, vehicle_entries):
        """Return the minimum same-lane bumper gap and minimum TTC."""
        nearest_distance = float("inf")
        minimum_ttc = float("inf")
        for item in vehicle_entries:
            local_position = item.get("local_position")
            if local_position is None:
                continue
            if local_position.x <= 0.0:
                continue
            if abs(local_position.y) > parameters.HIGHWAY_RISK_LATERAL_THRESHOLD:
                continue

            # ACC와 동일하게 물체 중심거리에서 Ego 길이를 빼 앞 범퍼 간격으로 사용
            distance = max(0.0, local_position.x - parameters.VEHICLE_LENGTH)
            object_velocity = float(item["object_info"].velocity)
            closing_speed = max(0.0, vehicle_state.velocity - object_velocity)
            ttc = (
                distance / closing_speed
                if closing_speed > 0.1
                else float("inf")
            )
            nearest_distance = min(nearest_distance, distance)
            minimum_ttc = min(minimum_ttc, ttc)
        return nearest_distance, minimum_ttc

    def _update_highway_acc_override(self, distance, ttc):
        if distance == float("inf"):
            self._highway_acc_override = False
            return

        if self._highway_acc_override:
            # 거리와 TTC가 모두 해제 기준을 벗어나야 Lattice로 돌아갑니다.
            self._highway_acc_override = (
                distance < parameters.HIGHWAY_ACC_RELEASE_DISTANCE
                or ttc < parameters.HIGHWAY_ACC_RELEASE_TTC
            )
        else:
            self._highway_acc_override = (
                distance <= parameters.HIGHWAY_ACC_ENTER_DISTANCE
                or ttc <= parameters.HIGHWAY_ACC_ENTER_TTC
            )

    def _is_inside_static_zone(self, vehicle_state):
        x = vehicle_state.position.x
        y = vehicle_state.position.y
        if self.zone_bounds is not None:
            min_x, max_x, min_y, max_y = self.zone_bounds
            return min_x <= x <= max_x and min_y <= y <= max_y
        if self.zone_start is None or self.zone_end is None:
            return False
        return self._distance_sq_to_segment(
            x, y, self.zone_start, self.zone_end
        ) <= self.zone_half_width ** 2

    # 기존 호출/테스트 호환용: _is_inside_zone은 정적장애물 구간을 의미합니다.
    def _is_inside_zone(self, vehicle_state):
        return self._is_inside_static_zone(vehicle_state)

    def _is_inside_highway_zone(self, vehicle_state):
        if self.highway_lane_planner is not None:
            return self.highway_lane_planner.lane_map.match(
                vehicle_state.position,
                heading=vehicle_state.yaw,
            ) is not None
        if len(self.highway_zone_points) < 2:
            return False
        x = vehicle_state.position.x
        y = vehicle_state.position.y
        threshold_sq = self.highway_zone_half_width ** 2
        return any(
            self._distance_sq_to_segment(x, y, start, end) <= threshold_sq
            for start, end in zip(
                self.highway_zone_points[:-1], self.highway_zone_points[1:]
            )
        )

    @staticmethod
    def _is_inside_highway_right_turn_zone(vehicle_state):
        """Use a directed corridor so the policy never activates before start."""
        start_x, start_y = HIGHWAY_RIGHT_TURN_CENTERLINE[0]
        next_x, next_y = HIGHWAY_RIGHT_TURN_CENTERLINE[1]
        segment_x = next_x - start_x
        segment_y = next_y - start_y
        length_sq = segment_x * segment_x + segment_y * segment_y
        relative_x = vehicle_state.position.x - start_x
        relative_y = vehicle_state.position.y - start_y
        start_progress = (
            relative_x * segment_x + relative_y * segment_y
        ) / length_sq
        if start_progress < 0.0:
            return False
        return any(
            LatticePlanningProcess._distance_sq_to_segment(
                vehicle_state.position.x,
                vehicle_state.position.y,
                start,
                end,
            ) <= HIGHWAY_RIGHT_TURN_HALF_WIDTH ** 2
            for start, end in zip(
                HIGHWAY_RIGHT_TURN_CENTERLINE[:-1],
                HIGHWAY_RIGHT_TURN_CENTERLINE[1:],
            )
        )

    @staticmethod
    def _distance_sq_to_segment(x, y, start, end):
        start_x, start_y = start
        end_x, end_y = end
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        segment_length_sq = segment_x ** 2 + segment_y ** 2
        if segment_length_sq < 1e-9:
            return (x - start_x) ** 2 + (y - start_y) ** 2

        ratio = (
            (x - start_x) * segment_x + (y - start_y) * segment_y
        ) / segment_length_sq
        ratio = max(0.0, min(1.0, ratio))
        nearest_x = start_x + ratio * segment_x
        nearest_y = start_y + ratio * segment_y
        return (x - nearest_x) ** 2 + (y - nearest_y) ** 2

    def _has_any_zone(self):
        has_static_zone = self.zone_bounds is not None or (
            self.zone_start is not None and self.zone_end is not None
        )
        return has_static_zone or len(self.highway_zone_points) >= 2

    def _update_log(self, result, vehicle_state, ego_header_seq, ego_source_ns):
        if (result.mode == "roundabout_entry" and ego_header_seq is not None
                and ego_source_ns is not None):
            # Emitted every tick (not deduped) so the collected supervisor.log
            # carries a row for every sensor frame's exact (seq, source_ns) --
            # situation_labeling's expert_instructions.py joins on that exact
            # key and does not interpolate between sparse transitions.
            details = result.debug_details.get("roundabout_entry", {})
            print("[EXPERT_STATE] " + json.dumps({
                "ego_header_seq": ego_header_seq,
                "ego_source_ns": ego_source_ns,
                "mode": "roundabout_entry",
                "roundabout_phase": details.get("phase"),
                "roundabout_entering": details.get("entering"),
                "roundabout_committed": details.get("committed"),
                "roundabout_front_id": details.get("front_id"),
                "roundabout_conflict_id": details.get("blocker_unique_id"),
                "planner_target_speed_mps": result.acc_target_velocity,
                "ego_speed_mps": vehicle_state.velocity,
            }))
        state = (
            result.mode,
            result.active,
            result.acc_override,
            result.reason,
            result.long_obstacle_detected,
            round(result.selected_offset, 2),
        )
        if result.mode == "roundabout_entry":
            details = result.debug_details.get("roundabout_entry", {})
            state += (details.get("front_id"), details.get("blocker_unique_id"),
                      details.get("following_speed_adjusted", False))
        if state == self._last_log_state:
            return
        self._last_log_state = state

        if result.mode == "roundabout_entry":
            details = result.debug_details.get("roundabout_entry", {})
            print(
                "[ROUNDABOUT] {} target={:.1f}km/h conflict_id={} ttc={:.2f}s "
                "front_id={} gap={} hold_gap={:.2f}m entering={} merged={} "
                "ceiling={:.1f}km/h adjusted={}".format(
                    details.get("phase"), result.acc_target_velocity * 3.6,
                    details.get("blocker_unique_id"), result.ttc,
                    details.get("front_id"), details.get("front_gap"),
                    details.get("stop_gap", 0.0),
                    details.get("entering", False), details.get("committed", False),
                    details.get("speed_ceiling", result.acc_target_velocity) * 3.6,
                    details.get("following_speed_adjusted", False),
                )
            )
        elif result.mode == "static_obstacle":
            print("[STATIC] {} offset={:+.2f}m target={:.1f}km/h".format(
                result.reason, result.selected_offset, result.acc_target_velocity * 3.6
            ))
            details = result.debug_details.get("static_avoidance", {})
            if details.get("error") or result.reason == "static_recorded_no_complete_route":
                print("[STATIC_FAILURE] reason={} error={} search={} rejections={} "
                      "tracking_error_m={} heading_error_deg={}".format(
                          result.reason, details.get("error"), details.get("corridor_search"),
                          details.get("search_rejections"), details.get("tracking_error_m"),
                          details.get("heading_error_deg")))
        elif result.acc_override:
            ttc_text = "inf" if result.ttc == float("inf") else "{:.2f}s".format(result.ttc)
            print(
                "[LATTICE] highway ACC PRIORITY: gap={:.2f}m, TTC={}, reason={}".format(
                    result.nearest_vehicle_distance, ttc_text, result.reason
                )
            )
        elif result.active and result.mode == "highway_lane":
            print(
                "[LATTICE] highway lane planning ON: "
                "state={}, {} -> {}, offset={:+.2f}m".format(
                    result.highway_state,
                    result.current_link_id,
                    result.target_link_id,
                    result.selected_offset,
                )
            )
        elif result.active and result.mode == "highway":
            print(
                "[LATTICE] highway vehicle avoidance ON: "
                "selected_offset={:+.2f}m, cost-based (no forced side)".format(
                    result.selected_offset
                )
            )
        elif result.active:
            print(
                "[LATTICE] static obstacle avoidance ON: "
                "selected_offset={:+.2f}m (negative=right), "
                "long_obb={}, speed_scale={:.2f}".format(
                    result.selected_offset,
                    result.long_obstacle_detected,
                    result.longitudinal_speed_scale,
                )
            )
        elif result.reason == "all_candidates_blocked":
            print("[LATTICE] all candidates blocked; keeping reference path")
