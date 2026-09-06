#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zone-aware process wrapper for static and highway lattice planning."""

from . import parameters
from .lattice_planner import LatticePlanner
from .structures import LatticePlanningResult
from ..highway_lane_planning import HighwayLanePlanner


# 신호등·정적장애물 구간의 중심선과 활성 반폭 [m]
STATIC_OBSTACLE_ZONE_START = (-60.18083478654391, -162.36788516811984)
STATIC_OBSTACLE_ZONE_END = (-60.057962937774064, -54.031780525899286)
STATIC_OBSTACLE_ZONE_HALF_WIDTH = 3.0

# 사용자가 측정한 고속도로 1,280점에서 100점 간격으로 추린 중심선입니다.
# 시작/끝 두 점만 잇지 않고 굴곡을 따라 구간 진입 여부를 판정합니다.
HIGHWAY_ZONE_CENTERLINE = (
    (58.094459788788775, 294.7137957632885),
    (62.24592546385352, 245.99639621904632),
    (62.529043341556694, 195.99795240535693),
    (63.68936341522996, 146.01971486184382),
    (66.59782941339266, 96.10501210884534),
    (67.16688948747576, 46.69911408341383),
    (67.30198107141482, -2.374116308774615),
    (67.48668301933553, -52.37363060058683),
    (67.74967001916039, -102.37253327899825),
    (68.51592550361202, -151.37422127003208),
    (73.55240368475383, -201.1185921310922),
    (72.38174073422384, -250.21225473270576),
    (70.32717092510104, -300.11746653752044),
    (70.41629737914103, -339.6167416577021),
)

# From this measured point, the official global path selects the dedicated
# right-turn/tollgate lane (MGeo link A2256W000153). Disable highway lateral
# planning only inside this downstream corridor so it cannot pull the vehicle
# back toward a previously selected lane.
HIGHWAY_RIGHT_TURN_CENTERLINE = (
    (75.41420370276319, -217.855053961277),    # merge end / A2256W000153 start
    (70.31, -327.68),                          # user-measured right-turn point
    (70.21414883446414, -365.74847655789927), # A2256W000153 end
)
HIGHWAY_RIGHT_TURN_HALF_WIDTH = 8.0


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
    ):
        # zone_* 인자는 기존 정적장애물 구간 테스트/설정을 위해 유지합니다.
        self.zone_bounds = zone_bounds
        self.zone_start = zone_start
        self.zone_end = zone_end
        self.zone_half_width = float(zone_half_width)
        self.planner = planner or LatticePlanner()

        self.highway_zone_points = tuple(highway_zone_points or ())
        self.highway_zone_half_width = float(highway_zone_half_width)
        # 정적 우측 선호의 이전 offset이 고속도로 비용에 섞이지 않도록 별도 인스턴스 사용
        self.highway_planner = highway_planner or LatticePlanner()
        self.highway_lane_planner = highway_lane_planner or (
            HighwayLanePlanner(map_data_dir) if map_data_dir else None
        )
        self._highway_acc_override = False

        self.last_result = None
        self._last_log_state = None

    def run(self, vehicle_state, reference_path, object_info_list):
        if self._is_inside_static_zone(vehicle_state):
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset()
            result = self._run_static_zone(
                vehicle_state, reference_path, object_info_list
            )
        elif self._is_inside_highway_zone(vehicle_state):
            result = self._run_highway_zone(
                vehicle_state, reference_path, object_info_list
            )
        else:
            self._highway_acc_override = False
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset()
            reason = "zone_not_configured" if not self._has_any_zone() else "outside_zone"
            result = LatticePlanningResult(reference_path, reason=reason)

        self._update_log(result)
        self.last_result = result
        return result

    def _run_static_zone(self, vehicle_state, reference_path, object_info_list):
        static_obstacles = [
            item["object_info"]
            for item in object_info_list
            if getattr(item["object_info"], "is_static", False)
        ]
        return self.planner.plan(vehicle_state, reference_path, static_obstacles)

    def _run_highway_zone(self, vehicle_state, reference_path, object_info_list):
        vehicle_entries = [
            item
            for item in object_info_list
            if not getattr(item["object_info"], "is_static", False)
            and getattr(item["object_info"], "type", -1) in (1, 2)
        ]
        if self._is_inside_highway_right_turn_zone(vehicle_state):
            self._highway_acc_override = False
            if self.highway_lane_planner is not None:
                self.highway_lane_planner.reset_lateral_state()
                safety_result = self.highway_lane_planner.plan(
                    vehicle_state, reference_path, vehicle_entries
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
                vehicle_state, reference_path, vehicle_entries
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

    def _update_log(self, result):
        state = (
            result.mode,
            result.active,
            result.acc_override,
            result.reason,
            result.long_obstacle_detected,
            round(result.selected_offset, 2),
        )
        if state == self._last_log_state:
            return
        self._last_log_state = state

        if result.acc_override:
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
