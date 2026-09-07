# -*- coding: utf-8 -*-
"""합류 충돌구역 점유구간 판단 테스트.

정책: Ego 가 구역을 완전히 비우고 상대차가 들어오기까지 확실한 여유가
있을 때만 감속 없이 통과(GO)하고, 조금이라도 겹칠 것 같으면 양보(YIELD)한다.
단, 이미 구역에 들어갔거나 멈출 수 없는 거리라면 감속하지 않고 비워준다.
"""
import math
import os
import unittest

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning import parameters
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.merge_conflict import (
    MergeOccupancy,
    travel_time,
)
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.vehicle_state import VehicleState

FIRST_EGO_LINK = "A2256W000411"
FIRST_OTHER_LINK = "A2256W000409"
FIRST_MERGED_LINK = "A2256W000420"
SECOND_EGO_LINK = "A2256W000422"
SECOND_OTHER_LINK = "A2256W000445"


class MergeConflictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        expert_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.lane_map = HighwayLaneMap(os.path.normpath(
            os.path.join(expert_dir, "..", "..", "map_data", "R_KR_PG_KATRI_2025")
        ))
        cls.first_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        cls.second_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lane_map)

    # ----- fixtures ---------------------------------------------------------
    def _zone_s(self, link_id, center):
        return self.lane_map.project(center, self.lane_map.links[link_id]).s

    def _ego_at(self, link_id, center, back, velocity):
        s = max(0.0, self._zone_s(link_id, center) - back)
        point, yaw, _ = self.lane_map.point_at_s(self.lane_map.links[link_id], s)
        return VehicleState(point.x, point.y, yaw, velocity), s

    def _object_at(self, link_id, s, velocity, unique_id=1, ego_state=None):
        point, yaw, _ = self.lane_map.point_at_s(
            self.lane_map.links[link_id], max(0.0, s)
        )
        info = ObjectInfo(
            point.x, point.y, velocity, 1,
            name="2015_Kia_K5", size_x=4.5, size_y=2.0,
            heading_deg=math.degrees(yaw), unique_id=unique_id,
        )
        local = Point(1.0, 0.0)
        if ego_state is not None:
            dx = point.x - ego_state.position.x
            dy = point.y - ego_state.position.y
            local = Point(
                dx * math.cos(ego_state.yaw) + dy * math.sin(ego_state.yaw),
                -dx * math.sin(ego_state.yaw) + dy * math.cos(ego_state.yaw),
            )
        return {"object_info": info, "local_position": local}

    def _other_back(self, link_id, center, back, velocity, ego_state=None):
        return self._object_at(
            link_id,
            self._zone_s(link_id, center) - back,
            velocity,
            ego_state=ego_state,
        )

    def _reference_path(self, link_id, s, distance=140.0):
        link = self.lane_map.links[link_id]
        match = self.lane_map.project(link.points[0], link)
        match.s = s
        return [
            item[0]
            for item in self.lane_map.route_points(match, distance, 0.5)
        ]

    def _plan(self, link_id, center, ego_back, ego_v, entries):
        ego, s = self._ego_at(link_id, center, ego_back, ego_v)
        return ego, self.planner.plan(
            ego, self._reference_path(link_id, s), entries
        )

    def _merge_debug(self, result):
        return "\n".join(result.debug_details.get("merge_checks", []))

    # ----- 사용자가 요구한 정책 --------------------------------------------
    def test_goes_at_full_speed_when_overwhelmingly_first(self):
        """압도적으로 먼저 도착하면 감속하지 않는다."""
        other = self._other_back(FIRST_OTHER_LINK, self.first_point, 130.0, 20.0)
        _, result = self._plan(
            FIRST_EGO_LINK, self.first_point, 45.0, 20.0, [other]
        )

        self.assertEqual("EGO", self.planner._predictive_merge_priority)
        self.assertTrue(
            math.isinf(result.debug_details["raw_merge_limit"]),
            self._merge_debug(result),
        )

    def test_yields_when_arrival_windows_overlap(self):
        """도착 구간이 겹치면 양보한다."""
        ego_back, ego_v = 70.0, 20.0
        other = self._other_back(FIRST_OTHER_LINK, self.first_point, 62.0, 20.0)
        _, result = self._plan(
            FIRST_EGO_LINK, self.first_point, ego_back, ego_v, [other]
        )

        self.assertEqual(
            "OTHER", self.planner._predictive_merge_priority,
            self._merge_debug(result),
        )
        limit = result.debug_details["raw_merge_limit"]
        self.assertFalse(math.isinf(limit))
        self.assertLessEqual(limit, parameters.PREDICTIVE_MERGE_MAX_SPEED)

    def test_yields_when_only_slightly_first(self):
        """조금이라도 겹칠 것 같으면 먼저 도착해도 양보한다.

        이것이 사용자가 요구한 핵심 정책이다. 진입시점만 비교하던 이전
        구현은 여기서 GO 를 확정해 뒤에서 추돌당했다.
        """
        ego_back, ego_v = 70.0, 20.0
        # Ego 가 약간 먼저 도착하지만 구역을 비우기 전에 상대차가 들어온다.
        other = self._other_back(FIRST_OTHER_LINK, self.first_point, 78.0, 20.0)
        _, result = self._plan(
            FIRST_EGO_LINK, self.first_point, ego_back, ego_v, [other]
        )

        self.assertEqual(
            "OTHER", self.planner._predictive_merge_priority,
            self._merge_debug(result),
        )

    def test_requires_extra_lead_against_a_faster_vehicle(self):
        """상대차가 더 빠르면 합류 직후 추돌을 피할 만큼 앞서야 GO 한다."""
        ego_back, ego_v = 70.0, 15.0
        slow_other = self._other_back(
            FIRST_OTHER_LINK, self.first_point, 125.0, 15.0
        )
        _, slow_result = self._plan(
            FIRST_EGO_LINK, self.first_point, ego_back, ego_v, [slow_other]
        )
        slow_priority = self.planner._predictive_merge_priority

        self.planner = HighwayLanePlanner(lane_map=self.lane_map)
        fast_other = self._other_back(
            FIRST_OTHER_LINK, self.first_point, 125.0, 28.0
        )
        _, fast_result = self._plan(
            FIRST_EGO_LINK, self.first_point, ego_back, ego_v, [fast_other]
        )

        self.assertEqual("EGO", slow_priority, self._merge_debug(slow_result))
        self.assertEqual(
            "OTHER", self.planner._predictive_merge_priority,
            self._merge_debug(fast_result),
        )

    def test_far_vehicle_beyond_decision_horizon_does_not_slow_ego(self):
        """판단 지평 밖의 차량 때문에 감속하지 않는다(과도 감속 방지).

        구역까지 26초 걸리는 저속 차량은 지금 판단 대상이 아니다.
        """
        crawling_far_behind = self._other_back(
            FIRST_OTHER_LINK, self.first_point, 60.0, 2.0
        )
        _, result = self._plan(
            FIRST_EGO_LINK, self.first_point, 120.0, 20.0,
            [crawling_far_behind],
        )

        self.assertTrue(
            math.isinf(result.debug_details["raw_merge_limit"]),
            self._merge_debug(result),
        )

    # ----- 진입 후에는 감속하지 않는다 --------------------------------------
    def test_commits_to_clearing_once_inside_the_zone(self):
        """구역 안에서는 제동하지 않는다. 측면·후면을 내주게 된다."""
        other = self._other_back(FIRST_OTHER_LINK, self.first_point, 10.0, 20.0)
        _, result = self._plan(
            FIRST_EGO_LINK, self.first_point, 0.0, 15.0, [other]
        )

        self.assertEqual("EGO", self.planner._predictive_merge_priority)
        self.assertTrue(self.planner._merge_zone_committed)
        self.assertTrue(
            math.isinf(result.debug_details["raw_merge_limit"]),
            self._merge_debug(result),
        )

    def test_commits_to_clearing_past_the_point_of_no_return(self):
        """멈출 수 없는 거리에서는 어정쩡하게 감속하지 않고 통과한다."""
        ego_v = 25.0
        stopping = ego_v * ego_v / (2.0 * parameters.MERGE_YIELD_MAX_DECEL)
        # 정지거리보다 짧게 남은 지점
        back = parameters.PREDICTIVE_MERGE_ENTRY_RADIUS + stopping - 5.0
        other = self._other_back(FIRST_OTHER_LINK, self.first_point, 20.0, 25.0)
        _, result = self._plan(
            FIRST_EGO_LINK, self.first_point, back, ego_v, [other]
        )

        self.assertEqual("EGO", self.planner._predictive_merge_priority)
        self.assertIn("point of no return", self._merge_debug(result))

    def test_emergency_stop_remains_available_after_committing(self):
        """통과 확정 뒤에도 정면 충돌이 임박하면 정지한다."""
        ego, s = self._ego_at(SECOND_EGO_LINK, self.second_point, 0.0, 20.0)
        blocking = self._object_at(
            SECOND_OTHER_LINK,
            self._zone_s(SECOND_OTHER_LINK, self.second_point),
            15.0,
            ego_state=ego,
        )
        result = self.planner.plan(
            ego, self._reference_path(SECOND_EGO_LINK, s), [blocking]
        )

        self.assertTrue(self.planner._merge_zone_committed)
        self.assertTrue(result.acc_override)
        self.assertEqual("highway_lane_emergency_stop", result.reason)
        self.assertEqual(0.0, result.acc_target_velocity)

    # ----- 양보 목표속도 ---------------------------------------------------
    def test_yield_target_decreases_toward_the_zone_boundary(self):
        """양보 목표속도는 구역 경계에 가까워질수록 낮아진다."""
        limits = []
        for back in (90.0, 70.0, 55.0):
            self.planner = HighwayLanePlanner(lane_map=self.lane_map)
            other = self._other_back(
                FIRST_OTHER_LINK, self.first_point, 20.0, 8.0
            )
            _, result = self._plan(
                FIRST_EGO_LINK, self.first_point, back, 12.0, [other]
            )
            self.assertEqual("OTHER", self.planner._predictive_merge_priority)
            limits.append(result.debug_details["raw_merge_limit"])

        self.assertTrue(
            limits[0] > limits[1] > limits[2],
            "yield limits should tighten as Ego closes in: {}".format(limits),
        )

    def test_stalled_vehicle_in_the_zone_brings_ego_to_a_stop(self):
        """구역 안에 멈춰 있는 차량 앞에서는 경계까지 감속해 정지한다.

        멀리서 목표속도를 즉시 0 으로 떨어뜨리면 급제동이 되므로, 정지 곡선을
        따라 내려가 경계 근처에서 0 이 되어야 한다.
        """
        stalled_s = self._zone_s(FIRST_OTHER_LINK, self.first_point)
        limits = []
        for back in (40.0, 20.0):
            self.planner = HighwayLanePlanner(lane_map=self.lane_map)
            stalled = self._object_at(FIRST_OTHER_LINK, stalled_s, 0.0)
            _, result = self._plan(
                FIRST_EGO_LINK, self.first_point, back, 6.0, [stalled]
            )
            self.assertEqual(
                "OTHER", self.planner._predictive_merge_priority,
                self._merge_debug(result),
            )
            limit = result.debug_details["raw_merge_limit"]
            self.assertFalse(math.isinf(limit))
            limits.append(limit)

        self.assertGreater(limits[0], limits[1], limits)
        self.assertLess(limits[1], 10.0, self._merge_debug(result))

    def test_yield_releases_after_the_blocking_vehicle_clears(self):
        """막고 있던 차량이 합류 링크를 지나면 양보를 해제한다."""
        blocking = self._other_back(
            FIRST_OTHER_LINK, self.first_point, 25.0, 12.0
        )
        self._plan(FIRST_EGO_LINK, self.first_point, 80.0, 12.0, [blocking])
        self.assertEqual("OTHER", self.planner._predictive_merge_priority)

        cleared = self._object_at(FIRST_MERGED_LINK, 60.0, 20.0)
        hold_cycles = int(
            parameters.MERGE_ZONE_DROPOUT_HOLD_TIME
            * parameters.PLANNER_UPDATE_RATE
        ) + 3
        for _ in range(hold_cycles):
            _, result = self._plan(
                FIRST_EGO_LINK, self.first_point, 80.0, 12.0, [cleared]
            )
        self.assertEqual(
            "EGO", self.planner._predictive_merge_priority,
            self._merge_debug(result),
        )

    def test_one_frame_dropout_holds_the_yield(self):
        """막고 있던 차량이 한 프레임 사라져도 양보를 유지한다."""
        blocking = self._other_back(
            FIRST_OTHER_LINK, self.first_point, 25.0, 12.0
        )
        self._plan(FIRST_EGO_LINK, self.first_point, 80.0, 12.0, [blocking])
        self.assertEqual("OTHER", self.planner._predictive_merge_priority)

        _, result = self._plan(
            FIRST_EGO_LINK, self.first_point, 80.0, 12.0, []
        )
        self.assertEqual(
            "OTHER", self.planner._predictive_merge_priority,
            self._merge_debug(result),
        )

    # ----- 상대차 식별 -----------------------------------------------------
    def test_lead_vehicle_on_own_route_is_not_a_merge_opponent(self):
        """같은 차선 선행차는 합류 상대가 아니라 ACC 대상이다."""
        config = self.planner._first_merge_state.config
        self.assertFalse(
            self.planner._merges_into_zone(
                FIRST_EGO_LINK, FIRST_EGO_LINK, config
            )
        )
        self.assertTrue(
            self.planner._merges_into_zone(
                FIRST_EGO_LINK, FIRST_OTHER_LINK, config
            )
        )

    def test_object_key_distinguishes_identical_car_models(self):
        """MORAI name 은 차량 모델명이라 중복된다. unique_id 로 구분해야 한다."""
        first = self._object_at(FIRST_OTHER_LINK, 10.0, 10.0, unique_id=11)
        second = self._object_at(FIRST_OTHER_LINK, 40.0, 10.0, unique_id=22)
        self.assertEqual(
            first["object_info"].name, second["object_info"].name
        )
        self.assertNotEqual(
            self.planner._merge_zone_object_key(first, None),
            self.planner._merge_zone_object_key(second, None),
        )

    # ----- 단위/기하 -------------------------------------------------------
    def test_travel_time_accounts_for_acceleration(self):
        """정지 상태의 Ego 도 가속을 반영해야 합류 교착이 생기지 않는다."""
        self.assertTrue(math.isinf(travel_time(50.0, 0.0, 0.0, 20.0)))
        accelerating = travel_time(50.0, 0.0, 1.5, 20.0)
        self.assertFalse(math.isinf(accelerating))
        self.assertAlmostEqual(math.sqrt(2.0 * 50.0 / 1.5), accelerating, delta=0.1)
        # 이미 최고속도면 등속과 같다
        self.assertAlmostEqual(
            50.0 / 20.0, travel_time(50.0, 20.0, 1.5, 20.0), delta=1e-6
        )

    def test_occupancy_inside_flag_uses_entry_distance(self):
        inside = MergeOccupancy("a", 10.0, 0.0, 20.0, 0.0, 2.0)
        outside = MergeOccupancy("b", 10.0, 5.0, 25.0, 0.5, 2.5)
        self.assertTrue(inside.inside)
        self.assertFalse(outside.inside)

    def test_zone_exit_distance_is_beyond_the_entry_distance(self):
        """탈출 거리는 진입 거리보다 커야 점유구간이 성립한다."""
        state = self.planner._first_merge_state
        config = state.config
        ego, _ = self._ego_at(FIRST_EGO_LINK, self.first_point, 60.0, 20.0)
        match = self.planner._resolve_ego_lane(ego)
        occupancy = self.planner._zone_occupancy_from_match(
            state, match, 20.0,
            0.5 * parameters.VEHICLE_LENGTH,
            0.5 * parameters.VEHICLE_WIDTH,
            "EGO",
            center_offset=0.5 * parameters.VEHICLE_LENGTH,
            is_ego=True,
        )
        self.assertIsNotNone(occupancy)
        self.assertGreater(occupancy.exit_distance, occupancy.entry_distance)
        self.assertGreater(occupancy.exit_time, occupancy.entry_time)
        self.assertGreaterEqual(
            occupancy.exit_distance - occupancy.entry_distance,
            2.0 * config.radius - 1.0,
        )

    def test_watch_radius_covers_the_emergency_stopping_distance(self):
        """감시 반경이 정지거리보다 짧으면 양보 판단이 실행 불가능해진다."""
        highway_speed = 28.0
        stopping = highway_speed * highway_speed / (
            2.0 * parameters.MERGE_YIELD_MAX_DECEL
        )
        for radius in (
            parameters.PREDICTIVE_MERGE_EGO_RADIUS,
            parameters.CONSERVATIVE_MERGE_EGO_RADIUS,
        ):
            self.assertGreater(radius, stopping)


if __name__ == "__main__":
    unittest.main()
