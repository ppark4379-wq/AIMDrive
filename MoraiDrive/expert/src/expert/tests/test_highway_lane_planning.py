import math
import os
import unittest

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning import parameters
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.planning.highway_lane_planning.structures import HighwayCandidatePath
from expert.src.planning.lattice_planning.planning_process import LatticePlanningProcess
from expert.src.vehicle_state import VehicleState


class HighwayLanePlanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        expert_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map_dir = os.path.normpath(
            os.path.join(expert_dir, "..", "..", "map_data", "R_KR_PG_KATRI_2025")
        )
        cls.lane_map = HighwayLaneMap(map_dir)

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lane_map)

    def test_prediction_horizon_is_short_enough_for_post_acc_avoidance(self):
        self.assertLessEqual(parameters.PREDICTION_HORIZON, 1.5)
        predicted_distance = (
            parameters.ACC_PREPARE_TARGET_SPEED * parameters.PREDICTION_HORIZON
        )
        self.assertLessEqual(predicted_distance, 20.0)

    def test_rviz_prediction_contains_one_future_pose_at_short_horizon(self):
        entry = self._object_entry("A2256W000408", 70.0, 10.0)
        matches = self.planner._match_objects([entry])
        paths = self.planner._prediction_paths(matches)

        self.assertEqual(1, len(paths))
        self.assertEqual(2, len(paths[0]))
        expected_distance = 10.0 * parameters.PREDICTION_HORIZON
        self.assertAlmostEqual(
            expected_distance,
            paths[0][0].distance(paths[0][1]),
            delta=0.15,
        )

    def _vehicle_state(self, link_id, s, velocity):
        lane_link = self.lane_map.links[link_id]
        point, yaw, _ = self.lane_map.point_at_s(lane_link, s)
        return VehicleState(point.x, point.y, yaw, velocity)

    def _object_entry(self, link_id, s, velocity):
        lane_link = self.lane_map.links[link_id]
        point, yaw, _ = self.lane_map.point_at_s(lane_link, s)
        object_info = ObjectInfo(
            point.x,
            point.y,
            velocity,
            1,
            size_x=4.5,
            size_y=2.0,
            heading_deg=math.degrees(yaw),
        )
        return {"object_info": object_info, "local_position": Point(1.0, 0.0)}

    def _reference_path(self, link_id, s=0.0, distance=100.0):
        lane_link = self.lane_map.links[link_id]
        match = self.lane_map.project(lane_link.points[0], lane_link)
        match.s = s
        return [
            item[0]
            for item in self.lane_map.route_points(match, distance, 0.5)
        ]

    def test_all_multilane_sections_have_centerlines_and_exclude_lane_one(self):
        sections = {}
        for lane_link in self.lane_map.links.values():
            sections.setdefault(lane_link.section_index, []).append(lane_link)
            self.assertGreater(len(lane_link.points), 100)

        self.assertEqual([0, 1, 2, 3, 4, 5], sorted(sections))
        for section_index, links in sections.items():
            if section_index == 0:
                continue
            excluded = [link for link in links if link.excluded]
            self.assertEqual(1, len(excluded))
            self.assertEqual(1, excluded[0].lane_number)

    def test_lane_two_cannot_select_excluded_leftmost_lane(self):
        lane_two = self.lane_map.links["A2256W000418"]
        neighbors = self.lane_map.legal_neighbors(lane_two)

        self.assertNotIn("A2256W000429", [link.id for _, link in neighbors])
        self.assertIn("A2256W000410", [link.id for _, link in neighbors])

    def test_highway_activation_uses_all_lane_links_not_route_corridor_only(self):
        ego = self._vehicle_state("A2256W000418", 50.0, 15.0)
        process = LatticePlanningProcess(highway_lane_planner=self.planner)

        self.assertTrue(process._is_inside_highway_zone(ego))
        self.assertFalse(
            process._is_inside_highway_zone(
                VehicleState(x=40.0, y=150.0, yaw=0.0, velocity=0.0)
            )
        )

    def test_right_turn_zone_returns_official_global_path(self):
        process = LatticePlanningProcess(highway_lane_planner=self.planner)
        reference_path = [Point(70.31, -327.68), Point(70.21, -365.75)]
        ego = VehicleState(
            x=74.0,
            y=-330.0,
            yaw=math.radians(-90.0),
            velocity=12.0,
        )
        self.planner.changed_lane_hold = True
        self.planner.state = self.planner.ACC

        result = process.run(ego, reference_path, [])

        self.assertIs(reference_path, result.path)
        self.assertFalse(result.active)
        self.assertEqual("highway_right_turn_global_path", result.reason)
        self.assertEqual(self.planner.KEEP, self.planner.state)
        self.assertFalse(self.planner.changed_lane_hold)

    def test_right_turn_policy_does_not_activate_before_start_point(self):
        process = LatticePlanningProcess(highway_lane_planner=self.planner)
        before_start = VehicleState(
            x=75.4,
            y=-210.0,
            yaw=math.radians(-90.0),
            velocity=12.0,
        )

        self.assertFalse(
            process._is_inside_highway_right_turn_zone(before_start)
        )

    def test_right_turn_policy_starts_after_merge(self):
        process = LatticePlanningProcess(highway_lane_planner=self.planner)
        early_entry = VehicleState(
            x=75.1,
            y=-225.0,
            yaw=math.radians(-90.0),
            velocity=15.0,
        )

        self.assertTrue(
            process._is_inside_highway_right_turn_zone(early_entry)
        )

    def test_right_turn_global_path_still_applies_vehicle_acc(self):
        process = LatticePlanningProcess(highway_lane_planner=self.planner)
        ego = self._vehicle_state("A2256W000153", 115.0, 20.0)
        lead = self._object_entry("A2256W000153", 135.0, 8.0)
        reference_path = self._reference_path("A2256W000153", 115.0)

        result = process.run(ego, reference_path, [lead])

        self.assertIs(reference_path, result.path)
        self.assertTrue(result.acc_override)
        self.assertLess(result.acc_target_velocity, ego.velocity)
        self.assertTrue(result.reason.startswith("highway_right_turn_"))

    def test_leftmost_lane_is_forced_to_safe_right_lane(self):
        ego = self._vehicle_state("A2256W000421", 50.0, 15.0)
        result = self.planner.plan(
            ego,
            self._reference_path("A2256W000421", 50.0),
            [],
        )

        self.assertTrue(result.active)
        self.assertEqual("CHANGE_RIGHT", result.highway_state)
        self.assertEqual("A2256W000435", result.target_link_id)
        self.assertFalse(self.lane_map.links[result.target_link_id].excluded)

    def test_predicted_lead_collision_selects_safe_adjacent_lane(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 20.0)
        lead = self._object_entry("A2256W000408", 75.0, 12.0)
        prepare = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 50.0),
            [lead],
        )
        slowed_ego = self._vehicle_state("A2256W000408", 50.0, 15.0)
        result = self.planner.plan(
            slowed_ego,
            self._reference_path("A2256W000408", 50.0),
            [lead],
        )

        self.assertTrue(prepare.acc_override)
        self.assertEqual("highway_lane_acc_prepare", prepare.reason)
        self.assertLessEqual(prepare.acc_target_velocity, 13.0)
        self.assertTrue(result.active)
        self.assertLessEqual(result.acc_target_velocity, 15.0)
        self.assertFalse(result.acc_override)
        self.assertEqual("highway_lane_safe_change", result.reason)
        self.assertNotEqual("A2256W000408", result.target_link_id)
        self.assertFalse(self.lane_map.links[result.target_link_id].excluded)

    def test_slower_lead_triggers_lane_change_before_prediction_collision(self):
        ego = self._vehicle_state("A2256W000408", 40.0, 20.0)
        lead = self._object_entry("A2256W000408", 84.0, 15.0)
        prepare = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 40.0),
            [lead],
        )
        slowed_ego = self._vehicle_state("A2256W000408", 40.0, 14.0)
        result = self.planner.plan(
            slowed_ego,
            self._reference_path("A2256W000408", 40.0),
            [lead],
        )

        self.assertEqual("highway_lane_acc_prepare", prepare.reason)
        self.assertLessEqual(prepare.acc_target_velocity, 13.0)
        self.assertTrue(result.active)
        self.assertEqual("highway_lane_safe_change", result.reason)
        self.assertIn(result.highway_state, ("CHANGE_LEFT", "CHANGE_RIGHT"))

    def test_lane_change_path_persists_and_holds_target_lane(self):
        lead = self._object_entry("A2256W000408", 84.0, 15.0)
        fast_ego = self._vehicle_state("A2256W000408", 40.0, 20.0)
        self.planner.plan(
            fast_ego,
            self._reference_path("A2256W000408", 40.0),
            [lead],
        )
        slowed_ego = self._vehicle_state("A2256W000408", 40.0, 14.0)
        selected = self.planner.plan(
            slowed_ego,
            self._reference_path("A2256W000408", 40.0),
            [lead],
        )
        selected_end = selected.path[-1]
        middle = selected.path[min(10, len(selected.path) - 2)]
        following = selected.path[min(11, len(selected.path) - 1)]
        middle_ego = VehicleState(
            middle.x,
            middle.y,
            math.atan2(following.y - middle.y, following.x - middle.x),
            14.0,
        )
        in_progress = self.planner.plan(
            middle_ego,
            self._reference_path("A2256W000408", 40.0),
            [lead],
        )

        self.assertEqual("highway_lane_change_in_progress", in_progress.reason)
        self.assertAlmostEqual(selected_end.x, in_progress.path[-1].x, places=5)
        self.assertAlmostEqual(selected_end.y, in_progress.path[-1].y, places=5)

        target = self.lane_map.links[selected.target_link_id]
        target_ego = self._vehicle_state(
            target.id, min(60.0, target.length - 1.0), 14.0
        )
        original_lane_reference = self._reference_path("A2256W000408", 40.0)
        held = self.planner.plan(
            target_ego,
            original_lane_reference,
            [],
        )
        self.assertEqual("KEEP_LANE", held.highway_state)
        self.assertEqual("highway_lane_hold_changed_lane", held.reason)
        self.assertTrue(held.active)
        self.assertIsNot(held.path, original_lane_reference)
        held_match = self.lane_map.project(held.path[0], target)
        self.assertLess(held_match.distance, 0.01)

    def test_committed_lane_change_keeps_lateral_state_while_acc_brakes(self):
        candidate = HighwayCandidatePath(layer="lane_right")
        candidate.points = [Point(0.0, 0.0), Point(1.0, -0.1)]
        candidate.direction = "right"
        candidate.offset = -3.5
        candidate.valid = False
        self.planner.state = self.planner.CHANGE_RIGHT

        result = self.planner._result_during_change(
            candidate,
            nearest_distance=8.0,
            ttc=1.0,
            acc_target_velocity=5.0,
            common={},
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("CHANGE_RIGHT", result.highway_state)
        self.assertIs(candidate.points, result.path)

    def test_immediate_risk_produces_real_acc_speed_limit(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 20.0)
        lead = self._object_entry("A2256W000408", 62.0, 5.0)
        result = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 50.0),
            [lead],
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("ACC_FOLLOW", result.highway_state)
        self.assertLess(result.acc_target_velocity, ego.velocity)

    def test_same_speed_lead_with_infinite_ttc_does_not_start_lane_change(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 20.0)
        lead = self._object_entry("A2256W000408", 95.0, 20.0)
        reference_path = self._reference_path("A2256W000408", 50.0)

        result = self.planner.plan(ego, reference_path, [lead])

        self.assertTrue(math.isinf(result.ttc))
        self.assertFalse(result.acc_override)
        self.assertEqual("KEEP_LANE", result.highway_state)
        self.assertEqual("highway_lane_no_obstacle_global_path", result.reason)
        self.assertIs(reference_path, result.path)

    def test_no_obstacle_keeps_global_reference_path(self):
        ego = self._vehicle_state("A2256W000408", 20.0, 20.0)
        reference_path = self._reference_path("A2256W000408", 20.0)
        result = self.planner.plan(
            ego,
            reference_path,
            [],
        )

        self.assertEqual("KEEP_LANE", result.highway_state)
        self.assertIs(reference_path, result.path)
        self.assertFalse(result.active)
        self.assertEqual("highway_lane_no_obstacle_global_path", result.reason)

    def test_local_forward_fallback_starts_acc_when_link_match_fails(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 25.0)
        object_info = ObjectInfo(
            999.0,
            999.0,
            10.0,
            1,
            size_x=4.5,
            size_y=2.0,
        )
        entry = {
            "object_info": object_info,
            "local_position": Point(30.0, 0.3),
        }
        result = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 50.0),
            [entry],
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("highway_lane_acc_prepare", result.reason)
        self.assertLessEqual(result.acc_target_velocity, 13.0)

    def test_fast_rear_vehicle_blocks_that_target_lane(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 20.0)
        entries = [
            self._object_entry("A2256W000408", 75.0, 12.0),
            self._object_entry("A2256W000434", 35.0, 30.0),
        ]
        result = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 50.0),
            entries,
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("ACC_FOLLOW", result.highway_state)
        self.assertEqual("A2256W000408", result.target_link_id)

    def test_fast_rear_vehicle_uses_relative_speed_clearance(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 14.0)
        target = self.lane_map.links["A2256W000434"]
        target_projection = self.lane_map.project(ego.position, target)
        candidate = self.planner._build_candidate(
            ego,
            target,
            target_projection,
            direction="left",
            source_link=self.lane_map.links["A2256W000408"],
        )
        rear_vehicle = self._object_entry(
            "A2256W000434",
            max(0.0, target_projection.s - 30.0),
            30.0,
        )
        matches = self.planner._match_objects([rear_vehicle])

        self.assertFalse(
            self.planner._target_lane_has_clearance(
                target, candidate, matches, ego.velocity
            )
        )

    def test_converging_lane_vehicle_triggers_merge_acc(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000445"]
        ego_s = max(0.0, ego_link.length - 40.0)
        other_s = max(0.0, other_link.length - 40.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 20.0)
        converging_vehicle = self._object_entry(
            other_link.id, other_s, 20.0
        )

        result = self.planner.plan(
            ego,
            self._reference_path(ego_link.id, ego_s),
            [converging_vehicle],
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("ACC_FOLLOW", result.highway_state)
        self.assertEqual("highway_lane_merge_acc", result.reason)
        self.assertLess(result.acc_target_velocity, ego.velocity)
        self.assertEqual(ego_link.id, result.target_link_id)

    def test_designated_merge_yields_even_when_ego_reaches_first(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000445"]
        ego_s = max(0.0, ego_link.length - 20.0)
        other_s = max(0.0, other_link.length - 50.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 20.0)
        joining_rear = self._object_entry(other_link.id, other_s, 20.0)
        reference_path = self._reference_path(ego_link.id, ego_s)

        result = self.planner.plan(ego, reference_path, [joining_rear])

        self.assertTrue(result.acc_override)
        self.assertEqual("ACC_FOLLOW", result.highway_state)
        self.assertEqual("highway_lane_merge_acc", result.reason)
        self.assertLessEqual(
            result.acc_target_velocity,
            parameters.CONSERVATIVE_MERGE_MAX_SPEED,
        )

    def test_same_node_different_successors_are_not_merge_conflict(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000423"]
        ego_s = max(0.0, ego_link.length - 20.0)
        other_s = max(0.0, other_link.length - 10.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 15.0)
        different_exit_vehicle = self._object_entry(
            other_link.id, other_s, 15.0
        )
        reference_path = self._reference_path(ego_link.id, ego_s)

        result = self.planner.plan(
            ego, reference_path, [different_exit_vehicle]
        )

        self.assertFalse(result.acc_override)
        self.assertEqual("highway_lane_no_obstacle_global_path", result.reason)
        self.assertIs(reference_path, result.path)

    def test_designated_merge_yield_uses_safe_zone_speed_cap(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000445"]
        ego_s = max(0.0, ego_link.length - 40.0)
        other_s = max(0.0, other_link.length - 20.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 20.0)
        joining_vehicle = self._object_entry(other_link.id, other_s, 20.0)

        result = self.planner.plan(
            ego,
            self._reference_path(ego_link.id, ego_s),
            [joining_vehicle],
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("highway_lane_merge_acc", result.reason)
        self.assertLessEqual(
            result.acc_target_velocity,
            parameters.CONSERVATIVE_MERGE_MAX_SPEED,
        )

    def test_safe_zone_keeps_yielding_while_vehicle_enters_successor(self):
        ego_link = self.lane_map.links["A2256W000422"]
        ego_s = max(0.0, ego_link.length - 20.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 20.0)
        merged_vehicle = self._object_entry("A2256W000153", 5.0, 15.0)

        result = self.planner.plan(
            ego,
            self._reference_path(ego_link.id, ego_s),
            [merged_vehicle],
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("highway_lane_merge_acc", result.reason)
        self.assertLessEqual(
            result.acc_target_velocity,
            parameters.CONSERVATIVE_MERGE_MAX_SPEED,
        )

    def test_merge_acc_one_frame_dropout_recovers_without_toggle(self):
        first_limit = self.planner._stabilize_merge_speed_limit(5.0, 20.0)
        dropout_limit = self.planner._stabilize_merge_speed_limit(
            float("inf"), 20.0
        )

        self.assertEqual(5.0, first_limit)
        self.assertFalse(math.isinf(dropout_limit))
        self.assertGreater(dropout_limit, first_limit)
        self.assertLessEqual(
            dropout_limit - first_limit,
            parameters.MERGE_ACC_RECOVERY_RATE
            / parameters.PLANNER_UPDATE_RATE,
        )

    def test_unmatched_crossing_vehicle_triggers_trajectory_merge_acc(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 20.0)
        ego_match = self.lane_map.match(ego.position, heading=ego.yaw)
        collision_time = 2.0
        ego_future, ego_future_yaw, _ = self.lane_map.advance_default(
            ego_match, ego.velocity * collision_time
        )
        object_speed = 20.0
        object_yaw = ego_future_yaw + math.pi / 2.0
        ego_center_x = (
            ego_future.x
            + 0.5 * parameters.VEHICLE_LENGTH * math.cos(ego_future_yaw)
        )
        ego_center_y = (
            ego_future.y
            + 0.5 * parameters.VEHICLE_LENGTH * math.sin(ego_future_yaw)
        )
        object_x = (
            ego_center_x
            - object_speed * collision_time * math.cos(object_yaw)
        )
        object_y = (
            ego_center_y
            - object_speed * collision_time * math.sin(object_yaw)
        )
        object_info = ObjectInfo(
            object_x,
            object_y,
            object_speed,
            1,
            size_x=4.5,
            size_y=2.0,
            heading_deg=math.degrees(object_yaw),
        )
        dx = object_x - ego.position.x
        dy = object_y - ego.position.y
        entry = {
            "object_info": object_info,
            "local_position": Point(
                math.cos(ego.yaw) * dx + math.sin(ego.yaw) * dy,
                -math.sin(ego.yaw) * dx + math.cos(ego.yaw) * dy,
            ),
        }

        self.assertEqual([], self.planner._match_objects([entry]))
        result = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 50.0),
            [entry],
        )

        self.assertTrue(result.acc_override)
        self.assertEqual("highway_lane_merge_acc", result.reason)
        self.assertLess(result.acc_target_velocity, ego.velocity)

    def test_near_merge_without_vehicle_does_not_enable_acc(self):
        ego_link = self.lane_map.links["A2256W000423"]
        ego_s = max(0.0, ego_link.length - 40.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 20.0)
        reference_path = self._reference_path(ego_link.id, ego_s)

        result = self.planner.plan(ego, reference_path, [])

        self.assertFalse(result.acc_override)
        self.assertEqual("KEEP_LANE", result.highway_state)
        self.assertEqual("highway_lane_no_obstacle_global_path", result.reason)
        self.assertIs(reference_path, result.path)

    def test_blocked_adjacent_lanes_fall_back_to_acc(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 20.0)
        entries = [
            self._object_entry("A2256W000408", 75.0, 12.0),
            self._object_entry("A2256W000434", 49.0, 20.0),
            self._object_entry("A2256W000179", 49.0, 20.0),
        ]
        prepare = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 50.0),
            entries,
        )
        slowed_ego = self._vehicle_state("A2256W000408", 50.0, 15.0)
        result = self.planner.plan(
            slowed_ego,
            self._reference_path("A2256W000408", 50.0),
            entries,
        )

        self.assertEqual("highway_lane_acc_prepare", prepare.reason)
        self.assertTrue(result.acc_override)
        self.assertEqual("ACC_FOLLOW", result.highway_state)
        self.assertEqual("highway_lane_no_safe_change", result.reason)
        self.assertEqual("A2256W000408", result.target_link_id)


if __name__ == "__main__":
    unittest.main()
