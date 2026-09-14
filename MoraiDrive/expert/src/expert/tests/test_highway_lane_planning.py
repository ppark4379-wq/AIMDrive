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

    def test_prediction_horizon_is_short_enough_for_dynamic_avoidance(self):
        self.assertLessEqual(parameters.PREDICTION_HORIZON, 1.5)

    def test_time_based_cycle_threshold_scales_with_control_dt(self):
        self.planner.set_control_dt(0.02)

        self.assertEqual(
            5,
            self.planner._cycles_for_reference_count(
                parameters.PREDICTIVE_MERGE_CLEAR_CYCLES
            ),
        )
        self.assertEqual(
            50,
            self.planner._cycles_for_reference_count(
                parameters.CONSERVATIVE_MERGE_CLEAR_CYCLES
            ),
        )

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

    def _object_entry(self, link_id, s, velocity, name=""):
        lane_link = self.lane_map.links[link_id]
        point, yaw, _ = self.lane_map.point_at_s(lane_link, s)
        object_info = ObjectInfo(
            point.x,
            point.y,
            velocity,
            1,
            name=name,
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

    def _checked_merge_scene(self, second=False, ego_back=35.0, other_back=20.0,
                             ego_speed=20.0, other_speed=15.0):
        ego_id, other_id, center = (
            ("A2256W000422", "A2256W000445", parameters.CONSERVATIVE_MERGE_POINT)
            if second else ("A2256W000411", "A2256W000409", parameters.PREDICTIVE_MERGE_POINT)
        )
        ego_s = self.lane_map.project(Point(*center), self.lane_map.links[ego_id]).s - ego_back
        other_s = self.lane_map.project(Point(*center), self.lane_map.links[other_id]).s - other_back
        ego = self._vehicle_state(ego_id, max(0.0, ego_s), ego_speed)
        entry = self._object_entry(other_id, max(0.0, other_s), other_speed, name="merge_opponent")
        entry["object_info"].unique_id = 12
        dx = entry["object_info"].position.x - ego.position.x
        dy = entry["object_info"].position.y - ego.position.y
        entry["local_position"] = Point(math.cos(ego.yaw) * dx + math.sin(ego.yaw) * dy,
                                        -math.sin(ego.yaw) * dx + math.cos(ego.yaw) * dy)
        return self.planner.plan(ego, self._reference_path(ego_id, max(0.0, ego_s)),
                                 [entry], desired_velocity=22.0)

    def test_all_multilane_sections_have_expected_prohibited_lanes(self):
        sections = {}
        for lane_link in self.lane_map.links.values():
            sections.setdefault(lane_link.section_index, []).append(lane_link)
            self.assertGreater(len(lane_link.points), 100)

        self.assertEqual([0, 1, 2, 3, 4, 5], sorted(sections))
        for section_index, links in sections.items():
            if section_index == 0:
                continue
            excluded_ids = {link.id for link in links if link.excluded}
            expected_ids = {link.id for link in links if link.lane_number == 1}
            expected_ids.update(
                link.id
                for link in links
                if link.id in parameters.PROHIBITED_LINK_IDS
            )
            self.assertEqual(expected_ids, excluded_ids)

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
        self.assertTrue(result.active)
        self.assertTrue(math.isinf(result.acc_target_velocity))
        self.assertFalse(result.acc_override)
        self.assertEqual("highway_lane_safe_change", result.reason)
        self.assertNotEqual("A2256W000408", result.target_link_id)
        self.assertFalse(self.lane_map.links[result.target_link_id].excluded)

    def test_slower_lead_triggers_lane_change_before_prediction_collision(self):
        ego = self._vehicle_state("A2256W000408", 40.0, 20.0)
        lead = self._object_entry("A2256W000408", 84.0, 15.0)
        result = self.planner.plan(
            ego,
            self._reference_path("A2256W000408", 40.0),
            [lead],
        )

        self.assertTrue(result.active)
        self.assertFalse(result.acc_override)
        self.assertTrue(math.isinf(result.acc_target_velocity))
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

    def test_lane_change_cooldown_uses_acc_instead_of_changing_back(self):
        lane_link = self.lane_map.links["A2256W000435"]
        ego = self._vehicle_state(lane_link.id, 50.0, 20.0)
        lead = self._object_entry(lane_link.id, 75.0, 10.0)
        self.planner._lane_change_cooldown_origin = Point(
            ego.position.x, ego.position.y
        )
        self.planner._last_lane_change_direction = self.planner.CHANGE_LEFT
        self.planner.changed_lane_hold = True

        result = self.planner.plan(
            ego,
            self._reference_path(lane_link.id, 50.0),
            [lead],
        )

        self.assertEqual("ACC_FOLLOW", result.highway_state)
        self.assertEqual("highway_lane_change_cooldown", result.reason)
        self.assertTrue(result.acc_override)
        self.assertGreater(
            result.debug_details["lane_change_cooldown_remaining"], 0.0
        )

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

    def test_committed_safe_lane_change_drops_source_lane_acc_limit(self):
        candidate = HighwayCandidatePath(layer="lane_right")
        candidate.points = [Point(0.0, 0.0), Point(1.0, -0.1)]
        candidate.direction = "right"
        candidate.offset = -3.5
        candidate.valid = True
        self.planner.state = self.planner.CHANGE_RIGHT

        result = self.planner._result_during_change(
            candidate,
            nearest_distance=30.0,
            ttc=float("inf"),
            acc_target_velocity=5.0,
            common={},
        )

        self.assertFalse(result.acc_override)
        self.assertTrue(math.isinf(result.acc_target_velocity))
        self.assertEqual("highway_lane_change_in_progress", result.reason)

    def test_committed_lane_change_applies_acc_for_close_front_vehicle(self):
        candidate = HighwayCandidatePath(layer="lane_right")
        candidate.points = [Point(0.0, 0.0), Point(1.0, -0.1)]
        candidate.direction = "right"
        candidate.offset = -3.5
        candidate.valid = True
        self.planner.state = self.planner.CHANGE_RIGHT

        result = self.planner._result_during_change(
            candidate,
            nearest_distance=7.23,
            ttc=float("inf"),
            acc_target_velocity=5.32,
            common={},
        )

        self.assertTrue(result.acc_override)
        self.assertEqual(5.32, result.acc_target_velocity)
        self.assertEqual("highway_lane_change_acc", result.reason)

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

    def test_rightmost_lane_preference_applies_with_same_speed_lead(self):
        ego = self._vehicle_state("A2256W000408", 50.0, 20.0)
        lead = self._object_entry("A2256W000408", 95.0, 20.0)
        reference_path = self._reference_path("A2256W000408", 50.0)

        result = self.planner.plan(ego, reference_path, [lead])

        self.assertTrue(math.isinf(result.ttc))
        self.assertFalse(result.acc_override)
        self.assertEqual("CHANGE_LEFT", result.highway_state)
        self.assertEqual("A2256W000434", result.target_link_id)
        self.assertEqual("highway_lane_second_merge_prefer_left", result.reason)

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
        self.assertLessEqual(
            result.acc_target_velocity,
            parameters.ACC_UNMATCHED_HAZARD_SPEED,
        )

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

    def test_rightmost_lane_is_prohibited_for_driving_and_avoidance(self):
        source = self.lane_map.links["A2256W000408"]
        neighbor_ids = {
            target.id for _, target in self.lane_map.legal_neighbors(source)
        }

        self.assertFalse(self.lane_map.links["A2256W000411"].excluded)
        self.assertTrue(self.lane_map.links["A2256W000179"].excluded)
        self.assertNotIn("A2256W000179", neighbor_ids)

        ego = self._vehicle_state("A2256W000179", 50.0, 20.0)
        result = self.planner.plan(
            ego, self._reference_path("A2256W000179", 50.0), []
        )
        self.assertTrue(result.active)
        self.assertNotEqual("A2256W000179", result.target_link_id)
        self.assertFalse(self.lane_map.links[result.target_link_id].excluded)

    def test_acc_releases_stopped_vehicle_after_safe_gap_reopens(self):
        result = self.planner._acc_speed_limit(
            ego_speed=0.0,
            gap=30.0,
            ttc=float("inf"),
        )

        self.assertTrue(math.isinf(result))

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

    def test_vehicle_on_target_lane_successor_blocks_lane_change(self):
        source = self.lane_map.links["A2256W000409"]
        target = self.lane_map.links["A2256W000411"]
        ego = self._vehicle_state(source.id, max(0.0, source.length - 10.0), 20.0)
        target_projection = self.lane_map.project(ego.position, target)
        candidate = self.planner._build_candidate(
            ego,
            target,
            target_projection,
            direction="right",
            source_link=source,
        )
        successor_vehicle = self._object_entry("A2256W000420", 5.0, 5.0)
        matches = self.planner._match_objects([successor_vehicle])

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
        self.assertEqual("highway_lane_emergency_stop", result.reason)
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertEqual(ego_link.id, result.target_link_id)

    def test_first_merge_yields_when_other_prediction_reaches_circle(self):
        ego_link = self.lane_map.links["A2256W000411"]
        other_link = self.lane_map.links["A2256W000409"]
        merge_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        ego = self._vehicle_state(
            ego_link.id, max(0.0, ego_zone_s - 35.0), 20.0
        )
        approaching_other = self._object_entry(
            other_link.id, max(0.0, other_zone_s - 20.0), 15.0,
            name="first_merge_vehicle",
        )
        self.assertGreater(
            approaching_other["object_info"].position.distance(merge_point),
            parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
        )

        result = self.planner.plan(
            ego,
            self._reference_path(ego_link.id, max(0.0, ego_zone_s - 35.0)),
            [approaching_other],
        )

        self.assertEqual("OTHER", self.planner._predictive_merge_priority)
        self.assertTrue(result.acc_override)
        self.assertLessEqual(
            result.acc_target_velocity,
            parameters.PREDICTIVE_MERGE_MAX_SPEED,
        )

    def test_first_merge_ignores_unrelated_vehicle_already_on_successor(self):
        ego_link = self.lane_map.links["A2256W000411"]
        other_link = self.lane_map.links["A2256W000409"]
        merge_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        ego = self._vehicle_state(
            ego_link.id, max(0.0, ego_zone_s - 35.0), 20.0
        )
        actual_winner = self._object_entry(
            other_link.id,
            max(0.0, other_zone_s - 20.0),
            15.0,
            name="actual_crossing_winner",
        )
        unrelated_downstream = self._object_entry(
            "A2256W000420",
            parameters.PREDICTIVE_MERGE_PASS_DISTANCE - 1.0,
            10.0,
            name="unrelated_downstream_vehicle",
        )

        result = self.planner.plan(
            ego,
            self._reference_path(ego_link.id, max(0.0, ego_zone_s - 35.0)),
            [actual_winner, unrelated_downstream],
        )

        self.assertEqual("OTHER", self.planner._predictive_merge_priority)
        self.assertEqual(2, result.debug_details["merge_safety"]["tracks"])
        self.assertIsNotNone(result.debug_details["merge_safety"]["conflict"])
        self.assertTrue(result.acc_override)

    def test_first_merge_stopped_ego_can_launch_after_vehicle_passes(self):
        result = self._checked_merge_scene(ego_back=10, other_back=0,
                                           ego_speed=0, other_speed=10)
        self.assertGreater(result.acc_target_velocity, 0.0)
        self.assertIsNone(result.debug_details["merge_safety"]["conflict"])

    def test_first_merge_near_simultaneous_arrivals_require_gap(self):
        result = self._checked_merge_scene(ego_back=35, other_back=20,
                                           ego_speed=20, other_speed=15)
        self.assertLess(result.acc_target_velocity, 20.0)
        self.assertIsNotNone(result.debug_details["merge_safety"]["conflict"])

    def test_first_merge_clear_arrivals_do_not_get_fixed_brake_point_cap(self):
        result = self._checked_merge_scene(ego_back=60, other_back=0,
                                           ego_speed=20, other_speed=25)
        self.assertEqual(22.0, result.acc_target_velocity)
        self.assertEqual("GO", result.debug_details["merge_safety"]["decision"])

    def test_first_merge_does_not_force_30kph_into_collision(self):
        result = self._checked_merge_scene(ego_back=10, other_back=5,
                                           ego_speed=10, other_speed=10)
        self.assertLess(result.acc_target_velocity, parameters.MERGE_MIN_YIELD_SPEED)

    def test_first_merge_releases_old_speed_cap_when_vehicle_has_cleared(self):
        held = self._checked_merge_scene(ego_back=25, other_back=30,
                                         ego_speed=15, other_speed=25)
        self.assertLess(held.acc_target_velocity, 15.0)
        self.planner._merge_speed_limit_state = 2.0
        cleared = self._checked_merge_scene(ego_back=60, other_back=0,
                                            ego_speed=15, other_speed=25)
        self.assertEqual("GO", cleared.debug_details["merge_safety"]["decision"])
        self.assertEqual(22.0, cleared.acc_target_velocity)

    def test_first_merge_keeps_speed_after_ego_enters_first(self):
        ego_link = self.lane_map.links["A2256W000411"]
        other_link = self.lane_map.links["A2256W000409"]
        merge_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        ego = self._vehicle_state(ego_link.id, ego_zone_s, 20.0)
        reference_path = self._reference_path(ego_link.id, ego_zone_s)
        self.planner._merge_speed_limit_state = 0.0

        entered = self.planner.plan(ego, reference_path, [])
        self.assertEqual("EGO", self.planner._predictive_merge_priority)
        self.assertFalse(entered.acc_override)

        late_vehicle = self._object_entry(
            other_link.id,
            max(0.0, other_zone_s - 10.0),
            15.0,
            name="late_first_merge_vehicle",
        )
        late_vehicle["local_position"] = Point(-10.0, 0.0)
        result = self.planner.plan(ego, reference_path, [late_vehicle])

        self.assertEqual("EGO", self.planner._predictive_merge_priority)
        self.assertFalse(
            result.acc_override,
            (result.reason, result.acc_target_velocity, result.debug_details),
        )
        self.assertTrue(math.isinf(result.acc_target_velocity))

    def test_first_merge_ego_priority_does_not_exempt_fast_rear(self):
        self.planner._predictive_merge_priority = "EGO"
        result = self._checked_merge_scene(ego_back=25, other_back=30,
                                           ego_speed=15, other_speed=25)
        self.assertTrue(result.acc_override)
        self.assertLess(result.acc_target_velocity, 15.0)
        self.assertIsNotNone(result.debug_details["merge_safety"]["conflict"])

    def test_first_merge_rechecks_new_vehicle_after_go(self):
        clear = self._checked_merge_scene(ego_back=60, other_back=0,
                                          ego_speed=20, other_speed=25)
        self.assertEqual("GO", clear.debug_details["merge_safety"]["decision"])
        danger = self._checked_merge_scene(ego_back=25, other_back=30,
                                           ego_speed=15, other_speed=25)
        self.assertLess(danger.acc_target_velocity, 15.0)

    def test_first_merge_ignores_vehicle_outside_monitored_right_lane(self):
        ego_link = self.lane_map.links["A2256W000411"]
        ignored_link = self.lane_map.links["A2256W000410"]
        merge_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        ignored_zone_s = self.lane_map.project(merge_point, ignored_link).s
        ego = self._vehicle_state(
            ego_link.id, max(0.0, ego_zone_s - 35.0), 20.0
        )
        ignored_vehicle = self._object_entry(
            ignored_link.id,
            max(0.0, ignored_zone_s - 30.0),
            15.0,
            name="ignored_main_lane_vehicle",
        )

        result = self.planner.plan(
            ego,
            self._reference_path(ego_link.id, max(0.0, ego_zone_s - 35.0)),
            [ignored_vehicle],
        )

        self.assertNotEqual("OTHER", self.planner._predictive_merge_priority)
        self.assertTrue(
            math.isinf(result.debug_details["raw_merge_limit"])
        )

    def test_first_merge_releases_stop_quickly_after_prediction_clears(self):
        ego_link = self.lane_map.links["A2256W000411"]
        merge_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        stopped_ego = self._vehicle_state(
            ego_link.id, max(0.0, ego_zone_s - 25.0), 0.0
        )
        reference_path = self._reference_path(
            ego_link.id, max(0.0, ego_zone_s - 25.0)
        )
        self.planner._predictive_merge_priority = "OTHER"
        self.planner._merge_speed_limit_state = 0.0

        result = None
        for _ in range(parameters.PREDICTIVE_MERGE_CLEAR_CYCLES):
            result = self.planner.plan(stopped_ego, reference_path, [])

        self.assertIsNotNone(result)
        self.assertNotEqual("OTHER", self.planner._predictive_merge_priority)
        self.assertFalse(result.acc_override)
        self.assertTrue(math.isinf(result.acc_target_velocity))

    def test_first_merge_track_dropout_retains_predicted_obstacle(self):
        self._checked_merge_scene(ego_back=25, other_back=30, ego_speed=15, other_speed=25)
        guard = self.planner.merge_safety
        before = len(guard.tracks)
        ego_link = self.lane_map.links["A2256W000411"]
        position = self.lane_map.project(Point(*parameters.PREDICTIVE_MERGE_POINT), ego_link).s - 25
        ego = self._vehicle_state(ego_link.id, position, 15.0)
        result = self.planner.plan(ego, self._reference_path(ego_link.id, position), [],
                                   desired_velocity=22.0)
        self.assertEqual(before, len(guard.tracks))
        self.assertLess(result.acc_target_velocity, 15.0)

    def test_designated_merge_does_not_brake_for_time_separated_arrivals(self):
        result = self._checked_merge_scene(second=True, ego_back=60, other_back=0,
                                           ego_speed=20, other_speed=25)
        self.assertIsNone(result.debug_details["merge_safety"]["conflict"])
        self.assertEqual(22.0, result.acc_target_velocity)

    def test_second_merge_middle_lane_is_already_preferred(self):
        middle_link = self.lane_map.links["A2256W000422"]
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        merge_s = self.lane_map.project(merge_point, middle_link).s
        ego_s = max(0.0, merge_s - 30.0)
        ego = self._vehicle_state(middle_link.id, ego_s, 20.0)

        result = self.planner.plan(
            ego,
            self._reference_path(middle_link.id, ego_s),
            [],
        )

        self.assertEqual("KEEP_LANE", result.highway_state)
        self.assertEqual(middle_link.id, result.target_link_id)

    def test_second_merge_middle_lane_holds_at_photo_location(self):
        source_link = self.lane_map.links["A2256W000434"]
        photo_position = Point(70.45, -103.79)
        source_s = self.lane_map.project(photo_position, source_link).s
        ego = self._vehicle_state(source_link.id, source_s, 20.0)

        result = self.planner.plan(
            ego,
            self._reference_path(source_link.id, source_s),
            [],
        )

        self.assertEqual("KEEP_LANE", result.highway_state)
        self.assertEqual(source_link.id, result.target_link_id)

    def test_second_merge_middle_uses_acc_when_left_is_unavailable(self):
        middle_link = self.lane_map.links["A2256W000434"]
        ego_s = 130.0
        ego = self._vehicle_state(middle_link.id, ego_s, 20.0)
        lead = self._object_entry("A2256W000422", 10.0, 18.0)
        original_clearance = self.planner._target_lane_has_clearance

        def clearance(target_link, *args):
            if target_link.lane_number == 2:
                return False
            return original_clearance(target_link, *args)

        self.planner._target_lane_has_clearance = clearance
        try:
            result = self.planner.plan(
                ego,
                self._reference_path(middle_link.id, ego_s),
                [lead],
            )
        finally:
            self.planner._target_lane_has_clearance = original_clearance

        self.assertEqual("ACC_FOLLOW", result.highway_state)
        self.assertNotEqual("CHANGE_RIGHT", result.highway_state)
        self.assertEqual(middle_link.id, result.target_link_id)
        self.assertTrue(result.acc_override)

    def test_second_merge_rightmost_lane_moves_toward_middle(self):
        source_link = self.lane_map.links["A2256W000408"]
        photo_position = Point(67.5, -103.79)
        source_s = self.lane_map.project(photo_position, source_link).s
        ego = self._vehicle_state(source_link.id, source_s, 20.0)

        result = self.planner.plan(
            ego,
            self._reference_path(source_link.id, source_s),
            [],
        )

        self.assertEqual("CHANGE_LEFT", result.highway_state)
        self.assertEqual("A2256W000434", result.target_link_id)
        self.assertEqual("highway_lane_second_merge_prefer_left", result.reason)

    def test_second_merge_right_lane_preference_starts_before_photo_location(self):
        source_link = self.lane_map.links["A2256W000408"]
        source_s = 40.0
        ego = self._vehicle_state(source_link.id, source_s, 20.0)
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        self.assertGreater(ego.position.distance(merge_point), 100.0)
        self.assertLess(
            ego.position.distance(merge_point),
            parameters.SECOND_MERGE_LANE_PREFERENCE_RADIUS,
        )

        result = self.planner.plan(
            ego,
            self._reference_path(source_link.id, source_s),
            [],
        )

        self.assertEqual("CHANGE_LEFT", result.highway_state)
        self.assertEqual("A2256W000434", result.target_link_id)

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
        self.assertEqual("highway_lane_merge_clear", result.reason)
        self.assertIs(reference_path, result.path)

    def test_designated_merge_keeps_ego_priority_when_ego_enters_first(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000445"]
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        ego_s = max(0.0, ego_link.length - 40.0)
        other_s = max(0.0, other_zone_s - 25.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 20.0)
        joining_vehicle = self._object_entry(other_link.id, other_s, 20.0)

        result = self.planner.plan(
            ego,
            self._reference_path(ego_link.id, ego_s),
            [joining_vehicle],
        )

        self.assertFalse(result.acc_override)
        self.assertEqual("EGO", self.planner._merge_zone_priority)

    def test_second_merge_actual_ego_entry_overrides_early_other_prediction(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000445"]
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        predicted_winner = self._object_entry(
            other_link.id,
            max(0.0, other_zone_s - 25.0),
            30.0,
            name="second_noisy_predicted_winner",
        )

        self.planner.plan(
            self._vehicle_state(
                ego_link.id, max(0.0, ego_zone_s - 30.0), 20.0
            ),
            self._reference_path(
                ego_link.id, max(0.0, ego_zone_s - 30.0)
            ),
            [predicted_winner],
        )
        self.assertEqual("OTHER", self.planner._merge_zone_priority)

        still_approaching = self._object_entry(
            other_link.id,
            max(0.0, other_zone_s - 22.0),
            20.0,
            name="second_noisy_predicted_winner",
        )
        result = self.planner.plan(
            self._vehicle_state(ego_link.id, ego_zone_s, 20.0),
            self._reference_path(ego_link.id, ego_zone_s),
            [still_approaching],
        )

        self.assertEqual("EGO", self.planner._merge_zone_priority)
        self.assertFalse(result.acc_override)
        self.assertTrue(math.isinf(result.acc_target_velocity))

    def test_second_merge_rechecks_fast_rear_after_go(self):
        clear = self._checked_merge_scene(second=True, ego_back=60, other_back=0,
                                          ego_speed=20, other_speed=25)
        self.assertEqual("GO", clear.debug_details["merge_safety"]["decision"])
        danger = self._checked_merge_scene(second=True, ego_back=25, other_back=30,
                                           ego_speed=15, other_speed=25)
        self.assertLess(danger.acc_target_velocity, 15.0)
        self.assertTrue(danger.debug_details["merge_safety"]["conflict"]["rear"])

    def test_second_merge_new_collision_is_not_ignored_after_go(self):
        self.planner._merge_zone_priority = "EGO"
        result = self._checked_merge_scene(second=True, ego_back=0, other_back=0,
                                           ego_speed=20, other_speed=20)
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertEqual("highway_lane_emergency_stop", result.reason)

    def test_second_merge_ego_priority_ignores_trajectory_obb_stop(self):
        ego_link = self.lane_map.links["A2256W000422"]
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        ego = self._vehicle_state(ego_link.id, ego_zone_s, 20.0)
        reference_path = self._reference_path(ego_link.id, ego_zone_s)
        self.planner._merge_zone_priority = "EGO"
        original = self.planner._trajectory_merge_speed_limit
        self.planner._trajectory_merge_speed_limit = lambda *args: 0.0

        try:
            result = self.planner.plan(ego, reference_path, [])
        finally:
            self.planner._trajectory_merge_speed_limit = original

        self.assertEqual("EGO", self.planner._merge_zone_priority)
        self.assertFalse(result.acc_override)
        self.assertTrue(math.isinf(result.acc_target_velocity))

    def test_designated_merge_still_stops_for_immediate_collision(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000445"]
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        waiting_ego = self._vehicle_state(
            ego_link.id, max(0.0, ego_zone_s - 30.0), 20.0
        )
        entering_other = self._object_entry(
            other_link.id, other_zone_s, 15.0, name="merge_vehicle"
        )

        approach = self.planner.plan(
            waiting_ego,
            self._reference_path(ego_link.id, max(0.0, ego_zone_s - 30.0)),
            [entering_other],
        )
        self.assertEqual("OTHER", self.planner._merge_zone_priority)
        self.assertTrue(approach.acc_override)
        self.assertLess(approach.acc_target_velocity, waiting_ego.velocity)

        zone_ego = self._vehicle_state(ego_link.id, ego_zone_s, 10.0)
        stopped = self.planner.plan(
            zone_ego,
            self._reference_path(ego_link.id, ego_zone_s),
            [entering_other],
        )
        self.assertTrue(stopped.acc_override)
        self.assertEqual(0.0, stopped.acc_target_velocity)

    def test_second_merge_zone_keeps_30kph_when_boundary_is_occupied(self):
        ego_link = self.lane_map.links["A2256W000422"]
        other_link = self.lane_map.links["A2256W000445"]
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        ego_zone_s = self.lane_map.project(merge_point, ego_link).s
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        zone_ego = self._vehicle_state(ego_link.id, ego_zone_s, 10.0)
        entering_other = self._object_entry(
            other_link.id, other_zone_s, 15.0, name="rolling_merge_vehicle"
        )
        matches = self.planner._match_objects([entering_other])
        self.planner._merge_zone_priority = "OTHER"
        self.planner._merge_zone_other_key = "rolling_merge_vehicle"

        limit = self.planner._merge_zone_priority_speed_limit(
            zone_ego,
            ego_link,
            matches,
            [entering_other],
            set(parameters.CONSERVATIVE_MERGE_SUCCESSORS),
            merge_point,
        )

        self.assertAlmostEqual(parameters.MERGE_MIN_YIELD_SPEED, limit)

    def test_safe_zone_keeps_yielding_while_vehicle_enters_successor(self):
        ego_link = self.lane_map.links["A2256W000422"]
        ego_s = max(0.0, ego_link.length - 20.0)
        ego = self._vehicle_state(ego_link.id, ego_s, 20.0)
        other_link = self.lane_map.links["A2256W000445"]
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        other_zone_s = self.lane_map.project(merge_point, other_link).s
        entering_vehicle = self._object_entry(
            other_link.id, other_zone_s, 15.0, name="merge_vehicle"
        )
        self.planner.plan(
            ego,
            self._reference_path(ego_link.id, ego_s),
            [entering_vehicle],
        )
        merged_vehicle = self._object_entry(
            "A2256W000153", 5.0, 15.0, name="merge_vehicle"
        )

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
