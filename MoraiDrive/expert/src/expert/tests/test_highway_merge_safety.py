import math
from collections import deque
from pathlib import Path
import unittest
from unittest.mock import patch

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState
from expert.src.planning.highway import parameters as p
from expert.src.planning.highway.lane_map import HighwayLaneMap
from expert.src.planning.highway.planner import HighwayLanePlanner


class HighwayMergeSafetyTest(unittest.TestCase):
    def test_second_merge_accepts_smaller_clear_gap_without_changing_default(self):
        speed = 40./3.6
        radius = p.CONSERVATIVE_MERGE_ENTRY_RADIUS
        exit_time = 1. + (2*radius+p.VEHICLE_LENGTH)/speed
        obj = ObjectInfo(0., 0., 10., 1, size_x=4.5, size_y=2.)
        touching = [({'object_info': obj}, None, exit_time+1.2)]
        args = (1., speed, touching, radius)
        # A clear 1.2 s / 12 m gap remains a yield under the first merge defaults.
        self.assertTrue(HighwayLanePlanner._merge_arrivals_overlap(*args))
        self.assertFalse(HighwayLanePlanner._merge_arrivals_overlap(
            *args, clear_time_margin=p.CONSERVATIVE_MERGE_GO_CLEAR_TIME_MARGIN,
            clear_distance=p.CONSERVATIVE_MERGE_CLOSE_GAP_YIELD_DISTANCE))

    def test_second_merge_still_requires_both_clear_time_and_distance(self):
        speed = 40./3.6
        radius = p.CONSERVATIVE_MERGE_ENTRY_RADIUS
        exit_time = 1. + (2*radius+p.VEHICLE_LENGTH)/speed
        for gap_time, other_speed in ((.8, 30.), (1.2, 5.), (-.2, 10.)):
            with self.subTest(gap_time=gap_time, other_speed=other_speed):
                obj = ObjectInfo(0., 0., other_speed, 1, size_x=4.5, size_y=2.)
                touching = [({'object_info': obj}, None, exit_time+gap_time)]
                self.assertTrue(HighwayLanePlanner._merge_arrivals_overlap(
                    1., speed, touching, radius,
                    clear_time_margin=p.CONSERVATIVE_MERGE_GO_CLEAR_TIME_MARGIN,
                    clear_distance=p.CONSERVATIVE_MERGE_CLOSE_GAP_YIELD_DISTANCE))

    @classmethod
    def setUpClass(cls):
        map_dir = Path(__file__).resolve().parents[3] / "map_data" / "R_KR_PG_KATRI_2025"
        cls.lane_map = HighwayLaneMap(str(map_dir))

    def test_both_merges_with_actuator_delay_and_moving_front_or_rear(self):
        for ego_id, other_id, center in (
            ("A2256W000411", "A2256W000409", p.PREDICTIVE_MERGE_POINT),
            ("A2256W000422", "A2256W000445", p.CONSERVATIVE_MERGE_POINT),
        ):
            for ego_back, other_back, speed, other_speed in (
                (75.0, 50.0, 16.0, 12.0),
                (65.0, 80.0, 16.0, 24.0),
                (40.0, 20.0, 17.0, 15.0),
            ):
                with self.subTest(merge=ego_id, ego_back=ego_back, other_speed=other_speed):
                    planner = HighwayLanePlanner(lane_map=self.lane_map)
                    planner.set_control_dt(0.1)
                    ego_origin = self.lane_map.project(Point(*center), self.lane_map.links[ego_id])
                    other_origin = self.lane_map.project(Point(*center), self.lane_map.links[other_id])
                    ego_origin.s -= ego_back
                    other_origin.s -= other_back
                    ego_distance = other_distance = 0.0
                    delay = deque([speed] * 3)
                    minimum_speed = speed
                    steps = 0
                    for step in range(160):
                        point, yaw, link = self.lane_map.advance_default(ego_origin, ego_distance)
                        other_point, other_yaw, other_link = self.lane_map.advance_default(other_origin, other_distance)
                        ego = VehicleState(point.x, point.y, yaw, speed)
                        obj = ObjectInfo(other_point.x, other_point.y, other_speed, 1,
                                         size_x=4.5, size_y=2.0, heading_deg=math.degrees(other_yaw), unique_id=12)
                        match = self.lane_map.project(point, link)
                        object_match = self.lane_map.project(other_point, other_link)
                        path = [pose[0] for pose in self.lane_map.route_points(match, 140.0, 0.5)]
                        entry = {"object_info": obj}
                        target = planner.merge_safety.limit(ego, match, [(entry, object_match)],
                                                            [entry], path, desired_velocity=22.0)
                        if target is None:
                            break
                        self.assertFalse(planner._obb_collision(point, yaw, other_point, other_yaw, obj),
                                         (step, speed, planner.merge_safety.details.get("conflict")))
                        delay.append(target)
                        applied_target = delay.popleft()
                        change = max(-0.3, min(0.15, applied_target - speed))
                        ego_distance += (speed + 0.5 * change) * 0.1
                        speed += change
                        other_distance += other_speed * 0.1
                        minimum_speed = min(minimum_speed, speed)
                        steps += 1
                    self.assertGreater(steps, 30)
                    self.assertGreater(ego_distance, ego_back + 10.0)
                    self.assertGreater(minimum_speed, 7.5)

    def _second_merge_rear_scene(self, lateral=1.1, rear_speed=13.89, front=False):
        # Synthetic scene anchored at the reported Ego pose, not an NPC replay.
        planner = HighwayLanePlanner(lane_map=self.lane_map)
        yaw = math.radians(-85.37)
        ego = VehicleState(73.13, -192.17, yaw, 10.11)
        match = self.lane_map.project(ego.position, self.lane_map.links["A2256W000422"])
        rear = ObjectInfo(ego.position.x - 30*math.cos(yaw) - lateral*math.sin(yaw),
                          ego.position.y - 30*math.sin(yaw) + lateral*math.cos(yaw),
                          rear_speed, 1, size_x=4.5, size_y=2.,
                          heading_deg=math.degrees(yaw), unique_id=27)
        objects = [rear]
        if front:
            objects.append(ObjectInfo(ego.position.x + 14*math.cos(yaw),
                                      ego.position.y + 14*math.sin(yaw), 0., 1,
                                      size_x=4.5, size_y=2., heading_deg=math.degrees(yaw), unique_id=28))
        entries = [{"object_info": obj} for obj in objects]
        matches = [(entry, self.lane_map.match(entry["object_info"].position, heading=yaw))
                   for entry in entries]
        path = [Point(ego.position.x + d*math.cos(yaw), ego.position.y + d*math.sin(yaw))
                for d in range(151)]
        return planner, ego, match, matches, entries, path

    def test_second_merge_rear_safe_hold_survives_speed_control(self):
        import json
        from expert.src.planning.highway.structures import HighwayPlanningResult
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.control.path_controller import PathController
        for lateral in (1.1, 1.8):
            with self.subTest(lateral=lateral):
                planner, ego, match, matches, entries, path = self._second_merge_rear_scene(lateral)
                full_result = planner.plan(ego, path, entries, desired_velocity=7.83,
                                           allow_lane_change=False)
                limit = full_result.acc_target_velocity
                self.assertAlmostEqual(ego.velocity, limit)
                self.assertEqual("EGO", planner._merge_zone_priority)
                self.assertEqual("second_merge_committed_speed", planner.merge_safety.details["cause"])
                self.assertIsNone(planner.merge_safety.details["conflict"])
                result = HighwayPlanningResult(path, debug_details={"merge_zone_priorities": {"SECOND_ENTRY": "EGO"}})
                planned = AutonomousDriving._merge_ego_curvature_speed_floor(7.83, ego.velocity, result)
                self.assertAlmostEqual(ego.velocity, min(planned, limit))
                config = json.loads((Path(__file__).resolve().parents[1]
                                     / "src/config/config.json").read_text())
                controller = PathController(config)
                command = controller.control_path(path, ego, min(planned, limit), result)
                self.assertEqual(0., command.brake)

    def test_first_merge_does_not_use_second_merge_recovery(self):
        planner, ego, match, matches, entries, path = self._second_merge_rear_scene()
        with patch.object(planner.merge_safety, "zone_for", return_value="FIRST_PREDICTIVE"):
            limit = planner.merge_safety.limit(ego, match, matches, entries, path, 7.83)
        self.assertEqual(0., limit)
        self.assertNotIn("hold_conflict", planner.merge_safety.details)

    def test_second_merge_before_entry_still_yields(self):
        planner, ego, match, matches, entries, path = self._second_merge_rear_scene()
        # Real route-progress gate must stay closed before the conflict area exit.
        center = self.lane_map.project(Point(*p.CONSERVATIVE_MERGE_POINT), match.lane_link)
        match.s = center.s - p.CONSERVATIVE_MERGE_ENTRY_RADIUS - 5.
        self.assertFalse(planner.merge_safety._second_merge_entered(match))
        self.assertEqual(0., planner.merge_safety.limit(ego, match, matches, entries, path, 7.83))

    def test_second_merge_hold_respects_front_speed_limit(self):
        planner, ego, match, matches, entries, path = self._second_merge_rear_scene()
        limit = planner._merge_acc_speed_limit(ego, match, matches, entries, path,
                                               desired_velocity=7.83, forward_speed_limit=5.)
        self.assertLessEqual(limit, 5.)
        self.assertNotIn("hold_conflict", planner.merge_safety.details)

    def test_second_merge_front_obstacle_blocks_hold(self):
        planner, ego, match, matches, entries, path = self._second_merge_rear_scene(front=True)
        limit = planner.merge_safety.limit(ego, match, matches, entries, path, 7.83)
        self.assertLessEqual(limit, 7.83)
        self.assertNotEqual("second_merge_rear_safe_hold", planner.merge_safety.details.get("cause"))

    def test_second_merge_mitigates_rear_risk_without_claiming_safe_go(self):
        planner, ego, match, matches, entries, path = self._second_merge_rear_scene(rear_speed=20.)
        limit = planner.merge_safety.limit(ego, match, matches, entries, path, 7.83)
        details = planner.merge_safety.details
        self.assertEqual("MITIGATE", details["decision"])
        self.assertGreater(limit, 0.)
        self.assertLessEqual(limit, ego.velocity)
        self.assertGreater(details["conflict"]["time"], details["stop_conflict"]["time"])
        self.assertEqual("second_merge_rear_risk_mitigation", details["cause"])

    def test_second_merge_rear_mitigation_cannot_hide_another_front_hazard(self):
        planner, ego, match, matches, entries, path = self._second_merge_rear_scene(rear_speed=20.)
        def conflict(*args, **kwargs):
            if kwargs.get("ignore_ids"):
                return dict(time=4.5, id="npc:28", rear=False, kind="obb")
            return dict(time=2. if args[3] == 0. else 4., id="npc:27", rear=True, kind="gap")
        with patch.object(planner.merge_safety, "_conflict", side_effect=conflict):
            limit = planner.merge_safety.limit(ego, match, matches, entries, path, 7.83)
        self.assertEqual(0., limit)
        self.assertEqual("STOP", planner.merge_safety.details["decision"])

    def test_second_merge_entry_speed_survives_repeated_profile_drops(self):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.planning.highway.structures import HighwayPlanningResult
        from expert.src.control.path_controller import PathController
        import json
        planner = HighwayLanePlanner(lane_map=self.lane_map)
        controller = PathController(json.loads((Path(__file__).resolve().parents[1]
                                               / "src/config/config.json").read_text()))
        # Sequential logged Ego locations and profile speeds. No exact NPC snapshot
        # exists, so this isolates the proven profile-driven braking regression.
        for y, profile, speed in [(-170.76,13.79,12.41),(-183.11,10.73,11.81),
                                  (-193.75,7.29,9.52),(-202.77,7.29,8.52),(-207.43,7.29,7.69)]:
            ego = VehicleState(73., y, -math.pi/2, speed)
            match = self.lane_map.project(ego.position, self.lane_map.links["A2256W000422"])
            path = [Point(73., y-d) for d in range(151)]
            limit = planner.merge_safety.limit(ego, match, [], [], path, profile)
            self.assertGreaterEqual(limit, 12.4)
            result = HighwayPlanningResult(path, debug_details={"merge_safety": dict(planner.merge_safety.details)})
            planned = AutonomousDriving._merge_ego_curvature_speed_floor(profile, ego.velocity, result)
            command = controller.control_path(path, ego, min(planned,limit,p.SECOND_MERGE_SLOWDOWN_MAX_SPEED), result)
            self.assertEqual(0., command.brake)
        self.assertEqual(12.5, planner.merge_safety.second_entry_speed)
        planner.merge_safety.reset()
        self.assertIsNone(planner.merge_safety.second_entry_speed)

    def test_second_merge_committed_speed_never_overrides_front_stop(self):
        planner, ego, match, matches, entries, path = self._second_merge_rear_scene()
        planner.merge_safety.second_entry_speed = 12.4
        self.assertEqual(0., planner.merge_safety.limit(ego, match, matches, entries, path,
                                                      0., forward_speed_limit=0.))
        self.assertNotIn("speed_floor", planner.merge_safety.details)

    def test_opening_front_gap_uses_measured_lead_speed(self):
        limit = HighwayLanePlanner._acc_speed_limit(12.0, 15.0, float("inf"), lead_speed=20.0)
        self.assertGreaterEqual(limit, 12.0)

    def test_close_stopped_lead_still_brakes(self):
        limit = HighwayLanePlanner._acc_speed_limit(12.0, 4.0, 0.33, lead_speed=0.0)
        self.assertEqual(0.0, limit)

    def test_validated_clear_ceiling_is_not_held_at_old_low_speed(self):
        planner = HighwayLanePlanner(lane_map=self.lane_map)
        planner._merge_guard_active = True
        planner._merge_speed_limit_state = 2.0
        self.assertEqual(18.0, planner._stabilize_merge_speed_limit(18.0, 12.0))


if __name__ == "__main__":
    unittest.main()
