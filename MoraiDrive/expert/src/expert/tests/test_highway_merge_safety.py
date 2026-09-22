import math
from collections import deque
from pathlib import Path
import unittest

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState
from expert.src.planning.highway_lane_planning import parameters as p
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner


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
