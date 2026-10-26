"""Regression coverage for the 17:02 run's late lead/second-merge braking."""
import math
from pathlib import Path
import unittest

from AIM2026.Expert.src.config.parameters import highway as p
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap


class EarlyBrakingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lanes = HighwayLaneMap(str(Path(__file__).resolve().parents[2] /
            'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.planner.set_control_dt(.05)

    def ego(self, lane, y, speed):
        m = self.lanes.project(Point(70., y), self.lanes.links['A2256W000'+lane])
        return VehicleState(m.x, m.y, m.yaw, speed), m

    def test_closing_reserve_slows_early_without_commanding_stop(self):
        # A faster Ego acquires the 40 km/h lead at 30 m bumper clearance.
        # Exercise feedback dynamics, not repeated replay of old Ego poses.
        speed, lead_speed, gap = 19.5, 11., 30.
        minimum_gap, minimum_target = gap, float('inf')
        first = None
        for _ in range(300):
            closing = max(0., speed-lead_speed)
            target = self.planner._acc_speed_limit(speed, gap,
                gap/closing if closing > .1 else float('inf'), lead_speed)
            target = min(20., target)
            first = target if first is None else first
            # Conservative proportional response, with finite actuator force.
            acceleration = max(-3., min(1.5, .6*(target-speed)))
            speed = max(0., speed+acceleration*.05)
            gap += (lead_speed-speed)*.05
            minimum_gap = min(minimum_gap, gap)
            minimum_target = min(minimum_target, target)
        self.assertLess(first, 14.)
        self.assertGreater(minimum_gap, p.ACC_STANDSTILL_DISTANCE)
        self.assertGreater(minimum_target, 0.)
        self.assertAlmostEqual(speed, lead_speed, delta=.5)

    def test_empty_slot_can_acquire_new_front_target_after_commit_not_rear(self):
        ego, origin = self.ego('409', 210., 18.)
        entries = []
        for identity, delta in ((1,-15.), (20,25.)):
            point, yaw, _ = self.lanes.advance_default(origin, delta) if delta > 0 else (
                Point(ego.position.x, ego.position.y+15.), ego.yaw, None)
            obj = ObjectInfo(point.x,point.y,11.,2,unique_id=identity,size_x=4.5,
                             size_y=2.,heading_deg=math.degrees(yaw))
            entries.append({'object_info':obj})
        matches = self.planner._match_objects(entries)
        follow = self.planner.entry_merge.following
        follow.update(ego,origin.lane_link,matches,100.,committed=True,speed_ceiling=20.)
        self.assertEqual(follow.details['lead_id'],20)

    def test_front_gap_uses_rear_axle_consistently_with_collision_boxes(self):
        ego, origin = self.ego('409',210.,11.)
        point,yaw,_ = self.lanes.advance_default(origin,20.)
        obj = ObjectInfo(point.x,point.y,11.,2,unique_id=1,size_x=4.5,size_y=2.,heading_deg=math.degrees(yaw))
        entries = [{'object_info':obj}]
        gap,_ = self.planner._front_risk(ego,origin,self.planner._match_objects(entries),entries)
        self.assertAlmostEqual(gap,20.-p.VEHICLE_LENGTH-2.25,delta=.05)

    def test_second_approach_limit_begins_before_taper_and_survives_lane_handoff(self):
        ego, match = self.ego('434',-90.,16.)
        safety = self.planner.merge_safety
        self.assertEqual(safety.zone_for(match),'SECOND_ENTRY')
        self.assertFalse(safety._second_merge_entered(match))
        path = [r[0] for r in self.lanes.route_points(match,160.,.5)]
        limit = safety.limit(ego,match,[],[],path,desired_velocity=22.22)
        self.assertLessEqual(limit,p.SECOND_MERGE_SLOWDOWN_MAX_SPEED)
        self.planner.changed_lane_hold=True
        self.planner.target_link_id=match.lane_link.id
        result=self.planner.plan(ego,path,[],desired_velocity=22.22)
        self.assertLessEqual(result.acc_target_velocity,p.SECOND_MERGE_SLOWDOWN_MAX_SPEED)

    def test_yield_recovery_is_gradual_only_while_slow_path_is_safe(self):
        ego,match=self.ego('422',-160.,4.)
        safety=self.planner.merge_safety
        safety.zone='SECOND_ENTRY'
        safety.yield_speed=0.
        path=[r[0] for r in self.lanes.route_points(match,160.,.5)]
        speed=safety.limit(ego,match,[],[],path,desired_velocity=12.5)
        self.assertAlmostEqual(speed,ego.velocity+p.MERGE_ACC_RECOVERY_RATE*.05)
        # A real overlapping front vehicle still overrides recovery at once.
        front=ObjectInfo(ego.position.x,ego.position.y-2.,0.,2,
                         unique_id=99,size_x=4.5,size_y=2.,heading_deg=math.degrees(ego.yaw))
        entries=[{'object_info':front}]
        speed=safety.limit(ego,match,self.planner._match_objects(entries),entries,path,desired_velocity=12.5)
        self.assertEqual(speed,0.)

    def test_recovery_does_not_keep_braking_in_front_of_close_rear_vehicle(self):
        ego,match=self.ego('422',-165.,10.)
        rear,rear_match=self.ego('422',-150.,10.)
        obj=ObjectInfo(rear.position.x,rear.position.y,10.,2,unique_id=29,
                       size_x=4.5,size_y=2.,heading_deg=math.degrees(rear.yaw))
        entries=[{'object_info':obj}]
        safety=self.planner.merge_safety
        safety.zone='SECOND_ENTRY'
        safety.yield_speed=0.
        path=[r[0] for r in self.lanes.route_points(match,160.,.5)]
        speed=safety.limit(ego,match,self.planner._match_objects(entries),entries,path,desired_velocity=12.5)
        self.assertGreaterEqual(speed,ego.velocity)


if __name__ == '__main__':
    unittest.main()
