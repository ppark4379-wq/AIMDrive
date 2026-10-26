"""Rear clearance must not create a stop in front of an established follower."""
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner


class SecondMergeRearSpacingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.lanes = HighwayLaneMap(str(root/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.safety = self.planner.merge_safety
        self.safety.zone = 'SECOND_ENTRY'
        self.ego = VehicleState(0., 0., 0., 5.)
        self.match = NS(lane_link=self.lanes.links['A2256W000445'], s=40.)
        self.other_match = NS(lane_link=self.match.lane_link, s=30.)

    def car(self, x=-4.5, y=0., speed=5., heading=0.):
        return ObjectInfo(x, y, speed, 2, unique_id=37, size_x=4.5,
                          size_y=2., heading_deg=heading)

    def is_follower(self, car, entered=True):
        with patch.object(self.safety, '_second_merge_entered', return_value=entered):
            return self.safety._second_merge_follower(self.ego, self.match, car, self.other_match)

    def test_close_rear_gap_does_not_revoke_follower(self):
        # 2.25 m bumper clearance, below the 3 m merge spacing threshold.
        self.assertTrue(self.is_follower(self.car()))
        self.assertTrue(self.is_follower(self.car(x=-3.0)))

    def test_side_converging_front_and_overlapping_cars_are_not_exempt(self):
        self.assertFalse(self.is_follower(self.car(y=3.5)))
        self.assertFalse(self.is_follower(self.car(heading=40.)))
        self.assertFalse(self.is_follower(self.car(x=10.)))
        self.assertFalse(self.is_follower(self.car(x=-1.)))
        self.assertFalse(self.is_follower(self.car(), entered=False))
        self.safety.zone = 'FIRST_PREDICTIVE'
        self.assertFalse(self.is_follower(self.car()))

    def test_recorded_stopped_geometry_is_recognized_without_map_centerline_alignment(self):
        self.ego = VehicleState(71.24675335135544, -180.846789768897,
                                math.radians(-84.94260669747747), 0.)
        car = self.car(71.99589593792022, -176.18717020234536,
                       heading=-87.21881799024541)
        self.match = self.lanes.project(self.ego.position, self.lanes.links['A2256W000445'])
        self.other_match = self.lanes.match(car.position, heading=math.radians(car.heading_deg))
        self.assertTrue(self.is_follower(car))

    def conflict(self, car, predicted_x=None):
        predictions = [[(car.position, math.radians(car.heading_deg))]]
        if predicted_x is not None:
            predictions.append([(Point(predicted_x, car.position.y), math.radians(car.heading_deg))])
        return self.safety._conflict(self.ego, [Point(0.,0.),Point(100.,0.)],
                                    [0.,100.], 5., [('rear',(car,self.other_match,0.))],
                                    [[poses] for poses in predictions])

    def test_only_rear_spacing_is_exempt_not_current_or_future_body_collision(self):
        self.safety.second_rear_gap_ids = {'rear'}
        self.assertIsNone(self.conflict(self.car()))
        self.assertEqual(self.conflict(self.car(x=-1.))['kind'], 'obb')
        self.assertEqual(self.conflict(self.car(), predicted_x=0.)['kind'], 'obb')
        # A future/current front clearance still applies even to the same ID.
        self.assertIsNotNone(self.conflict(self.car(x=9.)))

    def test_non_follower_rear_spacing_is_still_checked(self):
        hazard = self.conflict(self.car())
        self.assertEqual(hazard['kind'], 'gap')
        self.assertTrue(hazard['rear'])

    def test_limit_keeps_driving_with_close_follower_and_still_stops_for_front(self):
        rear = self.car()
        tracks = [('rear',(rear,self.other_match,0.))]
        with patch.object(self.safety, 'zone_for', return_value='SECOND_ENTRY'), \
             patch.object(self.safety, '_second_merge_entered', return_value=True), \
             patch.object(self.safety, '_update_tracks', return_value=tracks), \
             patch.object(self.safety, '_shares_corridor', return_value=True), \
             patch.object(self.safety, '_route', return_value=([Point(0.,0.),Point(100.,0.)],[0.,100.])), \
             patch.object(self.safety, '_pose', side_effect=lambda obj,match,t: [(Point(obj.position.x+obj.velocity*t,obj.position.y),0.)]):
            speed = self.safety.limit(self.ego,self.match,[],[],desired_velocity=5.)
            self.assertEqual(speed,5.)
            self.assertEqual(self.safety.details['rear_spacing_only_exempt'],['rear'])
            front = self.car(x=3.,speed=0.)
            tracks.append(('front',(front,self.other_match,0.)))
            self.assertEqual(self.safety.limit(self.ego,self.match,[],[],desired_velocity=5.),0.)
            self.assertEqual(self.safety.details['conflict']['id'],'front')

    def test_reset_clears_exemptions(self):
        self.safety.second_rear_gap_ids.add('rear')
        self.safety.reset()
        self.assertFalse(self.safety.second_rear_gap_ids)

    def test_recorded_32kph_approach_accelerates_instead_of_stopping_for_rear(self):
        ego=VehicleState(69.84642723,-164.90075903,math.radians(-83.9),8.81190459)
        match=self.lanes.match(ego.position,heading=ego.yaw)
        rear=ObjectInfo(71.46,-143.95,52.9/3.6,2,unique_id=47,
                       size_x=4.73,size_y=2.04,heading_deg=-88.2)
        entries=[{'object_info':rear}]
        matches=self.planner._match_objects(entries)
        path=[v[0]for v in self.lanes.route_points(match,130.,.5)]
        self.assertFalse(self.safety._second_merge_entered(match))
        target=self.safety.limit(ego,match,matches,entries,path,desired_velocity=12.5)
        self.assertEqual(target,12.5)
        self.assertTrue(self.safety.details['launch_verified'])
        self.assertEqual(self.safety.details['decision'],'GO')
        # A real stopped front body must still prohibit this acceleration.
        front=ObjectInfo(70.,-169.,0.,2,unique_id=99,size_x=4.5,size_y=2.,heading_deg=-85.)
        entries.append({'object_info':front})
        self.assertEqual(self.safety.limit(ego,match,self.planner._match_objects(entries),entries,
                                          path,desired_velocity=12.5),0.)


if __name__ == '__main__':
    unittest.main()
