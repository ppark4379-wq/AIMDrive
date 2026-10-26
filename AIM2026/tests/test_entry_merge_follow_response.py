"""Speed-matched admission and the recorded 16:18 one-tick brake regression."""
import math
from pathlib import Path
from unittest.mock import patch
import unittest

from AIM2026.Expert.src.config.parameters import highway as p
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.control.path_controller import PathController


class EntryFollowResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.lanes = HighwayLaneMap(str(root/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.source = self.lanes.links[p.ENTRY_MERGE_SOURCE]
        self.target = self.lanes.links[p.ENTRY_MERGE_TARGET]

    def step(self, ego, entries):
        match = self.lanes.project(ego.position,self.source)
        reference = [row[0] for row in self.lanes.route_points(match,150.,.5)]
        return self.planner.plan(ego,reference,entries,desired_velocity=16.6666667)

    def test_matches_leader_before_turn_and_admits_matched_trajectory(self):
        ego = VehicleState(62.14,249.58,math.radians(-89.),11.88)
        lead = ObjectInfo(66.,232.8,10.,-1,unique_id=8,size_x=4.55,size_y=1.98,heading_deg=-89.)
        result = self.step(ego,[{'object_info':lead}])
        self.assertEqual(result.reason,'entry_merge_approach')
        self.assertLessEqual(result.acc_target_velocity,10.)
        self.assertTrue(any(a.get('speed_policy')=='match_lead'
                            for a in result.debug_details['entry_merge']['attempts']))
        ego.velocity = result.acc_target_velocity
        result = self.step(ego,[{'object_info':lead}])
        self.assertEqual(result.reason,'entry_merge_start')
        self.assertAlmostEqual(result.acc_target_velocity,10.)
        self.assertEqual(result.debug_details['entry_merge']['speed_policy'],'match_lead')
        # The rejected close-gap acceleration must not produce a stop next tick.
        result = self.step(ego,[{'object_info':lead}])
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertNotEqual(result.reason,'entry_merge_front_hazard')

    def recorded_maneuver(self):
        planner = self.planner
        start = VehicleState(62.74897801480256,168.70136533631012,-1.554447211514818,.0004572778)
        candidate = planner._build_candidate(start,self.target,
            self.lanes.project(start.position,self.target),'left',self.source,22.098498,
            prediction_horizon=p.MERGE_SAFETY_HORIZON)
        candidate.entry_merge = candidate.compact_static = True
        candidate.safety_speed = candidate.entry_speed_ceiling = 6.97584141373164
        candidate.speed_limit = candidate.maneuver_speed_limit = candidate.safety_speed
        planner.active_candidate = candidate
        planner.state = planner.CHANGE_LEFT
        planner.source_link_id,planner.target_link_id = self.source.id,self.target.id
        planner.entry_merge.following.command = candidate.safety_speed
        ego = VehicleState(64.43182894145139,158.4848008370027,-1.272772533832692,5.219495561387804)
        planner.set_control_dt(.0348059)
        data = [
            (1,66.58954382,128.61986234,11.07470214,4.91983515,2.03224659,-87.59178653,False),
            (24,69.97114977,148.57618718,10.79909706,4.90355933,2.06586075,-89.85853407,False),
            (30,74.39369987,149.89584214,13.83104513,4.73364514,2.22389865,-77.96600512,False),
            (29,66.23585002,172.20907731,10.82802105,4.78466642,1.97014940,-84.93590786,False),
            (31,73.42761306,178.89818346,11.97213412,4.40924519,1.98173928,-87.11313509,False),
            (33,79.76107744,142.97400488,4.83649397,3.78422296,2.17156339,-168.71129525,True),
        ]
        entries=[]
        for identity,x,y,v,length,width,yaw,uncertain in data:
            obj=ObjectInfo(x,y,v,-1,unique_id=identity,size_x=length,size_y=width,heading_deg=yaw)
            obj.motion_uncertain=uncertain
            entries.append({'object_info':obj})
        return ego,entries

    def test_recorded_distant_heading_outlier_does_not_issue_zero(self):
        ego,entries=self.recorded_maneuver()
        result=self.step(ego,entries)
        info=result.debug_details['entry_merge']
        self.assertEqual(result.reason,'entry_merge_confirm_crossing')
        self.assertAlmostEqual(result.acc_target_velocity,ego.velocity)
        self.assertEqual(info['conflict']['object_id'],29)
        self.assertEqual(info['front_conflict']['object_id'],33)
        self.assertEqual(info['crossing_confirmation']['object_id'],33)
        self.assertFalse(info['verified'])
        # The next frame corrects the crossing heading: resume the SAME path.
        path=self.planner.active_candidate
        entries[-1]['object_info'].heading_deg=-117.5506864
        entries[-1]['object_info'].position=Point(79.77,140.91)
        entries[-1]['object_info'].velocity=7.3864
        result=self.step(ego,entries)
        self.assertNotEqual(result.reason,'entry_merge_front_hazard')
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertIs(self.planner.active_candidate,path)

    def test_confirmation_cannot_hide_a_persistent_crossing(self):
        ego,entries=self.recorded_maneuver()
        for _ in range(8):
            result=self.step(ego,entries)
        self.assertEqual(result.reason,'entry_merge_front_hazard')
        self.assertEqual(result.acc_target_velocity,0.)

    def test_confirmed_crossing_and_close_body_are_not_delayed(self):
        ego,entries=self.recorded_maneuver()
        entries[-1]['object_info'].motion_uncertain=False
        result=self.step(ego,entries)
        self.assertEqual(result.reason,'entry_merge_front_hazard')
        self.planner.reset()
        ego,entries=self.recorded_maneuver()
        obj=entries[-1]['object_info']
        obj.position=Point(ego.position.x+1.,ego.position.y-3.)
        obj.velocity=0.
        result=self.step(ego,[{'object_info':obj}])
        self.assertEqual(result.acc_target_velocity,0.)

    def test_boost_remains_active_above_old_3mps_gate_but_not_for_stops(self):
        ego,entries=self.recorded_maneuver()
        result=self.step(ego,[])
        self.assertGreater(ego.velocity,3.)
        self.assertGreater(result.acc_target_velocity,ego.velocity)
        self.assertEqual(PathController._entry_merge_launch_acceleration_gain(
            result.acc_target_velocity,ego,result),p.ENTRY_MERGE_LAUNCH_P_GAIN)
        self.assertIsNone(PathController._entry_merge_launch_acceleration_gain(0.,ego,result))
        result.debug_details['entry_merge']['phase']='COMPLETE'
        self.assertIsNone(PathController._entry_merge_launch_acceleration_gain(8.,ego,result))

    def test_matched_lead_does_not_override_a_fast_rear_vehicle(self):
        ego=VehicleState(62.14,249.58,math.radians(-89.),10.)
        lead=ObjectInfo(66.,232.8,10.,-1,unique_id=8,size_x=4.55,size_y=1.98,heading_deg=-89.)
        rear=ObjectInfo(66.,253.,15.,-1,unique_id=9,size_x=4.55,size_y=1.98,heading_deg=-89.)
        result=self.step(ego,[{'object_info':lead},{'object_info':rear}])
        self.assertNotEqual(result.reason,'entry_merge_start')
        self.assertIsNone(self.planner.active_candidate)

    def seed_recorded_lost_target(self, age=16.7):
        # 22:39 replay: ID 22 vanished near y=252; Ego later stopped at y=169.
        # Its last gap was only ~3 m, so the old forward-ROI-only release never ran.
        following = self.planner.entry_merge.following
        origin = self.lanes.project(Point(66.6, 252.), self.target)
        following.lead = dict(identity=22, s=origin.s, gap=3.2, speed=5.788,
                              length=6.35, x=66.6, y=252., yaw=-math.pi/2, age=0.)
        following.age = age
        following.command = 0.
        self.planner.set_control_dt(.1)
        return VehicleState(62.75, 168.89, math.radians(-89.), 0.)

    def test_recorded_lost_target_cannot_lock_empty_merge_from_standstill(self):
        ego = self.seed_recorded_lost_target()
        for _ in range(15):
            result = self.step(ego, [])
            self.assertNotEqual(result.debug_details['entry_merge']['target_event'], 'target_lost')
            if result.reason == 'entry_merge_start':
                break
        else:
            self.fail('Expired ID still prevented a clear stopped-state merge')
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertEqual(result.target_link_id, p.ENTRY_MERGE_TARGET)

    def test_expired_target_reselects_current_bus_and_does_not_force_merge(self):
        ego = self.seed_recorded_lost_target()
        bus = ObjectInfo(66.4, 169., 0., -1, unique_id=55, size_x=11.,
                         size_y=2.6, heading_deg=-89., is_static=True)
        result = self.step(ego, [{'object_info':bus}])
        details = result.debug_details['entry_merge']
        self.assertEqual(details['expired_lead_id'], 22)
        self.assertEqual(details['lead_id'], 55)
        self.assertEqual(details['target_event'], 'reselected_after_dropout')
        self.assertNotEqual(result.reason, 'entry_merge_start')
        self.assertIsNone(self.planner.active_candidate)

    def test_short_dropout_still_holds_and_same_id_reconnects(self):
        ego = self.seed_recorded_lost_target(age=0.)
        following = self.planner.entry_merge.following
        self.planner._requested_speed = 10.
        following.update(ego, self.target, [], 30.)
        self.assertEqual(following.details['target_event'], 'predicted_short_dropout')
        self.assertEqual(following.details['lead_id'], 22)
        obj = ObjectInfo(66.6, 251., 5.8, -1, unique_id=22, size_x=6.35,
                         size_y=2.6, heading_deg=-90.)
        entries = [{'object_info':obj}]
        following.update(ego, self.target, self.planner._match_objects(entries), 30.)
        self.assertEqual(following.details['lead_id'], 22)
        self.assertEqual(following.details['lead_age'], 0.)
        self.assertIsNone(following.details['expired_lead_id'])

    def test_expired_target_still_rejects_a_fast_approaching_rear_car(self):
        ego = self.seed_recorded_lost_target()
        rear = ObjectInfo(66.4, 181., 25., -1, unique_id=56, size_x=4.8,
                          size_y=2., heading_deg=-90.)
        for _ in range(8):
            result = self.step(ego, [{'object_info':rear}])
            self.assertNotEqual(result.reason, 'entry_merge_start')
        self.assertIsNone(self.planner.active_candidate)

    def test_target_expiry_keeps_an_already_committed_path(self):
        ego, _ = self.recorded_maneuver()
        candidate = self.planner.active_candidate
        following = self.planner.entry_merge.following
        following.lead = dict(identity=22, s=1., gap=3.2, speed=5.8,
                              length=6.35, x=66.6, y=252., yaw=-math.pi/2, age=0.)
        following.age = 20.
        result = self.step(ego, [])
        self.assertIs(self.planner.active_candidate, candidate)
        self.assertEqual(result.target_link_id, p.ENTRY_MERGE_TARGET)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertIsNone(result.debug_details['entry_merge']['lead_id'])


if __name__=='__main__':
    unittest.main()
