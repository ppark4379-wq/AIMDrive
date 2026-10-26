"""18:33 run: parallel track 28's side padding must not command STOP."""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from AIM2026.Expert.src.config.parameters import highway as p
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.control.path_controller import PathController


class ParallelVehicleClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.lanes = HighwayLaneMap(str(cls.root/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.planner.changed_lane_hold = True
        self.planner.target_link_id = 'A2256W000409'
        self.ego = VehicleState(66.56257664848818,148.44274187553674,-1.5645075373114823,11.973406473795572)
        self.neighbor = ObjectInfo(68.88758355704465,145.89509751450012,4.16733455657959,2,
            unique_id=28,size_x=5.937471091747284,size_y=2.418747901916504,heading_deg=-89.80884675241263)
        self.neighbor.motion_uncertain = True
        self.lead = ObjectInfo(66.58923114296971,128.72514474326545,9.505193710327148,2,
            unique_id=22,size_x=5.201255977153778,size_y=2.1926627159118652,heading_deg=-89.98976864125335)
        self.entries = [{'object_info':o} for o in (self.lead,self.neighbor)]
        self.match = self.lanes.project(self.ego.position,self.lanes.links['A2256W000409'])
        self.path = [v[0] for v in self.lanes.route_points(self.match,150.,.5)]

    def plan(self):
        return self.planner.plan(self.ego,self.path,self.entries,desired_velocity=22.222,
                                 allow_lane_change=False)

    def test_neighbor_jitter_and_duplicate_only_change_keep_lane_view(self):
        p = self.planner
        truck = ObjectInfo(70.1,147.,10.,2,unique_id=5,size_x=7.,size_y=2.8,heading_deg=-90.)
        truck.motion_uncertain = False
        entries = [{'object_info':truck}]
        p._keep_lane_observations(self.ego,self.match,entries)
        truck.position = Point(68.45,146.)
        view = p._keep_lane_observations(self.ego,self.match,entries)
        self.assertGreater(view[0]['object_info'].position.x,69.5)
        self.assertEqual(truck.position.x,68.45)
        self.assertIn(5,p._keep_lane_evidence['corrected'])
        truck.position = Point(70.1,145.)
        duplicate = ObjectInfo(68.24,145.,10.,2,unique_id=23,size_x=6.5,size_y=2.62,heading_deg=-90.)
        duplicate.motion_uncertain = True
        entries.append({'object_info':duplicate})
        view = p._keep_lane_observations(self.ego,self.match,entries)
        self.assertEqual([5],[e['object_info'].unique_id for e in view])
        self.assertEqual(len(entries),2)
        # The same raw boxes must BOTH constrain our own lateral maneuver.
        p.state=p.CHANGE_LEFT
        self.assertIs(p._keep_lane_observations(self.ego,self.match,entries),entries)

    def test_noisy_neighbor_exception_does_not_hide_stopped_own_lane_box(self):
        truck=ObjectInfo(70.1,147.,10.,2,unique_id=5,size_x=7.,size_y=2.8,heading_deg=-90.)
        truck.motion_uncertain=False
        self.planner._keep_lane_observations(self.ego,self.match,[{'object_info':truck}])
        stopped=ObjectInfo(66.6,145.,0.,2,unique_id=23,size_x=6.5,size_y=2.62,heading_deg=-90.)
        stopped.motion_uncertain=True
        entries=[{'object_info':truck},{'object_info':stopped}]
        view=self.planner._keep_lane_observations(self.ego,self.match,entries)
        self.assertIn(stopped,[e['object_info'] for e in view])

    def test_lateral_anchor_expires_instead_of_hiding_persistent_overlap(self):
        truck=ObjectInfo(70.1,147.,10.,2,unique_id=5,size_x=7.,size_y=2.8,heading_deg=-90.)
        truck.motion_uncertain=False
        self.planner._keep_lane_observations(self.ego,self.match,[{'object_info':truck}])
        truck.position=Point(68.,146.)
        self.planner.set_control_dt(.1)
        for _ in range(6):
            view=self.planner._keep_lane_observations(self.ego,self.match,[{'object_info':truck}])
        self.assertIs(view[0]['object_info'],truck)

    def test_recorded_side_padding_overlap_follows_front_without_stopping(self):
        obj=self.neighbor
        self.assertTrue(self.planner._obb_collision(self.ego.position,self.ego.yaw,
            obj.position,math.radians(obj.heading_deg),obj))
        result=self.plan()
        self.assertGreater(result.acc_target_velocity,10.)
        self.assertLess(result.acc_target_velocity,self.ego.velocity)
        self.assertIn('npc:28',result.debug_details['merge_safety']['parallel_side_margin_only'])
        self.assertNotEqual(result.debug_details['merge_safety']['decision'],'STOP')
        self.assertTrue(result.candidates[0].valid)

    def test_actual_neighbor_body_overlap_still_stops(self):
        self.neighbor.position=Point(self.ego.position.x+1.6,self.ego.position.y-2.)
        self.assertEqual(self.plan().acc_target_velocity,0.)

    def test_missing_lane_changed_heading_and_disabled_assumption_keep_margin(self):
        match=self.lanes.match(self.neighbor.position,math.radians(self.neighbor.heading_deg))
        margin=self.planner._parallel_lane_margin
        self.assertEqual(margin(self.ego,self.match,self.neighbor,None),p.COLLISION_MARGIN)
        with patch.object(p,'ASSUME_HIGHWAY_VEHICLES_KEEP_LANE',False):
            self.assertEqual(margin(self.ego,self.match,self.neighbor,match),p.COLLISION_MARGIN)
            self.assertEqual(self.plan().acc_target_velocity,0.)
        self.neighbor.heading_deg=-65.
        self.assertEqual(margin(self.ego,self.match,self.neighbor,match),p.COLLISION_MARGIN)

    def test_our_lane_change_and_converging_lanes_keep_full_margin(self):
        match=self.lanes.match(self.neighbor.position,math.radians(self.neighbor.heading_deg))
        self.planner.state=self.planner.CHANGE_LEFT
        self.assertEqual(self.planner._parallel_lane_margin(self.ego,self.match,self.neighbor,match),p.COLLISION_MARGIN)
        self.planner.state=self.planner.KEEP
        for a,b,y in [('411','409',200.),('408','434',-90.),('445','422',-170.)]:
            own=self.lanes.project(Point(70.,y),self.lanes.links['A2256W000'+a])
            other=self.lanes.project(Point(70.,y),self.lanes.links['A2256W000'+b])
            ego=VehicleState(own.x,own.y,own.yaw,10.)
            obj=ObjectInfo(other.x,other.y,10.,2,heading_deg=math.degrees(other.yaw))
            self.assertEqual(self.planner._parallel_lane_margin(ego,own,obj,other),p.COLLISION_MARGIN)

    def test_future_real_overlap_not_ignored_for_parallel_track(self):
        safety=self.planner.merge_safety
        safety.zone='FIRST_PREDICTIVE'
        safety.ego_lane_match=self.match
        other=self.lanes.match(self.neighbor.position,math.radians(self.neighbor.heading_deg))
        tracks=[('npc:28',(self.neighbor,other,0.))]
        # Current side padding only; next sample genuinely intersects our body.
        predictions=[[[(self.neighbor.position,math.radians(self.neighbor.heading_deg))]],
                     [[(Point(self.ego.position.x,self.ego.position.y-3.),self.ego.yaw)]]]
        hazard=safety._conflict(self.ego,self.path,self.planner._cumulative_distance(self.path),10.,tracks,predictions)
        self.assertIsNotNone(hazard)
        self.assertEqual(hazard['time'],p.MERGE_SAFETY_STEP)
        self.assertEqual(hazard['kind'],'obb')

    def test_verified_release_uses_measured_speed_and_clears_old_brake(self):
        self.entries=[]
        self.ego.velocity=4.89209598965115
        self.planner.merge_safety.zone='FIRST_PREDICTIVE'
        self.planner.merge_safety.yield_speed=0.
        result=self.plan()
        self.assertGreaterEqual(result.acc_target_velocity,self.ego.velocity)
        config=json.loads((self.root/'AIM2026/Expert/src/config/config.json').read_text())
        controller=PathController(config)
        controller.previous_target_velocity=0.
        controller.previous_acc_cmd=-1.
        with patch.object(controller.pid,'get_output',return_value=-.1):
            command=controller._calculate_control(self.ego,result.path,result.acc_target_velocity,result)[0]
            self.assertGreaterEqual(command,0.)
            # A lower signal/front limit is never overridden by release metadata.
            self.assertEqual(controller._calculate_control(self.ego,result.path,0.,result)[0],-1.)

    def recorded_unmatched_case(self):
        # 19:05:10 run, 28.813 s: track 22 exists in exactly one inference frame.
        ego = VehicleState(70.28924688778352, 111.90297676669434,
                           -1.5158237837312916, 15.161852306789822)
        obj = ObjectInfo(86.32432322, 86.81029785, 14.993276596069336, 2,
                         unique_id=22, size_x=2.8072609305381775,
                         size_y=1.0952095985412598, heading_deg=156.3072502474908)
        obj.motion_uncertain = True
        planner = HighwayLanePlanner(lane_map=self.lanes)
        planner.changed_lane_hold = True
        planner.target_link_id = 'A2256W000410'
        match = self.lanes.project(ego.position, self.lanes.links[planner.target_link_id])
        path = [row[0] for row in self.lanes.route_points(match, 150., .5)]
        return planner, ego, obj, match, path

    def test_single_frame_unmatched_heading_does_not_invent_cut_in(self):
        planner, ego, obj, match, path = self.recorded_unmatched_case()
        entries = [{'object_info': obj}]
        self.assertEqual(planner._match_objects(entries), [])
        with patch.object(p, 'ASSUME_HIGHWAY_VEHICLES_KEEP_LANE', False):
            old = planner.plan(ego, path, entries, desired_velocity=80/3.6, allow_lane_change=False)
            self.assertEqual(old.acc_target_velocity, 0.)
        # Reusing the same observation must not mature its motion confidence.
        for _ in range(12):
            result = planner.plan(ego, path, entries, desired_velocity=80/3.6, allow_lane_change=False)
            self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
            self.assertIn('road-parallel ids=[22]', ' '.join(result.debug_details['merge_checks']))
        self.assertAlmostEqual(obj.velocity, 14.993276596069336)
        self.assertAlmostEqual(obj.heading_deg, 156.3072502474908)

    def test_confirmed_unmatched_crossing_still_brakes(self):
        planner, ego, obj, match, path = self.recorded_unmatched_case()
        obj.motion_uncertain = False
        result = planner.plan(ego, path, [{'object_info': obj}], desired_velocity=80/3.6,
                              allow_lane_change=False)
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_no_provisional_exemption_during_own_change_or_merge(self):
        planner, ego, obj, match, _ = self.recorded_unmatched_case()
        entries = [{'object_info': obj}]
        planner.state = planner.CHANGE_LEFT
        self.assertEqual(planner._trajectory_merge_speed_limit(ego, match, [], entries), 0.)
        planner.state = planner.KEEP
        with patch.object(planner.merge_safety, 'zone_for', return_value='SECOND_ENTRY'):
            self.assertEqual(planner._trajectory_merge_speed_limit(ego, match, [], entries), 0.)

    def test_current_unmatched_body_overlap_is_checked_immediately(self):
        planner, ego, obj, match, _ = self.recorded_unmatched_case()
        obj.position = Point(ego.position.x, ego.position.y-2.)
        self.assertEqual(planner._trajectory_merge_speed_limit(
            ego, match, [], [{'object_info': obj}]), 0.)

    def test_cleared_fallback_drops_stale_stop_but_preserves_front_stop(self):
        planner, ego, obj, match, path = self.recorded_unmatched_case()
        planner._merge_speed_limit_state = 0.
        clear = planner.plan(ego, path, [], desired_velocity=80/3.6, allow_lane_change=False)
        self.assertTrue(clear.debug_details['highway_brake_release']['verified'])
        self.assertGreaterEqual(clear.acc_target_velocity, ego.velocity)
        controller = PathController(json.loads((self.root/'AIM2026/Expert/src/config/config.json').read_text()))
        controller.previous_target_velocity = 0.
        controller.previous_acc_cmd = -1.
        with patch.object(controller.pid, 'get_output', return_value=-.5):
            self.assertGreaterEqual(controller._calculate_control(ego, clear.path, 80/3.6, clear)[0], 0.)
            self.assertLess(controller._calculate_control(ego, clear.path, ego.velocity-2., clear)[0], 0.)
            self.assertEqual(controller._calculate_control(ego, clear.path, 0., clear)[0], -1.)
        front = ObjectInfo(ego.position.x, ego.position.y-2., 0., 2,
                           unique_id=99, size_x=4.5, size_y=2., heading_deg=-90.)
        stop = planner.plan(ego, path, [{'object_info':front}], desired_velocity=80/3.6,
                            allow_lane_change=False)
        self.assertEqual(stop.acc_target_velocity, 0.)


if __name__ == '__main__':
    unittest.main()
