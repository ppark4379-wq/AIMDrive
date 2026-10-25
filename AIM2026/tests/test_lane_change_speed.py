"""Ordinary lane-change speed commitment and independent stop authorities."""
import contextlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.planning.common.target_speed import target_speed
from AIM2026.Expert.src.control.path_controller import PathController


class LaneChangeSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.lanes = HighwayLaneMap(str(cls.root/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.planner._requested_speed = 16.
        self.source = self.lanes.links['A2256W000410']
        self.target = self.lanes.links['A2256W000418']
        match = self.lanes.project(Point(70., 220.), self.source)
        self.ego = VehicleState(match.x, match.y, match.yaw, 12.)

    def candidate(self, distance=55.):
        return self.planner._build_candidate(self.ego, self.target,
            self.lanes.project(self.ego.position, self.target), 'left', self.source, distance)

    def result(self, candidate, acc=0., merge=0.):
        self.planner.state = self.planner.CHANGE_LEFT
        return self.planner._result_during_change(candidate, 4., .5, acc,
            dict(debug_details={'filtered_merge_limit': merge}))

    def test_full_maneuver_holds_entry_speed_despite_later_lower_road_profile(self):
        candidate = self.candidate()
        self.planner._evaluate_lane_change(candidate, self.ego, [], [])
        self.assertTrue(candidate.valid)
        self.assertEqual(candidate.safety_speed, 12.)
        self.planner._requested_speed = 6.
        self.ego.velocity = 11.8
        self.planner._evaluate_lane_change(candidate, self.ego, [], [], committed=True)
        self.assertTrue(candidate.valid)
        result = self.result(candidate)
        self.assertEqual(result.acc_target_velocity, 12.)
        self.assertTrue(result.debug_details['lane_change_cruise']['verified'])

    def test_complete_planner_passes_static_car_without_carrying_old_lane_acc(self):
        match = self.lanes.project(self.ego.position, self.source)
        self.assertIsNone(self.planner.merge_safety.zone_for(match))
        lead = self.lanes.project(Point(70., 180.), self.source)
        obj = ObjectInfo(lead.x,lead.y,0.,2,is_static=True,unique_id=9,size_x=4.5,size_y=1.8,
                         heading_deg=math.degrees(lead.yaw))
        reference = [row[0] for row in self.lanes.route_points(match,100.,.5)]
        result = self.planner.plan(self.ego,reference,[{'object_info':obj}],desired_velocity=16.)
        self.assertIn(result.highway_state, (self.planner.CHANGE_LEFT,self.planner.CHANGE_RIGHT))
        self.assertLess(result.debug_details['front_acc_limit'],self.ego.velocity)
        self.assertEqual(result.acc_target_velocity,self.ego.velocity)
        self.assertEqual(result.debug_details['lane_change_cruise']['speed'],self.ego.velocity)

    def test_rejects_short_curve_instead_of_starting_with_reduced_speed(self):
        candidate = self.candidate(22.)
        candidate.compact_static = candidate.static_escape = True
        candidate.speed_limit = 6.
        self.planner._evaluate_lane_change(candidate, self.ego, [], [])
        self.assertFalse(candidate.valid)
        self.assertEqual(candidate.prediction_conflict['kind'], 'change_speed_not_supported')

    def test_conflicting_gap_does_not_retry_lower_speeds_before_or_during_change(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                candidate = self.candidate()
                candidate.static_escape = True
                calls = []
                def check(c, e, entries, matches, speed, **kw):
                    calls.append(speed)
                    return {'time': 3., 'rear': False, 'kind': 'gap'} if speed > 8. else None
                with patch.object(self.planner.lane_change_safety, 'check', side_effect=check):
                    self.planner._evaluate_lane_change(candidate, self.ego, [], [], committed=committed)
                self.assertEqual(calls[0],12.)
                self.assertTrue(all(speed >= 12. for speed in calls))
                self.assertFalse(candidate.valid)

    def test_new_front_body_still_stops_on_the_committed_path(self):
        candidate = self.candidate()
        self.planner._evaluate_lane_change(candidate, self.ego, [], [])
        a,b = candidate.points[:2]
        obj = ObjectInfo(a.x, a.y, 0., 2, unique_id=99,
                         size_x=4.5, size_y=2., heading_deg=math.degrees(self.ego.yaw))
        traffic = [{'object_info': obj}]
        self.planner.obstacle_motion.update(traffic, .1)
        self.planner._evaluate_lane_change(candidate, self.ego, traffic,
            self.planner._match_objects(traffic), committed=True)
        result = self.result(candidate)
        self.assertEqual(result.acc_target_velocity, 0.)
        self.assertEqual(result.reason, 'highway_lane_change_emergency_stop')
        self.assertIs(result.path, candidate.points)

    def test_rear_risk_does_not_mask_a_simultaneous_front_collision(self):
        candidate = self.candidate()
        traffic = [{'object_info': NS(unique_id=1)}, {'object_info': NS(unique_id=2)}]
        def check(c,e,entries,matches,speed,**kw):
            if any(v['object_info'].unique_id==1 for v in entries):
                return dict(time=.2,rear=True,kind='collision',object_id=1)
            return dict(time=.4,rear=False,kind='collision',object_id=2)
        with patch.object(self.planner.lane_change_safety, 'check', side_effect=check):
            self.planner._evaluate_lane_change(candidate,self.ego,traffic,[],committed=True)
        self.assertEqual(candidate.prediction_conflict['object_id'], 2)
        self.assertEqual(self.result(candidate).acc_target_velocity, 0.)

    def test_start_from_rest_has_positive_validated_speed(self):
        self.ego.velocity = 0.
        candidate = self.candidate()
        candidate.speed_limit = 8.
        self.planner._evaluate_lane_change(candidate,self.ego,[],[])
        self.assertTrue(candidate.valid)
        self.assertEqual(candidate.change_cruise_speed, 8.)

    def test_slow_change_accelerates_on_same_path_instead_of_latching_8_kph(self):
        self.ego.velocity = 2.42672151989407  # Recorded 23:00 run: 8.7 km/h.
        candidate = self.candidate()
        self.planner._evaluate_lane_change(candidate,self.ego,[],[])
        entry = candidate.change_cruise_speed
        points = candidate.points
        self.planner._evaluate_lane_change(candidate,self.ego,[],[],committed=True)
        self.assertTrue(candidate.valid)
        self.assertGreater(candidate.change_cruise_speed, entry + 4.)
        self.assertIs(candidate.points, points)
        self.assertEqual(candidate.change_entry_speed, entry)
        faster = candidate.change_cruise_speed
        self.planner._requested_speed = 1.
        self.planner._evaluate_lane_change(candidate,self.ego,[],[],committed=True)
        self.assertEqual(candidate.change_cruise_speed, faster)

    def test_faster_change_is_rejected_when_front_traffic_blocks_acceleration(self):
        self.ego.velocity = 3.
        candidate = self.candidate()
        self.planner._evaluate_lane_change(candidate,self.ego,[],[])
        hazard = dict(time=2.,rear=False,kind='collision',object_id=11)
        with patch.object(self.planner.lane_change_safety,'check',
                          side_effect=lambda c,e,es,ms,v,**kw: hazard if v > 3. else None):
            self.planner._evaluate_lane_change(candidate,self.ego,[],[],committed=True)
        self.assertTrue(candidate.valid)
        self.assertEqual(candidate.change_cruise_speed,3.)
        self.assertEqual(candidate.acceleration_conflict,hazard)

    def test_recorded_provisional_speed_does_not_create_stationary_ghost(self):
        obj=ObjectInfo(69.0959153158,34.1435005426,10.6294536591,-1,
            unique_id=16,size_x=5.162866831,size_y=2.268596411,
            heading_deg=-89.559068072)
        obj.motion_uncertain=True
        match=self.lanes.match(obj.position,heading=math.radians(obj.heading_deg))
        predictions=self.planner.merge_safety._pose(obj,match,.8)
        point=Point(67.1990047861,41.5668199448)
        yaw=math.radians(-89.6414766961)
        self.assertTrue(self.planner._obb_collision(point,yaw,obj.position,
                                                   math.radians(obj.heading_deg),obj))
        self.assertTrue(all(not self.planner._obb_collision(point,yaw,p,a,obj)
                            for p,a in predictions))
        self.assertEqual(obj.velocity,10.6294536591)
        obj.velocity=0.
        self.assertTrue(any(self.planner._obb_collision(point,yaw,p,a,obj)
                           for p,a in self.planner.merge_safety._pose(obj,match,.8)))
        obj.velocity=30.;obj.speed_unknown=True
        self.assertTrue(any(self.planner._obb_collision(point,yaw,p,a,obj)
                           for p,a in self.planner.merge_safety._pose(obj,match,.8)))

    def test_provisional_front_acc_keeps_available_velocity(self):
        match=self.lanes.project(self.ego.position,self.source)
        pt=self.lanes.advance_default(match,20.)[0]
        obj=ObjectInfo(pt.x,pt.y,10.,-1,unique_id=5,size_x=5.,size_y=2.,
                       heading_deg=math.degrees(match.yaw))
        obj.motion_uncertain=True
        entries=[{'object_info':obj}]
        self.planner.obstacle_motion.update(entries,.03)
        self.planner._front_risk(self.ego,match,self.planner._match_objects(entries),entries)
        self.assertEqual(self.planner._front_lead_speed,10.)
        obj.speed_unknown=True
        self.planner._front_risk(self.ego,match,self.planner._match_objects(entries),entries)
        self.assertEqual(self.planner._front_lead_speed,0.)

    def test_first_merge_guard_ends_on_shared_downstream_lane(self):
        lane=self.lanes.links['A2256W000420']
        entering=self.lanes.project(lane.points[0],lane)
        self.assertEqual(self.planner.merge_safety.zone_for(entering),'FIRST_PREDICTIVE')
        passed=self.lanes.project(Point(67.2,51.),lane)
        self.assertIsNone(self.planner.merge_safety.zone_for(passed))

    def test_low_speed_change_completes_without_braking_in_bicycle_rollout(self):
        source=self.lanes.links['A2256W000420']
        target=self.lanes.links['A2256W000430']
        match=self.lanes.project(Point(67.2,50.),source)
        ego=VehicleState(match.x,match.y,match.yaw,2.42672)
        planner=self.planner
        candidate=planner._build_candidate(ego,target,self.lanes.project(ego.position,target),
                                          'left',source,55.)
        planner._evaluate_lane_change(candidate,ego,[],[])
        planner.active_candidate=candidate;planner.state=planner.CHANGE_LEFT
        planner.source_link_id=source.id;planner.target_link_id=target.id
        planner.changed_lane_hold=planner.entry_merge.completed=True
        reference=[r[0] for r in self.lanes.route_points(match,180.,.5)]
        controller=PathController(json.loads((self.root/'AIM2026/Expert/src/config/config.json').read_text()))
        completed=False
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(600):
                result=planner.plan(ego,reference,[],desired_velocity=60/3.6)
                if planner.state not in (planner.CHANGE_LEFT,planner.CHANGE_RIGHT):
                    completed=True
                    break
                acc,_,steer=controller._calculate_control(ego,result.path,result.acc_target_velocity,result)
                self.assertGreaterEqual(acc,0.)
                steer/=controller.steering_sign
                dt=1/30
                ego.velocity+=3*acc*dt
                ego.yaw+=ego.velocity/3*math.tan(steer)*dt
                ego.position=Point(ego.position.x+ego.velocity*math.cos(ego.yaw)*dt,
                                   ego.position.y+ego.velocity*math.sin(ego.yaw)*dt)
        self.assertTrue(completed)
        self.assertGreater(ego.velocity*3.6,35.)
        self.assertEqual(result.target_link_id,target.id)

    def test_final_speed_ignores_old_curve_but_keeps_signal_and_regulatory_limits(self):
        candidate = self.candidate()
        self.planner._evaluate_lane_change(candidate, self.ego, [], [])
        result = self.result(candidate)
        owner = NS(_merge_ego_curvature_speed_floor=lambda v,*a:v,
            _hold_merge_ego_curvature_speed=lambda v,*a:v,
            _second_merge_slowdown_band=lambda v,*a:v,
            _legacy_acc_entries=lambda *a:[], _smooth_static_target_speed=lambda v,*a:v,
            adaptive_cruise_control=NS(check_object=lambda *a:None,
                get_target_velocity=lambda v,t:t),
            traffic_light_stop_controller=NS(get_target_velocity=lambda **kw:float('inf')),
            path_manager=NS(current_waypoint=0,get_velocity_without_curvature=lambda:16.))
        args=(owner,self.ego,result.path,result.path,result,4.,False,float('inf'),[],[],None)
        self.assertEqual(target_speed(*args)[0],12.)
        owner.path_manager.get_velocity_without_curvature=lambda:8.
        self.assertEqual(target_speed(*args)[0],8.)
        owner.traffic_light_stop_controller.get_target_velocity=lambda **kw:0.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(target_speed(*args)[0],0.)

    def test_pid_releases_previous_brake_but_honors_new_stop(self):
        candidate=self.candidate()
        self.planner._evaluate_lane_change(candidate,self.ego,[],[])
        result=self.result(candidate)
        config=json.loads((self.root/'AIM2026/Expert/src/config/config.json').read_text())
        controller=PathController(config)
        controller.previous_acc_cmd=-.5
        controller.previous_target_velocity=12.
        self.ego.velocity=11.9
        with patch.object(controller.pid,'get_output',return_value=-.1):
            self.assertGreaterEqual(controller._calculate_control(self.ego,result.path,12.,result)[0],0.)
        self.assertEqual(controller._calculate_control(self.ego,result.path,0.,result)[0],-1.)


class LaneKeepingPredictionTest(unittest.TestCase):
    def setUp(self):
        from AIM2026.Expert.src.planning.highway.common.merge_safety import MergeSafety
        from AIM2026.Expert.src.config.parameters import highway
        self.settings = highway
        self.match = NS(x=10., y=3.5, yaw=0., lane_link=NS(road_id='4000104'))
        self.obj = NS(position=Point(10.,3.5), velocity=10., heading_deg=-20., size_x=4.5, size_y=2.)
        lanes = NS(_angle_difference=lambda a,b: math.atan2(math.sin(a-b),math.cos(a-b)),
                   advance_options=lambda match,d: [(Point(match.x+d,match.y),0.,match.lane_link)])
        self.safety = MergeSafety(NS(lane_map=lanes))

    def test_adjacent_vehicle_heading_noise_does_not_create_lateral_entry(self):
        with patch.object(self.settings,'ASSUME_HIGHWAY_VEHICLES_KEEP_LANE',True):
            point,_=self.safety._pose(self.obj,self.match,2.)[0]
            self.assertAlmostEqual(point.y,3.5)
            self.assertAlmostEqual(point.x,30.)
        with patch.object(self.settings,'ASSUME_HIGHWAY_VEHICLES_KEEP_LANE',False):
            point,_=self.safety._pose(self.obj,self.match,2.)[0]
            self.assertLess(point.y,1.)

    def test_actual_overlap_is_not_erased_and_other_roads_keep_cut_in_prediction(self):
        from AIM2026.Expert.src.planning.highway.common.collision_check import CollisionPredictionMixin
        self.obj.position=Point(10.,.5)
        with patch.object(self.settings,'ASSUME_HIGHWAY_VEHICLES_KEEP_LANE',True):
            point,yaw=self.safety._pose(self.obj,self.match,0.)[0]
            self.assertEqual(point.y,.5)
            self.assertTrue(CollisionPredictionMixin._obb_collision(Point(8.,0.),0.,point,yaw,self.obj))
            self.obj.position=Point(10.,3.5)
            self.match.lane_link.road_id='other_road'
            point,_=self.safety._pose(self.obj,self.match,2.)[0]
            self.assertLess(point.y,1.)

    def test_mapped_convergence_is_still_predicted(self):
        self.safety.lane_map.advance_options=lambda match,d: [(Point(10.+d,0.),0.,match.lane_link)]
        with patch.object(self.settings,'ASSUME_HIGHWAY_VEHICLES_KEEP_LANE',True):
            point,_=self.safety._pose(self.obj,self.match,2.)[0]
            self.assertEqual(point.y,0.)


if __name__=='__main__': unittest.main()
