"""Synthetic traffic anchored at reported poses; NPC trajectories were not recorded."""
import json
import math
from pathlib import Path
import unittest
from expert.src.control.path_controller import PathController
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.planning.highway_lane_planning.structures import HighwayPlanningResult


class LoggedHighwayStopsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.map = HighwayLaneMap(str(root.parents[1] / 'map_data/R_KR_PG_KATRI_2025'))
        cls.config = json.loads((root/'src/config/config.json').read_text())

    def scene(self, second=True):
        planner = HighwayLanePlanner(lane_map=self.map)
        ego = (VehicleState(68.32,-152.70,math.radians(-86.57),0.) if second else
               VehicleState(66.52,194.87,math.radians(-95.32),0.))
        link = self.map.links['A2256W000445' if second else 'A2256W000409']
        match = self.map.project(ego.position,link)
        path = [pose[0] for pose in self.map.route_points(match,140.,.5)]
        return planner,ego,match,path

    def test_stopped_prediction_cannot_rotate_34_degrees(self):
        planner,ego,match,path = self.scene()
        points,station = planner.merge_safety._route(ego,match,path,12.5)
        _,old_heading = planner._pose_at_distance(points,station,0.)
        self.assertGreater(abs(math.degrees(old_heading-ego.yaw)),30.)
        for target in (0.,.5,12.5):
            for t in (0.,.1,.2):
                distance = planner.merge_safety.travel(0.,target,t)
                point,yaw = planner.merge_safety.ego_pose(ego,points,station,distance)
                self.assertEqual(0.,ego.position.distance(point))
                self.assertEqual(ego.yaw,yaw)
        _,yaw = planner.merge_safety.ego_pose(ego,points,station,1e-6)
        self.assertLess(abs(yaw-ego.yaw),1e-6)

    def test_can_go_while_parallel_neighbor_is_still_present(self):
        planner,ego,match,path = self.scene()
        obj = ObjectInfo(ego.position.x+3.,ego.position.y-3.,11.11,1,
            size_x=4.5,size_y=2.,heading_deg=-90.,unique_id=25)
        self.assertFalse(planner._obb_collision(ego.position,ego.yaw,obj.position,-math.pi/2,obj))
        target = planner.merge_safety.limit(ego,match,[],[{'object_info':obj}],path,desired_velocity=12.5)
        self.assertEqual(12.5,target)
        self.assertEqual('GO',planner.merge_safety.details['decision'])

    def test_first_merge_passed_source_obstacle_does_not_lock_stop(self):
        planner,ego,match,path = self.scene(False)
        obj = ObjectInfo(62.2,186.,0.,2,is_static=True,
            size_x=4.69,size_y=2.2,heading_deg=-90.,unique_id=13)
        target = planner.merge_safety.limit(ego,match,[],[{'object_info':obj}],path,desired_velocity=26.62)
        self.assertEqual(26.62,target,planner.merge_safety.details)

    def test_actual_body_overlap_still_stops_in_both_merges(self):
        for second in (False,True):
            planner,ego,match,path = self.scene(second)
            obj = ObjectInfo(ego.position.x,ego.position.y-2.,0.,1,
                size_x=4.5,size_y=2.,heading_deg=-90.,unique_id=31)
            target = planner.merge_safety.limit(ego,match,[],[{'object_info':obj}],path,desired_velocity=12.5)
            self.assertEqual(0.,target)
            self.assertEqual('obb',planner.merge_safety.details['conflict']['kind'])

    def test_real_cut_in_is_not_ignored_as_neighbor(self):
        planner,ego,match,path = self.scene()
        obj = ObjectInfo(ego.position.x+4.,ego.position.y-2.,6.,1,
            size_x=4.5,size_y=2.,heading_deg=-140.,unique_id=31)
        target = planner.merge_safety.limit(ego,match,[],[{'object_info':obj}],path,desired_velocity=12.5)
        self.assertLess(target,12.5)
        self.assertIsNotNone(planner.merge_safety.details['conflict'])

    def test_matched_cut_in_heading_also_predicts_lateral_motion(self):
        planner,ego,_,_ = self.scene(False)
        lane = self.map.links['A2256W000410']
        point,yaw,_ = self.map.point_at_s(lane,60.)
        obj = ObjectInfo(point.x,point.y,10.,1,size_x=4.5,size_y=2.,
                         heading_deg=math.degrees(yaw)-25.,unique_id=31)
        match = self.map.project(point,lane)
        predicted,_ = planner.merge_safety._pose(obj,match,1.)[0]
        nominal,_,_ = self.map.advance_default(match,10.)
        self.assertLess(predicted.x,nominal.x-1.)

    def test_normal_lane_change_waits_for_heading_alignment(self):
        planner,_,_,_ = self.scene(False)
        start = VehicleState(62.2,256.,-math.pi/2,14.26)
        target = self.map.links['A2256W000409']
        candidate = planner._build_candidate(start,target,self.map.project(start.position,target),
            'left',self.map.links['A2256W000411'],transition_distance=55.)
        candidate.static_escape=True
        candidate.compact_static=False
        candidate.speed_limit=float('inf')
        candidate.maneuver_speed_limit=14.26
        planner.active_candidate=candidate
        planner.state=planner.CHANGE_LEFT
        planner.source_link_id='A2256W000411'
        planner.target_link_id=target.id
        ego=VehicleState(65.74,216.47,math.radians(-82.19),17.51)
        reference=[p[0] for p in self.map.route_points(self.map.project(ego.position,target),100.,.5)]
        result=planner.plan(ego,reference,[],desired_velocity=27.78)
        self.assertEqual('CHANGE_LEFT',result.highway_state)
        self.assertIsNotNone(planner.active_candidate)

    def test_lane_change_holds_entry_speed_instead_of_100_kmh(self):
        planner,_,_,_ = self.scene(False)
        ego=VehicleState(62.2,245.,-math.pi/2,14.26)
        target=self.map.links['A2256W000409']
        candidate=planner._build_candidate(ego,target,self.map.project(ego.position,target),
            'left',self.map.links['A2256W000411'],transition_distance=55.)
        candidate.static_escape=True
        planner._requested_speed=27.78
        planner._evaluate_lane_change(candidate,ego,[],[])
        self.assertTrue(candidate.valid)
        self.assertEqual(14.26,candidate.safety_speed)
        planner._evaluate_lane_change(candidate,ego,[],[],committed=True)
        self.assertEqual(14.26,candidate.safety_speed)

    def test_acc_transition_does_not_kick_full_brake_but_emergency_does(self):
        controller=PathController(self.config)
        controller.previous_target_velocity=27.78
        controller.previous_acc_cmd=.805
        ego=VehicleState(0.,0.,0.,17.51)
        path=[Point(float(i),0.) for i in range(101)]
        result=HighwayPlanningResult(path,highway_state='ACC_FOLLOW',acc_target_velocity=15.76)
        command,_,_=controller._calculate_control(ego,path,15.76,result)
        self.assertGreater(command,-.5)
        self.assertLess(command,0.)
        command,_,_=controller._calculate_control(ego,path,0.,result)
        self.assertEqual(-1.,command)


if __name__=='__main__':
    unittest.main()
