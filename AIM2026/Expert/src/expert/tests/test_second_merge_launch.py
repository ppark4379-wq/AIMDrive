"""Fast second-merge launch must clear rear traffic and obey lower speed limits."""
import contextlib
import io
import json
import math
from pathlib import Path
import unittest

from expert.src.control.path_controller import PathController
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway.lane_map import HighwayLaneMap
from expert.src.planning.highway.planner import HighwayLanePlanner
from expert.src.planning.highway.structures import HighwayPlanningResult
from expert.src.vehicle_state import VehicleState


class SecondMergeLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.map = HighwayLaneMap(str(root.parents[1] / 'map_data/R_KR_PG_KATRI_2025'))
        cls.config = json.loads((root / 'src/config/config.json').read_text())

    def scene(self, rear_distance=50., rear_speed=13.89):
        planner = HighwayLanePlanner(lane_map=self.map)
        ego = VehicleState(68.44, -155.47, math.radians(-86.1), 0.)
        match = self.map.project(ego.position, self.map.links['A2256W000445'])
        path = [row[0] for row in self.map.route_points(match, 140., .5)]
        rear_match = self.map.project(Point(71.7, ego.position.y + rear_distance),
                                      self.map.links['A2256W000434'])
        pos, yaw, _ = self.map.point_at_s(rear_match.lane_link, rear_match.s)
        obj = ObjectInfo(pos.x, pos.y, rear_speed, 1, size_x=4.5, size_y=2.,
                         heading_deg=math.degrees(yaw), unique_id=31)
        entry = dict(object_info=obj, local_position=(pos-ego.position).rotate(-ego.yaw))
        return planner, ego, match, path, entry, rear_match

    def decide(self, distance=50., speed=13.89):
        planner, ego, match, path, entry, rear_match = self.scene(distance, speed)
        target = planner.merge_safety.limit(ego, match, [(entry, rear_match)], [entry],
                                           path, desired_velocity=12.5)
        return planner, ego, path, target

    def test_far_rear_allows_ego_to_launch_first_instead_of_waiting(self):
        planner, _, _, target = self.decide()
        self.assertEqual(12.5, target)
        details = planner.merge_safety.details
        self.assertTrue(details['requested_conflict']['rear'])
        self.assertTrue(details['launch_verified'])
        self.assertEqual('GO', details['decision'])
        self.assertIsNone(details['launch_conflict'])

    def test_close_or_fast_rear_still_blocks_fast_launch(self):
        for distance, speed in ((40., 13.89), (50., 25.)):
            planner, _, _, target = self.decide(distance, speed)
            self.assertFalse(planner.merge_safety.details.get('launch_verified', False))
            self.assertLess(target, 12.5)

    def test_actual_front_body_overlap_cancels_launch(self):
        planner, ego, match, path, entry, rear_match = self.scene()
        planner.merge_safety.second_launch_active = True
        obj = ObjectInfo(ego.position.x, ego.position.y-2., 0., 1,
                         size_x=4.5, size_y=2., heading_deg=-90., unique_id=25)
        target = planner.merge_safety.limit(ego, match, [], [dict(object_info=obj)],
                                           path, desired_velocity=12.5)
        self.assertEqual(0., target)
        self.assertFalse(planner.merge_safety.second_launch_active)
        self.assertFalse(planner.merge_safety.details.get('launch_verified', False))

    def test_stop_release_immediately_uses_full_available_acceleration(self):
        planner, ego, path, target = self.decide()
        result = HighwayPlanningResult(path, highway_state='ACC_FOLLOW',
            acc_target_velocity=target, debug_details={'merge_safety':planner.merge_safety.details})
        controller = PathController(self.config)
        controller.previous_target_velocity = 0.
        controller.previous_acc_cmd = -1.
        pedal, _, _ = controller._calculate_control(ego, path, target, result)
        self.assertEqual(1., pedal)
        pedal, _, _ = controller._calculate_control(ego, path, 0., result)
        self.assertEqual(-1., pedal)

    def test_boost_is_not_applied_to_first_merge_or_lower_signal_acc_target(self):
        planner, ego, path, target = self.decide()
        result = HighwayPlanningResult(path, debug_details={'merge_safety':planner.merge_safety.details})
        gain = PathController._second_merge_launch_acceleration_gain
        self.assertIsNotNone(gain(target, ego, result))
        self.assertIsNone(gain(3., ego, result))
        result.debug_details['merge_safety']['zone'] = 'FIRST_PREDICTIVE'
        self.assertIsNone(gain(target, ego, result))

    def test_real_controller_clears_rear_without_contact_or_repeated_stop(self):
        planner, ego, _, path, entry, origin = self.scene()
        controller = PathController(self.config)
        controller.previous_target_velocity = 0.
        controller.previous_acc_cmd = -1.
        dt = .05
        planner.set_control_dt(dt)
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(120):
                obj = entry['object_info']
                pos, yaw, _ = self.map.advance_default(origin, 13.89*step*dt)
                obj.position, obj.heading_deg = pos, math.degrees(yaw)
                entry['local_position'] = (pos-ego.position).rotate(-ego.yaw)
                result = planner.plan(ego, path, [entry], desired_velocity=12.5, allow_lane_change=False)
                target = min(12.5, result.acc_target_velocity)
                self.assertGreater(target, .05, (step, result.debug_details))
                controller._configure_runtime(sync_mode=False, control_dt=dt)
                pedal, _, steer = controller._calculate_control(ego, result.path, target, result)
                speed = max(0., ego.velocity + (3. if pedal >= 0. else 5.)*pedal*dt)
                yaw = ego.yaw
                ego = VehicleState(ego.position.x+speed*math.cos(yaw)*dt,
                    ego.position.y+speed*math.sin(yaw)*dt,
                    yaw+speed/3.*math.tan(steer/controller.steering_sign)*dt, speed)
                self.assertFalse(planner._obb_collision(ego.position, ego.yaw, pos,
                                  math.radians(obj.heading_deg), obj), step)
                if step == 59:
                    self.assertGreater(speed, 7.)
        self.assertGreater(ego.velocity, 10.)


    def test_logged_neighbor_gap_does_not_interrupt_second_merge_launch(self):
        # Exact predicted poses from the report; length is recoverable from the
        # logged bumper gap. Width is a conservative synthetic 2.16 m because
        # the log did not contain the measured width.
        cases = [
            ((69.18496667646068, -158.44218713324855), -84.53357700313765,
             (71.69688179139857, -155.1937642472353), -89.65202711607162),
            ((68.74276535534239, -157.33261093421345), -80.24203744376831,
             (71.808285863032, -167.28330845003055), -88.77815756847194),
        ]
        for xy, heading, other_xy, other_heading in cases:
            planner = HighwayLanePlanner(lane_map=self.map)
            ego = VehicleState(*xy, math.radians(heading), 0.)
            obj = ObjectInfo(*other_xy, 13.89, 1, size_x=4.7, size_y=2.16,
                             heading_deg=other_heading, unique_id=25)
            tracks = [('npc:25', [obj, None, 0.])]
            predictions = [[[(obj.position, math.radians(other_heading))]]]
            points = [ego.position, Point(ego.position.x, ego.position.y-100.)]
            planner.merge_safety.zone = 'FIRST_PREDICTIVE'
            hazard = planner.merge_safety._conflict(ego, points, [0.,100.], 0., tracks, predictions)
            self.assertEqual('gap', hazard['kind'])
            planner.merge_safety.zone = 'SECOND_ENTRY'
            self.assertIsNone(planner.merge_safety._conflict(
                ego, points, [0.,100.], 0., tracks, predictions))

    def test_real_same_lane_bumper_gap_is_still_enforced(self):
        planner = HighwayLanePlanner(lane_map=self.map)
        ego = VehicleState(0., 0., 0., 0.)
        obj = ObjectInfo(8., 0., 0., 1, size_x=4.7, size_y=2.16, heading_deg=0.)
        planner.merge_safety.zone = 'SECOND_ENTRY'
        self.assertFalse(planner._obb_collision(ego.position, ego.yaw, obj.position, 0., obj))
        hazard = planner.merge_safety._conflict(ego, [ego.position, Point(100.,0.)],
            [0.,100.], 0., [('front',[obj,None,0.])], [[[(obj.position,0.)]]])
        self.assertEqual('gap', hazard['kind'])
        self.assertFalse(hazard['rear'])

    def stopped_report_scene(self, speed=.25):
        planner = HighwayLanePlanner(lane_map=self.map)
        ego = VehicleState(68.68, -157.63, math.radians(-80.8), speed)
        match = self.map.project(ego.position, self.map.links['A2256W000445'])
        path = [row[0] for row in self.map.route_points(match, 140., .5)]
        # Approximate current rear position reconstructed from the logged
        # five-second prediction. NPC speed/width are synthetic assumptions.
        origins = [self.map.project(Point(71.7, y), self.map.links[link])
                   for y,link in ((-109.1,'A2256W000434'),(-173.8,'A2256W000422'))]
        entries = []
        for identifier, origin in zip((31,25), origins):
            pos,yaw,_ = self.map.point_at_s(origin.lane_link, origin.s)
            obj = ObjectInfo(pos.x,pos.y,13.89,1,size_x=4.7,size_y=2.16,
                             heading_deg=math.degrees(yaw),unique_id=identifier)
            entries.append(dict(object_info=obj,local_position=(pos-ego.position).rotate(-ego.yaw)))
        return planner,ego,match,path,entries,origins

    def test_reported_stopped_pose_can_launch_before_far_rear(self):
        planner,ego,match,path,entries,origins = self.stopped_report_scene()
        target = planner.merge_safety.limit(ego,match,list(zip(entries,origins)),entries,path,
                                           desired_velocity=12.5)
        self.assertEqual(12.5,target,planner.merge_safety.details)
        self.assertTrue(planner.merge_safety.details['launch_verified'])

    def test_brief_stop_above_old_start_threshold_can_revalidate_boost(self):
        planner,ego,match,path,entries,origins = self.stopped_report_scene(speed=1.94)
        planner.merge_safety.second_launch_active = False
        target = planner.merge_safety.limit(ego,match,list(zip(entries,origins)),entries,path,
                                           desired_velocity=12.5)
        self.assertEqual(12.5,target,planner.merge_safety.details)
        self.assertTrue(planner.merge_safety.details['launch_verified'])

    def test_reported_pose_launch_with_front_and_rear_controller_rollout(self):
        for plant_acceleration in (2.5,3.):
            with self.subTest(plant_acceleration=plant_acceleration):
                planner,ego,_,path,entries,origins = self.stopped_report_scene()
                controller = PathController(self.config)
                controller.previous_target_velocity = 0.
                controller.previous_acc_cmd = -1.
                dt = .05
                planner.set_control_dt(dt)
                with contextlib.redirect_stdout(io.StringIO()):
                    for step in range(140):
                        for entry,origin in zip(entries,origins):
                            pos,yaw,_ = self.map.advance_default(origin,13.89*step*dt)
                            obj = entry['object_info']
                            obj.position,obj.heading_deg = pos,math.degrees(yaw)
                            entry['local_position'] = (pos-ego.position).rotate(-ego.yaw)
                        result = planner.plan(ego,path,entries,desired_velocity=12.5,allow_lane_change=False)
                        target = min(12.5,result.acc_target_velocity)
                        self.assertGreater(target,.05,(step,result.debug_details))
                        controller._configure_runtime(sync_mode=False,control_dt=dt)
                        pedal,_,steer = controller._calculate_control(ego,result.path,target,result)
                        speed = max(0.,ego.velocity+(plant_acceleration if pedal>=0. else 5.)*pedal*dt)
                        yaw = ego.yaw
                        ego = VehicleState(ego.position.x+speed*math.cos(yaw)*dt,
                            ego.position.y+speed*math.sin(yaw)*dt,
                            yaw+speed/3.*math.tan(steer/controller.steering_sign)*dt,speed)
                        for entry,origin in zip(entries,origins):
                            obj = entry['object_info']
                            pos,angle,_ = self.map.advance_default(origin,13.89*(step+1)*dt)
                            self.assertFalse(planner._obb_collision(ego.position,ego.yaw,pos,angle,obj),
                                             (step,obj.unique_id))
                self.assertGreater(ego.velocity,10.)



    def test_lower_front_limit_must_not_authorize_fast_launch_prediction(self):
        planner,ego,match,path,entries,origins = self.stopped_report_scene()
        planner.merge_safety.limit(ego,match,list(zip(entries,origins)),entries,path,
                                  desired_velocity=12.5,forward_speed_limit=3.)
        self.assertFalse(planner.merge_safety.details.get('launch_verified',False))

    def test_physical_follower_requires_completed_alignment_and_rear_clearance(self):
        planner,ego,match,path,entries,origins = self.stopped_report_scene()
        safety = planner.merge_safety
        safety.zone = 'SECOND_ENTRY'
        obj = entries[0]['object_info']
        self.assertFalse(safety._second_merge_follower(ego,match,obj,origins[0]))
        lane = self.map.links['A2256W000422']
        aligned = self.map.project(Point(72.5,-185.),lane)
        pos,yaw,_ = self.map.point_at_s(lane,aligned.s)
        from expert.src.planning.highway import parameters as p
        ego = VehicleState(pos.x-.5*p.VEHICLE_LENGTH*math.cos(yaw),
                           pos.y-.5*p.VEHICLE_LENGTH*math.sin(yaw),yaw,10.)
        match = self.map.project(ego.position,self.map.links['A2256W000445'])
        pos,heading,_ = self.map.point_at_s(lane,aligned.s-15.)
        obj.position,obj.heading_deg = pos,math.degrees(heading)
        other_match = self.map.project(pos,lane)
        self.assertTrue(safety._second_merge_follower(ego,match,obj,other_match))
        obj.heading_deg -= 25.
        self.assertFalse(safety._second_merge_follower(ego,match,obj,other_match))
        pos,heading,_ = self.map.point_at_s(lane,aligned.s-5.)
        obj.position,obj.heading_deg = pos,math.degrees(heading)
        self.assertFalse(safety._second_merge_follower(ego,match,obj,self.map.project(pos,lane)))
        safety.zone = 'FIRST_PREDICTIVE'
        pos,heading,_ = self.map.point_at_s(lane,aligned.s-15.)
        obj.position,obj.heading_deg = pos,math.degrees(heading)
        self.assertFalse(safety._second_merge_follower(ego,match,obj,self.map.project(pos,lane)))


if __name__ == '__main__':
    unittest.main()
