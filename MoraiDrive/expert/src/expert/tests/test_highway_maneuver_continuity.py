"""Regression for the exhausted 420->408 lane-change path in the driving log."""
import contextlib
import io
import json
import math
from pathlib import Path
import unittest

from expert.src.control.path_controller import PathController
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.vehicle_state import VehicleState


class ManeuverContinuityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.map = HighwayLaneMap(str(root.parents[1]/'map_data/R_KR_PG_KATRI_2025'))
        cls.config = json.loads((root/'src/config/config.json').read_text())

    def scene(self):
        planner = HighwayLanePlanner(lane_map=self.map)
        source,target = (self.map.links['A2256W000'+n] for n in ('430','420'))
        start = self.map.project(Point(70.9,14.),source)
        ego = VehicleState(start.x,start.y,start.yaw,13.89)
        planner._requested_speed = 22.8
        candidate = planner._build_candidate(ego,target,self.map.project(ego.position,target),
                                             'right',source,transition_distance=55.)
        planner.active_candidate = candidate
        planner.state = planner.CHANGE_RIGHT
        planner.target_link_id,planner.source_link_id = target.id,source.id
        return planner,ego,candidate

    def reference(self, ego):
        match = self.map.match(ego.position,heading=ego.yaw)
        return [row[0] for row in self.map.route_points(match,160.,.5)]

    def test_cross_section_does_not_replace_target_with_source_successor(self):
        for x in (70.5,71.5):
            planner,_,candidate = self.scene()
            ego = VehicleState(x,2.,-math.pi/2,13.89)
            result = planner.plan(ego,self.reference(ego),[],desired_velocity=22.8)
            self.assertEqual('A2256W000408',result.target_link_id)
            self.assertEqual('A2256W000408',candidate.target_link_id)
            self.assertEqual('CHANGE_RIGHT',result.highway_state)
            self.assertGreater(result.acc_target_velocity,1.)

    def test_logged_stale_target_and_two_point_path_recovers_completed_change(self):
        planner,_,candidate = self.scene()
        planner.target_link_id = 'A2256W000434'
        candidate.points = [Point(67.60,-70.07),Point(67.61,-70.57)]
        candidate.cost = float('inf')
        ego = VehicleState(67.51,-70.45,math.radians(-89.71),.02)
        result = planner.plan(ego,self.reference(ego),[],desired_velocity=22.79)
        self.assertEqual('A2256W000408',result.target_link_id)
        self.assertNotEqual('CHANGE_RIGHT',result.highway_state)
        self.assertIsNone(planner.active_candidate)
        self.assertGreater(len(result.path),20)
        self.assertGreater(result.acc_target_velocity,1.)
        self.assertTrue(any((point-ego.position).rotate(-ego.yaw).x > 10. for point in result.path))

    def test_unfinished_change_replenishes_tail_instead_of_creeping_at_endpoint(self):
        planner,_,candidate = self.scene()
        ego = VehicleState(68.4,-70.45,math.radians(-89.71),.26)
        candidate.points = [Point(67.60,-70.07),Point(67.61,-70.57)]
        result = planner.plan(ego,self.reference(ego),[],desired_velocity=22.79)
        self.assertEqual('CHANGE_RIGHT',result.highway_state)
        self.assertEqual('A2256W000408',result.target_link_id)
        self.assertGreater(len(result.path),20)
        self.assertGreater(result.acc_target_velocity,1.)
        self.assertIsNone(result.debug_details.get('lane_change_conflict'))

    def test_replenished_tail_is_checked_for_real_obstacles(self):
        planner,_,candidate = self.scene()
        ego = VehicleState(68.4,-70.45,math.radians(-89.71),10.)
        candidate.points = [Point(67.60,-70.07),Point(67.61,-70.57)]
        match = self.map.project(Point(67.6,-85.),self.map.links['A2256W000408'])
        obj = ObjectInfo(match.x,match.y,0.,2,is_static=True,size_x=4.7,size_y=2.1,
                         heading_deg=math.degrees(match.yaw),unique_id=77)
        entry = dict(object_info=obj,local_position=(obj.position-ego.position).rotate(-ego.yaw))
        result = planner.plan(ego,self.reference(ego),[entry],desired_velocity=22.79)
        self.assertLess(result.acc_target_velocity,10.)
        self.assertGreater(len(result.path),20)

    def test_real_controller_completes_boundary_change_and_keeps_driving(self):
        planner,ego,candidate = self.scene()
        controller = PathController(self.config)
        dt = .05
        planner.set_control_dt(dt)
        completed = False
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(160):
                result = planner.plan(ego,self.reference(ego),[],desired_velocity=13.89)
                if planner.active_candidate is None:
                    completed = True
                self.assertGreater(len(result.path),10,(step,result.reason))
                self.assertGreater(result.acc_target_velocity,1.,(step,result.debug_details))
                self.assertTrue(any((point-ego.position).rotate(-ego.yaw).x > 5. for point in result.path),step)
                controller._configure_runtime(sync_mode=False,control_dt=dt)
                pedal,_,steer = controller._calculate_control(ego,result.path,min(13.89,result.acc_target_velocity),result)
                speed = max(0.,ego.velocity+(3. if pedal>=0 else 5.)*pedal*dt)
                yaw = ego.yaw
                ego = VehicleState(ego.position.x+speed*math.cos(yaw)*dt,
                    ego.position.y+speed*math.sin(yaw)*dt,
                    yaw+speed/3.*math.tan(steer/controller.steering_sign)*dt,speed)
        self.assertTrue(completed)
        self.assertLess(ego.position.y,-80.)
        self.assertGreater(ego.velocity,10.)


if __name__ == '__main__':
    unittest.main()
