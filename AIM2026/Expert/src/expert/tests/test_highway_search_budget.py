"""Optional trajectory search cannot stall asynchronous obstacle braking."""
import math
import random
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import test_highway_static_merge_escape as fixture
from expert.src.localization.point import Point
from expert.src.control.path_controller import PathController
from expert.src.planning.highway.planner import HighwayLanePlanner
from expert.src.planning.highway import parameters
from expert.src.vehicle_state import VehicleState
from expert.src.obstacle.object_info import ObjectInfo


class HighwaySearchBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StaticMergeEscapeTest.setUpClass()

    def scene(self):
        scene = fixture.StaticMergeEscapeTest()
        scene.setUp()
        scene.ego = VehicleState(62.19,247.94,math.radians(-89.8),11.4)
        scene.blocker = ObjectInfo(62.5684,186.498,0.,2,is_static=True,
                                  size_x=4.69,size_y=2.2,heading_deg=-89.64,unique_id=13)
        scene.planner.search_time_budget = .08
        return scene

    def test_timeout_rejects_unverified_changes_and_does_not_accelerate_toward_static(self):
        scene = self.scene()
        calls = iter([0., 1.])
        with patch('expert.src.planning.highway.planner.time.monotonic',
                   side_effect=lambda: next(calls,1.)):
            result = scene.plan()
        self.assertTrue(result.debug_details['lane_search_budget_exhausted'])
        self.assertEqual('ACC_FOLLOW', result.highway_state)
        self.assertEqual('A2256W000411', result.target_link_id)
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertLessEqual(result.acc_target_velocity,scene.ego.velocity)
        self.assertFalse(any(c.valid for c in result.candidates if c.direction!='keep'))

    def test_normal_safe_escape_still_starts_within_budget(self):
        scene = self.scene()
        with patch('expert.src.planning.highway.planner.time.monotonic',return_value=0.):
            result = scene.plan()
        self.assertEqual('CHANGE_LEFT', result.highway_state)
        self.assertTrue(scene.planner.active_candidate.lane_change_verified)
        self.assertFalse(result.debug_details['lane_search_budget_exhausted'])

    def test_exhausted_search_brakes_before_static_with_real_controller(self):
        import contextlib
        import io
        scene = self.scene()
        controller = PathController(scene.config)
        dt = .05
        scene.planner.set_control_dt(dt)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(240):
                # Force optional search exhaustion directly; a two-value clock
                # can be consumed by another timer before the search begins.
                with patch.object(scene.planner, 'search_timed_out', return_value=True):
                    result = scene.plan()
                ego = scene.ego
                controller._configure_runtime(sync_mode=False, control_dt=dt)
                pedal, _, steer = controller._calculate_control(
                    ego,result.path,min(27.78,result.acc_target_velocity),result)
                speed = max(0.,ego.velocity+(3. if pedal>=0 else 5.)*pedal*dt)
                scene.ego = VehicleState(ego.position.x+speed*math.cos(ego.yaw)*dt,
                    ego.position.y+speed*math.sin(ego.yaw)*dt,
                    ego.yaw+speed/3.*math.tan(steer/controller.steering_sign)*dt,speed)
                self.assertFalse(scene.planner._obb_collision(scene.ego.position,scene.ego.yaw,
                    scene.blocker.position,math.radians(scene.blocker.heading_deg),scene.blocker))
        self.assertLess(scene.ego.velocity,.3)

    def test_runtime_budget_is_disabled_when_simulator_ticks_are_synchronous(self):
        scene = self.scene()
        controller = PathController(scene.config)
        controller.lattice_planning_process = SimpleNamespace(highway_lane_planner=scene.planner)
        controller._configure_runtime(sync_mode=False,control_dt=.033)
        self.assertEqual(parameters.LANE_CHANGE_SEARCH_TIME_BUDGET,scene.planner.search_time_budget)
        controller._configure_runtime(sync_mode=True,control_dt=.033)
        self.assertIsNone(scene.planner.search_time_budget)

    def test_scalar_geometry_matches_original_distance_and_curvature(self):
        rng = random.Random(2409)
        for _ in range(50):
            points=[Point(rng.uniform(-100,100),rng.uniform(-100,100)) for _ in range(12)]
            points.insert(5,points[4])
            cumulative=[0.]
            for a,b in zip(points,points[1:]):cumulative.append(cumulative[-1]+a.distance(b))
            actual=HighwayLanePlanner._cumulative_distance(points)
            for a,b in zip(cumulative,actual):self.assertAlmostEqual(a,b,places=9)
            maximum=0.
            for a,b,c in zip(points,points[1:],points[2:]):
                denominator=a.distance(b)*b.distance(c)*a.distance(c)
                if denominator>=1e-9:
                    area=abs((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x))
                    maximum=max(maximum,2.*area/denominator)
            self.assertAlmostEqual(maximum,HighwayLanePlanner._max_curvature(points),places=12)

if __name__=='__main__':unittest.main()
