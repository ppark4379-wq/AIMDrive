"""Second-merge-only curvature bypass; traffic, legal and terminal limits stay."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from expert.src.autonomous_driving import AutonomousDriving
from expert.src.localization.path_manager import PathManager
from expert.src.localization.point import Point
from expert.src.planning.highway.structures import HighwayPlanningResult
from expert.src.vehicle_state import VehicleState


class HighwayPathSpeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        map_dir = Path(__file__).resolve().parents[3] / 'map_data/R_KR_PG_KATRI_2025'
        xy = np.loadtxt(str(map_dir / '2026_molit_comp_global_path.txt'))
        cls.route = [Point(*p[:2]) for p in xy]

    def setUp(self):
        self.manager = PathManager(self.route, False, 50)
        self.manager.set_velocity_profile(100., .15, 15)

    def execute(self, ego, planner_limit=float('inf'), signal_limit=float('inf'), acc_limit=float('inf')):
        driver = AutonomousDriving.__new__(AutonomousDriving)
        driver.path_manager = self.manager
        driver.path = self.route
        driver._merge_curvature_floor_zone = driver._merge_curvature_floor_speed = None
        driver._configure_runtime = lambda **kwargs: None
        driver.forward_object_detector = SimpleNamespace(detect_object=lambda *args, **kwargs: [])
        speeds = {}
        def plan(vehicle, path, entries, desired_velocity, **kwargs):
            speeds['prediction'] = desired_velocity
            return HighwayPlanningResult(path, acc_target_velocity=planner_limit)
        driver.lattice_planning_process = SimpleNamespace(run=plan,
            roundabout_planner=SimpleNamespace(active=False),
            _roundabout_entry_state=lambda ego: {'in_approach': False})
        driver.adaptive_cruise_control = SimpleNamespace(
            check_object=lambda *args: None, pedestrian_released=False,
            get_target_velocity=lambda current, planned: min(acc_limit, planned))
        driver.traffic_light_stop_controller = SimpleNamespace(
            get_target_velocity=lambda **kwargs: min(signal_limit, kwargs['planned_velocity']))
        def control(vehicle, path, target, result, **kwargs):
            speeds['control'] = target
            return 0., 0., 0.
        driver._calculate_control = control
        driver.execute(ego, [], None)
        return speeds

    def test_logged_low_profile_is_bypassed_before_prediction_and_control(self):
        for x, y in ((75.22, -196.50), (75.27, -204.99), (75.35, -212.84)):
            with self.subTest(y=y):
                ego = VehicleState(x, y, -math.pi/2, 10.)
                _, original = self.manager.get_local_path(ego)
                self.assertAlmostEqual(7.2937, original, places=3)
                speeds = self.execute(ego)
                self.assertAlmostEqual(100./3.6, speeds['prediction'])
                self.assertEqual(speeds['prediction'], speeds['control'])

    def test_bypass_is_limited_to_second_merge_and_travel_direction(self):
        for x, y, yaw, expected in ((75.,-180.,-math.pi/2,True),
                (75.,-212.,-math.pi/2,True), (75.,-220.,-math.pi/2,False),
                (71.,-115.,-math.pi/2,False), (66.,150.,-math.pi/2,False),
                (110.,-180.,-math.pi/2,False), (75.,-180.,math.pi/2,False)):
            with self.subTest(x=x,y=y,yaw=yaw):
                self.assertEqual(expected, AutonomousDriving._second_merge_curvature_bypass(
                    VehicleState(x,y,yaw,10.)))

    def test_curvature_profile_resumes_after_merge(self):
        ego = VehicleState(75.46,-220.16,math.radians(-90.78),10.)
        _, original = self.manager.get_local_path(ego)
        self.assertAlmostEqual(9.008, original, places=3)
        speeds = self.execute(ego)
        self.assertEqual(original, speeds['prediction'])
        self.assertEqual(original, speeds['control'])

    def test_front_acc_merge_stop_and_signal_still_override(self):
        ego = VehicleState(75.27,-204.99,-math.pi/2,8.)
        for limits, expected in (({'planner_limit':0.},0.), ({'signal_limit':0.},0.),
                                 ({'acc_limit':4.},4.), ({'planner_limit':5.},5.)):
            with self.subTest(limits=limits):
                speeds = self.execute(ego, **limits)
                self.assertGreater(speeds['prediction'],7.29)
                self.assertEqual(expected,speeds['control'])

    def test_existing_approach_speed_band_is_used_in_prediction_too(self):
        speeds = self.execute(VehicleState(73.,-150.,-math.pi/2,10.))
        self.assertEqual(45./3.6, speeds['prediction'])
        self.assertEqual(speeds['prediction'], speeds['control'])

    def test_legal_speed_zone_and_terminal_stop_are_not_bypassed(self):
        path = [Point(i*.5,0.) for i in range(401)]
        manager = PathManager(path,False,50)
        manager.set_velocity_profile(100.,.15,15,[dict(name='test_zone',
            start_point=(50.,0.),end_point=(90.,0.),max_velocity=20.)])
        manager.get_local_path(VehicleState(60.,0.,0.,10.))
        self.assertEqual(20./3.6,manager.get_velocity_without_curvature())
        manager._last_waypoint_idx = None
        manager.get_local_path(VehicleState(199.,0.,0.,2.))
        self.assertEqual(2.,manager.get_velocity_without_curvature())


if __name__ == '__main__':
    unittest.main()
