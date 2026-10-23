"""Shared controller works with expert planner events and standalone VLDrive."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.control.path_controller import PathController
from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState


class SharedPathControllerTest(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'src/config/config.json').read_text())
        self.path=[Point(float(i),0.) for i in range(1,21)]
        self.ego=VehicleState(0.,0.,0.,0.)
        self.context=SimpleNamespace(mode='model_waypoints',active=False,reason='model_waypoints',
                                     debug_details={},highway_state='OFF')

    def test_vldrive_does_not_require_an_acc_object(self):
        core=PathController(self.config)
        self.assertFalse(hasattr(core,'adaptive_cruise_control'))
        command=core.control_path(self.path,self.ego,5/3.6,self.context)
        self.assertGreater(command.accel,0.)
        self.assertEqual(command.brake,0.)
        self.assertEqual(core.pid.p_gain,.075)

    def test_pedestrian_release_is_explicit_and_preserves_default_brake_ramp(self):
        outputs=[]
        for released in (False,True):
            core=PathController(self.config)
            core.previous_target_velocity=0.
            core.previous_acc_cmd=-1.
            outputs.append(core._calculate_control(self.ego,self.path,2.,self.context,
                                                   pedestrian_released=released)[0])
        self.assertLess(outputs[0],0.)
        self.assertGreater(outputs[1],0.)

    def test_expert_uses_the_shared_control_and_updated_helpers(self):
        for method in ('_calculate_control','_configure_runtime','_recorded_lookahead_limits',
                       '_roundabout_entry_acceleration_gain'):
            self.assertIs(getattr(AutonomousDriving,method),getattr(PathController,method))
        core=PathController(self.config)
        context=SimpleNamespace(active=True,reason='static_recorded_follow',debug_details={
            'static_avoidance':{'planner':'recorded_route_graph','preview_curvature':.02}})
        limits=core._recorded_lookahead_limits(SimpleNamespace(velocity=10.),context)
        self.assertAlmostEqual(limits[0],6.5)


if __name__=='__main__':unittest.main()
