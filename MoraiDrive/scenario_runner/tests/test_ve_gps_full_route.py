"""Scenario lifecycle must never switch the GPS driver back to simulator GT."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from MoraiDrive.scenario_runner.tools.run_ve_gps_full_route import configure_scenario, GpsController
from run_ve_gps_expert import prepare


class GpsFullRouteTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(speed_cap_kmh=60., seed=123, stop_before_tunnel=False,
            static_preset=None, ego_trace=None, sim_profile='normal', signal_head=Path('/unused/head.pt'),
            bind_ip='0.0.0.0',sim_ip='127.0.0.1',status_port=8000,gps_port=10506,imu_port=10507,
            control_port=7601,control_source_port=7600,
            mission='',tunnel_dynamic_speed=None,tunnel_candidate_id=None,tunnel_layout=None,gt_expert=False,
            tunnel_vehicle_behavior=None,tunnel_vehicle_random=False,
            moving_carry_s=0.,slow_window_s=0.,stationary_hold_s=0.)

    def test_previous_ros_config_cannot_start_parallel_ve_or_gt_inputs(self):
        base = dict(ve_objects_checkpoint='E3',ve_signal_head='old',ve_full_route_signals=True,
            expert_ros_remaps=['/Object_topic:=/Object_topic'], sync_runtime={'enabled':True},
            highway_merge_static_vehicle={'enabled':True},traffic_light_wait_override={'enabled':True},
            dynamic_obstacle={'pre_signal_guard':{'enabled':True}},roundabout={'enabled':True})
        original = copy.deepcopy(base)
        cfg = configure_scenario(base, self.args())
        self.assertEqual(base, original)
        self.assertNotIn('ve_objects_checkpoint',cfg)
        self.assertFalse(cfg['sync_runtime']['enabled'])
        self.assertFalse(cfg['highway_merge_static_vehicle']['enabled'])
        self.assertFalse(cfg['traffic_light_wait_override']['enabled'])
        self.assertTrue(cfg['dynamic_obstacle']['pre_signal_guard']['enabled'])
        self.assertTrue(cfg['roundabout']['enabled'])
        self.assertEqual(cfg['expert_ros_remaps'], [])
        self.assertTrue(cfg['static_verified_pair'])
        self.assertEqual(cfg['max_laps'],1)

    def test_driver_uses_same_route_snapshot_and_no_second_teleport_or_gt_ports(self):
        with tempfile.TemporaryDirectory() as d:
            context = SimpleNamespace(gps_output=Path(d),gps_args=self.args(),
                global_cfg={'paths':{}},cfg={'ve_stop_before_tunnel':True},route_points=[(1.,2.,3.),(4.,5.,6.)])
            controller = GpsController(Mock(),context)
            command = controller.command()
            self.assertIn('--drive',command)
            for disallowed in ('--from-start','--object-port','--signal-port','--signal-source'):
                self.assertNotIn(disallowed,command)
            self.assertIn('run_ve_gps_expert.py', ' '.join(command))
            saved = json.loads(controller.context_path.read_text())
            self.assertEqual(saved['route_points'], [[1.,2.,3.],[4.,5.,6.]])
            self.assertTrue(saved['scenario_config']['ve_stop_before_tunnel'])
            self.assertNotIn('--drive',controller.command(check=True))

    def test_child_refuses_a_route_different_from_the_scene_before_planner_setup(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args()
            args.scenario_context = Path(d)/'context.json'
            args.scenario_context.write_text(json.dumps(dict(schema='ve_gps_scenario_context_v1',
                global_config={'paths':{'mgeo_root':'unused'}},scenario_config={},route_points=[[9.,9.,9.]])))
            context = Mock(route_points=[(1.,2.,3.)])
            with patch('scenario_runner.scenarios.full_loop_scenario.FullLoopScenario',return_value=context), \
                 patch('scenario_runner.utils.map_loader.MGeoMapLoader'):
                with self.assertRaisesRegex(ValueError,'경로가 다릅니다'):
                    prepare(args,Path(d))
            context.setup_expert_controller.assert_not_called()

    def test_stop_terminates_actual_driver_not_unused_ros_controller(self):
        context = SimpleNamespace(process=Mock(),log=Mock())
        process, log = context.process, context.log
        with patch('MoraiDrive.scenario_runner.tools.run_ve_gps_full_route.stop_child') as stop:
            GpsController.stop_process(context)
        stop.assert_called_once_with(process)
        log.close.assert_called_once()
        self.assertIsNone(context.process)


if __name__ == '__main__':
    unittest.main()
