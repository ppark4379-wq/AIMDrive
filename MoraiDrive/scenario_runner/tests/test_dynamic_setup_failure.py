import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from scenario_runner.scenarios.full_loop_parts import dynamic_obstacle as dynamic
from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
from scenario_runner.core.full_loop_drive_base import FullLoopDriveBase


class DynamicSetupFailureTests(unittest.TestCase):
    def test_stopline_guard_failure_keeps_cleanup_metrics_initialized(self):
        context = object.__new__(FullLoopScenario)
        context.cfg = {}
        context.route_definition = {'zones': {'dynamic_obstacle': {}}}
        context.traffic_light_stops = []
        context.dynamic_obstacle_event_index = 0
        with patch.object(dynamic, '_strict_assessment'), \
             patch.object(dynamic, '_dynamic_cfg', return_value={}), \
             patch.object(dynamic, '_catalogue_entries', return_value=[]), \
             patch.object(dynamic, '_expand_catalogue', return_value=[]), \
             patch.object(dynamic, '_choose_schedule', return_value=[{'id':'event'}]), \
             patch.object(dynamic, '_signal_guard_cfg', return_value={'enabled':True}):
            with self.assertRaisesRegex(RuntimeError,'no verified traffic-light stop'):
                dynamic.spawn_dynamic_obstacle(context)
        self.assertFalse(context.dynamic_validation['goal_reached'])
        with patch.object(FullLoopDriveBase, 'cleanup'), \
             patch.object(FullLoopScenario, 'cleanup_spawned_actors'), \
             patch.object(dynamic, '_persist_progress'):
            context.cleanup()
        self.assertTrue(context.dynamic_validation['cleanup_verified'])
        self.assertFalse(context.dynamic_validation['safety_validation_passed'])
