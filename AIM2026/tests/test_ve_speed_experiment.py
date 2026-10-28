"""Isolated tests for the removable raw-VE-speed experiment."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from AIM2026.Expert.ve_speed_experiment import use_ve_speed
from AIM2026.Expert.runtime import Inputs
from AIM2026.Expert.run_highway import ExpertController


class VeSpeedExperimentTest(unittest.TestCase):
    def box(self, **changes):
        return dict(dict(track_id=3, center_world=[66., 250.], motion_state='moving',
                         planning_heading_deg=-90., speed_mps=43./3.6,
                         planning_speed_mps=19.8/3.6, planning_speed_source='position_history'), **changes)

    def test_raw_speed_preserves_tracking_and_input_packet(self):
        packet = {'boxes': [self.box()]}
        result = use_ve_speed(packet)['boxes'][0]
        self.assertAlmostEqual(result['planning_speed_mps']*3.6, 43.)
        self.assertAlmostEqual(result['tracked_planning_speed_mps']*3.6, 19.8)
        for field in ('track_id', 'center_world', 'motion_state', 'planning_heading_deg'):
            self.assertEqual(result[field], packet['boxes'][0][field])
        self.assertEqual(packet['boxes'][0]['planning_speed_source'], 'position_history')

    def test_stationary_and_invalid_raw_keep_existing_speed(self):
        boxes = [self.box(motion_state='stationary', planning_speed_mps=0.)]
        boxes += [self.box(speed_mps=value) for value in (None, -1., float('nan'), float('inf'), True, '12')]
        result = use_ve_speed({'boxes': boxes})
        for old, new in zip(boxes, result['boxes']):
            self.assertEqual(new['planning_speed_mps'], old['planning_speed_mps'])
            self.assertEqual(new['planning_speed_source'], old['planning_speed_source'])

    def test_uncertain_raw_is_used_without_changing_confidence_label(self):
        result = use_ve_speed({'boxes': [self.box(motion_state='uncertain')]})['boxes'][0]
        self.assertEqual(result['motion_state'], 'uncertain')
        self.assertEqual(result['planning_speed_mps'], result['speed_mps'])

    def test_runtime_defaults_to_tracker_and_option_runs_after_processing(self):
        row = dict(generation='test', checkpoint_sha256='sha', sequence=1, boxes=[])
        original = {'boxes': [self.box()]}
        for enabled in (False, True):
            tracker = Mock(update=Mock(return_value=original))
            bridge = Inputs('test', 'sha', tracker, dedup=False, ve_object_speed=enabled)
            with patch('AIM2026.Expert.runtime.decode'):
                self.assertEqual(bridge.accept(json.dumps(row)), 'detection')
            expected = 43. if enabled else 19.8
            self.assertAlmostEqual(bridge.packet['boxes'][0]['planning_speed_mps']*3.6, expected)
        self.assertEqual(original['boxes'][0]['planning_speed_source'], 'position_history')

    def test_launcher_forwards_option_only_when_requested(self):
        controller = object.__new__(ExpertController)
        controller.scenario = SimpleNamespace(args=SimpleNamespace(
            expert_python='python', config='config', speed_cap_kmh=None), driver_output='output')
        controller.child_path = str
        controller.context_path, controller.stop_file = 'context', 'stop'
        self.assertNotIn('--ve-object-speed', controller.command())
        controller.scenario.args.ve_object_speed = True
        self.assertIn('--ve-object-speed', controller.command())
        self.assertIn('--ve-object-speed', controller.command(check=True))


if __name__ == '__main__':
    unittest.main()
