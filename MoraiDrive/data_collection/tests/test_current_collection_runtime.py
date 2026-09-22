"""Current-only collection selection and preservation of historical evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / 'data_collection/internal_sensor_record'
sys.path[:0] = [str(PROJECT / 'data_collection'), str(PROJECT), str(INTERNAL)]
from async_capture.journal import digest
from async_capture.sensor_record_admission import (
    _verify_producer_snapshot, verify_sealed_directory,
)

RUNNER = INTERNAL / 'tools/run_proven_three_lap_scenario.py'


class CurrentCollectionRuntimeTests(unittest.TestCase):
    def invoke(self, command, env=None):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment.update(env or {})
        return subprocess.run(command, env=environment, cwd=PROJECT,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=45)

    def test_all_nine_scenarios_use_current_expert_without_frozen_tree(self):
        self.assertFalse((INTERNAL / 'proven_three_lap_runtime').exists())
        for scenario in ('full_loop', 'full_loop_normal', 'full_loop_static',
                         'full_loop_dynamic', 'full_loop_roundabout',
                         'full_loop_roundabout_nonuniform',
                         'full_loop_roundabout_speed_random',
                         'full_loop_roundabout_brake',
                         'full_loop_highway'):
            with self.subTest(scenario=scenario):
                result = self.invoke([
                    sys.executable, '-B', str(RUNNER), '--check',
                    '--scenario', scenario, '--nearby-actors',
                    '--time-mode', 'simulation_free_run', '--max-speed-kmh', '60',
                    '--collision-policy', 'record', '--raw-camera-transport', 'bridge_dropped',
                ], {'MORAI_EXPERT_ROOT': '/unused/frozen/expert'})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('expert_mode=current; expert_root=' + str(PROJECT / 'expert'), result.stdout)

    def test_removed_mode_is_rejected_by_both_python_entrypoints(self):
        for script, args in (
            (RUNNER, ['--check', '--expert-runtime', 'proven']),
            (INTERNAL / 'tools/run_sensor_record_scenario.py',
             ['--name', 'unused', '--pid', '1', '--scenario-runtime', 'proven']),
        ):
            with self.subTest(script=script.name):
                result = self.invoke([sys.executable, '-B', str(script)] + args)
                self.assertEqual(result.returncode, 2)
                self.assertIn('invalid choice', result.stderr)
                self.assertIn('current', result.stderr)

    def test_removed_campaign_mode_fails_before_machine_preparation(self):
        result = self.invoke(['bash', str(INTERNAL / 'tools/collect_scenario_laps.sh'), '1'], {
            'MORAI_SCENARIO_RUNTIME': 'proven',
            'MORAI_PROJECT_ROOT': '/unused/project',
        })
        self.assertEqual(result.returncode, 2)
        self.assertIn('proven mode has been removed', result.stderr)
        self.assertNotIn('[PREPARE]', result.stdout)

    def test_explicit_current_remains_compatible(self):
        command = [sys.executable, '-B', str(RUNNER), '--check']
        default = self.invoke(command)
        explicit = self.invoke(command + ['--expert-runtime', 'current'])
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(default.stdout, explicit.stdout)

    def test_current_root_accepts_updated_config_without_old_hash(self):
        spec = importlib.util.spec_from_file_location('current_runtime_test', RUNNER)
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', list(sys.path)), patch.dict(os.environ):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            config = project / 'expert/src/expert/src/config/config.json'
            config.parent.mkdir(parents=True)
            with patch.object(module, 'PROJECT_ROOT', project):
                for content in ('{"version": 1}', '{"version": 2}'):
                    config.write_text(content)
                    self.assertEqual(module.require_expert_root('current'), project / 'expert')
                with self.assertRaises(ValueError):
                    module.require_expert_root('proven')

    def test_prepared_native_output_requires_exact_empty_owned_layout(self):
        spec = importlib.util.spec_from_file_location('prepared_output_test', RUNNER)
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', list(sys.path)), patch.dict(os.environ):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'campaign_lap_001'
            output.mkdir()
            for role in module.PREPARED_NATIVE_ROLES:
                (output / role).mkdir()
            (output / 'supervisor.log').write_text('')
            (output / 'SESSION.json').write_text(json.dumps({
                'applied': True,
                'plan': {'session': output.name},
            }))

            module.require_prepared_output_root(output)
            (output / 'plan.json').write_text('{}')
            with self.assertRaises(ValueError):
                module.require_prepared_output_root(output)
            (output / 'plan.json').unlink()
            (output / 'CAMERA_1' / 'early.png').write_bytes(b'x')
            with self.assertRaises(ValueError):
                module.require_prepared_output_root(output)

    def test_outer_capture_passes_prepared_output_only_for_direct_layout(self):
        source = (
            INTERNAL / 'tools/run_sensor_record_scenario.py'
        ).read_text()
        self.assertIn("if direct:\n        command.append('--prepared-output-root')", source)

    def test_historical_snapshot_survives_removal_but_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'old_proven_lap'
            snapshot_root = run / 'software_snapshot'
            source = snapshot_root / 'tools/run_proven_three_lap_scenario.py'
            source.parent.mkdir(parents=True)
            source.write_text('# archived code before mode removal\n')
            producer = {str(source.relative_to(snapshot_root)): digest(source)}
            snapshot = {'root': '/previous/computer/software_snapshot', 'files': producer}
            plan = run / 'plan.json'
            plan.write_text(json.dumps({'arguments': {'expert_runtime': 'proven'},
                'proven_runtime_manifest': {'path': '/removed/PROVEN_RUNTIME_MANIFEST.json', 'sha256': 'historical'}}))
            (run / 'COMPLETE.json').write_text(json.dumps({'files': {'plan.json': digest(plan)}}))
            self.assertTrue(_verify_producer_snapshot(snapshot, producer, transferred_run=run)['valid'])
            verify_sealed_directory(run)
            source.write_text('# changed archived code\n')
            self.assertFalse(_verify_producer_snapshot(snapshot, producer, transferred_run=run)['valid'])
            plan.write_text('{}')
            with self.assertRaises(ValueError):
                verify_sealed_directory(run)


if __name__ == '__main__':
    unittest.main()
