"""Pre-drive self/anonymous contact records are journaled and exempted; every new contact still stops the run."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..core.startup_contact import StartupContactBaseline, StartupContactExemption

try:
    from MoraiDrive.scenario_runner.tools import run_ve_gps_full_route as route_runner
    from scenario_runner.core.full_loop_drive_base import FullLoopDriveBase
except ImportError:                  # the launcher needs Linux-only modules; these checks run under WSL/Linux
    route_runner = None


class FakeDriveBase:
    """Stands in for the drive-loop scenario: replays scripted Ego states."""
    def __init__(self, states):
        self.states = list(states)
        self.grpc = SimpleNamespace(world=None)

    def get_ego_motion_state(self):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return dict(x=0.0, y=0.0, collision_objects=list(state))


class Scenario(StartupContactExemption, FakeDriveBase):
    contact_label = 'Test'

    def __init__(self, states, output, strict=False):
        FakeDriveBase.__init__(self, states)
        self.output = Path(output)
        self.args = SimpleNamespace(strict_collisions=strict)


class StartupContactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def begin(self, scenario):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            scenario.begin_startup_contact_baseline()
        return printed.getvalue()

    def test_a_self_record_before_the_drive_is_exempted_and_journaled(self):
        scenario = Scenario([['Ego'], ['Ego'], ['Ego']], self.output)
        printed = self.begin(scenario)
        self.assertIn('INTERACTIVE', printed)
        self.assertIn('not a collision-free guarantee', printed)
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], [])
        events = [json.loads(line) for line in (self.output / 'contact_events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['raw_collision_objects'], ['Ego'])
        self.assertEqual(events[-1]['startup_contact_ignored'], ['Ego'])
        before = json.loads((self.output / 'before_placement.json').read_text())
        self.assertEqual(before['collision_objects'], ['Ego'])

    def test_a_new_contact_next_to_the_self_record_still_stops_the_run(self):
        scenario = Scenario([['Ego'], ['Ego'], ['Ego', '1']], self.output)
        self.begin(scenario)
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], [])
        state = scenario.get_ego_motion_state()
        self.assertEqual(state['collision_objects'], ['1'])
        self.assertEqual(state['contact_classification']['1'], 'new_report')

    def test_a_cleared_self_record_that_comes_back_is_a_new_contact(self):
        scenario = Scenario([['Ego'], ['Ego'], [], ['Ego']], self.output)
        self.begin(scenario)
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], [])
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], [])
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], ['Ego'])

    def test_only_a_self_record_gets_an_exemption(self):
        scenario = Scenario([['7'], ['7']], self.output)
        self.begin(scenario)
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], ['7'])

    def test_strict_mode_records_the_same_evidence_but_exempts_nothing(self):
        scenario = Scenario([['Ego'], ['Ego']], self.output, strict=True)
        printed = self.begin(scenario)
        self.assertNotIn('INTERACTIVE', printed)
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], ['Ego'])
        self.assertTrue((self.output / 'contact_events.jsonl').is_file())

    def test_no_report_before_the_drive_means_no_exemption_to_apply(self):
        scenario = Scenario([[], [], ['Ego']], self.output)
        self.begin(scenario)
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], [])
        self.assertEqual(scenario.get_ego_motion_state()['collision_objects'], ['Ego'])

    def test_a_removed_numeric_actor_label_is_exempt_only_while_the_actor_stays_gone(self):
        baseline = StartupContactBaseline({'collision_objects': ['Ego', '19']}, live_actor_ids={'Ego'})
        self.assertEqual(baseline.apply({'collision_objects': ['Ego', '19']}, {'Ego'})['collision_objects'], [])
        # the id is now live again (reused): the exemption is revoked, so it counts as a new contact
        self.assertEqual(baseline.apply({'collision_objects': ['Ego', '19']}, {'Ego', '19'})['collision_objects'], ['19'])


@unittest.skipIf(route_runner is None, 'GPS launcher needs Linux modules')
class GpsLauncherStartupContactTests(unittest.TestCase):
    def scenario(self, scenario_class, strict=False):
        scenario = object.__new__(scenario_class)
        scenario.gps_args = SimpleNamespace(strict_collisions=strict)
        scenario.gps_output = Path('run-output')
        return scenario

    def test_both_gps_scenario_classes_use_the_exemption(self):
        for scenario_class in (route_runner.GpsMission, route_runner.GpsFullLoop):
            self.assertTrue(issubclass(scenario_class, StartupContactExemption))
        scenario = self.scenario(route_runner.GpsMission, strict=True)
        self.assertEqual(scenario._contact_output_dir(), Path('run-output'))
        self.assertTrue(scenario._strict_contact_mode())
        self.assertFalse(self.scenario(route_runner.GpsMission)._strict_contact_mode())

    def test_baseline_is_taken_before_the_drive_setup_moves_the_ego(self):
        calls = []
        scenario = self.scenario(route_runner.GpsFullLoop)
        with patch.object(StartupContactExemption, 'begin_startup_contact_baseline',
                          side_effect=lambda: calls.append('baseline')), \
             patch.object(FullLoopDriveBase, 'configure_drive_from_start',
                          side_effect=lambda: calls.append('configure') or True):
            self.assertTrue(scenario.configure_drive_from_start())
        self.assertEqual(calls, ['baseline', 'configure'])


if __name__ == '__main__':
    unittest.main()
