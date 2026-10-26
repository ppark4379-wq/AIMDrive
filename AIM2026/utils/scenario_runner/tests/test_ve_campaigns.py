import importlib.util
import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'tools/ve_full_route_campaign.py'
spec = importlib.util.spec_from_file_location('campaign', path)
route_campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route_campaign)


class FullRouteCampaignTests(unittest.TestCase):
    def test_success_requires_goal_and_clean_exit(self):
        self.assertEqual(route_campaign.classify(0, 'GOAL REACHED'), 'success')
        self.assertNotEqual(route_campaign.classify(1, 'GOAL REACHED'), 'success')
        self.assertNotEqual(route_campaign.classify(0, 'setup complete'), 'success')
        self.assertNotEqual(route_campaign.classify(0, 'GOAL REACHED', 'wall_timeout'), 'success')

    def test_failure_categories(self):
        self.assertEqual(route_campaign.classify(1, 'collision during scenario verification'), 'collision')
        self.assertEqual(route_campaign.classify(1, 'FATAL_STALL'), 'stalled_180s')
        self.assertEqual(route_campaign.classify(1, 'AIMDrive expert running.\nRuntimeError: x'), 'runtime_error')
        self.assertEqual(route_campaign.classify(1, 'VE process exit'), 'startup_error')

    def test_disabled_timeout_banner_does_not_hide_input_failure(self):
        text = 'AIMDrive expert running. timeout=disabled\nRuntimeError: VE/자차 입력이 2초 이상 끊겼습니다'
        self.assertEqual(route_campaign.classify(1, text), 'perception_input_stale')
        self.assertEqual(route_campaign.classify(1, 'AIMDrive expert running. timeout=disabled\nRuntimeError: other'), 'runtime_error')
        self.assertEqual(route_campaign.classify(1, 'RuntimeError: Dynamic obstacle crossing timeout: id=x'), 'timeout')

    def test_fresh_random_campaigns_and_reproducible_logged_seed(self):
        with patch.object(route_campaign.secrets, 'randbits', side_effect=[101, 202]):
            root_a, seeds_a = route_campaign.trial_seeds(30)
            root_b, seeds_b = route_campaign.trial_seeds(30)
        self.assertNotEqual(root_a, root_b)
        self.assertNotEqual(seeds_a, seeds_b)
        self.assertEqual(len(set(seeds_a)), 30)
        self.assertEqual((root_a, seeds_a), route_campaign.trial_seeds(30, root_a))
        self.assertTrue(all(0 < seed < 2**31 for seed in seeds_a))

    def test_default_goal_is_before_tunnel_with_explicit_full_loop_option(self):
        args = route_campaign.parse_arguments(['--output', '/tmp/not-created'])
        self.assertTrue(args.stop_before_tunnel)
        self.assertIsNone(args.seed)
        self.assertFalse(route_campaign.parse_arguments(['--output', '/tmp/not-created', '--include-tunnel']).stop_before_tunnel)

    def test_aim_backend_path_and_trial_command_preserve_common_mission_options(self):
        args=route_campaign.parse_arguments(['--output','/tmp/not-created','--include-tunnel',
            '--gps-blackout-hold','--aim2026-config','AIM2026/config.yaml',
            '--mission','full_loop_tunnel_normal','--speed-cap-kmh','40'])
        self.assertEqual(args.aim2026_config,route_campaign.REPO/'AIM2026/config.yaml')
        command=route_campaign.trial_command(args,123,Path('/tmp/ego.csv'))
        self.assertEqual(command[command.index('--aim2026-config')+1],str(args.aim2026_config))
        self.assertEqual(command[command.index('--speed-cap-kmh')+1],'40.0')
        self.assertEqual(command[command.index('--mission')+1],'full_loop_tunnel_normal')
        self.assertIn('--gps-blackout-hold',command)
        self.assertNotIn('--simulate-gps-blackout',command)

    def test_aim_backend_incompatible_flags_rejected_before_execution(self):
        base=['--output','/tmp/not-created','--include-tunnel','--aim2026-config','AIM2026/config.yaml']
        for extra in ([],['--gps-blackout-hold','--simulate-gps-blackout'],
                      ['--gps-blackout-hold','--gps-latency-s','.1']):
            with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):route_campaign.parse_arguments(base+extra)

    def test_aim_backend_missing_or_malformed_config_rejected(self):
        base=['--output','/tmp/not-created','--include-tunnel','--gps-blackout-hold']
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'configuration with spaces.yaml'
            for value in ('Localization: []','Localization:\n  enabled: "true"'):
                file.write_text(value)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    route_campaign.parse_arguments(base+['--aim2026-config',str(file)])
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                route_campaign.parse_arguments(base+['--aim2026-config',str(file.parent/'missing.yaml')])

    def test_aim_backend_does_not_require_dr_when_localization_explicitly_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'disabled.yaml';file.write_text('Localization:\n  enabled: false')
            args=route_campaign.parse_arguments(['--output','/tmp/not-created','--aim2026-config',str(file)])
            self.assertFalse(args.gps_blackout_hold)
            self.assertTrue(args.stop_before_tunnel)

    def test_stationary_jitter_and_real_movement(self):
        watch = route_campaign.MovementWatch(180)
        self.assertFalse(watch.stalled(500))
        watch.update(10, 20, 0)
        watch.update(10.1, 20.1, 179)
        self.assertFalse(watch.stalled(179))
        self.assertTrue(watch.stalled(180))
        watch.update(13, 20, 181)
        self.assertFalse(watch.stalled(181))
        watch.update(float('nan'), 20, 350)
        self.assertTrue(watch.stalled(361))

    def test_static_stuck_needs_stopped_ego_and_recent_static_trouble(self):
        watch = route_campaign.StaticStuckWatch(30)
        watch.update(0, 0, 0)
        self.assertFalse(watch.stuck(40))                       # stopped (e.g. red light), no static trouble
        watch.console('[PLAN] reason=static_recorded_follow state=OFF', 10)
        self.assertFalse(watch.stuck(40))                       # following a path is not trouble
        watch.console('[PLAN] reason=static_recorded_blocked state=OFF', 35)
        self.assertTrue(watch.stuck(40))
        self.assertFalse(watch.stuck(66))                       # trouble not seen for 30 s
        watch.update(5, 0, 41)                                  # moved 5 m: counter restarts
        self.assertFalse(watch.stuck(60))

    def test_static_trouble_patterns(self):
        for line in ('[PLAN] reason=static_recorded_planning state=OFF', '[PLAN] reason=static_recorded_blocked state=OFF',
                     '[STATIC_FAILURE] reason=static_recorded_blocked error=remaining_path_collision'):
            self.assertTrue(route_campaign.STATIC_TROUBLE.search(line), line)
        for line in ('[PLAN] reason=static_recorded_follow', '[PLAN] reason=traffic_light_stop', '[STATIC] {phase: planning}'):
            self.assertFalse(route_campaign.STATIC_TROUBLE.search(line), line)

    def test_static_stuck_option_defaults_to_30_seconds(self):
        self.assertEqual(route_campaign.parse_arguments(['--output', '/tmp/x']).static_stuck_seconds, 30.)
        self.assertEqual(route_campaign.parse_arguments(['--output', '/tmp/x', '--static-stuck-seconds', '0']).static_stuck_seconds, 0.)

    def test_driver_log_location_is_read_from_trial_console(self):
        line = '[GPS+VE 주행 로그] /x/ve_gps_full_route/20261017_194601_620229/driver.console.log'
        self.assertEqual(route_campaign.DRIVER_LOG.search(line).group(1), '/x/ve_gps_full_route/20261017_194601_620229')

    def test_ignore_signals_option_is_off_by_default(self):
        self.assertEqual(route_campaign.parse_arguments(['--output', '/tmp/x']).ignore_signals, '')
        self.assertEqual(route_campaign.parse_arguments(['--output', '/tmp/x', '--ignore-signals', 'C1,C2']).ignore_signals, 'C1,C2')

    def test_signal_hold_after_static_trouble_is_not_static_stuck(self):
        watch = route_campaign.StaticStuckWatch(30)
        watch.update(0, 0, 0)
        watch.console('[STATIC_REPLAN] committed path blocked for 1.0s while stopped', 5)
        red = "[STATIC] {'phase': 'follow', 'speed_limits': {'legacy_acc': 14.2, 'signal': 0.0, 'static': 7.8}, 'x': 1}"
        for t in range(6, 40):
            watch.console(red, t)
        self.assertFalse(watch.stuck(40))                       # stopped for a red light, not the static planner
        green = "[STATIC] {'phase': 'blocked', 'speed_limits': {'legacy_acc': 14.2, 'signal': 14.2, 'static': 0.0}, 'x': 1}"
        watch.console(green, 41)
        watch.console('[PLAN] reason=static_recorded_blocked state=OFF', 41)
        self.assertFalse(watch.stuck(60))
        self.assertTrue(watch.stuck(70))                        # 30 s blocked after the last signal hold

    def test_speed_cap_defaults_to_60_and_is_validated(self):
        self.assertEqual(route_campaign.parse_arguments(['--output', '/tmp/x']).speed_cap_kmh, 60.)
        self.assertEqual(route_campaign.parse_arguments(['--output', '/tmp/x', '--speed-cap-kmh', '90']).speed_cap_kmh, 90.)
        with self.assertRaises(SystemExit):
            route_campaign.parse_arguments(['--output', '/tmp/x', '--speed-cap-kmh', '0'])

    def test_sensor_only_is_rejected_as_unsupported(self):
        valid = ['--output', '/tmp/not-created', '--sensor-only', '--mission', 'full_loop_tunnel_normal']
        # Not part of this port: it needs VLDrive sensor-audit modules that main does not have.
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            route_campaign.parse_arguments(valid)


# Bounded overnight runs: lifecycle, failure persistence and no fake PASS.
import json
from unittest.mock import Mock, patch

from MoraiDrive.scenario_runner.legacy_expert.run_ve_hz_check import campaign as hz_campaign, findings, wait_background_start


class HzCampaignTests(unittest.TestCase):
    def run_campaign(self, root, failure=None, interrupted=False):
        sessions = []
        clock = [100.]
        def factory(*args, **kwargs):
            self.assertFalse(kwargs['observe_gt'])
            session = Mock(error=None, rejected=0)
            session.sender.poll.return_value = None
            session.performance_line.return_value = 'test Hz'
            session.snapshot.return_value = dict(ve_fresh=True)
            if failure:
                session.await_ready.side_effect = failure
            sessions.append(session)
            return session
        def report(folder):
            path = folder/'timing_report.json'
            path.write_text(json.dumps(dict(accepted_outputs=10,jpeg_warning_occurrences=2,
                input_selection=dict(lidar=dict(reused_selected_sample=0)))))
            return path
        def sleep(seconds):
            if interrupted:
                raise KeyboardInterrupt()
            clock[0] += seconds
        with patch('MoraiDrive.scenario_runner.legacy_expert.run_ve_hz_check.time.monotonic',side_effect=lambda:clock[0]), \
             patch('MoraiDrive.scenario_runner.legacy_expert.run_ve_hz_check.time.sleep',side_effect=sleep), \
             patch('MoraiDrive.scenario_runner.legacy_expert.run_ve_hz_check.process_rss_mib',return_value=123.):
            code = hz_campaign(root,'E3',4.,2,factory,Mock(),lambda:False,report)
        return code, sessions, json.loads((root/'campaign_report.json').read_text())

    def test_success_is_measuring_complete_not_a_semantic_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,sessions,report = self.run_campaign(Path(tmp))
        self.assertEqual(code,0)
        self.assertEqual(report['status'],'COMPLETE')
        self.assertEqual(report['completed_segments'],2)
        self.assertEqual(report['segments_with_issues'],2)
        self.assertEqual(report['control_commands_sent'],0)
        self.assertFalse(report['performance_passed_automatically'])
        for session in sessions:
            session.close.assert_called_once()

    def test_failed_start_stops_repetition_preserves_report_and_closes_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,sessions,report = self.run_campaign(Path(tmp),RuntimeError('camera missing'))
        self.assertEqual(code,1)
        self.assertEqual(len(sessions),1)
        sessions[0].close.assert_called_once()
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['reason'],'camera missing')
        self.assertEqual(report['completed_segments'],0)

    def test_interruption_closes_model_and_keeps_partial_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,sessions,report = self.run_campaign(Path(tmp),interrupted=True)
        self.assertEqual(code,130)
        sessions[0].close.assert_called_once()
        self.assertEqual(report['status'],'INTERRUPTED')
        self.assertFalse(report['segments'][0]['duration_completed'])

    def test_correct_reassembly_still_reports_remaining_fragment_losses(self):
        report = dict(accepted_outputs=100,jpeg_warning_occurrences=0,
            lidar_gate_counter_differs_from_selected_sample=0,
            sensors=dict(rgb_front=dict(final_counters=dict(counters=dict(
                frames_started=120,end_markers=110,discarded_torn_images=10)))),
            input_selection=dict(lidar=dict(reused_selected_sample=0)))
        self.assertEqual(findings(report),dict(rgb_front=dict(discarded_torn_images=10)))

    def test_background_success_requires_fresh_input_and_failures_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            child=Mock()
            child.poll.return_value=None
            states=[dict(status='RUNNING',phase='model_and_sensors_starting'),
                    dict(status='RUNNING',phase='measuring',input_fresh=True,performance='9Hz')]
            (root/'status.json').write_text(json.dumps(states[0]))
            with patch('MoraiDrive.scenario_runner.legacy_expert.run_ve_hz_check.time.sleep',
                       side_effect=lambda _: (root/'status.json').write_text(json.dumps(states[1]))) as sleep:
                self.assertEqual(wait_background_start(root,child),0)
                sleep.assert_called_once()
            (root/'status.json').write_text(json.dumps(dict(status='FAILED',reason='missing lidar')))
            self.assertEqual(wait_background_start(root,child),1)


if __name__ == "__main__":
    unittest.main()
