import importlib.util
import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'tools/ve_full_route_campaign.py'
spec = importlib.util.spec_from_file_location('campaign', path)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_success_requires_goal_and_clean_exit(self):
        self.assertEqual(campaign.classify(0, 'GOAL REACHED'), 'success')
        self.assertNotEqual(campaign.classify(1, 'GOAL REACHED'), 'success')
        self.assertNotEqual(campaign.classify(0, 'setup complete'), 'success')
        self.assertNotEqual(campaign.classify(0, 'GOAL REACHED', 'wall_timeout'), 'success')

    def test_failure_categories(self):
        self.assertEqual(campaign.classify(1, 'collision during scenario verification'), 'collision')
        self.assertEqual(campaign.classify(1, 'FATAL_STALL'), 'stalled_180s')
        self.assertEqual(campaign.classify(1, 'AIMDrive expert running.\nRuntimeError: x'), 'runtime_error')
        self.assertEqual(campaign.classify(1, 'VE process exit'), 'startup_error')

    def test_disabled_timeout_banner_does_not_hide_input_failure(self):
        text = 'AIMDrive expert running. timeout=disabled\nRuntimeError: VE/자차 입력이 2초 이상 끊겼습니다'
        self.assertEqual(campaign.classify(1, text), 'perception_input_stale')
        self.assertEqual(campaign.classify(1, 'AIMDrive expert running. timeout=disabled\nRuntimeError: other'), 'runtime_error')
        self.assertEqual(campaign.classify(1, 'RuntimeError: Dynamic obstacle crossing timeout: id=x'), 'timeout')

    def test_fresh_random_campaigns_and_reproducible_logged_seed(self):
        with patch.object(campaign.secrets, 'randbits', side_effect=[101, 202]):
            root_a, seeds_a = campaign.trial_seeds(30)
            root_b, seeds_b = campaign.trial_seeds(30)
        self.assertNotEqual(root_a, root_b)
        self.assertNotEqual(seeds_a, seeds_b)
        self.assertEqual(len(set(seeds_a)), 30)
        self.assertEqual((root_a, seeds_a), campaign.trial_seeds(30, root_a))
        self.assertTrue(all(0 < seed < 2**31 for seed in seeds_a))

    def test_default_goal_is_before_tunnel_with_explicit_full_loop_option(self):
        args = campaign.parse_arguments(['--output', '/tmp/not-created'])
        self.assertTrue(args.stop_before_tunnel)
        self.assertIsNone(args.seed)
        self.assertFalse(campaign.parse_arguments(['--output', '/tmp/not-created', '--include-tunnel']).stop_before_tunnel)

    def test_aim_backend_path_and_trial_command_preserve_common_mission_options(self):
        args=campaign.parse_arguments(['--output','/tmp/not-created','--include-tunnel',
            '--gps-blackout-hold','--aim2026-config','AIM2026/config.yaml',
            '--mission','full_loop_tunnel_normal','--speed-cap-kmh','40'])
        self.assertEqual(args.aim2026_config,campaign.REPO/'AIM2026/config.yaml')
        command=campaign.trial_command(args,123,Path('/tmp/ego.csv'))
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
                with self.assertRaises(SystemExit):campaign.parse_arguments(base+extra)

    def test_aim_backend_missing_or_malformed_config_rejected(self):
        base=['--output','/tmp/not-created','--include-tunnel','--gps-blackout-hold']
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'configuration with spaces.yaml'
            for value in ('Localization: []','Localization:\n  enabled: "true"'):
                file.write_text(value)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    campaign.parse_arguments(base+['--aim2026-config',str(file)])
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                campaign.parse_arguments(base+['--aim2026-config',str(file.parent/'missing.yaml')])

    def test_aim_backend_does_not_require_dr_when_localization_explicitly_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'disabled.yaml';file.write_text('Localization:\n  enabled: false')
            args=campaign.parse_arguments(['--output','/tmp/not-created','--aim2026-config',str(file)])
            self.assertFalse(args.gps_blackout_hold)
            self.assertTrue(args.stop_before_tunnel)

    def test_stationary_jitter_and_real_movement(self):
        watch = campaign.MovementWatch(180)
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
        watch = campaign.StaticStuckWatch(30)
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
            self.assertTrue(campaign.STATIC_TROUBLE.search(line), line)
        for line in ('[PLAN] reason=static_recorded_follow', '[PLAN] reason=traffic_light_stop', '[STATIC] {phase: planning}'):
            self.assertFalse(campaign.STATIC_TROUBLE.search(line), line)

    def test_static_stuck_option_defaults_to_30_seconds(self):
        self.assertEqual(campaign.parse_arguments(['--output', '/tmp/x']).static_stuck_seconds, 30.)
        self.assertEqual(campaign.parse_arguments(['--output', '/tmp/x', '--static-stuck-seconds', '0']).static_stuck_seconds, 0.)

    def test_driver_log_location_is_read_from_trial_console(self):
        line = '[GPS+VE 주행 로그] /x/ve_gps_full_route/20261017_194601_620229/driver.console.log'
        self.assertEqual(campaign.DRIVER_LOG.search(line).group(1), '/x/ve_gps_full_route/20261017_194601_620229')

    def test_ignore_signals_option_is_off_by_default(self):
        self.assertEqual(campaign.parse_arguments(['--output', '/tmp/x']).ignore_signals, '')
        self.assertEqual(campaign.parse_arguments(['--output', '/tmp/x', '--ignore-signals', 'C1,C2']).ignore_signals, 'C1,C2')

    def test_signal_hold_after_static_trouble_is_not_static_stuck(self):
        watch = campaign.StaticStuckWatch(30)
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
        self.assertEqual(campaign.parse_arguments(['--output', '/tmp/x']).speed_cap_kmh, 60.)
        self.assertEqual(campaign.parse_arguments(['--output', '/tmp/x', '--speed-cap-kmh', '90']).speed_cap_kmh, 90.)
        with self.assertRaises(SystemExit):
            campaign.parse_arguments(['--output', '/tmp/x', '--speed-cap-kmh', '0'])


if __name__ == '__main__':
    unittest.main()
