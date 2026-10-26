"""Tunnel regression checks: hold reports."""

import contextlib, importlib.util, io, json, math, tempfile, unittest, re
from pathlib import Path


# Analyze tunnel hold

ANALYZER_MODULE = Path(__file__).resolve().parents[1]/'tools/analyze_tunnel_hold.py'
analyzer_spec = importlib.util.spec_from_file_location('analyze_tunnel_hold', ANALYZER_MODULE)
analyzer = importlib.util.module_from_spec(analyzer_spec)
analyzer_spec.loader.exec_module(analyzer)


def matched(receipt, estimate=(3., 4.), real=(0., 0.), **extra):
    return dict(gps_received_ns=receipt, dr_received_ns=receipt,
                estimated_xy=list(estimate), real_gps_xy=list(real), yaw_grid_rad=0.,
                speed_mps=10., hold_s=receipt/1e9, distance_m=receipt/1e8,
                time_basis='receipt_monotonic', alignment_method='interpolated_dr_history', **extra)


def row(mode, comparison=None, moment=1., position=(0., 0.), **extra):
    gps = dict(position_mode=mode, hold_s=moment, dead_reckoning=dict(distance_m=moment*10), **extra)
    if comparison is not None:
        gps['simulated_comparisons'] = comparison
    return dict(monotonic=moment, position=dict(x=position[0], y=position[1],
                yaw_grid_rad=0., speed_mps=10.), control_pose_selected_samples=dict(gps=gps))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def trial(self, number=1, result='success', rows=None, baseline=True, console_extra=''):
        trial = self.root/('trial_%02d' % number)
        # A path with spaces exercises the historical console announcement parser.
        runtime = self.root/('runtime with spaces %d' % number)
        (runtime/'driver').mkdir(parents=True)
        trial.mkdir()
        (trial/'result.json').write_text(json.dumps(dict(result=result, elapsed_s=20)))
        (trial/'console.log').write_text('[GPS+VE 주행 로그] %s/driver.console.log\n%s' % (runtime, console_extra))
        (runtime/'driver/control.jsonl').write_text(''.join(json.dumps(value)+'\n' for value in rows or []))
        if baseline is not None:
            folder = runtime/'tunnel_baseline/run1'
            folder.mkdir(parents=True)
            (folder/'baseline.json').write_text(json.dumps(dict(started_before_entry=True,
                mapped_entry_observed=True, mapped_exit_observed=baseline)))
        return trial, runtime

    def test_receipt_matched_window_is_deduplicated_including_gps_recovery(self):
        a, b = matched(1_000_000_000), matched(2_000_000_000, estimate=(0., 2.))
        rows = [row('gps'), row('dead_reckoning', [a]),
                row('dead_reckoning', [a]), row('gps', [a, b], moment=3.)]
        trial, _ = self.trial(rows=rows)
        report, samples = analyzer.analyze_trial(trial)
        self.assertEqual(len(samples), 2)
        self.assertEqual(report['completion'], 'complete')
        self.assertTrue(report['accuracy_aggregate_eligible'])
        self.assertTrue(report['gps_recovery_observed'])
        self.assertEqual(report['diagnostics']['duplicate_comparisons'], 2)
        errors = report['comparisons'][analyzer.MATCHED]['errors']
        self.assertEqual(errors['error_2d_m']['max'], 5.)
        self.assertEqual(errors['error_2d_m']['median'], 3.5)
        self.assertAlmostEqual(errors['error_2d_m']['p95'], 4.85)
        self.assertEqual(report['comparisons'][analyzer.MATCHED]['exit_last_observed_error']['gps_received_ns'], 2_000_000_000)

    def test_mismatched_receipt_time_is_not_called_matched_or_fallback(self):
        bad = matched(1_000_000_000)
        bad['dr_received_ns'] += 1
        trial, _ = self.trial(rows=[row('dead_reckoning', [bad],
            simulated_real_gps_xy=[0, 0], simulated_real_gps_age_s=0)])
        report, samples = analyzer.analyze_trial(trial)
        self.assertFalse(samples)
        self.assertEqual(report['diagnostics']['invalid_comparisons'], 1)
        self.assertIn('no_comparison_samples', report['exclusions'])

    def test_legacy_approximation_and_new_comparisons_never_pool(self):
        legacy = row('dead_reckoning', position=(11., 2.),
                     simulated_real_gps_xy=[0., 0.], simulated_real_gps_age_s=1.)
        self.trial(rows=[legacy, row('dead_reckoning', [matched(1_000_000_000)]), row('gps')])
        report, samples = analyzer.analyze(self.root)
        self.assertEqual({item['method'] for item in report['aggregates']}, {analyzer.MATCHED, analyzer.LEGACY})
        old = next(item for item in samples if item['method'] == analyzer.LEGACY)
        self.assertEqual((old['longitudinal_m'], old['lateral_m']), (1., 2.))
        self.assertAlmostEqual(old['error_2d_m'], math.sqrt(5))

    def test_expired_input_excluded_even_when_result_says_success(self):
        self.trial(rows=[row('dead_reckoning', [matched(1_000_000_000)]), row('gps')],
                   console_extra='reason=sensors_expired_during_inference')
        self.trial(2, rows=[row('dead_reckoning', [matched(2_000_000_000)])])
        report, _ = analyzer.analyze(self.root)
        self.assertEqual(report['observed_trials'], 2)
        self.assertEqual(report['eligible_complete_trials'], 1)
        self.assertEqual(report['exclusion_reason_counts'][analyzer.EXPIRY], 1)
        eligible = [item for item in report['aggregates'] if item['scope'] == 'eligible_complete']
        self.assertEqual(eligible[0]['trials'], ['trial_02'])

    def test_partial_and_failure_results_remain_diagnostic(self):
        self.trial(rows=[row('dead_reckoning', [matched(1_000_000_000)])], baseline=False)
        self.trial(2, result='collision', rows=[row('dead_reckoning', [matched(2_000_000_000)]), row('gps')])
        report, _ = analyzer.analyze(self.root)
        self.assertEqual(report['eligible_complete_trials'], 0)
        self.assertEqual(report['completion_counts'], dict(partial=2))
        self.assertTrue(all(group['scope'] == 'excluded_partial' for group in report['aggregates']))
        self.assertIsNone(report['trials'][0]['comparisons'][analyzer.MATCHED]['exit_last_observed_error'])

    def test_explicit_incomplete_boundaries_override_gps_recovery(self):
        trial, _ = self.trial(rows=[row('dead_reckoning', [matched(1_000_000_000)]), row('gps')], baseline=False)
        report, _ = analyzer.analyze_trial(trial)
        self.assertEqual(report['completion'], 'partial')
        self.assertEqual(report['completion_evidence'], 'insufficient')
        self.assertTrue(report['gps_recovery_observed'])

    def test_old_complete_trial_can_use_weaker_recovery_evidence(self):
        trial, _ = self.trial(rows=[row('dead_reckoning', [matched(1_000_000_000)]), row('gps')], baseline=None)
        report, _ = analyzer.analyze_trial(trial)
        self.assertEqual(report['completion'], 'complete')
        self.assertEqual(report['completion_evidence'], 'gps_recovery_after_dr')
        self.assertTrue(any('weaker' in warning for warning in report['warnings']))

    def test_direct_driver_log_supported_and_malformed_rows_counted(self):
        _, runtime = self.trial(rows=[row('dead_reckoning', [matched(1_000_000_000)])])
        path = runtime/'driver/control.jsonl'
        with path.open('a') as stream:
            stream.write('not JSON\n[]\n')
        report, samples = analyzer.analyze(path)
        self.assertEqual(len(samples), 1)
        self.assertEqual(report['trials'][0]['diagnostics']['malformed_control_lines'], 2)
        self.assertEqual(report['trials'][0]['completion'], 'partial')

    def test_sensor_only_direct_trial_and_driver_share_result_and_boundary_evidence(self):
        trial = self.root/'trial_01'
        (trial/'driver').mkdir(parents=True)
        (trial/'result.json').write_text(json.dumps(dict(result='success', elapsed_s=12)))
        (trial/'baseline.json').write_text(json.dumps(dict(started_before_entry=True,
            mapped_entry_observed=True, mapped_exit_observed=True)))
        (trial/'driver/control.jsonl').write_text(json.dumps(row('dead_reckoning', [matched(1_000_000_000)]))+'\n')
        for path in (trial, trial/'driver', trial/'driver/control.jsonl'):
            with self.subTest(path=path):
                report, samples = analyzer.analyze(path)
                self.assertEqual(report['eligible_complete_trials'], 1)
                self.assertEqual(report['trials'][0]['trial'], 'trial_01')
                self.assertEqual(len(samples), 1)

    def test_blocked_empty_campaign_reports_zero_trials_and_writes_artifacts(self):
        (self.root/'report.json').write_text(json.dumps(dict(requested=5, trials=[], status='BLOCKED')))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(analyzer.main([str(self.root), '--no-plot']), 0)
        folder = self.root/'tunnel_hold_analysis'
        report = json.loads((folder/'analysis.json').read_text())
        self.assertEqual(report['observed_trials'], 0)
        self.assertEqual(report['eligible_complete_trials'], 0)
        self.assertEqual(report['assessment'], 'not_assessed_no_application_tolerance')
        self.assertTrue((folder/'samples.csv').is_file())
        self.assertTrue((folder/'trials.csv').is_file())

    def test_actual_speed_hold_and_signed_errors_follow_yaw(self):
        comparison = matched(1_000_000_000, estimate=(3., 4.))
        comparison['yaw_grid_rad'] = math.pi/2
        trial, _ = self.trial(rows=[row('dead_reckoning', [comparison], moment=4.)])
        report, samples = analyzer.analyze_trial(trial)
        self.assertAlmostEqual(samples[0]['longitudinal_m'], 4.)
        self.assertAlmostEqual(samples[0]['lateral_m'], -3.)
        self.assertEqual(report['hold_s'], 4.)
        self.assertEqual(report['distance_m'], 40.)
        self.assertEqual(report['actual_speed_kmh']['median'], 36.)


if __name__ == "__main__":
    unittest.main()
