"""--runs N: repeat a scenario launcher and keep one diagnosis report per run (no MORAI)."""
import contextlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from AIM2026.Expert import run_repeat
from AIM2026.Expert.run_repeat import run_many, split_repeat_options


class FakeScenario:
    """Stands in for a launcher: each call writes the folder a real run would leave behind."""
    def __init__(self, scenarios, outcomes):
        self.scenarios, self.outcomes, self.calls = Path(scenarios), list(outcomes), []

    def __call__(self, argv):
        self.calls.append(list(argv))
        number = len(self.calls)
        status, error, headline = self.outcomes.pop(0)
        run = self.scenarios / ('highway-2026-%02d' % number)
        (run / 'failure_report').mkdir(parents=True)
        seed = int(argv[argv.index('--seed') + 1]) if '--seed' in argv else 900 + number
        recorded = 'FAILED' if status == 'RAISE' else status          # a real launcher reports FAILED, then raises
        (run / 'report.json').write_text(json.dumps(dict(status=recorded, error=error, plan=dict(seed=seed))), encoding='utf-8')
        (run / 'failure_report' / 'report.html').write_text('<html>report %d</html>' % number, encoding='utf-8')
        (run / 'failure_report' / 'report_data.json').write_text(json.dumps(dict(summary=dict(headline=headline))), encoding='utf-8')
        if status == 'RAISE':
            raise RuntimeError(error)


class SplitOptionTests(unittest.TestCase):
    def test_repeat_options_are_removed_and_the_rest_is_untouched(self):
        options, rest = split_repeat_options(['--runs', '10', '--static-obstacle-count', '2', '--pause-s', '3',
                                              '--stop-on-failure', '--same-seed', '--seed', '124', '--rviz'])
        self.assertEqual((options.runs, options.pause_s, options.seed), (10, 3., 124))
        self.assertTrue(options.stop_on_failure and options.same_seed)
        self.assertEqual(rest, ['--static-obstacle-count', '2', '--rviz'])

    def test_defaults_mean_a_single_run_and_keep_the_seed_for_the_launcher(self):
        options, rest = split_repeat_options(['--seed', '7', '--check'])
        self.assertEqual(options.runs, 1)
        self.assertEqual(rest, ['--check', '--seed', '7'])
        self.assertEqual(split_repeat_options([])[0].runs, 1)

    def test_equals_form_and_invalid_values(self):
        self.assertEqual(split_repeat_options(['--runs=4'])[0].runs, 4)
        for bad in (['--runs', '0'], ['--runs', '-2'], ['--pause-s', '-1']):
            with self.assertRaises(SystemExit):
                split_repeat_options(bad)

    def test_run_timeout_default_and_validation(self):
        self.assertEqual(split_repeat_options([])[0].run_timeout_s, 1200.)
        self.assertEqual(split_repeat_options(['--run-timeout-s', '0'])[0].run_timeout_s, 0.)
        with self.assertRaises(SystemExit):
            split_repeat_options(['--run-timeout-s', '-5'])

    def test_abbreviations_are_not_guessed(self):
        options, rest = split_repeat_options(['--run', '5'])
        self.assertEqual(options.runs, 1)
        self.assertEqual(rest, ['--run', '5'])


class RunManyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.scenarios, self.batches = self.root / 'scenarios', self.root / 'batches'
        self.scenarios.mkdir()
        self.sleeps = []

    def run_batch(self, outcomes, *extra, argv=('--config', 'x.yaml'), **kwargs):
        options, rest = split_repeat_options(['--runs', str(len(outcomes)), *extra])
        fake = FakeScenario(self.scenarios, outcomes)
        with contextlib.redirect_stdout(io.StringIO()):
            code = run_many(fake, list(argv) + rest, options, label='highway', scenarios_dir=self.scenarios,
                            batches_dir=self.batches, sleep=self.sleeps.append, **kwargs)
        return code, fake, next(self.batches.glob('highway-*'))

    def test_ten_runs_leave_ten_reports_and_one_index(self):
        outcomes = [('COMPLETE', None, 'ok %d' % i) for i in range(10)]
        code, fake, batch = self.run_batch(outcomes, '--seed', '100')
        self.assertEqual(code, 0)
        self.assertEqual(len(list((batch / 'reports').glob('*.html'))), 10)
        self.assertEqual([call[call.index('--seed') + 1] for call in fake.calls], [str(100 + i) for i in range(10)])
        self.assertTrue(all('--validation-events' in call and '--config' in call for call in fake.calls))
        summary = json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))
        self.assertEqual((summary['finished_runs'], summary['planned_runs'], summary['complete']), (10, 10, True))
        self.assertEqual(summary['counts'], {'COMPLETE': 10})
        index = (batch / 'index.html').read_text(encoding='utf-8')
        self.assertEqual(index.count('<a href="reports/'), 10)
        self.assertIn('ok 3', index)                       # the diagnosis headline of that run
        self.assertEqual(self.sleeps, [10.] * 9)           # a pause between runs, not after the last

    def test_the_final_summary_names_every_run_that_did_not_complete(self):
        options, rest = split_repeat_options(['--runs', '4', '--seed', '10'])
        fake = FakeScenario(self.scenarios, [('COMPLETE', None, 'a'), ('FAILED', 'collision with 1', 'b'),
                                             ('COMPLETE', None, 'c'), ('STOPPED', None, 'd')])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            run_many(fake, rest, options, label='highway', scenarios_dir=self.scenarios, batches_dir=self.batches,
                     sleep=self.sleeps.append)
        text = printed.getvalue()
        self.assertIn('[Batch] not COMPLETE: 2 of 4', text)
        self.assertIn('run 2  FAILED', text)
        self.assertIn('run_02_FAILED_seed11.html', text)
        self.assertIn('reason: collision with 1', text)
        self.assertIn('run 4  STOPPED', text)
        self.assertNotIn('run 1  COMPLETE', text)
        self.assertIn('log folder:', text)

    def test_the_final_summary_says_so_when_every_run_completed(self):
        options, rest = split_repeat_options(['--runs', '2'])
        fake = FakeScenario(self.scenarios, [('COMPLETE', None, 'a'), ('COMPLETE', None, 'b')])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            run_many(fake, rest, options, label='highway', scenarios_dir=self.scenarios, batches_dir=self.batches,
                     sleep=self.sleeps.append)
        self.assertIn('[Batch] every run COMPLETE', printed.getvalue())
        self.assertNotIn('not COMPLETE', printed.getvalue())

    def test_a_failed_run_is_kept_and_the_batch_continues(self):
        code, fake, batch = self.run_batch([('COMPLETE', None, 'a'), ('FAILED', 'collision', 'b'), ('COMPLETE', None, 'c')],
                                           '--seed', '1')
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.calls), 3)
        rows = json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))['rows']
        self.assertEqual([r['status'] for r in rows], ['COMPLETE', 'FAILED', 'COMPLETE'])
        self.assertEqual(rows[1]['error'], 'collision')
        self.assertTrue((batch / 'reports' / 'run_02_FAILED_seed2.html').is_file())

    def test_an_exception_from_a_run_is_a_result_not_the_end(self):
        code, fake, batch = self.run_batch([('RAISE', 'MORAI died', 'x'), ('COMPLETE', None, 'y')])
        rows = json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))['rows']
        self.assertEqual(len(rows), 2)
        self.assertIn('MORAI died', rows[0]['error'])
        self.assertEqual(code, 1)

    def test_stop_on_failure_stops_after_the_first_bad_run(self):
        code, fake, batch = self.run_batch([('FAILED', 'e', 'a'), ('COMPLETE', None, 'b')], '--stop-on-failure')
        self.assertEqual(len(fake.calls), 1)
        summary = json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))
        self.assertEqual((summary['finished_runs'], summary['complete']), (1, False))

    def test_deferred_collision_review_is_not_counted_as_a_clean_pass(self):
        code, fake, batch = self.run_batch(
            [('REVIEW_REQUIRED', None, 'distant spawn report'), ('COMPLETE', None, 'ok')],
            '--stop-on-failure')
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.calls), 1)
        summary = json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['counts'], {'REVIEW_REQUIRED': 1})

    def test_ctrl_c_in_a_run_stops_the_batch_and_keeps_what_was_done(self):
        code, fake, batch = self.run_batch([('COMPLETE', None, 'a'), ('STOPPED', None, 'b'), ('COMPLETE', None, 'c')])
        self.assertEqual(len(fake.calls), 2)
        self.assertFalse(json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))['complete'])
        self.assertEqual(code, 1)

    def test_same_seed_repeats_one_seed(self):
        _, fake, _ = self.run_batch([('COMPLETE', None, 'a')] * 3, '--seed', '5', '--same-seed')
        self.assertEqual([call[call.index('--seed') + 1] for call in fake.calls], ['5', '5', '5'])

    def test_without_a_seed_the_launcher_chooses_and_the_index_shows_the_one_it_used(self):
        _, fake, batch = self.run_batch([('COMPLETE', None, 'a')] * 2)
        self.assertTrue(all('--seed' not in call for call in fake.calls))
        rows = json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))['rows']
        self.assertEqual([r['seed'] for r in rows], [901, 902])

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'the time limit needs POSIX signals')
    def test_a_stuck_run_is_cut_off_and_the_batch_goes_on(self):
        options, rest = split_repeat_options(['--runs', '2', '--run-timeout-s', '0.3', '--pause-s', '0'])
        fake = FakeScenario(self.scenarios, [('COMPLETE', None, 'second')])
        calls = []

        def run_once(argv):
            calls.append(argv)
            if len(calls) == 1:
                time.sleep(5)                                   # a hung scenario: no watchdog would ever end it
            return fake(argv)
        started = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            code = run_many(run_once, rest, options, label='highway', scenarios_dir=self.scenarios,
                            batches_dir=self.batches, sleep=self.sleeps.append)
        self.assertLess(time.monotonic() - started, 4)
        rows = json.loads((next(self.batches.glob('highway-*')) / 'batch_summary.json').read_text(encoding='utf-8'))['rows']
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['status'], 'FAILED')
        self.assertIn('run exceeded --run-timeout-s', rows[0]['error'])
        self.assertEqual(rows[1]['status'], 'COMPLETE')
        self.assertEqual(code, 1)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0.)      # no alarm is left running

    def test_zero_means_no_time_limit(self):
        marker = object()
        self.assertIs(run_repeat._with_time_limit(marker, 0), marker)

    def test_bad_arguments_stop_before_the_mistake_is_repeated(self):
        options, rest = split_repeat_options(['--runs', '3'])

        def rejects(argv):
            raise SystemExit(2)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            run_many(rejects, rest, options, label='highway', scenarios_dir=self.scenarios, batches_dir=self.batches,
                     sleep=self.sleeps.append)

    def test_error_text_is_escaped_in_the_index(self):
        _, _, batch = self.run_batch([('FAILED', '<script>alert(1)</script>', '<b>x</b>')])
        index = (batch / 'index.html').read_text(encoding='utf-8')
        self.assertNotIn('<script>alert', index)
        self.assertIn('&lt;script&gt;', index)

    def test_a_run_without_a_report_is_listed_without_a_link(self):
        options, rest = split_repeat_options(['--runs', '1'])

        def silent(argv):
            return 0
        with contextlib.redirect_stdout(io.StringIO()):
            code = run_many(silent, rest, options, label='highway', scenarios_dir=self.scenarios, batches_dir=self.batches,
                            sleep=self.sleeps.append)
        batch = next(self.batches.glob('highway-*'))
        rows = json.loads((batch / 'batch_summary.json').read_text(encoding='utf-8'))['rows']
        self.assertEqual((rows[0]['status'], rows[0]['report']), ('NO_REPORT', None))
        self.assertEqual(code, 1)


class LauncherHookupTests(unittest.TestCase):
    """The launchers hand --runs to run_many; run_roundabout inherits it through run_highway.main."""
    def setUp(self):
        try:
            from AIM2026.Expert import run_highway, run_roundabout
        except ImportError as error:
            self.skipTest('launcher imports need the scenario runner environment: %s' % error)
        self.run_highway, self.run_roundabout = run_highway, run_roundabout

    def test_highway_and_every_roundabout_variant_repeat_with_their_own_mission(self):
        cases = [(self.run_highway.main, [], 'highway')] + [
            (self.run_roundabout.main, ['--variant', v], 'roundabout' + ('' if v == 'normal' else '_' + v))
            for v in ('normal', 'nonuniform', 'speed_random', 'brake')]
        for entry, extra, label in cases:
            with self.subTest(label=label), patch.object(run_repeat, 'run_many', return_value=0) as many:
                self.assertEqual(entry(extra + ['--runs', '10', '--seed', '124']), 0)
            options = many.call_args.args[2]
            self.assertEqual((options.runs, options.seed), (10, 124))
            self.assertNotIn('--runs', many.call_args.args[1])
            self.assertEqual(many.call_args.kwargs['label'], label)

    def test_check_never_repeats(self):
        with patch.object(run_repeat, 'run_many') as many, \
             patch.object(self.run_highway, 'load_configuration', side_effect=RuntimeError('stop here')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'stop here'):
                self.run_highway.main(['--runs', '5', '--check'])
        many.assert_not_called()


if __name__ == '__main__':
    unittest.main()
