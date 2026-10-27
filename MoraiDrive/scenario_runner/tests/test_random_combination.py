import io
import unittest
from unittest.mock import patch

from scenario_runner.tools import run_aim_dynamic as runner


class RandomCombinationTests(unittest.TestCase):
    def collect(self, extra=(), count=64):
        with patch.object(runner, 'run_once', side_effect=[0] * (count - 1) + [130]) as run, \
                patch('sys.stdout', new_callable=io.StringIO):
            runner.main(['--repeat', '--random-combination', '--seed', '123', *extra])
        return [(c.args[0].opposing_crosswalk, tuple(c.args[0].roads)) for c in run.call_args_list]

    def test_repeat_varies_and_seed_reproduces_all_combinations(self):
        choices = self.collect(count=256)
        self.assertEqual(choices, self.collect(count=256))
        self.assertEqual(len(set(choices)), 16)

    def test_fixed_parts(self):
        self.assertEqual({c for c, _ in self.collect(['--opposing-crosswalk', '2'])}, {2})
        choices = self.collect(['--roads', '1', '5'])
        self.assertEqual({r for _, r in choices}, {(1, 5)})
        self.assertEqual({c for c, _ in choices}, {2, 3})

    def test_once_and_config(self):
        with patch.object(runner, 'run_once', return_value=0) as run:
            runner.main(['--repeat', '--random-combination', '--once', '--seed', '123'])
        run.assert_called_once()
        args = run.call_args.args[0]
        config = runner.build_dynamic_config({}, 1, opposing_crosswalk=args.opposing_crosswalk, roads=args.roads)
        self.assertEqual(config['dynamic_obstacle']['required_count'], 2 + len(args.roads))

    def test_invalid_combinations(self):
        for extra in (['--random-locations', '2'], ['--only-location', '5'],
                      ['--dynamic-location-mix'], ['--dynamic-obstacle-count', '2'],
                      ['--roads', '1', '1']):
            with patch.object(runner, 'run_once') as run, \
                    patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                runner.main(['--random-combination', *extra])
            run.assert_not_called()
