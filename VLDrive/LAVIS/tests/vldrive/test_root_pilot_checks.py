import copy
import unittest
from tools.morai.root_pilot_checks import category, summarize, gate, compare


class RootPilotChecks(unittest.TestCase):
    @staticmethod
    def passing_pair():
        baseline = dict(initial_trainable_sha256='same', sampling=[1, 2], seed=607,
            optimizer={'lr': 1e-5}, schedule_updates=9540, provenance={'data': 'same'},
            updates=320, frozen_unchanged=True,
            controlled_settings={'model': {'memory_size': 10}, 'train_window_stride': 5},
            initial_task_probe=dict(trajectory=[[[1., 2.]]], selector_waypoint_l2=1e-7,
                selector_weighted_pruning_l2=1., selector_weighted_reconstruction_l2=200.),
            after={'curved': {'fde_m': 4.}}, gate={'passed': False})
        candidate = copy.deepcopy(baseline)
        candidate['initial_task_probe']['selector_waypoint_l2'] = 1e-3
        candidate['after']['curved']['fde_m'] = 1.
        candidate['gate']['passed'] = True
        return baseline, candidate

    def test_category_is_actual_trajectory_not_instruction(self):
        self.assertEqual(category([[0, 0]] * 5), 'stationary')
        self.assertEqual(category([[3, -5]] * 5), 'moving_left')
        self.assertEqual(category([[-3, -5]] * 5), 'moving_right')

    def test_barely_turning_not_hidden_by_direction_accuracy(self):
        rows = [dict(category='moving_left', prediction=[[.6, -5]] * 5,
                     actual_future=[[4., -5]] * 5, errors_m=[3.4] * 5)]
        metrics = summarize(rows)['curved']
        self.assertEqual(metrics['direction_accuracy'], 1.)
        self.assertAlmostEqual(metrics['lateral_error_m'], 3.4)

    def test_count_only_and_mismatched_trials_rejected(self):
        base = dict(initial_trainable_sha256='same', sampling=[1], seed=607, optimizer={'lr': 1e-5},
            schedule_updates=9540, provenance={'data': 'same'},
            updates=320, controlled_settings={}, frozen_unchanged=True,
            initial_task_probe=dict(trajectory=[1], selector_waypoint_l2=1e-7,
                                    selector_weighted_pruning_l2=1., selector_weighted_reconstruction_l2=2.),
            after={'curved': {'fde_m': 4.}}, gate={'passed': False})
        candidate = copy.deepcopy(base)
        candidate['initial_task_probe']['selector_waypoint_l2'] = 1.
        self.assertFalse(compare(base, candidate)['passed'])
        candidate['gate']['passed'] = True
        candidate['after']['curved']['fde_m'] = 1.
        self.assertTrue(compare(base, candidate)['passed'])
        candidate['seed'] = 99
        self.assertFalse(compare(base, candidate)['passed'])

    def test_baseline_selected_when_candidate_has_no_advantage(self):
        base = dict(initial_trainable_sha256='same', sampling=[1], seed=607, optimizer={},
            schedule_updates=9540, provenance={}, updates=320, controlled_settings={}, frozen_unchanged=True,
            initial_task_probe=dict(trajectory=[1], selector_waypoint_l2=1e-7,
                                    selector_weighted_pruning_l2=1., selector_weighted_reconstruction_l2=2.),
            after={'curved': {'fde_m': 1.}}, gate={'passed': True})
        self.assertEqual(compare(base, copy.deepcopy(base))['selected_mode'], 'legacy')

    def test_equal_auxiliary_gradients_and_controlled_trial_select_candidate(self):
        result = compare(*self.passing_pair())
        self.assertTrue(result['passed'])
        self.assertEqual(result['selected_mode'], 'retained_output_st')
        self.assertTrue(result['auxiliary_selector_gradient_unchanged'])

    def test_auxiliary_gradient_changes_reject_even_better_waypoints(self):
        for field in ('selector_weighted_pruning_l2', 'selector_weighted_reconstruction_l2'):
            with self.subTest(field=field):
                baseline, candidate = self.passing_pair()
                candidate['initial_task_probe'][field] *= 1.01
                result = compare(baseline, candidate)
                self.assertFalse(result['passed'])
                self.assertIsNone(result['selected_mode'])
                self.assertTrue(result['matched_experiment'])
                self.assertFalse(result['auxiliary_selector_gradient_unchanged'])

    def test_auxiliary_comparison_documents_small_numeric_tolerance(self):
        baseline, candidate = self.passing_pair()
        for field in ('selector_weighted_pruning_l2', 'selector_weighted_reconstruction_l2'):
            candidate['initial_task_probe'][field] *= 1. + 5e-7
        result = compare(baseline, candidate)
        self.assertTrue(result['passed'])
        self.assertEqual(result['auxiliary_gradient_relative_tolerance'], 1e-6)

    def test_different_completed_updates_reject_comparison(self):
        baseline, candidate = self.passing_pair()
        candidate['updates'] = 160
        result = compare(baseline, candidate)
        self.assertFalse(result['passed'])
        self.assertFalse(result['matched_experiment'])

    def test_nonparameter_model_and_data_config_mismatches_reject(self):
        for change in ('memory_size', 'train_window_stride'):
            with self.subTest(change=change):
                baseline, candidate = self.passing_pair()
                if change == 'memory_size':
                    candidate['controlled_settings']['model'][change] = 5
                else:
                    candidate['controlled_settings'][change] = 1
                # Identical tensor hashes do not establish equal runtime semantics.
                self.assertEqual(baseline['initial_trainable_sha256'], candidate['initial_trainable_sha256'])
                result = compare(baseline, candidate)
                self.assertFalse(result['passed'])
                self.assertFalse(result['matched_experiment'])

    def test_frozen_parameter_change_in_either_arm_blocks_selection(self):
        for arm in (0, 1):
            with self.subTest(arm=arm):
                pair = self.passing_pair()
                pair[arm]['frozen_unchanged'] = False
                result = compare(*pair)
                self.assertFalse(result['passed'])
                self.assertIsNone(result['selected_mode'])

    def test_initial_forward_mismatch_rejects_backward_only_claim(self):
        baseline, candidate = self.passing_pair()
        candidate['initial_task_probe']['trajectory'][0][0][0] += .01
        result = compare(baseline, candidate)
        self.assertFalse(result['passed'])
        self.assertFalse(result['identical_initial_forward'])


if __name__ == '__main__':
    unittest.main()
