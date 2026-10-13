"""Guard evidence ordering and pure-color/permission separation, without a VE."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'tools/evaluation/evaluate_ve_expert_connection.py'
spec = importlib.util.spec_from_file_location('connection_runner', str(MODULE))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ConnectionContractTests(unittest.TestCase):
    def setUp(self):
        self.document = dict(entries=[dict(frame_id=10 + i, sensor_time_ns=100 + i,
                                           object_valid=True) for i in range(4)])
        self.sequence = dict(entry_indices=[1, 2], frame_ids=[11, 12])

    def test_exact_chronological_selection(self):
        entries = runner.validate_selection(self.sequence, self.document)
        self.assertEqual([row['frame_id'] for row in entries], [11, 12])

    def test_frame_identity_rejects_different_sealed_row(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            runner.validate_selection(dict(self.sequence, frame_ids=[12, 13]), self.document)

    def test_reordered_sequence_rejected(self):
        with self.assertRaisesRegex(ValueError, 'chronological'):
            runner.validate_selection(dict(entry_indices=[2, 1], frame_ids=[12, 11]), self.document)

    def test_duplicate_sequence_frame_rejected(self):
        with self.assertRaises(ValueError):
            runner.validate_selection(dict(entry_indices=[1, 1], frame_ids=[11, 11]), self.document)

    def test_nonadvancing_physical_time_rejected(self):
        self.document['entries'][2]['sensor_time_ns'] = 100
        with self.assertRaisesRegex(ValueError, 'sensor time'):
            runner.validate_selection(self.sequence, self.document)

    def test_unverified_objects_are_not_background(self):
        self.document['entries'][2]['object_valid'] = False
        with self.assertRaisesRegex(ValueError, 'object supervision'):
            runner.validate_selection(self.sequence, self.document)

    def test_green_correct_color_can_have_invalid_left_permission(self):
        entry = dict(traffic_light=1, traffic_signal=dict(light_id='left',
                         raw_signal_bits=16, signal_movement='left'),
                     route_permission=dict(valid=True, target=False))
        row = runner.color_diagnostic([-4., 4.], entry)
        self.assertTrue(row['color_correct'])
        self.assertTrue(row['color_only_counterfactual_false_go'])
        self.assertIsNone(row['predicted_route_permission'])
        self.assertIsNone(row['predicted_light_id'])
        self.assertFalse(row['expert_signal_stop_evaluated'])

    def test_arrow_is_masked_color_not_invented_class(self):
        entry = dict(traffic_light=-1, traffic_signal=dict(raw_signal_bits=33),
                     route_permission=dict(valid=True, target=True))
        row = runner.color_diagnostic([-1., 1.], entry)
        result = runner.summarize_colors([row])
        self.assertEqual(result['color_observed_frames'], 0)
        self.assertEqual(result['masked_color_frames'], 1)
        self.assertIsNone(result['color_accuracy'])
        self.assertFalse(result['predicted_route_permission_available'])

    def test_unknown_permission_is_not_false_go(self):
        row = runner.color_diagnostic([-1., 1.], dict(traffic_light=-1))
        self.assertIsNone(row['color_only_counterfactual_false_go'])
        self.assertEqual(runner.summarize_colors([row])['bound_route_permission_frames'], 0)

    def test_nonfinite_model_color_fails(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            runner.color_diagnostic([0., float('nan')], dict(traffic_light=0))

    def test_global_path_hash_must_match_manifest_navigation(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gp.txt'
            np.savetxt(path, [[0., 0., 0.], [1., 0., 0.]])
            route = dict(path=str(path), sha256=runner.digest(path), format='whitespace_xyz',
                         coordinate_frame='projected_grid', points_count=2, xy_columns=[0, 1])
            with self.assertRaisesRegex(ValueError, 'navigation provenance'):
                runner.load_route(dict(route=route), dict(navigation_contract=dict(global_path_sha256='wrong')),
                                  lambda value: Path(value))

    def test_global_path_is_not_future_waypoint_array(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gp.txt'
            np.savetxt(path, [[0., 0., 3.], [1., 0., 3.]])
            checksum = runner.digest(path)
            route = dict(path=str(path), sha256=checksum, format='whitespace_xyz',
                         coordinate_frame='projected_grid', points_count=2, xy_columns=[0, 1])
            values = runner.load_route(dict(route=route),
                       dict(navigation_contract=dict(global_path_sha256=checksum), waypoints=[999., 999.]),
                       lambda value: Path(value))
            self.assertEqual(values.tolist(), [[0., 0.], [1., 0.]])

    def test_numeric_summary_excludes_flags_and_nulls(self):
        result = runner.summarize_planning([dict(diagnostics=dict(path_error_m=2., agree=True, missing=None)),
                                           dict(diagnostics=dict(path_error_m=4., agree=False))])
        self.assertEqual(result['numeric_means']['diagnostics.path_error_m'], 3.)
        self.assertNotIn('diagnostics.agree', result['numeric_means'])
        self.assertEqual(result['boolean_counts']['diagnostics.agree.true'], 1)

    def test_collision_comparison_distinguishes_planner_and_perception_limits(self):
        rows = [dict(comparison=dict(decision_agreement=False, obstacle_flag_agreement=False,
                                    gt_path_collision=False, ve_path_collision=True)),
                dict(comparison=dict(decision_agreement=True, obstacle_flag_agreement=True,
                                     gt_path_collision=True, ve_path_collision=True))]
        result = runner.summarize_planning(rows)['paired_geometry_comparison']
        self.assertEqual(result['decision_agreement_rate'], .5)
        self.assertEqual(result['ve_collision_gt_clear_frames'], 1)
        self.assertEqual(result['both_paths_collide_frames'], 1)


if __name__ == '__main__':
    unittest.main()
