"""Independent checkpoint evaluation must score objects for the exact frames."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, Subset

from utils.evaluation import morai_transfer_evaluation as evaluation


class AdmittedDataset(Dataset):
    def __init__(self):
        # Deliberately differ from raw manifest order: diagnostics must follow
        # admitted dataset rows, not the original list of manifest entries.
        self.rows = [(None, {'frame_id': i, 'sensor_time_ns': 1000 + i,
                             'object_valid': i != 4}, None, None) for i in (5, 2, 4)]
        self.map_basis = None
        self.coverage = {'object_valid': 2}
        self.excluded = {'missing_asset': 2}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index][1]


class StandaloneEvaluationTest(unittest.TestCase):
    def test_workers_use_spawn_with_isolated_explicit_generator(self):
        loader = evaluation.make_evaluation_loader(AdmittedDataset(), 2)
        self.assertEqual(loader.multiprocessing_context.get_start_method(), 'spawn')
        self.assertEqual(loader.num_workers, 2)
        self.assertEqual(loader.prefetch_factor, 2)
        self.assertFalse(loader.pin_memory)
        self.assertFalse(loader.persistent_workers)
        self.assertEqual(loader.generator.initial_seed(), 0)
        policy = evaluation.evaluation_loader_policy(2)
        self.assertEqual(policy['multiprocessing_context'], 'spawn')
        self.assertEqual(policy['implementation_sha256'], evaluation.digest(
            evaluation.ROOT / 'data/src/loaders/morai_dataloader.py'))

    def test_zero_workers_preserve_global_rng_and_exact_order(self):
        before = torch.get_rng_state().clone()
        loader = evaluation.make_evaluation_loader(AdmittedDataset(), 0)
        self.assertIsNone(loader.multiprocessing_context)
        self.assertEqual([int(batch['frame_id'][0]) for batch in loader], [5, 2, 4])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_nested_subset_metadata_follows_reordered_indices(self):
        data = AdmittedDataset()
        selected = Subset(Subset(data, [2, 0, 1]), [1, 0])
        self.assertEqual([e['frame_id'] for e in evaluation.ordered_object_entries(selected)], [5, 4])

    def _run_and_check(self, limit, expected):
        data = AdmittedDataset()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, manifest_path = root / 'index.json', root / 'manifest.json'
            index_path.write_text('{}')
            manifest_path.write_text('{}')
            manifest = {'entries': [{'frame_id': i} for i in range(1, 6)]}
            args = argparse.Namespace(threads=1, workers=0, device='cpu', output=root / 'evaluation',
                checkpoint=root / 'checkpoint.pth', checkpoint_sha256=None, velocity_source='ego_topic',
                manifest_index=[index_path], split='val', diagnostic_limit=limit,
                allow_map_basis_migration=False)
            calls = []

            def inspect_evaluation(model, loader, device, **kwargs):
                self.assertEqual(kwargs['detection_loss_policy'], 'lmdrive_lav_v1')
                entries = kwargs['object_entries']
                ids = [entry['frame_id'] for entry in entries]
                observed_ids = [int(batch['frame_id'][0]) for batch in loader]
                self.assertEqual(ids, expected)
                self.assertEqual(ids, observed_ids)
                calls.append(ids)
                return {'frames': len(ids), 'object_detection': {'frames': len(ids)}}

            with patch.object(evaluation, '_resolve', return_value=object()), \
                 patch.object(evaluation, '_load_model', return_value=(object(), {}, 'checkpoint_hash')), \
                 patch.object(evaluation, 'load_publication', return_value=(
                     {'val': [{'lap': 3}]}, {'val': [(manifest_path, manifest)]})), \
                 patch.object(evaluation, 'SensorRecordDataset', return_value=data), \
                 patch.object(evaluation, 'evaluate', side_effect=inspect_evaluation), \
                 patch('builtins.print'):
                result = evaluation.run(args)
            self.assertEqual(calls, [expected])
            self.assertEqual(result['results'][0]['evaluated_frames'], len(expected))
            self.assertEqual(result['results'][0]['metrics']['object_detection']['frames'], len(expected))
            self.assertEqual(result['object_detection_policy']['schema'], 'morai_oriented_bev_object_diagnostic_v2')
            code_path = 'utils/evaluation/morai_object_metrics.py'
            self.assertEqual(result['metric_code_sha256'][code_path], evaluation.digest(evaluation.ROOT / code_path))
            saved_config = json.loads((args.output / 'config.json').read_text())
            saved_report = json.loads((args.output / 'report.json').read_text())
            self.assertEqual(saved_config['metric_code_sha256'], saved_report['metric_code_sha256'])
            self.assertEqual(saved_config['object_detection_policy'], saved_report['object_detection_policy'])
            self.assertEqual(saved_config['loader_policy'], saved_report['loader_policy'])
            self.assertEqual(saved_report['loader_policy']['workers'], args.workers)

    def test_diagnostic_limit_applies_after_admission(self):
        self._run_and_check(2, [5, 2])

    def test_no_limit_covers_all_admitted_rows(self):
        self._run_and_check(0, [5, 2, 4])

    def test_limit_larger_than_dataset_preserves_full_coverage(self):
        self._run_and_check(100, [5, 2, 4])


if __name__ == '__main__':
    unittest.main()
