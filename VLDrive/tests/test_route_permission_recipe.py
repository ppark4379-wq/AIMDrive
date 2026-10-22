"""CPU public-recipe smoke: admitted rows -> head update -> sealed-base reload."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from models.ve.route_permission import SENSOR_INPUT_KEYS
from tests.test_morai_route_permission_data import entry
from tests.test_route_permission_head import NativeEncoderFixture
from training.vision_encoder.recipes import route_permission as recipe


class AdmittedFixture:
    """Record admission of every frame before the permission-only subset."""
    constructed = []
    sampled = []

    def __init__(self, manifests, resolver, velocity_source):
        del resolver
        if velocity_source != 'ego_topic':
            raise ValueError('Unexpected velocity contract')
        self.map_basis = {'schema': 'test_grid'}
        self.rows, self.row_manifest_paths = [], []
        for path, doc in manifests:
            for row in doc['entries']:
                self.rows.append(('native', copy.deepcopy(row), None, 0.))
                self.row_manifest_paths.append(str(path))
        self.constructed.append([row[1]['frame_id'] for row in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index][1]
        self.sampled.append(row['frame_id'])
        value = float(row['route_permission']['raw_signal_bits'] == 33)
        inputs = {name: torch.full((3, 2, 2), value)
                  for name in SENSOR_INPUT_KEYS if name.startswith('rgb_')}
        inputs.update(lidar=torch.zeros(4, 4), num_points=torch.tensor(4),
                      velocity=torch.zeros(1), target_point=torch.ones(2))
        targets = dict(frame_id=torch.tensor(row['frame_id'], dtype=torch.int64),
                       sensor_time_ns=torch.tensor(row['sensor_time_ns'], dtype=torch.int64),
                       traffic_light=torch.tensor(row['traffic_light']))
        return inputs, targets


class RoutePermissionRecipeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(283)
        self.base = NativeEncoderFixture().eval()
        self.base_state = copy.deepcopy(self.base.state_dict())
        self.loaded_models = []
        self.manifests = {}
        for split, offset in (('train', 0), ('val', 100)):
            rows = []
            for i, raw in enumerate((1, 16, 33, 33)):
                row = entry(raw)
                row['frame_id'] = offset + i
                if i == 3:
                    row['route_permission']['valid'] = False
                    row['route_permission']['signal_movement'] = None
                rows.append(row)
            self.manifests[split] = [(Path(split + '_manifest.json'), {'entries': rows})]
        self.index = dict(publications=[{'path': 'sealed', 'sha256': 'b' * 64}],
                          manifest_scenarios={split + '_manifest.json': 'full_loop_traffic_light'
                                              for split in ('train', 'val')})
        AdmittedFixture.constructed = []
        AdmittedFixture.sampled = []

    def load_model(self, checkpoint, expected_sha, device):
        del checkpoint
        self.assertEqual(expected_sha, 'a' * 64)
        self.assertEqual(device, 'cpu')
        base = copy.deepcopy(self.base)
        self.loaded_models.append(base)
        return base, dict(morai_encoder_contract=dict(
            velocity_input='causal_ego_topic_speed_mps_50ms', map_basis={'schema': 'test_grid'})), 'a' * 64

    def test_public_train_then_evaluate_preserves_base_and_masks_invalid_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps(dict(checkpoint='base.pt', checkpoint_sha256='a' * 64,
                                              manifest_indices=['published'], velocity_source='ego_topic')))
            args = recipe.argument_parser().parse_args([
                '--config', str(config), '--output', str(root / 'train'), '--mode', 'train',
                '--device', 'cpu', '--epochs', '1', '--workers', '0', '--threads', '1'])
            with patch.object(recipe, 'load_publications', return_value=(self.index, self.manifests)), \
                 patch.object(recipe, 'SensorRecordDataset', AdmittedFixture), \
                 patch.object(recipe, '_load_model', side_effect=self.load_model):
                trained = recipe.run(args)
                head_path = root / 'train' / 'head-epoch-001.pt'
                self.assertTrue(head_path.is_file())
                self.assertEqual(trained['training_updates'], 3)
                self.assertEqual(trained['base_encoder_updates'], 0)
                self.assertTrue(trained['base_encoder_unchanged'])
                eval_args = recipe.argument_parser().parse_args([
                    '--config', str(config), '--output', str(root / 'evaluation'),
                    '--mode', 'evaluate', '--device', 'cpu', '--workers', '0', '--threads', '1',
                    '--head-checkpoint', str(head_path)])
                evaluated = recipe.run(eval_args)
            self.assertEqual(evaluated['training_updates'], 0)
            self.assertEqual(evaluated['metrics']['confusion_matrix'], trained['after']['confusion_matrix'])
            self.assertEqual(evaluated['metrics']['loss'], trained['after']['loss'])
            self.assertEqual(evaluated['metrics']['evaluated_permission_frames'], 3)
            self.assertEqual(evaluated['metrics']['source_excluded_permission_frames'], 1)
            self.assertFalse(evaluated['automatic_deployment'])
            self.assertEqual(AdmittedFixture.constructed, [[0, 1, 2, 3], [100, 101, 102, 103]] * 2)
            self.assertNotIn(3, AdmittedFixture.sampled)
            self.assertNotIn(103, AdmittedFixture.sampled)
            for base in self.loaded_models:
                self.assertFalse(any(p.requires_grad for p in base.parameters()))
                for name, tensor in base.state_dict().items():
                    torch.testing.assert_close(tensor, self.base_state[name], rtol=0, atol=0)
            payload = torch.load(str(head_path), map_location='cpu', weights_only=True)
            self.assertEqual(payload['base_checkpoint_sha256'], 'a' * 64)
            self.assertTrue(all(name.startswith('network.') for name in payload['state_dict']))


if __name__ == '__main__':
    unittest.main()
