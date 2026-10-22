"""The opt-in perception recipe must disconnect only waypoint supervision."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training.vision_encoder.losses.morai_transfer import multitask_loss
from training.vision_encoder.recipes.transfer import (
    add_arguments, configure_transfer_scope, photometric_jitter, resolve_args,
)
from training.vision_encoder.utils.morai_resume_state import (
    PROTECTED_IMPLEMENTATIONS, apply_rng_state, apply_augmentation_rng_state,
    capture_progress, epoch_order, restore_progress,
)


def example(waypoint_nan=False):
    torch.manual_seed(83)
    detection = torch.randn(2, 4, 8, requires_grad=True)
    waypoints = torch.randn(2, 5, 2, requires_grad=True)
    light = torch.randn(2, 2, requires_grad=True)
    target = torch.zeros(2, 4, 8)
    target[:, 1, :] = torch.tensor([1., .2, -.1, 1., 0., .3, .7, .5])
    targets = dict(detection=target, object_valid=torch.ones(2, dtype=torch.bool),
                   traffic_light=torch.tensor([0, 1]),
                   waypoint_valid=torch.ones(2, dtype=torch.bool),
                   waypoints=torch.full((2, 5, 2), float('nan') if waypoint_nan else .5))
    return (detection, waypoints, light), targets


class TinyNativeShape(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('rgb_backbone', 'lidar_backbone', 'encoder', 'decoder',
                     'traffic_pred_head', 'traffic_light_pred_head',
                     'waypoints_generator', 'stop_sign_head'):
            setattr(self, name, nn.Linear(2, 2))
        self.query_embed = nn.Parameter(torch.ones(6, 2))


class PerceptionRecipeTests(unittest.TestCase):
    def test_default_loss_is_unchanged_and_waypoint_supervised(self):
        outputs, targets = example()
        total, tasks = multitask_loss(outputs, targets, detection_loss_policy='lmdrive_lav_v1')
        expected = (.5 * tasks['detection'] + .5 * tasks['waypoint_l1']
                    + .05 * tasks['object_speed'] + .1 * tasks['traffic_light_ce'])
        torch.testing.assert_close(total, expected, rtol=0, atol=0)
        total.backward()
        self.assertGreater(outputs[1].grad.abs().sum(), 0.)

    def test_zero_waypoint_weight_ignores_invalid_future_and_preserves_perception_gradients(self):
        outputs, targets = example(waypoint_nan=True)
        total, tasks = multitask_loss(outputs, targets, detection_loss_policy='lmdrive_lav_v1',
                                     waypoint_loss_weight=0.)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(float(tasks['waypoint_l1']), 0.)
        self.assertFalse(tasks['waypoint_l1'].requires_grad)
        total.backward()
        self.assertIsNone(outputs[1].grad)
        for value in (outputs[0].grad, outputs[2].grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(value.abs().sum(), 0.)
        # A changed future trajectory cannot change the perception objective.
        second_outputs, second_targets = example()
        second, _ = multitask_loss(second_outputs, second_targets,
                                   detection_loss_policy='lmdrive_lav_v1', waypoint_loss_weight=0.)
        second.backward()
        for i in (0, 2):
            torch.testing.assert_close(outputs[i].grad, second_outputs[i].grad, rtol=0, atol=0)

    def test_zero_waypoint_weight_allows_absent_future_target(self):
        outputs, targets = example()
        del targets['waypoints']
        del targets['waypoint_valid']
        loss, _ = multitask_loss(outputs, targets, waypoint_loss_weight=0.)
        self.assertTrue(torch.isfinite(loss))

    def test_zero_weight_freezes_generator_but_keeps_shared_queries_and_heads_trainable(self):
        model = TinyNativeShape()
        expected = configure_transfer_scope(model, 'full', 0.)
        self.assertNotIn('waypoints_generator.', expected)
        self.assertTrue(model.query_embed.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.traffic_pred_head.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.traffic_light_pred_head.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.waypoints_generator.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.stop_sign_head.parameters()))
        before = copy.deepcopy(model.waypoints_generator.state_dict())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=.01, weight_decay=.1)
        optimizer.zero_grad(set_to_none=True)
        loss = sum(p.square().sum() for p in model.parameters() if p.requires_grad)
        loss.backward()
        optimizer.step()
        for name, parameter in model.waypoints_generator.state_dict().items():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_default_scope_remains_full(self):
        model = TinyNativeShape()
        expected = configure_transfer_scope(model, 'full', .5)
        self.assertIn('waypoints_generator.', expected)
        self.assertTrue(all(p.requires_grad for p in model.waypoints_generator.parameters()))

    def test_cli_default_and_invalid_weights(self):
        parser = add_arguments(argparse.ArgumentParser())
        self.assertEqual(parser.parse_args([]).waypoint_loss_weight, .5)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text(json.dumps({}))
            base = ['--config', str(config), '--manifest-index', '/tmp/unused_index',
                    '--checkpoint', '/tmp/unused_checkpoint']
            for value in ('nan', 'inf', '-0.1'):
                args = parser.parse_args(base + ['--waypoint-loss-weight=' + value])
                with self.assertRaisesRegex(ValueError, 'Invalid'):
                    resolve_args(args)
            args = parser.parse_args(base + ['--checkpoint-selection', 'perception_guarded'])
            with self.assertRaisesRegex(ValueError, 'waypoint-loss-weight 0'):
                resolve_args(args)
            args = parser.parse_args(base + ['--checkpoint-selection', 'perception_guarded',
                                             '--waypoint-loss-weight', '0'])
            resolve_args(args)
            self.assertEqual(args.waypoint_loss_weight, 0.)

    def test_loss_rejects_nonfinite_or_negative_weight(self):
        outputs, targets = example()
        for value in (float('nan'), float('inf'), -.5):
            with self.assertRaisesRegex(ValueError, 'Waypoint loss weight'):
                multitask_loss(outputs, targets, waypoint_loss_weight=value)


class AugmentedPerceptionResumeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119)
        self.model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.4), nn.Linear(4, 2))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=5e-6, weight_decay=.01)
        self.generators = {'photometric': torch.Generator().manual_seed(101),
                           'target_point_dropout': torch.Generator().manual_seed(103)}
        self.contract = {'map_basis': 'projected_grid'}
        self.prov = dict(
            schema='morai_visual_transfer_v1', args={'seed': 42},
            publications=[{'path': 'index', 'sha256': 'sealed'}],
            coverage={'train': {'frames': 7}}, trainable_scope='full',
            loss='0.5*detection+0.0*waypoint_L1+0.05*object_speed_L1+0.1*light_CE',
            batch_size=1, frame_sampling={'policy': 'all', 'selected_frames': 7},
            augmentation='photometric_shared_per_frame p=0.5',
            augmentation_rng_contract=dict(policy='main_process_torch_generators_v1',
                generator_names=['photometric', 'target_point_dropout'],
                photometric_probability=.5, target_point_dropout_probability=.2),
            velocity_source='ego_topic', velocity_contract='causal',
            waypoint_source='sealed', target_point='route25m dropout .2',
            batchnorm='frozen stats', stop_sign='frozen', encoder_contract=self.contract,
            optimizer={'name': 'AdamW', 'learning_rate': 5e-6, 'weight_decay': .01,
                       'schedule': 'constant', 'restored': False},
            implementation_sha256={**{k: 'same' for k in PROTECTED_IMPLEMENTATIONS},
                                    'training/vision_encoder/recipes/transfer.py': 'same_recipe'})
        self.order = epoch_order(7, 42, 0)
        for index in self.order[:2]:
            self.update(self.model, self.opt, self.generators, index)
        state = capture_progress(self.model, self.opt, self.prov, self.contract,
            epoch_index=0, order=self.order, cursor=2, global_step=2,
            include_cuda=False, augmentation_generators=self.generators)
        self.source = dict(state_dict=copy.deepcopy(self.model.state_dict()),
            optimizer=copy.deepcopy(self.opt.state_dict()), pretraining=copy.deepcopy(self.prov),
            morai_encoder_contract=self.contract, epoch=1, steps=2, training_state=state)

    @staticmethod
    def update(model, optimizer, generators, index):
        inputs = {'rgb_front': torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4) + index * .01,
                  'target_point': torch.tensor([[1., 2.]])}
        inputs = photometric_jitter(inputs, generators['photometric'], .5,
                                   (.485, .456, .406), (.229, .224, .225))
        dropped = torch.rand(1, generator=generators['target_point_dropout']) < .2
        target = inputs['target_point'].masked_fill(dropped[:, None], 0.)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs['rgb_front'].mean(dim=(2, 3))) - target).square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def test_augmented_resume_matches_next_loss_weights_and_adam_moments_exactly(self):
        expected = self.update(self.model, self.opt, self.generators, self.order[2])
        model = copy.deepcopy(self.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=.01)
        generators = {name: torch.Generator().manual_seed(999) for name in self.generators}
        restored = restore_progress(self.source, model, optimizer, self.prov, self.contract,
                                    dataset_size=7, seed=42)
        torch.rand(5)  # Loader setup may consume the process RNG.
        apply_rng_state(restored['pending_rng_state'])
        apply_augmentation_rng_state(restored['pending_augmentation_rng_state'], generators)
        actual = self.update(model, optimizer, generators, self.order[2])
        self.assertEqual(actual, expected)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, self.model.state_dict()[name], rtol=0, atol=0)
        for pid, values in optimizer.state_dict()['state'].items():
            for name, tensor in values.items():
                torch.testing.assert_close(tensor, self.opt.state_dict()['state'][pid][name], rtol=0, atol=0)

    def test_changed_waypoint_objective_or_missing_rng_is_rejected_before_mutation(self):
        for kind in ('objective', 'missing_rng', 'changed_augmentation', 'changed_augmentation_code'):
            source = copy.deepcopy(self.source)
            if kind == 'objective':
                source['pretraining']['loss'] = self.prov['loss'].replace('0.0*waypoint', '0.5*waypoint')
            elif kind == 'missing_rng':
                del source['training_state']['augmentation_rng_state']
            elif kind == 'changed_augmentation':
                source['pretraining']['augmentation_rng_contract']['photometric_probability'] = .75
            else:
                source['pretraining']['implementation_sha256']['training/vision_encoder/recipes/transfer.py'] = 'different'
            model = copy.deepcopy(self.model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=.01)
            before = copy.deepcopy(model.state_dict())
            with self.assertRaises(ValueError):
                restore_progress(source, model, optimizer, self.prov, self.contract,
                                 dataset_size=7, seed=42)
            self.assertFalse(optimizer.state)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
