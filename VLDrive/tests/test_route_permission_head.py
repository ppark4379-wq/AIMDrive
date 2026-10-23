"""CPU checks for the additive movement-conditioned signal head and sidecar."""
import copy
import io
import unittest

import torch
from torch import nn

from models.ve.route_permission import (
    SENSOR_INPUT_KEYS, RoutePermissionEncoder, RoutePermissionHead,
    build_route_permission_checkpoint, load_route_permission_checkpoint,
)
from training.vision_encoder.losses.route_permission import route_permission_loss


class NativeEncoderFixture(nn.Module):
    """Native five-output/hook interface with a small deterministic CPU trunk."""
    def __init__(self):
        super().__init__()
        self.return_feature = False
        self.backbone = nn.Sequential(nn.Linear(12, 256), nn.Dropout(.5))
        self.traffic_light_pred_head = nn.Linear(256, 2)
        self.stop_sign_head = nn.Linear(256, 2)
        self.traffic_pred_head = nn.Linear(256, 32)
        self.waypoints_generator = nn.Linear(256, 10)
        self.calls = 0
        self.fail_after_light = False
        self.skip_light = False

    def forward(self, inputs):
        self.calls += 1
        token = self.backbone(inputs['rgb_front'].flatten(1))
        light = (torch.zeros(len(token), 2) if self.skip_light
                 else self.traffic_light_pred_head(token))
        if self.fail_after_light:
            raise RuntimeError('Native forward failed')
        return (self.traffic_pred_head(token).reshape(-1, 4, 8),
                self.waypoints_generator(token).reshape(-1, 5, 2), light,
                self.stop_sign_head(token), token[:, None, :].expand(-1, 4, -1))


def sensors(batch=3):
    result = {name: torch.randn(batch, 3, 2, 2) for name in SENSOR_INPUT_KEYS if name.startswith('rgb_')}
    result.update(lidar=torch.randn(batch, 4, 4), num_points=torch.full((batch,), 4),
                  velocity=torch.ones(batch, 1), target_point=torch.ones(batch, 2))
    return result


class RoutePermissionHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(178)
        torch.set_num_threads(1)

    def test_indices_and_onehot_match_and_planned_movement_can_change_permission(self):
        head = RoutePermissionHead()
        feature = torch.randn(3, 256)
        indices = torch.tensor([0, 1, 2])
        expected = head(feature, indices)
        actual = head(feature, torch.eye(3))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (3, 2))
        # Same sensor evidence may legitimately mean straight=allow,left=deny.
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.network[0].weight[0, 256] = 1.
            head.network[0].weight[1, 257] = 1.
            head.network[2].weight[1, 0] = 1.
            head.network[2].weight[0, 1] = 1.
        logits = head(torch.zeros(2, 256), torch.tensor([0, 1]))
        self.assertEqual(logits.argmax(1).tolist(), [1, 0])

    def test_unknown_or_soft_movement_is_never_silently_valid(self):
        head = RoutePermissionHead()
        for movement in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([1.]),
                         torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([[.5, .5, 0.]]),
                         torch.tensor([[float('nan'), 0., 1.]])):
            with self.subTest(movement=movement), self.assertRaises((ValueError, TypeError)):
                head(torch.zeros(1, 256), movement)

    def test_single_forward_preserves_all_native_outputs_bit_exactly(self):
        base = NativeEncoderFixture().eval()
        inputs = sensors()
        expected = base(inputs)
        wrapper = RoutePermissionEncoder(base).eval()
        before_calls = base.calls
        output = wrapper(inputs, torch.tensor([0, 1, 2]))
        self.assertEqual(base.calls, before_calls + 1)
        self.assertEqual(set(output), {'base_outputs', 'route_permission'})
        self.assertEqual(output['route_permission'].shape, (3, 2))
        for observed, original in zip(output['base_outputs'], expected):
            torch.testing.assert_close(observed, original, rtol=0, atol=0)
        self.assertEqual(len(base.traffic_light_pred_head._forward_pre_hooks), 0)
        # Base parameters are registered only under base_encoder, with no alias
        # to the old light head introduced by the feature hook.
        all_ids = [id(parameter) for parameter in wrapper.parameters()]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(wrapper.state_dict()), len(base.state_dict()) + len(wrapper.permission_head.state_dict()))

    def test_freeze_keeps_base_eval_unchanged_and_only_head_gets_gradients(self):
        base = NativeEncoderFixture()
        wrapper = RoutePermissionEncoder(base).train()
        self.assertFalse(base.training)
        self.assertTrue(wrapper.permission_head.training)
        before = copy.deepcopy(base.state_dict())
        optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=.01)
        output = wrapper(sensors(), torch.tensor([0, 1, 2]))
        loss, _ = route_permission_loss(output['route_permission'], torch.tensor([0, 1, 0]),
                                       torch.ones(3, dtype=torch.bool), torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertFalse(any(parameter.grad is not None for parameter in base.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum())
                            for p in wrapper.permission_head.parameters()))
        optimizer.step()
        for name, parameter in base.state_dict().items():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_explicit_joint_mode_propagates_permission_gradient_into_visual_trunk(self):
        base = NativeEncoderFixture()
        wrapper = RoutePermissionEncoder(base, freeze_encoder=False).eval()
        logits = wrapper(sensors(), torch.tensor([0, 1, 2]))['route_permission']
        loss, _ = route_permission_loss(logits, torch.tensor([0, 1, 0]),
                                       torch.ones(3, dtype=torch.bool), torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertGreater(float(base.backbone[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(wrapper.permission_head.network[0].weight.grad.abs().sum()), 0.)
        self.assertIsNone(base.traffic_light_pred_head.weight.grad)
        self.assertIsNone(base.traffic_pred_head.weight.grad)

    def test_gt_fields_are_rejected_and_unknown_movement_fails_before_sensor_forward(self):
        base = NativeEncoderFixture()
        wrapper = RoutePermissionEncoder(base)
        for name in ('raw_signal_bits', 'light_id', 'distance_to_stopline',
                     'route_permission', 'traffic_light'):
            inputs = sensors()
            inputs[name] = torch.zeros(3)
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, 'no GT'):
                wrapper(inputs, torch.tensor([0, 1, 2]))
        with self.assertRaisesRegex(ValueError, 'Unknown planned movement'):
            wrapper(sensors(), torch.tensor([0, -1, 2]))
        self.assertEqual(base.calls, 0)

    def test_hook_is_removed_after_native_failure_and_missing_light_call(self):
        base = NativeEncoderFixture()
        wrapper = RoutePermissionEncoder(base)
        base.fail_after_light = True
        with self.assertRaisesRegex(RuntimeError, 'Native forward failed'):
            wrapper(sensors(), torch.tensor([0, 1, 2]))
        self.assertEqual(len(base.traffic_light_pred_head._forward_pre_hooks), 0)
        base.fail_after_light = False
        base.skip_light = True
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            wrapper(sensors(), torch.tensor([0, 1, 2]))
        self.assertEqual(len(base.traffic_light_pred_head._forward_pre_hooks), 0)

    def test_feature_only_base_mode_is_rejected(self):
        base = NativeEncoderFixture()
        base.return_feature = True
        with self.assertRaisesRegex(ValueError, 'return_feature=False'):
            RoutePermissionEncoder(base)

    def test_head_sidecar_roundtrip_has_no_base_weights_and_checks_base_sha(self):
        head = RoutePermissionHead(hidden_dim=32).double().eval()
        sha = 'a' * 64
        payload = build_route_permission_checkpoint(head, sha)
        self.assertEqual(set(payload['state_dict']), set(head.state_dict()))
        self.assertFalse(any('base_encoder' in name for name in payload['state_dict']))
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        restored = load_route_permission_checkpoint(torch.load(buffer, weights_only=True), sha).eval()
        features = torch.randn(3, 256, dtype=torch.float64)
        movement = torch.tensor([0, 1, 2])
        torch.testing.assert_close(restored(features, movement), head(features, movement), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            load_route_permission_checkpoint(payload, 'b' * 64)
        damaged = copy.deepcopy(payload)
        damaged['state_dict']['network.0.weight'] = torch.zeros(32, 256)
        with self.assertRaisesRegex(ValueError, 'Invalid permission head tensor'):
            load_route_permission_checkpoint(damaged, sha)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            build_route_permission_checkpoint(head, 'not_a_hash')


class RoutePermissionLossTests(unittest.TestCase):
    def test_red_plus_arrow_supervises_permission_even_when_native_color_is_unknown(self):
        # These metadata values deliberately never enter the model/loss API.
        metadata = {'raw_signal_bits': 33, 'traffic_light': -1,
                    'route_permission': {'target': True, 'valid': True}}
        logits = torch.tensor([[2., -1.]], requires_grad=True)
        loss, terms = route_permission_loss(logits,
            torch.tensor([int(metadata['route_permission']['target'])]),
            torch.tensor([metadata['route_permission']['valid']]), torch.tensor([1]))
        loss.backward()
        self.assertEqual(int(terms['route_permission_valid_count']), 1)
        self.assertGreater(float(logits.grad[0, 0]), 0.)
        self.assertLess(float(logits.grad[0, 1]), 0.)

    def test_empty_mask_ignores_nan_logits_labels_and_unknown_movement(self):
        logits = torch.full((2, 2), float('nan'), requires_grad=True)
        loss, terms = route_permission_loss(logits, torch.full((2,), float('nan')),
                                           torch.zeros(2, dtype=torch.bool), torch.tensor([-1, -1]))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(int(terms['route_permission_valid_count']), 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)

    def test_masked_rows_cannot_change_valid_loss_or_receive_gradient(self):
        logits = torch.tensor([[1., 2.], [float('nan'), float('nan')]], requires_grad=True)
        loss, _ = route_permission_loss(logits, torch.tensor([1., float('nan')]),
                                       torch.tensor([True, False]), torch.tensor([1, -1]))
        expected = nn.functional.cross_entropy(logits[:1], torch.tensor([1]))
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[1], torch.zeros(2), rtol=0, atol=0)

    def test_active_unknown_movement_and_invalid_label_are_errors(self):
        for labels, movement in ((torch.tensor([1]), torch.tensor([-1])),
                                 (torch.tensor([-1]), torch.tensor([1])),
                                 (torch.tensor([float('nan')]), torch.tensor([1]))):
            with self.assertRaises(ValueError):
                route_permission_loss(torch.zeros(1, 2), labels, torch.tensor([True]), movement)


if __name__ == '__main__':
    unittest.main()
