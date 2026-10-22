import unittest
import torch
from models.ve.signal_state import (SignalStateHead, SIGNAL_CODES, extract_signal_feature,
                                    encode_checkpoint, decode_checkpoint, decode_signal)
from tests.test_route_permission_head import NativeEncoderFixture, sensors
from data.src.loaders.morai_route_permission import permission_target
from training.vision_encoder.recipes.signal_state import signal_target, metrics


class SignalStateTests(unittest.TestCase):
    def test_codes_abstention_and_invalid(self):
        rows = decode_signal(torch.eye(3) * 20)
        self.assertEqual([r['code'] for r in rows], list(SIGNAL_CODES))
        self.assertIsNone(decode_signal(torch.zeros(1, 3))[0]['code'])
        self.assertIsNone(decode_signal(torch.full((1, 3), float('nan')))[0]['code'])

    def test_native_outputs_unchanged_and_only_sidecar_updates(self):
        torch.manual_seed(42)
        base = NativeEncoderFixture().eval().requires_grad_(False)
        inputs = sensors(); expected = base(inputs)
        before = {k: v.clone() for k, v in base.state_dict().items()}
        token, observed = extract_signal_feature(base, inputs)
        for a, b in zip(expected, observed):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        head = SignalStateHead(); opt = torch.optim.AdamW(head.parameters())
        loss = torch.nn.functional.cross_entropy(head(token), torch.tensor([0, 1, 2]))
        loss.backward(); opt.step()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        for k, v in base.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertFalse(base.traffic_light_pred_head._forward_pre_hooks)
        with self.assertRaises(ValueError):
            extract_signal_feature(base, dict(inputs, raw_signal_bits=torch.tensor([1])))
        base.fail_after_light = True
        with self.assertRaises(RuntimeError): extract_signal_feature(base, inputs)
        self.assertFalse(base.traffic_light_pred_head._forward_pre_hooks)

    def test_perception_forward_hook_is_single_forward_and_removed(self):
        base = NativeEncoderFixture().eval()
        calls = []
        def perception(model, inputs):
            calls.append(1)
            return model(inputs)
        token, _ = extract_signal_feature(base, sensors(), perception)
        self.assertEqual(len(calls), 1)
        self.assertEqual(token.shape, (3, 256))
        self.assertFalse(base.traffic_light_pred_head._forward_pre_hooks)

    def test_checkpoint_identity_and_mapping(self):
        head = SignalStateHead(); payload = encode_checkpoint(head, 'a'*64)
        x = torch.randn(4, 256)
        torch.testing.assert_close(head(x), decode_checkpoint(payload, 'a'*64)(x))
        with self.assertRaises(ValueError): decode_checkpoint(payload, 'b'*64)
        payload['signal_codes'] = [1, 33, 16]
        with self.assertRaises(ValueError): decode_checkpoint(payload, 'a'*64)

    def test_native_binary_code_mapping_and_abstention(self):
        from models.ve.signal_state import decode_native_signal
        logits=torch.tensor([[20.,0.],[0.,20.],[0.,0.],[float('nan'),0.]])
        self.assertEqual([r['code'] for r in decode_native_signal(logits)],[1,16,None,None])
        with self.assertRaises(ValueError): decode_native_signal(torch.zeros(1,3))

    def test_live_perception_without_waypoint_target_keeps_strict_sensor_contract(self):
        base = NativeEncoderFixture().eval()
        full = sensors()
        live = {k: v for k, v in full.items() if k != 'target_point'}
        def perception(model, inputs):
            self.assertNotIn('target_point', inputs)
            return model(inputs)
        expected, _ = extract_signal_feature(base, full)
        observed, _ = extract_signal_feature(base, live, perception)
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            extract_signal_feature(base, live)  # Full waypoint forward still requires target.
        for key in live:
            with self.assertRaises(ValueError):
                extract_signal_feature(base, {k:v for k,v in live.items() if k != key}, perception)
        for key in ('raw_signal_bits', 'traffic_light', 'route_permission'):
            with self.assertRaises(ValueError):
                extract_signal_feature(base, dict(live, **{key: torch.tensor([33])}), perception)
        self.assertFalse(base.traffic_light_pred_head._forward_pre_hooks)

    def test_missing_gt_cannot_fall_back_to_color(self):
        self.assertFalse(permission_target({'traffic_light': 1})['valid'])
        for bits in [None, True, 32, 48, 4]:
            self.assertEqual(signal_target(dict(valid=True,raw_signal_bits=bits)),-1)
        self.assertEqual(signal_target(dict(valid=False,raw_signal_bits=33)),-1)
        self.assertEqual([signal_target(dict(valid=True,raw_signal_bits=b)) for b in SIGNAL_CODES],[0,1,2])

    def test_wrong_left_allow_is_counted_even_when_aggregate_accuracy_looks_good(self):
        logits=torch.tensor([[0.,0.,20.],[0.,20.,0.],[0.,0.,20.]])
        meta=[dict(raw_signal_bits=b,stopline_distance_m=5,manifest_path='lap_val') for b in SIGNAL_CODES]
        report=metrics(logits,torch.tensor([0,1,2]),meta)['overall']
        self.assertEqual(report['false_left_allow_confidence_0_9'],1)
        self.assertEqual(report['confusion_gt_rows_pred_columns'],[[0,0,1],[0,1,0],[0,0,1]])
