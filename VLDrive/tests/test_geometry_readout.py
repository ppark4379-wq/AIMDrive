import unittest
import torch
from training.vision_encoder.losses.morai_geometry_readout import (
    GeometryReadoutDelta, geometry_readout_loss,
)


class GeometryReadoutTests(unittest.TestCase):
    def test_optimizer_cannot_change_confidence_speed_or_upstream(self):
        torch.manual_seed(7)
        x = torch.randn(1, 2500, 64, requires_grad=True)
        last = torch.nn.Linear(64, 8)
        base = last(x)
        readout = GeometryReadoutDelta()
        self.assertTrue(torch.equal(readout(x, base), base))
        optimizer = torch.optim.AdamW(readout.parameters(), lr=.001, weight_decay=.1)
        old_weight = last.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            pred = readout(x, base)
            pred[..., 1:7].square().mean().backward()
            optimizer.step()
        pred = readout(x, base)
        self.assertTrue(torch.equal(pred[..., [0, 7]], base[..., [0, 7]]))
        self.assertIsNone(x.grad)
        self.assertIsNone(last.weight.grad)
        self.assertTrue(torch.equal(old_weight, last.weight))
        weight, bias = readout.fold_into(last)
        self.assertTrue(torch.equal(weight[[0, 7]], last.weight[[0, 7]]))
        self.assertTrue(torch.equal(bias[[0, 7]], last.bias[[0, 7]]))
        folded = torch.nn.functional.linear(x.detach(), weight, bias)
        torch.testing.assert_close(folded, pred, atol=1e-6, rtol=1e-5)

    def test_positive_geometry_moves_toward_gt(self):
        target = torch.zeros(1, 2500, 8)
        target[0, 0] = torch.tensor([1., 0., 0., 1., 0., 2./7, 1./4, 0.])
        pred = target.clone(); pred[0, 0, 1] += .2; pred[0, 0, 5] += .1
        pred.requires_grad_(True)
        ids = torch.full((1, 2500), -1, dtype=torch.long); ids[0, 0] = 0
        loss, _ = geometry_readout_loss(pred, target, ids, pred.detach(), ids >= 0)
        loss.backward()
        self.assertGreater(float(pred.grad[0, 0, 1]), 0.)
        self.assertGreater(float(pred.grad[0, 0, 5]), 0.)
        self.assertEqual(float(pred.grad[..., [0, 7]].abs().max()), 0.)

    def test_deferred_gt_does_not_supervise(self):
        target = torch.zeros(1, 2500, 8)
        target[0, 0] = torch.tensor([1., 0., 0., 1., 0., 2./7, 1./4, 0.])
        pred = (target + .02).requires_grad_(True)
        ids = torch.full((1, 2500), -1, dtype=torch.long); ids[0, 0] = 0
        loss, _ = geometry_readout_loss(pred, target, ids, pred.detach(), ids < -1)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(pred.grad.abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
