"""An interrupted validation must never discard completed training updates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from training.vision_encoder.recipes.transfer import evaluate_and_checkpoint

class RecoveryTest(unittest.TestCase):
    def test_validation_failure_keeps_new_weights_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint-last.pth.tar'
            payload=dict(state_dict={'weight':torch.tensor([3.])},optimizer={'step':1000},steps=1000)
            with patch('training.vision_encoder.recipes.transfer.evaluate',side_effect=RuntimeError('loader stalled')):
                with self.assertRaisesRegex(RuntimeError,'loader stalled'):
                    evaluate_and_checkpoint(p,payload,None,None,'cpu')
            restored=torch.load(p,weights_only=True)
            self.assertEqual(restored['state_dict']['weight'].item(),3.)
            self.assertEqual(restored['optimizer']['step'],1000)
            self.assertFalse(restored['validation_completed'])
            self.assertIsNone(restored['metrics'])

    def test_validation_success_publishes_metrics_after_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint-last.pth.tar'
            def evaluate(*a,**kw):
                self.assertFalse(torch.load(p,weights_only=True)['validation_completed'])
                return {'loss':.1}
            with patch('training.vision_encoder.recipes.transfer.evaluate',side_effect=evaluate):
                metrics=evaluate_and_checkpoint(p,{'steps':10},None,None,'cpu')
            self.assertEqual(metrics,{'loss':.1})
            self.assertTrue(torch.load(p,weights_only=True)['validation_completed'])
if __name__=='__main__':unittest.main()
