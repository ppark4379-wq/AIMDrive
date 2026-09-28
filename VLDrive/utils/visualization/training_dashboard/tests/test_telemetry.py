"""Small synthetic fixtures only; never load or train a real VLDrive checkpoint."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from training.common.runtime.reproducibility import seed_all
from training.connector_llm.training import train
from utils.visualization.training_dashboard.telemetry import TrainingTelemetry


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(precision='fp32',target_retention=.3)
        self.connector=QFormerMEFAConnector(QFormerMEFAConfig(visual_dim=8,num_visual_tokens=6,
            language_dim=16,query_count=2,memory_size=2,qformer_hidden_size=8,qformer_num_layers=1,
            qformer_num_attention_heads=2,qformer_intermediate_size=16,reconstruction_heads=2,
            reconstruction_layers=1,reconstruction_reduction='mean'))
        self.masks=[]

    def forward(self,item):
        output=self.connector(item['model_inputs']['visual_feature'],compute_reconstruction=True)
        self.masks.append(output['selection_mask'].detach().cpu().clone())
        return {'loss':output['language_tokens'].square().mean()+output['cycle_loss']}

    def gradient_groups(self):
        return dict(selector='connector.ccdp.',mefa='connector.temporal_encoder.',
                    qformer='connector.qformer.',connector_projection='connector.qformer_to_language.')


class TelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.previous_threads)

    def compare(self,device):
        seed_all(607); initial=Fixture().to(device)
        data=[{'model_inputs':{'visual_feature':torch.randn(3,6,8)}} for _ in range(3)]
        results=[]
        for enabled in (False,True):
            model=copy.deepcopy(initial); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            seed_all(800)
            with tempfile.TemporaryDirectory() as tmp:
                log=Path(tmp)/'training.jsonl'
                train(model,optimizer,data,[0,1,2],start_update=0,updates=3,schedule_updates=6,
                    seed=607,device=device,log_path=log,lr=1e-4,telemetry_enabled=enabled,gradient_log_interval=1)
                rows=[json.loads(s) for s in log.read_text().splitlines()]
            results.append((model,rows,torch.get_rng_state().clone(),
                            torch.cuda.get_rng_state().clone() if device=='cuda' else None))
        a,b=results
        self.assertTrue(torch.equal(a[2],b[2]))
        if device=='cuda': self.assertTrue(torch.equal(a[3],b[3]))
        for name,value in a[0].state_dict().items(): torch.testing.assert_close(value,b[0].state_dict()[name],rtol=0,atol=0)
        for x,y in zip(a[0].masks,b[0].masks): self.assertTrue(torch.equal(x,y))
        self.assertEqual([r['loss'] for r in a[1]],[r['loss'] for r in b[1]])
        self.assertEqual([r['grad_norm'] for r in a[1]],[r['grad_norm'] for r in b[1]])
        self.assertEqual(len(b[0].connector._forward_hooks),0)
        self.assertTrue(all(r['module_gradients']['selector']>0 for r in b[1]))
        self.assertTrue(all('p_keep_std' in r for r in b[1]))
        self.assertTrue(all('p_keep_std' not in r for r in a[1]))

    def test_cpu_bitwise_rng_masks_losses_weights_unchanged(self): self.compare('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_bitwise_rng_masks_losses_weights_unchanged(self): self.compare('cuda')

    def test_hook_is_removed_on_failure_and_observation_never_breaks_forward(self):
        model=Fixture()
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with TrainingTelemetry(model,'cpu') as telemetry:
                telemetry._observe(model.connector,None,{})
                self.assertIn('telemetry_error',telemetry.values())
                raise RuntimeError('test failure')
        self.assertEqual(len(model.connector._forward_hooks),0)


if __name__=='__main__': unittest.main()
