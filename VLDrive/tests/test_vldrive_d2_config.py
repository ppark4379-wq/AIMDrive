"""The default training config's D2 sampler and attached mean denominator."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from models.vldrive import VLDriveTrainingConfig


def test_default_training_config_uses_d2():
    root = Path(__file__).resolve().parents[1]
    d2 = json.loads((root / 'configs/vldrive_train.json').read_text())
    expected = dict(training_variant='morai_repair_v1', reconstruction_reduction='mean',
                    detach_reconstruction_loss_mask=False, retention_logit_weight=0.,
                    selector_task_gradient_mode='legacy', inference_selection_mode='argmax')
    assert {key: d2['model'][key] for key in expected} == expected
    VLDriveTrainingConfig(**d2['model']).validate()


def connector():
    return QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=8, num_visual_tokens=4, language_dim=16, query_count=2,
        memory_size=2, qformer_hidden_size=16, qformer_num_layers=1,
        qformer_num_attention_heads=2, qformer_intermediate_size=32,
        reconstruction_layers=1, reconstruction_heads=2,
        reconstruction_reduction='mean', detach_reconstruction_loss_mask=False,
        retention_logit_weight=0., selector_task_gradient_mode='legacy',
        inference_selection_mode='argmax'))


def test_d2_train_uses_real_hard_gumbel_despite_eval_argmax():
    net = connector().train()
    assert net.ccdp.temperature == net.ccdp.gumbel_noise_scale == 1.
    with patch('models.connector.ccdp.F.gumbel_softmax', wraps=F.gumbel_softmax) as sampler:
        output = net(torch.randn(1, 3, 4, 8), compute_reconstruction=True)
    assert sampler.call_count > 0
    for call in sampler.call_args_list:
        assert call.kwargs['tau'] == 1. and call.kwargs['hard'] is True
    assert torch.isfinite(output['reconstruction_loss'])


def test_d2_mean_has_per_frame_quotient_rule_mask_gradient():
    net = connector().eval()

    class Zero(torch.nn.Module):
        def forward(self, value):
            return value * 0.

    net.reconstruction_transformer = Zero()
    features = torch.arange(64, dtype=torch.float32).reshape(1, 2, 4, 8) / 8 + 1
    mask = torch.tensor([[[1., 0., 0., 0.], [1., 1., 1., 0.]]], requires_grad=True)
    loss = net(features, hard_mask=mask, compute_reconstruction=True)['reconstruction_loss']
    energy = features.square().mean(-1)
    dropped_count = (1 - mask.detach()).sum(-1)
    frame_mean = ((1 - mask.detach()) * energy).sum(-1) / dropped_count
    expected_gradient = (frame_mean[..., None] - energy) / dropped_count[..., None] / 2
    torch.testing.assert_close(loss, frame_mean.mean())
    torch.testing.assert_close(torch.autograd.grad(loss, mask)[0], expected_gradient,
                               rtol=1e-5, atol=1e-5)
