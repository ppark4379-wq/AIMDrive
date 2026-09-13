"""Experimental CCDP stabilization: preserve learning, remove mask exemption.

The short synthetic fit is a regression test of the actual connector graph,
not evidence that MORAI driving or dynamic pruning already converges.
"""

import json

import pytest
import torch

from models.ccdp import CCDPTokenSparsifier
from models.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def connector(**overrides):
    config = dict(
        visual_dim=32, num_visual_tokens=12, language_dim=64,
        query_count=4, target_retention=.3, memory_size=2,
        qformer_hidden_size=64, qformer_num_layers=1,
        qformer_num_attention_heads=4, qformer_intermediate_size=128,
        reconstruction_layers=1, reconstruction_heads=4,
    )
    config.update(overrides)
    return QFormerMEFAConnector(QFormerMEFAConfig(**config))


class ZeroReconstruction(torch.nn.Module):
    def forward(self, inputs):
        # Retain a zero-valued graph path to isolate the loss-weight gradient.
        return inputs * 0.


def test_opt_in_removes_direct_exemption_reward_at_all_keep():
    torch.manual_seed(9)
    model = connector()
    model.reconstruction_transformer = ZeroReconstruction()
    features = torch.randn(1, 12, 32)
    languages = torch.randn(1, 12, 64)
    gradients = []
    for detached in (False, True):
        model.config.detach_reconstruction_loss_mask = detached
        mask = torch.ones(1, 12, requires_grad=True)
        result = model._reconstruct(
            features, languages, torch.arange(12)[None],
            torch.ones(1, 12, dtype=torch.bool), mask,
        )
        assert result['reconstruction_loss'].item() == 0.
        gradient, = torch.autograd.grad(result['reconstruction_loss'], mask)
        gradients.append(gradient)
    torch.testing.assert_close(gradients[0], -features.square().sum(-1))
    torch.testing.assert_close(gradients[1], torch.zeros_like(gradients[1]))
    print(json.dumps(dict(
        check='direct_mask_gradient_at_all_keep',
        legacy_l2=float(gradients[0].norm()), repaired_l2=float(gradients[1].norm()),
    )))


def test_repaired_reconstruction_still_trains_selector_through_eq8():
    torch.manual_seed(11)
    model = connector(detach_reconstruction_loss_mask=True, reconstruction_reduction='mean')
    output = model(torch.randn(1, 3, 12, 32), selection_mode='gumbel', compute_reconstruction=True)
    output['reconstruction_loss'].backward()
    grads = [p.grad for p in model.ccdp.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.square().sum()) for g in grads) > 0.
    assert model.reconstruction_transformer.layers[0].linear1.weight.grad.norm() > 0


def test_budgeted_inference_is_explicit_and_does_not_change_train_sampling():
    torch.manual_seed(17)
    model = connector(inference_selection_mode='topk')
    # Equal scores make the distinction deterministic: argmax selects none
    # and rescues one, top-k selects four, Gumbel has a stochastic count.
    for parameter in model.ccdp.score_mlp.parameters():
        parameter.data.zero_()
    features = torch.randn(1, 8, 12, 32)
    model.eval()
    assert torch.all(model(features, compute_reconstruction=False)['selected_count_per_frame'] == 4)
    assert torch.all(model(features, selection_mode='argmax', compute_reconstruction=False)['selected_count_per_frame'] == 1)
    model.train()
    counts = model(features, compute_reconstruction=False)['selected_count_per_frame']
    assert torch.any(counts != 4)


def test_full_size_synthetic_connector_learns_gumbel_retention():
    torch.manual_seed(31)
    model = connector(
        visual_dim=256, num_visual_tokens=106, query_count=32,
        detach_reconstruction_loss_mask=True, reconstruction_reduction='mean',
        inference_selection_mode='topk',
        retention_logit_weight=100.,
    ).train()
    features = torch.randn(1, 3, 106, 256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.)
    retention, probabilities, reconstruction = [], [], []
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        output = model(features, compute_reconstruction=True)
        if step == 0:
            for name, loss in (('pruning', output['pruning_loss'] * 10),
                               ('reconstruction', output['reconstruction_loss']),
                               ('logit', output['retention_logit_loss'] * model.config.retention_logit_weight)):
                g, = torch.autograd.grad(loss, model.ccdp.score_mlp[-1].bias, retain_graph=True)
                print(json.dumps(dict(initial_gradient_source=name, last_bias_gradient=g.tolist())))
        output['cycle_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 39, 79, 159, 239):
            print(json.dumps(dict(step=step + 1,
                sampled_retention=float(output['selector']['retention'].detach()),
                soft_retention=float(output['selector']['soft_retention'].detach()),
                pruning=float(output['pruning_loss'].detach()),
                reconstruction=float(output['reconstruction_loss'].detach()))))
        if step >= 220:
            retention.append(float(output['selector']['retention'].detach()))
            probabilities.append(float(output['selector']['soft_retention'].detach()))
            reconstruction.append(float(output['reconstruction_loss'].detach()))
    mean_retention = sum(retention) / len(retention)
    mean_probability = sum(probabilities) / len(probabilities)
    print(json.dumps(dict(
        check='synthetic_connector_fit', updates=240, tokens=106, channels=256,
        last20_sampled_retention=mean_retention, last20_soft_retention=mean_probability,
        last20_reconstruction_mean=sum(reconstruction) / len(reconstruction),
    )))
    assert .20 < mean_retention < .40
    assert .20 < mean_probability < .40
    assert all(0. < r < 1. for r in retention)
    assert all(torch.isfinite(torch.tensor(reconstruction)))


@pytest.mark.parametrize('logit_gap', [-30., 30.])
def test_logit_regularizer_can_recover_saturated_selector(logit_gap):
    torch.manual_seed(37)
    selector = CCDPTokenSparsifier(dim=256)
    features = torch.randn(3, 106, 256)
    selector.requires_grad_(False)
    score = selector.score_mlp[-1]
    with torch.no_grad():
        score.weight.zero_()
        score.bias.copy_(torch.tensor([-logit_gap / 2, logit_gap / 2]))
    score.bias.requires_grad_(True)
    optimizer = torch.optim.SGD([score.bias], lr=.5)
    output = selector(features)
    first_gradient, = torch.autograd.grad(output['retention_logit_loss'], score.bias)
    assert first_gradient.abs().min() > .29
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        selector(features)['retention_logit_loss'].backward()
        optimizer.step()
    output = selector(features)
    probability = float(output['soft_retention'].detach())
    print(json.dumps(dict(check='saturation_recovery', initial_gap=logit_gap,
                         first_bias_gradient=first_gradient.tolist(), final_probability=probability)))
    assert abs(probability - .3) < .005
