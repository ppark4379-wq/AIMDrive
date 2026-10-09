"""Telemetry observes actual joint forwards without changing predictions or gradients."""
import copy
from types import SimpleNamespace

import pytest
import torch

from .test_situation_instruction import model as language_fixture
from utils.visualization.training_dashboard.telemetry import TrainingTelemetry


def test_last_frame_only_ego_error_and_zone_grouping():
    telemetry = TrainingTelemetry(torch.nn.Module(), 'cpu')
    # Older frames intentionally have huge errors and conflicting zone IDs.
    prediction = torch.full((3, 3, 5, 2), 999.)
    prediction[:, -1] = torch.tensor([3., 4.])
    prediction[1, -1] *= 2
    telemetry._observe_language(SimpleNamespace(training=True), (),
        SimpleNamespace(predicted_waypoints=prediction.reshape(3, 3, 10), token_metrics=None))
    item = dict(model_inputs=dict(zone_ids=torch.tensor([[4,4,1],[4,4,1],[4,4,-1]])),
                training_targets=dict(future_trajectory=torch.zeros(3,3,5,2)))
    result = telemetry.values(item=item)
    assert result['train_ade_m'] == pytest.approx(20/3)
    assert result['train_fde_m'] == pytest.approx(20/3)
    assert result['trajectory_samples'] == 3
    assert result['zone_metrics'] == {'1':dict(samples=2, ade_sum_m=15., fde_sum_m=15.)}
    assert telemetry.prediction is None


@pytest.mark.parametrize('enabled_inputs', [False, True])
def test_actual_forward_token_accounting_and_bitwise_training_parity(monkeypatch, enabled_inputs):
    torch.manual_seed(9)
    wrapper = torch.nn.Module()
    wrapper.language_model = language_fixture(monkeypatch, path=enabled_inputs,
                                              situation=enabled_inputs, zone=enabled_inputs)
    wrapper.gradient_groups = lambda: dict(navigation_mlp='language_model.navigation_mlp.',
                                          zone_embedding='language_model.zone_embedding.') if enabled_inputs else {}
    visual = torch.randn(2, 3, 4, 32)
    mask = torch.tensor([[[1,1,1,0]]*3, [[1,1,0,0]]*3], dtype=torch.bool)
    target = torch.randn(2, 3, 5, 2)
    args = dict(visual_attention_mask=mask, predict_per_frame=True)
    if enabled_inputs:
        args.update(zone_ids=torch.tensor([[1,1,1],[-1,-1,-1]]),
            situation_instructions=[['Continue driving.', None, 'Remain stopped.'], [None,None,None]],
            path_features=dict(ego_relation=torch.randn(2,3,3),path_context=torch.randn(2,3,5),
                               path_valid=torch.ones(2,3,dtype=torch.bool)))
    item = dict(model_inputs={'zone_ids':args['zone_ids']} if enabled_inputs else {},
                training_targets=dict(future_trajectory=target))
    initial = copy.deepcopy(wrapper.state_dict())
    results = []
    for monitor in (False, True):
        wrapper.load_state_dict(initial)
        wrapper.zero_grad(set_to_none=True)
        torch.manual_seed(43)
        with TrainingTelemetry(wrapper, 'cpu', enabled=monitor, gradient_interval=1) as telemetry:
            out = wrapper.language_model(['Follow.', 'Follow the path.'], visual, **args)
            loss = (out.predicted_waypoints.reshape(2,3,5,2) - target).abs().mean()
            loss.backward()
            values = telemetry.values(item=item)
            norms = telemetry.gradients(1)
            if monitor:
                expected = torch.linalg.vector_norm(out.predicted_waypoints.reshape(2,3,5,2)[:,-1].detach()-target[:,-1],dim=-1)
                assert values['train_ade_m'] == pytest.approx(expected.mean().item())
                assert values['train_fde_m'] == pytest.approx(expected[:,-1].mean().item())
                assert sum(values[k] for k in ('navigation_tokens','situation_tokens','zone_tokens','visual_tokens')) == values['llm_valid_tokens']
                assert values['llm_padded_tokens'] == out.sequence.inputs_embeds.shape[1]
                assert values['llm_valid_tokens'] == out.sequence.attention_mask.float().sum(1).mean()
                assert values['visual_tokens'] == 7.5
                assert values['navigation_tokens'] == (9. if enabled_inputs else 3.)
                assert values['zone_tokens'] == (1.5 if enabled_inputs else 0.)
                assert values['situation_tokens'] == (4. if enabled_inputs else 0.)
                if enabled_inputs:
                    assert norms['zone_embedding'] > 0 and norms['navigation_mlp'] > 0
            else:
                assert values == {} and out.token_metrics is None
            results.append((out.predicted_waypoints.detach().clone(),
                            {n:p.grad.clone() for n,p in wrapper.named_parameters() if p.grad is not None},
                            torch.get_rng_state().clone()))
        assert len(wrapper.language_model._forward_hooks) == 0
        assert not hasattr(wrapper.language_model, 'collect_training_metrics')
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][2], results[1][2])
    for name, gradient in results[0][1].items():
        assert torch.equal(gradient, results[1][1][name])


def test_language_observer_cleanup_after_exception(monkeypatch):
    wrapper = torch.nn.Module()
    wrapper.language_model = language_fixture(monkeypatch, path=False, situation=False)
    with pytest.raises(RuntimeError):
        with TrainingTelemetry(wrapper, 'cpu'):
            raise RuntimeError('interrupted')
    assert not wrapper.language_model._forward_hooks
    assert not hasattr(wrapper.language_model, 'collect_training_metrics')
