"""Backward-only CCDP candidate checks, not a claim of learned driving quality.

No optimizer updates or pretrained-asset downloads. The task-gradient check
uses the real VLDrive forward and L1 waypoint loss with a tiny random LLaMA.
"""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from lavis.models.vldrive.ddia_attention import install_ddia_attention
from lavis.models.vldrive.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
from lavis.models.vldrive import vldrive_model


@pytest.fixture(autouse=True)
def cpu_threads_one():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def make_connector():
    return QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=256, num_visual_tokens=106, language_dim=64, query_count=32,
        memory_size=2, qformer_hidden_size=64, qformer_num_layers=1,
        qformer_num_attention_heads=4, qformer_intermediate_size=128,
        reconstruction_layers=1, reconstruction_heads=4,
    ))


def candidates():
    torch.manual_seed(121)
    baseline = make_connector().train()
    candidate = copy.deepcopy(baseline)
    candidate.config.selector_task_gradient_mode = 'retained_output_st'
    candidate.config.validate()
    return baseline, candidate


def paired_outputs(baseline, candidate, features, reconstruct=True):
    torch.manual_seed(122)
    first = baseline(features, compute_reconstruction=reconstruct)
    torch.manual_seed(122)
    second = candidate(features, compute_reconstruction=reconstruct)
    return first, second


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_surrogate_exact_identity_and_known_selected_only_derivative(dtype):
    torch.manual_seed(123)
    language = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=dtype, requires_grad=True)
    indices = torch.tensor([[0, 2, -1], [1, 2, 3]])
    valid = indices >= 0
    result = QFormerMEFAConnector._retained_output_task_surrogate(language, mask, indices, valid)
    torch.testing.assert_close(result, language, rtol=0, atol=0)
    cotangent = torch.randn_like(result)
    language_gradient, mask_gradient = torch.autograd.grad((result*cotangent).sum(), (language, mask))
    torch.testing.assert_close(language_gradient, cotangent, rtol=0, atol=0)
    expected = torch.zeros_like(mask)
    values = (language.detach()*cotangent).sum(-1) * valid
    expected.scatter_add_(1, indices.clamp_min(0), values)
    torch.testing.assert_close(mask_gradient, expected, rtol=0, atol=0)
    assert not mask_gradient[mask.detach() == 0].any()


@pytest.mark.parametrize('training', [True, False])
def test_entire_connector_forward_is_exactly_unchanged(training):
    baseline, candidate = candidates()
    baseline.train(training)
    candidate.train(training)
    features = torch.randn(1, 2, 106, 256)
    original, changed = paired_outputs(baseline, candidate, features)
    for name in ('language_tokens', 'selection_mask', 'selected_indices', 'selected_valid_mask',
                 'selected_count_per_frame', 'initial_query_embeddings', 'qformer_hidden',
                 'enhanced_memory', 'pruning_loss', 'reconstruction_loss', 'cycle_loss'):
        torch.testing.assert_close(original[name], changed[name], rtol=0, atol=0)
    assert original['language_tokens'].shape == changed['language_tokens'].shape
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    assert candidate.config.inference_selection_mode == 'argmax'


@pytest.mark.parametrize('loss_name', ['pruning_loss', 'reconstruction_loss', 'cycle_loss'])
def test_auxiliary_gradients_remain_exactly_original(loss_name):
    baseline, candidate = candidates()
    features = torch.randn(1, 2, 106, 256)
    original, changed = paired_outputs(baseline, candidate, features)
    first = torch.autograd.grad(original[loss_name], tuple(baseline.parameters()), allow_unused=True)
    second = torch.autograd.grad(changed[loss_name], tuple(candidate.parameters()), allow_unused=True)
    assert len(first) == len(second)
    for old, new in zip(first, second):
        if old is None:
            assert new is None
        else:
            torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_actual_ddia_waypoint_l1_reaches_selector_without_changing_prediction(monkeypatch):
    baseline, candidate = candidates()
    torch.manual_seed(124)
    tiny = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048))
    install_ddia_attention(tiny)
    monkeypatch.setattr(vldrive_model, 'load_tinyllama',
        lambda *args, **kwargs: SimpleNamespace(model=tiny, tokenizer=None))
    driving_model = vldrive_model.VLDriveModel(visual_feature_dim=64, use_visual_projection=False,
                                             precision='fp32').eval()
    instruction = {'input_ids': torch.tensor([[1, 2, 3]]),
                   'attention_mask': torch.ones(1, 3, dtype=torch.long)}
    features = torch.randn(1, 2, 106, 256)
    targets = {'future_trajectory': torch.tensor([[[[1., .2], [2., .5], [3., 1.], [4., 1.5], [5., 2.]],
                                                 [[1., .2], [2., .5], [3., 1.], [4., 1.5], [5., 2.]]]])}
    original, changed = paired_outputs(baseline, candidate, features, reconstruct=False)
    predictions, gradients, losses = [], [], []
    for connector, connected in ((baseline, original), (candidate, changed)):
        predicted = driving_model(instruction, connected['language_tokens'],
                                  connected['selected_valid_mask'], predict_per_frame=True)
        output = dict(trajectory=predicted.predicted_waypoints.reshape(1, 2, 5, 2),
                      cycle_loss=connected['cycle_loss'], pruning_loss=connected['pruning_loss'],
                      reconstruction_loss=connected['reconstruction_loss'])
        waypoint_loss = VLDriveTrainingModel.loss(output, targets)['waypoint_loss']
        gradients.append(torch.autograd.grad(waypoint_loss, tuple(connector.parameters()), allow_unused=True))
        predictions.append(output['trajectory'].detach())
        losses.append(waypoint_loss.detach())
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    norms = [0., 0.]
    for (name, _), old, new in zip(baseline.named_parameters(), *gradients):
        if name.startswith('ccdp.'):
            for index, gradient in enumerate((old, new)):
                if gradient is not None:
                    norms[index] += float(gradient.square().sum())
        elif name.startswith('visual_input_norm.'):
            # CCDP consumes normalized input: its additional mask derivative
            # also reaches this upstream norm by the unchanged chain rule.
            assert old is not None and new is not None
            assert torch.isfinite(new).all()
        elif old is None:
            assert new is None
        else:
            torch.testing.assert_close(old, new, rtol=0, atol=0)
    norms = [value**.5 for value in norms]
    assert norms[1] > norms[0] * 10
    assert norms[1] > 0 and all(torch.isfinite(torch.tensor(norms)))
    print(json.dumps(dict(check='actual_tiny_llama_ddia_waypoint_l1_backward',
        tokens=106, channels=256, frames=2, optimizer_updates=0,
        same_forward_max_delta=float((predictions[0]-predictions[1]).abs().max()),
        waypoint_loss=float(losses[0]), original_selector_gradient_l2=norms[0],
        candidate_selector_gradient_l2=norms[1],
        caveat='random tiny model; selected-only surrogate; no evidence of MORAI convergence')))


def test_training_variant_requires_explicit_paper_objective_probe():
    VLDriveTrainingConfig().validate()
    VLDriveTrainingConfig(training_variant='paper_objective_st_probe',
                          selector_task_gradient_mode='retained_output_st').validate()
    with pytest.raises(ValueError, match='requires explicit'):
        VLDriveTrainingConfig(selector_task_gradient_mode='retained_output_st').validate()
    with pytest.raises(ValueError, match='requires explicit'):
        VLDriveTrainingConfig(training_variant='paper_objective_st_probe').validate()
    with pytest.raises(ValueError):
        VLDriveTrainingConfig(training_variant='paper_objective_st_probe',
            selector_task_gradient_mode='retained_output_st', inference_selection_mode='topk').validate()
    with pytest.raises(ValueError):
        QFormerMEFAConfig(selector_task_gradient_mode='retained_output_st',
                          detach_reconstruction_loss_mask=True).validate()
