#!/usr/bin/env python3
"""CPU-only CCDP forward/backward audit; never optimize or alter checkpoints."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def projection_audit(projection, label):
    """A same-forward surrogate comparison, not a proposed production fix."""
    import torch
    from torch.nn import functional as F
    from lavis.models.vldrive.qformer_mefa import QFormerMEFAConnector
    projection = copy.deepcopy(projection).double()
    torch.manual_seed(992)
    channels = projection[0].normalized_shape[0]
    features = F.layer_norm(torch.randn(1, 106, channels, dtype=torch.float64), (channels,))
    mask = torch.zeros(1, 106, dtype=torch.float64)
    mask[:, ::3] = 1.
    mask.requires_grad_(True)
    packed, _, indices = QFormerMEFAConnector._pack_by_mask(features, mask)
    current = projection(packed)
    # On binary selected masks these two forwards are identical. Their
    # derivatives differ because a scalar gate before LayerNorm is normalized.
    post_projection_gate, _, _ = QFormerMEFAConnector._pack_by_mask(projection(features), mask)
    cotangent = torch.randn_like(current)
    current_gradient, = torch.autograd.grad((current*cotangent).sum(), mask, retain_graph=True)
    post_gradient, = torch.autograd.grad((post_projection_gate*cotangent).sum(), mask)
    step = 1e-3
    plus, minus = mask.detach().clone(), mask.detach().clone()
    plus[0, 0] += step
    minus[0, 0] -= step
    with torch.no_grad():
        plus_output = projection(QFormerMEFAConnector._pack_by_mask(features, plus)[0])
        minus_output = projection(QFormerMEFAConnector._pack_by_mask(features, minus)[0])
        numerical = float(((plus_output-minus_output)*cotangent).sum()/(2*step))
    return dict(label=label, selected_tokens=int(indices.shape[1]),
        same_forward_max_delta=float((current-post_projection_gate).abs().max().detach()),
        current_mask_gradient_l2=float(current_gradient.norm()),
        same_forward_gate_after_projection_gradient_l2=float(post_gradient.norm()),
        attenuation_ratio=float(current_gradient.norm()/post_gradient.norm()),
        dropped_mask_gradient_max=float(current_gradient[mask.detach()==0].abs().max()),
        first_kept_analytic_gradient=float(current_gradient[0, 0]),
        first_kept_central_difference=numerical,
        interpretation='Physical gather removes dropped query gradients; LayerNorm suppresses the retained scalar-gate gradient. Full trajectory causality is not established by this local VJP.')


def gumbel_audit():
    import torch
    from torch.nn import functional as F
    from lavis.models.vldrive.ccdp import CCDPTokenSparsifier
    logits = torch.tensor([[-10., 10.], [-1., 1.], [0., 0.], [1., -1.], [10., -10.]])
    probabilities = logits.softmax(-1)
    torch.manual_seed(12)
    noise = -torch.empty(8192, 5, 2).exponential_().log()
    correct = (logits+noise).argmax(-1)
    logprob = (logits.log_softmax(-1)+noise).argmax(-1)
    double_softmax = (probabilities+noise).argmax(-1)
    selector = CCDPTokenSparsifier(256)
    torch.manual_seed(77)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = selector(torch.randn(1, 106, 256))
    saturation = {}
    for dtype in (torch.float32, torch.bfloat16):
        gaps = torch.arange(0., 30., .125, dtype=torch.float32)
        scores = torch.stack((torch.zeros_like(gaps), gaps), -1).to(dtype).requires_grad_(True)
        probs = scores.softmax(-1)[..., 1]
        derivative, = torch.autograd.grad(probs.sum(), scores)
        first = int(torch.nonzero(probs==1.)[0])
        saturation[str(dtype)] = dict(first_gap_rounded_probability_one=float(gaps[first]),
            gradient_at_first_rounding=derivative[first].float().tolist(),
            note='Probability equal to one is not proof that every logit gradient is exactly zero.')
    return dict(keep_probability=probabilities[:, 1].tolist(),
        logits_sampling_keep_fraction=correct.float().mean(0).tolist(),
        logits_vs_logprob_mask_disagreements=int((correct!=logprob).sum()),
        treating_probabilities_as_pytorch_logits_keep_fraction=double_softmax.float().mean(0).tolist(),
        cpu_autocast_dtypes={key: str(output[key].dtype) for key in ('logits','probabilities','mask','pruning_loss')},
        saturation=saturation,
        interpretation='PyTorch gumbel_softmax takes logits. Logits and log probabilities are equivalent; feeding probabilities as logits changes the distribution. CUDA autocast promotes softmax/log to FP32, unlike this CPU probe, so CPU BF16 behavior does not establish the CUDA training dtype.')


def reconstruction_audit(connector):
    import torch
    torch.manual_seed(31)
    features = torch.randn(1, 3, 106, 256)
    output = connector(features, selection_mode='gumbel', compute_reconstruction=True)
    mask = output['selection_mask'].reshape(-1, 106)
    squared_error = (output['normalized_features'].reshape(-1, 106, 256).detach()
                     - output['reconstructed_features'].float()).square()
    direct = ((1.-mask)*squared_error.detach().sum(-1)).sum(-1).mean()
    bias = connector.ccdp.score_mlp[-1].bias
    def grad(value):
        result, = torch.autograd.grad(value, bias, retain_graph=True)
        return result.detach()
    total_gradient = grad(output['reconstruction_loss'])
    direct_gradient = grad(direct)
    graph = dict(
        pruning_weighted_bias_gradient=grad(output['pruning_loss']*10).tolist(),
        reconstruction_total_bias_gradient=total_gradient.tolist(),
        reconstruction_direct_loss_mask_bias_gradient=direct_gradient.tolist(),
        reconstruction_input_and_query_bias_gradient=(total_gradient-direct_gradient).tolist(),
    )
    expected = ((1.-mask)*squared_error.sum(-1)).sum(-1).mean()
    channel_mean = ((1.-mask)*squared_error.mean(-1)).sum(-1).mean()
    dropped_mean = (((1.-mask)*squared_error.sum(-1)).sum(-1)
                    / ((1.-mask).sum(-1)*256).clamp_min(1)).mean()
    valid = output['selected_valid_mask'].reshape(-1, output['selected_valid_mask'].shape[-1])
    indices = output['selected_indices'].reshape(valid.shape)
    aligned = output['aligned_sparse_tokens']
    expected_base = connector.mask_embedding.expand(3, -1, -1).clone()
    for frame in range(3):
        expected_base[frame, indices[frame, valid[frame]]] = aligned[frame, valid[frame]]
    return dict(synthetic_config=connector.config.to_dict(),
        sampled_retention=float(output['selector']['retention'].detach()),
        total_queries_per_frame=output['selected_count_per_frame'].tolist(),
        full_kv_memory_shape=list(output['enhanced_memory'].shape),
        equation8_scatter_and_mask_forward_max_delta=float((expected_base-output['reconstruction_base_input']).abs().max().detach()),
        paper_sum_implemented=float(output['reconstruction_loss'].detach()),
        explicit_token_and_channel_sum=float(expected.detach()),
        channel_mean_token_sum_diagnostic=float(channel_mean.detach()),
        dropped_element_mean_diagnostic=float(dropped_mean.detach()),
        sum_to_channelmean_ratio=float((expected/channel_mean).detach()),
        weighted_pruning=float((output['pruning_loss']*10).detach()),
        gradient_split=graph,
        interpretation='A deterministic single forward/backward of a small randomly initialized Q-Former, not the trained full model. Denominator probes do not alter the graph used for training, and do not identify the authors unspecified channel reduction.')


def full_connector_probe(checkpoint, saved):
    """Original 768x12 trained connector; fixed leaf masks isolate its VJP."""
    import numpy as np
    import torch
    from lavis.models.vldrive.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
    description = json.loads((checkpoint.parent/'model.json').read_text())
    model = QFormerMEFAConnector(QFormerMEFAConfig(**description['connector']),
        qformer_local_pretrained_path=description['config']['qformer_checkpoint'])
    state = {key[len('connector.'):]: value for key, value in saved['trainable_state'].items()
             if key.startswith('connector.')}
    loading = model.load_state_dict(state, strict=False)
    allowed_missing = ('learnable_query_tokens', 'qformer.bert.embeddings.position_ids',
                       'qformer.bert.embeddings.word_embeddings.',
                       'qformer.bert.embeddings.position_embeddings.',
                       'qformer.bert.embeddings.token_type_embeddings.', 'qformer.bert.pooler.')
    if loading.unexpected_keys or any(not key.startswith(allowed_missing) for key in loading.missing_keys):
        raise ValueError('Unexpected trained connector state mismatch: '+str(loading))
    model.eval().requires_grad_(False)
    cache = ROOT/'.runtime/morai_vldrive/full_loop_tokens_epoch25'
    entry = next(json.loads(line) for line in (cache/'manifest.jsonl').read_text().splitlines()
                 if json.loads(line)['frame_id'] == 2865)
    feature_path = cache/entry['visual_feature']['path']
    if entry['split'] != 'train' or digest(feature_path) != entry['visual_feature']['sha256']:
        raise ValueError('Train feature integrity mismatch')
    features = torch.from_numpy(np.load(feature_path).copy()).float()[None, None]
    normalized = model.visual_input_norm(features)
    flat = normalized.reshape(1, 106, 256)
    memory = model.temporal_encoder(normalized)['enhanced_features'].reshape(1, 106, 256)
    valid = torch.ones(1, 106, dtype=torch.bool)
    torch.manual_seed(1623)
    cotangent = torch.randn(1, 106, model.config.language_dim)/106
    reports, outputs = {}, {}
    for location in ('production_gate_before_projection_ln', 'gate_after_projection_before_qformer_ln',
                     'additional_gate_after_complete_connector'):
        mask = torch.ones(1, 106, requires_grad=True)
        if location == 'gate_after_projection_before_qformer_ln':
            query = model._pack_by_mask(model.retained_query_projection(flat), mask)[0]
        else:
            query = model.retained_query_projection(model._pack_by_mask(flat, mask)[0])
        hidden, _ = model._run_qformer(query, valid, memory, valid)
        language = model.qformer_to_language(hidden)
        if location == 'additional_gate_after_complete_connector':
            language = language*mask.unsqueeze(-1)
        scalar = (language*cotangent).sum()
        gradient, = torch.autograd.grad(scalar, mask)
        outputs[location] = language.detach()
        reports[location] = dict(mask_gradient_l2=float(gradient.norm()),
            mask_gradient_abs_max=float(gradient.abs().max()), linear_probe_value=float(scalar.detach()))
    reference = outputs['production_gate_before_projection_ln']
    for name, value in reports.items():
        value['forward_max_delta'] = float((outputs[name]-reference).abs().max())
    return dict(frame=2865, frames=1, history='single-frame zero history, not original 11-frame trajectory replay',
        qformer_width=model.config.qformer_hidden_size, qformer_layers=model.config.qformer_num_layers,
        precision='CPU FP32', query_count=106, kv_count=106,
        allowed_frozen_missing_keys=loading.missing_keys, probes=reports,
        interpretation='Same-forward, different-backward local counterfactuals with no optimizer. Q-Former itself also starts with LayerNorm, so moving the gate only past the first projection is not a demonstrated fix. The linear language-token probe is not an actual waypoint/LLM loss.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=ROOT/'outputs/diagnosis_root_20260911/ccdp_audit.json')
    parser.add_argument('--checkpoint', type=Path,
        default=ROOT/'outputs/morai_paper_last_encoder_20260910T055054Z/model-last.pt')
    parser.add_argument('--full-connector-probe', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = ROOT/'outputs/diagnosis_root_20260911'
    if output.parent != allowed or not output.name.startswith('ccdp_audit') or output.exists():
        raise ValueError('Use a new ccdp_audit filename in the authorized diagnosis output directory')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from lavis.models.vldrive.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
    files = [ROOT/'lavis/models/vldrive/ccdp.py', ROOT/'lavis/models/vldrive/qformer_mefa.py',
             ROOT/'lavis/models/vldrive/training_model.py']
    before = {str(path.relative_to(ROOT)): digest(path) for path in files}
    torch.manual_seed(31)
    connector = QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=256, num_visual_tokens=106, language_dim=64, query_count=32,
        memory_size=2, qformer_hidden_size=64, qformer_num_layers=1,
        qformer_num_attention_heads=4, qformer_intermediate_size=128,
        reconstruction_layers=1, reconstruction_heads=4,
        reconstruction_formulation='eq8', reconstruction_reduction='sum',
        detach_reconstruction_loss_mask=False, retention_logit_weight=0.,
    ))
    result = dict(optimizer_updates=0, gpu_used=False, cpu_threads=torch.get_num_threads(),
        source_hashes=before,
        projection_synthetic=projection_audit(connector.retained_query_projection, 'random local projection'),
        gumbel=gumbel_audit(), reconstruction=reconstruction_audit(connector))
    path = args.checkpoint.resolve()
    integrity = json.loads(path.with_suffix(path.suffix+'.sha256.json').read_text())
    if path.stat().st_size != integrity['bytes'] or digest(path) != integrity['sha256']:
        raise ValueError('Checkpoint integrity mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = saved['trainable_state']
    linear_weight = state['connector.retained_query_projection.1.weight']
    projection = torch.nn.Sequential(torch.nn.LayerNorm(linear_weight.shape[1]),
                                    torch.nn.Linear(linear_weight.shape[1], linear_weight.shape[0]))
    projection.load_state_dict({key[len('connector.retained_query_projection.'):]: value
                               for key, value in state.items() if key.startswith('connector.retained_query_projection.')})
    result['projection_trained_checkpoint'] = projection_audit(projection, 'original model-last retained-query projection')
    result['checkpoint_sha256'] = integrity['sha256']
    if args.full_connector_probe:
        result['trained_full_connector_probe'] = full_connector_probe(path, saved)
    result['source_unchanged'] = before == {str(path.relative_to(ROOT)): digest(path) for path in files}
    if not result['source_unchanged']:
        raise RuntimeError('Model source changed during diagnosis')
    result['primary_sources'] = [
        'https://arxiv.org/html/2511.06256v1',
        'https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.gumbel_softmax.html',
        'https://docs.pytorch.org/docs/2.8/amp.html',
        'https://github.com/raoyongming/DynamicViT/blob/master/models/dyvit.py',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
