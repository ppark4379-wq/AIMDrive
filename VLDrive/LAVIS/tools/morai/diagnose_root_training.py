#!/usr/bin/env python3
"""Read-only training arithmetic audit; CPU-only synthetic probes, no retraining."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def adamw_probe():
    from lavis.runners.vldrive_trainer import CPUAdamW
    parameter = nn.Parameter(torch.tensor([1., -1., .125], dtype=torch.bfloat16))
    initial = parameter.detach().clone()
    reference = nn.Parameter(parameter.detach().float().clone())
    optimizer = CPUAdamW([parameter], lr=1e-5, weight_decay=.06)
    standard = torch.optim.AdamW([reference], lr=1e-5, weight_decay=.06, foreach=False)
    max_master_error = 0.
    first_bf16_change = None
    for step in range(400):
        gradient = torch.tensor([.1, -.2, .0001], dtype=torch.bfloat16)
        parameter.grad = gradient.clone()
        reference.grad = gradient.float().clone()
        optimizer.step()
        standard.step()
        max_master_error = max(max_master_error, float((optimizer.masters[0]-reference).abs().max()))
        assert torch.equal(parameter.detach(), reference.detach().to(torch.bfloat16))
        if first_bf16_change is None and not torch.equal(parameter.detach(), initial):
            first_bf16_change = step + 1
        optimizer.zero_grad()
        standard.zero_grad()
    return dict(steps=400, fp32_master_vs_standard_adamw_max_error=max_master_error,
                bf16_first_visible_change_update=first_bf16_change,
                final_bf16=parameter.tolist(), final_master=optimizer.masters[0].tolist(),
                equivalent_at_every_step=True,
                conclusion='FP32 masters preserve sub-BF16-ULP updates; no lost-update bug in the tested CPU optimizer arithmetic.')


def ddia_probe():
    from transformers import LlamaConfig, LlamaForCausalLM
    from lavis.models.vldrive.ddia_attention import install_ddia_attention, set_ddia_modality_ids
    from lavis.models.vldrive.multimodal_sequence import build_multimodal_sequence
    torch.manual_seed(61)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=64)
    direct = install_ddia_attention(LlamaForCausalLM(config))
    checkpointed = copy.deepcopy(direct)
    checkpointed.model.gradient_checkpointing_enable()
    tokens = torch.randn((1, 9, 32))
    modality = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1]])
    mask = torch.ones((1, 9), dtype=torch.long)
    positions = torch.arange(9)[None]
    gradient_sets, outputs = [], []
    for model in (direct, checkpointed):
        model.train()
        inputs = tokens.clone().requires_grad_(True)
        set_ddia_modality_ids(model, modality)
        hidden = model.model(inputs_embeds=inputs, attention_mask=mask,
                             position_ids=positions, use_cache=False).last_hidden_state
        hidden[:, -1, :16].sum().backward()
        outputs.append(hidden.detach())
        gradient_sets.append({name: value.grad.clone() for name, value in model.named_parameters()
                              if value.grad is not None})
    assert gradient_sets[0].keys() == gradient_sets[1].keys()
    gradient_error = max(float((gradient_sets[0][key]-gradient_sets[1][key]).abs().max())
                         for key in gradient_sets[0])
    direct.eval()
    def prediction(value):
        set_ddia_modality_ids(direct, modality)
        return direct.model(inputs_embeds=value, attention_mask=mask,
                            position_ids=positions, use_cache=False).last_hidden_state
    with torch.no_grad():
        baseline = prediction(tokens)
        future = tokens.clone(); future[:, -3:] += 10.
        future_output = prediction(future)
        altered_text = tokens.clone(); altered_text[:, 1] *= -1
        altered_text_output = prediction(altered_text)
        self_changed = tokens.clone(); self_changed[:, -1] += torch.arange(32)
        self_changed_output = prediction(self_changed)
    # Real HF 4.31 DDIA block math/gradient checkpointing, not a mocked LLM.
    return dict(output_checkpoint_max_error=float((outputs[0]-outputs[1]).abs().max()),
                gradient_checkpoint_max_error=gradient_error,
                future_visual_to_past_max_error=float((baseline[:, :6]-future_output[:, :6]).abs().max()),
                changed_instruction_to_last_visual_max_delta=float((baseline[:, -1]-altered_text_output[:, -1]).abs().max()),
                changed_current_token_to_own_output_max_delta=float((baseline[:, -1]-self_changed_output[:, -1]).abs().max()),
                conclusion='Tiny CPU test preserves gradients and frame causality; text reaches visual outputs. Strict visual <j attention still retains current token through decoder residuals.')


def update_audit(run):
    logs = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines() if line]
    schedule = len(logs)
    maximum_lr_error = max(abs(row['lr'] - 1e-5*.5*(1+math.cos(math.pi*i/schedule)))
                           for i, row in enumerate(logs))
    result = dict(updates=len(logs), contiguous_update_ids=[row['update'] for row in logs] == list(range(1, schedule+1)),
                  cosine_lr_max_error=maximum_lr_error, lr_first=logs[0]['lr'], lr_last=logs[-1]['lr'])
    phases = {'all': logs, 'first_200': logs[:200], 'updates_201_1000': logs[200:1000], 'last_1000': logs[-1000:]}
    for name, phase in phases.items():
        norms = np.asarray([row['grad_norm'] for row in phase])
        clip = np.minimum(1., 10./(norms+1e-6))
        result[name] = dict(updates=len(phase), preclip_gradient_norm_median=float(np.median(norms)),
                            clip_factor_median=float(np.median(clip)),
                            clipped_fraction=float((norms > 10).mean()),
                            waypoint_l1_mean=float(np.mean([r['waypoint_loss'] for r in phase])),
                            reconstruction_mean=float(np.mean([r['reconstruction_loss'] for r in phase])))
    result['first_update'] = logs[0]
    result['limitations'] = 'Global clipping scales every gradient, but AdamW is partly scale-invariant; small clip factors alone do not prove suppressed parameter updates.'
    return result


def source_audit(run):
    provenance = read(run/'provenance.json')
    paths = ['lavis/models/vldrive/ddia_attention.py', 'lavis/models/vldrive/vldrive_model.py',
             'lavis/models/vldrive/multimodal_sequence.py', 'lavis/models/vldrive/tinyllama_backbone.py',
             'lavis/models/vldrive/visual_token_interface.py', 'lavis/runners/vldrive_trainer.py',
             'lavis/models/vldrive/training_model.py', 'lavis/datasets/datasets/morai/expert_vldrive_dataset.py']
    return {path: dict(current_sha256=sha(ROOT/path), training_sha256=provenance['source'].get(path),
                       same_as_trained=sha(ROOT/path) == provenance['source'].get(path)) for path in paths}


def actual_parameter_audit(run):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    checkpoint_path = run/'model-last.pt'
    integrity = read(checkpoint_path.with_suffix('.pt.sha256.json'))
    actual_hash = sha(checkpoint_path)
    if actual_hash != integrity['sha256'] or checkpoint_path.stat().st_size != integrity['bytes']:
        raise ValueError('Final checkpoint integrity mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    saved = checkpoint['trainable_state']
    settings = read(run/'resolved_config.json')
    asset = Path(settings['model']['language_checkpoint'])
    tokenizer = AutoTokenizer.from_pretrained(str(asset), local_files_only=True)
    result = dict(checkpoint_sha256=actual_hash, updates=checkpoint['updates'], weights={}, instruction_embedding_rows={})
    probes = ['model.layers.%d.self_attn.%s.weight' % (layer, part)
              for layer in (0, 10, 21) for part in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    with safe_open(asset/'model.safetensors', framework='pt', device='cpu') as original:
        for key in probes:
            final = saved['language_model.tinyllama.'+key]
            # Compare against initialization *after* its BF16 cast, not FP32 base.
            base = original.get_tensor(key).to(dtype=final.dtype)
            a, b = final.flatten()[::37].float(), base.flatten()[::37].float()
            delta = a-b
            result['weights'][key] = dict(sampled_values=a.numel(), changed_fraction=float((delta != 0).float().mean()),
                                         mean_abs_delta=float(delta.abs().mean()), max_abs_delta=float(delta.abs().max()),
                                         relative_l2=float(delta.norm()/b.norm().clamp_min(1e-12)))
        final_embeddings = saved['language_model.tinyllama.model.embed_tokens.weight']
        original_embeddings = original.get_tensor('model.embed_tokens.weight')
        for text in ('left', 'right', 'straight', 'route'):
            token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
            base = original_embeddings[token_ids].to(final_embeddings.dtype).float()
            final = final_embeddings[token_ids].float()
            result['instruction_embedding_rows'][text] = dict(token_ids=token_ids,
                changed_fraction=float((final != base).float().mean()), mean_abs_delta=float((final-base).abs().mean()),
                relative_l2=float((final-base).norm()/base.norm().clamp_min(1e-12)))
    # mmap touches optimizer scalars and small slices, not all multi-GB masters.
    full = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    optimizer = full['optimizer']
    steps = Counter(int(value['step']) for value in optimizer['optimizer']['state'].values())
    groups = optimizer['optimizer']['param_groups']
    result['optimizer_checkpoint'] = dict(implementation=optimizer['implementation'],
        parameter_state_steps=dict(steps), master_count=len(optimizer['masters']),
        groups=[{key: value for key, value in group.items() if key != 'params'} for group in groups],
        full_checkpoint_hash_reverified=False,
        note='Metadata read via mmap from local checkpoint; model-only weights above independently SHA-verified.')
    master_errors = []
    master_hidden_updates = []
    # Serialized order follows named_parameters() when no optimizer groups used.
    for position, (name, weight) in enumerate(saved.items()):
        if 'self_attn.q_proj.weight' not in name and 'waypoints_predictor' not in name:
            continue
        master = optimizer['masters'][position]
        if master.shape != weight.shape:
            raise ValueError('Original optimizer parameter ordering mismatch')
        sample = master.reshape(-1)[::997]
        model_values = weight.reshape(-1)[::997]
        master_errors.append(float((sample.to(weight.dtype)-model_values).abs().max()))
        master_hidden_updates.append(float((sample-model_values.float()).abs().mean()))
    result['optimizer_checkpoint']['master_cast_vs_model_sample_max_error'] = max(master_errors)
    result['optimizer_checkpoint']['sub_bf16_master_residual_mean_abs'] = float(np.mean(master_hidden_updates))
    return result


def frame_loss_audit(run):
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    settings = read(run/'resolved_config.json')
    data = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
        split_file=ROOT/settings['split_file'], split='train', sequence_length=settings['sequence_length'],
        instructions=settings['instructions'])
    exposures = Counter()
    per_instruction = Counter()
    indices = read(run/'window_indices.json')['train']
    for index in indices:
        begin, end = data.windows[index]
        rows = data.entries[begin:end]
        assert len({r['instruction'] for r in rows}) == 1
        per_instruction[rows[-1]['instruction']] += len(rows)
        exposures.update(int(r['frame_id']) for r in rows)
    reference = data.describe()
    trained = read(run/'provenance.json')['datasets']['train']
    return dict(description_matches_original_training=reference == trained,
                anchors=len(indices), frame_supervision_occurrences_per_epoch=sum(exposures.values()),
                unique_supervised_frames=len(exposures), exposure_count_histogram=dict(Counter(exposures.values())),
                instruction_frame_supervision_occurrences=dict(per_instruction),
                note='All actual-window frame trajectories are trained, not only last-frame evaluation anchors; every original window has one unchanged instruction.')


def projection_initialization_audit(run):
    from safetensors import safe_open
    from transformers import AutoTokenizer, logging
    from lavis.models.vldrive.qformer_mefa import QFormerMEFAConfig, build_qformer
    logging.set_verbosity_error()
    settings = read(run/'resolved_config.json')
    checkpoint = torch.load(run/'model-last.pt', map_location='cpu', weights_only=False, mmap=True)
    saved = checkpoint['trainable_state']
    tokenizer = AutoTokenizer.from_pretrained(settings['model']['language_checkpoint'], local_files_only=True)
    text_embeddings = saved['language_model.tinyllama.model.embed_tokens.weight'].float()
    result = dict(text_embeddings={}, projection={}, query_ffn_initialization={})
    for text in ('Proceed ahead and make a right turn.', 'Proceed ahead and make a left turn.',
                 'Please proceed straight at the next intersection.', 'Proceed along this route.'):
        ids = tokenizer(text)['input_ids']
        values = text_embeddings[ids]
        norms = values.norm(dim=-1)
        result['text_embeddings'][text] = dict(token_ids=ids, token_l2=norms.tolist(),
            mean_token_l2=float(norms.mean()), content_only_mean_token_l2=float(norms[1:].mean()),
            embedding_rms=float(values.square().mean().sqrt()))
    prefix = 'connector.qformer_to_language.'
    gamma, beta = saved[prefix+'0.weight'].float(), saved[prefix+'0.bias'].float()
    weight, bias = saved[prefix+'1.weight'].float(), saved[prefix+'1.bias'].float()
    projected_mean = weight@beta+bias
    expected_squared_l2 = ((weight*gamma[None]).square().sum()+projected_mean.square().sum())
    result['projection'] = dict(layernorm_gamma_mean=float(gamma.mean()),
        layernorm_gamma_rms=float(gamma.square().mean().sqrt()), layernorm_beta_rms=float(beta.square().mean().sqrt()),
        linear_weight_rms=float(weight.square().mean().sqrt()), linear_weight_frobenius=float(weight.norm()),
        linear_bias_l2=float(bias.norm()), linear_bias_rms=float(bias.square().mean().sqrt()),
        hypothetical_unit_variance_input_expected_l2=float(expected_squared_l2.sqrt()),
        hypothetical_unit_variance_input_expected_rms=float((expected_squared_l2/len(bias)).sqrt()),
        note='Projection weights are actual final values. Expected output norm assumes independent zero-mean unit-variance hidden components BEFORE learned affine LN; not measured real Q-Former activations.')
    ffn_result = result['query_ffn_initialization']
    asset = Path(settings['model']['qformer_checkpoint'])
    config = QFormerMEFAConfig(**read(run/'model.json')['connector'])
    torch.manual_seed(607)
    qformer = build_qformer(config, local_pretrained_path=str(asset)).eval()
    newly_initialized = qformer.state_dict()
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    with safe_open(asset/'model.safetensors', framework='pt', device='cpu') as original:
        ffn_result['bert_asset_has_query_ffn_keys'] = any('_query' in key for key in original.keys())
        ffn_result['layers'] = []
        for layer in (0, 5, 11):
            normal = 'bert.encoder.layer.%d.intermediate.dense.weight' % layer
            query = normal.replace('.intermediate.', '.intermediate_query.')
            base = original.get_tensor(normal)
            current_normal, current_query = newly_initialized[normal], newly_initialized[query]
            trained_normal, trained_query = saved['connector.qformer.'+normal], saved['connector.qformer.'+query]
            ffn_result['layers'].append(dict(layer=layer, bert_original_sha256=tensor_sha(base),
                newly_loaded_ordinary_ffn_sha256=tensor_sha(current_normal),
                newly_initialized_query_ffn_sha256=tensor_sha(current_query),
                trained_ordinary_ffn_sha256=tensor_sha(trained_normal),
                loaded_ordinary_equals_bert=torch.equal(current_normal, base),
                initialized_query_equals_bert=torch.equal(current_query, base),
                trained_unused_ordinary_equals_bert=torch.equal(trained_normal, base),
                new_query_std=float(current_query.std()), bert_std=float(base.std()),
                trained_query_std=float(trained_query.std())))
    ffn_result['limitation'] = ('Seed607 here isolates the builder; it does not replay the original random-generator '
        'advancement through TinyLlama/connector initialization. New query hashes are NOT claimed to equal '
        'the exact original run initial hashes. Asset absence + builder control flow + ordinary BERT hash matches '
        'establish missing query-FFN pretrain transfer, not its causal impact on driving.')
    full = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    states = full['optimizer']['optimizer']['state']
    unused = [name for index, name in enumerate(saved) if index not in states]
    result['unused_trainable_parameters'] = dict(tensors=len(unused),
        parameter_count=sum(saved[name].numel() for name in unused), names=unused,
        note='Ordinary text FFNs unused because connector is visual-query-only; this is overhead, not a cut visual gradient path.')
    return result


def rope_precision_audit(run):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
    from lavis.models.vldrive.ddia_attention import apply_rope, compose_ddia_logits
    torch.manual_seed(812)
    length, dim, heads = 1201, 64, 4
    positions = torch.arange(length)[None]
    q_source, k_source = torch.randn(1, heads, length, dim), torch.randn(1, heads, length, dim)
    rotary = LlamaRotaryEmbedding(dim, max_position_embeddings=2048)
    frequency = 1.0/(10000.0 ** (torch.arange(0, dim, 2).float()/dim))
    angles = positions.float().unsqueeze(-1)*frequency
    modality = torch.ones((1, length), dtype=torch.long)
    modality[:, :10] = 0
    report = dict(head_dim=dim, heads=heads, positions_inclusive=[0, length-1], instruction_tokens=10,
                  reference='Installed HF Transformers 4.31 LlamaRotaryEmbedding: FP32 cos/sin cache then cast',
                  dtypes={})
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        q, k = q_source.to(dtype), k_source.to(dtype)
        ours_q, ours_k = apply_rope(q, positions), apply_rope(k, positions)
        cos, sin = rotary(q, seq_len=length)
        reference_q, reference_k = apply_rotary_pos_emb(q, k, cos, sin, positions)
        diff = ours_q.float()-reference_q.float()
        quantized_angles = angles.to(dtype).float()
        angle_error = quantized_angles-angles
        ours_logits, allowed = compose_ddia_logits(q, k, ours_q, ours_k, modality)
        reference_logits, _ = compose_ddia_logits(q, k, reference_q, reference_k, modality)
        ours_logits, reference_logits = ours_logits.float(), reference_logits.float()
        cross_error = (ours_logits[:, :, 10:, :10]-reference_logits[:, :, 10:, :10]).abs()
        visual_mask = allowed[:, :, 10:, 10:].expand(1, heads, -1, -1)
        visual_diff = (ours_logits[:, :, 10:, 10:]-reference_logits[:, :, 10:, 10:])[visual_mask]
        visual_reference = reference_logits[:, :, 10:, 10:][visual_mask]
        ours_probability = ours_logits.masked_fill(~allowed, float('-inf')).softmax(-1)
        reference_probability = reference_logits.masked_fill(~allowed, float('-inf')).softmax(-1)
        tv = (ours_probability-reference_probability).abs().sum(-1)*.5
        top1 = (ours_probability.argmax(-1) != reference_probability.argmax(-1)).float()
        # Same content at adjacent positions must retain approximately cos(1)
        # similarity in the first frequency, independent of absolute position.
        basis = torch.zeros((1, 1, length, dim), dtype=dtype)
        basis[..., 0] = 1
        ours_basis = apply_rope(basis, positions).float()
        ref_basis = apply_rotary_pos_emb(basis, basis, cos, sin, positions)[0].float()
        ours_neighbor = (ours_basis[:, :, 1:]*ours_basis[:, :, :-1]).sum(-1).flatten()
        ref_neighbor = (ref_basis[:, :, 1:]*ref_basis[:, :, :-1]).sum(-1).flatten()
        selected_positions = [1, 127, 255, 256, 257, 511, 512, 513, 767, 768, 1023, 1024,
                              1025, 1100, 1151, 1152, 1165, 1175, 1200]
        result = dict(rotated_query_max_abs_error=float(diff.abs().max()),
            rotated_query_rms_error=float(diff.square().mean().sqrt()),
            rotated_query_relative_l2=float(diff.norm()/reference_q.float().norm()),
            angle_quantization_max_error_rad=float(angle_error.abs().max()),
            angle_quantization_rms_error_rad=float(angle_error.square().mean().sqrt()),
            visual_visual_score_max_abs_error=float(visual_diff.abs().max()),
            visual_visual_score_rms_error=float(visual_diff.square().mean().sqrt()),
            visual_visual_score_relative_l2=float(visual_diff.norm()/visual_reference.norm()),
            visual_instruction_raw_score_max_abs_error=float(cross_error.max()),
            text_text_score_max_abs_error=float((ours_logits[:, :, :10, :10]-reference_logits[:, :, :10, :10]).abs().max()),
            first_frequency_unique_quantized_angles=int(quantized_angles[0, :, 0].unique().numel()),
            adjacent_first_frequency_angle_collisions=int((quantized_angles[0, 1:, 0] == quantized_angles[0, :-1, 0]).sum()),
            neighbor_first_frequency_similarity_examples=[dict(position=p,
                original_angle=float(angles[0, p, 0]), quantized_angle=float(quantized_angles[0, p, 0]),
                original_previous_angle=float(angles[0, p-1, 0]), quantized_previous_angle=float(quantized_angles[0, p-1, 0]),
                current_similarity=float(ours_neighbor[p-1]), hf_similarity=float(ref_neighbor[p-1]),
                exact_cos_1=math.cos(1)) for p in selected_positions],
            by_query_position={})
        for label, start, stop in [('short_10_255', 10, 256), ('mid_256_511', 256, 512),
                                   ('long_512_1023', 512, 1024), ('long_1024_1200', 1024, length)]:
            block = diff[:, :, start:stop]
            result['by_query_position'][label] = dict(
                rotated_query_relative_l2=float(block.norm()/reference_q[:, :, start:stop].float().norm()),
                attention_total_variation_mean=float(tv[:, :, start:stop].mean()),
                attention_top1_changed_fraction=float(top1[:, :, start:stop].mean()))
        report['dtypes'][str(dtype)] = result
    report['classification'] = ('Confirmed numerical incompatibility with HF RoPE: casting angles before trig '
        'quantizes positional phase. The BF16 cast occurs explicitly before cos/sin, so CUDA/autocast cannot '
        'restore discarded phase bits; exact kernel-level errors may vary by device. This does NOT yet '
        'prove the size of its contribution to the trained model turning failure. Raw cross-modal DDIA '
        'scores are unchanged directly, while visual/visual rotary scores and shared normalization change.')
    report['synthetic_scope'] = ('Queries/keys are unit-normal random tensors, not actual trained layer activations. '
        'The basis example independently demonstrates violation of relative-position similarity. '
        'No rotary/model source or checkpoint was modified.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/morai_paper_last_encoder_20260910T055054Z')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/diagnosis_root_20260911/training_audit.json')
    parser.add_argument('--projection-only', action='store_true')
    parser.add_argument('--rope-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an earlier diagnostic artifact')
    report = dict(scope='CPU-only; no existing code, weights, datasets or training runs mutated',
                  paper_source='https://arxiv.org/html/2511.06256v1#S3.S5',
                  source_identity=source_audit(args.run))
    operations = [('rope_precision', lambda: rope_precision_audit(args.run))] if args.rope_only else (
        [('projection_and_initialization', lambda: projection_initialization_audit(args.run))] if args.projection_only else (
            ('adamw_arithmetic', adamw_probe), ('ddia_checkpoint_and_causality', ddia_probe),
            ('training_logs', lambda: update_audit(args.run)),
            ('actual_parameters', lambda: actual_parameter_audit(args.run)),
            ('frame_supervision', lambda: frame_loss_audit(args.run))))
    for name, operation in operations:
        print('Auditing '+name, flush=True)
        report[name] = operation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
