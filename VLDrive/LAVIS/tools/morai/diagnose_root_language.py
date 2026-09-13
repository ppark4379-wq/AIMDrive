#!/usr/bin/env python3
"""Bounded original-model inference hooks: text -> hidden -> waypoint readout."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from offline_train_replay import ROOT, bootstrap, read, write, sha256, np, torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[2865, 2970, 3320])
    args = parser.parse_args()
    output = args.output.resolve()
    assert ROOT in output.parents and not output.exists()
    torch.set_num_threads(2)
    torch.manual_seed(607)
    settings = read(args.run / 'resolved_config.json')
    bootstrap(ROOT.parent)
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    from lavis.models.vldrive.ddia_attention import apply_rope, repeat_kv
    from lavis.runners.vldrive_trainer import parameter_hash, source_hashes
    from lavis.runners.morai_trainer import autocast
    free, total = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        raise RuntimeError('Require at least 6 GiB free; never displace paused training')
    checkpoint_path = args.run / 'model-last.pt'
    digest = read(checkpoint_path.with_suffix('.pt.sha256.json'))
    assert checkpoint_path.stat().st_size == digest['bytes']
    assert sha256(checkpoint_path) == digest['sha256']
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda').eval()
    # Old checkpoint predates opt-in repair fields. Require original behaviour
    # after dataclass default expansion; do not represent this as exact resume.
    assert asdict(VLDriveTrainingConfig(**state['model_config'])) == asdict(model.config)
    assert model.config.training_variant == 'paper_independent'
    assert parameter_hash(model, trainable=False) == state['frozen_hash']
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    assert expected == set(state['trainable_state'])
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in expected:
                p.copy_(state['trainable_state'][name])
    assert parameter_hash(model, trainable=True) == state['trainable_hash']
    assert state['updates'] == 9540
    del state
    saved = {int(row['sample_id'].rsplit('/', 1)[1]): row for row in read(args.run / 'train_predictions.json')}
    cache = Path(settings['feature_cache'])
    rows = [json.loads(line) for line in (cache / 'manifest.jsonl').read_text().splitlines()]
    rows = {row['frame_id']: row for row in rows if row['split'] == 'train'}
    instructions = [json.loads(line) for line in Path(settings['instructions']).read_text().splitlines()]
    all_results = []
    trace = {}
    handles = []

    def reference_rope(value, position, theta):
        frequency = 1.0 / (theta ** (torch.arange(0, value.shape[-1], 2, device=value.device).float() / value.shape[-1]))
        angle = position.float().unsqueeze(-1) * frequency
        angle = torch.cat((angle, angle), -1).unsqueeze(1)
        cosine, sine = angle.cos().to(value.dtype), angle.sin().to(value.dtype)
        first, second = value.chunk(2, -1)
        return value * cosine + torch.cat((-second, first), -1) * sine

    def attention_hook(index):
        def hook(module, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            position = kwargs['position_ids']
            b, length, _ = hidden.shape
            q = module.q_proj(hidden[:, -1:]).view(b, 1, module.num_heads, module.head_dim).transpose(1, 2)
            k = module.k_proj(hidden).view(b, length, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            q_reference = reference_rope(q, position[:, -1:], module.rope_theta)
            k_reference = repeat_kv(reference_rope(k, position, module.rope_theta), module.num_key_value_groups)
            qr = apply_rope(q, position[:, -1:], module.rope_theta)
            kr = apply_rope(k, position, module.rope_theta)
            k = repeat_kv(k, module.num_key_value_groups)
            kr = repeat_kv(kr, module.num_key_value_groups)
            scale = module.head_dim ** -.5
            text = module.modality_ids[:, None, None, :] == 0
            raw, rotated = (q @ k.transpose(-1, -2)) * scale, (qr @ kr.transpose(-1, -2)) * scale
            logits = torch.where(text, raw, rotated)
            logits[..., -1] = torch.finfo(logits.dtype).min
            weights = torch.softmax(logits.float(), -1)
            mass = (weights * text).sum(-1)
            trace.setdefault(index, {})['text_attention_mean'] = float(mass.mean())
            trace[index]['text_attention_max_head'] = float(mass.max())
            text_count = int((module.modality_ids[0] == 0).sum())
            trace[index]['text_token_attention_mean_heads'] = weights[0, :, 0, :text_count].mean(0).cpu().tolist()
            reference_logits = torch.where(text, raw, (q_reference @ k_reference.transpose(-1, -2)) * scale)
            reference_logits[..., -1] = torch.finfo(reference_logits.dtype).min
            reference_weights = torch.softmax(reference_logits.float(), -1)
            trace[index]['rope_reference'] = dict(
                dtype=str(q.dtype), sequence_length=length,
                attention_total_variation_mean_head=float((weights - reference_weights).abs().sum(-1).mean() / 2),
                reference_text_attention_mean=float((reference_weights * text).sum(-1).mean()),
                current_text_attention_mean=float(mass.mean()),
                top_key_changed_head_fraction=float((weights.argmax(-1) != reference_weights.argmax(-1)).float().mean()))
        return hook

    def layer_hook(index):
        def hook(module, args, outputs):
            hidden = outputs[0]
            trace.setdefault(index, {})['last_hidden'] = hidden[0, -1].float().cpu().numpy()
        return hook

    for index, layer in enumerate(model.language_model.tinyllama.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(attention_hook(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(layer_hook(index)))

    def emit():
        output.parent.mkdir(parents=True, exist_ok=True)
        write(output, dict(checkpoint_sha256=digest['sha256'], optimizer_updates=0, split='train_only',
            original_paper_config=asdict(model.config), current_source=source_hashes(ROOT),
            note='Post-repair code loaded with original defaults; original trainable/frozen hashes verified; baseline predictions checked. Hooks are observational. Counterfactual instructions have no new driving GT.',
            results=all_results))

    with torch.inference_mode():
        for frame in args.frames:
            assert frame in saved
            feature = np.stack([np.load(cache / rows[f]['visual_feature']['path']) for f in range(frame-10, frame+1)])
            x = torch.from_numpy(feature.astype(np.float32))[None].cuda()
            correct = next(item['text'] for item in instructions if item['start_frame'] <= frame <= item['stop_frame'])
            with autocast(model):
                connected = model.connector(x, compute_reconstruction=False)
            visual_rms = float(connected['language_tokens'].float().square().mean().sqrt())
            snapshots = {}
            predictions = {}
            instruction_texts = dict(correct=correct, right='Proceed ahead and make a right turn.',
                                     left='Proceed ahead and make a left turn.', generic='Proceed along this route.')
            for label, text in instruction_texts.items():
                trace.clear()
                encoded = model.language_model._tokenize([text], 'cuda')
                embeddings = model.language_model.tinyllama.get_input_embeddings()(encoded['input_ids'])
                with autocast(model):
                    language = model.language_model([text], connected['language_tokens'],
                        connected['selected_valid_mask'], predict_per_frame=True)
                predicted = language.predicted_waypoints[0, -1].float().reshape(5, 2).cpu().numpy()
                last = language.hidden_states[0, -1].float()
                head = model.language_model.waypoints_predictor
                first = torch.nn.functional.linear(last, head[0].weight.float(), head[0].bias.float()).relu()
                fp32 = torch.nn.functional.linear(first, head[2].weight.float(), head[2].bias.float()).reshape(5, 2)
                snapshots[label] = {index: dict(value) for index, value in trace.items()}
                snapshots[label]['readout'] = last.cpu().numpy()
                predictions[label] = dict(instruction=text, prediction=predicted.tolist(),
                    text_tokens=model.language_model.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].cpu().tolist()),
                    text_embedding_rms=float(embeddings.float().square().mean().sqrt()), visual_embedding_rms=visual_rms,
                    fp32_head_prediction=fp32.cpu().tolist(),
                    relu_active_fraction=float((first > 0).float().mean()),
                    text_attention_by_layer=[trace[index]['text_attention_mean'] for index in sorted(trace)],
                    text_token_attention_by_layer=[trace[index]['text_token_attention_mean_heads'] for index in sorted(trace)],
                    rope_reference_by_layer=[trace[index]['rope_reference'] for index in sorted(trace)])
                if label == 'correct':
                    discrepancy = float(np.max(np.abs(predicted - np.asarray(saved[frame]['trajectory']))))
                    assert discrepancy < .04, ('Baseline changed', frame, discrepancy)
            comparisons = {}
            for first, second in [('right', 'left'), ('correct', 'generic')]:
                per_layer = []
                for index in range(len(model.language_model.tinyllama.model.layers)):
                    a, b = snapshots[first][index]['last_hidden'], snapshots[second][index]['last_hidden']
                    per_layer.append(dict(layer=index, relative_l2=float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-20)),
                                          max_abs=float(np.max(np.abs(a-b)))))
                a, b = snapshots[first]['readout'], snapshots[second]['readout']
                comparisons[first + '_vs_' + second] = dict(layers=per_layer,
                    readout_relative_l2=float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-20)),
                    readout_max_abs=float(np.max(np.abs(a-b))))
            row = dict(frame=frame, baseline_max_abs_difference_m=discrepancy,
                       predictions=predictions, comparisons=comparisons)
            all_results.append(row)
            emit()
            print(json.dumps(dict(frame=frame, baseline_difference=discrepancy,
                right_left_readout_relative_l2=comparisons['right_vs_left']['readout_relative_l2'],
                original_prediction=predictions['correct']['prediction'],
                text_attention=predictions['correct']['text_attention_by_layer'])), flush=True)
    for handle in handles:
        handle.remove()
    print('LANGUAGE_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
