#!/usr/bin/env python3
"""Read-only pilot checkpoint diagnostics: gradients, selection shift, text sensitivity.

No optimizer is constructed. Diagnostic selector-mode overrides are scoped hooks;
they are not training changes or proposed inference policies. Hashes before and
after must match. Only new diagnostic artifacts under LAVIS/outputs are written.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lavis.common.morai_runtime import configure_runtime, bootstrap, sha256
configure_runtime()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


@contextmanager
def selector_mode(connector, mode):
    def override(module, args, kwargs):
        return args, dict(kwargs, selection_mode=mode)
    handle = connector.register_forward_pre_hook(override, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-prefix', default='morai_root_pilot_20260910T214044Z')
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / 'outputs' not in output.parents:
        raise ValueError('Diagnostic output must stay under LAVIS/outputs')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'full_graph.json'
    if destination.exists() or (output / 'selector_initial.pt').exists():
        raise ValueError('Diagnostic artifacts already exist; never overwrite prior evidence')
    import numpy as np
    import torch
    backend = bootstrap(ROOT.parent)
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    from lavis.runners.vldrive_trainer import (seed_all, source_hashes, restore_model_checkpoint,
                                             parameter_hash, preserve_rng_state)
    from lavis.runners.morai_trainer import autocast, batch
    from tools.morai.root_pilot_checks import summarize, category
    import fcntl
    lock = (ROOT / '.runtime/training_runs/last_encoder.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    sources = source_hashes(ROOT)
    result = dict(schema='pilot-full-graph-audit-v1', optimizer_updates=0, training_started=False,
                  source_sha256=sources, diagnostic_script_sha256=sha256(Path(__file__)),
                  interpretation='Bounded checkpoint observations, not a new trained model or closed-loop score', arms={})
    for arm in ('legacy', 'retained_output_st'):
        run = ROOT / 'outputs' / (args.run_prefix + '_' + arm)
        settings, report, provenance = (read(run / name) for name in
            ('resolved_config.json', 'report.json', 'provenance.json'))
        if sources != provenance['source'] or backend != provenance['backend']:
            raise ValueError('Production source/backend changed since pilot')
        for name, digest in provenance['pilot_sources'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError('Pilot source changed: ' + name)
        dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
            split_file=settings['split_file'], instructions=settings['instructions'], split='train',
            sequence_length=settings['sequence_length'], instruction_boundary_policy='exclude')
        if dataset.describe() != provenance['datasets']['train']:
            raise ValueError('Original train data contract changed')
        seed_all(settings['seed'])
        print(json.dumps(dict(phase='load', arm=arm)), flush=True)
        model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda')
        if parameter_hash(model, trainable=True) != report['initial_trainable_sha256']:
            raise ValueError('Fresh initialization did not reproduce pilot')
        if arm == 'legacy':
            torch.save({n: p.detach().cpu().clone() for n, p in model.named_parameters()
                        if n.startswith(('connector.visual_input_norm.', 'connector.ccdp.'))}, output / 'selector_initial.pt')
            print('SELECTOR_INITIAL_SAVED', flush=True)
        checkpoint = run / 'model-last.pt'
        restore_model_checkpoint(checkpoint, model, provenance=provenance)
        frozen_hash, trainable_hash = parameter_hash(model, trainable=False), parameter_hash(model, trainable=True)
        model.eval()
        pilot_indices = report['sampling']['evaluation_indices']
        categories = report['sampling']['detail']['category_indices']
        original_rows = read(run / 'after_000320_predictions.json')
        original = {r['window_index']: r for r in original_rows}
        arm_result = dict(checkpoint_sha256=sha256(checkpoint), updates=report['updates'],
                          gradient_probes=[], selection_mode_probes={}, instruction_probes=[])
        result['arms'][arm] = arm_result

        def snapshot():
            groups = {
                'selector': 'connector.ccdp.', 'input_norm': 'connector.visual_input_norm.',
                'mefa': 'connector.temporal_encoder.', 'qformer': 'connector.qformer.',
                'language_projection': 'connector.qformer_to_language.',
                'reconstruction': 'connector.reconstruction_',
                'waypoint_head': 'language_model.waypoints_predictor.',
                'llm': 'language_model.tinyllama.model.',
            }
            accumulators = {key: [] for key in groups}
            selector_parts, bias = [], {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float()
                for key, prefix in groups.items():
                    if name.startswith(prefix):
                        accumulators[key].append(grad.square().sum())
                if name.startswith('connector.ccdp.'):
                    selector_parts.append(grad.reshape(-1).cpu())
                    if name.endswith('.bias') and parameter.numel() == 2:
                        bias[name] = grad.cpu().tolist()
            norms = {key: float(torch.stack(parts).sum().sqrt()) if parts else 0.
                     for key, parts in accumulators.items()}
            return norms, torch.cat(selector_parts) if selector_parts else torch.zeros(1), bias

        # Distinct actual left/right/stop cases; all11 frames remain supervised.
        for label in ('moving_left', 'moving_right', 'stationary'):
            index = categories[label][0]
            with preserve_rng_state():
                seed_all(1307 + index)
                model.train()
                model.zero_grad(set_to_none=True)
                item = batch(dataset[index], 'cuda')
                with autocast(model):
                    prediction = model.predict(item['model_inputs'], compute_auxiliary=True)
                    waypoint = model.loss(prediction, item['training_targets'])['waypoint_loss']
                losses = dict(waypoint=waypoint, pruning=10 * prediction['pruning_loss'],
                              reconstruction=prediction['reconstruction_loss'])
                probe = dict(category=label, window_index=index,
                    sample_id=original[index]['sample_id'], losses={k: float(v.detach()) for k, v in losses.items()},
                    sampled_retention=float(prediction['sampled_retention'].mean()),
                    soft_retention=float(prediction['soft_retention'].mean()), gradient_norms={}, bias_gradients={})
                vectors = {}
                for name, loss in losses.items():
                    model.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=name != 'reconstruction')
                    norms, vector, bias = snapshot()
                    probe['gradient_norms'][name], vectors[name], probe['bias_gradients'][name] = norms, vector, bias
                probe['selector_cosines'] = {}
                for a, b in (('waypoint', 'reconstruction'), ('pruning', 'reconstruction')):
                    denominator = float(vectors[a].norm() * vectors[b].norm())
                    probe['selector_cosines'][a + '_vs_' + b] = float(torch.dot(vectors[a], vectors[b])) / max(denominator, 1e-30)
                combined = sum(vectors.values())
                probe['selector_total_vs_reconstruction_cosine'] = float(torch.dot(combined, vectors['reconstruction'])) / max(
                    float(combined.norm() * vectors['reconstruction'].norm()), 1e-30)
                arm_result['gradient_probes'].append(probe)
                model.zero_grad(set_to_none=True)
                del prediction, waypoint, losses, loss, vectors, combined
                print(json.dumps(dict(phase='gradients', arm=arm, case=label,
                                      norms=probe['gradient_norms']['reconstruction'])), flush=True)
                write(destination, result)
        model.eval()

        # Keep model/dropout in eval; change ONLY selector sampling to reproduce
        # stochastic training decisions. No fixed top-k count is tested/adopted.
        for mode in ('argmax', 'gumbel'):
            trials = []
            for trial_seed in ((607,) if mode == 'argmax' else (607, 608, 609)):
                rows, counts, objectives, shifts, last_indices = [], [], [], [], []
                for index in pilot_indices:
                    item = batch(dataset[index], 'cuda')
                    with torch.no_grad(), preserve_rng_state():
                        seed_all(trial_seed + index)
                        captures = {}
                        def observe(module, args, output):
                            captures['last_token_indices'] = output['selected_indices'][0].gather(
                                1, (output['selected_count_per_frame'][0] - 1)[:, None]).squeeze(1).cpu().tolist()
                        handle = model.connector.register_forward_hook(observe)
                        try:
                            with selector_mode(model.connector, mode), autocast(model):
                                prediction = model.predict(item['model_inputs'], compute_auxiliary=True)
                        finally:
                            handle.remove()
                        gt = item['training_targets']['future_trajectory'][0, -1].float().cpu().numpy()
                        pred = prediction['trajectory'][0, -1].float().cpu().numpy()
                        losses = model.loss(prediction, item['training_targets'])
                        objectives.append({key: float(losses[key]) for key in
                                           ('loss', 'waypoint_loss', 'pruning_loss', 'reconstruction_loss')})
                        counts.extend(prediction['selected_count'][0].tolist())
                        last_indices.extend(captures['last_token_indices'])
                        shift = float(np.linalg.norm(pred - np.asarray(original[index]['prediction']), axis=-1).mean())
                        if mode == 'argmax' and shift != 0:
                            raise ValueError('Original saved evaluation was not reproduced exactly')
                        shifts.append(shift)
                        rows.append(dict(window_index=index, sample_id=original[index]['sample_id'], category=category(gt),
                            actual_future=gt.tolist(), prediction=pred.tolist(), errors_m=np.linalg.norm(pred - gt, axis=1).tolist(),
                            last_readout_source_index=captures['last_token_indices'][-1]))
                from collections import Counter
                trial = dict(seed=trial_seed, metrics=summarize(rows), retained_fraction=float(np.mean(counts) / 106),
                    count_min=min(counts), count_max=max(counts),
                    last_readout_index_counts=dict(Counter(last_indices)),
                    prediction_mean_shift_from_saved_argmax_m=float(np.mean(shifts)),
                    mean_objectives={key: float(np.mean([r[key] for r in objectives])) for key in objectives[0]})
                trials.append(trial)
                write(output / ('predictions_%s_%s_%d.json' % (arm, mode, trial_seed)), rows)
                print(json.dumps(dict(phase='selection', arm=arm, mode=mode, trial=trial)), flush=True)
            arm_result['selection_mode_probes'][mode] = trials
            write(destination, result)

        # Language sensitivity on the same visual clips; counterfactual texts
        # have no new trajectory GT and are not scored as correct driving.
        for label in ('moving_left', 'moving_right'):
            index = categories[label][-1]
            item = batch(dataset[index], 'cuda')
            texts = {'original': item['model_inputs']['instruction'],
                     'left': ['Proceed ahead and make a left turn.'],
                     'right': ['Proceed ahead and make a right turn.']}
            predictions = {}
            with torch.no_grad(), preserve_rng_state():
                for text_label, text in texts.items():
                    inputs = dict(item['model_inputs'], instruction=text)
                    with autocast(model):
                        prediction = model.predict(inputs, compute_auxiliary=False)
                    predictions[text_label] = prediction['trajectory'][0, -1].float().cpu().numpy()
            arm_result['instruction_probes'].append(dict(window_index=index, category=label,
                original_instruction=texts['original'], predictions={k: v.tolist() for k, v in predictions.items()},
                right_left_endpoint_lateral_delta_m=float(abs(predictions['right'][-1, 0] - predictions['left'][-1, 0])),
                note='Counterfactual input sensitivity only; no synthetic ground truth'))
        if parameter_hash(model, trainable=True) != trainable_hash or parameter_hash(model, trainable=False) != frozen_hash:
            raise ValueError('Read-only diagnostics altered weights')
        arm_result.update(trainable_hash_unchanged=True, frozen_hash_unchanged=True)
        write(destination, result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if source_hashes(ROOT) != sources:
        raise ValueError('Production sources changed during diagnostics')
    result['complete'] = True
    write(destination, result)
    print('PILOT_FULL_GRAPH_DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
