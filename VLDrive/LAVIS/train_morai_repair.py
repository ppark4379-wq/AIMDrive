#!/usr/bin/env python3
"""Explicit MORAI repair ablation: frozen epoch25 encoder, gated downstream training.

This is not the paper-identical path. A small train-only overfit gate must pass
before the complete balanced train run is allowed to start. No validation data
is used for training, gating, or model selection.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, bootstrap, sha256, add_path_arguments, resolve_settings
configure_runtime()
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pending.replace(path)


def category(gt):
    import numpy as np
    gt = np.asarray(gt)
    if float(np.linalg.norm(gt, axis=-1).max()) < .5:
        return 'stationary'
    if float(gt[-1, 0]) >= 2.:
        return 'moving_left'
    if float(gt[-1, 0]) <= -2.:
        return 'moving_right'
    return 'moving_straight'


def summarize(rows):
    import numpy as np
    groups = {}
    for label in ('all', 'stationary', 'moving_straight', 'moving_left', 'moving_right', 'curved', 'moving'):
        values = [r for r in rows if label == 'all' or r['category'] == label
                  or label == 'curved' and r['category'] in ('moving_left', 'moving_right')
                  or label == 'moving' and r['category'] != 'stationary']
        if not values:
            continue
        errors = np.asarray([r['errors_m'] for r in values])
        groups[label] = dict(windows=len(values), ade_m=float(errors.mean()), fde_m=float(errors[:, -1].mean()))
        if label in ('moving_left', 'moving_right', 'curved'):
            correct = [r['prediction'][-1][0]*r['actual_future'][-1][0] > 0
                       and abs(r['prediction'][-1][0]) >= .5 for r in values]
            groups[label]['direction_accuracy'] = float(np.mean(correct))
    return groups


def check_gate(before, after, telemetry, config, updates):
    curved = after.get('curved', {})
    tests = dict(
        minimum_updates=updates >= config['minimum_updates'],
        both_turn_directions=all(k in after for k in ('moving_left', 'moving_right')),
        curved_absolute_error=curved.get('ade_m', float('inf')) <= config['curved_ade_max_m'],
        curved_improvement=curved.get('ade_m', float('inf')) <= before['curved']['ade_m']*config['curved_ade_ratio_max'],
        direction_left=after.get('moving_left', {}).get('direction_accuracy', 0.) >= config['direction_accuracy_min'],
        direction_right=after.get('moving_right', {}).get('direction_accuracy', 0.) >= config['direction_accuracy_min'],
        sampled_retention=config['retention_min'] <= telemetry['sampled_retention'] <= config['retention_max'],
        soft_retention=config['retention_min'] <= telemetry['soft_retention'] <= config['retention_max'],
        selector_not_saturated=telemetry['saturated_keep_fraction'] <= config['saturated_keep_fraction_max'],
    )
    return dict(passed=all(tests.values()), checks=tests, training_selector=telemetry,
                note='Top-k inference count is enforced and is NOT evidence of learned dynamic retention.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-only', action='store_true')
    add_path_arguments(parser)
    args = parser.parse_args()
    settings = resolve_settings(read(args.config), args)
    settings['split_file'] = str((args.path_root/settings['split_file']).resolve())
    settings['warm_start'] = str((args.path_root/settings['warm_start']).resolve())
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new output inside LAVIS; prior runs are never overwritten')
    output.mkdir(parents=True)
    write(output/'resolved_config.json', settings)
    started = time.monotonic()
    state = dict(pid=os.getpid(), output=str(output), encoder_retrained=False, split='train_only')

    def status(phase, **fields):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_s=time.monotonic()-started, **fields)
        write(output/'progress.json', state)
        print(json.dumps(state, allow_nan=False), flush=True)

    try:
        status('validating')
        import numpy as np
        import torch
        from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
        from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
        from lavis.runners.vldrive_trainer import (seed_all, source_hashes, stride_indices,
            balanced_training_indices, warm_start_model_checkpoint, make_optimizer, preflight, train,
            save_model_checkpoint, save_checkpoint, periodic_checkpoint_path, parameter_hash, preserve_rng_state)
        from lavis.runners.morai_trainer import batch, autocast
        torch.set_num_threads(settings['cpu_threads'])
        seed_all(settings['seed'])
        backend = bootstrap(ROOT.parent)
        dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
            split_file=settings['split_file'], instructions=settings['instructions'], split='train',
            sequence_length=settings['sequence_length'],
            instruction_boundary_policy=settings['instruction_boundary_policy'])
        config = VLDriveTrainingConfig(**settings['model'])
        if config.training_variant != 'morai_repair_v1' or config.input_mode != 'cached':
            raise ValueError('This runner requires the explicit frozen-cache repair variant')
        if sha256(Path(config.visual_checkpoint)) != dataset.checkpoint_sha256:
            raise ValueError('Frozen encoder/cache mismatch')
        # Verify the immutable existing TRAIN cache in full before optimization.
        for entry in dataset.entries:
            dataset._array(entry['visual_feature'], (106, 256), 'visual feature')
            dataset._array(entry['future_trajectory'], (5, 2), 'actual future trajectory')
        unique = stride_indices(len(dataset), settings['train_window_stride'])
        full_draws, full_sampling = balanced_training_indices(dataset, indices=unique,
            draws=settings['draws_per_epoch'], seed=settings['seed'])
        pilot_unique = []
        for values in full_sampling['category_indices'].values():
            if not values:
                raise ValueError('Pilot requires examples of stop, straight, left and right')
            positions = np.linspace(0, len(values)-1, min(settings['pilot_windows_per_category'], len(values)), dtype=int)
            pilot_unique.extend(values[int(i)] for i in positions)
        pilot_unique = sorted(set(pilot_unique))
        pilot_draws, pilot_sampling = balanced_training_indices(dataset, indices=pilot_unique,
            draws=settings['draws_per_epoch'], seed=settings['seed'])
        write(output/'sampling.json', dict(full=full_sampling, pilot=pilot_sampling,
            unique_train_evaluation_indices=unique, pilot_evaluation_indices=pilot_unique,
            note='Training resamples rare categories. Evaluation indices are unique, never oversampled.'))
        provenance = dict(backend=backend, source=source_hashes(ROOT), datasets={'train': dataset.describe()},
            repair_sources={str(p.relative_to(ROOT)): sha256(p) for p in (
                Path(__file__).resolve(), ROOT/'tools/morai/build_global_path_instructions.py')})
        write(output/'provenance.json', provenance)
        status('loading_model', verified_train_frames=len(dataset.entries), pilot_windows=len(pilot_unique))
        model = VLDriveTrainingModel(config, device=settings['device'])
        transfer = warm_start_model_checkpoint(settings['warm_start'], model, provenance=provenance,
            reset_prefixes=tuple(settings['reset_prefixes']),
            transfer_reason='Authorized MORAI downstream repair; same frozen epoch25 visual cache; reset saturated selector, fresh optimizer.')
        write(output/'warm_start.json', transfer)
        write(output/'model.json', model.describe())
        frozen_hash = parameter_hash(model, trainable=False)
        status('preflight')
        checks = preflight(model, dataset[pilot_unique[0]], settings['device'])
        write(output/'preflight.json', checks)
        if not checks['passed']:
            raise RuntimeError('Repair preflight failed')
        optimizer = make_optimizer(model, **settings['optimizer'])

        @torch.no_grad()
        @preserve_rng_state()
        def evaluate(indices, name):
            model.eval()
            rows = []
            for index in indices:
                sample = dataset[index]
                item = batch(sample, settings['device'])
                with autocast(model):
                    prediction = model.predict(item['model_inputs'], compute_auxiliary=False)
                gt = item['training_targets']['future_trajectory'][0, -1].float().cpu().numpy()
                pred = prediction['trajectory'][0, -1].float().cpu().numpy()
                rows.append(dict(window_index=index, sample_id=sample['metadata']['sample_ids'][-1],
                    instruction=sample['model_inputs']['instruction'], category=category(gt),
                    actual_future=gt.tolist(), prediction=pred.tolist(),
                    errors_m=np.linalg.norm(pred-gt, axis=1).tolist(),
                    selected_count=prediction['selected_count'][0].cpu().tolist()))
            groups = summarize(rows)
            write(output/(name+'_predictions.json'), rows)
            write(output/(name+'_metrics.json'), groups)
            model.train()
            return groups

        status('pilot_baseline')
        baseline = evaluate(pilot_unique, 'pilot_before')
        status('pilot_training', pilot_before=baseline)
        schedule_updates = settings['pilot_max_updates']+settings['full_epochs']*len(full_draws)
        current = 0
        last_gate = None
        pilot_logs = []
        while current < settings['pilot_max_updates']:
            stop = min(current+settings['pilot_check_every'], settings['pilot_max_updates'])
            log_path = output/('pilot_training_%06d.jsonl' % current)
            def pilot_progress(step):
                if step % 10 == 0:
                    status('pilot_training', updates=step, pilot_max_updates=settings['pilot_max_updates'])
            train(model, optimizer, dataset, pilot_draws, start_update=current, updates=stop,
                schedule_updates=schedule_updates, seed=settings['seed'], device=settings['device'],
                log_path=log_path, lr=settings['optimizer']['lr'], on_update=pilot_progress)
            current = stop
            pilot_logs.extend(json.loads(line) for line in log_path.read_text().splitlines() if line)
            recent = pilot_logs[-min(64, len(pilot_logs)):]
            telemetry = {k: float(np.mean([r[k] for r in recent])) for k in
                         ('sampled_retention', 'soft_retention', 'saturated_keep_fraction')}
            status('pilot_evaluating', updates=current)
            after = evaluate(pilot_unique, 'pilot_%06d' % current)
            last_gate = check_gate(baseline, after, telemetry, settings['gate'], current)
            write(output/('gate_%06d.json' % current), dict(last_gate, metrics=after, updates=current))
            save_model_checkpoint(output/'pilot-model-last.pt', model, updates=current,
                provenance=provenance, settings=settings, metrics=after)
            status('pilot_checked', updates=current, gate=last_gate, pilot_after=after)
            if last_gate['passed']:
                break
        write(output/'pilot_result.json', dict(before=baseline, after=after, gate=last_gate, updates=current))
        save_checkpoint(output/'pilot-checkpoint.pt', model, optimizer, updates=current,
                        provenance=provenance, settings=settings)
        if not last_gate['passed'] or args.pilot_only:
            status('pilot_passed' if last_gate['passed'] else 'pilot_gate_failed', updates=current,
                   full_training_started=False)
            return

        full_start, full_stop = current, current+settings['full_epochs']*len(full_draws)
        # The global LR schedule was fixed before the pilot; early gate passage
        # simply stops before the originally reserved maximum schedule endpoint.
        settings['completed_pilot_updates'] = full_start
        settings['full_stop_update'] = full_stop
        write(output/'resolved_config.json', settings)
        full_eval_subset = sorted(set(pilot_unique + unique[::max(1, len(unique)//96)]))
        full_before = evaluate(unique, 'full_before')
        status('full_training', updates=current, full_start_update=full_start, full_stop_update=full_stop,
               full_before=full_before, gate=last_gate)

        def progress(step):
            if step % 10 == 0:
                status('full_training', updates=step, remaining_updates=full_stop-step)
            interval = settings['checkpoint_every']
            if interval > 0 and step % interval == 0 and step < full_stop:
                path = periodic_checkpoint_path(output, step, interval, settings['checkpoint_slots'])
                status('checkpointing', updates=step, checkpoint=str(path))
                save_checkpoint(path, model, optimizer, updates=step, provenance=provenance, settings=settings)
                status('full_training', updates=step)
            if (step-full_start) % len(full_draws) == 0:
                epoch = (step-full_start)//len(full_draws)
                metrics = evaluate(full_eval_subset, 'epoch_%02d_train' % epoch)
                save_model_checkpoint(output/'model-last.pt', model, updates=step,
                    provenance=provenance, settings=settings, metrics=metrics)
                write(output/'latest_train_metrics.json', dict(epoch=epoch, updates=step, metrics=metrics))
                status('full_training', updates=step, epoch=epoch, latest_train_metrics=metrics)

        for epoch in range(settings['full_epochs']):
            epoch_start = full_start+epoch*settings['draws_per_epoch']
            draws, sampling = balanced_training_indices(dataset, indices=unique,
                draws=settings['draws_per_epoch'], seed=settings['seed']+epoch)
            write(output/('epoch_%02d_sampling.json' % (epoch+1)), dict(sampling, indices=draws))
            train(model, optimizer, dataset, draws, start_update=epoch_start,
                updates=epoch_start+len(draws), schedule_updates=schedule_updates,
                sampling_step_offset=epoch_start,
                seed=settings['seed'], device=settings['device'],
                log_path=output/('training_epoch_%02d.jsonl' % (epoch+1)),
                lr=settings['optimizer']['lr'], on_update=progress)
        status('final_evaluation', updates=full_stop)
        final = evaluate(unique, 'train_final')
        save_model_checkpoint(output/'model-last.pt', model, updates=full_stop,
            provenance=provenance, settings=settings, metrics=final)
        save_checkpoint(output/'checkpoint.pt', model, optimizer, updates=full_stop,
                        provenance=provenance, settings=settings)
        if parameter_hash(model, trainable=False) != frozen_hash or sha256(Path(config.visual_checkpoint)) != dataset.checkpoint_sha256:
            raise RuntimeError('Frozen assets changed during downstream training')
        write(output/'report.json', dict(phase='complete', pilot=last_gate, full_before=full_before,
            after_train=final, total_updates=full_stop, full_updates=full_stop-full_start,
            encoder_retrained=False, frozen_unchanged=True,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
        status('complete', updates=full_stop, final_train=final)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
