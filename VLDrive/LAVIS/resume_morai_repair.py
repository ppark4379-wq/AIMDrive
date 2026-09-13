#!/usr/bin/env python3
"""Resume a gated MORAI repair full-phase checkpoint into a new output directory.

No warm start, selector reset, pilot bypass, configuration change, or validation
sampling is allowed. Model, optimizer, RNG, cosine schedule and the within-epoch
balanced sampling cursor are restored. Source and frozen-asset hashes must match.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from train_morai_repair import ROOT, read, write, category, summarize


def acquire_training_lock(root=ROOT):
    """Share the launcher's lifetime lock before allocating any GPU resources."""
    import fcntl
    root = Path(root).resolve()
    path = (root / '.runtime/training_runs/last_encoder.lock').resolve()
    if root not in path.parents:
        raise ValueError('Training lock must remain inside the selected LAVIS checkout')
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = path.open('a')
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        guard.close()
        raise RuntimeError('Another last-encoder training worker is already running') from None
    except Exception:
        guard.close()
        raise
    return guard


def validate_full_phase(settings, pilot, *, updates=None):
    """Reject pilot snapshots and accidental changes to the training horizon."""
    gate = pilot.get('gate', {})
    if gate.get('passed') is not True or not gate.get('checks') or not all(gate['checks'].values()):
        raise ValueError('Recovery requires the original successful pilot gate')
    required = ('completed_pilot_updates', 'full_stop_update', 'full_epochs', 'draws_per_epoch')
    if any(not isinstance(settings.get(key), int) or settings[key] < 1 for key in required):
        raise ValueError('Only full-phase settings with an explicit horizon can resume')
    start = settings['completed_pilot_updates']
    stop = settings['full_stop_update']
    if (pilot.get('updates') != start
            or stop != start + settings['full_epochs'] * settings['draws_per_epoch']):
        raise ValueError('Pilot/full-phase update contract changed')
    if settings.get('checkpoint_every') != 200 or settings.get('checkpoint_slots') != 2:
        raise ValueError('Recovery requires the documented 200-update/two-slot checkpoint policy')
    if settings.get('model', {}).get('training_variant') != 'morai_repair_v1':
        raise ValueError('Checkpoint is not a MORAI repair experiment')
    if updates is not None and (not isinstance(updates, int) or not start <= updates < stop):
        raise ValueError('Checkpoint must be inside an unfinished full training phase')
    return start, stop


def remaining_epochs(settings, updates):
    """Global update positions with local balanced-draw shuffle boundaries."""
    start = settings['completed_pilot_updates']
    count = settings['draws_per_epoch']
    first_epoch = (updates - start) // count
    for epoch in range(first_epoch, settings['full_epochs']):
        epoch_start = start + epoch * count
        yield dict(epoch=epoch, seed=settings['seed'] + epoch,
                   sampling_step_offset=epoch_start,
                   start_update=max(updates, epoch_start), updates=epoch_start + count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='An integrity-checked full-phase checkpoint_slot0/1.pt')
    parser.add_argument('--output', type=Path, required=True,
                        help='A new directory inside this LAVIS checkout')
    args = parser.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if (ROOT not in checkpoint.parents or not checkpoint.is_file()
            or not checkpoint.with_suffix('.sha256.json').is_file()):
        raise ValueError('Use an existing locally produced full checkpoint and its integrity sidecar')
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new output directory inside LAVIS; do not overwrite a previous run')
    parent = checkpoint.parent
    settings = read(parent / 'resolved_config.json')
    pilot = read(parent / 'pilot_result.json')
    full_start, full_stop = validate_full_phase(settings, pilot)
    original_sampling = read(parent / 'sampling.json')
    original_provenance = read(parent / 'provenance.json')
    # This handle remains live for the entire recovery. It is the exact lock
    # used by launch_repair_training.py and launch_last_encoder_training.py.
    guard = acquire_training_lock()
    output.mkdir(parents=True)
    started = time.monotonic()
    state = dict(pid=os.getpid(), output=str(output), resumed_from=str(checkpoint),
                 encoder_retrained=False, split='train_only')

    def status(phase, **fields):
        import json
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_s=time.monotonic() - started, **fields)
        write(output / 'progress.json', state)
        print(json.dumps(state, allow_nan=False), flush=True)

    try:
        status('validating_resume')
        import numpy as np
        import torch
        from lavis.common.morai_runtime import bootstrap, sha256
        from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
        from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
        from lavis.runners.vldrive_trainer import (seed_all, source_hashes, stride_indices,
            balanced_training_indices, make_optimizer, train, save_checkpoint,
            save_model_checkpoint, periodic_checkpoint_path, parameter_hash, preserve_rng_state)
        from lavis.runners.morai_trainer import restore_checkpoint, batch, autocast

        torch.set_num_threads(settings['cpu_threads'])
        seed_all(settings['seed'])
        backend = bootstrap(ROOT.parent)
        dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
            split_file=settings['split_file'], instructions=settings['instructions'], split='train',
            sequence_length=settings['sequence_length'],
            instruction_boundary_policy=settings['instruction_boundary_policy'])
        config = VLDriveTrainingConfig(**settings['model'])
        if config.input_mode != 'cached' or sha256(Path(config.visual_checkpoint)) != dataset.checkpoint_sha256:
            raise ValueError('Frozen vision encoder/cache contract changed')
        for entry in dataset.entries:
            dataset._array(entry['visual_feature'], (106, 256), 'visual feature')
            dataset._array(entry['future_trajectory'], (5, 2), 'actual future trajectory')
        unique = stride_indices(len(dataset), settings['train_window_stride'])
        _, audit = balanced_training_indices(dataset, indices=unique,
            draws=settings['draws_per_epoch'], seed=settings['seed'])
        if (original_sampling['unique_train_evaluation_indices'] != unique
                or original_sampling['full'] != audit):
            raise ValueError('Original train candidates or balanced sampling classification changed')
        pilot_indices = original_sampling['pilot_evaluation_indices']
        if (not pilot_indices or len(pilot_indices) != len(set(pilot_indices))
                or any(index not in unique for index in pilot_indices)):
            raise ValueError('Invalid original pilot evaluation indices')
        provenance = dict(backend=backend, source=source_hashes(ROOT), datasets={'train': dataset.describe()},
            repair_sources={str(path.relative_to(ROOT)): sha256(path) for path in (
                ROOT / 'train_morai_repair.py', ROOT / 'tools/morai/build_global_path_instructions.py')})
        if provenance != original_provenance:
            raise ValueError('Original repair code, backend, instructions, split, or asset provenance changed')
        status('loading_checkpoint')
        model = VLDriveTrainingModel(config, device=settings['device'])
        optimizer = make_optimizer(model, **settings['optimizer'])
        saved = restore_checkpoint(checkpoint, model, optimizer, provenance=provenance)
        if saved['settings'] != settings:
            raise ValueError('Checkpoint settings do not match the original resolved configuration')
        current = int(saved['updates'])
        validate_full_phase(settings, pilot, updates=current)
        frozen_hash = saved['frozen_hash']
        del saved
        write(output / 'resolved_config.json', settings)
        write(output / 'provenance.json', provenance)
        write(output / 'sampling.json', original_sampling)
        write(output / 'pilot_result.json', pilot)
        write(output / 'model.json', model.describe())
        write(output / 'resume.json', dict(checkpoint=str(checkpoint),
            checkpoint_integrity=read(checkpoint.with_suffix('.sha256.json')),
            previous_output=str(parent), restored_updates=current,
            remaining_updates=full_stop-current, model_optimizer_rng_restored=True,
            selector_reset=False, source_settings_unchanged=True,
            recovery_tool_sha256=sha256(Path(__file__).resolve()),
            sampling_policy='balanced bags at seed+full_epoch, shuffle at base seed with local epoch cursor',
            full_start_update=full_start, full_stop_update=full_stop))

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
            metrics = summarize(rows)
            write(output / (name+'_predictions.json'), rows)
            write(output / (name+'_metrics.json'), metrics)
            model.train()
            return metrics

        full_eval_subset = sorted(set(pilot_indices + unique[::max(1, len(unique)//96)]))
        schedule_updates = settings['pilot_max_updates'] + settings['full_epochs'] * settings['draws_per_epoch']
        restored_updates = current

        def progress(step):
            if step % 10 == 0:
                status('full_training', updates=step, remaining_updates=full_stop-step)
            interval = settings['checkpoint_every']
            if step % interval == 0 and step < full_stop:
                path = periodic_checkpoint_path(output, step, interval, settings['checkpoint_slots'])
                status('checkpointing', updates=step, checkpoint=str(path))
                save_checkpoint(path, model, optimizer, updates=step, provenance=provenance, settings=settings)
                status('full_training', updates=step)
            if (step-full_start) % settings['draws_per_epoch'] == 0:
                epoch = (step-full_start) // settings['draws_per_epoch']
                metrics = evaluate(full_eval_subset, 'epoch_%02d_train' % epoch)
                save_model_checkpoint(output / 'model-last.pt', model, updates=step,
                    provenance=provenance, settings=settings, metrics=metrics)
                write(output / 'latest_train_metrics.json', dict(epoch=epoch, updates=step, metrics=metrics))
                status('full_training', updates=step, epoch=epoch, latest_train_metrics=metrics)

        status('full_training', updates=current, full_start_update=full_start, full_stop_update=full_stop,
               remaining_updates=full_stop-current, gate=pilot['gate'])
        for segment in remaining_epochs(settings, current):
            draws, epoch_audit = balanced_training_indices(dataset, indices=unique,
                draws=settings['draws_per_epoch'], seed=segment['seed'])
            write(output / ('epoch_%02d_sampling.json' % (segment['epoch']+1)), dict(epoch_audit, indices=draws))
            train(model, optimizer, dataset, draws, start_update=segment['start_update'],
                updates=segment['updates'], sampling_step_offset=segment['sampling_step_offset'],
                schedule_updates=schedule_updates, seed=settings['seed'], device=settings['device'],
                log_path=output / ('training_epoch_%02d.jsonl' % (segment['epoch']+1)),
                lr=settings['optimizer']['lr'], on_update=progress)
        status('final_evaluation', updates=full_stop)
        final = evaluate(unique, 'train_final')
        save_model_checkpoint(output / 'model-last.pt', model, updates=full_stop,
            provenance=provenance, settings=settings, metrics=final)
        save_checkpoint(output / 'checkpoint.pt', model, optimizer, updates=full_stop,
                        provenance=provenance, settings=settings)
        if (parameter_hash(model, trainable=False) != frozen_hash
                or sha256(Path(config.visual_checkpoint)) != dataset.checkpoint_sha256):
            raise RuntimeError('Frozen assets changed during resumed downstream training')
        write(output / 'report.json', dict(phase='complete', pilot=pilot['gate'], after_train=final,
            total_updates=full_stop, full_updates=full_stop-full_start,
            resumed_from=str(checkpoint), restored_updates=restored_updates,
            updates_this_invocation=full_stop-restored_updates,
            encoder_retrained=False, frozen_unchanged=True,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
        status('complete', updates=full_stop, final_train=final)
    except Exception as error:
        status('failed', error=str(error))
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    main()
