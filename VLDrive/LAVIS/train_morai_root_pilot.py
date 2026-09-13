#!/usr/bin/env python3
"""One arm of the authorized RoPE-fixed, frozen-epoch25 CCDP comparison.

Fresh local pretrained TinyLlama/BERT, original instructions and objectives.
This bounded runner never auto-starts a full run. Resume is strict, not transfer.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, add_path_arguments, resolve_settings, sha256
configure_runtime()
from tools.morai.root_pilot_checks import category, summarize, gate
ROOT = Path(__file__).resolve().parent


def write(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('legacy', 'retained_output_st'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--verify-only', action='store_true', help='Real GPU forward/backward checks, zero optimizer steps')
    add_path_arguments(parser)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Direct CLI and detached launcher share the same lock. The launcher passes
    # its locked descriptor, keeping ownership even if its parent exits.
    import fcntl
    lock_path = ROOT / '.runtime/training_runs/last_encoder.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if args.lock_fd is None:
        training_lock = lock_path.open('a')
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        training_lock = os.fdopen(os.dup(args.lock_fd), 'a')
        if os.fstat(training_lock.fileno()).st_ino != lock_path.stat().st_ino:
            raise ValueError('Inherited training lock does not match expected file')
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = resolve_settings(json.loads(args.config.read_text()), args)
    settings['split_file'] = str((args.path_root / settings['split_file']).resolve())
    settings['model'].update(selector_task_gradient_mode=args.arm,
        training_variant='paper_independent' if args.arm == 'legacy' else 'paper_objective_st_probe')
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a NEW output directory inside LAVIS')
    output.mkdir(parents=True)
    write(output / 'resolved_config.json', settings)
    started = time.monotonic()
    state = dict(pid=os.getpid(), arm=args.arm, output=str(output), validation_used=False,
                 fresh_pretrained_initialization=True, encoder_retrained=False)

    def status(phase, **values):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_s=time.monotonic() - started, **values)
        write(output / 'progress.json', state)
        print(json.dumps(state), flush=True)

    try:
        import numpy as np
        import torch
        from lavis.common.morai_runtime import bootstrap
        from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
        from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
        from lavis.runners.vldrive_trainer import (seed_all, source_hashes, stride_indices,
            balanced_training_indices, make_optimizer, preflight, train, save_checkpoint,
            restore_checkpoint, save_model_checkpoint, parameter_hash, preserve_rng_state)
        from lavis.runners.morai_trainer import batch, autocast
        torch.set_num_threads(settings['cpu_threads'])
        seed_all(settings['seed'])
        status('validating_data')
        backend = bootstrap(ROOT.parent)
        dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
            split_file=settings['split_file'], instructions=settings['instructions'], split='train',
            sequence_length=settings['sequence_length'], instruction_boundary_policy='exclude')
        config = VLDriveTrainingConfig(**settings['model'])
        config.validate()
        if sha256(Path(config.visual_checkpoint)) != dataset.checkpoint_sha256:
            raise ValueError('Frozen encoder/cache mismatch')
        for entry in dataset.entries:
            dataset._array(entry['visual_feature'], (106, 256), 'visual feature')
            dataset._array(entry['future_trajectory'], (5, 2), 'actual future trajectory')
        unique = stride_indices(len(dataset), settings['train_window_stride'])
        _, categories = balanced_training_indices(dataset, indices=unique, draws=32, seed=settings['seed'])
        pilot = []
        for values in categories['category_indices'].values():
            positions = np.linspace(0, len(values) - 1, min(settings['windows_per_category'], len(values)), dtype=int)
            pilot.extend(values[int(p)] for p in positions)
        pilot = sorted(set(pilot))
        draws, draw_info = balanced_training_indices(dataset, indices=pilot,
            draws=settings['max_updates'], seed=settings['seed'])
        sampling = dict(evaluation_indices=pilot, training_draws=draws, detail=draw_info,
                        instruction_boundary_policy='exclude', shared_by_both_arms=True)
        write(output / 'sampling.json', sampling)
        provenance = dict(backend=backend, source=source_hashes(ROOT), datasets={'train': dataset.describe()},
            pilot_sources={str(p.relative_to(ROOT)): sha256(p) for p in (
                Path(__file__).resolve(), ROOT / 'tools/morai/root_pilot_checks.py')})
        write(output / 'provenance.json', provenance)
        status('loading_fresh_pretrained_model', verified_frames=len(dataset.entries))
        model = VLDriveTrainingModel(config, device=settings['device'])
        initial_hash = parameter_hash(model, trainable=True)
        frozen_hash = parameter_hash(model, trainable=False)
        write(output / 'model.json', model.describe())
        curved_index = draw_info['category_indices']['moving_left'][0]
        status('gpu_preflight')
        checks = preflight(model, dataset[curved_index], settings['device'])
        write(output / 'preflight.json', checks)
        if not checks['passed']:
            raise RuntimeError('Actual GPU gradient/causality/input preflight failed')

        @preserve_rng_state()
        def task_probe():
            model.train()
            model.zero_grad(set_to_none=True)
            item = batch(dataset[curved_index], settings['device'])
            with autocast(model):
                prediction = model.predict(item['model_inputs'], compute_auxiliary=True)
                waypoint = model.loss(prediction, item['training_targets'])['waypoint_loss']
            selector = [p for n, p in model.named_parameters() if n.startswith('connector.ccdp.') and p.requires_grad]
            values = {}
            for name, loss in (
                    ('waypoint', waypoint),
                    ('weighted_pruning', config.pruning_weight * prediction['pruning_loss']),
                    ('weighted_reconstruction', config.reconstruction_weight * prediction['reconstruction_loss'])):
                # HF 4.31 uses reentrant checkpointing, which does not support
                # autograd.grad(inputs=...). Exercise the real backward API.
                model.zero_grad(set_to_none=True)
                loss.backward(retain_graph=name != 'weighted_reconstruction')
                values['selector_' + name + '_l2'] = float(sum(
                    (p.grad.detach().float().square().sum() for p in selector if p.grad is not None),
                    waypoint.new_zeros((), dtype=torch.float32)).sqrt())
            values.update(trajectory=prediction['trajectory'].detach().float().cpu().tolist(),
                          waypoint_loss=float(waypoint.detach()), window_index=curved_index,
                          sampled_retention=float(prediction['sampled_retention'].mean()),
                          note='Actual eleven-frame waypoint L1 through TinyLlama/DDIA, not a linear token probe')
            model.zero_grad(set_to_none=True)
            return values

        initial_probe = task_probe()
        write(output / 'initial_task_probe.json', initial_probe)
        if not all(np.isfinite(v) for k, v in initial_probe.items() if k.endswith('_l2')):
            raise RuntimeError('Nonfinite actual task/auxiliary selector gradients')

        @torch.no_grad()
        @preserve_rng_state()
        def evaluate(name):
            model.eval()
            rows, counts = [], []
            for index in pilot:
                sample = dataset[index]
                item = batch(sample, settings['device'])
                with autocast(model):
                    prediction = model.predict(item['model_inputs'], compute_auxiliary=False)
                gt = item['training_targets']['future_trajectory'][0, -1].float().cpu().numpy()
                pred = prediction['trajectory'][0, -1].float().cpu().numpy()
                counts.extend(prediction['selected_count'][0].cpu().tolist())
                rows.append(dict(window_index=index, sample_id=sample['metadata']['sample_ids'][-1],
                    instruction=sample['model_inputs']['instruction'], category=category(gt.tolist()),
                    actual_future=gt.tolist(), prediction=pred.tolist(), errors_m=np.linalg.norm(pred - gt, axis=1).tolist()))
            metrics = summarize(rows)
            write(output / (name + '_predictions.json'), rows)
            write(output / (name + '_metrics.json'), metrics)
            model.train()
            return metrics, float(np.mean(counts)) / 106.

        status('initial_train_evaluation')
        before, _ = evaluate('before')
        controlled = copy.deepcopy(settings)
        for key in ('selector_task_gradient_mode', 'training_variant'):
            controlled['model'].pop(key, None)
        common = dict(arm=args.arm, seed=settings['seed'], optimizer=settings['optimizer'],
            controlled_settings=controlled,
            schedule_updates=settings['schedule_updates'], initial_trainable_sha256=initial_hash,
            sampling=sampling, provenance=provenance, initial_task_probe=initial_probe, before=before)
        write(output / 'initial_report.json', common)
        if args.verify_only:
            write(output / 'report.json', dict(common, passed=True, optimizer_updates=0))
            status('verified_zero_updates', optimizer_updates=0)
            return
        optimizer = make_optimizer(model, **settings['optimizer'])
        current = 0
        if args.checkpoint:
            checkpoint = restore_checkpoint(args.checkpoint, model, optimizer, provenance=provenance)
            if checkpoint['settings'] != settings:
                raise ValueError('Strict resume requires identical sampling, arm, config and schedule')
            current = int(checkpoint['updates'])
            del checkpoint
        if not 0 <= current < settings['max_updates']:
            raise ValueError('No remaining bounded pilot updates')
        while current < settings['max_updates']:
            stop = min(current + settings['check_every'], settings['max_updates'])
            log_path = output / ('training_%06d.jsonl' % current)
            status('pilot_training', updates=current, max_updates=settings['max_updates'])
            def progress(step):
                if step % 10 == 0:
                    status('pilot_training', updates=step, max_updates=settings['max_updates'])
            train(model, optimizer, dataset, draws, start_update=current, updates=stop,
                schedule_updates=settings['schedule_updates'], seed=settings['seed'], device=settings['device'],
                log_path=log_path, lr=settings['optimizer']['lr'], on_update=progress)
            current = stop
            recent = [json.loads(line) for line in log_path.read_text().splitlines()][-64:]
            telemetry = {k: float(np.mean([row[k] for row in recent])) for k in
                         ('sampled_retention', 'soft_retention', 'saturated_keep_fraction')}
            status('pilot_evaluation', updates=current)
            after, inference_retention = evaluate('after_%06d' % current)
            telemetry['inference_retention'] = inference_retention
            result = gate(before, after, telemetry, settings['gate'], current)
            report = dict(common, after=after, gate=result, updates=current,
                          frozen_unchanged=parameter_hash(model, trainable=False) == frozen_hash,
                          elapsed_s=time.monotonic() - started)
            write(output / 'report.json', report)
            write(output / ('gate_%06d.json' % current), result)
            save_model_checkpoint(output / 'model-last.pt', model, updates=current,
                provenance=provenance, settings=settings, metrics=after)
            # Two fixed slots preserve the preceding complete optimizer/RNG state.
            slot = (current // settings['check_every'] - 1) % 2
            save_checkpoint(output / ('checkpoint_slot%d.pt' % slot), model, optimizer,
                updates=current, provenance=provenance, settings=settings)
            if not report['frozen_unchanged']:
                raise RuntimeError('Frozen weights changed')
            # No early success shortcut: both arms receive exactly the same updates.
            status('pilot_checkpointed', updates=current, gate=result)
        status('pilot_passed' if result['passed'] else 'pilot_gate_failed', updates=current, gate=result)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
