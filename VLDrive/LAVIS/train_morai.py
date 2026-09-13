#!/usr/bin/env python3
"""Complete offline MORAI model: preflight/train/evaluate/resume, one graph."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[0]
import os
sys.path.insert(0, str(ROOT))
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, add_path_arguments, resolve_settings
configure_runtime()


def write_json(path, data):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'train', 'evaluate'), default='train')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--updates', type=int, help='Total optimizer updates, including resumed updates')
    parser.add_argument('--train-windows', type=int, help='0 selects every valid training window')
    parser.add_argument('--val-windows', type=int, help='0 selects every independent validation window')
    add_path_arguments(parser)
    args = parser.parse_args()
    settings = resolve_settings(json.loads(args.config.read_text()), args)
    for name in ('updates', 'train_windows', 'val_windows'):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if args.output.exists():
        raise ValueError('Use a new output directory; existing experiments are never overwritten')
    output = args.output.resolve()
    if ROOT not in output.parents:
        raise ValueError('Training outputs must stay inside LAVIS')
    output.mkdir(parents=True)
    write_json(output / 'resolved_config.json', settings)
    from lavis.common.morai_runtime import bootstrap
    backend = bootstrap(ROOT.parent)
    import torch
    from lavis.datasets.datasets.morai.training_dataset import MoraiTrainingDataset
    from lavis.models.vldrive.morai_model import MoraiDrivingModel, MoraiDrivingConfig
    from lavis.runners.morai_trainer import (seed_all, subset_indices, source_hashes, parameter_hash,
        preflight, evaluate, make_optimizer, train, save_checkpoint, restore_checkpoint)
    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    device = settings.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this configured raw visual experiment')
    config = MoraiDrivingConfig(**settings['model'])
    config.language_checkpoint = str((ROOT / config.language_checkpoint).resolve())
    config.visual_checkpoint = str((ROOT / config.visual_checkpoint).resolve())
    dataset_root = (ROOT / settings['dataset']).resolve()
    split_file = dataset_root / settings['split_file']
    instruction_file = (ROOT / settings['instructions']).resolve() if settings.get('instructions') else None
    print('Validating complete train/validation artifacts, teacher XY/speed, splits and receive causality...', flush=True)
    datasets = {split: MoraiTrainingDataset(dataset_root, split_file=split_file, split=split,
        sequence_length=settings['sequence_length'], input_mode=config.input_mode,
        instructions=instruction_file) for split in ('train', 'val')}
    if not config.visual_sha256:
        config.visual_sha256 = datasets['train'].base.checkpoint_sha256
    elif config.visual_sha256 != datasets['train'].base.checkpoint_sha256:
        raise ValueError('Configured visual checkpoint differs from dataset provenance')
    indices = {split: subset_indices(len(datasets[split]), settings[split + '_windows']) for split in datasets}
    provenance = dict(backend=backend, source=source_hashes(ROOT), datasets={k: v.describe() for k, v in datasets.items()})
    write_json(output / 'provenance.json', provenance)
    write_json(output / 'window_indices.json', indices)
    print('Loading actual pretrained visual encoder and lightweight LM...', flush=True)
    model = MoraiDrivingModel(config, device=device)
    write_json(output / 'model.json', model.describe())
    optimizer = make_optimizer(model, **settings.get('optimizer', {}))
    start_update = 0
    if args.checkpoint:
        saved = restore_checkpoint(args.checkpoint, model, optimizer, provenance=provenance)
        start_update = saved['updates']
        if args.mode != 'evaluate' and saved['settings'] != {k: v for k, v in settings.items() if k != 'updates'}:
            raise ValueError('Resume requires the same data selection, optimizer and random schedule')
    report = dict(schema='morai-goal-stack-report-v1', full_goal_graph=config.input_mode == 'raw'
        and config.language_mode == 'ddia' and config.ccdp and config.mefa,
        start_update=start_update, semantic_annotations=datasets['train'].describe()['semantic_annotation_count'],
        sample_counts={k: len(v) for k, v in indices.items()})
    if args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError('Evaluation requires a trained checkpoint')
        def evaluation_progress(completed, total):
            progress = dict(phase='evaluating', completed_windows=completed, total_windows=total)
            write_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
        evaluation_progress(0, len(indices['val']))
        report['validation'] = evaluate(model, datasets['val'], indices['val'], device,
            save_predictions=output / 'val_predictions.json', on_progress=evaluation_progress)
        report.update(trainable_hash=parameter_hash(model, trainable=True),
                      frozen_hash=parameter_hash(model, trainable=False), optimizer_updates=0)
    else:
        if settings['updates'] < start_update:
            raise ValueError('Total updates cannot precede resumed step')
        print('Checking real-data gradients, teacher input rejection and future-to-past causality...', flush=True)
        report['preflight'] = preflight(model, datasets['train'][indices['train'][0]], device)
        write_json(output / 'preflight.json', report['preflight'])
        if not report['preflight']['passed']:
            raise RuntimeError('Goal-stack preflight failed; inspect ' + str(output / 'preflight.json'))
        if args.mode == 'train':
            before_hash = parameter_hash(model, trainable=True)
            frozen_hash = parameter_hash(model, trainable=False)
            report['before'] = {k: evaluate(model, datasets[k], indices[k], device) for k in datasets}
            write_json(output / 'progress.json', dict(report, phase='before_training'))
            # Preflight uses Gumbel draws; resume must restore the exact saved RNG cursor afterwards.
            if args.checkpoint:
                restore_checkpoint(args.checkpoint, model, optimizer, provenance=provenance)
            def periodic_checkpoint(step):
                interval = settings.get('checkpoint_every', 80)
                if interval > 0 and step % interval == 0 and step < settings['updates']:
                    path = output / 'checkpoint_{:06d}.pt'.format(step)
                    save_checkpoint(path, model, optimizer, updates=step, provenance=provenance,
                        settings={k: v for k, v in settings.items() if k != 'updates'})
                    write_json(output / 'progress.json', dict(report, phase='training', completed_updates=step,
                                                              checkpoint=str(path)))
            train(model, optimizer, datasets['train'], indices['train'], start_update=start_update,
                  updates=settings['updates'], seed=settings['seed'], device=device, log_path=output / 'training.jsonl',
                  on_update=periodic_checkpoint)
            # Save completed optimization BEFORE potentially long independent evaluation.
            save_checkpoint(output / 'checkpoint.pt', model, optimizer, updates=settings['updates'], provenance=provenance,
                            settings={k: v for k, v in settings.items() if k != 'updates'})
            write_json(output / 'progress.json', dict(report, phase='optimization_complete', completed_updates=settings['updates']))
            report['after'] = {k: evaluate(model, datasets[k], indices[k], device,
                save_predictions=output / (k + '_predictions.json')) for k in datasets}
            report.update(optimizer_updates=settings['updates'] - start_update,
                total_updates=settings['updates'], trainable_hash=parameter_hash(model, trainable=True),
                frozen_hash=parameter_hash(model, trainable=False))
            report['checks'] = dict(trainable_changed=report['trainable_hash'] != before_hash,
                frozen_unchanged=report['frozen_hash'] == frozen_hash,
                small_train_objective_decreased=report['after']['train']['objective'] < report['before']['train']['objective'],
                finite_metrics=all(torch.isfinite(torch.tensor(v['objective'])).item() for v in report['after'].values()))
            if not report['checks']['frozen_unchanged']:
                raise RuntimeError('Frozen parameters or buffers changed')
        else:
            report['optimizer_updates'] = 0
    if device == 'cuda':
        report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    report['completed_wall_time'] = time.time()
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
