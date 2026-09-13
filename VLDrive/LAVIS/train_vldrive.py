#!/usr/bin/env python3
"""Paper-equation VLDrive training on instruction-aligned Stage6E MORAI data."""
import argparse
import json
import os
from pathlib import Path

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, add_path_arguments, resolve_settings
configure_runtime()
ROOT = Path(__file__).resolve().parent


def write(path, data):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('train', 'preflight', 'evaluate'), default='train')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--instructions', type=Path, help='Actual sourced navigation event JSONL')
    parser.add_argument('--updates', type=int, help='Stop after this many total updates; keeps the configured cosine schedule')
    add_path_arguments(parser)
    args = parser.parse_args()
    settings = resolve_settings(json.loads(args.config.read_text()), args)
    if args.instructions:
        settings['instructions'] = str(args.instructions.expanduser().resolve())
    if not settings.get('instructions') or not Path(settings['instructions']).is_file():
        raise ValueError('VLDrive requires --instructions with actual sourced navigation events. No text is fabricated from routes or teacher labels.')
    if settings['sequence_length'] < settings['model'].get('memory_size', 10) + 1:
        raise ValueError('Paper preflight requires at least memory_size+1 real frames to exercise Z=10')
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new output directory inside this LAVIS checkout')
    output.mkdir(parents=True)
    write(output / 'resolved_config.json', settings)
    import torch
    from lavis.common.morai_runtime import bootstrap
    backend = bootstrap(ROOT.parent)
    from lavis.datasets.datasets.morai.vldrive_dataset import VLDriveTrainingDataset
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    from lavis.runners.vldrive_trainer import (seed_all, subset_indices, source_hashes, parameter_hash,
        preflight, evaluate, make_optimizer, train, save_checkpoint, restore_checkpoint)
    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    device = settings.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')
    config = VLDriveTrainingConfig(**settings['model'])
    config.validate()
    root = Path(settings['dataset'])
    print('Validating actual artifacts, teacher labels, independent splits and instruction availability...', flush=True)
    datasets = {split: VLDriveTrainingDataset(root, split_file=root / settings['split_file'], split=split,
        sequence_length=settings['sequence_length'], input_mode=config.input_mode,
        instructions=settings['instructions']) for split in ('train', 'val')}
    digest = datasets['train'].base.checkpoint_sha256
    if config.visual_sha256 and config.visual_sha256 != digest:
        raise ValueError('Visual checkpoint differs from dataset provenance')
    config.visual_sha256 = digest
    indices = {split: subset_indices(len(dataset), settings[split + '_windows']) for split, dataset in datasets.items()}
    provenance = dict(backend=backend, source=source_hashes(ROOT), datasets={k: v.describe() for k, v in datasets.items()})
    write(output / 'provenance.json', provenance)
    write(output / 'window_indices.json', indices)
    print('Loading frozen visual encoder and full trainable pretrained TinyLlama/DDIA...', flush=True)
    model = VLDriveTrainingModel(config, device=device)
    write(output / 'model.json', model.describe())
    optimizer = make_optimizer(model, **settings.get('optimizer', {}))
    schedule_updates = settings.get('epochs', 15) * len(indices['train'])
    updates = schedule_updates if args.updates is None else args.updates
    start_update = 0
    if args.checkpoint:
        state = restore_checkpoint(args.checkpoint, model, optimizer, provenance=provenance)
        start_update = state['updates']
        if state['settings'] != settings:
            raise ValueError('Resume requires identical settings, sampling and cosine schedule')
        del state
    report = dict(training_path='frozen_encoder_dynamic_ccdp_mefa_tinyllama_ddia_l1',
                  start_update=start_update, semantic_annotations=True, schedule_updates=schedule_updates)
    if args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError('Evaluation requires --checkpoint')
        report['validation'] = evaluate(model, datasets['val'], indices['val'], device,
                                        save_predictions=output / 'val_predictions.json')
    else:
        if not start_update <= updates <= schedule_updates:
            raise ValueError('Update limit must be within the configured schedule')
        print('Checking gradients, temporal causality and forbidden teacher inputs...', flush=True)
        report['preflight'] = preflight(model, datasets['train'][indices['train'][0]], device)
        write(output / 'preflight.json', report['preflight'])
        if not report['preflight']['passed']:
            raise RuntimeError('VLDrive preflight failed; inspect preflight.json')
        if args.mode == 'train':
            frozen = parameter_hash(model, trainable=False)
            before = parameter_hash(model, trainable=True)
            report['before'] = {k: evaluate(model, datasets[k], indices[k], device) for k in datasets}
            if args.checkpoint:
                state = restore_checkpoint(args.checkpoint, model, optimizer, provenance=provenance)
                del state  # Release loaded state before full-LM training.
            def periodic(step):
                interval = settings.get('checkpoint_every', 100)
                if interval > 0 and step % interval == 0 and step < updates:
                    save_checkpoint(output / ('checkpoint_%06d.pt' % step), model, optimizer,
                                    updates=step, provenance=provenance, settings=settings)
            train(model, optimizer, datasets['train'], indices['train'], start_update=start_update,
                updates=updates, schedule_updates=schedule_updates, seed=settings['seed'], device=device,
                log_path=output / 'training.jsonl', lr=settings.get('optimizer', {}).get('lr', 1e-5), on_update=periodic)
            save_checkpoint(output / 'checkpoint.pt', model, optimizer, updates=updates,
                            provenance=provenance, settings=settings)
            report['after'] = {k: evaluate(model, datasets[k], indices[k], device,
                save_predictions=output / (k + '_predictions.json')) for k in datasets}
            report.update(optimizer_updates=updates - start_update,
                trainable_changed=parameter_hash(model, trainable=True) != before,
                frozen_unchanged=parameter_hash(model, trainable=False) == frozen)
            if not report['frozen_unchanged']:
                raise RuntimeError('Frozen encoder/base weights changed')
    if device == 'cuda':
        report['peak_gpu_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    write(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
