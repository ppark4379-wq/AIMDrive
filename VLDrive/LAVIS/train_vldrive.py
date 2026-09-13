#!/usr/bin/env python3
"""Paper-equation VLDrive training on instruction-aligned MORAI data."""
import argparse
import json
import os
from pathlib import Path

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, add_path_arguments, resolve_settings, sha256
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
    parser.add_argument('--instructions', type=Path, help='Verified global-path navigation instruction JSONL')
    parser.add_argument('--feature-cache', type=Path, help='Completed frozen [106,256] token cache')
    parser.add_argument('--split-file', type=Path, help='Chronological MORAI train/purge/validation split')
    parser.add_argument('--updates', type=int, help='Stop after this many total updates; keeps the configured cosine schedule')
    add_path_arguments(parser)
    args = parser.parse_args()
    settings = resolve_settings(json.loads(args.config.read_text()), args)
    if args.instructions:
        settings['instructions'] = str(args.instructions.expanduser().resolve())
    if args.feature_cache:
        settings['feature_cache'] = str(args.feature_cache.expanduser().resolve())
    if args.split_file:
        settings['split_file'] = str(args.split_file.expanduser().resolve())
    if not settings.get('instructions') or not Path(settings['instructions']).is_file():
        raise ValueError('VLDrive requires verified navigation instructions generated from the recorded global path.')
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
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    from lavis.runners.vldrive_trainer import (seed_all, subset_indices, stride_indices, source_hashes, parameter_hash,
        preflight, evaluate, make_optimizer, train, save_checkpoint, restore_checkpoint,
        save_model_checkpoint, restore_model_checkpoint, periodic_checkpoint_path)
    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    device = settings.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')
    config = VLDriveTrainingConfig(**settings['model'])
    config.validate()
    root = Path(settings['dataset'])
    print('Validating actual artifacts, teacher labels, independent splits and instruction availability...', flush=True)
    dataset_format = settings.get('dataset_format', 'stage6e')
    if dataset_format == 'morai_expert_cache_v1':
        if config.input_mode != 'cached':
            raise ValueError('MORAI expert cache path requires model.input_mode=cached')
        from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
        split_path = Path(settings['split_file'])
        if not split_path.is_absolute():
            split_path = (ROOT / split_path).resolve()
        cache = Path(settings.get('feature_cache', ''))
        if not cache.is_dir():
            raise ValueError('Completed MORAI feature cache is missing: ' + str(cache))
        datasets = {split: MoraiExpertVLDriveDataset(
            root, cache_root=cache, split_file=split_path, split=split,
            sequence_length=settings['sequence_length'], instructions=settings['instructions'],
        ) for split in ('train', 'val')}
    elif dataset_format == 'stage6e':
        from lavis.datasets.datasets.morai.vldrive_dataset import VLDriveTrainingDataset
        datasets = {split: VLDriveTrainingDataset(root, split_file=root / settings['split_file'], split=split,
            sequence_length=settings['sequence_length'], input_mode=config.input_mode,
            instructions=settings['instructions']) for split in ('train', 'val')}
    else:
        raise ValueError('Unknown VLDrive dataset_format: ' + str(dataset_format))
    digest = datasets['train'].checkpoint_sha256 if hasattr(datasets['train'], 'checkpoint_sha256') else datasets['train'].base.checkpoint_sha256
    visual_asset = Path(config.visual_checkpoint)
    if not visual_asset.is_file() or sha256(visual_asset) != digest:
        raise ValueError('Prepared visual checkpoint does not match the frozen token cache')
    if config.visual_sha256 and config.visual_sha256 != digest:
        raise ValueError('Visual checkpoint differs from dataset provenance')
    config.visual_sha256 = digest
    indices = {}
    for split, dataset in datasets.items():
        stride_key = split + '_window_stride'
        if stride_key in settings:
            indices[split] = stride_indices(len(dataset), settings[stride_key])
        else:
            indices[split] = subset_indices(len(dataset), settings[split + '_windows'])
    provenance = dict(backend=backend, source=source_hashes(ROOT), datasets={k: v.describe() for k, v in datasets.items()})
    write(output / 'provenance.json', provenance)
    write(output / 'window_indices.json', indices)
    print('Loading cached frozen visual tokens and full trainable pretrained TinyLlama/DDIA...', flush=True)
    model = VLDriveTrainingModel(config, device=device)
    write(output / 'model.json', model.describe())
    schedule_updates = settings.get('epochs', 15) * len(indices['train'])
    updates = schedule_updates if args.updates is None else args.updates
    start_update = 0
    optimizer = None
    if args.checkpoint and args.mode != 'evaluate':
        optimizer = make_optimizer(model, **settings.get('optimizer', {}))
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
        restore_model_checkpoint(args.checkpoint, model, provenance=provenance)
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
            if optimizer is None:
                optimizer = make_optimizer(model, **settings.get('optimizer', {}))
            frozen = parameter_hash(model, trainable=False)
            before = parameter_hash(model, trainable=True)
            report['before'] = {k: evaluate(model, datasets[k], indices[k], device) for k in datasets}
            def periodic(step):
                interval = settings.get('checkpoint_every', 100)
                if interval > 0 and step % interval == 0 and step < updates:
                    path = periodic_checkpoint_path(output, step, interval, settings.get('checkpoint_slots', 0))
                    print(json.dumps(dict(event='checkpoint_start', updates=step, path=str(path))), flush=True)
                    save_checkpoint(path, model, optimizer,
                                    updates=step, provenance=provenance, settings=settings)
                    print(json.dumps(dict(event='checkpoint_complete', updates=step, path=str(path))), flush=True)
            baseline_row = dict(epoch=start_update // len(indices['train']), updates=start_update,
                                validation=report['before']['val'])
            best = dict(fde_m=report['before']['val']['fde_m'], epoch=baseline_row['epoch'],
                        metrics=baseline_row, scope='current_invocation_including_starting_weights')
            save_model_checkpoint(output / 'model-best.pt', model, updates=start_update,
                provenance=provenance, settings=settings, metrics=baseline_row)
            epoch_metrics = output / 'epoch_metrics.jsonl'
            def end_epoch(epoch, step):
                validation = evaluate(model, datasets['val'], indices['val'], device)
                row = dict(epoch=int(epoch) + 1, updates=int(step), validation=validation)
                with epoch_metrics.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
                save_model_checkpoint(output / 'model-last.pt', model, updates=step,
                    provenance=provenance, settings=settings, metrics=row)
                if validation['fde_m'] < best['fde_m']:
                    best.update(fde_m=validation['fde_m'], epoch=int(epoch) + 1, metrics=row)
                    save_model_checkpoint(output / 'model-best.pt', model, updates=step,
                        provenance=provenance, settings=settings, metrics=row)
                print(json.dumps(dict(event='epoch_validation', **row), allow_nan=False), flush=True)
            train(model, optimizer, datasets['train'], indices['train'], start_update=start_update,
                updates=updates, schedule_updates=schedule_updates, seed=settings['seed'], device=device,
                log_path=output / 'training.jsonl', lr=settings.get('optimizer', {}).get('lr', 1e-5),
                on_update=periodic, on_epoch=end_epoch)
            report['after'] = {k: evaluate(model, datasets[k], indices[k], device,
                save_predictions=output / (k + '_predictions.json')) for k in datasets}
            final_row = dict(epoch=updates // len(indices['train']), updates=updates,
                             validation=report['after']['val'])
            save_model_checkpoint(output / 'model-last.pt', model, updates=updates,
                provenance=provenance, settings=settings, metrics=final_row)
            if best['epoch'] is None or report['after']['val']['fde_m'] < best['fde_m']:
                best.update(fde_m=report['after']['val']['fde_m'], epoch=final_row['epoch'], metrics=final_row)
                save_model_checkpoint(output / 'model-best.pt', model, updates=updates,
                    provenance=provenance, settings=settings, metrics=final_row)
            save_checkpoint(output / 'checkpoint.pt', model, optimizer, updates=updates,
                            provenance=provenance, settings=settings)
            report.update(optimizer_updates=updates - start_update, best=best,
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
