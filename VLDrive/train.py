#!/usr/bin/env python3
"""Train VLDrive on MORAI data.

  stage 1, vision encoder pre-training
      python train.py vision-encoder --config configs/vision_encoder_train.json

  stage 2, VLDrive (frozen encoder -> CCDP -> MEFA -> TinyLlama -> waypoints)
      python train.py vldrive --config configs/vldrive_train.json \
          --output outputs/logs/vldrive-run1

Both stages take every training parameter from their config file.  Stage 1
accepts no other option, so nothing on the command line or in another module
can override a configured value; edit or copy the config instead.

Stage 1 writes checkpoint-last.pth.tar / model-best.pth.tar, metrics.jsonl and
feature_contract.json.  Stage 2 writes model-best.pt / model-last.pt,
checkpoint.pt (weights + optimizer, for --checkpoint resume), training.jsonl,
epoch_metrics.jsonl and report.json.

Stage 2 reads visual tokens from a cache built by tools/cache_vision_tokens.py
and instructions built by tools/build_instructions.py, so the frozen encoder
never runs during training.  --mode preflight stops after the gradient and
causality checks; --mode evaluate scores a checkpoint without training.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import add_path_arguments, configure_runtime, resolve_settings, sha256

configure_runtime()


def write(path, data):
    """Write JSON atomically so an interrupted run never leaves half a file."""
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


# --------------------------------------------------------------------------- stage 1

def add_vision_encoder_arguments(parser):
    from utils import vision_encoder_training
    return vision_encoder_training.add_arguments(parser)


def run_vision_encoder(parser, args):
    from utils import vision_encoder_training
    return vision_encoder_training.run(vision_encoder_training.config_from_args(parser, args))


# --------------------------------------------------------------------------- stage 2

def add_vldrive_arguments(parser):
    parser.add_argument('--config', type=Path, required=True,
                        help='training config, e.g. configs/vldrive_train.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='new run directory, e.g. outputs/logs/vldrive-run1')
    parser.add_argument('--mode', choices=('train', 'preflight', 'evaluate'), default='train')
    parser.add_argument('--checkpoint', type=Path, help='resume from checkpoint.pt, or score model-best.pt')
    parser.add_argument('--instructions', type=Path, help='verified global-path navigation instruction JSONL')
    parser.add_argument('--feature-cache', type=Path, help='completed frozen [106,256] token cache')
    parser.add_argument('--split-file', type=Path, help='chronological MORAI train/purge/validation split')
    parser.add_argument('--updates', type=int,
                        help='stop after this many total updates; keeps the configured cosine schedule')
    add_path_arguments(parser)
    return parser


def run_vldrive(args):
    settings = resolve_settings(json.loads(args.config.read_text()), args)
    for key in ('instructions', 'feature_cache', 'split_file'):
        override = getattr(args, key)
        if override:
            settings[key] = str(override.expanduser().resolve())
    if not settings.get('instructions') or not Path(settings['instructions']).is_file():
        raise ValueError('VLDrive requires verified navigation instructions generated from the recorded global path.')
    if settings['sequence_length'] < settings['model'].get('memory_size', 10) + 1:
        raise ValueError('Paper preflight requires at least memory_size+1 real frames to exercise Z=10')
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new output directory inside this VLDrive checkout')
    output.mkdir(parents=True)
    write(output / 'resolved_config.json', settings)

    import torch
    from utils.runtime import bootstrap
    backend = bootstrap(ROOT)
    from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
    from utils.trainer import (seed_all, subset_indices, stride_indices, source_hashes, parameter_hash,
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
    dataset_format = settings.get('dataset_format', 'morai_expert_cache_v1')
    if dataset_format != 'morai_expert_cache_v1':
        raise ValueError('Unknown VLDrive dataset_format: ' + str(dataset_format))
    if config.input_mode != 'cached':
        raise ValueError('MORAI expert cache path requires model.input_mode=cached')
    from datasets.morai_expert_dataset import MoraiExpertVLDriveDataset
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

    digest = datasets['train'].checkpoint_sha256
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
    provenance = dict(backend=backend, source=source_hashes(ROOT),
                      datasets={k: v.describe() for k, v in datasets.items()})
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
    return report


# --------------------------------------------------------------------------- entry

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest='stage', required=True, metavar='stage')
    stage1 = stages.add_parser('vision-encoder', help='stage 1: MORAI vision encoder pre-training')
    add_vision_encoder_arguments(stage1)
    stage2 = stages.add_parser('vldrive', help='stage 2: VLDrive training on cached visual tokens')
    add_vldrive_arguments(stage2)
    args = parser.parse_args()
    if args.stage == 'vision-encoder':
        run_vision_encoder(stage1, args)
    else:
        run_vldrive(args)


if __name__ == '__main__':
    main()
