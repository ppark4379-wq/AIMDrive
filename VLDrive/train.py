#!/usr/bin/env python3
"""Train VLDrive on MORAI data.

  stage 1, vision encoder pre-training
      python train.py vision-encoder --config training/vldrive_train.json

  stage 2, VLDrive (frozen encoder -> CCDP -> MEFA -> TinyLlama -> waypoints)
      python train.py --config vldrive_train.json

Stage 2 is the default. A bare config filename is also searched in configs/.
Its output directory defaults to outputs/logs/vldrive_<timestamp>.

Both stages take every training parameter from their config file.  Stage 1
accepts no other option, so nothing on the command line or in another module
can override a configured value; edit or copy the config instead.

Stage 1 writes checkpoint-last.pth.tar / model-best.pth.tar, metrics.jsonl and
feature_contract.json.  Stage 2 writes model-best.pt / model-last.pt,
checkpoint.pt (weights + optimizer, for --checkpoint resume), training.jsonl,
epoch_metrics.jsonl and report.json.

Stage 2 reads visual tokens from a cache built by data/src/processing/cache_vision_tokens.py
and instructions built by data/src/processing/build_instructions.py, so the frozen encoder
never runs during training.  --mode preflight stops after the gradient and
causality checks; --mode evaluate scores a checkpoint without training.
"""
import argparse
from datetime import datetime
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common.runtime import add_path_arguments, configure_runtime, resolve_settings, sha256

configure_runtime()


def write(path, data):
    """Write JSON atomically so an interrupted run never leaves half a file."""
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


# --------------------------------------------------------------------------- stage 1

def add_vision_encoder_arguments(parser):
    from training.vision_encoder.recipes import pretrain as vision_encoder_training
    return vision_encoder_training.add_arguments(parser)


def run_vision_encoder(parser, args):
    from training.vision_encoder.recipes import pretrain as vision_encoder_training
    return vision_encoder_training.run(vision_encoder_training.config_from_args(parser, args))


# --------------------------------------------------------------------------- stage 2

def add_vldrive_arguments(parser):
    parser.add_argument('--config', type=Path, required=True,
                        help='training config, e.g. training/vldrive_train.json')
    parser.add_argument('--output', type=Path,
                        help='new run directory; default: outputs/logs/vldrive_<timestamp>')
    parser.add_argument('--mode', choices=('train', 'preflight', 'evaluate'), default='train')
    parser.add_argument('--checkpoint', type=Path, help='resume from checkpoint.pt, or score model-best.pt')
    parser.add_argument('--instructions', type=Path, help='verified instruction JSONL or LMDrive navigation bundle directory')
    parser.add_argument('--feature-cache', type=Path, help='completed frozen [106,256] token cache')
    parser.add_argument('--split-file', type=Path, help='chronological MORAI train/purge/validation split')
    parser.add_argument('--updates', type=int,
                        help='stop after this many total updates; keeps the configured cosine schedule')
    add_path_arguments(parser)
    return parser


def run_vldrive(args):
    from utils.config.config_files import load_config
    settings = resolve_settings(load_config(args.config), args)
    for key in ('instructions', 'feature_cache', 'split_file'):
        override = getattr(args, key)
        if override:
            settings[key] = str(override.expanduser().resolve())
    batch_size = settings.get('batch_size', 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    validation_enabled = settings.get('validation_enabled', True)
    if type(validation_enabled) is not bool:
        raise ValueError('validation_enabled must be boolean')
    if type(settings.get('initial_evaluation', True)) is not bool:
        raise ValueError('initial_evaluation must be boolean')
    full_route = settings.get('include_validation_in_train', False)
    if type(full_route) is not bool or (full_route and validation_enabled):
        raise ValueError('Full-route training requires validation_enabled=false')
    if full_route and settings.get('dataset_format') != 'navigation_token_cache_v1':
        raise ValueError('Full-route training requires navigation_token_cache_v1')
    splits = ('train', 'val') if validation_enabled else ('train',)
    warm_start_path = settings.get('warm_start_checkpoint')
    if warm_start_path and not args.checkpoint and args.mode != 'evaluate':
        if not Path(warm_start_path).is_file():
            raise ValueError('Warm-start checkpoint is missing: ' + str(warm_start_path))
    instruction_path = Path(settings.get('instructions') or '')
    valid_instructions = ((instruction_path / 'manifest.json').is_file()
                          if settings.get('dataset_format') == 'navigation_token_cache_v1'
                          else instruction_path.is_file())
    if not settings.get('instructions') or not valid_instructions:
        raise ValueError('VLDrive requires verified navigation instructions generated from the recorded global path.')
    if settings['sequence_length'] < settings['model'].get('memory_size', 10) + 1:
        raise ValueError('Paper preflight requires at least memory_size+1 real frames to exercise Z=10')
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new output directory inside this VLDrive checkout')
    output.mkdir(parents=True)
    write(output / 'resolved_config.json', settings)

    import torch
    from utils.common.runtime import bootstrap
    backend = bootstrap(ROOT)
    from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
    from training.connector_llm.training import (seed_all, subset_indices, stride_indices, source_hashes, parameter_hash,
        preflight, evaluate, make_optimizer, train, save_checkpoint, restore_checkpoint,
        save_model_checkpoint, restore_model_checkpoint, periodic_checkpoint_path,
        warm_start_model_checkpoint)
    from functools import partial
    evaluate = partial(evaluate, seed=settings.get('evaluation_seed', settings['seed']))
    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    device = settings.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')
    config = VLDriveTrainingConfig(**settings['model'])
    config.validate()
    path_settings = settings.get('path_instruction', {})
    path_cache = path_settings.get('feature_cache') if config.path_navigation else None
    if config.path_navigation and (not path_cache or settings.get('dataset_format') != 'navigation_token_cache_v1'):
        raise ValueError('Path navigation requires navigation_token_cache_v1 and path_instruction.feature_cache')
    if path_cache:
        path_cache = ROOT / path_cache
    root = Path(settings['dataset'])
    print('Validating actual artifacts, teacher labels, independent splits and instruction availability...', flush=True)
    dataset_format = settings.get('dataset_format', 'morai_expert_cache_v1')
    if dataset_format not in ('morai_expert_cache_v1', 'navigation_token_cache_v1'):
        raise ValueError('Unknown VLDrive dataset_format: ' + str(dataset_format))
    if config.input_mode != 'cached':
        raise ValueError('MORAI expert cache path requires model.input_mode=cached')
    from data.src.loaders.morai_expert_dataset import MoraiExpertVLDriveDataset
    split_path = Path(settings['split_file'])
    if not split_path.is_absolute():
        split_path = (ROOT / split_path).resolve()
    cache = Path(settings.get('feature_cache', ''))
    if not cache.is_dir():
        raise ValueError('Completed MORAI feature cache is missing: ' + str(cache))
    if dataset_format == 'navigation_token_cache_v1':
        from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
        language = settings.get('navigation_sampling', {})
        datasets = {split: NavigationCacheDataset(
            cache, settings['instructions'], split, settings['sequence_length'], settings['seed'],
            train_variants=language.get('train_variants', list(range(8))),
            validation_variants=language.get('validation_variants', list(range(8))),
            include_validation_in_train=full_route,
            frame_sampling_interval=settings.get('frame_sampling_interval', 1),
            path_feature_cache=path_cache,
        ) for split in splits}
    else:
        datasets = {split: MoraiExpertVLDriveDataset(
            root, cache_root=cache, split_file=split_path, split=split,
            sequence_length=settings['sequence_length'], instructions=settings['instructions'],
            frame_sampling_interval=settings.get('frame_sampling_interval', 1),
        ) for split in splits}

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
    token_folder = settings.get('navigation_sampling', {}).get('token_cache')
    if config.path_navigation and token_folder:
        raise ValueError('Legacy text token cache does not contain the new direction templates; disable token_cache')
    if token_folder:
        token_meta = Path(token_folder) / 'token_meta.json'
        provenance['instruction_tokens'] = dict(meta_sha256=sha256(token_meta),
                                                meta=json.loads(token_meta.read_text()))
    write(output / 'provenance.json', provenance)
    write(output / 'window_indices.json', indices)
    print('Loading cached frozen visual tokens and full trainable pretrained TinyLlama/DDIA...', flush=True)
    model = VLDriveTrainingModel(config, device=device)
    from training.connector_llm.stages import apply_instruction_stage
    apply_instruction_stage(model, path_settings.get('stage', 'joint') if config.path_navigation else 'joint')
    if token_folder:
        if dataset_format != 'navigation_token_cache_v1':
            raise ValueError('Navigation token cache requires the navigation dataset')
        from navigation.token_cache import CachedNavigationTokenizer
        model.language_model.tokenizer = CachedNavigationTokenizer(
            model.language_model.tokenizer, ROOT / token_folder,
            sha256(instruction_path / 'manifest.json'), sha256(cache / 'manifest.jsonl'))
    write(output / 'model.json', model.describe())
    steps_per_epoch = math.ceil(len(indices['train']) / batch_size)
    schedule_updates = settings.get('epochs', 15) * steps_per_epoch
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
    if warm_start_path and not args.checkpoint and args.mode != 'evaluate':
        print('Loading previous model weights; starting a fresh optimizer and learning-rate schedule...', flush=True)
        transfer = warm_start_model_checkpoint(
            warm_start_path, model, provenance=provenance,
            new_prefixes=path_settings.get('new_module_prefixes', ()) if config.path_navigation else (),
            transfer_reason=f'Continue from saved weights with batch_size={batch_size}, include_validation_in_train={full_route}; fresh optimizer and additional configured epochs.')
        write(output / 'warm_start.json', transfer)
        report['warm_start'] = dict(source_checkpoint=transfer['source_checkpoint'],
            source_updates=transfer['source_updates'], optimizer_restored=False,
            additional_epochs=settings.get('epochs', 15))
        print(json.dumps(dict(event='warm_start', **report['warm_start'])), flush=True)
    if args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError('Evaluation requires --checkpoint')
        restored = restore_model_checkpoint(args.checkpoint, model, provenance=provenance,
                                            evaluation_source_compatibility=True)
        report['evaluation_source_compatibility'] = restored['evaluation_source_compatibility']
        del restored
        evaluation_split = 'val' if validation_enabled else 'train'
        report['validation' if validation_enabled else 'training'] = evaluate(
            model, datasets[evaluation_split], indices[evaluation_split], device,
            save_predictions=output / (evaluation_split + '_predictions.json'))
    else:
        if not start_update <= updates <= schedule_updates:
            raise ValueError('Update limit must be within the configured schedule')
        print('Checking gradients, temporal causality and forbidden teacher inputs...', flush=True)
        report['preflight'] = preflight(model, datasets['train'][indices['train'][0]], device)
        if token_folder:
            report['preflight']['instruction_token_cache'] = model.language_model.tokenizer.describe()
        write(output / 'preflight.json', report['preflight'])
        if not report['preflight']['passed']:
            raise RuntimeError('VLDrive preflight failed; inspect preflight.json')
        if args.mode == 'train':
            from utils.visualization.training_dashboard.launcher import launch_dashboard
            launch_dashboard(output, settings.get('dashboard'))
            if optimizer is None:
                optimizer = make_optimizer(model, **settings.get('optimizer', {}))
            frozen = parameter_hash(model, trainable=False)
            before = parameter_hash(model, trainable=True)
            initial_evaluation = settings.get('initial_evaluation', True)
            report['initial_evaluation'] = dict(enabled=initial_evaluation)
            if initial_evaluation:
                print('Initial full evaluation started; optimizer updates begin after evaluation and baseline checkpoint saving.', flush=True)
                report['before'] = {k: evaluate(model, datasets[k], indices[k], device) for k in datasets}
                print('Initial full evaluation complete.', flush=True)
            else:
                report['before'] = None
                print('Initial full evaluation skipped (initial_evaluation=false).', flush=True)

            def periodic(step):
                interval = settings.get('checkpoint_every', 100)
                if interval > 0 and step % interval == 0 and step < updates:
                    path = periodic_checkpoint_path(output, step, interval, settings.get('checkpoint_slots', 0))
                    print(json.dumps(dict(event='checkpoint_start', updates=step, path=str(path))), flush=True)
                    save_checkpoint(path, model, optimizer,
                                    updates=step, provenance=provenance, settings=settings)
                    print(json.dumps(dict(event='checkpoint_complete', updates=step, path=str(path))), flush=True)

            score_split = 'val' if validation_enabled else 'train'
            score_name = 'validation' if validation_enabled else 'training'
            best = dict(fde_m=None, epoch=None, metrics=None,
                        scope=score_name + '_current_invocation_after_training')
            if initial_evaluation:
                baseline_row = dict(epoch=start_update // steps_per_epoch, updates=start_update,
                                    **{score_name: report['before'][score_split]})
                best.update(fde_m=report['before'][score_split]['fde_m'], epoch=baseline_row['epoch'],
                            metrics=baseline_row, scope=score_name + '_current_invocation_including_starting_weights')
                save_model_checkpoint(output / 'model-best.pt', model, updates=start_update,
                    provenance=provenance, settings=settings, metrics=baseline_row)
            epoch_metrics = output / 'epoch_metrics.jsonl'

            def end_epoch(epoch, step):
                validation = evaluate(model, datasets[score_split], indices[score_split], device)
                row = dict(epoch=int(epoch) + 1, updates=int(step), **{score_name: validation})
                with epoch_metrics.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
                save_model_checkpoint(output / 'model-last.pt', model, updates=step,
                    provenance=provenance, settings=settings, metrics=row)
                if best['epoch'] is None or validation['fde_m'] < best['fde_m']:
                    best.update(fde_m=validation['fde_m'], epoch=int(epoch) + 1, metrics=row)
                    save_model_checkpoint(output / 'model-best.pt', model, updates=step,
                        provenance=provenance, settings=settings, metrics=row)
                print(json.dumps(dict(event='epoch_' + score_name, **row), allow_nan=False), flush=True)

            train(model, optimizer, datasets['train'], indices['train'], start_update=start_update,
                updates=updates, schedule_updates=schedule_updates, seed=settings['seed'], device=device,
                log_path=output / 'training.jsonl', lr=settings.get('optimizer', {}).get('lr', 1e-5),
                on_update=periodic, on_epoch=end_epoch, batch_size=batch_size,
                telemetry_enabled=settings.get('logging', {}).get('telemetry_enabled', True),
                gradient_log_interval=settings.get('logging', {}).get('gradient_interval', 100))
            report['after'] = {k: evaluate(model, datasets[k], indices[k], device,
                save_predictions=output / (k + '_predictions.json')) for k in datasets}
            final_row = dict(epoch=updates // steps_per_epoch, updates=updates,
                             **{score_name: report['after'][score_split]})
            save_model_checkpoint(output / 'model-last.pt', model, updates=updates,
                provenance=provenance, settings=settings, metrics=final_row)
            if best['epoch'] is None or report['after'][score_split]['fde_m'] < best['fde_m']:
                best.update(fde_m=report['after'][score_split]['fde_m'], epoch=final_row['epoch'], metrics=final_row)
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

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith('-'):
        argv.insert(0, 'vldrive')
    if argv[0] == 'vldrive':
        # The prepared WSL packages were previously supplied via PYTHONPATH.
        # Load them before parser imports pull in model backends.
        dependencies = ROOT / '.runtime/stage2_wsl_deps'
        if sys.platform == 'linux' and dependencies.is_dir() and str(dependencies) not in sys.path:
            sys.path.insert(1, str(dependencies))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest='stage', required=True, metavar='stage')
    stage1 = stages.add_parser('vision-encoder', help='stage 1: MORAI vision encoder pre-training')
    add_vision_encoder_arguments(stage1)
    morai = stages.add_parser('vision-encoder-morai', help='MORAI Sensor Record multitask transfer from full LMDrive weights')
    from training.vision_encoder.recipes import transfer as morai_transfer_training
    morai_transfer_training.add_arguments(morai)
    stage2 = stages.add_parser('vldrive', help='stage 2: VLDrive training on cached visual tokens')
    add_vldrive_arguments(stage2)
    args = parser.parse_args(argv)
    if args.stage == 'vision-encoder':
        run_vision_encoder(stage1, args)
    elif args.stage == 'vision-encoder-morai':
        morai_transfer_training.run(args)
    else:
        if not args.config.is_file() and args.config.parent == Path('.'):
            args.config = ROOT / 'configs' / args.config.name
        if args.output is None:
            args.output = ROOT / 'outputs/logs' / ('vldrive_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        run_vldrive(args)


if __name__ == '__main__':
    main()
