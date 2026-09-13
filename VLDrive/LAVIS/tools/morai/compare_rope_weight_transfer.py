#!/usr/bin/env python3
"""Bounded same-weight historical/fixed RoPE comparison on original train clips.

This is an explicit, audited weight transfer into current code, NOT an exact
resume or retraining. Only apply_rope is monkeypatched within a try/finally
context. No optimizer exists, and all trainable/frozen hashes must stay fixed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

from offline_train_replay import ROOT, read, write, bootstrap, sha256, np, torch


def historical_cast_before_trig(x, position_ids, rope_theta=10000.0):
    """Historical numerical formula only, not a claim to restore all old code."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError('RoPE head dimension must be even')
    frequency = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = position_ids.float().unsqueeze(-1) * frequency
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1).to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * angles.cos() + rotated * angles.sin()


@contextmanager
def rope_variant(module, variant, observation):
    original = module.apply_rope
    if variant not in ('historical_cast_before_trig', 'fixed_fp32_trig'):
        raise ValueError('Unknown diagnostic RoPE variant')
    implementation = historical_cast_before_trig if variant.startswith('historical') else original

    def observed(x, position_ids, rope_theta=10000.0):
        if not observation:
            observation.update(activation_dtype=str(x.dtype), query_key_shape=list(x.shape),
                               max_position_id=int(position_ids.max()), calls=0)
        observation['calls'] += 1
        return implementation(x, position_ids, rope_theta)

    module.apply_rope = observed
    try:
        yield
    finally:
        module.apply_rope = original


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trajectory_metrics(prediction, target):
    errors = np.linalg.norm(prediction - target, axis=-1)
    return dict(prediction=prediction.tolist(), horizon_l2_m=errors.tolist(),
                ade_m=float(errors.mean()), fde_m=float(errors[-1]))


def self_test():
    from lavis.models.vldrive import ddia_attention
    original = ddia_attention.apply_rope
    positions = torch.arange(1202, dtype=torch.long)[None]
    value = torch.linspace(-1, 1, 1202 * 64).reshape(1, 1, 1202, 64).to(torch.bfloat16)
    results = {}
    for variant in ('historical_cast_before_trig', 'fixed_fp32_trig'):
        observations = {}
        with rope_variant(ddia_attention, variant, observations):
            results[variant] = ddia_attention.apply_rope(value, positions)
        assert ddia_attention.apply_rope is original
        assert observations['calls'] == 1 and observations['max_position_id'] == 1201
    difference = float((results['historical_cast_before_trig'].float()
                        - results['fixed_fp32_trig'].float()).abs().max())
    assert difference > .1, 'Probe must distinguish BF16 angle rounding'
    try:
        with rope_variant(ddia_attention, 'historical_cast_before_trig', {}):
            raise RuntimeError('intentional diagnostic failure')
    except RuntimeError:
        pass
    assert ddia_attention.apply_rope is original
    print(json.dumps(dict(cpu_self_test='passed', restoration_after_exception=True,
                          bf16_rope_max_abs_difference=difference)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frames', type=int, nargs='+', default=[2865, 2970, 3320])
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--self-test', action='store_true', help='Tiny CPU-only patch/restore test; no model assets')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.run is None or args.output is None:
        parser.error('--run and --output are required for weight-transfer inference')
    run, output = args.run.resolve(), args.output.resolve()
    if (ROOT not in output.parents or output.exists() or not 1 <= len(args.frames) <= 3
            or len(set(args.frames)) != len(args.frames)):
        raise ValueError('Use a new LAVIS output and one to three unique original train frames')
    settings = read(run / 'resolved_config.json')
    old_provenance = read(run / 'provenance.json')
    checkpoint = run / 'model-last.pt'
    backend = bootstrap(ROOT.parent)
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    from lavis.models.vldrive import ddia_attention
    from lavis.runners.vldrive_trainer import (
        warm_start_model_checkpoint, parameter_hash, source_hashes, seed_all, preserve_rng_state,
    )
    from lavis.runners.morai_trainer import autocast, batch

    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    if settings['model']['input_mode'] != 'cached' or settings['model']['precision'] != 'bf16':
        raise ValueError('This diagnostic targets the original BF16 frozen-cache run')
    dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
        split_file=ROOT / settings['split_file'], instructions=settings['instructions'],
        split='train', sequence_length=settings['sequence_length'], instruction_boundary_policy='exclude')
    if dataset.describe() != old_provenance['datasets']['train']:
        raise ValueError('Train cache, instruction, split or window semantics changed')
    if sha256(Path(settings['model']['visual_checkpoint'])) != dataset.checkpoint_sha256:
        raise ValueError('Original visual asset does not match the frozen cache')
    train_indices = read(run / 'window_indices.json')['train']
    saved_predictions = {row['sample_id']: row for row in read(run / 'train_predictions.json')}
    windows = {}
    for index in train_indices:
        begin, end = dataset.windows[index]
        entry = dataset.entries[end - 1]
        frame = int(entry['frame_id'])
        if frame in windows:
            raise ValueError('Frame IDs are ambiguous across original train routes')
        windows[frame] = index
    missing = set(args.frames) - set(windows)
    if missing:
        raise ValueError('Frames are not actual original train anchors: {}'.format(sorted(missing)))
    current_sources = source_hashes(ROOT)
    new_provenance = dict(backend=backend, source=current_sources,
        datasets={'train': dataset.describe()},
        diagnostic_sources={str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))})
    reason = ('Authorized train-only diagnostic: transfer every original final trainable weight '
              'with the same frozen assets/cache; compare historical BF16-angle RoPE and corrected '
              'FP32-trigonometry RoPE in the same current implementation. No reset or optimizer.')
    output.mkdir(parents=True)
    print(json.dumps(dict(phase='loading_same_original_weights', frames=args.frames)), flush=True)
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device=args.device)
    transfer = warm_start_model_checkpoint(checkpoint, model, provenance=new_provenance,
        reset_prefixes=(), transfer_reason=reason)
    if transfer['old_provenance'] != old_provenance or transfer['reset_old_keys'] or transfer['reset_new_keys']:
        raise ValueError('Original provenance differs or any trainable weight was reset')
    model.eval()
    before_trainable = parameter_hash(model, trainable=True)
    before_frozen = parameter_hash(model, trainable=False)
    original_state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if before_trainable != original_state['trainable_hash'] or before_frozen != original_state['frozen_hash']:
        raise ValueError('Weight transfer did not preserve exact original parameter hashes')
    del original_state
    write(output / 'weight_transfer.json', transfer)
    original_rope = ddia_attention.apply_rope
    results = []
    with torch.inference_mode(), preserve_rng_state():
        for frame in args.frames:
            index = windows[frame]
            begin, end = dataset.windows[index]
            entry = dataset.entries[end - 1]
            sample_id = entry['route_id'] + '/%04d' % frame
            saved = saved_predictions[sample_id]
            item = batch(dataset[index], args.device)
            target = item['training_targets']['future_trajectory'][0, -1].float().cpu().numpy()
            input_hash = tensor_sha256(item['model_inputs']['visual_feature'])
            variants, predictions, selections = {}, {}, {}
            for variant in ('historical_cast_before_trig', 'fixed_fp32_trig'):
                observation = {}
                started = time.perf_counter()
                with rope_variant(ddia_attention, variant, observation), autocast(model):
                    prediction = model.predict(item['model_inputs'], compute_auxiliary=False)
                assert ddia_attention.apply_rope is original_rope
                value = prediction['trajectory'][0, -1].float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite {} trajectory'.format(variant))
                selections[variant] = prediction['selected_count'][0].cpu().tolist()
                predictions[variant] = value
                variants[variant] = dict(trajectory_metrics(value, target), rope=observation,
                                        selected_count=selections[variant], elapsed_s=time.perf_counter() - started)
                del prediction
            if selections['historical_cast_before_trig'] != selections['fixed_fp32_trig']:
                raise ValueError('A non-RoPE selection difference confounds the pair')
            if tensor_sha256(item['model_inputs']['visual_feature']) != input_hash:
                raise ValueError('Cached visual inputs were mutated')
            old, new = predictions['historical_cast_before_trig'], predictions['fixed_fp32_trig']
            historical_saved_difference = float(np.abs(old - np.asarray(saved['trajectory'])).max())
            row = dict(frame_id=frame, window_index=index, sample_id=sample_id,
                instruction=item['model_inputs']['instruction'][0], input_feature_sha256=input_hash,
                actual_future=target.tolist(), variants=variants,
                fixed_minus_historical=dict(prediction_delta=(new - old).tolist(),
                    max_abs_m=float(np.abs(new - old).max()),
                    ade_m=variants['fixed_fp32_trig']['ade_m'] - variants['historical_cast_before_trig']['ade_m'],
                    fde_m=variants['fixed_fp32_trig']['fde_m'] - variants['historical_cast_before_trig']['fde_m']),
                historical_vs_saved_max_abs_m=historical_saved_difference,
                historical_reproduces_saved_within_1e_5=historical_saved_difference <= 1e-5)
            results.append(row)
            write(output / 'comparison.json', results)
            print(json.dumps(row), flush=True)
    after_trainable = parameter_hash(model, trainable=True)
    after_frozen = parameter_hash(model, trainable=False)
    if (before_trainable != after_trainable or before_frozen != after_frozen
            or ddia_attention.apply_rope is not original_rope or source_hashes(ROOT) != current_sources):
        raise ValueError('Diagnostic changed weights, frozen state, RoPE binding or production source')
    summary = dict(split='train_only', optimizer_updates=0, exact_resume=False,
        checkpoint=str(checkpoint), checkpoint_sha256=transfer['source_checkpoint_sha256'],
        source_updates=transfer['source_updates'], frames=args.frames,
        trainable_hash_before=before_trainable, trainable_hash_after=after_trainable,
        frozen_hash_before=before_frozen, frozen_hash_after=after_frozen,
        production_sources_unchanged=True, apply_rope_restored=True,
        historical_mean_ade_m=float(np.mean([row['variants']['historical_cast_before_trig']['ade_m'] for row in results])),
        fixed_mean_ade_m=float(np.mean([row['variants']['fixed_fp32_trig']['ade_m'] for row in results])),
        historical_mean_fde_m=float(np.mean([row['variants']['historical_cast_before_trig']['fde_m'] for row in results])),
        fixed_mean_fde_m=float(np.mean([row['variants']['fixed_fp32_trig']['fde_m'] for row in results])),
        note=('Only RoPE differs between paired forwards. The historical formula does not restore the '
              'entire historical source tree. Source/config transfer is recorded, and saved-prediction '
              'agreement is separately checked. These selected train clips are not a generalization test.'))
    write(output / 'summary.json', summary)
    print(json.dumps(dict(phase='COMPLETE', **summary)), flush=True)


if __name__ == '__main__':
    main()
