#!/usr/bin/env python3
"""Read-only representation extraction from the original trained model.

No optimizer or fitting. Reproduce historical RoPE in a scoped diagnostic hook,
require all saved 636 train predictions to match, and verify weights unchanged.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lavis.common.morai_runtime import configure_runtime, bootstrap, sha256
configure_runtime()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / 'outputs' not in output.parents:
        raise ValueError('Output must remain under LAVIS/outputs')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'representation_readout.json'
    matrix_file = output / 'representation_readout.npz'
    if destination.exists() or matrix_file.exists():
        raise ValueError('Never overwrite prior diagnostic artifacts')
    import fcntl
    lock = (ROOT / '.runtime/training_runs/last_encoder.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import torch
    bootstrap(ROOT.parent)
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    import lavis.models.vldrive.ddia_attention as attention
    from lavis.runners.vldrive_trainer import parameter_hash, source_hashes, seed_all
    from lavis.runners.morai_trainer import autocast
    torch.set_num_threads(2)
    run = ROOT / 'outputs/morai_paper_last_encoder_20260910T055054Z'
    settings = read(run / 'resolved_config.json')
    checkpoint = run / 'model-last.pt'
    digest = read(checkpoint.with_suffix('.pt.sha256.json'))
    assert checkpoint.stat().st_size == digest['bytes'] and sha256(checkpoint) == digest['sha256']
    sources = source_hashes(ROOT)
    seed_all(settings['seed'])
    print('LOADING_ORIGINAL_9540_CHECKPOINT', flush=True)
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda').eval()
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert asdict(VLDriveTrainingConfig(**state['model_config'])) == asdict(model.config)
    assert state['updates'] == 9540
    frozen = parameter_hash(model, trainable=False)
    assert frozen == state['frozen_hash']
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names == set(state['trainable_state'])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.copy_(state['trainable_state'][name])
    trainable = parameter_hash(model, trainable=True)
    assert trainable == state['trainable_hash']
    del state
    saved = read(run / 'train_predictions.json')
    cache = Path(settings['feature_cache'])
    rows = {r['frame_id']: r for r in map(json.loads, (cache / 'manifest.jsonl').read_text().splitlines())
            if r['split'] == 'train'}
    instructions = list(map(json.loads, Path(settings['instructions']).read_text().splitlines()))
    original_rope = attention.apply_rope

    def historical_rope(x, position_ids, rope_theta=10000.0):
        head_dim = x.shape[-1]
        frequency = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
        angles = position_ids.float().unsqueeze(-1) * frequency
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1).to(x.dtype)
        return x * angles.cos() + attention.rotate_half(x) * angles.sin()

    vectors = {key: [] for key in ('frame_ids', 'target', 'prediction', 'connector_last',
                                  'connector_mean', 'llm_readout', 'head_relu')}
    trace = {}
    def connector_hook(module, args, result):
        valid = result['selected_valid_mask'][0, -1]
        tokens = result['language_tokens'][0, -1, valid].float()
        trace['connector_last'] = tokens[-1].cpu().numpy()
        trace['connector_mean'] = tokens.mean(0).cpu().numpy()
    def head_input_hook(module, args):
        trace['llm_readout'] = args[0][0, -1].float().cpu().numpy()
    def head_relu_hook(module, args, result):
        trace['head_relu'] = result[0, -1].float().cpu().numpy()
    handles = [model.connector.register_forward_hook(connector_hook),
               model.language_model.waypoints_predictor.register_forward_pre_hook(head_input_hook),
               model.language_model.waypoints_predictor[1].register_forward_hook(head_relu_hook)]
    differences = []
    try:
        attention.apply_rope = historical_rope
        with torch.inference_mode():
            for i, record in enumerate(saved):
                frame = int(record['sample_id'].rsplit('/', 1)[1])
                window = [rows[f] for f in range(frame - 10, frame + 1)]
                assert all(b['native_sync_frame'] - a['native_sync_frame'] == 5
                           for a, b in zip(window, window[1:]))
                text = next(r['text'] for r in instructions if r['start_frame'] <= frame <= r['stop_frame'])
                features = []
                for row in window:
                    path = cache / row['visual_feature']['path']
                    assert sha256(path) == row['visual_feature']['sha256']
                    features.append(np.load(path, allow_pickle=False))
                target_path = cache / rows[frame]['future_trajectory']['path']
                assert sha256(target_path) == rows[frame]['future_trajectory']['sha256']
                x = torch.from_numpy(np.stack(features).astype(np.float32))[None].cuda()
                trace.clear()
                with autocast(model):
                    result = model.predict(dict(visual_feature=x, instruction=[text]), compute_auxiliary=False)
                pred = result['trajectory'][0, -1].float().cpu().numpy()
                difference = float(np.abs(pred - np.asarray(record['trajectory'])).max())
                # Historical formula must reproduce saved inference, not only
                # approximately transfer weights to a changed forward function.
                if difference != 0:
                    raise ValueError(('Historical saved prediction mismatch', frame, difference))
                differences.append(difference)
                vectors['frame_ids'].append(frame)
                vectors['target'].append(np.load(target_path, allow_pickle=False))
                vectors['prediction'].append(pred)
                for name in ('connector_last', 'connector_mean', 'llm_readout', 'head_relu'):
                    vectors[name].append(trace[name])
                if (i + 1) % 50 == 0:
                    print(json.dumps(dict(completed=i + 1, total=len(saved), max_prediction_difference=max(differences))), flush=True)
    finally:
        attention.apply_rope = original_rope
        for handle in handles:
            handle.remove()
    assert source_hashes(ROOT) == sources
    assert trainable == parameter_hash(model, trainable=True)
    assert frozen == parameter_hash(model, trainable=False)
    arrays = {name: np.asarray(value) for name, value in vectors.items()}
    np.savez(matrix_file, **arrays)
    write(destination, dict(complete=True, optimizer_updates=0, split='train_only',
        source_sha256=sources, checkpoint_sha256=digest['sha256'], original_updates=9540,
        sampled_anchors=len(saved), max_saved_prediction_difference_m=max(differences),
        trainable_hash_unchanged=True, frozen_hash_unchanged=True, apply_rope_restored=True,
        matrix_sha256=sha256(matrix_file), arrays={k: list(v.shape) for k, v in arrays.items()},
        note='Representations only. Historical RoPE is a scoped diagnostic override, not a production change. '
             'All 636 original train-anchor predictions must match exactly. No fitted replacement model.'))
    print('REPRESENTATION_READOUT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
