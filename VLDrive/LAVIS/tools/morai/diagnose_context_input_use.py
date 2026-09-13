#!/usr/bin/env python3
"""Bounded original-model input interventions. No fitting or weight changes."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lavis.common.morai_runtime import configure_runtime, bootstrap, sha256
configure_runtime()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / 'outputs' not in output.parents or output.exists():
        raise ValueError('Require a new diagnostic output under LAVIS/outputs')
    output.parent.mkdir(parents=True, exist_ok=True)
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
    from tools.morai.compare_rope_weight_transfer import historical_cast_before_trig
    torch.set_num_threads(2)
    run = ROOT / 'outputs/morai_paper_last_encoder_20260910T055054Z'
    settings = read(run / 'resolved_config.json')
    checkpoint = run / 'model-last.pt'
    digest = read(checkpoint.with_suffix('.pt.sha256.json'))
    assert sha256(checkpoint) == digest['sha256']
    sources = source_hashes(ROOT)
    seed_all(settings['seed'])
    print('LOADING_ORIGINAL_9540_FOR_READ_ONLY_INPUT_PROBES', flush=True)
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
    cache = Path(settings['feature_cache'])
    rows = {r['frame_id']: r for r in map(json.loads, (cache / 'manifest.jsonl').read_text().splitlines())
            if r['split'] == 'train'}
    instructions = list(map(json.loads, Path(settings['instructions']).read_text().splitlines()))
    saved = {int(r['sample_id'].rsplit('/', 1)[1]): r for r in read(run / 'train_predictions.json')}
    original_rope = attention.apply_rope
    results = []
    def text_for(frame):
        return next(r['text'] for r in instructions if r['start_frame'] <= frame <= r['stop_frame'])
    def features_for(frame):
        entries = [rows[f] for f in range(frame - 10, frame + 1)]
        assert all(b['native_sync_frame'] - a['native_sync_frame'] == 5 for a, b in zip(entries, entries[1:]))
        arrays = []
        for row in entries:
            path = cache / row['visual_feature']['path']
            assert sha256(path) == row['visual_feature']['sha256']
            arrays.append(np.load(path, allow_pickle=False))
        return torch.from_numpy(np.stack(arrays).astype(np.float32))[None].cuda()
    frames = [2865, 2870, 2970, 2975, 991, 1591, 2491, 2791]
    turns = set(frames[:4])
    try:
        attention.apply_rope = historical_cast_before_trig
        with torch.inference_mode():
            for frame in frames:
                x = features_for(frame)
                text = text_for(frame)
                variants = [('baseline_11', x, text), ('current_only_1', x[:, -1:], text),
                            ('recent_history_6', x[:, -6:], text)]
                if frame in turns:
                    variants += [('instruction_left', x, 'Proceed ahead and make a left turn.'),
                                 ('instruction_right', x, 'Proceed ahead and make a right turn.')]
                    donor = 2970 if frame < 2900 else 2865
                    swapped = x.clone()
                    swapped[:, -1] = features_for(donor)[:, -1]
                    variants.append((f'current_visual_from_opposite_turn_{donor}', swapped, text))
                else:
                    red = torch.from_numpy(np.load(cache / rows[frame-1]['visual_feature']['path']).astype(np.float32)).cuda()
                    replaced = x.clone()
                    replaced[:, -1] = red
                    variants.append(('current_all_tokens_from_previous_red_frame', replaced, text))
                    replaced_token = x.clone()
                    replaced_token[:, -1, 100] = red[100]
                    variants.append(('current_traffic_token_from_previous_red_frame', replaced_token, text))
                baseline = None
                records = []
                for name, inputs, sentence in variants:
                    with autocast(model):
                        prediction = model.predict(dict(visual_feature=inputs, instruction=[sentence]), compute_auxiliary=False)
                    value = prediction['trajectory'][0, -1].float().cpu().numpy()
                    if baseline is None:
                        baseline = value
                        if frame in saved:
                            assert np.array_equal(value, np.asarray(saved[frame]['trajectory'])), ('Baseline mismatch', frame)
                    change = value - baseline
                    records.append(dict(variant=name, history_frames=inputs.shape[1], instruction=sentence,
                        prediction=value.tolist(), max_abs_change_m=float(np.abs(change).max()),
                        endpoint_change_m=float(np.linalg.norm(change[-1])),
                        endpoint_lateral_change_m=float(change[-1, 0]),
                        selected_count=prediction['selected_count'][0].cpu().tolist()))
                    print(json.dumps(dict(frame_id=frame, variant=name, endpoint=value[-1].tolist(),
                        endpoint_change_m=records[-1]['endpoint_change_m'])), flush=True)
                gt_path = cache / rows[frame]['future_trajectory']['path']
                assert sha256(gt_path) == rows[frame]['future_trajectory']['sha256']
                results.append(dict(frame_id=frame, original_target=np.load(gt_path).tolist(),
                    baseline_saved_exact=(True if frame in saved else None), variants=records))
    finally:
        attention.apply_rope = original_rope
    assert source_hashes(ROOT) == sources
    assert trainable == parameter_hash(model, trainable=True)
    assert frozen == parameter_hash(model, trainable=False)
    report = dict(complete=True, optimizer_updates=0, split='train_only', results=results,
        checkpoint_sha256=digest['sha256'], original_updates=9540,
        trainable_hash_unchanged=True, frozen_hash_unchanged=True, apply_rope_restored=True,
        production_sources_unchanged=True,
        limitations='Input sensitivity only: altered observations/instructions do not have a new valid driving GT. '
                    'History length changes history content, MEFA context and positions together, not a pure history-causality test. '
                    'Opposite-turn visual swap also changes speed, place and scene. '
                    'Traffic token is a learned fused feature, not a pure signal-color field. '
                    'Historical RoPE is used only inside this diagnostic process. No training or production mutation.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('CONTEXT_INPUT_USE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
