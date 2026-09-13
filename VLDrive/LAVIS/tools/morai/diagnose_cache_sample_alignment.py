#!/usr/bin/env python3
"""Read-only raw-data -> frozen-token cache identity audit. No optimizer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=[1, 4], default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / 'outputs' not in output.parents or output.exists():
        raise ValueError('Require a new diagnostic file under LAVIS/outputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (ROOT / '.runtime/training_runs/last_encoder.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    script = ROOT / 'tools/morai/cache_expert_vision_tokens.py'
    spec = importlib.util.spec_from_file_location('cache_builder_audit', script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(607)
    run = ROOT / 'outputs/morai_paper_last_encoder_20260910T055054Z'
    config = read(run / 'resolved_config.json')
    cache = Path(config['feature_cache'])
    meta = read(cache / 'cache_meta.json')
    checkpoint = Path(config['model']['visual_checkpoint'])
    assert sha(checkpoint) == meta['checkpoint_sha256']
    assert sha(cache / 'manifest.jsonl') == meta['manifest_sha256']
    sources = {p: sha(REPO / p) for p in meta['implementation_sha256']}
    rows = [json.loads(line) for line in (cache / 'manifest.jsonl').read_text().splitlines()]
    train_rows = {int(row['frame_id']): row for row in rows if row['split'] == 'train'}
    print('AUDIT_ALL_4190_CACHE_RECORDS', flush=True)
    feature_hash = {}
    duplicate_features = []
    for i, row in enumerate(rows):
        for name, shape in [('visual_feature', (106, 256)), ('future_trajectory', (5, 2))]:
            ref = row[name]
            path = cache / ref['path']
            assert sha(path) == ref['sha256'], (name, row['frame_id'])
            assert path.stem == f"{row['frame_id']:04d}" and path.parent.name == row['route_id']
            array = np.load(path, allow_pickle=False)
            assert array.shape == shape and np.isfinite(array).all()
        digest = row['visual_feature']['sha256']
        if digest in feature_hash:
            duplicate_features.append([feature_hash[digest], row['frame_id']])
        feature_hash[digest] = row['frame_id']
        if (i + 1) % 1000 == 0:
            print(f'CACHE_HASH_OK {i + 1}', flush=True)

    dataset_args = SimpleNamespace(dataset_root=Path(config['dataset']),
        split_file=(ROOT / config['split_file']).resolve(), batch_size=1, workers=0)
    dataset, loader = builder.make_dataset_and_loader(dataset_args, 'train')
    from torch.utils.data import SequentialSampler
    assert isinstance(loader.sampler, SequentialSampler) and not loader.drop_last
    assert len(dataset) == len(train_rows)
    for index, (route_id, frame_id) in enumerate(dataset.route_frames):
        row = train_rows[frame_id]
        assert route_id == row['route_id']
    model, strict_load = builder.load_feature_model(checkpoint, torch.device('cuda'))
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    frames = [0, 1, 100, 580, 990, 991, 992, 1590, 1591, 2490, 2491,
              2790, 2791, 2820, 2835, 2850, 2865, 2885, 2947, 2955,
              2970, 2995, 3050, 3345]
    results = []
    for frame in frames:
        index = next(i for i, (_, f) in enumerate(dataset.route_frames) if f == frame)
        inputs, targets = dataset[index]
        assert int(targets['frame_id']) == frame
        assert int(targets['native_sync_frame']) == train_rows[frame]['native_sync_frame']
        gt = np.load(cache / train_rows[frame]['future_trajectory']['path'], allow_pickle=False)
        assert np.array_equal(gt, targets['waypoints'].numpy())
        group_begin = (index // args.batch_size) * args.batch_size
        group_indices = list(range(group_begin, min(group_begin + args.batch_size, len(dataset))))
        batch = loader.collate_fn([dataset[i] for i in group_indices])[0]
        batch = {key: value.to('cuda') for key, value in batch.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            fresh = model(batch)[index - group_begin].float().cpu().numpy().astype(np.float16)
        cached = np.load(cache / train_rows[frame]['visual_feature']['path'], allow_pickle=False)
        diff = fresh.astype(np.float32) - cached.astype(np.float32)
        distances = {}
        for offset in [-10, -5, -1, 0, 1, 5, 10]:
            other_frame = frame + offset
            if other_frame in train_rows:
                other = np.load(cache / train_rows[other_frame]['visual_feature']['path'], allow_pickle=False)
                distances[str(offset)] = float(np.linalg.norm(fresh.astype(np.float32) - other.astype(np.float32)))
        result = dict(frame_id=frame, native_sync_frame=int(targets['native_sync_frame']),
            max_abs_feature_difference=float(np.abs(diff).max()),
            mean_abs_feature_difference=float(np.abs(diff).mean()),
            exact_feature_match=bool(np.array_equal(fresh, cached)),
            neighbor_feature_l2_distances=distances,
            exact_target_match=True, current_speed=float(inputs['velocity'][0]))
        results.append(result)
        print(json.dumps(result), flush=True)
    assert all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items())
    assert sources == {p: sha(REPO / p) for p in sources}
    report = dict(complete=True, optimizer_updates=0, recomputation_batch_size=args.batch_size,
        all_cache_records=len(rows),
        artifact_hashes_checked=2 * len(rows), all_cache_hashes_shapes_finite_pass=True,
        duplicate_feature_artifact_hash_pairs=duplicate_features,
        raw_recomputed_train_frames=len(results), results=results,
        all_recomputed_features_exact=all(row['exact_feature_match'] for row in results),
        checkpoint_sha256=meta['checkpoint_sha256'], strict_load=strict_load,
        source_hashes=sources, source_changed_since_cache={p: sources[p] != h for p, h in meta['implementation_sha256'].items()},
        sequential_cache_loader_verified=True, weights_unchanged=True,
        production_sources_unchanged=True,
        limitations='All cache artifacts are checked; raw feature recomputation is sampled on train frames only. '
                    'This tests cache pairing, not simulator capture-content simultaneity. No dataset modification.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('CACHE_SAMPLE_ALIGNMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
