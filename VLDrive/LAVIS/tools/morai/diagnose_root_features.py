#!/usr/bin/env python3
"""Read-only train feature/head audit; no optimizer, no model changes, CPU only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / 'vision_encoder'))
from timm.models.memfuser import GRUWaypointsPredictor


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(value):
    value = np.asarray(value, dtype=np.float64)
    return dict(min=float(value.min()), median=float(np.median(value)),
                max=float(value.max()), mean=float(value.mean()))


def metrics(prediction, target, mask):
    if not np.any(mask):
        return dict(count=0)
    error = np.linalg.norm(prediction[mask] - target[mask], axis=-1)
    return dict(count=int(mask.sum()), ade_m=float(error.mean()), fde_m=float(error[:, -1].mean()),
                endpoint_lateral_mae_m=float(np.abs(prediction[mask, -1, 0] - target[mask, -1, 0]).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert ROOT in output.parents and not output.exists(), 'Use a new LAVIS output file'
    torch.set_num_threads(1)
    config = read(args.run / 'resolved_config.json')
    cache = Path(config['feature_cache'])
    rows = [json.loads(line) for line in (cache / 'manifest.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['split'] == 'train']
    assert len({row['route_id'] for row in rows}) == 1
    route = rows[0]['route_id']
    measurements = read(Path(config['dataset']) / 'data' / route / 'measurements_all.json')
    frames = np.array([row['frame_id'] for row in rows])
    feature = np.stack([np.load(cache / row['visual_feature']['path'], allow_pickle=False) for row in rows]).astype(np.float32)
    target = np.stack([np.load(cache / row['future_trajectory']['path'], allow_pickle=False) for row in rows])
    command = []
    red = []
    for frame in frames:
        item = measurements[int(frame)]
        theta = float(item['theta'])
        dx, dy = item['x_command'] - item['gps_x'], item['y_command'] - item['gps_y']
        command.append([-np.sin(theta) * dx + np.cos(theta) * dy,
                        -np.cos(theta) * dx - np.sin(theta) * dy])
        red.append(bool(item.get('is_red_light_present')))
    command = np.asarray(command, dtype=np.float32)
    red = np.asarray(red)
    moving = np.linalg.norm(target, axis=-1).max(axis=-1) > .5
    curved = moving & (np.abs(target[:, -1, 0]) > 1.)
    masks = dict(all=np.ones(len(rows), dtype=bool), stationary=~moving,
                 moving=moving, curved=curved, moving_straight=moving & ~curved)
    checkpoint = torch.load(config['model']['visual_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint['state_dict']
    head = GRUWaypointsPredictor(256).eval()
    prefix = 'waypoints_generator.'
    head.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
    traffic_head = torch.nn.Linear(256, 2).eval()
    prefix = 'traffic_light_pred_head.'
    traffic_head.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
    result = dict(split='train_only', train_frames=len(rows), optimizer_updates=0,
                  feature_cache=str(cache), frozen_encoder_sha256=config['model']['visual_sha256'],
                  note='Existing trained heads only; input interventions are diagnostic, not valid counterfactual driving labels.')
    with torch.inference_mode():
        logits = traffic_head(torch.from_numpy(feature[:, 100])).numpy()
        predicted_red = logits.argmax(-1) == 0
        result['traffic_head'] = dict(accuracy=float((predicted_red == red).mean()),
            red_count=int(red.sum()), predicted_red_count=int(predicted_red.sum()),
            false_red_count=int((predicted_red & ~red).sum()), missed_red_count=int((~predicted_red & red).sum()),
            transitions=[dict(frame=int(frames[i]), red=bool(red[i]), predicted_red=bool(predicted_red[i]),
                              logits=logits[i].tolist()) for i in range(1, len(rows)) if red[i] != red[i-1]])
        departures = []
        for frame in (991, 1591, 2491, 2791):
            indices = np.flatnonzero((frames >= frame - 2) & (frames <= frame + 30))
            departures.append(dict(first_green_frame=frame, timeline=[
                dict(frame=int(frames[i]), red_label=bool(red[i]), head_red=bool(predicted_red[i]),
                     speed_mps=float(measurements[int(frames[i])]['speed'])) for i in indices]))
        result['traffic_head']['departure_timelines'] = departures
        result['head_interventions'] = {}
        predictions = {}
        for mode in ('recorded_command', 'zero_command_lateral', 'mean_waypoint_tokens'):
            values = []
            for start in range(0, len(rows), 128):
                stop = min(start + 128, len(rows))
                tokens = feature[start:stop, 101:106].copy()
                point = command[start:stop].copy()
                if mode == 'zero_command_lateral':
                    point[:, 0] = 0
                if mode == 'mean_waypoint_tokens':
                    tokens[:] = feature[:, 101:106].mean(axis=0)
                values.append(head(torch.from_numpy(tokens), torch.from_numpy(point)).numpy())
            pred = np.concatenate(values)
            predictions[mode] = pred
            result['head_interventions'][mode] = {name: metrics(pred, target, mask) for name, mask in masks.items()}
    result['token_temporal_variation'] = {}
    for name, block in [('bev_100', feature[:, :100]), ('traffic_1', feature[:, 100:101]), ('waypoint_5', feature[:, 101:106])]:
        result['token_temporal_variation'][name] = dict(
            element_temporal_std=stats(block.std(axis=0)),
            per_frame_rms=stats(np.sqrt((block * block).mean(axis=(1, 2)))),
            centered_variance_fraction=float(np.var(block, axis=0).sum() / np.maximum((block * block).mean(axis=0).sum(), 1e-20)))
    counts = Counter(row['visual_feature']['sha256'] for row in rows)
    result['identical_cache_frames'] = dict(unique_hashes=len(counts), duplicated_frames=sum(n for n in counts.values() if n > 1))
    # This is only a representation diagnostic. Neighbour labels are not model inputs.
    # Excluding +/-25 frames reduces trivial adjacent-frame copying, but is NOT a new held-out split.
    saved = read(args.run / 'train_predictions.json')
    row_by_frame = {int(frame): index for index, frame in enumerate(frames)}
    query_indices = np.array([row_by_frame[int(row['sample_id'].rsplit('/', 1)[1])] for row in saved])
    reference_indices = np.arange(0, len(rows), 5)
    result['nearest_feature_diagnostic'] = {}
    for name, block in [('all_tokens', feature), ('waypoint_tokens', feature[:, 101:106])]:
        query = block[query_indices].reshape(len(query_indices), -1).astype(np.float64)
        reference = block[reference_indices].reshape(len(reference_indices), -1).astype(np.float64)
        distance = ((query * query).sum(-1)[:, None] + (reference * reference).sum(-1)[None, :] - 2 * query @ reference.T)
        distance = np.maximum(distance, 0)
        for exclusion in (0, 25):
            eligible = np.abs(frames[query_indices, None] - frames[reference_indices][None, :]) > exclusion
            nearest = reference_indices[np.where(eligible, distance, np.inf).argmin(-1)]
            pred = target[nearest]
            result['nearest_feature_diagnostic'][name + '_exclude_' + str(exclusion)] = {
                category: metrics(pred, target[query_indices], mask[query_indices]) for category, mask in masks.items()}
    result['original_downstream_same_anchors'] = {
        category: metrics(np.asarray([row['trajectory'] for row in saved]), target[query_indices], mask[query_indices])
        for category, mask in masks.items()}
    result['selected_frames'] = [dict(frame=int(frame), actual=target[row_by_frame[frame]].tolist(),
        head_predictions={mode: value[row_by_frame[frame]].tolist() for mode, value in predictions.items()})
        for frame in (990, 991, 995, 2830, 2865, 2955, 3320) if frame in row_by_frame]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
