#!/usr/bin/env python3
"""Read-only audit of the original completed run's TRAIN supervision.

No network, CUDA, simulator control, source mutation or model-weight loading.
Only a new JSON diagnostic report is written.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def ranges(values):
    values = sorted(set(int(i) for i in values))
    result = []
    for frame in values:
        if result and frame == result[-1]['stop_frame'] + 1:
            result[-1]['stop_frame'] = frame
        else:
            result.append(dict(start_frame=frame, stop_frame=frame))
    for row in result:
        row['frames'] = row['stop_frame'] - row['start_frame'] + 1
        row['duration_s_at_10hz'] = row['frames'] / 10.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Choose a new output file inside LAVIS')
    settings, provenance = read(run/'resolved_config.json'), read(run/'provenance.json')
    spec = read(ROOT/settings['split_file'])
    assert len(spec['routes']) == 1 and settings['sequence_length'] == 11
    route_spec = spec['routes'][0]
    route = Path(settings['dataset'])/'data'/route_spec['path']
    measurements = read(route/'measurements_all.json')
    instruction_path = Path(settings['instructions'])
    assert digest(instruction_path) == provenance['datasets']['train']['instruction_sha256']
    instructions = [json.loads(line) for line in instruction_path.read_text().splitlines() if line]
    start, stop = route_spec['train']['start'], route_spec['train']['stop']
    train_frames = list(range(start, stop + 1))
    assert stop + 25 < route_spec['validation']['start']
    commands = {}
    for frame in train_frames:
        matches = [row for row in instructions if row['start_frame'] <= frame <= row['stop_frame']]
        assert len(matches) == 1
        commands[frame] = matches[0]['command']
    cache = Path(settings['feature_cache'])
    entries = {int(row['frame_id']): row for row in
        (json.loads(line) for line in (cache/'manifest.jsonl').read_text().splitlines()) if row['split'] == 'train'}
    assert set(entries) == set(train_frames)
    targets, heading_changes = {}, {}
    cache_gt_max_difference = 0.
    for frame in train_frames:
        ego = measurements[frame]
        theta = ego['theta']
        target = []
        for offset in (5, 10, 15, 20, 25):
            future = measurements[frame + offset]
            assert future['native_sync_frame'] - ego['native_sync_frame'] == 5 * offset
            dx, dy = future['gps_x'] - ego['gps_x'], future['gps_y'] - ego['gps_y']
            target.append([-np.sin(theta)*dx + np.cos(theta)*dy, -np.cos(theta)*dx - np.sin(theta)*dy])
        targets[frame] = np.asarray(target, dtype=np.float32)
        ref = entries[frame]['future_trajectory']
        assert digest(cache/ref['path']) == ref['sha256']
        cached = np.load(cache/ref['path'], allow_pickle=False)
        cache_gt_max_difference = max(cache_gt_max_difference, float(np.abs(cached-targets[frame]).max()))
        heading_changes[frame] = float(np.rad2deg(np.arctan2(
            np.sin(measurements[frame+25]['theta']-theta), np.cos(measurements[frame+25]['theta']-theta))))
    assert cache_gt_max_difference < 1e-5
    valid, excluded = [], []
    for anchor in train_frames[10:]:
        window = list(range(anchor-10, anchor+1))
        assert all(measurements[b]['native_sync_frame']-measurements[a]['native_sync_frame'] == 5
                   for a, b in zip(window, window[1:]))
        (valid if len({commands[frame] for frame in window}) == 1 else excluded).append(window)
    train_indices = read(run/'window_indices.json')['train']
    selected = [valid[index] for index in train_indices]
    predictions = read(run/'train_predictions.json')
    assert len(predictions) == len(selected)
    for prediction, window in zip(predictions, selected):
        assert prediction['sample_id'] == route_spec['path'] + '/%04d' % window[-1]
    supervised = [frame for window in selected for frame in window]
    unique_supervised = sorted(set(supervised))
    anchors = [window[-1] for window in selected]

    def category(frame):
        target = targets[frame]
        if np.linalg.norm(target, axis=1).max() < .5:
            return 'stationary'
        if abs(target[-1, 0]) >= 2:
            return 'curved_left' if target[-1, 0] > 0 else 'curved_right'
        return 'moving_near_straight'

    def composition(frames):
        by_category = collections.Counter(category(frame) for frame in frames)
        by_command = collections.Counter(commands[frame] for frame in frames)
        curved = [frame for frame in frames if category(frame).startswith('curved_')]
        other = [frame for frame in curved if commands[frame] not in (1, 2)]
        generic = [frame for frame in curved if commands[frame] == 4]
        result = dict(count=len(frames), category_counts=dict(by_category), command_counts=dict(by_command),
            curved_count=len(curved), curved_with_generic_instruction=len(generic),
            curved_with_generic_fraction=len(generic)/len(curved) if curved else None,
            curved_without_left_or_right_instruction=len(other),
            curved_without_left_or_right_fraction=len(other)/len(curved) if curved else None,
            instruction_by_curve={key: dict(collections.Counter(commands[frame] for frame in frames if category(frame)==key))
                                  for key in ('curved_left', 'curved_right')})
        points = np.stack([targets[frame] for frame in frames])
        result['final_lateral_abs_quantiles_m'] = np.quantile(np.abs(points[:, -1, 0]), [.0, .5, .9, .99, 1.]).tolist()
        return result

    explicit_turns = []
    for instruction in instructions:
        if instruction['command'] not in (1, 2) or instruction['start_frame'] > stop:
            continue
        event = set(range(max(start, instruction['start_frame']), min(stop, instruction['stop_frame'])+1))
        actual = sorted(event.intersection(unique_supervised))
        event_anchors = sorted(event.intersection(anchors))
        explicit_turns.append(dict(command=instruction['command'], start_frame=min(event), stop_frame=max(event),
            frames=len(event), duration_s=len(event)/10, unique_supervised_frames=len(actual),
            supervision_coverage=len(actual)/len(event), missing_supervision_ranges=ranges(event-set(actual)),
            selected_anchors=event_anchors,
            first_supervised_frame=min(actual), first_selected_anchor=min(event_anchors)))
    curve_bouts = []
    for direction in ('curved_left', 'curved_right'):
        for row in ranges(frame for frame in train_frames if category(frame)==direction):
            row.update(category=direction,
                commands=dict(collections.Counter(commands[frame] for frame in range(row['start_frame'], row['stop_frame']+1))))
            curve_bouts.append(row)

    prediction_values = np.asarray([row['trajectory'] for row in predictions])
    truth = np.stack([targets[frame] for frame in anchors])
    errors = np.linalg.norm(prediction_values-truth, axis=2)
    stationary = np.array([category(frame)=='stationary' for frame in anchors])
    by_instruction = []
    for command in sorted(set(commands[frame] for frame in anchors)):
        mask = np.array([commands[frame]==command for frame in anchors])
        # Descriptive in-sample statistic only, not a retrained or deployed model.
        median = np.median(truth[mask], axis=0)
        median_errors = np.linalg.norm(truth[mask]-median[None], axis=2)
        by_instruction.append(dict(command=command, anchors=int(mask.sum()),
            actual_final_lateral_median_m=float(np.median(truth[mask, -1, 0])),
            model_final_lateral_median_m=float(np.median(prediction_values[mask, -1, 0])),
            model_ade_m=float(errors[mask].mean()), model_fde_m=float(errors[mask, -1].mean()),
            same_training_group_median_trajectory=median.tolist(),
            same_training_group_median_ade_m=float(median_errors.mean()),
            same_training_group_median_fde_m=float(median_errors[:, -1].mean())))
    elapsed_positions = np.asarray([[measurements[frame]['gps_x'], measurements[frame]['gps_y']] for frame in train_frames])
    payload = dict(run=str(run), split='train_only', model_loaded=False, gpu_used=False,
        instruction_sha256=digest(instruction_path),
        counts=dict(raw_train_frames=len(train_frames), valid_windows=len(valid),
                    instruction_boundary_excluded_windows=len(excluded), selected_training_windows=len(selected),
                    weighted_supervision_positions=len(supervised), unique_supervised_frames=len(unique_supervised)),
        unique_recorded_positions=len(np.unique(elapsed_positions, axis=0)),
        traveled_distance_m=float(np.linalg.norm(np.diff(elapsed_positions, axis=0), axis=1).sum()),
        elapsed_train_sim_time_s=(stop-start)/10,
        cache_gt_vs_actual_pose_max_difference_m=cache_gt_max_difference,
        composition={key: composition(frames) for key, frames in (
            ('raw_train_frames', train_frames), ('unique_supervised_frames', unique_supervised),
            ('weighted_training_positions', supervised), ('evaluation_anchors', anchors))},
        missing_unique_supervision_ranges=ranges(set(train_frames)-set(unique_supervised)),
        explicit_turn_events=explicit_turns, curved_future_bouts=sorted(curve_bouts, key=lambda row: row['start_frame']),
        excluded_anchor_composition=composition([window[-1] for window in excluded]),
        final_error_share=dict(stationary_ade_sum_fraction=float(errors[stationary].sum()/errors.sum()),
                               moving_ade_sum_fraction=float(errors[~stationary].sum()/errors.sum())),
        conditional_instruction_statistics=by_instruction,
        interpretation_limits=[
            'Curved labels use >=2m final lateral offset in the next2.5s; bouts are not independent intersections.',
            'Stationary uses maximum actual future displacement<0.5m; not current speed alone.',
            'All future targets stay before validation, including the designated purge frames.',
            'Conditional median uses the same train labels; descriptive counterexample, not independent evaluation.',
            'Imbalance and instruction context differ from LMDrive, but these facts alone do not prove an inability to fit the two explicit turns.',
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: payload[key] for key in ('counts','unique_recorded_positions','traveled_distance_m',
        'cache_gt_vs_actual_pose_max_difference_m','explicit_turn_events','final_error_share','conditional_instruction_statistics')}, indent=2))
    print('DATA_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
