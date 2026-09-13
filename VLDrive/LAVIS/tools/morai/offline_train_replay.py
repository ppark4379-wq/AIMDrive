#!/usr/bin/env python3
"""Train-only offline replay and comparison to recorded synchronized ego poses.

No simulator control and no optimizer updates. Uniform windows are reported
separately from deliberately selected diagnostic cases. Plots use vehicle
forward/left coordinates; the prediction head itself uses [left,-forward].
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, bootstrap, sha256
configure_runtime()
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def metrics(rows):
    if not rows:
        return {'samples': 0}
    errors = np.asarray([r['errors_m'] for r in rows])
    return dict(samples=len(rows), ade_m=float(errors.mean()), fde_m=float(errors[:, -1].mean()),
        ade_median_m=float(np.median(errors.mean(1))), ade_p90_m=float(np.quantile(errors.mean(1), .9)),
        fde_p90_m=float(np.quantile(errors[:, -1], .9)), horizon_l2_m=errors.mean(0).tolist(),
        fde_below_1m_fraction=float((errors[:, -1] < 1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=32)
    parser.add_argument('--raw-windows', type=int, default=2)
    parser.add_argument('--plots-only', action='store_true', help='Render already completed replay without rerunning inference')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if args.plots_only:
        if ROOT not in output.parents:
            raise ValueError('Report output must stay inside LAVIS')
        summary = read(output / 'train_analysis.json')
        render_plots(output, read(output / 'recorded_pose_comparison.json'), summary['diagnostic_cases'],
                     summary['epoch_loss'], read(run / 'resolved_config.json'))
        print(json.dumps(dict(phase='plots_complete', output=str(output))), flush=True)
        return
    if ROOT not in output.parents or output.exists() or args.windows < 1 or not 0 <= args.raw_windows <= 6:
        raise ValueError('Use a new output inside LAVIS and valid sample counts')
    output.mkdir(parents=True)
    settings, provenance = read(run / 'resolved_config.json'), read(run / 'provenance.json')
    report = read(run / 'report.json')
    if report.get('optimizer_updates') != report['schedule_updates']:
        raise ValueError('Training is not complete')
    checkpoint = run / 'model-last.pt'
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    from lavis.runners.vldrive_trainer import seed_all, source_hashes, restore_model_checkpoint
    from lavis.runners.morai_trainer import batch, autocast
    backend = bootstrap(ROOT.parent)
    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
        split_file=ROOT / settings['split_file'], instructions=settings['instructions'],
        split='train', sequence_length=settings['sequence_length'])
    if (backend != provenance['backend'] or source_hashes(ROOT) != provenance['source']
            or dataset.describe() != provenance['datasets']['train']):
        raise ValueError('Offline replay source/backend/train data differs from training')
    if sha256(Path(settings['model']['visual_checkpoint'])) != dataset.checkpoint_sha256:
        raise ValueError('Visual checkpoint/cache mismatch')
    train_indices = read(run / 'window_indices.json')['train']
    saved = read(run / 'train_predictions.json')
    if len(saved) != len(train_indices):
        raise ValueError('Train prediction/index count mismatch')

    @lru_cache(None)
    def measurements(route):
        return read(Path(settings['dataset']) / 'data' / route / 'measurements_all.json')

    def recorded_gt(route, frame):
        records = measurements(route)
        ego = records[frame]
        theta = float(ego['theta'])
        gt = []
        for offset in (5, 10, 15, 20, 25):
            future = records[frame + offset]
            assert future['native_sync_frame'] - ego['native_sync_frame'] == offset * 5
            dx, dy = future['gps_x'] - ego['gps_x'], future['gps_y'] - ego['gps_y']
            forward = np.cos(theta) * dx + np.sin(theta) * dy
            left = -np.sin(theta) * dx + np.cos(theta) * dy
            gt.append([left, -forward])
        return np.asarray(gt, dtype=np.float32), ego

    rows, cache_gt_delta = [], 0.0
    for index, prediction in zip(train_indices, saved):
        start, end = dataset.windows[index]
        entry = dataset.entries[end - 1]
        route, frame = entry['route_id'], int(entry['frame_id'])
        assert prediction['sample_id'] == route + '/%04d' % frame
        gt, ego = recorded_gt(route, frame)
        target = dataset._array(entry['future_trajectory'], (5, 2), 'future trajectory').numpy()
        cache_gt_delta = max(cache_gt_delta, float(np.max(np.abs(gt - target))))
        pred = np.asarray(prediction['trajectory'])
        stationary = float(np.linalg.norm(gt, axis=1).max()) < .5
        turning = abs(float(ego.get('steer', 0))) > .1
        category = 'stationary' if stationary else ('moving_turn' if turning else 'moving_straight')
        row = dict(window_index=index, sample_id=prediction['sample_id'], frame_id=frame, route_id=route,
            category=category, speed_mps=float(ego['speed']), steer=float(ego.get('steer', 0)),
            instruction=entry['instruction'], actual_future=gt.tolist(), prediction=pred.tolist(),
            errors_m=np.linalg.norm(pred-gt, axis=1).tolist(), selected_count=prediction['selected_count'])
        rows.append(row)
    assert cache_gt_delta < 1e-5
    all_metrics = metrics(rows)
    assert abs(all_metrics['ade_m'] - report['after']['train']['ade_m']) < 1e-5
    groups = {key: metrics([r for r in rows if r['category'] == key])
              for key in ('stationary', 'moving_straight', 'moving_turn')}
    groups['all_moving'] = metrics([r for r in rows if r['category'] != 'stationary'])
    counts = np.asarray([r['selected_count'] for r in rows])
    logs = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines() if line]
    epoch_loss = []
    for epoch in range(settings['epochs']):
        values = logs[epoch*len(train_indices):(epoch+1)*len(train_indices)]
        epoch_loss.append(dict(epoch=epoch+1, **{key:float(np.mean([r[key] for r in values]))
            for key in ('waypoint_loss', 'pruning_loss', 'reconstruction_loss')}))

    positions = np.linspace(0, len(rows)-1, min(args.windows, len(rows)), dtype=int).tolist()
    cases = {}
    for category in groups:
        if category == 'all_moving':
            continue
        subset = [i for i,r in enumerate(rows) if r['category'] == category]
        if subset:
            ordered = sorted(subset, key=lambda i: np.mean(rows[i]['errors_m']))
            cases[category+'_median'] = ordered[len(ordered)//2]
            cases[category+'_worst'] = ordered[-1]
    chosen = sorted(set(positions + list(cases.values())))
    summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), split='train_only',
        completed_updates=report['optimizer_updates'], training_before=report['before']['train'],
        training_after=all_metrics, groups=groups, cached_gt_vs_recorded_pose_max_difference_m=cache_gt_delta,
        token_selection=dict(mean=float(counts.mean()), min=int(counts.min()), max=int(counts.max()),
            expected_retention=.3, observed_retention=float(counts.mean()/106)), epoch_loss=epoch_loss,
        uniform_selection_count=len(positions), diagnostic_cases=cases)
    write(output / 'train_analysis.json', summary)
    write(output / 'recorded_pose_comparison.json', rows)
    print(json.dumps(dict(phase='analysis', train=all_metrics, groups=groups,
                         tokens=summary['token_selection'])), flush=True)

    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda')
    state = restore_model_checkpoint(checkpoint, model, provenance=provenance)
    assert state['updates'] == report['optimizer_updates']
    del state
    model.eval()
    replays = []
    with torch.inference_mode():
        for position in chosen:
            item = batch(dataset[rows[position]['window_index']], 'cuda')
            torch.cuda.synchronize()
            started = time.perf_counter()
            with autocast(model):
                pred = model.predict(item['model_inputs'], compute_auxiliary=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            value = pred['trajectory'][0,-1].float().cpu().numpy()
            row = dict(rows[position], prediction=value.tolist(),
                selected_count=pred['selected_count'][0].cpu().tolist(),
                errors_m=np.linalg.norm(value-np.asarray(rows[position]['actual_future']), axis=1).tolist(),
                replay_vs_saved_max_difference_m=float(np.max(np.abs(value-np.asarray(rows[position]['prediction'])))),
                elapsed_s=elapsed, selection='uniform' if position in positions else 'diagnostic')
            replays.append(row)
    summary['fresh_cached_replay'] = dict(all_selected=metrics(replays),
        uniform_only=metrics([r for r in replays if r['selection']=='uniform']),
        max_vs_saved_difference_m=max(r['replay_vs_saved_max_difference_m'] for r in replays))
    write(output / 'fresh_replay.json', replays)
    print(json.dumps(dict(phase='fresh_replay', **summary['fresh_cached_replay'])), flush=True)

    raw_results = []
    if args.raw_windows:
        from tools.morai.cache_expert_vision_tokens import make_dataset_and_loader, load_feature_model
        from lavis.common.morai_runtime import complete_visual_checkpoint_initialization
        raw_args = SimpleNamespace(dataset_root=Path(settings['dataset']), split_file=ROOT/settings['split_file'],
                                   batch_size=4, workers=0)
        raw_dataset, unused_loader = make_dataset_and_loader(raw_args, 'train')
        lookup = {key:i for i,key in enumerate(raw_dataset.route_frames)}
        with complete_visual_checkpoint_initialization():
            encoder, strict = load_feature_model(Path(settings['model']['visual_checkpoint']), torch.device('cuda'))
        raw_positions = [cases[key] for key in ('stationary_median', 'moving_turn_median', 'moving_straight_median')
                         if key in cases][:args.raw_windows]
        if len(raw_positions) < args.raw_windows:
            raw_positions = chosen[:args.raw_windows]
        with torch.inference_mode():
            for position in raw_positions:
                index = rows[position]['window_index']
                begin,end = dataset.windows[index]
                entries = dataset.entries[begin:end]
                inputs = [raw_dataset[lookup[(r['route_id'], int(r['frame_id']))]][0] for r in entries]
                from torch.utils.data._utils.collate import default_collate
                features = []
                for offset in range(0,len(inputs),4):
                    sensors = {k:v.cuda() for k,v in default_collate(inputs[offset:offset+4]).items()}
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        features.append(encoder(sensors).float())
                features = torch.cat(features)[None]
                with autocast(model):
                    pred = model.predict(dict(visual_feature=features, instruction=[rows[position]['instruction']]),
                                         compute_auxiliary=False)
                value = pred['trajectory'][0,-1].float().cpu().numpy()
                cached = batch(dataset[index], 'cuda')['model_inputs']['visual_feature']
                raw_results.append(dict(sample_id=rows[position]['sample_id'], source='RGB+LiDAR+speed -> encoder -> trained model',
                    actual_future=rows[position]['actual_future'], prediction=value.tolist(),
                    errors_m=np.linalg.norm(value-np.asarray(rows[position]['actual_future']),axis=1).tolist(),
                    cached_feature_mae=float((features-cached).abs().mean()),
                    prediction_vs_cached_max_difference_m=float(np.max(np.abs(value-np.asarray(rows[position]['prediction'])))),
                    strict_encoder_load=strict))
        summary['raw_sensor_replay'] = dict(metrics=metrics(raw_results), cases=raw_results)
    write(output / 'train_analysis.json', summary)
    render_plots(output, rows, cases, epoch_loss, settings)
    print(json.dumps(dict(phase='complete', output=str(output), raw=summary.get('raw_sensor_replay'))),flush=True)


def render_plots(output, rows, cases, epoch_loss, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 3.2*len(cases)), squeeze=False)
    for axrow, (name,position) in zip(axes, cases.items()):
        row = rows[position]
        frame = row['frame_id']
        with Image.open(Path(settings['dataset'])/'data'/row['route_id']/'rgb_front'/('%04d.jpg'%frame)) as im:
            axrow[0].imshow(im)
        axrow[0].axis('off')
        axrow[0].set_title('%s | frame %d | %.2f m/s' % (name,frame,row['speed_mps']))
        ax = axrow[1]
        for points,color,label in [(row['actual_future'],'tab:blue','Recorded expert'),(row['prediction'],'tab:orange','Model')]:
            p=np.vstack((np.zeros((1,2)),np.asarray(points)))
            ax.plot(-p[:,0],-p[:,1], 'o-',color=color,label=label,
                    markersize=7 if label == 'Recorded expert' else 4)
        ax.set(xlabel='Right (m)',ylabel='Forward (m)', title='ADE %.3f m | FDE %.3f m' %
               (np.mean(row['errors_m']),row['errors_m'][-1]))
        ax.set_aspect('equal', adjustable='datalim')
        if row['category'] == 'stationary':
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-1, 1); ax.set_ylim(-.25, 1.75)
        ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(output/'train_scenes.png',dpi=150); plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot([r['epoch'] for r in epoch_loss],[r['waypoint_loss'] for r in epoch_loss], 'o-')
    axes[0].set(xlabel='Epoch',ylabel='Mean waypoint L1',title='Training loss (all 11 frames)')
    axes[1].scatter([r['frame_id']/10 for r in rows],[np.mean(r['errors_m']) for r in rows],s=8)
    axes[1].set(xlabel='Recorded sample time (s)',ylabel='ADE (m)',title='Train anchor errors')
    axes[2].plot([r['epoch'] for r in epoch_loss],[r['pruning_loss'] for r in epoch_loss], 'o-')
    axes[2].set(xlabel='Epoch',ylabel='Pruning penalty',title='CCDP target retention: 30%')
    for ax in axes: ax.grid(True)
    fig.tight_layout(); fig.savefig(output/'train_summary.png',dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
