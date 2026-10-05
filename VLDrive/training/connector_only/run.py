"""Cache real sensors, train CCDP/MEFA with reconstruction, and pair history ablations.

Run from VLDrive: python training/connector_only/run.py --stage all
No LLM or future-waypoint labels enter this experiment.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
local_deps = ROOT / '.runtime/mefa_deps'
if local_deps.is_dir():
    sys.path.insert(0, str(local_deps))

import numpy as np
import torch
from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from utils.common.runtime import configure_runtime, sha256


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def noise_scale_at_update(schedule, update):
    if schedule is None:
        return 1.0
    start, end = schedule['start'], schedule['end']
    duration = schedule['decay_updates']
    if not 0 <= end <= start <= 1 or duration < 1 or update < 0:
        raise ValueError('Invalid Gumbel noise schedule')
    fraction = min(update / duration, 1.0)
    return start + (end - start) * fraction


def select_windows(times, speeds, length, interval, count, max_gap_ns):
    """Never bridge missing inputs or clock gaps; spread pilot anchors over the drive."""
    candidates = []
    span = (length - 1) * interval
    for end in range(span, len(times)):
        ids = list(range(end - span, end + 1, interval))
        if (all(speeds[i] is not None for i in ids)
                and all(0 < times[i] - times[i - 1] <= max_gap_ns
                        for i in range(end - span + 1, end + 1))):
            candidates.append(ids)
    if len(candidates) < count:
        raise ValueError('Not enough valid full-history windows')
    return [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]


def cache(cfg, output, chosen=None):
    from models.ve.vision_encoder import create_vision_encoder
    from training.connector_only.native_inputs import EgoNativeSensors
    weights = Path(cfg['visual_checkpoint'])
    blob = torch.load(weights, map_location='cpu', weights_only=True)
    recipe = blob['morai_encoder_contract']
    model = create_vision_encoder({'name': 'memfuser_baseline_e1d3_return_feature'}, pretrained_backbone=False)
    strict = str(model.load_state_dict(blob['state_dict'], strict=True))
    del blob
    model.to(cfg['device']).eval().requires_grad_(False)
    data_root = Path(cfg['dataset_root'])
    if chosen is None:
        chosen = []
        for category in ('dynamic_obstacle', 'static_obstacle'):
            episodes = sorted((data_root / category).glob('*/episode_000'))
            n, v = cfg['train_episodes_per_category'], cfg['validation_episodes_per_category']
            if min(n, v) < 1 or len(episodes) < n + v:
                raise ValueError('Insufficient independent episodes for both splits')
            chosen.extend((category, p, 'train') for p in episodes[:n])
            chosen.extend((category, p, 'validation') for p in episodes[-v:])
    identity = dict(config=cfg, visual_sha256=sha256(weights), encoder_contract=recipe,
                    episodes=[dict(path=str(p), split=s, category=c) for c, p, s in chosen],
                    sources={str(p.relative_to(ROOT)): sha256(p) for p in (
                        Path(__file__), Path(__file__).with_name('native_inputs.py'),
                        ROOT / 'data/src/loaders/navigation_sensor_record.py',
                        ROOT / 'data/src/loaders/morai_transfer_adapter.py',
                        ROOT / 'models/ve/vision_encoder.py', ROOT / 'models/ve/pointpillars.py',
                        ROOT / 'models/ve/resnet.py', ROOT / 'models/ve/transformer.py')})
    identity_path = output / 'cache_source.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Cache source/config changed; choose a new output directory')
    write_json(identity_path, identity)
    records, source_audits = [], {}
    cache_dir = output / 'features'
    cache_dir.mkdir(exist_ok=True)
    length = cfg['connector']['memory_size'] + 1
    for category, episode, split in chosen:
        emit('reading_episode', episode=str(episode), split=split)
        dataset = EgoNativeSensors(episode, recipe, category, cfg['max_sensor_gap_ms'])
        source_audits[dataset.identity] = dataset.provenance
        windows = select_windows(dataset.times, dataset.velocities, length, cfg['frame_interval'],
                                 cfg['windows_per_episode'], cfg['max_sensor_gap_ms'] * 1_000_000)
        for number, indices in enumerate(windows):
            key = hashlib.sha256(f'{dataset.identity}:{indices}'.encode()).hexdigest()[:24]
            path = cache_dir / (key + '.npy')
            record_path = cache_dir / (key + '.json')
            if path.exists() and record_path.exists():
                record = json.loads(record_path.read_text())
                if record['sha256'] != sha256(path):
                    raise ValueError('Changed cached feature bytes')
            else:
                features = []
                for index in indices:
                    inputs = {k: v.unsqueeze(0).to(cfg['device']) for k, v in dataset.inputs(index).items()}
                    with torch.inference_mode():
                        value = model(inputs)[0].float().cpu().numpy()
                    if value.shape != (106, 256) or not np.isfinite(value).all():
                        raise ValueError('Invalid encoder output')
                    features.append(value)
                array = np.stack(features).astype(np.float16)
                if not np.isfinite(array).all():
                    raise ValueError('Feature storage overflow')
                pending = path.with_suffix('.pending')
                with pending.open('wb') as stream:
                    np.save(stream, array, allow_pickle=False)
                pending.replace(path)
                record = dict(episode=dataset.identity, category=category, split=split,
                              indices=indices, times_ns=[dataset.times[i] for i in indices],
                              path=str(path.relative_to(output)), sha256=sha256(path))
                write_json(record_path, record)
            records.append(record)
            emit('cached_window', episode=dataset.identity, window=number + 1, total=len(windows))
    write_json(output / 'manifest.json', dict(records=records, strict_load=strict,
               visual_sha256=identity['visual_sha256'], encoder_contract=recipe, source_audits=source_audits,
               scope='pilot; episodes disjoint for connector, encoder pretraining overlap not established'))


def load_records(output, evaluation_only=False):
    manifest = json.loads((output / 'manifest.json').read_text())
    records = manifest['records']
    train_episodes = {r['episode'] for r in records if r['split'] == 'train'}
    val_episodes = {r['episode'] for r in records if r['split'] == 'validation'}
    if evaluation_only:
        if not val_episodes or train_episodes or any(r['split'] != 'validation' for r in records):
            raise ValueError('Evaluation-only manifest must contain validation records only')
    elif not train_episodes or not val_episodes or train_episodes & val_episodes:
        raise ValueError('Invalid episode split')
    for row in records:
        path = output / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('Changed feature cache')
        value = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32)).unsqueeze(0)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite cached features')
        row['tensor'] = value
    return records


def paired_evaluation(model, records, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for row in records:
            x = row['tensor'].to(device)
            normal = model(x, history_mode='correct', compute_reconstruction=True)
            mask = normal['selection_mask'].clone()
            zero = model(x, history_mode='zero', hard_mask=mask, compute_reconstruction=True)
            for key in ('selection_mask', 'selected_indices', 'selected_valid_mask', 'reconstruction_target'):
                if not torch.equal(normal[key], zero[key]):
                    raise AssertionError('Paired condition changed: ' + key)
            # Only the last frame has the full Z-frame history. Empty reconstruction
            # targets are excluded, never interpreted as a perfect zero loss.
            dropped = int(normal['dropped_mask'][-1].sum().item())
            valid = dropped > 0
            a = float(normal['reconstruction_loss_per_frame'][-1]) if valid else None
            b = float(zero['reconstruction_loss_per_frame'][-1]) if valid else None
            if valid and not np.isfinite([a, b]).all():
                raise ValueError('Nonfinite evaluation loss')
            rows.append(dict(episode=row['episode'], category=row['category'],
                             current_time_ns=row['times_ns'][-1], dropped_tokens=dropped,
                             retention=float(mask[0, -1].mean()), normal_lrec=a, zero_lrec=b,
                             delta_lrec=b-a if valid else None))
    valid = [r for r in rows if r['delta_lrec'] is not None]
    summary = dict(windows=len(rows), valid_windows=len(valid),
                   empty_target_windows=len(rows)-len(valid),
                   retention=float(np.mean([r['retention'] for r in rows])))
    for key in ('normal_lrec', 'zero_lrec', 'delta_lrec'):
        summary[key] = float(np.mean([r[key] for r in valid])) if valid else None
    summary['positive_delta_fraction'] = float(np.mean([r['delta_lrec'] > 0 for r in valid])) if valid else None
    summary['by_episode'] = {episode: dict(windows=len(group),
        delta_lrec=float(np.mean([r['delta_lrec'] for r in group])))
        for episode in sorted({r['episode'] for r in valid})
        if (group := [r for r in valid if r['episode'] == episode])}
    return dict(summary=summary, frames=rows)


def save_checkpoint(path, model, optimizer, epoch, cfg, output):
    payload = dict(schema='connector-reconstruction-v1', epoch=epoch,
                   connector_config=asdict(model.config), config=cfg,
                   manifest_sha256=sha256(output / 'manifest.json'),
                   cache_source_sha256=sha256(output / 'cache_source.json'),
                   state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
                   scope='pilot, randomly initialized Q-Former; no LLM')
    pending = path.with_suffix('.pending')
    torch.save(payload, pending)
    pending.replace(path)


def train(cfg, output, resume=False):
    records = load_records(output)
    training = [r for r in records if r['split'] == 'train']
    validation = [r for r in records if r['split'] == 'validation']
    model = QFormerMEFAConnector(QFormerMEFAConfig(**cfg['connector'])).to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    start = 0
    path = output / 'model-last.pt'
    if path.exists() and not resume:
        raise ValueError('Existing checkpoint; use --resume or a new output')
    if resume:
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (state['config'] != cfg or state['manifest_sha256'] != sha256(output / 'manifest.json')
                or state['cache_source_sha256'] != sha256(output / 'cache_source.json')):
            raise ValueError('Resume provenance mismatch')
        model.load_state_dict(state['state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        start = state['epoch']
        del state
    if not start:
        report = paired_evaluation(model, validation, cfg['device'])
        write_json(output / 'evaluation-initial.json', report)
        emit('initial_evaluation', **report['summary'])
    started = time.monotonic()
    for epoch in range(start + 1, cfg['epochs'] + 1):
        model.train()
        order = list(range(len(training)))
        random.shuffle(order)
        losses, gradients = [], {}
        for step, index in enumerate(order, 1):
            update = (epoch - 1) * len(training) + step - 1
            model.ccdp.gumbel_noise_scale = noise_scale_at_update(cfg.get('gumbel_noise_schedule'), update)
            x = training[index]['tensor'].to(cfg['device'])
            optimizer.zero_grad(set_to_none=True)
            result = model(x, selection_mode='gumbel', compute_reconstruction=True)
            # One supervised current frame, preceded by a complete context window.
            rec = result['reconstruction_loss_per_frame'][-1]
            pruning = result['selector']['pruning_loss_per_frame'][0, -1]
            loss = model.config.lambda_reconstruction * rec + model.config.lambda_pruning * pruning
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if step == 1:
                for group in ('ccdp.', 'temporal_encoder.', 'qformer.', 'reconstruction_transformer.'):
                    norm = sum(float(p.grad.detach().float().square().sum()) for name, p in model.named_parameters()
                               if name.startswith(group) and p.grad is not None) ** 0.5
                    gradients[group] = norm
                    if not np.isfinite(norm) or norm == 0:
                        raise ValueError('Missing/nonfinite gradient: ' + group)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append([float(rec.detach()), float(pruning.detach()),
                           float(result['selection_mask'][0, -1].detach().mean())])
            if step % 8 == 0:
                emit('training', epoch=epoch, step=step, steps=len(order), elapsed_s=round(time.monotonic()-started, 1))
        report = paired_evaluation(model, validation, cfg['device'])
        metrics = dict(epoch=epoch, train_lrec=float(np.mean(losses, axis=0)[0]),
                       gumbel_noise_scale=model.ccdp.gumbel_noise_scale,
                       train_lprun=float(np.mean(losses, axis=0)[1]),
                       train_retention=float(np.mean(losses, axis=0)[2]), gradients=gradients,
                       validation=report['summary'], elapsed_s=round(time.monotonic()-started, 1))
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        write_json(output / 'evaluation-last.json', report)
        save_checkpoint(path, model, optimizer, epoch, cfg, output)
        emit('epoch_complete', **metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('local_mefa.json'))
    parser.add_argument('--stage', choices=('cache', 'train', 'evaluate', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    configure_runtime()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    output = ROOT / cfg['output']
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required by this experiment')
    emit('start', stage=args.stage, output=str(output), torch=str(torch.__version__),
         device=torch.cuda.get_device_name(0) if cfg['device'] == 'cuda' else 'cpu')
    if args.stage in ('cache', 'all'):
        cache(cfg, output)
        torch.cuda.empty_cache()
    if args.stage in ('train', 'all'):
        train(cfg, output, args.resume)
    if args.stage == 'evaluate':
        state = torch.load(output / 'model-last.pt', map_location='cpu', weights_only=True)
        if state['manifest_sha256'] != sha256(output / 'manifest.json'):
            raise ValueError('Checkpoint dataset mismatch')
        model = QFormerMEFAConnector(QFormerMEFAConfig(**state['connector_config']))
        model.load_state_dict(state['state_dict'], strict=True)
        model.to(cfg['device'])
        validation = [r for r in load_records(output) if r['split'] == 'validation']
        report = paired_evaluation(model, validation, cfg['device'])
        write_json(output / 'evaluation-last.json', report)
        emit('evaluation', **report['summary'])


if __name__ == '__main__':
    main()
