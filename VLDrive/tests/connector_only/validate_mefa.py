"""Frozen-checkpoint MEFA history ablation on previously unused episodes."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.connector_only.run import (
    ROOT, QFormerMEFAConfig, QFormerMEFAConnector, cache, configure_runtime,
    emit, load_records, np, paired_evaluation, sha256, torch, write_json,
)


def episode_summary(frames, seed=607, bootstrap_samples=10000):
    """Equal episode weights; paired cluster bootstrap, stratified by category."""
    episodes = []
    for identity in sorted({r['episode'] for r in frames}):
        all_rows = [r for r in frames if r['episode'] == identity]
        valid = [r for r in all_rows if r['delta_lrec'] is not None]
        if not valid:
            continue
        a = float(np.mean([r['normal_lrec'] for r in valid]))
        b = float(np.mean([r['zero_lrec'] for r in valid]))
        episodes.append(dict(episode=identity, category=valid[0]['category'],
            windows=len(all_rows), valid_windows=len(valid), normal_lrec=a, zero_lrec=b, delta_lrec=b-a))
    if not episodes:
        raise ValueError('No valid reconstruction targets')
    categories = sorted({r['category'] for r in episodes})
    rng = np.random.default_rng(seed)
    draws = np.zeros((bootstrap_samples, 2))
    by_category = {}
    for category in categories:
        group = [r for r in episodes if r['category'] == category]
        pair = np.array([[r['normal_lrec'], r['zero_lrec']] for r in group])
        draw = pair[rng.integers(0, len(pair), size=(bootstrap_samples, len(pair)))].mean(axis=1)
        draws += draw * len(pair) / len(episodes)
        a, b = pair.mean(axis=0)
        by_category[category] = dict(episodes=len(group), normal_lrec=float(a), zero_lrec=float(b),
            delta_lrec=float(b-a), positive_episodes=sum(r['delta_lrec'] > 0 for r in group),
            delta_ci95=(np.quantile(draw[:, 1]-draw[:, 0], [.025, .975]).tolist()
                        if len(group) >= 2 else None))
    a = float(np.mean([r['normal_lrec'] for r in episodes]))
    b = float(np.mean([r['zero_lrec'] for r in episodes]))
    delta_draws = draws[:, 1] - draws[:, 0]
    relative = 100 * delta_draws / draws[:, 1] if np.all(draws[:, 1] > 0) else None
    ci = np.quantile(delta_draws, [.025, .975]).tolist()
    adequate_clusters = all(group['episodes'] >= 2 for group in by_category.values())
    return dict(episodes=episodes, summary=dict(
        episodes=len(episodes), total_episodes=len({r['episode'] for r in frames}),
        frames=len(frames), valid_frames=sum(r['delta_lrec'] is not None for r in frames),
        normal_lrec=a, zero_lrec=b, delta_lrec=b-a,
        relative_error_reduction_percent=100*(b-a)/b if b > 0 else None,
        delta_ci95=ci if adequate_clusters else None, relative_error_reduction_ci95_percent=(
            np.quantile(relative, [.025, .975]).tolist() if relative is not None and adequate_clusters else None),
        positive_episodes=sum(r['delta_lrec'] > 0 for r in episodes),
        positive_frames=sum(r['delta_lrec'] is not None and r['delta_lrec'] > 0 for r in frames),
        bootstrap_samples=bootstrap_samples, bootstrap_unit='episode, paired, category-stratified',
        bootstrap_seed=seed, by_category=by_category,
        evidence=('insufficient independent episodes for stratified uncertainty' if not adequate_clusters else
                  'positive mean history contribution on sampled drives' if ci[0] > 0 else
                  'negative mean history contribution on sampled drives' if ci[1] < 0 else
                  'mean history contribution not resolved by this sample')))


def choose_episodes(root, excluded, count, seed):
    rng = random.Random(seed)
    chosen = []
    for category in ('dynamic_obstacle', 'static_obstacle'):
        available = [p for p in sorted((root / category).glob('*/episode_000'))
                     if f'{category}/{p.parent.name}/{p.name}' not in excluded]
        if len(available) < count:
            raise ValueError('Insufficient unused episodes: ' + category)
        chosen.extend((category, p, 'validation') for p in sorted(rng.sample(available, count)))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/logs/connector_mefa_dynamic_anneal/model-last.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/logs/mefa_history_validation_20_drives')
    parser.add_argument('--stage', choices=('all', 'cache', 'evaluate'), default='all')
    parser.add_argument('--episodes-per-category', type=int, default=10)
    parser.add_argument('--windows-per-episode', type=int, default=8)
    parser.add_argument('--seed', type=int, default=607)
    parser.add_argument('--visual-checkpoint', type=Path,
                        help='Relocated encoder file; must match the training encoder SHA256')
    parser.add_argument('--dataset-root', type=Path,
                        help='Relocated dataset_label root; original checkpoint provenance is preserved')
    parser.add_argument('--cached-only', action='store_true',
                        help='Re-evaluate the checkpoint run validation cache; not new held-out drives')
    args = parser.parse_args()
    if min(args.episodes_per_category, args.windows_per_episode) < 1:
        parser.error('Episode/window counts must be positive')
    if args.cached_only and args.stage != 'evaluate':
        parser.error('--cached-only requires --stage evaluate')
    configure_runtime()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    old_manifest_path = checkpoint.parent / 'manifest.json'
    old_source_path = checkpoint.parent / 'cache_source.json'
    if (sha256(old_manifest_path) != state['manifest_sha256']
            or sha256(old_source_path) != state['cache_source_sha256']):
        raise ValueError('Checkpoint source provenance mismatch')
    previous = json.loads(old_manifest_path.read_text())
    excluded = {r['episode'] for r in previous['records']
                if not args.cached_only or r['split'] == 'train'}
    cfg = dict(state['config'])
    if args.visual_checkpoint is not None:
        cfg['visual_checkpoint'] = str(args.visual_checkpoint.resolve())
    if args.dataset_root is not None:
        cfg['dataset_root'] = str(args.dataset_root.resolve())
    if args.cached_only:
        identities = sorted({(r['category'], r['episode']) for r in previous['records']
                             if r['split'] == 'validation'})
        chosen = [(category, Path(cfg['dataset_root']) / identity, 'validation')
                  for category, identity in identities]
    else:
        if not Path(cfg['dataset_root']).is_dir():
            raise FileNotFoundError('Raw dataset drive/path unavailable: ' + cfg['dataset_root'])
        chosen = choose_episodes(Path(cfg['dataset_root']), excluded, args.episodes_per_category, args.seed)
    config = dict(dataset_root=cfg['dataset_root'], visual_checkpoint=cfg['visual_checkpoint'],
                  device=cfg['device'], windows_per_episode=args.windows_per_episode,
                  frame_interval=cfg['frame_interval'], max_sensor_gap_ms=cfg['max_sensor_gap_ms'],
                  connector=state['connector_config'])
    plan = dict(schema='frozen-mefa-history-ablation-v1', checkpoint=str(checkpoint),
                checkpoint_sha256=digest, seed=args.seed, excluded_previous_episodes=sorted(excluded),
                episodes=[dict(category=c, path=str(p), split=s) for c, p, s in chosen], config=config,
                primary_contrast='zero_history_lrec - correct_history_lrec',
                primary_unit='equal episode means', bootstrap='10000 paired episode resamples within category',
                practical_minimum_effect='not prespecified; report effect magnitude, do not claim practical sufficiency',
                scope='fixed trained checkpoint only; no retraining or random-token comparison',
                validation_reused_from_prior_experiment=args.cached_only)
    plan_path = output / 'evaluation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Evaluation plan changed; use another output')
    write_json(plan_path, plan)
    if not args.cached_only and sha256(Path(cfg['visual_checkpoint'])) != previous['visual_sha256']:
        raise ValueError('Different vision encoder checkpoint')
    emit('mefa_validation_plan', episodes=len(chosen), windows=len(chosen)*args.windows_per_episode,
         excluded_episodes=len(excluded), output=str(output))
    if args.stage in ('all', 'cache'):
        # Release optimizer/checkpoint storage before running the vision encoder.
        del state
        cache(config, output, chosen=chosen)
        torch.cuda.empty_cache()
    if args.stage in ('all', 'evaluate'):
        if args.stage == 'all':
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        records = ([r for r in load_records(checkpoint.parent) if r['split'] == 'validation']
                   if args.cached_only else load_records(output, evaluation_only=True))
        actual = {r['episode'] for r in records}
        expected = {f'{c}/{p.parent.name}/{p.name}' for c, p, _ in chosen}
        if actual != expected or actual & excluded:
            raise ValueError('Evaluation episodes changed or overlap prior experiment')
        manifest_path = old_manifest_path if args.cached_only else output / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['visual_sha256'] != previous['visual_sha256'] or manifest['encoder_contract'] != previous['encoder_contract']:
            raise ValueError('Evaluation encoder differs from training')
        model = QFormerMEFAConnector(QFormerMEFAConfig(**state['connector_config']))
        model.load_state_dict(state['state_dict'], strict=True)
        del state
        model.to(cfg['device']).eval().requires_grad_(False)
        versions = {k: v._version for k, v in model.named_parameters()}
        frames = []
        for episode in sorted(actual):
            subset = [r for r in records if r['episode'] == episode]
            report = paired_evaluation(model, subset, cfg['device'])
            frames.extend(report['frames'])
            emit('mefa_episode_evaluated', episode=episode, **report['summary'])
        if any(p._version != versions[k] or p.grad is not None for k, p in model.named_parameters()):
            raise AssertionError('Model modified during evaluation')
        if sha256(checkpoint) != digest:
            raise AssertionError('Checkpoint file changed during evaluation')
        result = episode_summary(frames, args.seed)
        result.update(frames=frames, plan=plan, checkpoint_unchanged=True,
                      manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path),
                      sources={str(p.relative_to(ROOT)): sha256(p) for p in (
                          Path(__file__), Path(__file__).with_name('run.py'),
                          Path(__file__).with_name('native_inputs.py'),
                          ROOT/'models/connector/qformer_mefa.py', ROOT/'models/connector/ccdp.py')},
                      limitations=['One trained checkpoint; not a multi-seed training comparison.',
                                   'Same capture campaigns; repetitions of routes may be correlated.',
                                   'Vision encoder pretraining overlap is unknown.',
                                   'Zero history is an inference ablation, not a separately trained baseline.',
                                   'Reconstruction benefit only, not driving performance.'] + (
                                   ['Prior validation reused, not an untouched test set.',
                                    'Only two episodes: stratified episode bootstrap is degenerate; '
                                    'do not interpret its interval as generalization evidence.']
                                   if args.cached_only else []))
        if args.cached_only:
            result['summary']['delta_ci95'] = None
            result['summary']['relative_error_reduction_ci95_percent'] = None
            result['summary']['evidence'] = 'Descriptive recheck only; too few independent drives for reliable uncertainty'
            for category in result['summary']['by_category'].values():
                category['delta_ci95'] = None
        write_json(output / 'mefa_result.json', result)
        emit('mefa_validation_complete', **result['summary'])


if __name__ == '__main__':
    main()
