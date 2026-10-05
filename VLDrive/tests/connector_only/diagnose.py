"""Separate fixed-budget ranking diagnostic; never changes a trained checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.connector_only.run import (
    QFormerMEFAConfig, QFormerMEFAConnector, emit, load_records,
    paired_evaluation, sha256, torch, write_json,
)
import numpy as np


def diagnose(output, selection_mode='argmax', random_seeds=20):
    state = torch.load(output / 'model-last.pt', map_location='cpu', weights_only=True)
    if state['manifest_sha256'] != sha256(output / 'manifest.json'):
        raise ValueError('Checkpoint/data mismatch')
    cfg = state['config']
    torch.set_num_threads(cfg['cpu_threads'])
    model = QFormerMEFAConnector(QFormerMEFAConfig(**state['connector_config']))
    model.load_state_dict(state['state_dict'], strict=True)
    model.to(cfg['device']).eval()
    records = [r for r in load_records(output) if r['split'] == 'validation']
    model.config.inference_selection_mode = selection_mode
    report = paired_evaluation(model, records, cfg['device'])
    report['policy'] = selection_mode
    report['checkpoint_sha256'] = sha256(output / 'model-last.pt')
    rows = []
    with torch.inference_mode():
        for row in records:
            x = row['tensor'].to(cfg['device'])
            learned = model(x, selection_mode=selection_mode, compute_reconstruction=True)
            learned_mask = learned['selection_mask']
            for seed in range(random_seeds):
                generator = torch.Generator(device=cfg['device']).manual_seed(seed)
                random_mask = torch.zeros_like(learned_mask)
                for t in range(x.shape[1]):
                    count = int(learned_mask[0, t].sum())
                    ids = torch.randperm(x.shape[2], generator=generator, device=cfg['device'])[:count]
                    random_mask[0, t, ids] = 1
                other = model(x, hard_mask=random_mask, selection_mode=selection_mode, compute_reconstruction=True)
                common = (learned_mask[0, -1] == 0) & (random_mask[0, -1] == 0)
                if not common.any():
                    continue
                target = learned['reconstruction_target'][-1, common]
                a = float((learned['reconstructed_features'][-1, common] - target).square().mean())
                b = float((other['reconstructed_features'][-1, common] - target).square().mean())
                rows.append(dict(episode=row['episode'], current_time_ns=row['times_ns'][-1],
                    random_seed=seed, common_dropped_tokens=int(common.sum()),
                    learned_mse=a, random_mse=b, delta_random_minus_learned=b-a))
    if not rows:
        raise ValueError('No common reconstruction targets for ranking evaluation')
    counts = [106-r['dropped_tokens'] for r in report['frames']]
    report['selection_counts'] = dict(min=min(counts), max=max(counts), mean=float(np.mean(counts)),
                                    one_token_frames=sum(k == 1 for k in counts))
    report['ranking_diagnostic'] = dict(
        interpretation='Same token budget; score only original tokens dropped in BOTH conditions. '
                       'Random masks are an inference perturbation, not a separately trained baseline.',
        learned_mse=float(np.mean([r['learned_mse'] for r in rows])),
        random_mse=float(np.mean([r['random_mse'] for r in rows])),
        delta_random_minus_learned=float(np.mean([r['delta_random_minus_learned'] for r in rows])),
        comparisons=len(rows), random_seeds=random_seeds,
        by_episode={episode: dict(
            learned_mse=float(np.mean([r['learned_mse'] for r in rows if r['episode']==episode])),
            random_mse=float(np.mean([r['random_mse'] for r in rows if r['episode']==episode])),
            delta=float(np.mean([r['delta_random_minus_learned'] for r in rows if r['episode']==episode])))
            for episode in sorted({r['episode'] for r in rows})},
        frames=rows)
    write_json(output / ('evaluation-' + selection_mode + '-diagnostic.json'), report)
    emit('selection_diagnostic', policy=selection_mode, **report['summary'], counts=report['selection_counts'], ranking={k: v for k, v in report['ranking_diagnostic'].items() if k != 'frames'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-mode', choices=('argmax', 'topk'), default='argmax')
    parser.add_argument('--random-seeds', type=int, default=20)
    args = parser.parse_args()
    if args.random_seeds < 1:
        parser.error('random-seeds must be positive')
    diagnose(args.output, args.selection_mode, args.random_seeds)
