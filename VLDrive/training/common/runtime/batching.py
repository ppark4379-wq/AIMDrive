"""Training batching; existing implementations relocated without changing calculations."""
import random
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def batch(sample, device):
    return batch_samples([sample], device)


def batch_samples(samples, device):
    return move(default_collate(samples), device)


def subset_indices(length, count):
    if count <= 0 or count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def stride_indices(length, stride):
    """Cover a chronological route uniformly without an arbitrary tiny subset."""
    stride = int(stride)
    if length < 1 or stride < 1:
        raise ValueError('length and stride must be positive')
    values = list(range(0, length, stride))
    if values[-1] != length - 1:
        values.append(length - 1)
    return values


def balanced_training_indices(dataset, *, indices=None, draws=640, seed=607,
                              stationary_distance_m=.5, curved_lateral_m=2.):
    """Balance train anchors, never mutate the unique chronological eval list.

    This resamples existing clips, not new driving events. Classification uses
    only the train anchor's actual future trajectory in [left,-forward] axes.
    """
    if getattr(dataset, 'split', None) != 'train':
        raise ValueError('Balanced sampling is restricted to the training split')
    if getattr(dataset, 'trajectory_coordinate_convention', None) != 'lmdrive_left_negative_forward':
        raise ValueError('Balanced sampling requires [left,-forward] trajectories')
    if (int(draws) != draws or draws < 4 or draws % 4
            or not 0 < stationary_distance_m < curved_lateral_m):
        raise ValueError('Use positive thresholds and draws divisible by four')
    candidates = list(range(len(dataset))) if indices is None else list(indices)
    if (not candidates or len(candidates) != len(set(candidates))
            or any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(dataset) for i in candidates)):
        raise ValueError('Training candidates must be unique valid dataset indices')
    categories = {key: [] for key in ('stationary', 'moving_straight', 'moving_left', 'moving_right')}
    for index in sorted(candidates):
        # The cached dataset can read just five GT points without loading an
        # entire 11-frame visual window merely to choose its sampling category.
        if hasattr(dataset, 'entries') and hasattr(dataset, 'windows') and hasattr(dataset, '_array'):
            _, end = dataset.windows[index]
            target = dataset._array(dataset.entries[end-1]['future_trajectory'], (5, 2), 'future trajectory')
        else:
            target = dataset[index]['training_targets']['future_trajectory'][-1]
        target = torch.as_tensor(target).detach().cpu().float().numpy()
        if target.shape != (5, 2) or not np.isfinite(target).all():
            raise ValueError('Invalid train trajectory for balanced sampling')
        if float(np.linalg.norm(target, axis=1).max()) < stationary_distance_m:
            key = 'stationary'
        elif abs(float(target[-1, 0])) >= curved_lateral_m:
            key = 'moving_left' if target[-1, 0] > 0 else 'moving_right'
        else:
            key = 'moving_straight'
        categories[key].append(int(index))
    empty = [key for key, values in categories.items() if not values]
    if empty:
        raise ValueError('Cannot balance empty training categories: ' + ', '.join(empty))
    generator = random.Random(int(seed))
    per_category = int(draws) // 4
    sampled = []
    for values in categories.values():
        sampled.extend(generator.choices(values, k=per_category))
    generator.shuffle(sampled)
    audit = dict(policy='train-anchor-four-category-with-replacement-v1', split='train',
        seed=int(seed), draws=int(draws), unique_candidates=len(candidates),
        unique_sampled=len(set(sampled)), category_indices=categories,
        candidate_counts={key: len(values) for key, values in categories.items()},
        draw_counts={key: per_category for key in categories},
        stationary_distance_m=float(stationary_distance_m), curved_lateral_m=float(curved_lateral_m),
        note='Repeated clips are oversampling, not independent driving events; evaluate unique chronological indices separately.')
    return sampled, audit

