"""Versioned offline predictions, separate from observations and GT."""
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def trajectory_metrics(prediction, ground_truth):
    if ground_truth is None:
        return dict(ade_m=None, fde_m=None)
    prediction, ground_truth = np.asarray(prediction), np.asarray(ground_truth)
    if prediction.shape != (5, 2) or ground_truth.shape != (5, 2):
        raise ValueError('Expected five future waypoint pairs')
    if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
        raise ValueError('Nonfinite trajectory')
    distances = np.linalg.norm(prediction - ground_truth, axis=-1)
    return dict(ade_m=float(distances.mean()), fde_m=float(distances[-1]))


class Results:
    def __init__(self, root, metadata):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(schema='vldrive-offline-replay-v1', status='preparing',
                             coordinate_convention='lmdrive_left_negative_forward',
                             situation_policy='predicted_only_in_static_zone_else_absent',
                             evaluation_scope='recorded_open_loop_training_data', **metadata)
        self.update()

    def update(self, **values):
        self.metadata.update(values)
        write_json(self.root / 'run.json', self.metadata)

    def episode(self, index, document):
        write_json(self.root / 'episodes' / f'{index:03d}.json', document)

    def prediction(self, index, value):
        path = self.root / 'predictions' / f'{index:03d}.jsonl'
        path.parent.mkdir(exist_ok=True)
        with path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
