"""Shared metrics for native camera/LiDAR cycle-shift controls."""
from collections import Counter

import numpy as np

from async_capture.alignment import distribution


VIEWS = ('front', 'left', 'right', 'rear')
SHIFTS = list(range(-3, 4))
OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def edge_summary(rows):
    valid = [row for row in rows if row['valid']]
    if not valid:
        return {'samples': 0}
    scores = np.array([row['scores_px'] for row in valid])
    zero = SHIFTS.index(0)
    blocks = {}
    for row in valid:
        blocks.setdefault(row['time_block'], []).append(row['scores_px'])
    block_scores = np.array([np.mean(value, axis=0) for value in blocks.values()])
    return {
        'samples': len(valid),
        'time_blocks': len(blocks),
        'mean_score_px_by_shift': np.mean(scores, axis=0).tolist(),
        'equal_block_mean_score_px_by_shift': np.mean(block_scores, axis=0).tolist(),
        'best_shift_histogram': dict(Counter(
            str(SHIFTS[index]) for index in np.argmin(scores, axis=1)
        )),
        'controls': {
            str(shift): {
                'paired_error_increase_px': distribution(scores[:, index] - scores[:, zero]),
                'fraction_zero_strictly_better': float(np.mean(scores[:, index] > scores[:, zero])),
                'equal_block_error_increase_px': distribution(
                    block_scores[:, index] - block_scores[:, zero]
                ),
            }
            for index, shift in enumerate(SHIFTS) if shift
        },
        'decision': 'diagnostic_only_no_per_frame_exposure_certificate',
    }
