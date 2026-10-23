"""Opt-in MORAI object decoding with stable mathematical confidence order.

The historical decoder remains unchanged. This helper uses raw objectness
logits for NMS priority and returned-box order, avoiding rounded sigmoid ties
at large positive logits. Admission, geometry, output scores and NMS semantics
are otherwise identical to ``morai_object_metrics.decode_object_boxes``.
Raw-logit confidence is not a prediction of box quality.
"""
import math

import numpy as np

from .morai_object_metrics import (
    NMS_IOU, _inside_bev, _partial, _rectangle, oriented_iou,
)


def decode_object_boxes_raw_logit(prediction):
    """Decode a raw (2500, 8) head using descending logit, then cell index.

    Return the historical ``(boxes, counts)`` schema, including unchanged
    sigmoid scores. Returned boxes retain raw-logit order for greedy matching.
    This is an explicit opt-in; the canonical historical decoder is untouched.
    """
    if hasattr(prediction, 'detach'):
        prediction = prediction.detach().cpu().numpy()
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (2500, 8):
        raise ValueError('Expected a raw (2500, 8) LAV detection head')
    finite_scores = np.isfinite(values[:, 0])
    selected = np.flatnonzero(finite_scores & (values[:, 0] >= 0.))
    counts = {'candidate_cells': int(len(selected)),
              'nonfinite_logits': int((~finite_scores).sum()),
              'invalid_geometry': 0, 'outside_bev': 0, 'nms_suppressed': 0}
    candidates = []
    for index in selected:
        p = values[index]
        if not np.isfinite(p).all():
            counts['invalid_geometry'] += 1
            continue
        row, col = divmod(int(index), 50)
        try:
            box = _rectangle([col - 24.5 + p[1] * 3.5, 29.5 - row + p[2] * 3.5],
                             [p[3], -p[4]], p[5] * 7., p[6] * 4.)
        except ValueError:
            counts['invalid_geometry'] += 1
            continue
        if not _inside_bev(box):
            counts['outside_bev'] += 1
            continue
        box.update(score=float(1. / (1. + math.exp(-p[0]))), cell=[row, col],
                   cell_index=int(index), partial=_partial(box))
        candidates.append((float(p[0]), box))
    candidates.sort(key=lambda item: (-item[0], item[1]['cell_index']))
    kept = []
    for _, candidate in candidates:
        if any(oriented_iou(candidate, earlier) > NMS_IOU for earlier in kept):
            counts['nms_suppressed'] += 1
        else:
            kept.append(candidate)
    counts['decoded_boxes'] = len(kept)
    return kept, counts
