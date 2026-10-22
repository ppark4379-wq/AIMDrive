"""Experimental offline geometry readout; never enabled by live decoding.

Use only model proposals, never GT. Retain detection count, scores and cell IDs
to isolate geometry from detection. A retained ID still denotes the baseline
query, while geometry_source_cell records where experimental geometry came from.
"""
import numpy as np
from .morai_object_metrics import decode_object_boxes, oriented_iou

POLICY = dict(score_threshold=.5, nms_iou=.3, center_gate_m=.75,
              min_relative_score=.7, size_ratio_min=.5, size_ratio_max=2.,
              min_cluster=3, max_cluster=16, min_mean_iou_gain=.02)


def consensus_boxes(raw):
    kept, _ = decode_object_boxes(raw)
    proposals, _ = decode_object_boxes(raw, nms_iou=None)
    # Reproduce greedy NMS ownership exactly, including its first suppressor.
    clusters = [[] for _ in kept]
    for p in proposals:
        for i, leader in enumerate(kept):
            if p['cell_index'] == leader['cell_index'] or oriented_iou(p, leader) > .3:
                clusters[i].append(p)
                break
    result = list(kept); changed = 0
    for i, leader in enumerate(kept):
        eligible = [p for p in clusters[i]
                    if np.linalg.norm(p['center'] - leader['center']) <= .75
                    and p['score'] >= leader['score'] * .7
                    and .5 <= p['length'] / leader['length'] <= 2.
                    and .5 <= p['width'] / leader['width'] <= 2.][:16]
        if len(eligible) < 3:
            continue
        similarity = np.array([[oriented_iou(a, b) for b in eligible] for a in eligible])
        averages = (similarity.sum(1) - 1.) / (len(eligible) - 1)
        best = int(averages.argmax())
        # Confidence sorting makes eligible[0] the original leader.
        if averages[best] < averages[0] + .02:
            continue
        selected = eligible[best]
        # Do not introduce overlap beyond existing NMS policy with another box.
        if any(oriented_iou(selected, p) > .3 for j, p in enumerate(result) if j != i):
            continue
        result[i] = dict(selected, score=leader['score'], cell=leader['cell'],
                         cell_index=leader['cell_index'], geometry_source_cell=selected['cell_index'])
        changed += 1
    return result, dict(changed=changed, boxes=len(kept))
