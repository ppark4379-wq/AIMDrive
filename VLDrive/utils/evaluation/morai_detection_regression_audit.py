"""Offline failure triage. Geometry associations are proxies, not actor IDs.

Keep standard IoU detection counts and geometry-association coverage separate.
Never interpret missing adequate boxes as proof of a missing visual feature.
"""
import numpy as np

from .morai_object_metrics import (
    _ground_truth, _match, _intersection, _geometry_error,
    decode_object_boxes, oriented_iou,
)


def reliable_center_pairs(truths, predictions, gate_m=2., margin_m=.5):
    """Mutual nearest centers with a runner-up margin in both directions.

    No IoU admission cutoff: a unique badly sized box still measures regression.
    Reject identity conflicts where another GT has appreciably greater overlap.
    """
    if not truths or not predictions:
        return {}
    d = np.array([[np.linalg.norm(g['center'] - p['center']) for p in predictions]
                  for g in truths])
    result = {}
    for gi, g in enumerate(truths):
        pi = int(d[gi].argmin())
        if d[gi, pi] > gate_m or int(d[:, pi].argmin()) != gi:
            continue
        row = np.sort(d[gi]); col = np.sort(d[:, pi])
        if len(row) > 1 and row[1] - row[0] < margin_m:
            continue
        if len(col) > 1 and col[1] - col[0] < margin_m:
            continue
        overlaps = [oriented_iou(t, predictions[pi]) for t in truths]
        if max(overlaps) > overlaps[gi] + .05:
            continue
        result[gi] = pi
    return result


def audit_frame(raw, entry):
    if entry.get('object_valid') is not True:
        return {'supervised': False}
    gt, _ = _ground_truth(entry)
    boxes, counts = decode_object_boxes(raw)
    pre, _ = decode_object_boxes(raw, nms_iou=None)
    low, _ = decode_object_boxes(raw, score_threshold=.2)
    policies = {'baseline': boxes, 'no_nms': pre, 'score_02': low}
    matches = {k: _match(v, gt, .5) for k, v in policies.items()}
    match03 = _match(boxes, gt, .3)
    pairs = reliable_center_pairs(gt, boxes)
    # One prediction covering multiple GT footprints is a review flag only.
    shared = {pi: [gi for gi, g in enumerate(gt)
                   if _intersection(p, g) / g['area'] >= .25]
              for pi, p in enumerate(boxes)}
    shared = {pi: ids for pi, ids in shared.items() if len(ids) > 1}
    rows = []
    for gi, g in enumerate(gt):
        if gi in matches['baseline']:
            reason = 'iou05_pass'
        elif gi in pairs:
            reason = 'unique_center_bbox_error'
        elif gi in match03:
            reason = 'iou03_only_association_uncertain'
        elif gi in matches['no_nms']:
            reason = 'nms_counterfactual_recovery'
        elif gi in matches['score_02']:
            reason = 'low_score_counterfactual_recovery'
        elif any(gi in ids for ids in shared.values()):
            reason = 'shared_box_review'
        else:
            reason = 'no_adequate_box_or_ambiguous'
        geo = None
        if gi in pairs:
            p = boxes[pairs[gi]]
            geo = _geometry_error(p, g)
            geo['corner_m'] = float(min(np.linalg.norm(
                np.roll(p['polygon'], shift, axis=0) - g['polygon'], axis=1).mean()
                for shift in range(4)))
            geo['iou'] = float(oriented_iou(p, g))
        neighbor = min((float(np.linalg.norm(g['center'] - t['center']))
                        for j, t in enumerate(gt) if j != gi), default=None)
        rows.append(dict(model=g['model_name'], instance_key=g['instance_key'],
                         reason=reason, geometry=geo, partial=g['partial'],
                         range_m=g['range_m'], nearest_gt_center_m=neighbor,
                         prediction_cell=boxes[pairs[gi]]['cell_index'] if gi in pairs else None,
                         shared_box_flag=any(gi in ids for ids in shared.values()),
                         recovered_without_nms=gi not in matches['baseline'] and gi in matches['no_nms'],
                         recovered_score02=gi not in matches['baseline'] and gi in matches['score_02']))
    return dict(supervised=True, gt=len(gt), decode=counts, rows=rows,
                shared_box_candidates=len(shared),
                detection={k: dict(tp=len(matches[k]), fp=len(v)-len(matches[k]),
                                   fn=len(gt)-len(matches[k])) for k, v in policies.items()})
