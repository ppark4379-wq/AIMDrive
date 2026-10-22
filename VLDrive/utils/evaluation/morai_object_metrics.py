"""Fixed-policy, class-agnostic oriented BEV object diagnostics for MORAI LAV.

This is a diagnostic of decoded objects, not 3D AP or class AP. It deliberately
uses fixed thresholds rather than selecting thresholds on validation results.
The head predicts no object class; model breakdowns condition recall on GT.
"""
import math
from collections import defaultdict

import numpy as np

CONFIDENCE = 0.5
NMS_IOU = 0.3
MATCH_IOUS = (0.3, 0.5)
BEV_EXTENT = (-25., 25., -20., 30.)  # right min/max, forward min/max, metres
_POLICY = {
    'schema': 'morai_oriented_bev_object_diagnostic_v2',
    'intersection_implementation': 'float64_sutherland_hodgman_local_origin_v1',
    'prediction_policy': 'lmdrive_lav_v1_raw_logits',
    'confidence_threshold': CONFIDENCE,
    'nms_iou_threshold': NMS_IOU,
    'matching_iou_thresholds': list(MATCH_IOUS),
    'axes': 'right_forward_metres',
    'bev_extent_m': list(BEV_EXTENT),
    'candidate_policy': 'all finite cells with sigmoid(logit) >= 0.5; no local-peak filter',
    'inclusion_policy': 'positive-area BEV intersection; full un-clipped boxes used for NMS and matching',
    'partial_box_policy': 'included and separately counted, even when center is outside BEV',
    'matching_policy': 'descending confidence, best unmatched GT IoU; ties use cell then GT input order',
    'range_policy': 'GT center horizontal radial distance from ego; [0,10), [10,20), [20,30), [30,40), [40,infinity) metres',
    'geometry_policy': 'upright BEV rectangle from heading_deg, full length and width, matching the target renderer; no height/roll/pitch evaluation',
    'invalid_prediction_policy': 'reject nonfinite output, nonpositive size, zero heading, or degenerate/unrepresentable float64 corners',
    'background_policy': 'false boxes per frame on valid frames with zero included GT boxes',
    'note': 'Class-agnostic fixed-threshold BEV counts, not AP. Per-model metrics are GT-conditioned recall. Counts are frame-object observations, not unique actors. No camera visibility or occlusion filter.',
}


def _key(value):
    return '{:.1f}'.format(value)


def _ratio(a, b):
    return float(a) / b if b else None


def _counts(tp, fp, fn):
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'precision': _ratio(tp, tp + fp), 'recall': _ratio(tp, tp + fn),
            'f1': _ratio(2 * tp, 2 * tp + fp + fn)}


def _range_label(distance):
    return ('0_10m' if distance < 10 else '10_20m' if distance < 20 else
            '20_30m' if distance < 30 else '30_40m' if distance < 40 else '40m_plus')


def _rectangle(center, heading, length, width):
    center, heading = np.asarray(center, dtype=np.float64), np.asarray(heading, dtype=np.float64)
    norm = float(np.linalg.norm(heading))
    if (not np.isfinite([*center, *heading, length, width, norm]).all()
            or length <= 0 or width <= 0 or norm < 1e-8):
        raise ValueError('Invalid oriented rectangle')
    heading = heading / norm
    along = heading * (length / 2.)
    across = np.array([-heading[1], heading[0]]) * (width / 2.)
    corners = np.array([center + along + across, center + along - across,
                        center - along - across, center - along + across])
    # Preserve float64 through clipping. Casting nearly coincident polygons to
    # float32 made OpenCV 5.0 return a triangle (half the overlap) for some
    # otherwise identical rectangles, corrupting both NMS and matched IoU.
    if not np.isfinite(corners).all():
        raise ValueError('Unrepresentable oriented rectangle')
    polygon = corners
    area = float(length * width)
    if (not math.isfinite(area) or area <= 0
            or np.any(np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1) == 0)):
        raise ValueError('Degenerate oriented rectangle')
    return {'center': center, 'heading': heading, 'length': float(length),
            'width': float(width), 'polygon': polygon, 'area': area,
            'bounds': np.array([corners[:, 0].min(), corners[:, 0].max(),
                                corners[:, 1].min(), corners[:, 1].max()])}


_ROI = _rectangle([0., 5.], [0., 1.], 50., 50.)


def _intersection(a, b):
    """Convex clipping in float64, translated near zero for stable shoelace area."""
    aa, bb = a['bounds'], b['bounds']
    if aa[0] >= bb[1] or bb[0] >= aa[1] or aa[2] >= bb[3] or bb[2] >= aa[3]:
        return 0.
    origin = b['center']
    subject = a['polygon'] - origin
    clip = b['polygon'] - origin

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    # _rectangle always emits clockwise corners. Inside is to the right of
    # each clip edge; no overlap tolerance or threshold adjustment is used.
    for i in range(4):
        if len(subject) < 3:
            return 0.
        start, end = clip[i - 1], clip[i]
        edge = end - start
        output = []
        previous = subject[-1]
        previous_distance = -cross(edge, previous - start)
        for point in subject:
            distance = -cross(edge, point - start)
            previous_inside, inside = previous_distance >= 0., distance >= 0.
            if inside != previous_inside:
                fraction = previous_distance / (previous_distance - distance)
                output.append(previous + fraction * (point - previous))
            if inside:
                output.append(point)
            previous, previous_distance = point, distance
        subject = np.asarray(output, dtype=np.float64)
    if len(subject) < 3:
        return 0.
    area = .5 * abs(np.sum(subject[:, 0] * np.roll(subject[:, 1], -1)
                           - subject[:, 1] * np.roll(subject[:, 0], -1)))
    return max(0., min(float(area), a['area'], b['area']))


def oriented_iou(a, b):
    """IoU of decoded rectangles returned by ``decode_object_boxes``."""
    intersection = _intersection(a, b)
    return intersection / (a['area'] + b['area'] - intersection)


def _inside_bev(box):
    return _intersection(box, _ROI) > 0.


def _partial(box):
    x0, x1, y0, y1 = box['bounds']
    return bool(x0 < -25. or x1 > 25. or y0 < -20. or y1 > 30.)


def decode_object_boxes(prediction, *, score_threshold=CONFIDENCE, nms_iou=NMS_IOU):
    """Raw (2500,8) LAV outputs -> NMS boxes and rejection counts.

    Each box contains numpy arrays for internal geometry. Use the frame helper
    for a JSON-safe result. Heading channels [right,-forward] are converted to
    [right,forward]; offsets *3.5, full length *7, full width *4.
    Defaults are the frozen evaluation policy. Diagnostic callers may disable
    NMS with None or explicitly sweep thresholds; live callers are unchanged.
    """
    if not 0. < score_threshold < 1.:
        raise ValueError('score_threshold must be between zero and one')
    if nms_iou is not None and not 0. <= nms_iou <= 1.:
        raise ValueError('nms_iou must be None or between zero and one')
    if hasattr(prediction, 'detach'):
        prediction = prediction.detach().cpu().numpy()
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (2500, 8):
        raise ValueError('Expected a raw (2500, 8) LAV detection head')
    finite_scores = np.isfinite(values[:, 0])
    logit_threshold = math.log(score_threshold / (1. - score_threshold))
    selected = np.flatnonzero(finite_scores & (values[:, 0] >= logit_threshold))
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
        candidates.append(box)
    candidates.sort(key=lambda b: (-b['score'], b['cell_index']))
    kept = []
    for candidate in candidates:
        if nms_iou is not None and any(oriented_iou(candidate, earlier) > nms_iou for earlier in kept):
            counts['nms_suppressed'] += 1
        else:
            kept.append(candidate)
    counts['decoded_boxes'] = len(kept)
    return kept, counts


def _ground_truth(entry):
    ego = entry['ego_pose']
    x, y, yaw = float(ego['x']), float(ego['y']), float(ego['yaw'])
    if not np.isfinite([x, y, yaw]).all():
        raise ValueError('Invalid ego pose in supervised frame')
    c, s = math.cos(yaw), math.sin(yaw)
    boxes, outside = [], 0
    for index, obj in enumerate(entry['objects']):
        if obj.get('geometry_valid') is not True:
            raise ValueError('Unverified GT geometry in supervised frame')
        geometry = [float(obj[k]) for k in ('x', 'y', 'heading_deg', 'length', 'width', 'height')]
        if not np.isfinite(geometry).all() or min(geometry[3:]) <= 0:
            raise ValueError('Invalid GT geometry in supervised frame')
        dx, dy = geometry[0] - x, geometry[1] - y
        right, forward = s * dx - c * dy, c * dx + s * dy
        angle = math.radians(geometry[2]) - yaw
        box = _rectangle([right, forward], [-math.sin(angle), math.cos(angle)], geometry[3], geometry[4])
        if not _inside_bev(box):
            outside += 1
            continue
        radius = math.hypot(right, forward)
        box.update(object_index=index, instance_key=str(obj.get('instance_key', obj.get('id', index))),
                   model_name=str(obj.get('model_name', 'unknown')), range_m=radius,
                   range_bin=_range_label(radius), partial=_partial(box))
        boxes.append(box)
    return boxes, outside


def _match(predictions, truths, threshold):
    available = set(range(len(truths)))
    matches = {}
    for pi, pred in enumerate(predictions):
        if not available:
            break
        best = max(sorted(available), key=lambda gi: oriented_iou(pred, truths[gi]))
        iou = oriented_iou(pred, truths[best])
        if iou >= threshold:
            matches[best] = (pi, iou)
            available.remove(best)
    return matches


def _geometry_error(pred, truth):
    dot = float(np.clip(abs(np.dot(pred['heading'], truth['heading'])), 0., 1.))
    return {'center_l2_m': float(np.linalg.norm(pred['center'] - truth['center'])),
            'length_abs_error_m': abs(pred['length'] - truth['length']),
            'width_abs_error_m': abs(pred['width'] - truth['width']),
            'heading_abs_error_deg_mod180': math.degrees(math.acos(dot))}


def evaluate_object_frame(prediction, entry):
    """JSON-safe per-frame diagnostics; never score an unsupervised frame.

    The entry must be the sealed manifest entry for this exact prediction and
    already share its corrected coordinate basis. No map/time correction is
    performed here. Malformed valid GT raises instead of becoming background.
    """
    if entry.get('object_valid') is not True:
        return {'supervised': False, 'skip_reason': 'object_valid_false_or_missing'}
    predictions, rejected = decode_object_boxes(prediction)
    truths, outside = _ground_truth(entry)
    matchings = {_key(t): _match(predictions, truths, t) for t in MATCH_IOUS}
    counts = {_key(t): _counts(len(matchings[_key(t)]), len(predictions) - len(matchings[_key(t)]),
                               len(truths) - len(matchings[_key(t)])) for t in MATCH_IOUS}
    rows = []
    for gi, truth in enumerate(truths):
        rows.append({'object_index': truth['object_index'], 'instance_key': truth['instance_key'],
                     'model_name': truth['model_name'], 'center_right_forward_m': truth['center'].tolist(),
                     'range_m': truth['range_m'], 'range_bin': truth['range_bin'],
                     'partial_box': truth['partial'],
                     'best_prediction_iou': max((oriented_iou(p, truth) for p in predictions), default=0.),
                     'matches': {k: (dict(prediction_index=m[gi][0], iou=m[gi][1],
                                          prediction_cell_index=predictions[m[gi][0]]['cell_index'],
                                          **_geometry_error(predictions[m[gi][0]], truth))
                                      if gi in m else None) for k, m in matchings.items()}})
    return {'supervised': True, 'gt_boxes': len(truths), 'gt_outside_bev': outside,
            'prediction_boxes': len(predictions), 'gt_partial_boxes': sum(b['partial'] for b in truths),
            'prediction_partial_boxes': sum(b['partial'] for b in predictions),
            'background_frame': not truths, 'background_false_boxes': len(predictions) if not truths else 0,
            'prediction_decode': rejected, 'matching': counts, 'gt_diagnostics': rows}


def _new_group():
    return {'gt_boxes': 0, 'partial_boxes': 0, 'best_iou_sum': 0.,
            'matching': {_key(t): {'tp': 0, 'errors': defaultdict(float)} for t in MATCH_IOUS}}


def _update_group(group, row):
    group['gt_boxes'] += 1
    group['partial_boxes'] += int(row['partial_box'])
    group['best_iou_sum'] += row['best_prediction_iou']
    for threshold, match in row['matches'].items():
        if match is not None:
            group['matching'][threshold]['tp'] += 1
            for key in ('center_l2_m', 'length_abs_error_m', 'width_abs_error_m', 'heading_abs_error_deg_mod180'):
                group['matching'][threshold]['errors'][key] += match[key]


def _group_result(group):
    result = {'gt_boxes': group['gt_boxes'], 'partial_boxes': group['partial_boxes'],
              'best_prediction_iou_mean': _ratio(group['best_iou_sum'], group['gt_boxes']), 'matching': {}}
    for threshold, values in group['matching'].items():
        result['matching'][threshold] = {'tp': values['tp'], 'fn': group['gt_boxes'] - values['tp'],
                                        'recall': _ratio(values['tp'], group['gt_boxes']),
                                        'matched_geometry_mean': {key: _ratio(values['errors'][key], values['tp'])
                                            for key in ('center_l2_m', 'length_abs_error_m', 'width_abs_error_m',
                                                        'heading_abs_error_deg_mod180')}}
    return result


class ObjectMetricsAccumulator:
    """Streaming accumulator without retaining predictions or per-frame rows."""
    def __init__(self):
        self.frames = 0
        self.supervised_frames = 0
        self.background_frames = 0
        self.background_false_boxes = 0
        self.totals = defaultdict(int)
        self.matching = {_key(t): defaultdict(int) for t in MATCH_IOUS}
        self.decode = defaultdict(int)
        self.by_model = defaultdict(_new_group)
        self.by_range = defaultdict(_new_group)
        self.overall_gt = _new_group()

    def update(self, prediction, entry):
        row = evaluate_object_frame(prediction, entry)
        self.frames += 1
        if not row['supervised']:
            return row
        self.supervised_frames += 1
        self.background_frames += int(row['background_frame'])
        self.background_false_boxes += row['background_false_boxes']
        for key in ('gt_boxes', 'gt_outside_bev', 'prediction_boxes', 'gt_partial_boxes', 'prediction_partial_boxes'):
            self.totals[key] += row[key]
        for key, value in row['prediction_decode'].items():
            self.decode[key] += value
        for threshold, counts in row['matching'].items():
            for key in ('tp', 'fp', 'fn'):
                self.matching[threshold][key] += counts[key]
        for gt in row['gt_diagnostics']:
            _update_group(self.overall_gt, gt)
            _update_group(self.by_model[gt['model_name']], gt)
            _update_group(self.by_range[gt['range_bin']], gt)
        return row

    def compute(self):
        """Return JSON-safe aggregate counts and GT-conditioned breakdowns."""
        return {'policy': dict(_POLICY), 'frames': self.frames,
                'supervised_frames': self.supervised_frames,
                'skipped_frames': self.frames - self.supervised_frames,
                **{key: self.totals[key] for key in ('gt_boxes', 'gt_outside_bev', 'prediction_boxes',
                                                     'gt_partial_boxes', 'prediction_partial_boxes')},
                'empty_frames': self.background_frames,
                'empty_frame_false_boxes': self.background_false_boxes,
                'false_positive_boxes_per_empty_frame': _ratio(self.background_false_boxes, self.background_frames),
                'prediction_decode': dict(self.decode),
                'thresholds': {k: {**{name: value for name, value in _counts(v['tp'], v['fp'], v['fn']).items()
                                             if name not in ('tp', 'fp', 'fn')},
                                    'true_positive': v['tp'], 'false_positive': v['fp'], 'false_negative': v['fn'],
                                    'false_boxes_per_supervised_frame': _ratio(v['fp'], self.supervised_frames)}
                               for k, v in self.matching.items()},
                'overall_gt': _group_result(self.overall_gt),
                'by_gt_model': {k: _group_result(v) for k, v in sorted(self.by_model.items())},
                'by_gt_range': {k: _group_result(v) for k, v in sorted(self.by_range.items())}}
