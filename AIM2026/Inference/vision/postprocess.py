"""Raw (2500, 8) detection head -> oriented boxes in map grid coordinates.

From VLDrive utils/evaluation/morai_object_metrics.decode_object_boxes and
morai_ve_expert_adapter.decode_world_predictions, without the GT evaluation.
Boxes are class-agnostic: no semantic class or persistent identity is produced.
The fixed policy is sigmoid >= 0.5, float64 oriented-IoU NMS at 0.3.
"""
import math

import numpy as np

from AIM2026.UDP.Detection import Box, MAX_BOX_SIZE_M, MAX_SPEED_MPS

CONFIDENCE = 0.5
NMS_IOU = 0.3
BEV_EXTENT = (-25., 25., -20., 30.)  # right min/max, forward min/max, metres
SPEED_SCALE = 8.                     # channel 7 * 8 = scalar speed in m/s


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
    # float64 throughout: float32 corners made near-identical boxes clip to a
    # triangle, corrupting NMS.
    if not np.isfinite(corners).all():
        raise ValueError('Unrepresentable oriented rectangle')
    area = float(length * width)
    if (not math.isfinite(area) or area <= 0
            or np.any(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1) == 0)):
        raise ValueError('Degenerate oriented rectangle')
    return {'center': center, 'heading': heading, 'length': float(length),
            'width': float(width), 'polygon': corners, 'area': area,
            'bounds': np.array([corners[:, 0].min(), corners[:, 0].max(),
                                corners[:, 1].min(), corners[:, 1].max()])}


_BEV = _rectangle([0., 5.], [0., 1.], 50., 50.)


def _intersection(a, b):
    """Convex clipping (Sutherland-Hodgman) in float64, near the origin for a stable area."""
    aa, bb = a['bounds'], b['bounds']
    if aa[0] >= bb[1] or bb[0] >= aa[1] or aa[2] >= bb[3] or bb[2] >= aa[3]:
        return 0.
    origin = b['center']
    subject = a['polygon'] - origin
    clip = b['polygon'] - origin

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    # _rectangle emits clockwise corners: inside is right of each clip edge.
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
    intersection = _intersection(a, b)
    return intersection / (a['area'] + b['area'] - intersection)


def _partial(box):
    x0, x1, y0, y1 = box['bounds']
    return bool(x0 < BEV_EXTENT[0] or x1 > BEV_EXTENT[1] or y0 < BEV_EXTENT[2] or y1 > BEV_EXTENT[3])


MERGE_IOMIN = 0.4
MERGE_MAX_ANGLE_DEG = 15.
MERGE_MAX_LENGTH_M = 14.


def merge_fragments(boxes):
    """Merge post-NMS boxes that are fragments of one long object (a truck seen as several boxes).

    Two boxes merge when the smaller lies mostly inside the larger (intersection / smaller area >= 0.4) and their
    headings agree within 15 degrees. The merged box keeps the higher-score box's heading, lateral centre and score,
    spans the union along that heading (at most 14 m) and takes the larger width. From VLDrive
    inference/ve_fragment_merge.py.
    """
    kept, merges = [], 0
    for box in boxes:                      # sorted by score, as NMS leaves them
        for i, k in enumerate(kept):
            h, hb = k['heading'] / np.linalg.norm(k['heading']), box['heading'] / np.linalg.norm(box['heading'])
            if math.degrees(math.acos(min(1., abs(float(h @ hb))))) > MERGE_MAX_ANGLE_DEG:
                continue
            o = oriented_iou(box, k)
            inter = o * (box['area'] + k['area']) / (1 + o)
            if inter / min(box['area'], k['area']) < MERGE_IOMIN:
                continue
            ends = [float((c - k['center']) @ h) for c in np.vstack([k['polygon'], box['polygon']])]
            lo, hi = min(ends), max(ends)
            if hi - lo > MERGE_MAX_LENGTH_M:
                continue
            merged = _rectangle(k['center'] + h * (lo + hi) / 2, h, hi - lo, max(k['width'], box['width']))
            merged.update(score=k['score'], cell_index=k['cell_index'], partial=_partial(merged))
            kept[i] = merged
            merges += 1
            break
        else:
            kept.append(box)
    return kept, merges


def decode_boxes(prediction, *, score_threshold=CONFIDENCE, nms_iou=NMS_IOU, merge=False):
    """(2500, 8) head -> ego-frame boxes (right, forward metres) after NMS, and rejection counts.

    Heading channels [right, -forward] become [right, forward]; offsets x3.5,
    full length x7, full width x4. Boxes must overlap the 50 x 50 m BEV window.
    """
    if not 0. < score_threshold < 1.:
        raise ValueError('score_threshold must be between zero and one')
    if nms_iou is not None and not 0. <= nms_iou <= 1.:
        raise ValueError('nms_iou must be None or between zero and one')
    if hasattr(prediction, 'detach'):
        prediction = prediction.detach().cpu().numpy()
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (2500, 8):
        raise ValueError('Expected a raw (2500, 8) detection head')
    finite_scores = np.isfinite(values[:, 0])
    logit_threshold = math.log(score_threshold / (1. - score_threshold))
    selected = np.flatnonzero(finite_scores & (values[:, 0] >= logit_threshold))
    counts = {'candidate_cells': int(len(selected)), 'nonfinite_logits': int((~finite_scores).sum()),
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
        if _intersection(box, _BEV) <= 0.:
            counts['outside_bev'] += 1
            continue
        box.update(score=float(1. / (1. + math.exp(-p[0]))), cell_index=int(index), partial=_partial(box))
        candidates.append(box)
    candidates.sort(key=lambda b: (-b['score'], b['cell_index']))
    kept = []
    for candidate in candidates:
        if nms_iou is not None and any(oriented_iou(candidate, earlier) > nms_iou for earlier in kept):
            counts['nms_suppressed'] += 1
        else:
            kept.append(candidate)
    if merge and nms_iou is not None:
        kept, counts['fragment_merges'] = merge_fragments(kept)
    counts['decoded_boxes'] = len(kept)
    return kept, counts


def to_world(center, heading, x, y, yaw):
    """Ego (right, forward) centre/heading -> map grid (x, y), heading degrees CCW from +x."""
    right, forward = map(float, center)
    hr, hf = map(float, heading)
    norm = math.hypot(hr, hf)
    if not all(math.isfinite(v) for v in (right, forward, hr, hf, norm, x, y, yaw)) or norm < 1e-8:
        raise ValueError('Invalid local box or ego pose')
    c, s = math.cos(yaw), math.sin(yaw)
    position = (x + c * forward + s * right, y + s * forward - c * right)
    hx, hy = c * hf + s * hr, s * hf - c * hr
    return position, math.degrees(math.atan2(hy, hx))


def decode_world_boxes(prediction, x, y, yaw_grid_rad, merge=False):
    """Detection head + the pose the inputs were captured at -> (UDP Box list, counts).

    Speed is the head's scalar speed (no direction); invalid values become None.
    Values outside the Detection wire limits would make the sender reject the
    whole frame, so such a speed becomes None and such a box is dropped (counted).
    """
    boxes, counts = decode_boxes(prediction, merge=merge)
    values = prediction.detach().float().cpu().numpy() if hasattr(prediction, 'detach') else np.asarray(prediction)
    result, invalid_speeds, oversize = [], 0, 0
    for box in boxes:
        if box['length'] > MAX_BOX_SIZE_M or box['width'] > MAX_BOX_SIZE_M:
            oversize += 1
            continue
        speed = float(values[box['cell_index'], 7]) * SPEED_SCALE
        if not math.isfinite(speed) or not 0. <= speed <= MAX_SPEED_MPS:
            speed, invalid_speeds = None, invalid_speeds + 1
        (wx, wy), heading_deg = to_world(box['center'], box['heading'], x, y, yaw_grid_rad)
        result.append(Box(wx, wy, heading_deg, float(box['width']), float(box['length']), speed))
    return result, dict(counts, invalid_speed_values=invalid_speeds, oversize_boxes=oversize)
