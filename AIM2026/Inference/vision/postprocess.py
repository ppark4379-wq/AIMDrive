"""Raw (2500, 8) detection head -> oriented boxes in map grid coordinates.

From VLDrive utils/evaluation/morai_object_metrics.decode_object_boxes and
morai_ve_expert_adapter.decode_world_predictions, without the GT evaluation.
Boxes are class-agnostic: no semantic class or persistent identity is produced.
The fixed policy is sigmoid >= 0.5, float64 oriented-IoU NMS at 0.3.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

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


LIDAR_SEED_RADIUS_M = 1.5     # the object's points start within this distance of the detector's centre ...
LIDAR_CLUSTER_GAP_M = .5      # ... and grow through neighbours closer than this ...
LIDAR_CLUSTER_MAX_M = 4.      # ... never farther than this from the centre (walls, other objects)
LIDAR_MIN_POINTS = 15
LIDAR_MIN_LONG_M = 1.5
LIDAR_MIN_SHOWN = .6          # the outline must show this share of max(detector length, LIDAR_SHOWN_FLOOR_M)
LIDAR_SHOWN_FLOOR_M = 4.      # a car seen only from behind (a 2 m line) must not pass as a barrier
LIDAR_MIN_ASPECT = 1.8        # clearly elongated; irregular piles (rockslide) keep the detector's box
LIDAR_MAX_SHORT_M = 2.5       # an outline wider than a parked car is two objects in one cluster: keep the detector's box
LIDAR_SMALL_HEAD_M = 2.        # a head box this short is a small prop (barricade, firewall), not a car ...
LIDAR_SMALL_MIN_LONG_M = 1.    # ... and its outline, this long ...
LIDAR_SMALL_SLACK_M = .6       # ... and at most this much longer than the head box, gives its heading
LIDAR_MAX_LONG_M = 5.5         # a small head box's cluster longer than a car runs into a wall: keep the head box


def lshape_axis(points):
    """Long-axis direction (radians) of the 2-D points by the L-shape closeness criterion, and the fitted extents.

    Every 1-degree rectangle orientation in [0, 90) is scored by how close the points lie to its nearest edge; the
    best one is the outline a car's visible sides form, independent of how many points each side has.
    """
    angles = np.radians(np.arange(90.))
    e1 = np.stack([np.cos(angles), np.sin(angles)], 1)
    e2 = np.stack([-np.sin(angles), np.cos(angles)], 1)
    c1, c2 = points @ e1.T, points @ e2.T
    d1 = np.minimum(c1 - c1.min(0), c1.max(0) - c1)
    d2 = np.minimum(c2 - c2.min(0), c2.max(0) - c2)
    best = int(np.argmax(np.sum(1. / np.maximum(np.minimum(d1, d2), .05), axis=0)))
    l1, l2 = float(np.ptp(c1[:, best])), float(np.ptp(c2[:, best]))
    axis = angles[best] if l1 >= l2 else angles[best] + math.pi / 2
    return axis, max(l1, l2), min(l1, l2)


LIDAR_VOXEL_M = .1


def object_points(points, center, nearest_only=False):
    """The LiDAR points connected to the detector's centre (gaps < LIDAR_CLUSTER_GAP_M), within LIDAR_CLUSTER_MAX_M.

    Connectivity is computed on 0.1 m voxels: the all-pairs version took up to 0.5 s on a frame next to a building
    wall (4000 points), which stalled the inference; on the benchmark both give the same points for 1401 of 1403 boxes.
    """
    near = points[np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) < LIDAR_CLUSTER_MAX_M]
    if not len(near):
        return near
    voxels, index = np.unique(np.floor(near / LIDAR_VOXEL_M).astype(np.int64), axis=0, return_inverse=True)
    index = index.reshape(-1)
    centres = np.zeros((len(voxels), 2))
    np.add.at(centres, index, near)
    centres /= np.bincount(index)[:, None]
    member = np.zeros(len(voxels), dtype=bool)
    distance = np.hypot(near[:, 0] - center[0], near[:, 1] - center[1])
    if nearest_only:
        # Grow from the one point nearest the centre: a second object within the seed radius (two firewalls side
        # by side 0.6 m apart) is then left out unless the points really connect.
        if distance.min() >= LIDAR_SEED_RADIUS_M:
            return near[:0]
        member[index[int(np.argmin(distance))]] = True
    else:
        member[index[distance < LIDAR_SEED_RADIUS_M]] = True
    tree = cKDTree(centres)
    frontier = np.flatnonzero(member)
    while len(frontier):
        reached = np.unique(np.concatenate(tree.query_ball_point(centres[frontier], LIDAR_CLUSTER_GAP_M)))
        frontier = reached[~member[reached]]
        member[frontier] = True
    return near[member[index]]


def refine_heading_with_lidar(boxes, points):
    """Orient elongated objects by the outline of their LiDAR points; returns the number of boxes changed.

    In the middle range (8-25 m) the cells covering one parked car or barrier predict headings up to 90 degrees apart
    and NMS picks a different cell from frame to frame, so the box turns; thin barriers also come out too short.
    The visible sides are measured directly by the LiDAR: the heading comes from their outline, the size is the larger
    of the detector's and the outline's (the outline only shows the visible sides) and the box is widened to cover
    the points. The forward sign stays the one closest to the head's heading. ``points`` are (right, forward) metres
    in the boxes' frame, already limited to body height.
    """
    if points is None or not len(points):
        return 0
    changed = 0
    for box in boxes:
        cluster = object_points(points, box['center'])
        if len(cluster) < LIDAR_MIN_POINTS:
            continue
        axis, long_side, short_side = lshape_axis(cluster)
        head_long = max(box['length'], box['width'])
        shown = long_side >= max(LIDAR_MIN_LONG_M, LIDAR_MIN_SHOWN * max(box['length'], LIDAR_SHOWN_FLOOR_M))
        # Small props (a 1.36 m firewall, 1.8-1.9 m concrete barricades) never show the 2.4 m the car rule asks, so
        # their head heading (28-32 deg median error, 78-83 deg p90) was kept: a firewall across the lane turned
        # 40-60 deg narrowed a 3.25 m tunnel gap to 2.87 m and blocked the planner. Their outline matches the head
        # box in size; a car the head boxed as a fragment shows a longer outline (its 1.9 m rear) and stays out.
        # Tunnel benchmark: props 1.6 deg median error with the outline; 7 car frames of thousands turned wrongly.
        small = head_long <= LIDAR_SMALL_HEAD_M and LIDAR_SMALL_MIN_LONG_M <= long_side <= head_long + LIDAR_SMALL_SLACK_M
        if head_long <= LIDAR_SMALL_HEAD_M and long_side > LIDAR_MAX_LONG_M:
            # A 0.4 x 0.9 m head box on a prop by the tunnel wall: the cluster ran 6 m along the wall and the box
            # came out 6.6 x 2.8 m, 3.5 m outside the prop. No car outline is that long.
            continue
        if not (shown or small) or long_side < LIDAR_MIN_ASPECT * max(short_side, .05):
            continue
        if short_side > LIDAR_MAX_SHORT_M:
            # Two barriers end to end at an angle, the detector's centre near the joint: both lie within the seed
            # radius and form one cluster (outline 5.3-5.8 x 2.6-2.9 m); its rectangle came out 5.4-6.6 x 3.6-4.2 m,
            # 1.3 m into the bypass, and blocked the tunnel planner where the GT boxes pass. Tunnel benchmark: 1 of
            # 2051 refined car frames and 2 of 369 open-door G80 frames are this wide.
            continue
        u = np.array([math.cos(axis), math.sin(axis)])
        if float(u @ box['heading']) < 0.:
            u = -u
        v = np.array([-u[1], u[0]])
        center = np.asarray(box['center'], dtype=np.float64)
        # The head's longer side goes along the measured axis: a barricade lying across the lane comes from the head
        # as 0.6 m long x 2.2 m wide; taking its width as the new width made it 2.2 m deep instead of 0.6.
        head_long, head_short = max(box['length'], box['width']), min(box['length'], box['width'])
        length, width = max(head_long, long_side), max(head_short, short_side)
        # cover both the detector's footprint (now along the measured axis) and every object point
        along, across = (cluster - center) @ u, (cluster - center) @ v
        if small and not shown:
            # A small prop's length is its own outline: the head box sat 0.3 m off the firewall and stretched it to
            # 1.7 m (1.36 m long), into the gap the planner needed.
            lo_u, hi_u = along.min(), along.max()
        else:
            lo_u, hi_u = min(-length / 2, along.min()), max(length / 2, along.max())
        lo_v, hi_v = min(-width / 2, across.min()), max(width / 2, across.max())
        refined = _rectangle(center + u * (lo_u + hi_u) / 2 + v * (lo_v + hi_v) / 2, u, hi_u - lo_u, hi_v - lo_v)
        refined.update({k: box[k] for k in ('score', 'cell_index')}, partial=_partial(refined),
                       head_heading=box['heading'])
        box.clear()
        box.update(refined)
        changed += 1
    return changed


LIDAR_CENTER_MIN_POINTS = 5
LIDAR_COVER_MAX_M = 3.        # grow a box over its cluster only while the result stays this small (not cars, not walls)


def _cover(box, cluster):
    """The box grown along its own axes to contain the cluster, or the box itself when that exceeds LIDAR_COVER_MAX_M."""
    u = np.asarray(box['heading'], dtype=np.float64)
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    center = np.asarray(box['center'], dtype=np.float64)
    along, across = (cluster - center) @ u, (cluster - center) @ v
    lo_u, hi_u = min(-box['length'] / 2, along.min()), max(box['length'] / 2, along.max())
    lo_v, hi_v = min(-box['width'] / 2, across.min()), max(box['width'] / 2, across.max())
    if max(hi_u - lo_u, hi_v - lo_v) > LIDAR_COVER_MAX_M or (hi_u - lo_u <= box['length'] and hi_v - lo_v <= box['width']):
        return box
    return dict(box, center=center + u * (lo_u + hi_u) / 2 + v * (lo_v + hi_v) / 2, length=hi_u - lo_u, width=hi_v - lo_v)


def decode_world_boxes(prediction, x, y, yaw_grid_rad, merge=False, lidar_points=None, cluster_points=None):
    """Detection head + the pose the inputs were captured at -> (UDP Box list, counts).

    Speed is the head's scalar speed (no direction); invalid values become None.
    Values outside the Detection wire limits would make the sender reject the
    whole frame, so such a speed becomes None and such a box is dropped (counted).
    ``lidar_points`` turns on the LiDAR heading fix; ``cluster_points`` (the same body points) adds each box's LiDAR
    cluster centre. The Expert uses it to tell one object reported twice from two objects and to judge whether a
    parked object moves: the cluster centre holds still where the detector's box centre jumps by up to a metre.
    A box is also grown to cover its cluster when the result stays within LIDAR_COVER_MAX_M: a barrier lying across
    the lane came out half its length (GT footprint covered 0.37 -> 0.98 in the tunnel benchmark, all objects
    0.74 -> 0.82, area x1.2). Boxes are never shrunk; cars and wall-attached clusters exceed the limit and stay as they are.
    """
    boxes, counts = decode_boxes(prediction, merge=merge)
    if lidar_points is not None:
        counts['lidar_heading'] = refine_heading_with_lidar(boxes, lidar_points)
        if counts['lidar_heading']:
            # Cells of one car that disagreed on the heading overlapped too little for NMS; once they share the
            # measured heading they are duplicates, so suppress (and merge) again.
            kept = []
            for box in boxes:   # still in score order
                if any(oriented_iou(box, earlier) > NMS_IOU for earlier in kept):
                    counts['nms_suppressed'] += 1
                else:
                    kept.append(box)
            boxes = kept
            if merge:
                boxes, merges = merge_fragments(boxes)
                counts['fragment_merges'] = counts.get('fragment_merges', 0) + merges
            counts['decoded_boxes'] = len(boxes)
    values = prediction.detach().float().cpu().numpy() if hasattr(prediction, 'detach') else np.asarray(prediction)
    result, invalid_speeds, oversize = [], 0, 0
    for box in boxes:
        if box['length'] > MAX_BOX_SIZE_M or box['width'] > MAX_BOX_SIZE_M:
            oversize += 1
            continue
        speed = float(values[box['cell_index'], 7]) * SPEED_SCALE
        if not math.isfinite(speed) or not 0. <= speed <= MAX_SPEED_MPS:
            speed, invalid_speeds = None, invalid_speeds + 1
        lidar_center = lidar_count = None
        if cluster_points is not None and len(cluster_points):
            cluster = object_points(cluster_points, box['center'])
            if len(cluster) >= LIDAR_CENTER_MIN_POINTS:
                lidar_center = to_world(cluster.mean(0), (0., 1.), x, y, yaw_grid_rad)[0]
                lidar_count = int(len(cluster))
                # Grow over this object's points only: with every seed within 1.5 m, two firewalls side by side became
                # one 2.2 x 1.7 m box (64-74 % of the frames in the competition tunnel) whose heading spun 60 degrees.
                box = _cover(box, object_points(cluster_points, box['center'], nearest_only=True))
        (wx, wy), heading_deg = to_world(box['center'], box['heading'], x, y, yaw_grid_rad)
        result.append(Box(wx, wy, heading_deg, float(box['width']), float(box['length']), speed,
                          lidar_center, lidar_count))
    return result, dict(counts, invalid_speed_values=invalid_speeds, oversize_boxes=oversize)
