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


def object_points(points, center, max_m=LIDAR_CLUSTER_MAX_M, as_index=False, nearest_only=False):
    """The LiDAR points connected to the detector's centre (gaps < LIDAR_CLUSTER_GAP_M), within ``max_m``.

    Connectivity is computed on 0.1 m voxels: the all-pairs version took up to 0.5 s on a frame next to a building
    wall (4000 points), which stalled the inference; on the benchmark both give the same points for 1401 of 1403 boxes.
    ``as_index`` returns the indices into ``points`` instead of the points.
    """
    selected = np.flatnonzero(np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) < max_m)
    near = points[selected]
    if not len(near):
        return selected if as_index else near
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
            return selected[:0] if as_index else near[:0]
        member[index[int(np.argmin(distance))]] = True
    else:
        member[index[distance < LIDAR_SEED_RADIUS_M]] = True
    # Offline score/NMS replay needs only numpy; scipy is for LiDAR refinement.
    from scipy.spatial import cKDTree
    tree = cKDTree(centres)
    frontier = np.flatnonzero(member)
    while len(frontier):
        reached = np.unique(np.concatenate(tree.query_ball_point(centres[frontier], LIDAR_CLUSTER_GAP_M)))
        frontier = reached[~member[reached]]
        member[frontier] = True
    return selected[member[index]] if as_index else near[member[index]]


def _outline_box(box, axis, cluster, long_side, short_side, length_from_cluster=False):
    """Rectangle along ``axis`` (forward sign closest to the box's) covering the box's footprint and every cluster
    point; length and width are at least the outline's. ``length_from_cluster``: the length is the cluster's alone
    (a large vehicle's outline shows its whole side; the detector's short box sits off its centre)."""
    u = np.array([math.cos(axis), math.sin(axis)])
    if float(u @ box['heading']) < 0.:
        u = -u
    v = np.array([-u[1], u[0]])
    center = np.asarray(box['center'], dtype=np.float64)
    # The head's longer side goes along the measured axis: a barricade lying across the lane comes from the head
    # as 0.6 m long x 2.2 m wide; taking its width as the new width made it 2.2 m deep instead of 0.6.
    head_long, head_short = max(box['length'], box['width']), min(box['length'], box['width'])
    length, width = max(head_long, long_side), max(head_short, short_side)
    along, across = (cluster - center) @ u, (cluster - center) @ v
    if length_from_cluster:
        lo_u, hi_u = along.min(), along.max()
    else:
        lo_u, hi_u = min(-length / 2, along.min()), max(length / 2, along.max())
    lo_v, hi_v = min(-width / 2, across.min()), max(width / 2, across.max())
    refined = _rectangle(center + u * (lo_u + hi_u) / 2 + v * (lo_v + hi_v) / 2, u, hi_u - lo_u, hi_v - lo_v)
    refined.update({k: box[k] for k in ('score', 'cell_index')}, partial=_partial(refined), head_heading=box['heading'])
    return refined


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
        # cover both the detector's footprint (now along the measured axis) and every object point
        # A small prop's length is its own outline: the head box sat 0.3 m off the firewall and stretched it to 1.7 m.
        refined = _outline_box(box, axis, cluster, long_side, short_side, length_from_cluster=small and not shown)
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


BIG_SCORE = .2              # head candidates down to this score are kept where the LiDAR shows a large vehicle
BIG_RADIUS_M = 9.           # a large vehicle's points are searched this far from the candidate centre ...
BIG_OPEN_M = 8.5            # ... and a cluster reaching past this runs into something longer (wall, guardrail)
BIG_LONG_M = (7., 13.5)     # outline length of a bus or truck
BIG_MIN_WIDTH_M = 1.2       # both sides seen (a wall or guardrail is a thin line)
BIG_MIN_HEIGHT_M = 1.5      # 90th percentile of the points' height above the road: vehicle sides, not low barriers
BIG_MAX_CANDIDATES = 30     # rejected candidates looked at per frame, best first
BIG_TALL_M = 1.6            # before any clustering a centre needs points this high above the road ...
BIG_TALL_POINTS = 5         # ... this many within BIG_TALL_RADIUS_M (highway dumps: 93-95 % of bus/truck cells,
BIG_TALL_RADIUS_M = 3.      # 3-25 % of car cells)
BIG_REPEAT_M = 1.5          # a candidate this close to one already judged not a bus is skipped


def _inside(point, box):
    """Whether the point lies within the box's rectangle."""
    u = np.asarray(box['heading'], dtype=np.float64)
    u = u / np.linalg.norm(u)
    d = np.asarray(point, dtype=np.float64) - box['center']
    return abs(float(d @ u)) <= box['length'] / 2 and abs(float(d @ np.array([-u[1], u[0]]))) <= box['width'] / 2


def _may_be_big(points, heights, center):
    """Cheap test before the wide search: the ordinary cluster runs to its radius limit and stands vehicle-tall."""
    idx = object_points(points, center, as_index=True)
    if len(idx) < LIDAR_MIN_POINTS:
        return False
    cluster = points[idx]
    return bool(np.hypot(cluster[:, 0] - center[0], cluster[:, 1] - center[1]).max() > LIDAR_CLUSTER_MAX_M - .5
                and np.percentile(heights[idx], 90) >= BIG_MIN_HEIGHT_M)


def big_vehicle(points, heights, center):
    """(axis, long side, short side, cluster) when the LiDAR cluster at ``center`` outlines a bus or truck, else None."""
    idx = object_points(points, center, BIG_RADIUS_M, as_index=True)
    if len(idx) < LIDAR_MIN_POINTS:
        return None
    cluster = points[idx]
    if (np.hypot(cluster[:, 0] - center[0], cluster[:, 1] - center[1]).max() > BIG_OPEN_M
            or np.percentile(heights[idx], 90) < BIG_MIN_HEIGHT_M):
        return None
    axis, long_side, short_side = lshape_axis(cluster)
    if not BIG_LONG_M[0] <= long_side <= BIG_LONG_M[1] or short_side < BIG_MIN_WIDTH_M:
        return None
    return axis, long_side, short_side, cluster


def apply_big_vehicles(prediction, boxes, points, heights):
    """Large vehicles (buses, trucks) from the detector's boxes and the LiDAR; returns (boxes, grown, rescued).

    The head misses about half the frames of a 9-12 m vehicle on the highway (its cells score 0.2-0.5) and reports the
    ones it finds at ~0.7 of their length, sometimes as two pieces. A kept box on a bus-sized LiDAR cluster is grown to
    the cluster; a rejected candidate (score >= BIG_SCORE) on such a cluster that no box covers yet is added. Boxes
    that end up on the same vehicle are suppressed as duplicates.
    """
    grown = rescued = 0
    tall = points[heights >= BIG_TALL_M]
    if len(tall) < BIG_TALL_POINTS:
        return boxes, 0, 0
    from scipy.spatial import cKDTree
    tree = cKDTree(tall)

    def tall_near(centres):
        return tree.query_ball_point(np.asarray(centres, dtype=np.float64).reshape(-1, 2), BIG_TALL_RADIUS_M,
                                     return_length=True) >= BIG_TALL_POINTS

    out = []
    screen = tall_near([b['center'] for b in boxes]) if boxes else []
    for box, near in zip(boxes, screen):
        found = big_vehicle(points, heights, box['center']) if near and _may_be_big(points, heights, box['center']) else None
        if found and found[1] > box['length'] + .5:
            box = _outline_box(box, found[0], found[3], found[1], found[2], length_from_cluster=True)
            grown += 1
        out.append(box)
    # Rejected cells are screened on the raw head before any box is built: the full decode with NMS over every
    # candidate >= BIG_SCORE stalled frames on the dense highway.
    values = prediction.detach().float().cpu().numpy() if hasattr(prediction, 'detach') else np.asarray(prediction)
    logits = values[:, 0]
    cells = np.flatnonzero(np.isfinite(logits) & (logits >= math.log(BIG_SCORE / (1. - BIG_SCORE)))
                           & (logits < math.log(CONFIDENCE / (1. - CONFIDENCE))))
    if len(cells):
        rows, cols = np.divmod(cells, 50)
        centres = np.stack([cols - 24.5 + values[cells, 1] * 3.5, 29.5 - rows + values[cells, 2] * 3.5], 1)
        cells = cells[tall_near(centres)]
        cells = cells[np.argsort(-logits[cells], kind='stable')][:BIG_MAX_CANDIDATES]
    if len(cells):
        masked = values.copy()
        masked[:, 0] = -50.
        masked[cells, 0] = values[cells, 0]
        candidates, _ = decode_boxes(masked, score_threshold=BIG_SCORE, nms_iou=None)
        failed = []         # neighbouring cells of one object: judge its cluster once
        for candidate in candidates:
            c = candidate['center']
            if (any(_inside(c, b) for b in out)
                    or any(abs(c[0] - f[0]) < BIG_REPEAT_M and abs(c[1] - f[1]) < BIG_REPEAT_M for f in failed)):
                continue
            found = big_vehicle(points, heights, c) if _may_be_big(points, heights, c) else None
            if not found:
                failed.append(c)
                continue
            out.append(_outline_box(candidate, found[0], found[3], found[1], found[2], length_from_cluster=True))
            rescued += 1
    if not grown and not rescued:
        return boxes, 0, 0
    kept = []
    for box in out:
        if not any(oriented_iou(box, earlier) > NMS_IOU for earlier in kept):
            kept.append(box)
    return kept, grown, rescued


def decode_world_boxes(prediction, x, y, yaw_grid_rad, merge=False, lidar_points=None, cluster_points=None,
                       large_vehicle_tracker=None, body_points_3d=None, observation_time=None, lidar_tracker=None,
                       big_points=None):
    """Detection head + the pose the inputs were captured at -> (UDP Box list, counts).

    Speed is the head's scalar speed, or measured world motion for LiDAR tracks.
    Invalid values become None.
    Values outside the Detection wire limits would make the sender reject the
    whole frame, so such a speed becomes None and such a box is dropped (counted).
    ``lidar_points`` turns on the LiDAR heading fix; ``cluster_points`` (the same body points) adds each box's LiDAR
    cluster centre. The Expert uses it to tell one object reported twice from two objects and to judge whether a
    parked object moves: the cluster centre holds still where the detector's box centre jumps by up to a metre.
    A box is also grown to cover its cluster when the result stays within LIDAR_COVER_MAX_M: a barrier lying across
    the lane came out half its length (GT footprint covered 0.37 -> 0.98 in the tunnel benchmark, all objects
    0.74 -> 0.82, area x1.2). Boxes are never shrunk; cars and wall-attached clusters exceed the limit and stay as they are.
    ``big_points`` = (points, heights above the road) turns on apply_big_vehicles.
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
    if big_points is not None and len(big_points[0]):
        boxes, counts['big_grown'], counts['big_rescued'] = apply_big_vehicles(prediction, boxes, *big_points)
        counts['decoded_boxes'] = len(boxes)
    values = prediction.detach().float().cpu().numpy() if hasattr(prediction, 'detach') else np.asarray(prediction)
    tracker = lidar_tracker if lidar_tracker is not None else large_vehicle_tracker
    if tracker is not None:
        if observation_time is None:
            raise ValueError('LiDAR tracking requires an observation time')
        boxes, tracking = tracker.apply(
            boxes, body_points_3d, (x, y, yaw_grid_rad), observation_time)
        counts.update(tracking)
    result, invalid_speeds, oversize = [], 0, 0
    for box in boxes:
        if box['length'] > MAX_BOX_SIZE_M or box['width'] > MAX_BOX_SIZE_M:
            oversize += 1
            continue
        speed = (box['tracked_speed'] if 'tracked_speed' in box
                 else float(values[box['cell_index'], 7]) * SPEED_SCALE)
        if speed is not None and (not math.isfinite(speed) or not 0. <= speed <= MAX_SPEED_MPS):
            speed, invalid_speeds = None, invalid_speeds + 1
        lidar_center = box.get('tracked_lidar_center')
        lidar_count = box.get('tracked_lidar_count')
        if lidar_center is None and cluster_points is not None and len(cluster_points):
            cluster = object_points(cluster_points, box['center'])
            if len(cluster) >= LIDAR_CENTER_MIN_POINTS:
                lidar_center = to_world(cluster.mean(0), (0., 1.), x, y, yaw_grid_rad)[0]
                lidar_count = int(len(cluster))
                # Grow over this object's points only: with every seed within 1.5 m, two firewalls side by side became
                # one 2.2 x 1.7 m box (64-74 % of the frames in the competition tunnel) whose heading spun 60 degrees.
                box = _cover(box, object_points(cluster_points, box['center'], nearest_only=True))
        (wx, wy), heading_deg = to_world(box['center'], box['heading'], x, y, yaw_grid_rad)
        result.append(Box(wx, wy, heading_deg, float(box['width']), float(box['length']), speed,
                          lidar_center, lidar_count, lidar_track_id=box.get('track_id'),
                          observed_only=box.get('observed_only', False),
                          observation_time=box.get('observation_time'), lidar_hits=box.get('lidar_hits')))
    return result, dict(counts, invalid_speed_values=invalid_speeds, oversize_boxes=oversize)
