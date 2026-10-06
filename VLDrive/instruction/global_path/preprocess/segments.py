"""Stable signed bend detection; these are geometric bends, not road semantics."""
import math

import numpy as np


def detect_bends(arc_m, curvature, *, closed, enter_curvature=.015, exit_curvature=.0075,
                 merge_gap_m=5., min_length_m=5., min_angle_deg=15.):
    """Seed above enter threshold, extend to exit threshold, merge short gaps.

    A significant opposite-sign seed always separates bends, even if that seed
    later fails length/angle filtering. Closed-seam events use end_m > length_m.
    """
    arc, k = np.asarray(arc_m, dtype=float), np.asarray(curvature, dtype=float)
    parameters = (enter_curvature, exit_curvature, merge_gap_m, min_length_m, min_angle_deg)
    if (not all(math.isfinite(v) and v > 0 for v in parameters)
            or exit_curvature > enter_curvature or min_angle_deg > 360):
        raise ValueError('Invalid bend segmentation thresholds')
    if (arc.ndim != 1 or k.shape != (len(arc) - 1,) or not len(k)
            or not np.isfinite(arc).all() or not np.isfinite(k).all()
            or arc[0] != 0 or np.any(np.diff(arc) <= 0)):
        raise ValueError('Invalid arc/curvature arrays')
    n, length = len(k), float(arc[-1])
    sign = np.where(abs(k) >= exit_curvature, np.sign(k), 0).astype(int)
    spans, start = [], 0
    for end in range(1, n + 1):
        if end == n or sign[end] != sign[start]:
            if sign[start]:
                spans.append([start, end, int(sign[start])])
            start = end

    def distance(index):
        lap, index = divmod(index, n)
        return lap * length + float(arc[index])

    def opposing_seed(start, end, direction):
        return np.any(direction * k[np.arange(start, end) % n] <= -enter_curvature)

    merged = []
    for start, end, direction in spans:
        if (merged and merged[-1][2] == direction
                and distance(start) - distance(merged[-1][1]) <= merge_gap_m
                and not opposing_seed(merged[-1][1], start, direction)):
            merged[-1][1] = end
        else:
            merged.append([start, end, direction])
    if closed and len(merged) > 1:
        first, last = merged[0], merged[-1]
        if (first[2] == last[2] and distance(first[0] + n) - distance(last[1]) <= merge_gap_m
                and not opposing_seed(last[1], first[0] + n, first[2])):
            merged = merged[1:-1] + [[last[0], first[1] + n, first[2]]]

    events = []
    for start, end, direction in merged:
        indices = np.arange(start, end) % n
        values = k[indices]
        peak = float(values[np.argmax(abs(values))])
        angle_deg = float(np.degrees(np.sum(values * np.diff(arc)[indices])))
        if (direction * peak < enter_curvature or distance(end) - distance(start) < min_length_m
                or direction * angle_deg < min_angle_deg):
            continue
        events.append(dict(id='bend_%04d' % len(events), direction='left' if direction > 0 else 'right',
            start_m=distance(start), end_m=distance(end), curv_event=peak, angle_deg=angle_deg))
    return events
