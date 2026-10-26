"""Drop a VE box that lies mostly inside a larger box of the same frame, before tracking.

NMS removes overlapping boxes only above IoU 0.3. A parked car is sometimes reported twice, for example as the whole car
and as a 1.5-2 m piece of it, or as two boxes turned against each other; their IoU stays below 0.3, so both reach the
tracker, the second one starts a new 'uncertain' track on the same object and the planner sees two obstacles. In the
tunnel benchmark this was 23 % of the stationary-to-uncertain switches.

Two real objects cannot share most of their footprint, so the smaller box is dropped when at least MIN_SHARE of its area
lies inside the larger one, or when both have about the same size and nearly the same centre (thin objects). A much smaller box (a pedestrian next to a parked car) is kept: the rule only applies when
the smaller box has at least MIN_AREA_RATIO of the larger one's area.
"""
import math

import numpy as np

from AIM2026.Inference.vision.postprocess import _intersection, _rectangle

MIN_SHARE = .5
MIN_AREA_RATIO = .2
# Thin objects (barriers 0.6-0.8 m wide) are reported twice with almost the same size, a few decimetres apart and a few
# degrees turned; their overlap is then only 30-45 % of either box. Such a pair is one object when the sizes agree and
# the centres are closer than a quarter of the longer side (at least SAME_CENTRE_MIN_M).
SAME_SIZE_RATIO = .5
SAME_CENTRE_SHARE = .25
SAME_CENTRE_MIN_M = .3


def _footprint(box):
    a = math.radians(box['heading_deg'])
    return _rectangle(box['center_world'], [math.cos(a), math.sin(a)], box['length_m'], box['width_m'])


def drop_contained(boxes):
    """(kept boxes, number dropped). Larger boxes are kept; order is preserved."""
    if len(boxes) < 2:
        return list(boxes), 0
    try:
        shapes = [_footprint(b) for b in boxes]
    except (ValueError, KeyError, TypeError):
        return list(boxes), 0
    order = sorted(range(len(boxes)), key=lambda i: -shapes[i]['area'])
    dropped = set()
    for rank, i in enumerate(order):
        if i in dropped:
            continue
        for j in order[rank + 1:]:
            if j in dropped:
                continue
            small, large = shapes[j]['area'], shapes[i]['area']
            if small < MIN_AREA_RATIO * large:
                continue
            if _intersection(shapes[i], shapes[j]) >= MIN_SHARE * small:
                dropped.add(j)
                continue
            longest = max(boxes[i]['length_m'], boxes[i]['width_m'], boxes[j]['length_m'], boxes[j]['width_m'])
            if (small >= SAME_SIZE_RATIO * large and math.dist(boxes[i]['center_world'], boxes[j]['center_world'])
                    <= max(SAME_CENTRE_MIN_M, SAME_CENTRE_SHARE * longest)):
                dropped.add(j)
    return [b for k, b in enumerate(boxes) if k not in dropped], len(dropped)
