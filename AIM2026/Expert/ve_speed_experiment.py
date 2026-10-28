"""Opt-in speed-source experiment; does not change the perception tracker."""
import math


def use_ve_speed(packet):
    """Use valid raw VE speed for moving/uncertain boxes, retaining audit values.

    Confirmed stationary objects keep their existing treatment. Missing/invalid
    raw speeds fall back to the tracker, never to zero. Copy boxes so this
    experiment cannot feed modified speeds back into tracking/memory state.
    """
    boxes = []
    for original in packet['boxes']:
        box = dict(original)
        box['tracked_planning_speed_mps'] = original.get('planning_speed_mps')
        box['tracked_planning_speed_source'] = original.get('planning_speed_source')
        raw = original.get('speed_mps')
        valid = (isinstance(raw, (int, float)) and not isinstance(raw, bool)
                 and math.isfinite(raw) and raw >= 0.)
        if original.get('motion_state') in ('moving', 'uncertain') and valid:
            box['planning_speed_mps'] = float(raw)
            box['planning_speed_source'] = 've_raw_experiment'
        boxes.append(box)
    return dict(packet, boxes=boxes, object_speed_source='ve_raw_experiment')
