"""Read-only planner snapshot for the original Expert RViz renderer (no ROS here)."""
import json
import math
from types import SimpleNamespace


def encode(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'__float__': str(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    # LaneMatch also has x/y: retain its lane before the generic point case.
    # Otherwise object_lane_matches loses lane_link and the RViz labels crash.
    if hasattr(value, 'lane_link'):
        return {'__object__': {'lane_link': encode(value.lane_link)}}
    if hasattr(value, 'x') and hasattr(value, 'y'):
        return {'__object__': {'x': float(value.x), 'y': float(value.y),
                               'z': float(getattr(value, 'z', 0.))}}
    if hasattr(value, 'tolist'):
        return encode(value.tolist())
    if all(hasattr(value, k) for k in ('section_index','lane_number','points','excluded')):
        fields = {k:getattr(value,k) for k in ('id','points','section_index','lane_number','excluded')}
    else:
        fields = vars(value)
    return {'__object__': {k:encode(v) for k,v in fields.items() if not k.startswith('_')}}


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if '__float__' in value:
            return float(value['__float__'])
        if '__object__' in value:
            return SimpleNamespace(**{k:decode(v) for k,v in value['__object__'].items()})
        return {k:decode(v) for k,v in value.items()}
    return value


def write_snapshot(target, result, vehicle, objects, selected_path, signal=None):
    if result is None:
        return
    document = encode(dict(result=result, vehicle=vehicle, objects=objects,
                           selected_path=selected_path, signal=signal))
    # JSON cannot preserve shared list identity; preserve the renderer's selected color explicitly.
    for row, candidate in zip(document['result']['__object__']['candidates'], result.candidates):
        # The corridor planner copies its chosen point list. Equal geometry is
        # still the selected path; this only affects the RViz display color.
        same_geometry = (bool(candidate.points) and len(candidate.points) == len(result.path)
                         and all(float(a.x) == float(b.x) and float(a.y) == float(b.y)
                                 for a, b in zip(candidate.points, result.path)))
        row['__object__']['selected_preview'] = (candidate.points is result.path
                                                or getattr(candidate,'selected_preview',False)
                                                or (candidate.valid and same_geometry))
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, allow_nan=False),encoding='utf-8')
    temporary.replace(target)
