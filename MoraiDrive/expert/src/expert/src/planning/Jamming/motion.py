"""Jamming-local speed classification; leave the transport and highway intact."""
import math
from ...obstacle.object_info import is_ve_tracked_unknown

# GT float noise around standstill, in m/s (0.18 km/h).
STATIONARY_SPEED = .05


def stationary(obj):
    if is_ve_tracked_unknown(obj):
        return obj.is_static and obj.name.endswith(':stationary')
    speed = float(obj.velocity)
    return (getattr(obj, 'type', -1) in (1, 2)
            and math.isfinite(speed) and abs(speed) <= STATIONARY_SPEED)


def moving(obj):
    if is_ve_tracked_unknown(obj):
        return not stationary(obj)
    speed = float(obj.velocity)
    return (getattr(obj, 'type', -1) in (0, 1, 2)
            and math.isfinite(speed) and abs(speed) > STATIONARY_SPEED)


def key(obj):
    uid = getattr(obj, 'unique_id', None)
    if uid is not None:
        return ('id', uid)
    name = getattr(obj, 'name', '')
    return ('name', name) if name else ('pose', round(obj.position.x, 1), round(obj.position.y, 1))
