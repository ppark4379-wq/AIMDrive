"""Stateless pedestrian clearance for the commanded Jamming path.

The legacy ACC's wide approach gate is useful at unmapped crosswalks, but
must not hold a tunnel stop after a pedestrian has passed the vehicle. Use
the same OBB and motion model as dynamic avoidance for this explicit scope.
"""
import math

from .motion import key


def pedestrian_conflict(path, ego, person, target_speed):
    """Return the earliest current/future body overlap, or None when clear.

    Coordinates must share a frame; both world and section-local paths work.
    Check the intended moving trajectory, not a previous zero ACC command.
    A zero-speed person is still checked, unlike the moving-vehicle filter.
    Every call uses fresh GT and has no crossing-history or release timer.
    """
    # Defer planner imports until the caller has built the planning process.
    # The highway/static planner packages share initialization dependencies.
    from .recorded_planner import DynamicObstaclePlanner
    from .dynamic_planner import HORIZON
    from ..highway import parameters as hp

    numbers = (ego.position.x, ego.position.y, ego.yaw, ego.velocity,
               person.position.x, person.position.y, person.velocity,
               person.heading_deg, person.size_x, person.size_y, target_speed)
    if len(path) < 2 or not all(math.isfinite(float(value)) for value in numbers):
        return dict(time=0., object_id=key(person), kind='pedestrian_prediction_unavailable')
    if not all(math.isfinite(point.x) and math.isfinite(point.y) for point in path):
        return dict(time=0., object_id=key(person), kind='pedestrian_prediction_unavailable')
    collisions = []
    target = max(0., float(target_speed))
    # The measured speed covers a slower response than nominal acceleration;
    # the fast bound protects an early restart before a crossing completes.
    speeds = dict.fromkeys((target, min(target, max(0., float(ego.velocity)))))
    for speed in speeds:
        for acceleration in (hp.LANE_CHANGE_PREDICT_ACCEL, hp.LANE_CHANGE_PREDICT_ACCEL_MAX):
            collision = DynamicObstaclePlanner.conflict(
                path, ego, [person], speed, HORIZON, acceleration)
            if collision is not None:
                collisions.append(collision)
            if speed <= max(0., ego.velocity):
                break
    return min(collisions, key=lambda value: value['time']) if collisions else None
