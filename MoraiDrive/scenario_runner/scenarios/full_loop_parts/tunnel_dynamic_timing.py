"""Pure forecasts for scheduling tunnel pedestrians without changing a driver.

The Ego forecast retains its measured speed and extrapolates only positive
measured acceleration. It does not predict braking, steering, signal waits, or
an obstacle response. An optional speed bound limits acceleration in this
forecast; it never changes Expert configuration or lowers measured speed.

NPC forecasts follow the declared polyline at the configured constant speed.
They approximate simulator traversal and polling, including an ABC turn, and
do not establish physical contact or guarantee an encounter in a live run.
"""

import math

from ...utils.transform_utils import (
    interpolate_on_polyline,
    project_distance_on_polyline,
)


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def travel_time(distance_m, speed_mps, acceleration_mps2=0.0, max_speed_mps=None):
    """Forecast signed arrival time in seconds, or ``math.inf`` if unreachable.

    Negative distances use distance/current speed as a past-time approximation;
    acceleration is never extrapolated backward. With no forward speed their
    arrival is already satisfied, so they return zero. Negative acceleration is
    treated as zero rather than forecasting a stop. A supplied speed bound
    below measured speed preserves that speed and prevents further acceleration.
    """
    distance = _finite(distance_m, "distance_m")
    speed = max(0.0, _finite(speed_mps, "speed_mps"))
    acceleration = max(0.0, _finite(acceleration_mps2, "acceleration_mps2"))
    bound = None
    if max_speed_mps is not None:
        bound = _finite(max_speed_mps, "max_speed_mps")
        if bound < 0.0:
            raise ValueError("max_speed_mps must be nonnegative")
        bound = max(speed, bound)
    if distance <= 0.0:
        return distance / speed if speed > 0.0 else 0.0
    if acceleration == 0.0 or (bound is not None and bound <= speed):
        return distance / speed if speed > 0.0 else math.inf

    def accelerating_time(length):
        # This form avoids cancellation when speed is high and length is short.
        return 2.0 * length / (
            speed + math.sqrt(speed * speed + 2.0 * acceleration * length))

    if bound is None:
        return accelerating_time(distance)
    time_to_bound = (bound - speed) / acceleration
    distance_to_bound = 0.5 * (speed + bound) * time_to_bound
    if distance <= distance_to_bound:
        return accelerating_time(distance)
    return time_to_bound + (distance - distance_to_bound) / bound


def motion_samples(route_points, plan):
    """Return NPC time, route progress, and signed lateral offset samples.

    Samples are at most 0.25 m apart along the declared waypoint polyline,
    beginning after 0.5 m of observed-equivalent movement. The final sample is
    at the final endpoint's arrival-tolerance boundary, rather than its center.
    Intermediate waypoints remain the declared turn locations; the forecast
    does not model frame delay or the simulator turning within their tolerance.
    Route progress uses the supplied route's coordinate frame, normally the
    clipped mission route. Positive lateral offset is left of that route.
    """
    if len(route_points) < 2:
        raise ValueError("route_points must contain at least two points")
    speed = _finite(plan["speed_mps"], "plan.speed_mps")
    if speed <= 0.0:
        raise ValueError("plan.speed_mps must be positive")
    tolerance = _finite(plan.get("arrival_tolerance_m", 0.0),
                        "plan.arrival_tolerance_m")
    if tolerance < 0.0:
        raise ValueError("plan.arrival_tolerance_m must be nonnegative")
    points = [(_finite(point["x"], "waypoint.x"),
               _finite(point["y"], "waypoint.y"))
              for point in plan["waypoints"]]
    if len(points) < 2:
        return []
    segments = []
    total = 0.0
    for start, end in zip(points, points[1:]):
        length = math.dist(start, end)
        if length > 0.0:
            segments.append((total, total + length, start, end))
            total += length
    endpoint_distance = max(0.0, total - tolerance)
    if endpoint_distance < 0.5 or not segments:
        return []

    distances = [0.5 + index * 0.25
                 for index in range(int(math.floor((endpoint_distance - 0.5) / 0.25)) + 1)]
    if endpoint_distance - distances[-1] > 1e-9:
        distances.append(endpoint_distance)
    samples = []
    segment_index = 0
    for distance in distances:
        while (segment_index < len(segments) - 1
               and distance > segments[segment_index][1]):
            segment_index += 1
        lo, hi, start, end = segments[segment_index]
        fraction = min(1.0, max(0.0, (distance - lo) / (hi - lo)))
        x = start[0] + fraction * (end[0] - start[0])
        y = start[1] + fraction * (end[1] - start[1])
        # A returning path can travel 0.5 m while remaining near its start.
        if math.hypot(x - points[0][0], y - points[0][1]) < 0.5 - 1e-9:
            continue
        route_s = project_distance_on_polyline(route_points, x, y)
        rx, ry, _, yaw_deg = interpolate_on_polyline(route_points, route_s)
        angle = math.radians(yaw_deg)
        samples.append(dict(
            time_sec=distance / speed,
            route_s=route_s,
            lateral_m=-(x - rx) * math.sin(angle) + (y - ry) * math.cos(angle),
        ))
    return samples


def encounter_start_window(samples, current_s, speed_mps,
                           acceleration_mps2=0.0, max_speed_mps=None, margin_m=3.0,
                           *, front_margin_m=None):
    """Return the signed envelope of feasible NPC spawn delays in seconds.

    For each NPC sample, Ego's arrival times at NPC route progress minus its
    front margin and plus its rear margin are shifted by the NPC sample time.
    ``front_margin_m`` can represent the distance from Ego's reported position
    to its front body extent. It defaults to ``margin_m`` and changes only the
    earliest arrival; the rear/latest bound and behind-Ego exclusion retain
    ``margin_m``. It does not add acceleration or alter the supplied forecast.
    Negative bounds are retained: a
    negative latest delay means that opportunity is already past. Samples
    wholly behind Ego are skipped, avoiding acceleration extrapolation into
    the past. No reachable samples return ``(math.inf, -math.inf)``.

    This is a longitudinal opportunity envelope, not a collision or lateral
    overlap test. Its enclosing interval can contain gaps between samples or
    separate path portions. Callers needing an exact union must retain those
    portions separately; an envelope alone cannot certify a chosen delay.
    """
    current = _finite(current_s, "current_s")
    margin = _finite(margin_m, "margin_m")
    if margin < 0.0:
        raise ValueError("margin_m must be nonnegative")
    front_margin = margin if front_margin_m is None else _finite(front_margin_m, "front_margin_m")
    if front_margin < 0.0:
        raise ValueError("front_margin_m must be nonnegative")
    earliest, latest = math.inf, -math.inf
    for sample in samples:
        npc_time = _finite(sample["time_sec"], "sample.time_sec")
        npc_s = _finite(sample["route_s"], "sample.route_s")
        if npc_s + margin < current:
            continue
        enter = travel_time(npc_s - front_margin - current, speed_mps,
                            acceleration_mps2, max_speed_mps)
        leave = travel_time(npc_s + margin - current, speed_mps,
                            acceleration_mps2, max_speed_mps)
        if not math.isfinite(enter):
            continue
        earliest = min(earliest, enter - npc_time)
        latest = max(latest, leave - npc_time)
    return earliest, latest


def goal_progress_s(route_length_m, cfg):
    """Approximate the existing normal-corridor mission goal progress gate.

    This matches the progress-ratio and route-end/goal-distance OR thresholds
    on a normal corridor. Curved paths that approach their goal spatially
    before their route end, cross-track failures, or later driver stops are
    not predicted. The route length excludes any Expert-only steering tail.
    """
    length = _finite(route_length_m, "route_length_m")
    if length < 0.0:
        raise ValueError("route_length_m must be nonnegative")
    goal_tolerance = _finite(cfg.get("goal_tolerance_m", 4.0), "goal_tolerance_m")
    arrival = _finite(cfg.get("expert_arrival_stop_distance_m",
                              max(goal_tolerance, 6.0)),
                      "expert_arrival_stop_distance_m")
    ratio = _finite(cfg.get("minimum_completion_progress_ratio", 0.0),
                    "minimum_completion_progress_ratio")
    return max(ratio * length, length - max(goal_tolerance, arrival))
