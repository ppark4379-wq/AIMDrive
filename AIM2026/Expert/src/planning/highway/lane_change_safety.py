"""Swept stationary geometry and time-aligned checks for highway lane changes."""

import math
from bisect import bisect_left
from copy import copy

from . import parameters as p


class LaneChangeSafety:
    def __init__(self, planner):
        self.planner = planner
        self.predictions = {}

    @staticmethod
    def first_merge_static_escape(candidate):
        return (getattr(candidate, 'static_escape', False)
                and getattr(candidate, 'source_link_id', None) == 'A2256W000411'
                and getattr(candidate, 'target_link_id', None) == 'A2256W000409')

    @staticmethod
    def travel(speed, target, time_value, acceleration=None):
        delay = min(p.LANE_CHANGE_CONTROL_DELAY, time_value)
        remaining = time_value - delay
        acceleration = p.LANE_CHANGE_PREDICT_ACCEL if acceleration is None else acceleration
        accel = acceleration if target >= speed else -p.LANE_CHANGE_PREDICT_DECEL
        ramp = min(remaining, abs((target - speed) / accel))
        return speed * delay + speed * ramp + 0.5 * accel * ramp * ramp + target * (remaining - ramp)

    def static_distance(self, points, entries, include_uncertain=False, margin=0.0):
        """Distance to first body overlap, using measured poses, including off-link objects."""
        objects = [entry['object_info'] for entry in entries
                   if (self.planner.obstacle_motion.is_static(entry['object_info'])
                       or (include_uncertain and getattr(entry['object_info'], 'speed_unknown', False)))]
        if margin > 0.0:
            # Geometry-only copies: never rewrite a perception object's speed
            # or motion state to test its possible stationary occupancy.
            objects = [copy(obj) for obj in objects]
            for obj in objects:
                obj.size_x += 2*margin
                obj.size_y += 2*margin
        if not objects or len(points) < 2:
            return float('inf')
        distance = 0.0
        for start, end in zip(points[:-1], points[1:]):
            yaw = math.atan2(end.y - start.y, end.x - start.x)
            length = math.hypot(end.x-start.x, end.y-start.y)
            for obj in objects:
                radius = p.VEHICLE_LENGTH + math.hypot(obj.size_x, obj.size_y) + 2.0
                if math.hypot(start.x-obj.position.x, start.y-obj.position.y) > radius:
                    continue
                if self.planner._obb_collision(start, yaw, obj.position, math.radians(obj.heading_deg), obj):
                    return distance
            distance += length
        return float('inf')

    def check(self, candidate, ego, entries, matches, target_speed, require_completion=True,
              committed=False):
        """Check nominal and faster acceleration; retain rear and front hazards."""
        if not committed and self.planner.search_timed_out():
            return dict(time=0.0, rear=False, kind='search_timeout')
        if len(self.predictions) > 20000:
            self.predictions.clear()
        predictions = self.predictions
        for acceleration in (p.LANE_CHANGE_PREDICT_ACCEL, p.LANE_CHANGE_PREDICT_ACCEL_MAX):
            conflict = self._check(candidate, ego, entries, matches, target_speed,
                                   require_completion, committed, acceleration, predictions)
            if conflict is not None:
                return conflict
            if target_speed <= max(0.0, ego.velocity):
                break
        return None

    def _check(self, candidate, ego, entries, matches, target_speed,
               require_completion, committed, acceleration, predictions):
        planner = self.planner
        points = candidate.points
        if len(points) < 2:
            return dict(time=0.0, rear=False, kind='path_end')
        compact = getattr(candidate, 'compact_static', False)
        speed = max(0.0, ego.velocity)
        if compact and speed > p.STATIC_CHANGE_MAX_SPEED:
            # The controller permits larger steering only after actually
            # slowing down, regardless of how quickly a prediction decelerates.
            return dict(time=0.0, rear=False, kind='turn_speed', speed=speed)
        cumulative = planner._cumulative_distance(points)
        compact = getattr(candidate, 'compact_static', False)
        curvatures = ([0.0] + [planner._max_curvature(points[i-1:i+2])
                              for i in range(1, len(points)-1)] + [0.0]) if compact else []
        endpoint = getattr(candidate, 'transition_end', points[-1])
        completion_index = min(range(len(points)), key=lambda i: (points[i].x-endpoint.x)**2 + (points[i].y-endpoint.y)**2)
        completion_distance = cumulative[completion_index]
        speed = max(0.0, ego.velocity)
        target_speed = max(0.0, target_speed)
        rear_time_gap = (p.TARGET_LANE_COMMITTED_REAR_TIME_GAP if committed
                         else p.TARGET_LANE_REAR_TIME_GAP)
        if self.first_merge_static_escape(candidate):
            rear_time_gap = min(rear_time_gap, p.FIRST_MERGE_STATIC_REAR_TIME_GAP)
        match_by_id = {id(entry['object_info']): match for entry, match in matches}
        initial_rear = {id(entry['object_info']) for entry in entries
                        if ((entry['object_info'].position.x - ego.position.x) * math.cos(ego.yaw)
                            + (entry['object_info'].position.y - ego.position.y) * math.sin(ego.yaw)) < 0.0}
        entry_merge = getattr(candidate, 'entry_merge', False)
        entry_gap_policy = (entry_merge
                            and getattr(candidate, 'source_link_id', None) == p.ENTRY_MERGE_SOURCE
                            and getattr(candidate, 'target_link_id', None) == p.ENTRY_MERGE_TARGET)
        stationary_distance = self.static_distance(points, entries,
            include_uncertain=entry_merge,
            margin=p.ENTRY_MERGE_OBSTACLE_MARGIN if entry_merge and not committed else 0.0)
        # Use the exact sampled path for stationary geometry even when time
        # samples could skip a small obstacle at highway speed.
        if require_completion and stationary_distance <= completion_distance + max(speed, target_speed) * p.LANE_CHANGE_SETTLE_TIME:
            return dict(time=stationary_distance / max(speed, target_speed, 0.1),
                        rear=False, kind='static_path', distance=stationary_distance)
        completion_time = None
        minimum_prediction_time = getattr(candidate, 'minimum_prediction_time', 0.0)
        if committed and getattr(candidate, 'entry_merge', False):
            # The full maneuver was checked at launch. Recheck its REMAINING
            # distance plus settling, not another rolling five seconds beyond
            # completion on every tick. Near-term collision checks still run.
            minimum_prediction_time = min(minimum_prediction_time, p.ENTRY_MERGE_COMMITTED_HORIZON)
        t = 0.0
        # At normal road speeds this is about 3--6 s. From rest use finite
        # acceleration to the requested speed, rather than teleporting ahead.
        max_time = max(12.0, minimum_prediction_time, completion_distance / max(target_speed, 1.0) + 4.0)
        while t <= max_time + 1e-9:
            if not committed and planner.search_timed_out():
                return dict(time=0.0, rear=False, kind='search_timeout')
            travelled = self.travel(speed, target_speed, t, acceleration)
            if travelled >= stationary_distance:
                return dict(time=t, rear=False, kind='static_path', distance=stationary_distance)
            if travelled > cumulative[-1] + 0.5:
                return dict(time=t, rear=False, kind='path_end')
            point, yaw = planner._pose_at_distance(points, cumulative, travelled)
            elapsed = max(0.0, t - p.LANE_CHANGE_CONTROL_DELAY)
            speed_at_t = (min(target_speed, speed + acceleration * elapsed)
                          if target_speed >= speed else max(target_speed, speed - p.LANE_CHANGE_PREDICT_DECEL * elapsed))
            if compact:
                index = min(len(points)-1, bisect_left(cumulative, travelled))
                curvature = max(curvatures[max(0, index-1):index+2])
                # A speed ceiling cannot instantly make a short curve drivable.
                # Match the controller's high-speed steering cap until slowed.
                high_speed_turn = (speed_at_t > p.STATIC_CHANGE_MAX_SPEED
                                   and curvature > math.tan(math.radians(p.LANE_CHANGE_STEERING_LIMIT_DEG)) / 3.0)
                if high_speed_turn or speed_at_t**2 * curvature > p.STATIC_CHANGE_LATERAL_ACCEL * 1.05:
                    return dict(time=t, rear=False, kind='turn_speed', speed=speed_at_t)
            if travelled <= 1e-9:
                point, yaw = ego.position, ego.yaw
            if travelled >= completion_distance and completion_time is None:
                completion_time = t
            for entry in entries:
                obj = entry['object_info']
                match = match_by_id.get(id(obj))
                key = (id(obj), obj.position.x, obj.position.y, obj.velocity, obj.heading_deg,
                       (match.lane_link.id, match.s, match.x, match.y, match.yaw) if match else None,
                       entry.get('prediction_age', 0.0), round(t, 6))
                if key not in predictions:
                    predictions[key] = planner.merge_safety._pose(obj, match, t + entry.get('prediction_age', 0.0))
                for center, angle in predictions[key]:
                    dx, dy = center.x - point.x, center.y - point.y
                    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
                    lateral = abs(-dx * math.sin(yaw) + dy * math.cos(yaw))
                    if planner._obb_collision(point, yaw, center, angle, obj):
                        return dict(time=t, rear=bool(longitudinal < 0.0 or id(obj) in initial_rear), kind='collision',
                                    object_id=getattr(obj, 'unique_id', None))
                    # A small future bumper margin, in addition to the larger
                    # initial target-lane clearance. Check only overlapping lanes.
                    source_lane = (match is not None
                                   and match.lane_link.id == getattr(candidate, 'source_link_id', None))
                    source_occupancy_only = ((getattr(candidate, 'entry_merge', False) and source_lane)
                        or (planner.obstacle_motion.is_static(obj) and (match is None or source_lane)))
                    # While leaving the source lane, check every moving AND
                    # stationary hypothesis against the swept body above. A
                    # longitudinal following gap on that abandoned lane would
                    # brake a safe pass whenever a provisional speed fluctuates.
                    # Target-lane and unmatched moving vehicles retain the gap.
                    # Gap checks use the other vehicle's lane direction. Ego's
                    # turning heading can point its rear axis across an entirely
                    # different lane and falsely block a change away from it.
                    gap_longitudinal = dx * math.cos(angle) + dy * math.sin(angle)
                    gap_lateral = abs(-dx * math.sin(angle) + dy * math.cos(angle))
                    if (not source_occupancy_only and math.cos(angle - yaw) > 0.8
                            and gap_lateral < 0.5 * (p.VEHICLE_WIDTH + max(1.0, obj.size_y))):
                        gap = (gap_longitudinal - p.VEHICLE_LENGTH - 0.5 * max(1.0, obj.size_x)
                               if gap_longitudinal >= 0 else -gap_longitudinal - 0.5 * max(1.0, obj.size_x))
                        # Only 411 -> 409 uses entry margins, before AND during
                        # the maneuver. Other changes retain the common margin.
                        required_gap = ((p.ENTRY_MERGE_FRONT_GAP if gap_longitudinal >= 0
                                         else p.ENTRY_MERGE_REAR_GAP)
                                        if entry_gap_policy else p.LANE_CHANGE_BUMPER_MARGIN)
                        if gap_longitudinal < 0:
                            required_gap += max(0.0, obj.velocity - speed_at_t) * rear_time_gap
                        if gap < required_gap:
                            return dict(time=t, rear=bool(longitudinal < 0.0 or id(obj) in initial_rear), kind='gap',
                                        object_id=getattr(obj, 'unique_id', None),
                                        gap=gap, required_gap=required_gap,
                                        gap_policy='entry_merge' if entry_gap_policy else 'lane_change')
            if (completion_time is not None
                    and t >= max(minimum_prediction_time, completion_time + p.LANE_CHANGE_SETTLE_TIME)):
                return None
            if (not require_completion and target_speed < 0.1 and completion_time is None
                    and travelled >= self.travel(speed, target_speed, max_time, acceleration) - 1e-6):
                completion_time = t
            t += p.LANE_CHANGE_CHECK_STEP
        return dict(time=max_time, rear=False, kind='incomplete_change')
