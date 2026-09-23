"""Two legal adjacent transitions, checked as one continuous maneuver."""

import copy
import math

from ...vehicle_state import VehicleState
from . import parameters as p


class MultiLanePlanning:
    def __init__(self, planner):
        self.planner = planner

    def candidates(self, ego, source, via, direction, entries, matches, static):
        planner = self.planner
        lane_map = planner.lane_map
        final = dict(lane_map.legal_neighbors(via)).get(direction)
        if final is None or final.id == source.id:
            return []
        candidates = []
        distances = (p.LANE_CHANGE_DISTANCE,) + (p.STATIC_CHANGE_DISTANCES if static else ())
        for distance in distances:
            first = planner._build_candidate(ego, via, lane_map.project(ego.position, via),
                                              direction, source, distance,
                                              prediction_horizon=p.MERGE_SAFETY_HORIZON)
            if not first.valid:
                continue
            point, yaw, via_at_end = lane_map.advance_default(lane_map.project(ego.position, via), distance)
            final_at_end = dict(lane_map.legal_neighbors(via_at_end)).get(direction)
            if final_at_end is None:
                continue
            final_origin = lane_map.project(ego.position, final)
            final_projection = lane_map.project(point, final_at_end)
            # A different road or a lane that disappears at the junction is
            # not a continuation of the initially selected final lane.
            if lane_map.forward_distance(final_origin, final_projection) is None:
                continue
            future_ego = VehicleState(point.x, point.y, yaw, ego.velocity)
            second = planner._build_candidate(future_ego, final_at_end, final_projection,
                                               direction, via_at_end, distance,
                                               prediction_horizon=p.MERGE_SAFETY_HORIZON)
            if not second.valid:
                continue
            end_index = min(range(len(first.points)), key=lambda i: first.points[i].distance(first.transition_end))
            candidate = copy.copy(second)
            candidate.points = first.points[:end_index + 1] + second.points[1:]
            candidate.source_link_id = source.id
            candidate.target_link_id = final.id
            candidate.target_lane = final.lane_number
            candidate.via_candidate = first
            candidate.via_link_id = via.id
            candidate.lane_change_count = 2
            candidate.offset = (source.lane_number - final.lane_number) * p.LANE_WIDTH
            candidate.transition_distance = distance * 2.0
            candidate.static_escape = static
            candidate.compact_static = static and distance < p.LANE_CHANGE_DISTANCE
            candidate.curvature_cost = planner._max_curvature(candidate.points)
            candidate.cost = 4.0 * candidate.curvature_cost + 0.5 + (
                2.0 if not lane_map.successors.get(final_at_end.id, ()) else 0.0)
            candidate.speed_limit = (planner._static_change_speed_limit(candidate.curvature_cost)
                                     if candidate.compact_static else float('inf'))
            if candidate.compact_static and candidate.curvature_cost > p.STATIC_CHANGE_MAX_CURVATURE:
                continue
            planner._evaluate_lane_change(candidate, ego, entries, matches)
            # Keep both initial front/rear gates, in addition to the full,
            # time-aligned body/gap checks through the second completion.
            for lane in (via, final):
                if not planner._target_lane_has_clearance(lane, candidate, matches,
                        min(ego.velocity, planner._requested_speed, candidate.speed_limit)):
                    candidate.valid = False
                    candidate.cost = float('inf')
            candidates.append(candidate)
            if candidate.valid:
                break
        return candidates

    def hold_intermediate(self, ego, candidate, entries, matches):
        """Drop the second leg if new traffic blocks it and the first is safe."""
        if getattr(candidate, 'lane_change_count', 1) != 2:
            return None
        planner = self.planner
        via = planner.lane_map.links[candidate.via_link_id]
        projection = planner.lane_map.project(ego.position, via)
        # Track the intermediate lane through link boundaries too.
        for _ in range(6):
            successor = planner.lane_map.default_successor(via.id)
            if projection.s < via.length - 0.1 or successor is None:
                break
            following = planner.lane_map.project(ego.position, planner.lane_map.links[successor])
            if following.distance >= projection.distance:
                break
            via, projection = following.lane_link, following
        lateral = ((ego.position.x - projection.x) * -math.sin(projection.yaw)
                   + (ego.position.y - projection.y) * math.cos(projection.yaw))
        toward_final = lateral if candidate.direction == 'left' else -lateral
        if toward_final > p.LANE_CHANGE_ABORT_MAX_LATERAL:
            return None
        # Start at the measured pose: replaying the old first leg would snap
        # sideways once ego has already begun the second transition.
        fallback = planner._build_candidate(ego, via, projection, candidate.direction,
            planner.lane_map.links[candidate.source_link_id],
            candidate.via_candidate.transition_distance,
            prediction_horizon=p.MERGE_SAFETY_HORIZON)
        fallback.static_escape = candidate.static_escape
        fallback.compact_static = candidate.compact_static
        fallback.speed_limit = (planner._static_change_speed_limit(planner._max_curvature(fallback.points))
                               if fallback.compact_static else float('inf'))
        if fallback.compact_static and fallback.curvature_cost > p.STATIC_CHANGE_MAX_CURVATURE:
            return None
        planner._evaluate_lane_change(fallback, ego, entries, matches)
        if not fallback.valid:
            return None
        planner.target_link_id = via.id
        planner.active_candidate = fallback
        planner._last_lane_checks.append('two-lane second leg blocked; keep intermediate={}'.format(via.id))
        return fallback

    @staticmethod
    def describe(candidate, details):
        if getattr(candidate, 'lane_change_count', 1) != 2:
            return
        details['multi_lane_change'] = dict(via=candidate.via_link_id,
                                           target=candidate.target_link_id,
                                           count=2)
        details['lane_checks'].insert(0, 'two-lane {} via={} target={}'.format(
            candidate.direction, candidate.via_link_id, candidate.target_link_id))
