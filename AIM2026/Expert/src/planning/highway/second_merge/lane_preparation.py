"""Reach the leftmost legal lane upstream, one adjacent maneuver at a time.

Wait for a gap on the source lane; optional active gap creation is disabled by
default. Admission and the committed maneuver
reuse the ordinary swept-body/front/rear checks, including held observations.
"""
import math
import time

from ....config.parameters import highway as p
from ....interfaces.geometry import Point
from ..common.types import HighwayPlanningResult


class SecondMergePreparation:
    SOURCE = 'A2256W000408'
    TARGET = 'A2256W000434'
    TAPER_SOURCE = 'A2256W000445'
    TAPER_TARGET = 'A2256W000422'
    # Lane 3 is an intermediate stop, not completion. Lane 1 is prohibited.
    NEXT_LEFT = {
        'A2256W000409': 'A2256W000410',
        'A2256W000410': 'A2256W000418',
        'A2256W000420': 'A2256W000430',
        'A2256W000430': 'A2256W000431',
        'A2256W000408': 'A2256W000434',
        'A2256W000434': 'A2256W000435',
    }
    APPROACH_LINKS = frozenset(NEXT_LEFT) | frozenset(NEXT_LEFT.values())
    EARLY_LINKS = frozenset(('A2256W000409', 'A2256W000410', 'A2256W000418'))
    TAPER_LINKS = frozenset(('A2256W000445', 'A2256W000422', 'A2256W000423'))

    def __init__(self, planner):
        self.planner = planner
        self.reset()

    def reset(self):
        self.observations = {}
        self.early_started = False
        self.clear_gap()

    def clear_gap(self):
        self.wait_speed = None
        self.follow_id = None

    def update(self, ego, match, entries, matches):
        if not self.applies(ego, match):
            self.reset()
            return
        # Continue across the y=89 link boundary even before the old start
        # point. The first merge/handoff and change cooldown still take
        # precedence in the planner; this never extends a committed maneuver.
        self.early_started |= match.lane_link.id in self.EARLY_LINKS
        for record in self.observations.values():
            record[2] += self.planner.control_dt
        matched = {id(entry['object_info']): lane for entry, lane in matches}
        for entry in entries:
            obj = entry['object_info']
            lane = matched.get(id(obj))
            key = self.planner._merge_zone_object_key(entry, lane)
            if key.startswith('link:'):
                key += ':{:.1f}:{:.1f}'.format(obj.position.x, obj.position.y)
            self.observations[key] = [obj, lane, 0.]
        self.observations = {key: record for key, record in self.observations.items()
                             if record[2] <= p.MERGE_TRACK_HOLD_TIME}

    def applies(self, ego, match):
        if not p.SECOND_MERGE_PREP_ENABLED:
            return False
        link = match.lane_link
        if link.id not in self.APPROACH_LINKS | self.TAPER_LINKS:
            return False
        if abs(self.planner.lane_map._angle_difference(ego.yaw, match.yaw)) > math.pi / 3:
            return False
        if link.id in self.EARLY_LINKS or self.early_started:
            return True
        if link.id in self.APPROACH_LINKS:
            start = self.planner.lane_map.project(Point(*p.SECOND_MERGE_PREP_START), link)
            return match.s >= start.s
        return True

    def on_target(self, match):
        return (match.lane_link.id in self.APPROACH_LINKS | self.TAPER_LINKS
                and match.lane_link.lane_number in p.SECOND_MERGE_PREFERRED_LANE_NUMBERS)

    def remaining_before_taper(self, match):
        """Room across continuous upstream links; never start inside the taper."""
        if match.lane_link.id not in self.APPROACH_LINKS:
            return 0.
        remaining = match.lane_link.length - match.s
        seen = {match.lane_link.id}
        next_id = self.planner.lane_map.default_successor(match.lane_link.id)
        while next_id in self.APPROACH_LINKS and next_id not in seen:
            seen.add(next_id)
            remaining += self.planner.lane_map.links[next_id].length
            next_id = self.planner.lane_map.default_successor(next_id)
        return remaining

    def _gap_speed(self, ego, target, matches, details):
        planner = self.planner
        projection = planner.lane_map.project(ego.position, target)
        vehicles = []
        for entry, match in matches:
            obj = entry['object_info']
            if getattr(obj, 'speed_unknown', False):
                continue  # Still checked for collision; do not invent a follow speed.
            if match.lane_link.id == target.id:
                signed = match.s - projection.s
            else:
                signed = planner.lane_map.forward_distance(projection, match)
                if signed is None:
                    behind = planner.lane_map.forward_distance(match, projection)
                    if behind is None:
                        continue
                    signed = -behind
            if not -p.SECOND_MERGE_PREP_REAR_RANGE <= signed <= p.SECOND_MERGE_PREP_FRONT_RANGE:
                continue
            key = planner._merge_zone_object_key(entry, match)
            gap = signed - p.VEHICLE_LENGTH - .5 * max(1., obj.size_x)
            vehicles.append((key, gap, max(0., obj.velocity)))
        chosen = next((v for v in vehicles if v[0] == self.follow_id), None)
        if chosen is None and vehicles:
            chosen = min(vehicles, key=lambda v: abs(v[1] + p.VEHICLE_LENGTH))
        ceiling = min(planner._requested_speed, p.SECOND_MERGE_PREP_SPEED_KPH / 3.6)
        if chosen is not None:
            self.follow_id, gap, speed = chosen
            ceiling = min(ceiling, max(0., speed + p.SECOND_MERGE_PREP_GAP_GAIN
                                      * (gap - p.SECOND_MERGE_PREP_FRONT_GAP)))
            details.update(follow_id=self.follow_id, front_bumper_gap=gap,
                           follow_speed=speed)
        else:
            self.follow_id = None
        if self.wait_speed is None:
            self.wait_speed = min(max(0., ego.velocity), planner._requested_speed)
        # Smooth only this preparation request. Actual front/merge emergency
        # limits are applied separately and must never be delayed by this ramp.
        step = p.SECOND_MERGE_PREP_DECEL * planner.control_dt
        self.wait_speed += max(-step, min(step, ceiling - self.wait_speed))
        details.update(gap_speed=ceiling, preparation_speed=self.wait_speed)
        return self.wait_speed

    def plan(self, ego, match, entries, matches, keep, acc_limit, common, allow_change):
        planner = self.planner
        if not self.applies(ego, match):
            self.reset()
            return None
        if planner.state in (planner.CHANGE_LEFT, planner.CHANGE_RIGHT):
            if getattr(planner.active_candidate, 'second_merge_preparation', False):
                common['debug_details']['second_merge_preparation'] = dict(
                    phase='CHANGE_LEFT', source=planner.source_link_id, target=planner.target_link_id)
            return None
        details = dict(phase='HOLD_LEFT' if self.on_target(match) else 'PREPARE',
                       source=match.lane_link.id,
                       target=(match.lane_link.id if self.on_target(match)
                               else self.NEXT_LEFT.get(match.lane_link.id)),
                       preferred_lanes=p.SECOND_MERGE_PREFERRED_LANE_NUMBERS)
        common['debug_details']['second_merge_preparation'] = details
        candidates = [keep]
        limited = acc_limit
        reason = 'highway_second_merge_hold_left'
        if self.on_target(match):
            self.clear_gap()
            planner.changed_lane_hold = True
        else:
            remaining = self.remaining_before_taper(match)
            enough_room = (match.lane_link.id in self.NEXT_LEFT and remaining >=
                           p.LANE_CHANGE_DISTANCE + p.SECOND_MERGE_PREP_END_MARGIN)
            details['remaining_before_taper'] = remaining
            if enough_room and allow_change:
                traffic, traffic_matches = planner._lane_change_traffic(entries, matches)
                target = next((link for direction, link in planner.lane_map.legal_neighbors(match.lane_link)
                               if direction == 'left' and link.id == self.NEXT_LEFT[match.lane_link.id]), None)
                if target is not None:
                    planner._search_deadline = (None if planner.search_time_budget is None else
                                               time.monotonic() + planner.search_time_budget)
                    candidate = planner._build_candidate(
                        ego, target, planner.lane_map.project(ego.position, target),
                        'left', source_link=match.lane_link,
                        transition_distance=p.LANE_CHANGE_DISTANCE,
                        prediction_horizon=p.MERGE_SAFETY_HORIZON)
                    candidate.second_merge_preparation = True
                    candidate.speed_limit = float('inf')
                    planner._evaluate_lane_change(candidate, ego, traffic, traffic_matches)
                    clearance = planner._target_lane_has_clearance(
                        target, candidate, traffic_matches, max(0., ego.velocity))
                    candidate.valid = candidate.valid and clearance
                    origin = planner._lane_change_cooldown_origin
                    cooldown = (origin is not None and ego.position.distance(origin)
                                < p.LANE_CHANGE_COOLDOWN_DISTANCE)
                    if cooldown:
                        candidate.valid = False
                        details['cooldown'] = True
                    candidates.append(candidate)
                    details['conflict'] = getattr(candidate, 'prediction_conflict', None)
                    details['target_clearance'] = bool(clearance)
                    if candidate.valid:
                        self.clear_gap()
                        planner.source_link_id = match.lane_link.id
                        planner.target_link_id = target.id
                        planner.active_candidate = candidate
                        planner.state = planner.CHANGE_LEFT
                        common['target_link_id'] = target.id
                        details['phase'] = 'CHANGE_LEFT'
                        planner._describe_change_speed(candidate, common['debug_details'])
                        return HighwayPlanningResult(
                            candidate.points, active=True, acc_override=True,
                            acc_target_velocity=candidate.safety_speed,
                            highway_state=planner.state, candidates=candidates,
                            selected_offset=candidate.offset,
                            reason='highway_second_merge_prepare_left', **common)
                    if p.SECOND_MERGE_PREP_SLOW_FOR_GAP:
                        details['phase'] = 'CREATE_GAP'
                        limited = min(limited, self._gap_speed(ego, target, traffic_matches, details))
                    else:
                        # A blocked optional lane change is not an obstacle on
                        # our current lane. Keep its normal ACC/merge ceiling;
                        # do not wait behind an adjacent rear car by slowing.
                        self.clear_gap()
                        details['phase'] = 'WAIT_FOR_GAP'
                        details['adjacent_gap_braking'] = False
                        # A blocked optional change alone imposes no speed cap.
                        # Actual front ACC / convergence approach limits remain
                        # in acc_limit; active gap creation has its own ceiling.
                reason = ('highway_second_merge_create_gap' if p.SECOND_MERGE_PREP_SLOW_FOR_GAP
                          else 'highway_second_merge_wait_for_gap')
            else:
                # Too late for a complete upstream maneuver: preserve the
                # original merge GO/YIELD policy, never force a taper crossing.
                self.clear_gap()
                details['phase'] = 'MERGE_ON_CURRENT_LANE' if not enough_room else 'FOLLOW_ONLY'
                reason = 'highway_second_merge_current_lane'
        if (not planner._merge_guard_active and not keep.valid
                and keep.collision_time <= p.MERGE_EMERGENCY_COLLISION_TIME):
            limited = 0.
        planner.state = planner.ACC if math.isfinite(limited) else planner.KEEP
        return HighwayPlanningResult(
            keep.points, active=True, acc_override=math.isfinite(limited),
            acc_target_velocity=limited, obstacle_detected=math.isfinite(limited),
            highway_state=planner.state, candidates=candidates, reason=reason, **common)
