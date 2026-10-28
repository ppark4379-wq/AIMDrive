"""411 -> 409: follow the entry lane, accept a full gap, finish one change.

The measured y=140 boundary supersedes the later MGeo link endpoint. Waiting
reserves a short, drivable launch trajectory before that boundary. This module
owns the first entry only; it never chooses a second lateral destination.
"""
import math

from ....interfaces.geometry import Point
from ....config.parameters import highway as p
from ..common.types import HighwayPlanningResult
from .follow_target import MergeFollowing


class EntryMerge:
    def __init__(self, planner):
        self.planner = planner
        self.following = MergeFollowing(planner)
        self.reset()

    def reset(self):
        self.wait_speed = float('inf')
        self.last_conflict = None
        self.completed = False
        self.handoff_remaining = 0.
        self.crossing_confirmation_age = 0.
        self.following.reset()

    @staticmethod
    def station_at_y(lane, y):
        for i, (a, b) in enumerate(zip(lane.points, lane.points[1:])):
            if min(a.y, b.y) <= y <= max(a.y, b.y) and abs(b.y-a.y) > 1e-9:
                ratio = (y-a.y)/(b.y-a.y)
                return lane.cumulative_s[i] + ratio*(lane.cumulative_s[i+1]-lane.cumulative_s[i])
        raise ValueError('Entry merge boundary is outside the configured lane')

    def plan(self, ego, match, entries, matches, allow_change):
        planner = self.planner
        candidate = planner.active_candidate
        committed = candidate is not None and getattr(candidate, 'entry_merge', False)
        handing_off = (self.completed and self.handoff_remaining > 0.
                       and match.lane_link.id == p.ENTRY_MERGE_TARGET)
        if not committed and not handing_off and match.lane_link.id != p.ENTRY_MERGE_SOURCE:
            return None
        source = planner.lane_map.links[p.ENTRY_MERGE_SOURCE]
        target = planner.lane_map.links[p.ENTRY_MERGE_TARGET]
        source_match = planner.lane_map.project(ego.position, source)
        boundary_s = self.station_at_y(source, p.ENTRY_MERGE_END_Y)
        finish_s = boundary_s - p.VEHICLE_LENGTH - p.ENTRY_MERGE_END_MARGIN
        wait_s = finish_s - p.ENTRY_MERGE_MIN_DISTANCE
        planner._last_lane_checks = []
        planner._merge_guard_active = False
        planner._merge_speed_limit_state = float('inf')
        planner.merge_safety._update_tracks(ego, entries, matches)
        traffic, traffic_matches = planner._lane_change_traffic(entries, matches)
        details = dict(phase='APPROACH', source=p.ENTRY_MERGE_SOURCE, target=p.ENTRY_MERGE_TARGET,
                       boundary_y=p.ENTRY_MERGE_END_Y, wait_s=wait_s, source_s=source_match.s,
                       committed=committed)
        common = dict(current_link_id=match.lane_link.id, target_link_id=p.ENTRY_MERGE_TARGET,
                      highway_lane_links=planner.lane_map.links.values(),
                      object_lane_matches=matches, predicted_object_paths=planner._prediction_paths(matches),
                      debug_details={'entry_merge': details, 'filtered_merge_limit': float('inf'),
                                     'ego_lane': match.lane_link.lane_number, 'ego_s': match.s,
                                     'total_vehicle_objects': len(entries), 'matched_vehicle_objects': len(matches)})
        if committed:
            return self._continue(ego, match, candidate, traffic, traffic_matches, common)
        if handing_off:
            self.handoff_remaining -= planner.control_dt
            speed = self.following.update(ego, target, traffic_matches, max(1., finish_s-source_match.s),
                                          committed=True)
            gap, ttc = planner._front_risk(ego, match, traffic_matches, traffic)
            speed = min(speed, self._handoff_front_limit(ego, gap, ttc, planner._front_lead_speed))
            keep = planner._build_candidate(ego, target, match, 'keep')
            common['debug_details']['entry_merge'].update(self.following.details)
            return self._result(keep, speed, common, 'FOLLOWING', 'entry_merge_following')

        keep = planner._build_candidate(ego, source, source_match, 'keep')
        # The waiting path never commands the simulator's later y=89 endpoint.
        keep.points = [pt for pt in keep.points
                       if planner.lane_map.project(pt, source).s <= wait_s]
        stop_point, _, _ = planner.lane_map.advance_default(source_match, max(0., wait_s-source_match.s))
        if len(keep.points) < 2:
            keep.points = [Point(ego.position.x, ego.position.y), stop_point]
        elif keep.points[-1].distance(stop_point) > .01:
            keep.points.append(stop_point)

        remaining = max(0., wait_s-source_match.s)
        usable = max(0., remaining-max(0., ego.velocity)*p.LANE_CHANGE_CONTROL_DELAY)
        approach = math.sqrt(2*p.ENTRY_MERGE_APPROACH_DECEL*usable)
        front_gap, front_ttc = planner._front_risk(ego, source_match, traffic_matches, traffic)
        front_cap = planner._acc_speed_limit(ego.velocity, front_gap, front_ttc, planner._front_lead_speed)
        static_gap = planner.lane_change_safety.static_distance(keep.points, traffic,
            include_uncertain=True, margin=p.ENTRY_MERGE_OBSTACLE_MARGIN)
        # If a stopped source-lane car blocks further progress, reserve lateral
        # launch room as well as collision clearance. Stopping bumper-to-bumper
        # leaves only an untrackably tight escape and causes stop/go turning.
        static_cap = math.sqrt(2*p.ENTRY_MERGE_APPROACH_DECEL
                               * max(0., static_gap-p.ENTRY_MERGE_MIN_DISTANCE))
        # Endpoint braking is monotonic while blocked: no stop/go creep to
        # chase an adjacent car or repeatedly restart the approach profile.
        self.wait_speed = min(self.wait_speed, approach)
        wait_target = min(planner._requested_speed, self.wait_speed, front_cap, static_cap)
        follow_speed = self.following.update(ego, target, traffic_matches, max(1., finish_s-source_match.s))
        details.update(self.following.details)
        # Match a leading target-lane car BEFORE turning; otherwise repeatedly
        # testing the road speed predicts catching it and never accepts its gap.
        wait_target = min(wait_target, follow_speed)
        if remaining <= p.ENTRY_MERGE_STOP_TOLERANCE:
            wait_target = 0.
            details['phase'] = 'WAIT_GAP'

        room = finish_s-source_match.s
        distance = min(max(p.ENTRY_MERGE_MIN_DISTANCE,
                           max(ego.velocity, follow_speed)*p.ENTRY_MERGE_CHANGE_TIME), room)
        legal = any(lane.id == target.id for _, lane in planner.lane_map.legal_neighbors(source))
        distances = [distance, min(p.LANE_CHANGE_DISTANCE, room)]
        if math.isfinite(static_gap):
            # Same adjacent target, shorter geometry when the source lane is
            # blocked. Never expand this search to another lateral destination.
            distances += [min(value, room) for value in p.STATIC_CHANGE_DISTANCES]
        # Prefer the duration sized to this maneuver, not the longer cruising
        # fallback. Longer paths may reach a newly visible source obstruction.
        distances = list(dict.fromkeys(round(value, 6) for value in distances if value >= 12.))
        details['attempts'] = []
        self.last_conflict = None
        observation_age = getattr(ego, 'object_observation_age_s', 0.)
        details['observation_age_s'] = observation_age
        fresh_gap = (math.isfinite(observation_age)
                     and observation_age <= p.ENTRY_MERGE_START_MAX_OBSERVATION_AGE)
        if not fresh_gap:
            # Re-evaluate on the next frame BEFORE crossing. Never turn this
            # admission gate into stop/go braking on an already committed path.
            distances = []
            self.last_conflict = dict(kind='await_fresh_merge_observation', age_s=observation_age,
                                     max_age_s=p.ENTRY_MERGE_START_MAX_OBSERVATION_AGE)
        if not (allow_change and legal and ego.position.y <= p.ENTRY_MERGE_START_Y):
            distances = []
        for distance in distances:
            proposal = planner._build_candidate(ego, target, planner.lane_map.project(ego.position, target),
                                                'left', source, distance,
                                                prediction_horizon=p.MERGE_SAFETY_HORIZON)
            proposal.entry_merge = True
            proposal.compact_static = distance < p.LANE_CHANGE_DISTANCE-.1
            endpoint = getattr(proposal, 'transition_end', None)
            geometry_valid = proposal.valid and endpoint is not None and len(proposal.points) >= 2
            transition_end = (min(range(len(proposal.points)),
                                 key=lambda i: proposal.points[i].distance(endpoint))
                              if geometry_valid else 0)
            # A distant raw-map join in the appended cruising tail is not the
            # curvature of this lateral maneuver.
            curvature = (planner._max_curvature(proposal.points[:transition_end+1])
                         if geometry_valid else float('inf'))
            curvature_speed = math.sqrt(p.ENTRY_MERGE_LATERAL_ACCEL/max(curvature, 1e-6))
            speed_limit = min(curvature_speed, p.STATIC_CHANGE_MAX_SPEED
                              if proposal.compact_static else float('inf'))
            if proposal.compact_static:
                geometry_valid = geometry_valid and curvature <= p.STATIC_CHANGE_MAX_CURVATURE
            fits = geometry_valid and endpoint.y >= p.ENTRY_MERGE_END_Y+p.VEHICLE_LENGTH+p.ENTRY_MERGE_END_MARGIN-.1
            close_speed = min(follow_speed, speed_limit)
            if ego.velocity > .5:
                close_speed = min(close_speed, ego.velocity + p.ENTRY_MERGE_LAUNCH_SPEED_DELTA)
            speeds = [('close_gap', close_speed)]
            lead_speed = self.following.details.get('lead_speed')
            lead_gap = self.following.details.get('lead_gap')
            desired_gap = self.following.details.get('desired_gap')
            if (lead_speed is not None and lead_gap is not None and desired_gap is not None
                    and lead_gap >= desired_gap and lead_speed > .5):
                matched_speed = min(lead_speed, speed_limit)
                if abs(matched_speed-speeds[0][1]) > .05:
                    speeds.append(('match_lead', matched_speed))
            for speed_policy, speed in speeds:
                # A speed-matched slot may be safe even though continuously
                # closing its gap would catch the leader during the maneuver.
                # If Ego is faster, match speed BEFORE beginning lateral motion.
                can_hold_speed = speed >= ego.velocity-.2 and speed > .5
                if geometry_valid and fits and speed > .5:
                    conflict = planner.lane_change_safety.check(proposal, ego, traffic, traffic_matches, speed)
                    if conflict is None and self._clearance(ego, target, traffic, traffic_matches, speed):
                        if not can_hold_speed:
                            wait_target = min(wait_target, speed)
                            self.last_conflict = {'kind': 'prepare_speed', 'target_speed': speed}
                            details['attempts'].append(dict(distance=distance, speed=speed,
                                speed_policy=speed_policy, conflict=self.last_conflict))
                            continue
                        proposal.maneuver_speed_limit = proposal.speed_limit = proposal.safety_speed = speed
                        proposal.entry_speed_ceiling = min(p.ENTRY_MERGE_CRUISE_SPEED, speed_limit)
                        # A matched-speed admission must not immediately switch
                        # back to its previously rejected faster setpoint.
                        proposal.entry_speed_policy = speed_policy
                        self.following.command = speed
                        proposal.lane_change_verified = True
                        proposal.prediction_conflict = None
                        planner.active_candidate = proposal
                        planner.source_link_id, planner.target_link_id = source.id, target.id
                        planner.state = planner.CHANGE_LEFT
                        self.last_conflict = None
                        details.update(distance=distance, launch_speed=ego.velocity, speed_policy=speed_policy)
                        return self._result(proposal, speed, common, 'MERGING', 'entry_merge_start')
                    self.last_conflict = conflict or {'kind': 'target_clearance'}
                else:
                    self.last_conflict = {'kind': 'approach_geometry_or_speed'}
                details['attempts'].append(dict(distance=distance, speed=speed,
                    speed_policy=speed_policy, conflict=self.last_conflict))
        details.update(target_speed=wait_target, remaining_to_wait=remaining, conflict=self.last_conflict)
        common['nearest_vehicle_distance'], common['ttc'] = front_gap, front_ttc
        common['debug_details'].update(front_acc_limit=front_cap, static_obstacle_distance=static_gap)
        planner.state = planner.KEEP if wait_target > .05 else planner.ACC
        common['target_link_id'] = source.id
        return HighwayPlanningResult(keep.points, active=True, acc_override=True,
            acc_target_velocity=wait_target, highway_state=planner.state, candidates=[keep],
            reason='entry_merge_wait_gap' if wait_target <= .05 else 'entry_merge_approach', **common)

    def _clearance(self, ego, target, entries, matches, speed):
        planner = self.planner
        origin = planner.lane_map.project(ego.position, target)
        by_id = {id(entry['object_info']): match for entry, match in matches}
        for entry in entries:
            obj = entry['object_info']
            match = by_id.get(id(obj))
            if match is None or match.lane_link.id != target.id:
                continue  # Full swept collision check above still covers unmatched objects.
            signed = match.s-origin.s
            other_speed = max(0., obj.velocity)
            if signed >= 0:
                needed = p.VEHICLE_LENGTH + .5*max(1., obj.size_x) + p.ENTRY_MERGE_FRONT_GAP
                needed += max(0., speed-other_speed)*p.TARGET_LANE_FRONT_TIME_GAP
            else:
                needed = .5*max(1., obj.size_x) + p.ENTRY_MERGE_REAR_GAP
                needed += max(0., other_speed-speed)*p.TARGET_LANE_REAR_TIME_GAP
            if abs(signed) < needed:
                return False
        return True

    def _continue(self, ego, match, candidate, entries, matches, common):
        planner = self.planner
        target = planner.lane_map.links[p.ENTRY_MERGE_TARGET]
        following_speed = self.following.update(ego, target, matches, 100., committed=True,
                                                speed_ceiling=candidate.entry_speed_ceiling)
        common['debug_details']['entry_merge'].update(self.following.details)
        # While crossing, a target-lane car alongside Ego is NOT already a
        # same-lane ACC leader. Validate speed on the actual swept path instead.
        # Keep the last accepted maneuver speed separate from the follower's
        # temporary gap-making setpoint; a bad frame must not poison recovery.
        held_speed = getattr(candidate, 'entry_last_verified_speed', candidate.safety_speed)
        speed = min(candidate.entry_speed_ceiling,
                    held_speed + p.ENTRY_MERGE_PROGRESS_ACCEL * planner.control_dt)
        lead_speed = self.following.details.get('lead_speed')
        if lead_speed is not None:
            # Once near the leader's speed, stop increasing the setpoint. The
            # full swept-path check still decides whether holding is possible.
            speed = min(speed, max(held_speed, lead_speed+p.ENTRY_MERGE_PROGRESS_LEAD_DELTA))
        common['debug_details']['entry_merge'].update(
            speed_policy='committed_progress', admitted_speed=held_speed,
            progress_requested_speed=speed, front_follow_limit=None)
        aligned = abs(planner.lane_map._angle_difference(ego.yaw, match.yaw)) < math.radians(5.)
        if match.lane_link.id == target.id and match.distance <= p.MANEUVER_COMPLETE_DISTANCE and aligned:
            self.handoff_remaining = p.ENTRY_MERGE_HANDOFF_TIME
            gap, ttc = planner._front_risk(ego, match, matches, entries)
            front_limit = self._handoff_front_limit(ego, gap, ttc, planner._front_lead_speed)
            speed = min(following_speed, front_limit)
            self.following.command = speed
            common['debug_details']['entry_merge']['front_follow_limit'] = front_limit
            planner.state = planner.KEEP
            planner.active_candidate = None
            planner.source_link_id = None
            planner.changed_lane_hold = True
            planner._last_lane_change_direction = planner.CHANGE_LEFT
            planner._lane_change_cooldown_origin = Point(ego.position.x, ego.position.y)
            keep = planner._build_candidate(ego, match.lane_link, match, 'keep')
            common['target_link_id'] = match.lane_link.id
            self.completed = True
            self.handoff_remaining = p.ENTRY_MERGE_HANDOFF_TIME
            return self._result(keep, speed, common, 'COMPLETE', 'entry_merge_complete')
        conflict = planner.lane_change_safety.check(candidate, ego, entries, matches, speed, committed=True)
        common['debug_details']['entry_merge']['conflict'] = conflict
        # Rear approach alone must not trigger a deceleration that worsens it.
        # Still examine all other objects: an earlier rear conflict cannot mask
        # a simultaneous front obstacle or cut-in.
        front_conflict = conflict
        remaining_entries = list(entries)
        while front_conflict and front_conflict.get('rear'):
            identity = front_conflict.get('object_id')
            reduced = [e for e in remaining_entries if getattr(e['object_info'], 'unique_id', None) != identity]
            if len(reduced) == len(remaining_entries):
                break
            remaining_entries = reduced
            front_conflict = planner.lane_change_safety.check(candidate, ego, remaining_entries, matches, speed, committed=True)
        info = common['debug_details']['entry_merge']
        info['front_conflict'] = front_conflict
        # Verify raising a matched-speed command. A rejected acceleration is
        # not a reason to brake if the admitted speed still clears the front.
        if front_conflict and speed > held_speed + 1e-6:
            held_front = self._front_conflict(candidate, ego, entries, matches, held_speed)
            if held_front is None:
                speed = held_speed
                self.following.command = speed
                front_conflict = None
                info['acceleration_rejected'] = info['front_conflict']
        if front_conflict and self._confirm_distant_crossing(
                candidate, ego, entries, matches, speed, front_conflict, info):
            # Hold measured speed briefly; do not accelerate toward an
            # unconfirmed crossing, or command full brake for one bad heading.
            return self._result(candidate, min(speed, max(0., ego.velocity)), common,
                                'MERGING', 'entry_merge_confirm_crossing')
        if front_conflict is None:
            self.crossing_confirmation_age = 0.
        if front_conflict and not front_conflict.get('rear'):
            # A future stationary hypothesis needs a stopping-distance ceiling,
            # not an immediate zero-speed command. Keep enough distance to stop
            # even if this new box is real; raw close hazards still brake below.
            distance = front_conflict.get('distance') if front_conflict.get('kind') == 'static_path' else None
            if distance is not None:
                available = max(0., distance-p.ACC_STANDSTILL_DISTANCE
                                - max(0., ego.velocity)*p.LANE_CHANGE_CONTROL_DELAY)
                stopping_speed = math.sqrt(2*p.ENTRY_MERGE_APPROACH_DECEL*available)
                urgent_distance = ego.velocity**2/(2*p.LANE_CHANGE_PREDICT_DECEL)
                if available > urgent_distance and stopping_speed > .5:
                    speed = min(speed, stopping_speed)
                    common['debug_details']['entry_merge']['stopping_speed'] = stopping_speed
                    return self._result(candidate, speed, common, 'MERGING', 'entry_merge_front_constraint')
            # Only a newly unsafe actual maneuver may lower the held speed.
            # Keep its lateral path, and validate any braking against rear traffic.
            safe_speed = None
            # Test the admitted/measured/leader speeds before a fine downward
            # search. The old 2 m/s steps could skip a viable moving slot and
            # jump to zero even though braking itself failed the rear check.
            options = [held_speed, max(0., ego.velocity)]
            lead_speed = info.get('lead_speed')
            if lead_speed is not None:
                options.append(lead_speed)
            options += [max(0., speed-.25*i) for i in range(1, int(math.ceil(speed/.25))+1)]
            # Prefer the fastest verified slot, not the first (possibly much
            # slower) measured-speed option. This also avoids needless braking
            # for a small future front-gap constraint.
            options = sorted({round(min(candidate.entry_speed_ceiling, v), 6)
                              for v in options}, reverse=True)
            for value in options:
                if planner.lane_change_safety.check(candidate, ego, entries, matches, value,
                                                    require_completion=False, committed=True) is None:
                    safe_speed = value
                    break
            if safe_speed is None or safe_speed < ego.velocity-.05:
                # Already crossing: stopping solely because every usable speed
                # misses a fast rear car's gap makes that approach worse. Keep
                # the fastest non-braking speed that still clears EVERY front
                # body on the remaining path. This is mitigation, not a claim
                # that the rear gap is safe; expose it explicitly in the log.
                for value in options:
                    if value < ego.velocity:
                        break
                    rear_conflict = planner.lane_change_safety.check(
                        candidate, ego, entries, matches, value, committed=True)
                    if (rear_conflict and rear_conflict.get('rear')
                            and rear_conflict.get('kind') == 'gap'
                            and self._front_conflict(candidate, ego, entries, matches, value) is None):
                        candidate.entry_last_verified_speed = value
                        self.following.command = value
                        info.update(speed_policy='rear_risk_progress',
                                    rear_gap_unresolved=rear_conflict)
                        return self._result(candidate, value, common,
                                            'MERGING', 'entry_merge_rear_risk_progress')
            if safe_speed is not None:
                candidate.entry_last_verified_speed = safe_speed
                self.following.command = safe_speed
                info['speed_policy'] = 'verified_front_adjustment'
                if safe_speed >= ego.velocity-.05:
                    return self._result(candidate, safe_speed, common,
                                        'MERGING', 'entry_merge_in_progress')
            speed = 0. if safe_speed is None else safe_speed
            return self._result(candidate, speed, common, 'EMERGENCY', 'entry_merge_front_hazard')
        candidate.entry_last_verified_speed = speed
        self.following.command = speed
        return self._result(candidate, speed, common, 'MERGING', 'entry_merge_in_progress')

    def _handoff_front_limit(self, ego, gap, ttc, lead_speed):
        """Grow the entry gap into ordinary ACC's gap without a one-tick jump.

        Closing-speed stopping room is applied immediately, even at completion.
        Only the desired following spacing changes gradually over the existing
        handoff period; a real close/stopped obstacle still requires braking.
        """
        if math.isinf(gap):
            return float('inf')
        if lead_speed is None:
            closing = 0. if math.isinf(ttc) else gap/max(.1, ttc)
            lead_speed = max(0., ego.velocity-closing)
        fraction = min(1., max(0., 1.-self.handoff_remaining/max(.001, p.ENTRY_MERGE_HANDOFF_TIME)))
        entry_gap = p.ENTRY_MERGE_FRONT_GAP+p.ENTRY_MERGE_FOLLOW_RESERVE
        regular_gap = p.ACC_STANDSTILL_DISTANCE+max(0., ego.velocity)*p.ACC_TIME_GAP
        desired_gap = entry_gap + fraction*(regular_gap-entry_gap)
        closing = max(0., ego.velocity-lead_speed)
        desired_gap += closing*p.LANE_CHANGE_CONTROL_DELAY + closing**2/(2.*p.ACC_APPROACH_DECEL)
        return max(0., lead_speed+p.ACC_DISTANCE_GAIN*(gap-desired_gap))

    def _front_conflict(self, candidate, ego, entries, matches, speed):
        """Do not let the first rear gap hide a second, genuine front hazard."""
        remaining = list(entries)
        while True:
            conflict = self.planner.lane_change_safety.check(
                candidate, ego, remaining, matches, speed, committed=True)
            if not conflict or not conflict.get('rear'):
                return conflict
            identity = conflict.get('object_id')
            reduced = [e for e in remaining if e['object_info'].unique_id != identity]
            if len(reduced) == len(remaining):
                return conflict
            remaining = reduced

    def _confirm_distant_crossing(self, candidate, ego, entries, matches, speed, conflict, info):
        """Bounded confirmation only for distant, new, off-lane VE crossings.

Near bodies, known vehicles, mapped traffic and insufficient stopping room
retain immediate intervention. This never makes uncertain objects stationary.
        """
        obj = next((e['object_info'] for e in entries
                    if e['object_info'].unique_id == conflict.get('object_id')), None)
        if (conflict.get('kind') != 'collision' or obj is None
                or not getattr(obj, 'motion_uncertain', False)
                or getattr(obj, 'speed_unknown', False)
                or any(e['object_info'] is obj for e, _ in matches)
                or ego.velocity <= .5):
            return False
        duration = p.ENTRY_MERGE_CROSSING_CONFIRM_TIME
        self.crossing_confirmation_age += max(0., self.planner.control_dt)
        v = max(0., ego.velocity)
        stopping_distance = (v*(p.LANE_CHANGE_CONTROL_DELAY + duration)
                             + v*v/(2*p.LANE_CHANGE_PREDICT_DECEL)
                             + p.ACC_STANDSTILL_DISTANCE)
        conflict_distance = self.planner.lane_change_safety.travel(v, speed, conflict['time'])
        # A currently close body or a nearer real stationary obstacle bypasses
        # confirmation, regardless of track maturity or its predicted heading.
        measured_clearance = ego.position.distance(obj.position) - p.VEHICLE_LENGTH - obj.size_x/2
        static_distance = self.planner.lane_change_safety.static_distance(candidate.points, entries)
        if (self.crossing_confirmation_age > duration
                or conflict_distance <= stopping_distance
                or measured_clearance <= stopping_distance
                or static_distance <= stopping_distance):
            return False
        others = [e for e in entries if e['object_info'] is not obj]
        if self._front_conflict(candidate, ego, others, matches, min(v, speed)) is not None:
            return False
        info['crossing_confirmation'] = dict(object_id=obj.unique_id,
            age=self.crossing_confirmation_age, limit=duration,
            predicted_distance=conflict_distance, stopping_distance=stopping_distance)
        return True

    def _result(self, candidate, speed, common, phase, reason):
        info = common['debug_details']['entry_merge']
        info.update(phase=phase, target_speed=speed, compact=getattr(candidate, 'compact_static', False),
                    committed=phase in ('MERGING', 'EMERGENCY'),
                    verified=(phase != 'EMERGENCY' and reason != 'entry_merge_confirm_crossing'
                              and not info.get('rear_gap_unresolved')))
        return HighwayPlanningResult(candidate.points, active=True, acc_override=True,
            acc_target_velocity=speed, highway_state=self.planner.state, selected_offset=candidate.offset,
            candidates=[candidate], reason=reason, **common)
