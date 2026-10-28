"""Persistent target and continuous following speed for the 411 -> 409 merge.

This chooses a longitudinal slot, never a steering destination or safety verdict.
All observed vehicles still go through the lane-change collision checks.
"""
import math
from bisect import bisect_right

from ....config.parameters import highway as p


class MergeFollowing:
    def __init__(self, planner):
        self.planner = planner
        self.reset()

    def reset(self):
        self.lead = None
        self.command = None
        self.age = 0.0
        self.expired_lead_id = None
        self.details = {}

    def _row(self, entry, match, origin):
        obj = entry['object_info']
        if match.lane_link.id != p.ENTRY_MERGE_TARGET:
            return None
        if (getattr(obj, 'speed_unknown', False) or not math.isfinite(obj.velocity)
                or abs(self.planner.lane_map._angle_difference(
                    math.radians(obj.heading_deg), match.yaw)) > math.radians(35.)):
            return None
        return dict(identity=getattr(obj, 'unique_id', None), s=match.s,
                    gap=match.s-origin.s, speed=max(0., obj.velocity),
                    length=max(1., obj.size_x), x=obj.position.x, y=obj.position.y,
                    yaw=math.radians(obj.heading_deg), age=entry.get('prediction_age', 0.))

    def _observe(self, rows, dt, committed):
        event = 'held'
        self.expired_lead_id = None
        if self.lead is not None:
            self.age += dt
            fresh = [r for r in rows if r['age'] == 0.]
            observed = next((r for r in fresh if r['identity'] == self.lead['identity']), None)
            if observed is None:
                # A new tracker ID may refer to the same physical vehicle.
                # Require an unambiguous position, direction AND speed match.
                expected = (self.lead['x'] + self.lead['speed']*self.age*math.cos(self.lead['yaw']),
                            self.lead['y'] + self.lead['speed']*self.age*math.sin(self.lead['yaw']))
                possible = [r for r in fresh
                            if math.hypot(r['x']-expected[0], r['y']-expected[1]) <= p.ENTRY_MERGE_REASSOC_DISTANCE
                            and abs(r['speed']-self.lead['speed']) <= p.ENTRY_MERGE_REASSOC_SPEED
                            and abs(self.planner.lane_map._angle_difference(r['yaw'], self.lead['yaw'])) < math.radians(15.)]
                if len(possible) == 1 and self.age <= p.ENTRY_MERGE_TARGET_HOLD:
                    observed = possible[0]
                    event = 'reassociated'
            if observed is not None:
                alpha = dt/(p.ENTRY_MERGE_SPEED_FILTER_TIME+dt)
                observed = dict(observed)
                observed['speed'] = self.lead['speed'] + alpha*(observed['speed']-self.lead['speed'])
                self.lead, self.age = observed, 0.
                return event
            if self.age <= p.ENTRY_MERGE_TARGET_HOLD:
                return 'predicted_short_dropout'
            # A following ID is not a permanent occupancy lock. After the
            # bounded dropout hold, select again from current observations.
            # This does NOT grant a merge: all observed/held traffic still
            # passes the full clearance and swept-path checks in EntryMerge.
            self.expired_lead_id = self.lead['identity']
            self.lead = None
            self.age = 0.
            event = 'target_expired'
        if self.lead is None:
            candidates = [r for r in rows if r['age'] == 0.
                          and -p.ENTRY_MERGE_SEARCH_REAR <= r['gap'] <= p.ENTRY_MERGE_SEARCH_FRONT
                          and (not committed or r['gap'] > p.VEHICLE_LENGTH + .5*r['length'])
                          and r['identity'] is not None]
            if candidates:
                self.lead = dict(min(candidates, key=lambda r: abs(r['gap'])))
                self.age = 0.
                event = 'reselected_after_dropout' if self.expired_lead_id is not None else 'selected'
        return event if self.lead or self.expired_lead_id is not None else 'empty'

    def update(self, ego, target, matches, room, committed=False, speed_ceiling=None):
        dt = min(.2, max(.001, self.planner.control_dt))
        origin = self.planner.lane_map.project(ego.position, target)
        rows = [r for entry, match in matches if (r := self._row(entry, match, origin)) is not None]
        event = self._observe(rows, dt, committed)
        released_rear_id = None
        if committed and self.lead is not None:
            gap = self.lead['s'] + self.lead['speed']*min(self.age, p.ENTRY_MERGE_TARGET_HOLD) - origin.s
            if gap <= 0.:
                # Once steering has begun (including the completion handoff),
                # an old target behind Ego cannot remain a longitudinal leader.
                # Only release the follow target; collision checks retain it.
                released_rear_id = self.lead['identity']
                self.lead = None
                self.age = 0.
                event = 'released_rear_target'
        limit = p.ENTRY_MERGE_CRUISE_SPEED if speed_ceiling is None else speed_ceiling
        if released_rear_id is not None and self.command is not None:
            # Do not carry the erroneous rear-following deceleration forward.
            # Real front/geometry limits are still applied by EntryMerge.
            self.command = max(self.command, min(limit, max(0., ego.velocity)))
        requested = min(limit, self.planner._requested_speed) if not committed else limit
        gap = desired_gap = None
        if self.lead:
            lead = self.lead
            gap = lead['s'] + lead['speed']*min(self.age, p.ENTRY_MERGE_TARGET_HOLD) - origin.s
            desired_gap = (p.VEHICLE_LENGTH + .5*lead['length'] + p.ENTRY_MERGE_FRONT_GAP
                           + p.ENTRY_MERGE_FOLLOW_RESERVE)
            # Resolve side-by-side occupancy BEFORE steering. Limited remaining
            # distance shortens the gap-making time, rather than creeping to the end.
            response = min(p.ENTRY_MERGE_GAP_TIME, max(1., room/max(ego.velocity, 1.)*.5))
            correction = max(-p.ENTRY_MERGE_GAP_SPEED_DELTA,
                             min(p.ENTRY_MERGE_GAP_SPEED_DELTA, (gap-desired_gap)/response))
            requested = min(requested, max(0., lead['speed']+correction))
        if self.command is not None:
            requested = max(self.command-p.ENTRY_MERGE_FOLLOW_DECEL*dt,
                            min(self.command+p.ENTRY_MERGE_FOLLOW_ACCEL*dt, requested))
        if (self.lead and gap >= desired_gap and self.lead['speed'] >= ego.velocity
                and requested < ego.velocity):
            # Gap creation has finished. Do not keep a lower old setpoint and
            # reject a safe turn solely because of our own slew limiter.
            requested = min(limit, ego.velocity)
        self.command = max(0., min(limit, requested))
        self.details = dict(lead_id=self.lead['identity'] if self.lead else None,
                            released_rear_id=released_rear_id,
                            expired_lead_id=self.expired_lead_id,
                            lead_age=self.age, target_event=event, lead_gap=gap,
                            desired_gap=desired_gap, follow_speed=self.command,
                            lead_speed=self.lead['speed'] if self.lead else None,
                            search_front=p.ENTRY_MERGE_SEARCH_FRONT, search_rear=p.ENTRY_MERGE_SEARCH_REAR,
                            search_polygon=self.search_polygon(target, origin.s))
        return self.command

    def search_polygon(self, target, origin_s):
        start = max(0., origin_s-p.ENTRY_MERGE_SEARCH_REAR)
        end = min(target.length, origin_s+p.ENTRY_MERGE_SEARCH_FRONT)
        stations = [start] + [s for s in target.cumulative_s if start < s < end] + [end]
        left, right = [], []
        for station in stations:
            i = max(0, min(len(target.points)-2, bisect_right(target.cumulative_s, station)-1))
            a, b = target.points[i:i+2]
            length = target.cumulative_s[i+1]-target.cumulative_s[i]
            f = (station-target.cumulative_s[i])/max(length, 1e-9)
            x, y = a.x+(b.x-a.x)*f, a.y+(b.y-a.y)*f
            yaw = math.atan2(b.y-a.y, b.x-a.x)
            dx, dy = -.5*p.LANE_WIDTH*math.sin(yaw), .5*p.LANE_WIDTH*math.cos(yaw)
            left.append([x+dx, y+dy]); right.append([x-dx, y-dy])
        return left+list(reversed(right))+left[:1]
