"""Class-independent yielding for objects entering the commanded path.

Legacy ACC follows objects already near the path. This additional ceiling
checks swept object bodies before they reach that narrow following gate.
It uses perception only; no scenario actor IDs, crossings or simulator truth.
"""
import math
import time

import numpy as np


class DynamicCrossingGuard:
    def __init__(self, vehicle_length=4.635, vehicle_width=2., horizon=6.,
                 max_distance=45., clearance=.5, stop_margin=4.,
                 deceleration=2.5, reaction_time=.6, release_delay=.6,
                 approach_speed=2., time_margin=1.):
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.horizon = horizon
        self.max_distance = max_distance
        self.clearance = clearance
        self.stop_margin = stop_margin
        self.deceleration = deceleration
        self.reaction_time = reaction_time
        self.release_delay = release_delay
        self.approach_speed = approach_speed
        self.time_margin = time_margin
        if any(not math.isfinite(v) or v <= 0 for v in (
                vehicle_length, vehicle_width, horizon, max_distance, clearance,
                stop_margin, deceleration, reaction_time, release_delay,
                approach_speed, time_margin)):
            raise ValueError('Dynamic crossing limits must be finite and positive')
        self.reset()

    def reset(self):
        """Discard ordinary-road yield memory when another planner owns speed."""
        self._hazards = {}
        self._released = {}
        self._acc_excluded = set()

    @staticmethod
    def _key(obj):
        key = getattr(obj, 'unique_id', None)
        return key if key is not None and key != -1 else id(obj)

    def following_entries(self, entries):
        """Leave leads/signals/uncertain objects with ACC; own observed crossings.

        Only call immediately after evaluate, on the same path and observations.
        A finite crossing ceiling replaces ACC for active crossings. Observed
        body clearance also releases ACC's wider center-distance gate.
        """
        return [entry for entry in entries
                if entry['object_info'].type == 3
                or self._key(entry['object_info']) not in self._acc_excluded]

    @staticmethod
    def _interval(position, velocity, lower, upper):
        if abs(velocity) < 1e-6:
            return (-float('inf'), float('inf')) if lower <= position <= upper else (1., 0.)
        a, b = (lower-position)/velocity, (upper-position)/velocity
        return min(a, b), max(a, b)

    def evaluate(self, path, ego, entries, desired_speed, *, now=None):
        now = time.monotonic() if now is None else now
        self._acc_excluded.clear()
        self._released = {key: seen for key, seen in self._released.items()
                          if now-seen <= self.horizon}
        if len(path) < 2:
            self._hazards.clear()
            return float('inf'), dict(state='no_path')
        xy = np.asarray(path, dtype=float)[:, :2]
        delta = np.diff(xy, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        valid = lengths > 1e-6
        starts, lengths, delta = xy[:-1][valid], lengths[valid], delta[valid]
        if not len(lengths):
            self._hazards.clear()
            return float('inf'), dict(state='no_path')
        tangent = delta/lengths[:, None]
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        stations = np.r_[0., np.cumsum(lengths)[:-1]]

        def project(position):
            relative = np.asarray(position)-starts
            along = np.clip(np.sum(relative*tangent, axis=1), 0., lengths)
            distances = np.linalg.norm(relative-along[:, None]*tangent, axis=1)
            i = int(np.argmin(distances))
            return float(stations[i]+along[i]), float(distances[i])

        ego_s, _ = project(ego.position)
        for entry in entries:
            obj = entry['object_info']
            if obj.type == 3:  # Signal state has its own stop authority.
                continue
            key = self._key(obj)
            held = key in self._hazards
            uncertain = (getattr(obj, 'speed_unknown', False)
                         or getattr(obj, 'motion_uncertain', False))
            if getattr(obj, 'speed_unknown', False):
                # The transport's speed bound is not an observed motion vector.
                speed = 0.
            else:
                speed = max(0., float(obj.velocity))
            if speed < .1 and not held and key not in self._released:
                continue  # Standing obstacles remain with path ACC/avoidance.
            angle = math.radians(obj.heading_deg)
            if not all(math.isfinite(v) for v in (*obj.position, speed, angle, obj.size_x, obj.size_y)):
                continue
            direction = np.array([math.cos(angle), math.sin(angle)])
            relative = np.asarray(obj.position)-starts
            along = np.sum(relative*tangent, axis=1)
            across = np.sum(relative*normal, axis=1)
            longitudinal = speed*(tangent@direction)
            lateral = speed*(normal@direction)
            half_length = .5*(max(.8, obj.size_x)*np.abs(tangent@direction)
                             + max(.8, obj.size_y)*np.abs(normal@direction))
            half_width = .5*(max(.8, obj.size_x)*np.abs(normal@direction)
                            + max(.8, obj.size_y)*np.abs(tangent@direction))
            widths = self.vehicle_width/2.+half_width+self.clearance
            distances = np.linalg.norm(
                relative-np.clip(along, 0., lengths)[:, None]*tangent, axis=1)
            nearest = int(np.argmin(distances))
            outside = bool(np.all(distances > widths))
            outward = math.copysign(1., across[nearest])*lateral[nearest]
            # A positive clearance observation is different from losing a
            # detection. Never infer clearance from an uncertain speed bound.
            cleared = (outside and not uncertain
                       and (outward > .2 or (key in self._released and outward >= 0.)))
            best = None
            for i in range(len(lengths)):
                width = widths[i]
                if abs(lateral[i]) < .2 and (not held or speed >= .1):
                    continue  # Parallel traffic is handled by following ACC.
                if abs(across[i]) > width and across[i]*lateral[i] >= 0.:
                    continue  # Already outside and departing from this path.
                enter, leave = self._interval(across[i], lateral[i], -width, width)
                begin, end = self._interval(along[i], longitudinal[i], 0., lengths[i])
                # Preserve the actual lateral exit time. A segment endpoint or
                # prediction horizon is not evidence that a body has cleared.
                exit_time = leave
                enter, leave = max(0., enter, begin), min(self.horizon, leave, end)
                if enter > leave:
                    continue
                # Nearest possible occupied station over this interval. Include
                # object length and the ego front, both in the path direction.
                t = enter if longitudinal[i] >= 0 else leave
                occupied_s = stations[i]+along[i]+longitudinal[i]*t
                distance = occupied_s-ego_s
                if distance+half_length[i] <= 0. or distance > self.max_distance:
                    continue
                stop_s = occupied_s-half_length[i]-self.vehicle_length-self.stop_margin
                point = starts[i]+(along[i]+longitudinal[i]*t)*tangent[i]
                if best is None or stop_s < best['stop_s']:
                    best = dict(object_id=key, stop_s=float(stop_s), point=point,
                                object_half_length=float(half_length[i]),
                                entry_time_s=float(enter), exit_time_s=float(exit_time),
                                motion_confident=not uncertain,
                                lateral_m=float(across[i]),
                                lateral_speed_mps=float(lateral[i]), seen=now)
            if best is not None:
                self._released.pop(key, None)
                self._hazards[key] = best
                if not uncertain:
                    self._acc_excluded.add(key)
            elif cleared:
                self._released[key] = now
                self._acc_excluded.add(key)
            else:
                self._released.pop(key, None)

        candidates = []
        for key, hazard in list(self._hazards.items()):
            station, side = project(hazard['point'])
            if now-hazard['seen'] > self.release_delay or side > self.vehicle_width/2.+self.clearance:
                del self._hazards[key]
                continue
            gap = station-ego_s-hazard['object_half_length']-self.vehicle_length
            available = max(0., gap-self.stop_margin)
            # v*reaction_time + v^2/(2*a) <= available stopping distance.
            ar = self.deceleration*self.reaction_time
            limit = max(0., math.sqrt(ar*ar+2*self.deceleration*available)-ar)
            if available <= .25:
                limit = 0.
            # Even instantaneous acceleration to the requested speed must not
            # reach the body until its predicted exit plus a time margin. Keep
            # the stopping envelope regardless: the object may slow or stop.
            age = max(0., now-hazard['seen'])
            exit_time = max(0., hazard['exit_time_s']-age)
            fastest_speed = max(0., ego.velocity, desired_speed)
            arrival = max(0., gap)/fastest_speed if fastest_speed > 1e-6 else float('inf')
            clears_before_arrival = (hazard['motion_confident']
                                     and exit_time+self.time_margin < arrival)
            if not clears_before_arrival:
                # Permit bounded progress from rest instead of locking a zero
                # target forever. Near the stop line the envelope still wins.
                limit = min(limit, max(self.approach_speed, ego.velocity))
            limit = min(limit, max(0., desired_speed))
            candidates.append((limit, dict(
                state='yield' if now == hazard['seen'] else 'clearance_hold',
                object_id=key, entry_time_s=hazard['entry_time_s'],
                exit_time_s=exit_time if math.isfinite(exit_time) else None,
                earliest_arrival_s=arrival if math.isfinite(arrival) else None,
                clears_before_arrival=clears_before_arrival,
                lateral_m=hazard['lateral_m'], lateral_speed_mps=hazard['lateral_speed_mps'],
                front_gap_m=float(gap), stop_margin_m=self.stop_margin,
                target_velocity=limit, horizon_s=self.horizon)))
        return min(candidates, key=lambda item: item[0]) if candidates else (
            float('inf'), dict(state='clear', horizon_s=self.horizon))
