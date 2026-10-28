"""Receding-horizon path for standing obstacles in the Jamming (tunnel) section.

The static-section planner (static_obstacle/corridor_optimizer.py) on the tunnel's accumulated occupancy map
(Expert/tunnel_map.py), in the section's forward-Y frame. Its drivable corridor is the mapped tunnel road (both lanes,
RoadBoundary) instead of the recorded routes: the body corners keep the road's planning margin from its edges.
Moving traffic reaches it only as boxes added to the map and a speed cap (traffic.py).
"""
from collections import deque
import time

import numpy as np
from shapely.geometry import LineString

from ..static_obstacle.corridor_optimizer import StaticCorridorPlanner, ESCAPE_DISTANCE
from ...config.parameters.jamming import (TIGHT_GAP_MARGIN, TIGHT_GAP_SPEED, TUNNEL_MAX_SPEED_KPH,
                                          TUNNEL_CLEARANCE_SPEED, FALLBACK_STALL_S, FALLBACK_STALL_SPEED,
                                          FALLBACK_CLEARANCES, FALLBACK_SPEED, FALLBACK_RELEASE_TICKS,
                                          FALLBACK_CONTACT_STALL_S, FALLBACK_CONTACT_STALL_SPEED)

ROAD_STEP = .25           # longitudinal spacing of the tabulated road edges [m]
ROAD_PROBE = 15.          # lateral half-length of the probe line that cuts the road polygon [m]


class TunnelCorridorPlanner(StaticCorridorPlanner):
    REASON = 'jamming_corridor_'
    # Objects are seen too small from afar and grow on the map as the car closes in: tight gaps are entered slowly.
    TIGHT_MARGIN = TIGHT_GAP_MARGIN
    TIGHT_SPEED = TIGHT_GAP_SPEED
    MAX_SPEED = TUNNEL_MAX_SPEED_KPH/3.6
    CLEARANCE_SPEED = TUNNEL_CLEARANCE_SPEED

    def __init__(self, base, occupancy, section, road_boundary):
        super().__init__(base, occupancy)
        self.section = section
        self.road_boundary = road_boundary
        self.margin = road_boundary.planning_margin_m
        self.road_y = self.road_low = self.road_high = None
        self.clock = time.monotonic             # replaced by the replay's clock in offline checks
        self._stall_since = None                # when the car began standing or creeping blocked
        self._fallback_active = False           # stays on until a path passes the normal clearance again
        self._fallback_stage = None             # index of the last stage that gave a path
        self._clear_ticks = 0                   # consecutive ticks with a normal path
        self._committed_relaxed = False         # the committed path came from a last-resort solve
        self._standing_since = None             # when the car came to a full stop, blocked
        self._contact_granted = False           # the contact stages are allowed until the last resort ends
        self.fallback_log = deque(maxlen=1000)  # (clock, stage clearance) of the recent ticks that used the last resort

    def run(self, ego, reference_path, global_path):
        result = super().run(ego, reference_path, global_path)
        if FALLBACK_STALL_S <= 0.:
            return result
        info = None if result is None else result.debug_details['static_avoidance']
        if info is None or info.get('phase') != 'blocked':
            if info is not None and info.get('source') == 'new':
                self._committed_relaxed = False     # a fresh normal solve replaced the committed path
            if self._fallback_active and info is not None and info.get('source') == 'previous' and self._committed_relaxed:
                # The normal solve still holds the last-resort path the car is on: that is no proof the gap is wide
                # enough, keep the last resort (and its speed) until a fresh normal solve passes.
                self._cap(result, info, None)
                return result
            # A fresh normal path: the last resort ends once it has held for FALLBACK_RELEASE_TICKS ticks (a path
            # that passes and fails tick by tick would start and stop the car again).
            self._clear_ticks += 1
            if not self._fallback_active or info is None or self._clear_ticks >= FALLBACK_RELEASE_TICKS:
                self._reset_fallback()
            return result
        self._clear_ticks = 0
        now = self.clock()
        if abs(ego.velocity) < FALLBACK_CONTACT_STALL_SPEED:
            self._standing_since = now if self._standing_since is None else self._standing_since
        else:
            self._standing_since = None
        if getattr(self.occupancy, 'virtual', None):
            self._reset_fallback()        # a mover is being overtaken or yielded to: its own rules decide, not this
            return result
        if not self._fallback_active:
            # Blocked, the planner still lets the car close in on the stop point 1 m before the obstacle, and the
            # point moves on as the map changes: the car crept forward twice (to 1.3 m/s) and a 0.3 m/s standing test
            # restarted the wait each time (9 s instead of 3, competition tunnel 10-29). Slow approach counts too.
            if abs(ego.velocity) >= FALLBACK_STALL_SPEED:
                self._stall_since = None
                return result
            if self._stall_since is None:
                self._stall_since = now
            if now-self._stall_since < FALLBACK_STALL_S:
                return result
            self._fallback_active = True
        relaxed = self._relaxed_plan(ego, reference_path, global_path)
        if relaxed is None:
            return result
        return relaxed

    def _reset_fallback(self):
        self._stall_since = None
        self._fallback_active = False
        self._fallback_stage = None
        self._standing_since = None
        self._contact_granted = False

    def _contact_allowed(self):
        """The contact stages only after a full stop of FALLBACK_CONTACT_STALL_S; once used, until the last resort ends
        (the car moving along that path is no reason to stop it again)."""
        if FALLBACK_CONTACT_STALL_S <= 0.:
            return True
        if not self._contact_granted and self._standing_since is not None:
            self._contact_granted = self.clock()-self._standing_since >= FALLBACK_CONTACT_STALL_S
        return self._contact_granted

    def _relaxed_plan(self, ego, reference_path, global_path):
        """The same solve with the required clearance lowered step by step (road edge condition unchanged); the stage
        that worked last tick goes first. Without any path, the committed path ahead of the car, if the car is on it."""
        order = list(range(len(FALLBACK_CLEARANCES)))
        if self._fallback_stage is not None:
            order.remove(self._fallback_stage)
            order.insert(0, self._fallback_stage)
        nominal = self.min_clearance
        # A failed attempt must leave no trace in the normal planning (its last solution, committed path and solve
        # cadence): every stage starts from the state before the attempt, and without any path it is restored.
        saved = (self.previous, self.committed, self._since_solve, self._held_error)
        try:
            for stage in order:
                if FALLBACK_CLEARANCES[stage] < 0. and not self._contact_allowed():
                    continue
                self.previous, self.committed, self._since_solve, self._held_error = saved
                self.min_clearance = FALLBACK_CLEARANCES[stage]
                self._since_solve = 1 << 30         # a held path of the stricter solve is not reused: solve afresh
                result = super().run(ego, reference_path, global_path)
                if result is None:
                    continue
                info = result.debug_details['static_avoidance']
                if info.get('phase') == 'blocked':
                    continue
                self._fallback_stage = stage
                self._committed_relaxed = True
                self._cap(result, info, FALLBACK_CLEARANCES[stage])
                return result
            self.previous, self.committed, self._since_solve, self._held_error = saved
        finally:
            self.min_clearance = nominal
        return None

    def _cap(self, result, info, clearance):
        """Speed cap and log of a last-resort tick; ``clearance`` None: the normal solve held the last-resort path."""
        result.acc_target_velocity = min(result.acc_target_velocity, FALLBACK_SPEED)
        info['fallback'] = dict(active=True, required_clearance_m=clearance, speed_cap_mps=FALLBACK_SPEED,
                                source='held' if clearance is None else 'relaxed')
        self.fallback_log.append((self.clock(), clearance))

    def _speed_clearance(self, x, y, yaw, clear):
        """Traffic boxes on the map include stretches a car may drive through (tunnel_map virtual boxes); the speed
        beside them follows the cars' own footprints, not those stretches."""
        if not getattr(self.occupancy, 'virtual', None):
            return clear
        with self.occupancy.footprints_only():
            return self.clearance(x, y, yaw)

    def _add_lane_bounds(self, ego):
        """Called once after the base setup: tabulate the road's lateral edges along the section."""
        start = min(float(ego.position.y), self.base.geometry.route_y[0])-5.
        self.road_y = np.arange(start, self.base.goal_y+self.base.planner.vehicle_length+10., ROAD_STEP)
        low, high = np.full(len(self.road_y), np.nan), np.full(len(self.road_y), np.nan)
        polygon = self.road_boundary._polygon
        for i, y in enumerate(self.road_y):
            centre = float(self.base.global_curve(y))
            probe = LineString([self.section.to_world((centre-ROAD_PROBE, y)), self.section.to_world((centre+ROAD_PROBE, y))])
            with self.road_boundary._predicate_lock:
                cut = probe.intersection(polygon)
            pieces = [cut] if cut.geom_type == 'LineString' else list(getattr(cut, 'geoms', []))
            spans = []
            for piece in pieces:
                if piece.is_empty or piece.geom_type != 'LineString':
                    continue
                xs = [self.section.to_local(point)[0] for point in piece.coords]
                spans.append((min(xs), max(xs)))
            if spans:
                # The span holding the route; at a junction the road splits and only that branch is drivable.
                a, b = min(spans, key=lambda s: 0. if s[0] <= centre <= s[1] else min(abs(s[0]-centre), abs(s[1]-centre)))
                low[i], high[i] = a, b
        self.road_low, self.road_high = low, high

    def _road(self, y):
        low = np.interp(y, self.road_y, np.nan_to_num(self.road_low, nan=np.inf))
        high = np.interp(y, self.road_y, np.nan_to_num(self.road_high, nan=-np.inf))
        return low, high

    def _corridor(self, ys, y0=None, x0=None):
        """Reference-point bounds: the road narrowed by the body half width and the road margin."""
        low, high = self._road(np.asarray(ys, dtype=float))
        inset = self.base.planner.vehicle_half_width+self.margin
        low, high = low+inset, high-inset
        if x0 is not None:
            keep = 1.-np.clip((np.asarray(ys)-y0)/ESCAPE_DISTANCE, 0., 1.)
            low = low-np.maximum(0., low-x0)*keep
            high = high+np.maximum(0., x0-high)*keep
        return low, high

    def _corridor_margin(self, x, y, yaw, y0, centre=None):
        """Signed distance [m] by which the body corners keep the road margin (negative: too close or outside)."""
        x, y, yaw = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float))
        low, high = self._corridor(y) if centre is None else centre
        margin = np.minimum(x-low, high-x)
        c, s = np.cos(yaw), np.sin(yaw)
        planner = self.base.planner
        for along in (0., planner.vehicle_length):
            for side in (-planner.vehicle_half_width, planner.vehicle_half_width):
                cx, cy = x+along*c-side*s, y+along*s+side*c
                road_low, road_high = self._road(cy)
                margin = np.minimum(margin, np.minimum(cx-road_low-self.margin, road_high-self.margin-cx))
        return np.where(y-y0 <= ESCAPE_DISTANCE, np.inf, margin)
