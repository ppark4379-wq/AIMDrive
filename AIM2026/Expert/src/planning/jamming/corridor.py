"""Receding-horizon path for standing obstacles in the Jamming (tunnel) section.

The static-section planner (static_obstacle/corridor_optimizer.py) on the tunnel's accumulated occupancy map
(Expert/tunnel_map.py), in the section's forward-Y frame. Its drivable corridor is the mapped tunnel road (both lanes,
RoadBoundary) instead of the recorded routes: the body corners keep the road's planning margin from its edges.
Moving traffic reaches it only as boxes added to the map and a speed cap (traffic.py).
"""
import numpy as np
import math
from shapely.geometry import LineString

from ..static_obstacle.corridor_optimizer import StaticCorridorPlanner, ESCAPE_DISTANCE
from ...config.parameters.jamming import (TIGHT_GAP_MARGIN, TIGHT_GAP_SPEED, TUNNEL_MAX_SPEED_KPH,
                                          TUNNEL_CLEARANCE_SPEED, TUNNEL_FORCE_RESUME_SPEED_KPH,
                                          YIELD_HORIZON_S, TRACK_STALE_S)
from ...config.parameters.static_obstacle import MAX_CURVATURE, CORRIDOR_MIN_CLEARANCE
from ...interfaces.geometry import Point
from ..lattice.structures import CandidatePath

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

    def _speed_clearance(self, x, y, yaw, clear):
        """Traffic boxes on the map include stretches a car may drive through (tunnel_map virtual boxes); the speed
        beside them follows the cars' own footprints, not those stretches."""
        if not getattr(self.occupancy, 'virtual', None):
            return clear
        with self.occupancy.footprints_only():
            return self.clearance(x, y, yaw)

    def least_risk_restart(self, ego, objects):
        """MORAI last resort: occupancy is a cost, road/steering are constraints.

        Compare equal-length smooth paths, including the previous endpoint.
        Signed body clearance scores current occupancy and observed movers.
        No positive obstacle clearance is required; this is not collision-free
        validation and must never be labelled path_validated by the caller.
        """
        x0, y0 = float(ego.position.x), float(ego.position.y)
        if not self.ready or self.road_y is None or math.sin(ego.yaw) <= .5:
            return None
        length = min(12., self.base.goal_y-y0)
        if length < 3.:
            return None
        yy = np.linspace(y0, y0+length, int(math.ceil(length/.1))+1)
        u = (yy-y0)/length
        blend = 10*u**3-15*u**4+6*u**5
        heading_blend = u-6*u**3+8*u**4-3*u**5
        slope0 = math.cos(ego.yaw)/math.sin(ego.yaw)
        low, high = self._corridor(np.array([yy[-1]]))
        if not np.isfinite([low[0], high[0]]).all() or low[0] > high[0]:
            return None
        reference = self.base.global_curve(yy)
        previous = getattr(self, '_restart_previous', None)
        prior = reference if previous is None else np.interp(yy, *previous)
        targets = np.unique(np.r_[np.linspace(low[0], high[0], 17),
                                  np.clip([x0, reference[-1], prior[-1]], low[0], high[0])])
        choices = []
        for target in targets:
            xx = x0+slope0*length*heading_blend+(target-x0)*blend
            dx = slope0*(1-18*u**2+32*u**3-15*u**4)+(target-x0)/length*(30*u**2-60*u**3+30*u**4)
            ddx = slope0/length*(-36*u+96*u**2-60*u**3)+(target-x0)/length**2*(60*u-180*u**2+120*u**3)
            yaw = np.arctan2(np.ones_like(dx), dx)
            curvature = np.abs(ddx)/(1+dx*dx)**1.5
            # Do NOT apply the normal planner's first-five-metres escape
            # exemption: every body corner must remain inside the road here.
            road_margin = self._corridor_margin(xx, yy, yaw, -1e9)
            if (not np.isfinite(xx).all() or np.any(np.isnan(road_margin))
                    or np.any(road_margin < 0.) or np.max(curvature) > MAX_CURVATURE):
                continue
            clear = np.asarray(self.clearance(xx, yy, yaw), dtype=float)
            stations = np.r_[0., np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
            times = np.minimum(stations/max(.1, TUNNEL_FORCE_RESUME_SPEED_KPH/3.6), YIELD_HORIZON_S)
            cx = xx[:, None]+self.circle_t*np.cos(yaw[:, None])
            cy = yy[:, None]+self.circle_t*np.sin(yaw[:, None])
            vehicle_conflict = False
            for obj in objects:
                protected_vehicle = obj.get('size') == 'vehicle'
                if obj['age'] > TRACK_STALE_S and not protected_vehicle:
                    continue
                vx, vy = obj['velocity']
                age = times+max(0., obj['age'])
                rx = cx-(obj['x']+vx*age)[:, None]
                ry = cy-(obj['y']+vy*age)[:, None]
                c, s = math.cos(obj['heading']), math.sin(obj['heading'])
                qx = np.abs(c*rx+s*ry)-obj['length']/2.
                qy = np.abs(-s*rx+c*ry)-obj['width']/2.
                signed = (np.hypot(np.maximum(qx, 0.), np.maximum(qy, 0.))
                          +np.minimum(np.maximum(qx, qy), 0.)-self.circle_r)
                if protected_vehicle and np.any(signed < CORRIDOR_MIN_CLEARANCE):
                    vehicle_conflict = True
                    break
                clear = np.minimum(clear, signed.min(axis=1))
            if vehicle_conflict or np.isnan(clear).any():
                continue
            clear = np.minimum(clear, 10.)
            overlap = float(np.mean(np.maximum(0., -clear)**2))
            shortfall = float(np.mean(np.maximum(0., .3-clear)**2))
            smooth = float(np.mean((xx-prior)**2)+.1*np.mean((xx-reference)**2)+np.mean(curvature**2))
            score = (round(overlap, 4), round(shortfall, 4), -round(float(clear.min()), 2), smooth)
            choices.append((score, xx, curvature, float(clear.min())))
        if not choices:
            return None
        score, xx, curvature, clearance = min(choices, key=lambda c: c[0])
        self._restart_previous = (yy.copy(), xx.copy())
        path = [Point(float(x), float(y)) for x, y in zip(xx, yy)]
        candidate = CandidatePath(layer='tunnel_last_resort')
        candidate.points, candidate.valid = path, clearance >= 0.
        candidate.obstacle_cost = score[0]+score[1]
        return path, candidate, dict(geometry_validated=True, obstacle_clearance_required=False,
                                    min_clearance_m=clearance, overlap_cost=score[0],
                                    clearance_cost=score[1], candidates=len(choices),
                                    max_curvature=float(curvature.max()))

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
