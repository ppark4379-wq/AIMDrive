"""Receding-horizon path for standing obstacles in the Jamming (tunnel) section.

The static-section planner (static_obstacle/corridor_optimizer.py) on the tunnel's accumulated occupancy map
(Expert/tunnel_map.py), in the section's forward-Y frame. Its drivable corridor is the mapped tunnel road (both lanes,
RoadBoundary) instead of the recorded routes: the body corners keep the road's planning margin from its edges.
Moving traffic reaches it only as boxes added to the map and a speed cap (traffic.py).
"""
import numpy as np
from shapely.geometry import LineString

from ..static_obstacle.corridor_optimizer import StaticCorridorPlanner, ESCAPE_DISTANCE
from ...config.parameters.jamming import (TIGHT_GAP_MARGIN, TIGHT_GAP_SPEED, TUNNEL_MAX_SPEED_KPH,
                                          TUNNEL_CLEARANCE_SPEED, FALLBACK_STALL_S, FALLBACK_STALL_SPEED,
                                          FALLBACK_CLEARANCES, FALLBACK_SPEED, FALLBACK_RELEASE_TICKS,
                                          FALLBACK_CONTACT_STALL_S, FALLBACK_CONTACT_STALL_SPEED,
                                          CURVATURE_OVERRIDE)

ROAD_STEP = .25           # longitudinal spacing of the tabulated road edges [m]
ROAD_PROBE = 15.          # lateral half-length of the probe line that cuts the road polygon [m]


class TunnelCorridorPlanner(StaticCorridorPlanner):
    REASON = 'jamming_corridor_'
    # Objects are seen too small from afar and grow on the map as the car closes in: tight gaps are entered slowly.
    TIGHT_MARGIN = TIGHT_GAP_MARGIN
    TIGHT_SPEED = TIGHT_GAP_SPEED
    MAX_SPEED = TUNNEL_MAX_SPEED_KPH/3.6
    CLEARANCE_SPEED = TUNNEL_CLEARANCE_SPEED
    # Last resort when blocked (jamming.py 5-1); the logic is the static-section planner's
    FALLBACK_STALL_S = FALLBACK_STALL_S
    FALLBACK_STALL_SPEED = FALLBACK_STALL_SPEED
    FALLBACK_CLEARANCES = FALLBACK_CLEARANCES
    FALLBACK_SPEED = FALLBACK_SPEED
    FALLBACK_CONTACT_STALL_S = FALLBACK_CONTACT_STALL_S
    FALLBACK_CONTACT_STALL_SPEED = FALLBACK_CONTACT_STALL_SPEED
    FALLBACK_RELEASE_TICKS = FALLBACK_RELEASE_TICKS
    CURVATURE_OVERRIDE = CURVATURE_OVERRIDE

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
