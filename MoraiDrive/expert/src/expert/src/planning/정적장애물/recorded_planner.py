"""Precompute one complete route through a recorded, northbound static zone.

Recordings are immutable. The envelope is empirical, NOT a surveyed road map.
All connectors and the global-route return undergo the same footprint checks.
"""

import csv
import heapq
import json
import math
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from ...localization.point import Point
from ..lattice_planning import parameters as p
from ..lattice_planning.structures import CandidatePath, LatticePlanningResult


ROOT = Path(__file__).resolve().parent
STEP = 0.2
MAX_CURVATURE = p.STATIC_MAX_CURVATURE
GLOBAL_REJOIN_POINT = (-59.97, -116.70, 28.37)
RETURN_WINDOW = 20.0
RETURN_MAX_CURVATURE = 0.12
TRACKING_SPEED = 5.0  # m/s, 2; leave steering-actuator settling time in this tight corridor.


def load_routes(root=ROOT):
    names = json.loads((root / "routes.json").read_text(encoding="utf-8"))["routes"]
    routes, report = [], []
    for name in names:
        if Path(name).name != name or not name.endswith(".csv"):
            raise ValueError("Invalid recording filename")
        with (root / "\ud6c4\ubcf4\uc804\uc5ed\uacbd\ub85c" / name).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        data = np.array([[float(r[k]) for k in ("x", "y", "yaw_rad")] for r in rows])
        if len(data) < 10 or not np.isfinite(data).all():
            raise ValueError(name + ": insufficient or non-finite data")
        delta = np.diff(data[:, :2], axis=0)
        distance = np.linalg.norm(delta, axis=1)
        forward = delta[:, 0] * np.cos(data[:-1, 2]) + delta[:, 1] * np.sin(data[:-1, 2])
        reverse = np.where(forward < -0.03)[0]
        trim = int(reverse[-1] + 1) if len(reverse) else 0
        if trim and distance[:trim].sum() > 3.0:
            raise ValueError(name + ": reverse segment is not an initial positioning error")
        data = data[trim:]
        if len(data) < 10 or np.max(np.linalg.norm(np.diff(data[:, :2], axis=0), axis=1)) > 1.0:
            raise ValueError(name + ": discontinuous recording")
        # This library belongs to this northbound segment, not arbitrary map links.
        keep = np.r_[True, np.diff(data[:, 1]) > 0.02]
        data = data[keep]
        if np.any(np.diff(data[:, 1]) <= 0) or data[-1, 1] - data[0, 1] < 30:
            raise ValueError(name + ": non-forward or unfinished recording")
        indices = np.unique(np.r_[np.arange(0, len(data), 4), len(data) - 1])
        knots = data[indices]
        if np.any(np.sin(knots[:, 2]) < 0.5):
            raise ValueError(name + ": heading outside the northbound corridor")
        curve = CubicHermiteSpline(knots[:, 1], knots[:, 0],
                                  np.cos(knots[:, 2]) / np.sin(knots[:, 2]))
        routes.append((name, data, curve))
        report.append(dict(name=name, recorded_points=len(rows), trimmed_prefix_points=trim,
                           start=data[0, :2].tolist(), end=data[-1, :2].tolist(),
                           length_m=float(np.linalg.norm(np.diff(data[:, :2], axis=0), axis=1).sum())))
    return routes, report


def connector(a, b):
    """Quintic x(y) matching position, heading and curvature at both ends."""
    y0, x0, d0, dd0 = a
    y1, x1, d1, dd1 = b
    length = y1 - y0
    if length <= 0:
        raise ValueError("Connector must progress forward")
    c = np.array([x0, d0 * length, dd0 * length * length / 2, 0., 0., 0.])
    c[3:] = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]),
        [x1 - c[:3].sum(), d1 * length - c[1] - 2*c[2], dd1*length*length - 2*c[2]])
    def evaluate(y, derivative=0):
        coefficients = np.polynomial.polynomial.polyder(c, derivative)
        return np.polynomial.polynomial.polyval((np.asarray(y)-y0)/length, coefficients) / length**derivative
    return evaluate


class RecordedStaticPlanner:
    def __init__(self, planner, root=ROOT, background=False):
        self.planner = planner
        self.executor = ThreadPoolExecutor(max_workers=1) if background else None
        self.pending = None
        self.error = None
        try:
            self.records, self.audit = load_routes(root)
        except (OSError, ValueError, KeyError) as exc:
            self.records, self.audit = [], []
            self.error = str(exc)
        self.reset()

    def reset(self):
        self.path = []
        self.progress = 0
        self.done = False
        self.route = []
        self.selected_names = []
        self.known = []
        self.last_attempt = -float("inf")
        self.curves = []
        self.goal_y = None
        self.rejoin_y = None
        self.plan_count = 0
        self.rejections = dict(curvature=0, envelope=0, collision=0)
        self.search_progress = 0
        self.handoff_y = None
        self.return_start_y = None

    @property
    def active(self):
        return bool(self.path)

    def contains(self, ego):
        if self.done and ego.position.y < -158:
            self.reset()
        return (not self.done and -65 < ego.position.x < -44
                and -164 < ego.position.y < GLOBAL_REJOIN_POINT[1] and math.sin(ego.yaw) > 0.5)

    def past_rejoin(self, ego):
        threshold = self.handoff_y if self.done and self.handoff_y is not None else GLOBAL_REJOIN_POINT[1]
        return (ego.position.y >= threshold
                and abs(ego.position.x-GLOBAL_REJOIN_POINT[0]) <= 3.
                and math.sin(ego.yaw) > .5)

    def _setup(self, global_path, ego):
        points = np.array([(v.x, v.y) for v in global_path
                           if -180 < v.y < -85 and -65 < v.x < -54])
        if len(points) < 10:
            raise ValueError("global return path unavailable")
        points = points[np.r_[True, np.diff(points[:, 1]) > 0.01]]
        if np.any(np.diff(points[:, 1]) <= 0):
            raise ValueError("global return path is not forward")
        self.rejoin_y = GLOBAL_REJOIN_POINT[1]
        self.goal_y = self.rejoin_y + 10.0
        if points[-1, 1] < self.goal_y:
            raise ValueError("global preview ends before complete return")
        self.route = list(global_path)
        self.global_curve = CubicSpline(points[:, 1], points[:, 0], bc_type="natural")
        self.global_x = lambda y: self.global_curve(y)
        self.curves = []
        for name, data, curve in self.records:
            start, end = data[0, 1], data[-1, 1]
            if end >= self.rejoin_y:
                raise ValueError(name + ": recording extends beyond the configured global handoff")
            global_slope = float(self.global_curve(self.rejoin_y, 1))
            tail = connector((end, float(curve(end)), float(curve(end, 1)), float(curve(end, 2))),
                             self._state(self.global_curve, self.rejoin_y))
            def extended(y, derivative=0, c=curve, t=tail, s=start, e=end, slope=global_slope):
                y = np.asarray(y)
                result = np.where(y <= e, c(np.maximum(s, y), derivative), t(y, derivative))
                before = float(c(s)) + float(c(s, 1)) * (y-s) if derivative == 0 else (float(c(s, 1)) if derivative == 1 else 0.)
                result = np.where(y < s, before, result)
                after = self.global_curve(y, derivative)
                return np.where(y > self.rejoin_y, after, result)
            self.curves.append(extended)
        self._make_envelope(min(ego.position.y, min(d[0, 1] for _, d, _ in self.records)) - 2)

    def _make_envelope(self, start):
        # Bounds come from recorded body sweeps, never a guessed extra lane width.
        spacing = .05
        self.envelope_y = np.arange(start-2, self.goal_y + self.planner.vehicle_length + 3, spacing)
        low, high = np.full(len(self.envelope_y), np.inf), np.full(len(self.envelope_y), -np.inf)
        ys = np.linspace(start, self.goal_y, int(math.ceil((self.goal_y-start)/spacing))+1)
        for curve in self.curves + [self._global_return_corridor]:
            xs, yaw = curve(ys), np.arctan2(1., curve(ys, 1))
            corners = []
            for along, side in ((-.1, -self.planner.vehicle_half_width-.1),
                                (self.planner.vehicle_length+.1, -self.planner.vehicle_half_width-.1),
                                (self.planner.vehicle_length+.1, self.planner.vehicle_half_width+.1),
                                (-.1, self.planner.vehicle_half_width+.1)):
                corners.append((xs+along*np.cos(yaw)-side*np.sin(yaw),
                                ys+along*np.sin(yaw)+side*np.cos(yaw)))
            # Intersect all four OBB edges with each scanline. Sparse side samples
            # missed the front edge on bends and carved false notches in the bounds.
            for (ax, ay), (bx, by) in zip(corners, corners[1:]+corners[:1]):
                first = np.ceil((np.minimum(ay, by)-self.envelope_y[0])/spacing).astype(int)
                last = np.floor((np.maximum(ay, by)-self.envelope_y[0])/spacing).astype(int)
                for offset in range(max(0, int(np.max(last-first)))+1):
                    index = first+offset
                    valid = (index <= last) & (index >= 0) & (index < len(low)) & (np.abs(by-ay) > 1e-9)
                    idx = index[valid]
                    fraction = (self.envelope_y[idx]-ay[valid])/(by[valid]-ay[valid])
                    x = ax[valid]+fraction*(bx[valid]-ax[valid])
                    np.minimum.at(low, idx, x)
                    np.maximum.at(high, idx, x)
        valid = np.isfinite(low)
        self.envelope_low = np.interp(self.envelope_y, self.envelope_y[valid], low[valid])
        self.envelope_high = np.interp(self.envelope_y, self.envelope_y[valid], high[valid])

    def _envelope_clear(self, xy, yaw, body_margin=0.):
        centers = np.array([c(xy[:, 1]) for c in self.curves + [self._global_return_corridor]])
        if np.any(xy[:, 0] < centers.min(axis=0)-.15) or np.any(xy[:, 0] > centers.max(axis=0)+.15):
            return False
        for along in (-body_margin, self.planner.vehicle_length+body_margin):
            for side in (-self.planner.vehicle_half_width-body_margin, self.planner.vehicle_half_width+body_margin):
                x = xy[:, 0] + along*np.cos(yaw) - side*np.sin(yaw)
                y = xy[:, 1] + along*np.sin(yaw) + side*np.cos(yaw)
                if np.any(x < np.interp(y, self.envelope_y, self.envelope_low)-.15) or np.any(x > np.interp(y, self.envelope_y, self.envelope_high)+.15):
                    return False
        return True

    def _global_return_corridor(self, y, derivative=0):
        # Only the final return window may use the existing global-route corridor.
        y = np.asarray(y)
        return np.where(y >= self.rejoin_y-RETURN_WINDOW,
                        self.global_curve(y, derivative), self.curves[0](y, derivative))

    def _smooth_return(self, complete, obstacles):
        """Replace only the recorded tail, before driving, with a validated C2 return."""
        self.handoff_y = self.rejoin_y
        self.return_start_y = self.rejoin_y
        route_curve = CubicSpline(complete[:, 1], complete[:, 0])
        start_min = max(float(complete[0, 1])+1., self.rejoin_y-RETURN_WINDOW)
        # The rear of the car must already have cleared the local static cluster.
        for obj in obstacles:
            if -75 < obj.position.x < -40 and complete[0, 1] <= obj.position.y < self.rejoin_y:
                radius = .5*math.hypot(max(obj.size_x, 1.), max(obj.size_y, 1.))
                start_min = max(start_min, obj.position.y+radius+self.planner.collision_margin+.2)
        options = []
        for start in np.arange(start_min, self.rejoin_y-3.9, .5):
            offset = float(route_curve(start)-self.global_x(start))
            for end in np.unique(np.r_[np.arange(start+4., self.rejoin_y, .5), self.rejoin_y]):
                curve = connector(self._state(route_curve, start), self._state(self.global_curve, end))
                ys = np.linspace(start, end, int(math.ceil((end-start)/.1))+1)
                curvature = np.abs(curve(ys,2))/(1+curve(ys,1)**2)**1.5
                residual = curve(ys)-self.global_x(ys)
                if (np.max(curvature) > RETURN_MAX_CURVATURE
                        or np.max(np.abs(residual)) > abs(offset)+.03
                        or np.any(residual*np.sign(offset) < -.03)):
                    continue
                edge = self._edge(curve, start, end, obstacles, body_margin=.2)
                tail = self._edge(self.global_curve, end, self.goal_y, obstacles)
                if edge is not None and tail is not None:
                    options.append((end, float(np.max(curvature)), start, edge, tail))
        if not options:
            return complete
        end, _, start, edge, tail = min(options, key=lambda item: (item[0], item[1]))
        result = np.vstack((complete[complete[:,1] < start-1e-6], edge, tail[1:]))
        points = [Point(x,y) for x,y in result]
        if self.planner._max_curvature(points) > MAX_CURVATURE:
            return complete
        self.return_start_y, self.handoff_y = float(start), float(end)
        return result

    def _early_handoff_ready(self, ego, reference_path, obstacles):
        if (not self.active or self.handoff_y is None or ego.position.y < self.handoff_y
                or len(reference_path) < 2):
            return False
        tangent = math.atan2(1., float(self.global_curve(ego.position.y, 1)))
        heading_error = abs(math.atan2(math.sin(ego.yaw-tangent), math.cos(ego.yaw-tangent)))
        forward = [point for point in reference_path if ego.position.y-.2 <= point.y <= self.goal_y]
        if len(forward) < 2:
            return False
        yaw = np.array([self.planner._path_heading(forward, i) for i in range(len(forward))])
        return (heading_error < math.radians(10.)
                and self.planner._distance_to_path(ego.position, reference_path) < .3
                and self.collision_free(np.array([ego.position]), np.array([ego.yaw]), obstacles or self.known)
                and self.collision_free(np.array(forward), yaw, obstacles or self.known, .12))

    def collision_free(self, xy, yaw, obstacles, extra=0.0):
        if not obstacles:
            return True
        xy, yaw = np.asarray(xy), np.asarray(yaw)
        forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
        left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        center = xy + .5*self.planner.vehicle_length*forward
        margin = self.planner.collision_margin + p.STATIC_TRACKING_MARGIN + extra
        length, width = .5*self.planner.vehicle_length+margin, self.planner.vehicle_half_width+margin
        for obj in obstacles:
            if not all(math.isfinite(v) for v in (obj.position.x, obj.position.y, obj.heading_deg, obj.size_x, obj.size_y)):
                return False
            angle = math.radians(obj.heading_deg)
            of = np.array([math.cos(angle), math.sin(angle)])
            ol = np.array([-of[1], of[0]])
            ox = .5*(obj.size_x if obj.size_x > 0 else 1.)
            oy = .5*(obj.size_y if obj.size_y > 0 else 1.)
            delta = np.array([obj.position.x, obj.position.y]) - center
            overlap = np.ones(len(xy), dtype=bool)
            for axis in (forward, left, np.broadcast_to(of, forward.shape), np.broadcast_to(ol, forward.shape)):
                radius = length*np.abs(np.sum(forward*axis, axis=1)) + width*np.abs(np.sum(left*axis, axis=1)) + ox*np.abs(axis @ of) + oy*np.abs(axis @ ol)
                overlap &= np.abs(np.sum(delta*axis, axis=1)) <= radius
            if np.any(overlap):
                return False
        return True

    def _edge(self, curve, y0, y1, obstacles, body_margin=0.):
        ys = np.linspace(y0, y1, max(2, int(math.ceil((y1-y0)/.1))+1))
        slope = curve(ys, 1)
        curvature = np.abs(curve(ys, 2)) / (1+slope*slope)**1.5
        if np.max(curvature) > MAX_CURVATURE:
            self.rejections['curvature'] += 1
            return None
        xy, yaw = np.column_stack((curve(ys), ys)), np.arctan2(1., slope)
        if not np.isfinite(xy).all() or not self._envelope_clear(xy, yaw, body_margin=body_margin):
            self.rejections['envelope'] += 1
            return None
        # Inflate by half the maximum translation/rotation between samples.
        step = float(np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        sweep = .5*step*(1+self.planner.vehicle_length*float(np.max(curvature)))
        if not self.collision_free(xy, yaw, obstacles, sweep):
            self.rejections['collision'] += 1
            return None
        return xy

    @staticmethod
    def _state(curve, y):
        return (y, float(curve(y)), float(curve(y, 1)), float(curve(y, 2)))

    def _search(self, ego, obstacles):
        start = float(ego.position.y)
        for i, curve in enumerate(self.curves):
            join = min(start+6., self.rejoin_y)
            if join <= start:
                continue
            entry = connector((start, ego.position.x, math.cos(ego.yaw)/max(.5, math.sin(ego.yaw)), 0.), self._state(curve, join))
            first = self._edge(entry, start, join, obstacles)
            rest = self._edge(curve, join, self.goal_y, obstacles)
            if first is not None and rest is not None:
                self.selected_names = [self.records[i][0]]
                return np.vstack((first, rest[1:]))
        # Every graph path must reach the global route at the handoff itself.
        # A cross-route connector must not carry a lateral offset beyond it.
        levels = np.unique(np.r_[np.arange(start, self.goal_y, 3.), self.rejoin_y, self.goal_y])
        source = (-1, 0)
        queue, costs, previous = [(0., source)], {source: 0.}, {}
        end = None
        edge_cache = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            route_id, level = node
            self.search_progress = max(self.search_progress, level)
            if level == len(levels)-1:
                end = node
                break
            for advance in (1, 2, 3, 4):
                target_level = level+advance
                if target_level >= len(levels):
                    continue
                if levels[level] < self.rejoin_y < levels[target_level]:
                    continue
                for target_id, curve in enumerate(self.curves):
                    if route_id == target_id and advance != 1:
                        continue
                    target = (target_id, target_level)
                    key = (node, target)
                    if key not in edge_cache:
                        if route_id == target_id:
                            transfer = curve
                        else:
                            a = self._state(self.curves[route_id], levels[level]) if route_id >= 0 else (
                                start, ego.position.x, math.cos(ego.yaw)/max(.5, math.sin(ego.yaw)), 0.)
                            transfer = connector(a, self._state(curve, levels[target_level]))
                        edge_cache[key] = self._edge(transfer, levels[level], levels[target_level], obstacles)
                    edge = edge_cache[key]
                    if edge is None:
                        continue
                    length = float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum())
                    new_cost = cost+length+(1.5 if route_id >= 0 and route_id != target_id else 0.)
                    if new_cost < costs.get(target, float("inf")):
                        costs[target], previous[target] = new_cost, (node, edge)
                        heapq.heappush(queue, (new_cost, target))
        if end is None:
            return None
        pieces, names = [], []
        while end != source:
            names.append(self.records[end[0]][0])
            end, edge = previous[end]
            pieces.append(edge)
        self.selected_names = list(dict.fromkeys(reversed(names)))
        pieces.reverse()
        return np.vstack([pieces[0]]+[piece[1:] for piece in pieces[1:]])

    def _result(self, points, valid, phase, error=None):
        candidate = CandidatePath(layer="recorded")
        candidate.points, candidate.valid = points, valid
        curvature = self.planner._max_curvature(points) if len(points) >= 3 else 0.
        speed = min(p.STATIC_MAX_SPEED_KPH/3.6, math.sqrt(p.STATIC_LATERAL_ACCEL/max(curvature, .001))) if valid else 0.
        if self.path:
            remaining = sum(a.distance(b) for a,b in zip(points, points[1:]))
            speed = min(speed, math.sqrt(2*p.STATIC_END_DECELERATION*max(0., remaining-1.)))
            # Falling path curvature does not mean the actual body has straightened.
            # Keep a stable tracking speed until the aligned global handoff.
            speed = min(speed, TRACKING_SPEED)
        return LatticePlanningResult(points, active=bool(self.path), mode="static_obstacle",
            obstacle_detected=True, acc_override=True, acc_target_velocity=speed,
            candidates=[candidate], reason="static_recorded_"+phase,
            debug_details={"static_avoidance": dict(phase=phase, path_validated=valid,
                planner="recorded_route_graph", boundary_source="recorded_swept_envelope_not_map",
                selected_routes=self.selected_names, recording_audit=self.audit, error=error,
                return_start_y=self.return_start_y, handoff_y=self.handoff_y,
                preplanned_complete=bool(self.path), plan_count=self.plan_count)})

    def plan(self, ego, reference_path, obstacles, prediction_path=None):
        if ((not self.active and self.past_rejoin(ego) and len(reference_path) >= 2)
                or self._early_handoff_ready(ego, reference_path, obstacles)):
            # Do not restart avoidance here, even if no maneuver was captured.
            if self.pending is not None:
                self.pending.cancel()
                self.pending = None
            self.path = []
            self.done = True
            return LatticePlanningResult(reference_path, mode="static_obstacle", reason="static_recorded_rejoined")
        if self.error:
            return self._result(reference_path, False, "library_error", self.error)
        if not self.active:
            if self.pending is not None:
                if not self.pending.done():
                    return self._result(reference_path, False, "planning")
                try:
                    complete = self.pending.result()
                except Exception as exc:
                    self.pending = None
                    return self._result(reference_path, False, "planning_error", str(exc))
                self.pending = None
                self.last_attempt = time.monotonic()
                if complete is None:
                    return self._result(reference_path, False, "no_complete_route")
                self.path = [Point(x,y) for x,y in complete]
            elif self.executor is not None:
                if abs(ego.velocity) > .3:
                    return self._result(reference_path, False, "settle_at_entry")
                if time.monotonic()-self.last_attempt >= 1.:
                    self.last_attempt = time.monotonic()
                    route = list(prediction_path if prediction_path is not None else reference_path)
                    self.pending = self.executor.submit(self._prepare, ego, route, list(obstacles))
                return self._result(reference_path, False, "planning")
        if not self.active:
            if time.monotonic()-self.last_attempt < 1.:
                return self._result(reference_path, False, "no_complete_route")
            self.last_attempt = time.monotonic()
            try:
                complete = self._prepare(ego, prediction_path if prediction_path is not None else reference_path, obstacles)
            except ValueError as exc:
                return self._result(reference_path, False, "no_global_return", str(exc))
            if complete is None:
                return self._result(reference_path, False, "no_complete_route")
            self.path = [Point(x,y) for x,y in complete]
        xy = np.array(self.path)
        nearest = int(np.argmin(np.linalg.norm(xy[max(0,self.progress-2):]-np.asarray(ego.position), axis=1)))+max(0,self.progress-2)
        error = ego.position.distance(self.path[nearest])
        if error > p.STATIC_RESET_DISTANCE or ego.position.y < self.path[0].y-3:
            self.reset()
            return self._result(reference_path, False, "reset_after_teleport")
        self.progress = max(self.progress, nearest)
        points = self.path[max(0, self.progress-1):]
        if obstacles:
            self.known = list(obstacles)
        yaw = np.array([self.planner._path_heading(points, i) for i in range(len(points))]) if len(points) >= 2 else np.array([])
        failure = None
        if len(points) < 3 or error >= .75:
            failure = 'tracking_error'
        elif not self._envelope_clear(np.array([ego.position]), np.array([ego.yaw])):
            failure = 'recorded_envelope'
        elif not self.collision_free(np.array([ego.position]), np.array([ego.yaw]), self.known):
            failure = 'ego_collision'
        elif not self.collision_free(np.array(points), yaw, self.known, .12):
            failure = 'remaining_path_collision'
        return self._result(points, failure is None, "follow" if failure is None else "blocked", failure)

    def _prepare(self, ego, route, obstacles):
        self._setup(route, ego)
        self.known = list(obstacles)
        self.plan_count += 1
        complete = self._search(ego, obstacles)
        return self._smooth_return(complete, obstacles) if complete is not None else None
