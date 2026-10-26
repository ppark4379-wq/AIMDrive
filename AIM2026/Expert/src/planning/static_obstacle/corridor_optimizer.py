"""Receding-horizon static-section path on the accumulated occupancy map.

The recorded planner commits one maneuver and then checks it pass/fail against every new VE frame, so small obstacle
changes stop the vehicle or leave it too close to swerve. Here the lateral path is re-optimized every tick from the
measured pose on the accumulated map (static_map.py): a slightly larger obstacle shifts the path instead of stopping.

Frame and corridor come from RecordedStaticPlanner (path as x(y) in the section's forward-Y frame, global curve,
recorded drivable corridor). Each tick:
1. Feasibility over stations y_i ahead and lateral samples x_j: circles covering the body keep MIN_CLEARANCE from
   occupied cells, and the body stays inside the recorded corridor.
2. Side choice by dynamic programming: clearance shortfall, distance from the reference and from the previous path.
3. Smoothing by bounded least squares (curvature and its change) inside the chosen free interval.
4. Validation with the path's own heading; speed from curvature, clearance and stopping distance.
The same MIN_CLEARANCE decides planning and stopping, so a valid plan never stops the car at its own pose.
"""
import math
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from ...interfaces.geometry import Point
from ...config.parameters import lattice as p
from ...config.parameters.static_obstacle import (
    MAX_CURVATURE,
    CORRIDOR_STATION_STEP,
    CORRIDOR_HORIZON,
    CORRIDOR_LATERAL_STEP,
    CORRIDOR_MIN_CLEARANCE,
    CORRIDOR_DESIRED_CLEARANCE,
    CORRIDOR_MAX_SLOPE,
    CORRIDOR_BODY_CIRCLES,
    CORRIDOR_WEIGHTS,
    CORRIDOR_LATERAL_ACCEL,
    CORRIDOR_CLEARANCE_SPEED,
    CORRIDOR_STOP_BUFFER,
    CORRIDOR_LANE_HALF_WIDTH,
    CORRIDOR_LANE_EDGE_MARGIN,
)
from ..lattice.structures import CandidatePath, LatticePlanningResult

DENSE_STEP = .25          # output/validation sample spacing along y [m]
VALIDATION_TOLERANCE = .05  # discretization allowance when re-checking the smoothed path [m]
CONTACT_CLEARANCE = .1    # the current pose is this close to an occupied cell: stop
ESCAPE_DISTANCE = 5.      # over this distance the required clearance ramps from the pose's own to MIN_CLEARANCE;
                          # the recorded-corridor check is also waived there (the car is already where it is) [m]
CURVATURE_SPEED_SKIP = 1.5  # the heading join right at the pose does not limit speed [m]
HOLD_TRACKING_ERROR = .4   # the committed path is kept only while the car is this close to it [m]
HOLD_HEADING_ERROR = math.radians(15.)  # ... and heads along it within this angle
MAX_REPAIRS = 3


class StaticCorridorPlanner:
    def __init__(self, base, occupancy):
        self.base = base              # RecordedStaticPlanner: geometry, global curve, recorded corridor
        self.occupancy = occupancy    # StaticObstacleMap: distance(x, y)
        length, half_width = base.planner.vehicle_length, base.planner.vehicle_half_width
        count = CORRIDOR_BODY_CIRCLES
        self.circle_t = (np.arange(count)+.5)*length/count
        self.circle_r = math.hypot(half_width, .5*length/count)
        self.ready = False
        self.previous = None          # (ys, xs) of the last station solution
        self.committed = None         # (yy, xx) dense samples of the last valid path
        self._global_xy = None

    # ------------------------------------------------------------------ geometry
    def goal_y(self):
        g = self.base.geometry
        return self.base.goal_y if self.ready else g.rejoin_point[1]+g.tail_length

    def contains(self, ego):
        g = self.base.geometry
        return (g.active_x[0] < ego.position.x < g.active_x[1]
                and g.route_y[0] < ego.position.y < self.goal_y() and math.sin(ego.yaw) > .5)

    def clearance(self, x, y, yaw):
        """Body clearance [m] to occupied cells for poses (rear reference point, heading)."""
        x, y, yaw = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float))
        c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
        d = self.occupancy.distance(x[..., None]+self.circle_t*c, y[..., None]+self.circle_t*s)
        return d.min(axis=-1)-self.circle_r

    def _corridor(self, ys, y0=None, x0=None):
        """Lateral bounds of the recorded drivable corridor; near the pose they also include the measured x."""
        centers = np.array([curve(ys) for curve in self.base.curves]+[self.base.global_curve(ys)])
        low, high = centers.min(axis=0)-.15, centers.max(axis=0)+.15
        if x0 is not None:
            keep = 1.-np.clip((np.asarray(ys)-y0)/ESCAPE_DISTANCE, 0., 1.)
            low = low-np.maximum(0., low-x0)*keep
            high = high+np.maximum(0., x0-high)*keep
        return low, high

    def _in_corridor(self, x, y, yaw):
        shape = np.shape(x)
        xy = np.column_stack((np.ravel(x), np.ravel(y)))
        return self.base._envelope_mask(xy, np.ravel(yaw), 0.).reshape(shape)

    # ------------------------------------------------------------------ planning steps
    @staticmethod
    def _required(ys, y0, pose_clear):
        """Required clearance per station: a pose that already lost clearance (the map grew) may still move away."""
        escape = min(CORRIDOR_MIN_CLEARANCE, max(CONTACT_CLEARANCE, pose_clear-VALIDATION_TOLERANCE))
        ramp = np.clip((np.asarray(ys)-y0)/ESCAPE_DISTANCE, 0., 1.)
        return escape+(CORRIDOR_MIN_CLEARANCE-escape)*ramp

    def _corridor_ok(self, x, y, yaw, y0):
        return self._in_corridor(x, y, yaw) | (np.asarray(y)-y0 <= ESCAPE_DISTANCE)

    def _feasibility(self, ys, xs, yaw, required, x0):
        low, high = self._corridor(ys, ys[0], x0)
        X, Y = np.meshgrid(xs, ys)
        YAW = np.broadcast_to(yaw[:, None], X.shape)
        clear = self.clearance(X, Y, YAW)
        feasible = ((X >= low[:, None]) & (X <= high[:, None]) & (clear >= required[:, None])
                    & self._corridor_ok(X, Y, YAW, ys[0]))
        return feasible, clear

    def _choose_side(self, xs, x_ref, feasible, clear, x0, previous):
        w = CORRIDOR_WEIGHTS
        steps = int(math.ceil(CORRIDOR_MAX_SLOPE*CORRIDOR_STATION_STEP/CORRIDOR_LATERAL_STEP))
        cost = np.full(len(xs), np.inf)
        cost[int(np.argmin(np.abs(xs-x0)))] = 0.
        back = np.zeros((len(x_ref), len(xs)), dtype=int)
        last = 0
        for i in range(1, len(x_ref)):
            node = (w['clearance']*np.maximum(0., CORRIDOR_DESIRED_CLEARANCE-clear[i])**2
                    + w['reference']*(xs-x_ref[i])**2)
            if previous is not None:
                node = node + w['previous']*(xs-previous[i])**2
            node[~feasible[i]] = np.inf
            best, arg = np.full(len(xs), np.inf), np.zeros(len(xs), dtype=int)
            index = np.arange(len(xs))
            for d in range(-steps, steps+1):
                source = index-d
                ok = (source >= 0) & (source < len(xs))
                value = np.full(len(xs), np.inf)
                value[ok] = cost[source[ok]]+w['slope']*(d*CORRIDOR_LATERAL_STEP/CORRIDOR_STATION_STEP)**2
                better = value < best
                best[better], arg[better] = value[better], source[better]
            cost = best+node
            back[i] = arg
            if not np.isfinite(cost).any():
                break
            last = i
            final = cost
        if last == 0:
            return [], 0
        j = int(np.argmin(final))
        chosen = [j]
        for i in range(last, 1, -1):
            j = back[i, j]
            chosen.append(int(j))
        return chosen[::-1], last          # j for stations 1..last

    @staticmethod
    def _interval(row, j):
        left = right = j
        while left > 0 and row[left-1]:
            left -= 1
        while right < len(row)-1 and row[right+1]:
            right += 1
        return left, right

    def _smooth(self, targets, lower, upper, x0, slope0):
        w, ds, n = CORRIDOR_WEIGHTS, CORRIDOR_STATION_STEP, len(targets)
        # Full sequence X = [x_-1, x_0, v_1..v_n] = S v + c
        S = np.zeros((n+2, n)); S[2:, :] = np.eye(n)
        c = np.zeros(n+2); c[0], c[1] = x0-ds*slope0, x0
        rows, rhs = [np.sqrt(w['target'])*np.eye(n)], [np.sqrt(w['target'])*np.asarray(targets)]
        second = np.array([[1., -2., 1.]])/ds**2
        D2 = np.zeros((n, n+2))
        for k in range(n):
            D2[k, k:k+3] = second
        rows.append(np.sqrt(w['curvature'])*D2@S); rhs.append(-np.sqrt(w['curvature'])*D2@c)
        if n > 1:
            D3 = np.zeros((n-1, n+2))
            for k in range(n-1):
                D3[k, k:k+4] = np.array([-1., 3., -3., 1.])/ds**3
            rows.append(np.sqrt(w['jerk'])*D3@S); rhs.append(-np.sqrt(w['jerk'])*D3@c)
        upper = np.maximum(upper, lower+1e-3)
        A, b = np.vstack(rows), np.concatenate(rhs)
        # BVLS divides by zero on nearly pinned variables; fall back to the trust-region solver, then to the targets.
        for method in ('bvls', 'trf'):
            with np.errstate(divide='ignore', invalid='ignore'):
                solution = lsq_linear(A, b, bounds=(lower, upper), method=method).x
            if np.isfinite(solution).all():
                return np.clip(solution, lower, upper)
        return np.clip(targets, lower, upper)

    def _dense(self, ys, xs, slope0):
        curve = CubicSpline(ys, xs, bc_type=((1, slope0), (2, 0.)))
        yy = np.arange(ys[0], ys[-1]+1e-9, DENSE_STEP)
        xx, slope, second = curve(yy), curve(yy, 1), curve(yy, 2)
        return yy, xx, np.arctan2(1., slope), np.abs(second)/(1+slope*slope)**1.5

    # ------------------------------------------------------------------ main entry
    def run(self, ego, reference_path, global_path):
        """Plan one tick. A numerical failure stops the car for this tick instead of ending the control loop."""
        begin = time.perf_counter()
        try:
            return self._plan(ego, reference_path, global_path, begin)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            ahead = Point(ego.position.x+.5*math.cos(ego.yaw), ego.position.y+.5*math.sin(ego.yaw))
            details = dict(planner='static_corridor', phase='blocked', path_validated=False,
                           error='planning_error: '+str(exc)[:160])
            return self._result([ego.position, ahead], 0., False, details, begin, [], np.zeros(0))

    def _plan(self, ego, reference_path, global_path, begin):
        if not self.ready:
            try:
                self.base._setup(global_path, ego)
            except ValueError:
                return None
            self._add_lane_left_bound(ego)
            self.ready = True
            self._global_xy = np.array([(q.x, q.y) for q in global_path])
        ds = CORRIDOR_STATION_STEP
        y0, x0 = float(ego.position.y), float(ego.position.x)
        end = min(y0+CORRIDOR_HORIZON, self.base.goal_y)
        if end-y0 < 3*ds:
            return None
        ys = y0+ds*np.arange(int((end-y0)/ds)+1)
        x_ref = self.base.global_curve(ys)
        yaw_ref = np.arctan2(1., self.base.global_curve(ys, 1))
        slope0 = float(np.clip(math.cos(ego.yaw)/max(.5, math.sin(ego.yaw)), -CORRIDOR_MAX_SLOPE, CORRIDOR_MAX_SLOPE))
        low, high = self._corridor(ys, y0, x0)
        xs = np.arange(low.min(), high.max()+1e-9, CORRIDOR_LATERAL_STEP)
        pose_clear = float(self.clearance(x0, y0, ego.yaw))
        required = self._required(ys, y0, pose_clear)
        held = self._held_path(ego, pose_clear)
        feasible, clear = self._feasibility(ys, xs, yaw_ref, required, x0)
        previous = None
        if self.previous is not None:
            previous = np.interp(ys, *self.previous, left=np.nan, right=np.nan)
            previous = np.where(np.isfinite(previous), previous, x_ref)
        chosen, last = self._choose_side(xs, x_ref, feasible, clear, x0, previous)
        blocked = last < len(ys)-1
        details = dict(planner='static_corridor', stations=len(ys), feasible_stations=last,
                       min_clearance_m=None, stop_distance_m=None, repairs=0)
        if not chosen:
            if held is not None:
                details['new_plan_error'] = 'no_feasible_station'
                return self._finish(ego, held, False, pose_clear, details, begin, source='previous')
            path, speed, valid = [ego.position, Point(x0, y0+.5)], 0., False
            details.update(phase='blocked', path_validated=False, error='no_feasible_station')
            return self._result(path, speed, valid, details, begin, [], np.zeros(0))
        n = len(chosen)
        intervals = [self._interval(feasible[i+1], j) for i, j in enumerate(chosen)]
        lower = np.array([xs[a] for a, _ in intervals])
        upper = np.array([xs[b] for _, b in intervals])
        # The side choice is quantized to the lateral grid; smooth it so the target does not inject curvature.
        targets = np.convolve(np.pad(xs[np.array(chosen)], 2, mode='edge'), np.ones(5)/5, mode='valid')
        targets = np.clip(targets, lower, upper)
        for repair in range(MAX_REPAIRS+1):
            solution = self._smooth(targets, lower, upper, x0, slope0)
            station_y = ys[:n+1]
            station_x = np.r_[x0, solution]
            yy, xx, yaw, curvature = self._dense(station_y, station_x, slope0)
            dense_clear = self.clearance(xx, yy, yaw)
            clear_ok = dense_clear >= self._required(yy, y0, pose_clear)-VALIDATION_TOLERANCE
            curvature_ok = curvature <= MAX_CURVATURE+1e-6
            corridor_ok = self._corridor_ok(xx, yy, yaw, y0)
            ok = clear_ok & curvature_ok & corridor_ok
            details['violations'] = dict(clearance=int((~clear_ok).sum()), curvature=int((~curvature_ok).sum()),
                                         corridor=int((~corridor_ok).sum()))
            if ok.all() or repair == MAX_REPAIRS:
                break
            # Re-judge the violating stations with the smoothed path's own heading and narrow their interval.
            details['repairs'] = repair+1
            for i in np.unique(np.clip(np.round((yy[~ok]-y0)/ds).astype(int), 1, n)):
                actual = np.interp(ys[i], yy, yaw)
                row = ((xs >= low[i]) & (xs <= high[i])
                       & (self.clearance(xs, np.full(len(xs), ys[i]), np.full(len(xs), actual))
                          >= required[i]+VALIDATION_TOLERANCE)
                       & self._corridor_ok(xs, np.full(len(xs), ys[i]), np.full(len(xs), actual), y0))
                j = int(np.argmin(np.abs(xs-solution[i-1])))
                if not row.any():
                    continue
                if not row[j]:
                    j = int(np.flatnonzero(row)[np.argmin(np.abs(np.flatnonzero(row)-j))])
                a, b = self._interval(row, j)
                lower[i-1], upper[i-1] = max(lower[i-1], xs[a]), min(upper[i-1], xs[b])
                targets[i-1] = np.clip(targets[i-1], lower[i-1], upper[i-1])
        if not ok.all():
            if held is not None:
                # The new solve failed but the committed path is still clear on the current map: keep it.
                details['new_plan_error'] = 'smoothed_path_violation'
                return self._finish(ego, held, False, pose_clear, details, begin, source='previous')
            bad = int(np.flatnonzero(~ok)[0])
            blocked = True
            yy, xx, yaw, curvature, dense_clear = (v[:max(bad, 2)] for v in (yy, xx, yaw, curvature, dense_clear))
            details['error'] = 'smoothed_path_violation'
        if held is not None and blocked and held[0][-1] > yy[-1]:
            details['new_plan_error'] = 'blocked_earlier_than_committed'
            return self._finish(ego, held, False, pose_clear, details, begin, source='previous')
        self.previous = (station_y, station_x)
        if not blocked:
            self.committed = (yy, xx)
        return self._finish(ego, (yy, xx, yaw, curvature, dense_clear), blocked, pose_clear, details, begin)

    def _held_path(self, ego, pose_clear):
        """The committed path ahead of the car, if the car is on it and it is still valid on the current map."""
        if self.committed is None:
            return None
        yy, xx = self.committed
        y0 = float(ego.position.y)
        d = np.hypot(xx-ego.position.x, yy-y0)
        k = int(np.argmin(d))
        if d[k] > HOLD_TRACKING_ERROR or len(yy)-k < 8:
            return None
        yy, xx = yy[k:], xx[k:]
        slope = np.gradient(xx, yy)
        yaw = np.arctan2(1., slope)
        heading_error = abs(math.atan2(math.sin(ego.yaw-yaw[0]), math.cos(ego.yaw-yaw[0])))
        if heading_error > HOLD_HEADING_ERROR:
            return None
        curvature = np.abs(np.gradient(slope, yy))/(1+slope*slope)**1.5
        clear = self.clearance(xx, yy, yaw)
        station = np.r_[0., np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
        # A path the car is already on may continue while it never gets closer than the car is now: beside an
        # obstacle that grew in the map, clearance cannot recover until the obstacle is passed.
        required = np.minimum(self._required(yy, y0, pose_clear),
                              min(CORRIDOR_MIN_CLEARANCE, max(CONTACT_CLEARANCE, pose_clear-VALIDATION_TOLERANCE)))
        ok = ((clear >= required-VALIDATION_TOLERANCE)
              & ((curvature <= MAX_CURVATURE+1e-6) | (station < CURVATURE_SPEED_SKIP))
              & self._corridor_ok(xx, yy, yaw, y0))
        if not ok.all():
            return None
        return yy, xx, yaw, curvature, clear

    def _finish(self, ego, dense, blocked, pose_clear, details, begin, source='new'):
        yy, xx, yaw, curvature, dense_clear = dense
        if source == 'previous':
            stations = np.arange(yy[0], yy[-1], CORRIDOR_STATION_STEP)
            self.previous = (stations, np.interp(stations, yy, xx))
        details['source'] = source
        # Speed: curvature, clearance, stopping before an impassable station.
        station = np.r_[0., np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
        base_speed, gain = CORRIDOR_CLEARANCE_SPEED
        # The first metres only join the measured heading to the plan; the tracker looks ahead past them.
        speed_curvature = np.where(station < CURVATURE_SPEED_SKIP, 0., curvature)
        limit = np.minimum.reduce([
            np.full(len(yy), p.STATIC_MAX_SPEED_KPH/3.6),
            np.sqrt(CORRIDOR_LATERAL_ACCEL/np.maximum(speed_curvature, 1e-6)),
            base_speed+gain*np.maximum(0., np.minimum(dense_clear, 10.)-CORRIDOR_MIN_CLEARANCE)])
        if blocked:
            stop = station[-1]-CORRIDOR_STOP_BUFFER
            limit = np.where(station >= stop, 0., limit)
            details['stop_distance_m'] = float(max(0., stop))
        for i in range(len(limit)-2, -1, -1):
            limit[i] = min(limit[i], math.sqrt(limit[i+1]**2+2*p.STATIC_END_DECELERATION*(station[i+1]-station[i])))
        speed = float(limit[0]) if pose_clear >= CONTACT_CLEARANCE else 0.
        path = [Point(float(a), float(b)) for a, b in zip(xx, yy)]
        if not blocked:
            path += self._global_tail(path[-1])
        details.update(phase='blocked' if blocked else 'follow', path_validated=not blocked,
                       min_clearance_m=float(np.min(dense_clear)) if len(dense_clear) else None,
                       pose_clearance_m=pose_clear, max_curvature=float(np.max(curvature)) if len(curvature) else 0.,
                       speed_policy_reason='corridor')
        return self._result(path, speed, not blocked, details, begin, path, curvature)

    def _add_lane_left_bound(self, ego):
        """All recordings bypass to the right, so the corridor stopped at the lane centre. Add the left half of the
        own lane (map lane width) as one more corridor curve; centre bounds and the body envelope widen together."""
        base = self.base
        offset = CORRIDOR_LANE_HALF_WIDTH-base.planner.vehicle_half_width-CORRIDOR_LANE_EDGE_MARGIN
        if offset <= 0.:
            return
        def lane_left(y, derivative=0, curve=base.global_curve):
            return curve(y, derivative)-(offset if derivative == 0 else 0.)
        base.curves.append(lane_left)
        base._center_bounds_cache = {}
        base._make_envelope(min(ego.position.y, min(d[0, 1] for _, d, _ in base.records))-2)

    def _global_tail(self, last, count=60):
        index = int(np.argmin(np.hypot(self._global_xy[:, 0]-last.x, self._global_xy[:, 1]-last.y)))
        return [Point(float(x), float(y)) for x, y in self._global_xy[index+1:index+1+count]]

    def _result(self, path, speed, valid, details, begin, points, curvature):
        details['replan_ms'] = round(1e3*(time.perf_counter()-begin), 1)
        candidate = CandidatePath(layer='corridor')
        candidate.points, candidate.valid = list(points), valid
        return LatticePlanningResult(path, active=True, mode='static_obstacle', obstacle_detected=True,
                                     acc_override=True, acc_target_velocity=speed, candidates=[candidate],
                                     reason='static_corridor_'+details['phase'],
                                     debug_details={'static_avoidance': details})
