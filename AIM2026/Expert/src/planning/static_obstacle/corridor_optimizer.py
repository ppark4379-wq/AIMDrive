"""Receding-horizon static-section path on the accumulated occupancy map.

The recorded planner commits one maneuver and then checks it pass/fail against every new VE frame, so small obstacle
changes stop the vehicle or leave it too close to swerve. Here the lateral path is re-optimized every tick from the
measured pose on the accumulated map (static_map.py): a slightly larger obstacle shifts the path instead of stopping.

Frame and corridor come from RecordedStaticPlanner (path as x(y) in the section's forward-Y frame, global curve,
recorded drivable corridor, widened by the map lanes). Each tick:
1. Feasibility over stations y_i ahead, lateral samples x_j and the heading of each DP step: circles covering the
   body keep MIN_CLEARANCE from occupied cells, and the body stays inside the corridor.
2. Side choice by dynamic programming: clearance shortfall, distance from the reference and from the previous path.
3. Smoothing by bounded least squares (curvature and its change) inside the chosen free interval.
4. Validation with the path's own heading. Violated margins are linearized in the station variables and added to
   the smoothing, a few rounds. Speed from curvature, clearance and stopping distance.
The same MIN_CLEARANCE decides planning and stopping, so a valid plan never stops the car at its own pose.
Last resort: blocked and standing (or creeping) for FALLBACK_STALL_S, the same solve is retried with the required
clearance lowered step by step (the corridor condition stays) and the car drives on slowly.
"""
from collections import deque
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from ...interfaces.geometry import Point
from ...vehicle_state import VehicleState
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
    CORRIDOR_MAP_LINKS,
    CORRIDOR_FALLBACK_STALL_S,
    CORRIDOR_FALLBACK_STALL_SPEED,
    CORRIDOR_FALLBACK_CLEARANCES,
    CORRIDOR_FALLBACK_SPEED,
    CORRIDOR_FALLBACK_CONTACT_STALL_S,
    CORRIDOR_FALLBACK_CONTACT_STALL_SPEED,
    CORRIDOR_FALLBACK_RELEASE_TICKS,
)
from ..lattice.structures import CandidatePath, LatticePlanningResult

DENSE_STEP = .25          # output/validation sample spacing along y [m]
VALIDATION_TOLERANCE = .05  # discretization allowance when re-checking the smoothed path [m]
CONTACT_CLEARANCE = .1    # floor of the escape clearance asked of a path starting this close to an occupied cell
ESCAPE_DISTANCE = 5.      # over this distance the required clearance ramps from the pose's own to MIN_CLEARANCE;
                          # the recorded-corridor check is also waived there (the car is already where it is) [m]
CURVATURE_SPEED_SKIP = 1.5  # the heading join right at the pose does not limit speed [m]
SCREEN_SLACK = .22          # the map's cheap bound is short by at most 1.5 cell diagonals (0.21 m at 0.1 m cells) [m]
HOLD_TRACKING_ERROR = .4   # the committed path is kept only while the car is this close to it [m]
HOLD_HEADING_ERROR = math.radians(15.)  # ... and heads along it within this angle
RESOLVE_EVERY = 6         # a still-valid committed path is kept for up to this many ticks between full solves
RESOLVE_SHORTFALL = 20.   # ... while it reaches within this distance of the horizon end [m]
SKIP_TRACKING_ERROR = .1  # ... and the car follows it this closely [m]; otherwise the path restarts from the pose
SKIP_HEADING_ERROR = math.radians(3.)
MAX_REPAIRS = 5
RETRY_WIDEN = .3          # widening of the retry's boxes beyond the grid's feasible cells [m]
REPAIR_MARGIN = .03       # a violated margin is linearized and asked to come out this far on the safe side [m]
CONSTRAINT_WEIGHT = 1e4   # weight of a linearized margin against the smoothing terms (unit-gradient rows)
CURVATURE_HOLD = .8       # with a curvature violation, points above this fraction of the limit are held too
GRADIENT_STEP = .02      # finite-difference step for margin gradients in x [m] and slope [m/m]


class StaticCorridorPlanner:
    REASON = 'static_corridor_'    # result reason prefix; + phase
    TIGHT_MARGIN = 0.            # path points within this of the minimum clearance are a tight gap [m]; 0: off
    TIGHT_SPEED = math.inf       # speed cap in a tight gap [m/s]
    MAX_SPEED = None             # speed cap [m/s]; None: the static-obstacle limit
    CLEARANCE_SPEED = CORRIDOR_CLEARANCE_SPEED   # (speed at the minimum clearance [m/s], gain per metre more)
    # Last resort when blocked (static_obstacle.py CORRIDOR_FALLBACK_*; the tunnel sets its own)
    FALLBACK_STALL_S = CORRIDOR_FALLBACK_STALL_S             # blocked and below FALLBACK_STALL_SPEED this long [s]; 0: off
    FALLBACK_STALL_SPEED = CORRIDOR_FALLBACK_STALL_SPEED     # [m/s]
    FALLBACK_CLEARANCES = CORRIDOR_FALLBACK_CLEARANCES       # required clearance per stage, tried in order [m]
    FALLBACK_SPEED = CORRIDOR_FALLBACK_SPEED                 # speed cap on a last-resort path [m/s]
    FALLBACK_CONTACT_STALL_S = CORRIDOR_FALLBACK_CONTACT_STALL_S         # full stop before a contact stage (< 0) [s]
    FALLBACK_CONTACT_STALL_SPEED = CORRIDOR_FALLBACK_CONTACT_STALL_SPEED  # [m/s]
    FALLBACK_RELEASE_TICKS = CORRIDOR_FALLBACK_RELEASE_TICKS # fresh normal paths in a row that end the last resort

    def __init__(self, base, occupancy, map_data_dir=None):
        self.base = base              # RecordedStaticPlanner: geometry, global curve, recorded corridor
        self.occupancy = occupancy    # StaticObstacleMap: distance(x, y)
        self.map_links = self._load_links(map_data_dir)   # centrelines of extra drivable map lanes
        length, half_width = base.planner.vehicle_length, base.planner.vehicle_half_width
        count = CORRIDOR_BODY_CIRCLES
        self.circle_t = (np.arange(count)+.5)*length/count
        self.circle_r = math.hypot(half_width, .5*length/count)
        self.ready = False
        self.min_clearance = CORRIDOR_MIN_CLEARANCE   # required body clearance [m]; relaxed only by the last resort
        self.previous = None          # (ys, xs) of the last station solution
        self.committed = None         # (yy, xx, yaw, curvature) dense samples of the last valid path
        self._since_solve = RESOLVE_EVERY   # ticks since the last full solve
        self._held_error = (math.inf, math.inf)   # (distance, heading) of the car from the committed path
        self._global_xy = None
        self.clock = time.monotonic             # replaced by the replay's clock in offline checks
        self._stall_since = None                # when the car began standing or creeping blocked
        self._fallback_active = False           # stays on until a path passes the normal clearance again
        self._fallback_stage = None             # index of the last stage that gave a path
        self._clear_ticks = 0                   # consecutive ticks with a normal path
        self._committed_relaxed = False         # the committed path came from a last-resort solve
        self._standing_since = None             # when the car came to a full stop, blocked
        self._contact_granted = False           # the contact stages are allowed until the last resort ends
        self.fallback_log = deque(maxlen=1000)  # (clock, stage clearance) of the recent ticks that used the last resort

    # ------------------------------------------------------------------ geometry
    def goal_y(self):
        g = self.base.geometry
        return self.base.goal_y if self.ready else g.rejoin_point[1]+g.tail_length

    def contains(self, ego):
        g = self.base.geometry
        return (g.active_x[0] < ego.position.x < g.active_x[1]
                and g.route_y[0] < ego.position.y < self.goal_y() and math.sin(ego.yaw) > .5)

    def clearance(self, x, y, yaw, screen=False):
        """Body clearance [m] to occupied cells for poses (rear reference point, heading).

        screen=True uses the map's cheap lower bound (up to one cell diagonal short) for many poses.
        """
        x, y, yaw = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float))
        c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
        options = dict(exact_below=0.) if screen else {}
        d = self.occupancy.distance(x[..., None]+self.circle_t*c, y[..., None]+self.circle_t*s, **options)
        return d.min(axis=-1)-self.circle_r

    def _speed_clearance(self, x, y, yaw, clear):
        """Clearance the speed is chosen by; the planning clearance here (a subclass may plan round more)."""
        return clear

    def _corridor(self, ys, y0=None, x0=None):
        """Lateral bounds of the recorded drivable corridor; near the pose they also include the measured x."""
        centers = np.array([curve(ys) for curve in self.base.curves]+[self.base.global_curve(ys)])
        low, high = centers.min(axis=0)-.15, centers.max(axis=0)+.15
        if x0 is not None:
            keep = 1.-np.clip((np.asarray(ys)-y0)/ESCAPE_DISTANCE, 0., 1.)
            low = low-np.maximum(0., low-x0)*keep
            high = high+np.maximum(0., x0-high)*keep
        return low, high

    def _corridor_margin(self, x, y, yaw, y0, centre=None):
        """Signed distance [m] by which a body stays inside the corridor (negative: outside), the test of
        base._envelope_mask: reference point within the centre bounds, body corners within the recorded body
        envelope. Waived (inf) within ESCAPE_DISTANCE of the pose. `centre` = (low, high) at y when known."""
        base = self.base
        x, y, yaw = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float))
        low, high = self._corridor(y) if centre is None else centre
        margin = np.minimum(x-low, high-x)
        c, s = np.cos(yaw), np.sin(yaw)
        for along in (0., base.planner.vehicle_length):
            for side in (-base.planner.vehicle_half_width, base.planner.vehicle_half_width):
                cx, cy = x+along*c-side*s, y+along*s+side*c
                margin = np.minimum(margin, np.minimum(
                    cx-(np.interp(cy, base.envelope_y, base.envelope_low)-.15),
                    np.interp(cy, base.envelope_y, base.envelope_high)+.15-cx))
        return np.where(y-y0 <= ESCAPE_DISTANCE, np.inf, margin)

    # ------------------------------------------------------------------ planning steps
    def _required(self, ys, y0, pose_clear):
        """Required clearance per station: a pose that already lost clearance (the map grew) may still move away."""
        minimum = self.min_clearance
        escape = min(minimum, max(CONTACT_CLEARANCE, pose_clear-VALIDATION_TOLERANCE))
        ramp = np.clip((np.asarray(ys)-y0)/ESCAPE_DISTANCE, 0., 1.)
        return escape+(minimum-escape)*ramp

    @staticmethod
    def _steps():
        return int(math.ceil(CORRIDOR_MAX_SLOPE*CORRIDOR_STATION_STEP/CORRIDOR_LATERAL_STEP))

    def _feasibility(self, ys, xs, required, x0):
        """Feasible cells and clearance per DP step d (lateral cells moved per station), shape (2*steps+1, n, m).

        The body at a station is judged with the heading of the step that reaches it: passing an obstacle close to
        the lane edge needs the body turned, and a body held at the reference heading would not fit.
        """
        steps = self._steps()
        low, high = self._corridor(ys, ys[0], x0)
        X, Y = np.meshgrid(xs, ys)
        inside = (X >= low[:, None]) & (X <= high[:, None])
        need = required[:, None]
        feasible = np.zeros((2*steps+1,)+X.shape, dtype=bool)
        clear = np.empty((2*steps+1,)+X.shape)
        straight = self.clearance(X, Y, np.full(X.shape, .5*math.pi), screen=True)
        for k, d in enumerate(range(-steps, steps+1)):
            yaw = math.atan2(1., d*CORRIDOR_LATERAL_STEP/CORRIDOR_STATION_STEP)
            # Turned by `turn` from straight, a body circle moves at most 2 t sin(turn/2): cells that far clear of
            # the straight body stay clear at this heading.
            reach = self.circle_t[-1]*2.*math.sin(.5*abs(yaw-.5*math.pi))
            near = inside & (straight < need+reach)
            c = straight.copy()
            if d != 0 and near.any():
                c[near] = self.clearance(X[near], Y[near], np.full(int(near.sum()), yaw), screen=True)
            # Measure exactly only where the cheap bound alone would reject a cell that may in fact keep the
            # required clearance (bound short by at most SCREEN_SLACK).
            doubtful = inside & (c < need) & (c >= need-SCREEN_SLACK)
            if doubtful.any():
                c[doubtful] = self.clearance(X[doubtful], Y[doubtful], np.full(int(doubtful.sum()), yaw))
            ok = inside & (c >= need)
            centre = (np.broadcast_to(low[:, None], X.shape)[ok], np.broadcast_to(high[:, None], X.shape)[ok])
            ok[ok] = self._corridor_margin(X[ok], Y[ok], yaw, ys[0], centre) >= 0.
            feasible[k], clear[k] = ok, c
        # A path's heading at a station lies between its incoming and outgoing steps, not exactly at the step's
        # quantized slope (0.1 m per station): accept a cell feasible at a neighbouring step heading too. Tight gaps
        # entered at an angle are otherwise lost to the grid; the smoothed path is validated with its own heading.
        masked = np.where(feasible, clear, -np.inf)
        relaxed = masked.copy()
        relaxed[1:] = np.maximum(relaxed[1:], masked[:-1])
        relaxed[:-1] = np.maximum(relaxed[:-1], masked[1:])
        ok = relaxed > -np.inf
        return ok, np.where(ok, relaxed, clear)

    def _choose_side(self, xs, x_ref, feasible, clear, x0, previous):
        """Cheapest lateral cell per station; returns cells and DP steps for stations 1..last."""
        w = CORRIDOR_WEIGHTS
        steps = self._steps()
        cost = np.full(len(xs), np.inf)
        cost[int(np.argmin(np.abs(xs-x0)))] = 0.
        back = np.zeros((len(x_ref), len(xs)), dtype=int)
        last, final = 0, None
        moves = np.arange(-steps, steps+1)
        source = np.arange(len(xs))[None, :]-moves[:, None]               # (move, cell) -> previous cell
        valid = (source >= 0) & (source < len(xs))
        source = np.clip(source, 0, len(xs)-1)
        # The slope is penalised relative to the reference's own slope: on a curving route an absolute penalty
        # pulls every solve straight, off the road's bend (0.2-0.5 m at the tunnel exit), and alternating solves
        # then flip the path left and right.
        step_slope = (moves*CORRIDOR_LATERAL_STEP/CORRIDOR_STATION_STEP)[:, None]
        node = np.where(feasible, w['clearance']*np.maximum(0., CORRIDOR_DESIRED_CLEARANCE-clear)**2, np.inf)
        for i in range(1, len(x_ref)):
            base = w['reference']*(xs-x_ref[i])**2
            if previous is not None:
                base = base + w['previous']*(xs-previous[i])**2
            reference_slope = (x_ref[i]-x_ref[i-1])/CORRIDOR_STATION_STEP
            value = (np.where(valid, cost[source], np.inf)+w['slope']*(step_slope-reference_slope)**2
                     + node[:, i]+base)
            k = np.argmin(value, axis=0)
            cost = value[k, np.arange(len(xs))]
            back[i] = source[k, np.arange(len(xs))]
            if not np.isfinite(cost).any():
                break
            last, final = i, cost
        if last == 0:
            return [], [], 0
        j = int(np.argmin(final))
        chosen, moves = [j], []
        for i in range(last, 0, -1):
            source = int(back[i, j])
            moves.append(j-source)
            if i > 1:
                chosen.append(source)
            j = source
        return chosen[::-1], moves[::-1], last

    @staticmethod
    def _interval(row, j):
        left = right = j
        while left > 0 and row[left-1]:
            left -= 1
        while right < len(row)-1 and row[right+1]:
            right += 1
        return left, right

    def _smooth(self, targets, lower, upper, x0, slope0, constraints=(), reference=None):
        """Bounded least squares; `constraints` are linearized margins (gradient g, value b) wanted as g @ v = b.

        Curvature and its change are penalised relative to ``reference`` (x of the route at y0-ds, y0, stations), so
        the path follows a curving road without cost instead of being pulled straight."""
        w, ds, n = CORRIDOR_WEIGHTS, CORRIDOR_STATION_STEP, len(targets)
        reference = np.zeros(n+2) if reference is None else np.asarray(reference, dtype=float)
        # Full sequence X = [x_-1, x_0, v_1..v_n] = S v + c
        S = np.zeros((n+2, n)); S[2:, :] = np.eye(n)
        c = np.zeros(n+2); c[0], c[1] = x0-ds*slope0, x0
        rows, rhs = [np.sqrt(w['target'])*np.eye(n)], [np.sqrt(w['target'])*np.asarray(targets)]
        for gradient, value in constraints:
            scale = math.sqrt(CONSTRAINT_WEIGHT)/max(np.linalg.norm(gradient), 1e-6)
            rows.append(scale*gradient[None, :]); rhs.append(np.array([scale*value]))
        second = np.array([[1., -2., 1.]])/ds**2
        D2 = np.zeros((n, n+2))
        for k in range(n):
            D2[k, k:k+3] = second
        rows.append(np.sqrt(w['curvature'])*D2@S); rhs.append(np.sqrt(w['curvature'])*D2@(reference-c))
        if n > 1:
            D3 = np.zeros((n-1, n+2))
            for k in range(n-1):
                D3[k, k:k+4] = np.array([-1., 3., -3., 1.])/ds**3
            rows.append(np.sqrt(w['jerk'])*D3@S); rhs.append(np.sqrt(w['jerk'])*D3@(reference-c))
        upper = np.maximum(upper, lower+1e-3)
        A, b = np.vstack(rows), np.concatenate(rhs)
        solution = self._box_qp(A.T@A, A.T@b, lower, upper, np.clip(targets, lower, upper))
        if solution is not None:
            return solution
        # BVLS divides by zero on nearly pinned variables; fall back to the trust-region solver, then to the targets.
        for method in ('bvls', 'trf'):
            with np.errstate(divide='ignore', invalid='ignore'):
                solution = lsq_linear(A, b, bounds=(lower, upper), method=method).x
            if np.isfinite(solution).all():
                return np.clip(solution, lower, upper)
        return np.clip(targets, lower, upper)

    @staticmethod
    def _box_qp(H, g, lower, upper, v, iterations=200):
        """min 1/2 v'Hv - g'v within bounds: projected Newton on the free variables (Bertsekas 1982).

        A few small dense solves instead of BVLS's least-squares solve per bound change; None if not converged.
        """
        def value(z):
            return .5*z@H@z-g@z
        scale = max(1., float(np.max(np.abs(g))))
        for _ in range(iterations):
            grad = H@v-g
            projected = v-np.clip(v-grad, lower, upper)
            if np.max(np.abs(projected)) <= 1e-9*scale:
                return v
            free = ~(((v <= lower) & (grad > 0.)) | ((v >= upper) & (grad < 0.)))
            step = -grad
            if free.any():
                with np.errstate(all='ignore'):
                    step[free] = -np.linalg.solve(H[np.ix_(free, free)], grad[free])
            if not np.isfinite(step).all():
                return None
            current, t = value(v), 1.
            while t > 1e-8:
                candidate = np.clip(v+t*step, lower, upper)
                if value(candidate) <= current+1e-4*grad@(candidate-v):
                    break
                t *= .5
            else:
                return None
            moved = np.max(np.abs(candidate-v))
            v = candidate
            if moved <= 1e-10:
                return v
        return None

    @staticmethod
    def _basis(station_y, x0, slope0, yy):
        """Dense x, slope and second derivative are affine in the station variables v: (base, matrix) each."""
        n = len(station_y)-1
        data = np.zeros((n+1, n+1)); data[0, 0] = x0; data[1:, 1:] = np.eye(n)
        start = np.zeros(n+1); start[0] = slope0
        curve = CubicSpline(station_y, data, bc_type=((1, start), (2, np.zeros(n+1))))
        return [(m[:, 0], m[:, 1:]) for m in (curve(yy), curve(yy, 1), curve(yy, 2))]

    def _dense(self, ys, xs, slope0):
        curve = CubicSpline(ys, xs, bc_type=((1, slope0), (2, 0.)))
        yy = np.arange(ys[0], ys[-1]+1e-9, DENSE_STEP)
        xx, slope, second = curve(yy), curve(yy, 1), curve(yy, 2)
        return yy, xx, np.arctan2(1., slope), np.abs(second)/(1+slope*slope)**1.5

    # ------------------------------------------------------------------ main entry
    def run(self, ego, reference_path, global_path):
        """Plan one tick; blocked for long, the last resort (FALLBACK_*) retries with lower clearance."""
        result = self._solve(ego, reference_path, global_path)
        if self.FALLBACK_STALL_S <= 0.:
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
            if not self._fallback_active or info is None or self._clear_ticks >= self.FALLBACK_RELEASE_TICKS:
                self._reset_fallback()
            return result
        self._clear_ticks = 0
        now = self.clock()
        if abs(ego.velocity) < self.FALLBACK_CONTACT_STALL_SPEED:
            self._standing_since = now if self._standing_since is None else self._standing_since
        else:
            self._standing_since = None
        if getattr(self.occupancy, 'virtual', None):
            self._reset_fallback()        # tunnel: a mover is being overtaken or yielded to, its own rules decide
            return result
        if not self._fallback_active:
            # Blocked, the planner still lets the car close in on the stop point 1 m before the obstacle, and the
            # point moves on as the map changes: the car crept forward twice (to 1.3 m/s) and a 0.3 m/s standing test
            # restarted the wait each time (9 s instead of 3, competition tunnel 10-29). Slow approach counts too.
            if abs(ego.velocity) >= self.FALLBACK_STALL_SPEED:
                self._stall_since = None
                return result
            if self._stall_since is None:
                self._stall_since = now
            if now-self._stall_since < self.FALLBACK_STALL_S:
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
        if self.FALLBACK_CONTACT_STALL_S <= 0.:
            return True
        if not self._contact_granted and self._standing_since is not None:
            self._contact_granted = self.clock()-self._standing_since >= self.FALLBACK_CONTACT_STALL_S
        return self._contact_granted

    def _relaxed_plan(self, ego, reference_path, global_path):
        """The same solve with the required clearance lowered step by step (corridor condition unchanged); the stage
        that worked last tick goes first. Without any path, the committed path ahead of the car, if the car is on it."""
        stages = self.FALLBACK_CLEARANCES
        order = list(range(len(stages)))
        if self._fallback_stage is not None:
            order.remove(self._fallback_stage)
            order.insert(0, self._fallback_stage)
        nominal = self.min_clearance
        # A failed attempt must leave no trace in the normal planning (its last solution, committed path and solve
        # cadence): every stage starts from the state before the attempt, and without any path it is restored.
        saved = (self.previous, self.committed, self._since_solve, self._held_error)
        try:
            for stage in order:
                if stages[stage] < 0. and not self._contact_allowed():
                    continue
                self.previous, self.committed, self._since_solve, self._held_error = saved
                self.min_clearance = stages[stage]
                self._since_solve = 1 << 30         # a held path of the stricter solve is not reused: solve afresh
                result = self._solve(ego, reference_path, global_path)
                if result is None:
                    continue
                info = result.debug_details['static_avoidance']
                if info.get('phase') == 'blocked':
                    continue
                self._fallback_stage = stage
                self._committed_relaxed = True
                self._cap(result, info, stages[stage])
                return result
            self.previous, self.committed, self._since_solve, self._held_error = saved
        finally:
            self.min_clearance = nominal
        return None

    def _cap(self, result, info, clearance):
        """Speed cap and log of a last-resort tick; ``clearance`` None: the normal solve held the last-resort path."""
        result.acc_target_velocity = min(result.acc_target_velocity, self.FALLBACK_SPEED)
        info['fallback'] = dict(active=True, required_clearance_m=clearance, speed_cap_mps=self.FALLBACK_SPEED,
                                source='held' if clearance is None else 'relaxed')
        self.fallback_log.append((self.clock(), clearance))

    def _solve(self, ego, reference_path, global_path):
        """One solve. A numerical failure stops the car for this tick instead of ending the control loop."""
        begin = time.perf_counter()
        try:
            return self._plan(ego, reference_path, global_path, begin)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            ahead = Point(ego.position.x+.5*math.cos(ego.yaw), ego.position.y+.5*math.sin(ego.yaw))
            details = dict(planner='static_corridor', phase='blocked', path_validated=False,
                           error='planning_error: '+str(exc)[:160])
            return self._result([ego.position, ahead], 0., False, details, begin, [], np.zeros(0))

    def prepare(self, global_path, start_y=None):
        """One-time setup on the global route (about 0.2 s: route curve, corridor, envelope, road table).

        Called at start-up so the first planning tick in the section does not carry it. ``start_y`` is the earliest
        pose y the corridor must cover (default: the section's route start)."""
        if self.ready:
            return True
        y = self.base.geometry.route_y[0] if start_y is None else min(float(start_y), self.base.geometry.route_y[0])
        anchor = VehicleState(0., y, .5*math.pi, 0.)
        try:
            self.base._setup(global_path, anchor)
        except ValueError:
            return False
        self._add_lane_bounds(anchor)
        self.ready = True
        self._global_xy = np.array([(q.x, q.y) for q in global_path])
        return True

    def _plan(self, ego, reference_path, global_path, begin):
        if not self.prepare(global_path, ego.position.y):
            return None
        ds = CORRIDOR_STATION_STEP
        y0, x0 = float(ego.position.y), float(ego.position.x)
        end = min(y0+CORRIDOR_HORIZON, self.base.goal_y)
        if end-y0 < 3*ds:
            return None
        pose_clear = float(self.clearance(x0, y0, ego.yaw))
        held = self._held_path(ego, pose_clear)
        # The committed path is re-checked on the current map every tick. While it holds and still reaches far
        # enough, the full solve runs only every RESOLVE_EVERY ticks; its cost would otherwise add to the
        # sensor-to-command latency on every frame (traffic-signal freshness, control delay).
        self._since_solve += 1
        if (held is not None and self._since_solve < RESOLVE_EVERY and held[0][-1] >= end-RESOLVE_SHORTFALL
                and self._held_error[0] <= SKIP_TRACKING_ERROR and self._held_error[1] <= SKIP_HEADING_ERROR):
            details = dict(planner='static_corridor', stations=0, feasible_stations=None, min_clearance_m=None,
                           stop_distance_m=None, repairs=0, solve='held')
            return self._finish(ego, held, False, pose_clear, details, begin, source='previous')
        self._since_solve = 0
        ys = y0+ds*np.arange(int((end-y0)/ds)+1)
        x_ref = self.base.global_curve(ys)
        slope0 = float(np.clip(math.cos(ego.yaw)/max(.5, math.sin(ego.yaw)), -CORRIDOR_MAX_SLOPE, CORRIDOR_MAX_SLOPE))
        low, high = self._corridor(ys, y0, x0)
        xs = np.arange(low.min(), high.max()+1e-9, CORRIDOR_LATERAL_STEP)
        required = self._required(ys, y0, pose_clear)
        feasible, clear = self._feasibility(ys, xs, required, x0)
        previous = None
        if self.previous is not None:
            previous = np.interp(ys, *self.previous, left=np.nan, right=np.nan)
            previous = np.where(np.isfinite(previous), previous, x_ref)
        chosen, moves, last = self._choose_side(xs, x_ref, feasible, clear, x0, previous)
        blocked = last < len(ys)-1
        details = dict(planner='static_corridor', stations=len(ys), feasible_stations=last,
                       min_clearance_m=None, stop_distance_m=None, repairs=0)
        if not chosen:
            if held is not None:
                details['new_plan_error'] = 'no_feasible_station'
                return self._finish(ego, held, False, pose_clear, details, begin, source='previous')
            path, speed, valid = [ego.position, Point(x0, y0+.5)], 0., False
            details.update(phase='blocked', path_validated=False, error='no_feasible_station')
            details['pose_clearance_m'] = round(pose_clear, 3)
            return self._result(path, speed, valid, details, begin, [], np.zeros(0))
        n = len(chosen)
        steps = self._steps()
        intervals = [self._interval(feasible[d+steps, i+1], j) for i, (j, d) in enumerate(zip(chosen, moves))]
        lower = np.array([xs[a] for a, _ in intervals])
        upper = np.array([xs[b] for _, b in intervals])
        # The side choice is quantized to the lateral grid; smooth it so the target does not inject curvature.
        targets = np.convolve(np.pad(xs[np.array(chosen)], 2, mode='edge'), np.ones(5)/5, mode='valid')
        targets = np.clip(targets, lower, upper)
        station_y = ys[:n+1]
        yy = np.arange(station_y[0], station_y[-1]+1e-9, DENSE_STEP)
        (X0, XB), (S0, SB), (K0, KB) = self._basis(station_y, x0, slope0, yy)
        centre = self._corridor(yy, y0, x0)
        required_dense = self._required(yy, y0, pose_clear)-VALIDATION_TOLERANCE
        route = self.base.global_curve(np.r_[y0-ds, station_y])   # x of the route at x_-1, x_0 and the stations
        constraints = []
        any_heading = feasible.any(axis=0)
        for attempt in range(2):
            if attempt:
                if ok.all():
                    break
                # The boxes hold each station inside the cells feasible at its DP step's heading. Entering a tight
                # gap at an angle, the path may need a position feasible only at a heading between the grid's
                # steps (the pose's own heading held for the first metre): retry once with the cells feasible at
                # any heading, widened by RETRY_WIDEN, the linearized margins and the validation above judging every
                # point at the path's own heading.
                intervals = [self._interval(any_heading[i+1], j) for i, j in enumerate(chosen)]
                lower = np.array([xs[a] for a, _ in intervals])-RETRY_WIDEN
                upper = np.array([xs[b] for _, b in intervals])+RETRY_WIDEN
                details['retry'] = 'any_heading_boxes'
            for repair in range(MAX_REPAIRS+1):
                solution = self._smooth(targets, lower, upper, x0, slope0, constraints, route)
                station_x = np.r_[x0, solution]
                yy, xx, yaw, curvature = self._dense(station_y, station_x, slope0)
                dense_clear = self.clearance(xx, yy, yaw)
                clear_margin = dense_clear-required_dense
                corridor_margin = self._corridor_margin(xx, yy, yaw, y0, centre)
                clear_ok, corridor_ok = clear_margin >= 0., corridor_margin >= 0.
                curvature_ok = curvature <= MAX_CURVATURE+1e-6
                ok = clear_ok & curvature_ok & corridor_ok
                details['violations'] = dict(clearance=int((~clear_ok).sum()), curvature=int((~curvature_ok).sum()),
                                             corridor=int((~corridor_ok).sum()))
                if ok.all() or repair == MAX_REPAIRS:
                    break
                details['repairs'] = repair+1
                # The DP judged each station at the heading of its own step; smoothing changes position and heading
                # together, and a body turned slightly differently can clip an obstacle or the corridor edge. Linearize
                # each violated margin in the station variables (through the dense point's position and heading) and
                # add it to the next solve; earlier linearizations stay, so the solve does not undo a fixed violation.
                slope, h = S0+SB@solution, GRADIENT_STEP
                def clear_at(k, x, s):
                    return self.clearance(x, yy[k], np.arctan2(1., s))-required_dense[k]
                def corridor_at(k, x, s):
                    return self._corridor_margin(x, yy[k], np.arctan2(1., s), y0, (centre[0][k], centre[1][k]))
                for bad, value, margin in ((~clear_ok, clear_margin, clear_at),
                                           (clear_ok & ~corridor_ok, corridor_margin, corridor_at)):
                    k = np.flatnonzero(bad)
                    if not len(k):
                        continue
                    x, s = xx[k], slope[k]
                    gradients = ((margin(k, x+h, s)-margin(k, x-h, s))[:, None]/(2*h)*XB[k]
                                 + (margin(k, x, s+h)-margin(k, x, s-h))[:, None]/(2*h)*SB[k])
                    constraints += [(g, g@solution+REPAIR_MARGIN-v) for g, v in zip(gradients, value[k])]
                if not curvature_ok.all():
                    # Pinning only the violating points pushes the bend onto their neighbours; hold every point near
                    # the limit too, at its current value or the cap, whichever is lower.
                    second = K0+KB@solution
                    cap = .9*MAX_CURVATURE*(1+slope**2)**1.5
                    for k in np.flatnonzero(curvature > CURVATURE_HOLD*MAX_CURVATURE):
                        constraints.append((KB[k], math.copysign(min(abs(second[k]), cap[k]), second[k])-K0[k]))
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
            self.committed = (yy, xx, yaw, curvature)
        return self._finish(ego, (yy, xx, yaw, curvature, dense_clear), blocked, pose_clear, details, begin)

    def _held_path(self, ego, pose_clear):
        """The committed path ahead of the car, if the car is on it and it is still valid on the current map."""
        if self.committed is None:
            return None
        yy, xx, yaw, curvature = self.committed
        y0 = float(ego.position.y)
        d = np.hypot(xx-ego.position.x, yy-y0)
        k = int(np.argmin(d))
        if d[k] > HOLD_TRACKING_ERROR or len(yy)-k < 8:
            return None
        # The spline's own heading and curvature, as validated when the path was planned: recomputing them by
        # finite differences moves the clearance by millimetres and drops a path accepted at the threshold.
        yy, xx, yaw, curvature = yy[k:], xx[k:], yaw[k:], curvature[k:]
        heading_error = abs(math.atan2(math.sin(ego.yaw-yaw[0]), math.cos(ego.yaw-yaw[0])))
        self._held_error = (float(d[k]), heading_error)
        if heading_error > HOLD_HEADING_ERROR:
            return None
        clear = self.clearance(xx, yy, yaw)
        station = np.r_[0., np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
        # A path the car is already on may continue while it never gets closer than the car is now: beside an
        # obstacle that grew in the map, clearance cannot recover until the obstacle is passed.
        required = np.minimum(self._required(yy, y0, pose_clear),
                              min(self.min_clearance, max(CONTACT_CLEARANCE, pose_clear-VALIDATION_TOLERANCE)))
        ok = ((clear >= required-VALIDATION_TOLERANCE)
              & ((curvature <= MAX_CURVATURE+1e-6) | (station < CURVATURE_SPEED_SKIP))
              & (self._corridor_margin(xx, yy, yaw, y0) >= 0.))
        if not ok.all():
            return None
        return yy, xx, yaw, curvature, clear

    def _finish(self, ego, dense, blocked, pose_clear, details, begin, source='new'):
        yy, xx, yaw, curvature, dense_clear = dense
        speed_clear = self._speed_clearance(xx, yy, yaw, dense_clear)
        if source == 'previous':
            stations = np.arange(yy[0], yy[-1], CORRIDOR_STATION_STEP)
            self.previous = (stations, np.interp(stations, yy, xx))
        details['source'] = source
        # Speed: curvature, clearance, stopping before an impassable station.
        station = np.r_[0., np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
        base_speed, gain = self.CLEARANCE_SPEED
        # The first metres only join the measured heading to the plan; the tracker looks ahead past them.
        speed_curvature = np.where(station < CURVATURE_SPEED_SKIP, 0., curvature)
        limit = np.minimum.reduce([
            np.full(len(yy), p.STATIC_MAX_SPEED_KPH/3.6 if self.MAX_SPEED is None else self.MAX_SPEED),
            np.sqrt(CORRIDOR_LATERAL_ACCEL/np.maximum(speed_curvature, 1e-6)),
            base_speed+gain*np.maximum(0., np.minimum(speed_clear, 10.)-CORRIDOR_MIN_CLEARANCE)])
        # A tight gap is entered slowly: an object seen too small from afar grows on the map as the car closes in,
        # and at this speed the car can still stop or take the re-solved path instead of driving into it.
        tight = speed_clear < CORRIDOR_MIN_CLEARANCE+self.TIGHT_MARGIN
        limit = np.where(tight, np.minimum(limit, self.TIGHT_SPEED), limit)
        if blocked:
            stop = station[-1]-CORRIDOR_STOP_BUFFER
            limit = np.where(station >= stop, 0., limit)
            details['stop_distance_m'] = float(max(0., stop))
        for i in range(len(limit)-2, -1, -1):
            limit[i] = min(limit[i], math.sqrt(limit[i+1]**2+2*p.STATIC_END_DECELERATION*(station[i+1]-station[i])))
        speed = float(limit[0])
        path = [Point(float(a), float(b)) for a, b in zip(xx, yy)]
        if not blocked:
            path += self._global_tail(path[-1])
        details.update(phase='blocked' if blocked else 'follow', path_validated=not blocked,
                       min_clearance_m=float(np.min(dense_clear)) if len(dense_clear) else None,
                       pose_clearance_m=pose_clear, max_curvature=float(np.max(curvature)) if len(curvature) else 0.,
                       speed_policy_reason='corridor')
        return self._result(path, speed, not blocked, details, begin, path, curvature)

    def _add_lane_bounds(self, ego):
        """The recordings cover only the lateral positions they were driven at. Add the map lanes as corridor
        curves: both halves of the own lane, and the right-turn lane at the section entrance, which an obstacle
        early in the section needs before the recorded bypasses start. Centre bounds and the body envelope widen
        together; between curves the corridor is filled."""
        base = self.base
        offset = CORRIDOR_LANE_HALF_WIDTH-base.planner.vehicle_half_width-CORRIDOR_LANE_EDGE_MARGIN
        if offset <= 0.:
            return
        for shift in (-offset, offset):
            def own_lane(y, derivative=0, curve=base.global_curve, shift=shift):
                return curve(y, derivative)+(shift if derivative == 0 else 0.)
            base.curves.append(own_lane)
        for points in self.map_links:
            for shift in (-offset, offset):
                curve = self._lane_curve(points, shift, base.global_curve)
                if curve is not None:
                    base.curves.append(curve)
        base._center_bounds_cache = {}
        base._make_envelope(min(ego.position.y, min(d[0, 1] for _, d, _ in base.records))-2)

    @staticmethod
    def _lane_curve(points, shift, outside):
        """x(y) of a link centreline moved `shift` to its right; the global curve outside the link's y-range.

        Only the part heading forward (at least 45 degrees from the x axis) is used, where x(y) is well defined.
        """
        heading = np.unwrap(np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0])))
        xy = points+shift*np.column_stack((np.sin(heading), -np.cos(heading)))
        keep = np.sin(heading) >= math.sin(math.radians(45.))
        if not keep[0]:
            return None
        end = int(np.argmin(keep)) if not keep.all() else len(keep)
        xy = xy[:end]
        xy = xy[np.r_[True, np.diff(xy[:, 1]) > .01]]
        if len(xy) < 4 or np.any(np.diff(xy[:, 1]) <= 0.):
            return None
        spline, lo, hi = CubicSpline(xy[:, 1], xy[:, 0]), float(xy[0, 1]), float(xy[-1, 1])
        def lane(y, derivative=0):
            y = np.asarray(y, dtype=float)
            on = (y >= lo) & (y <= hi)
            return np.where(on, spline(np.clip(y, lo, hi), derivative), outside(y, derivative))
        return lane

    @staticmethod
    def _load_links(map_data_dir):
        if map_data_dir is None:
            return []
        path = Path(map_data_dir)/'link_set.json'
        if not path.is_file():
            return []
        links = {link['idx']: link for link in json.loads(path.read_text(encoding='utf-8'))}
        return [np.asarray(links[name]['points'], dtype=float)[:, :2] for name in CORRIDOR_MAP_LINKS if name in links]

    def _global_tail(self, last, count=60):
        index = int(np.argmin(np.hypot(self._global_xy[:, 0]-last.x, self._global_xy[:, 1]-last.y)))
        return [Point(float(x), float(y)) for x, y in self._global_xy[index+1:index+1+count]]

    def _result(self, path, speed, valid, details, begin, points, curvature):
        details['replan_ms'] = round(1e3*(time.perf_counter()-begin), 1)
        candidate = CandidatePath(layer='corridor')
        candidate.points, candidate.valid = list(points), valid
        return LatticePlanningResult(path, active=True, mode='static_obstacle', obstacle_detected=True,
                                     acc_override=True, acc_target_velocity=speed, candidates=[candidate],
                                     reason=self.REASON+details['phase'],
                                     debug_details={'static_avoidance': details})
