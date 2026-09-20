"""Precompute one complete route through a recorded, northbound static zone.

Recordings are immutable. The envelope is empirical, NOT a surveyed road map.
All connectors and the global-route return undergo the same footprint checks.
"""

import csv
import copy
import json
import math
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, UnivariateSpline

from ...localization.point import Point
from ...vehicle_state import VehicleState
from ..lattice_planning import parameters as p
from ..lattice_planning.structures import CandidatePath, LatticePlanningResult
from .corridor_graph import search_corridor


# =============================================================================
# 정적 장애물 기록 경로 플래닝: 경로 자료·전역 경로 복귀
# 공통 속도·조향·사전 계획 설정은 ../lattice_planning/parameters.py의 2번 항목 참조.
# =============================================================================
ROOT = Path(__file__).resolve().parent # 기록 경로 목록 및 후보 경로 자료를 읽을 기준 폴더
STEP = 0.2                           # 현재 미사용: 이전 경로 샘플 간격 [m]
MAX_CURVATURE = p.STATIC_MAX_CURVATURE # 기록 경로 연결에 허용할 최대 곡률; 공통 설정 참조 [1/m]
GLOBAL_REJOIN_POINT = (-59.97, -116.70, 28.37) # 전역 경로 복귀 기준점 (x, y, z) [m]
RETURN_WINDOW = 20.0                 # 복귀점 앞에서 완만한 복귀 시작 위치를 탐색할 y 방향 범위 [m]
RETURN_MAX_CURVATURE = 0.12           # 기록 경로에서 전역 경로로 복귀할 때 허용할 최대 곡률 [1/m]
TRACKING_SPEED = p.STATIC_MAX_SPEED_KPH / 3.6 # 정적 회피 속도 상한의 제어용 환산값 [m/s]

# =============================================================================
# 정적 장애물 추종: 자세 오차에 따른 감속
# 시작 기준부터 최대 감속 기준까지 감속 강도를 선형으로 늘린다.
# 방향·횡오차가 모두 시작 기준의 절반 미만이 되면 감속 상태를 해제한다.
# =============================================================================
ALIGNMENT_SPEED = 1.0                 # 최대 자세 보정 감속의 목표속도 및 통로 추종 기준속도; 안전 정지는 0 [m/s]
ALIGNMENT_HEADING_ERROR = math.radians(2.0) # 방향 오차에 따른 감속 시작 기준; 2도 [rad]
ALIGNMENT_FULL_HEADING_ERROR = math.radians(4.0) # 방향 오차에 따른 최대 감속 기준; 4도 [rad]
ALIGNMENT_CROSS_TRACK_ERROR = .12      # 경로 횡오차에 따른 감속 시작 기준 [m]
ALIGNMENT_FULL_CROSS_TRACK_ERROR = .25 # 경로 횡오차에 따른 최대 감속 기준 [m]
ALIGNMENT_PREVIEW = 3.0               # 추종 제어에 전달할 전방 최대 곡률을 조회하는 경로 길이 [m]


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
    shape = np.broadcast(np.asarray(x0), np.asarray(x1)).shape
    c = np.zeros((6,) + shape)
    c[0], c[1], c[2] = x0, d0 * length, dd0 * length * length / 2
    c[3:] = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]),
        [x1 - c[:3].sum(axis=0), d1 * length - c[1] - 2*c[2], dd1*length*length - 2*c[2]])
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
        # Scenario Editor only: invalidate the cached maneuver whenever the
        # operator moves an obstacle or teleports Ego. Normal driving leaves
        # this disabled and therefore retains the proven one-plan policy.
        self.live_preview_enabled = False
        # Once the editor releases control, keep the accepted route and run
        # the same progress/tracking checks as production driving. This stays
        # latched so pausing cannot turn a moving Ego back into preview mode.
        self.live_drive_committed = False
        self._live_preview_obstacle_key = None
        self._live_preview_ego_pose = None
        self.preview_ego_on_route = False
        try:
            self.records, self.audit = load_routes(root)
        except (OSError, ValueError, KeyError) as exc:
            self.records, self.audit = [], []
            self.error = str(exc)
        self.reset()

    def reset(self):
        if self.pending is not None:
            self.pending.cancel()
            self.pending = None
        self.pending_key = None
        self.preplanned = False
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
        self.search_details = {}
        self.recovery_until_y = None
        self.recovery_count = 0
        self.heading_error = 0.
        self.tracking_error = 0.
        self.cross_track_error = 0.
        self.alignment_slow = False
        self._speed_profile_path = None
        self._cached_speed_profile = None
        self.curve_speed_limits = {}
        self.path_smoothing_deviation = 0.

    @staticmethod
    def _obstacle_key(obstacles):
        return tuple(sorted((float(o.position.x), float(o.position.y),
                             float(o.heading_deg), float(o.size_x), float(o.size_y))
                            for o in obstacles))

    @staticmethod
    def _preview_obstacle_key(obstacles):
        """Stable geometry key for interactive edits, ignoring float noise."""
        return tuple(sorted((round(float(o.position.x), 3),
                             round(float(o.position.y), 3),
                             round(float(o.heading_deg), 2),
                             round(float(o.size_x), 3),
                             round(float(o.size_y), 3))
                            for o in obstacles))

    def set_live_preview(self, enabled):
        enabled = bool(enabled)
        if enabled == self.live_preview_enabled:
            return
        self.live_preview_enabled = enabled
        self.live_drive_committed = False
        self._live_preview_obstacle_key = None
        self._live_preview_ego_pose = None
        self.preview_ego_on_route = False
        self.reset()

    @property
    def interactive_preview(self):
        return self.live_preview_enabled and not self.live_drive_committed

    def set_live_drive(self, enabled):
        """Commit the editor preview for real Expert tracking.

        Enabling is one-way until the Expert process restarts. The browser's
        stop action pauses commands without discarding planner progress.
        """
        if bool(enabled) and self.live_preview_enabled:
            self.live_drive_committed = True

    def refresh_live_preview(self, ego, obstacles):
        """Reset an editor preview after a material scene change.

        This hook is intentionally called only when the Expert was launched
        with the Scenario Editor private ROS parameter. Ego is stationary in
        that mode, so a 25 cm/3 degree threshold rejects status jitter while
        still detecting an operator teleport.
        """
        if not self.interactive_preview:
            return False
        obstacle_key = self._preview_obstacle_key(obstacles)
        ego_pose = (float(ego.position.x), float(ego.position.y), float(ego.yaw))
        changed = False
        if self._live_preview_obstacle_key is not None:
            changed = obstacle_key != self._live_preview_obstacle_key
        if self._live_preview_ego_pose is not None:
            previous_x, previous_y, previous_yaw = self._live_preview_ego_pose
            yaw_delta = abs(math.atan2(
                math.sin(ego_pose[2] - previous_yaw),
                math.cos(ego_pose[2] - previous_yaw),
            ))
            changed = changed or math.hypot(
                ego_pose[0] - previous_x, ego_pose[1] - previous_y
            ) > .25 or yaw_delta > math.radians(3.)
        if changed:
            self.reset()
        self._live_preview_obstacle_key = obstacle_key
        self._live_preview_ego_pose = ego_pose
        self.preview_ego_on_route = (
            -65 < ego.position.x < -44
            and -164 < ego.position.y < GLOBAL_REJOIN_POINT[1]
            and math.sin(ego.yaw) > .5
        )
        return changed

    def preplan(self, ego, global_path, obstacles):
        """Warm the complete maneuver using GT and a fixed future entry pose.

        The worker owns its geometry/state. A reset or a changed GT snapshot
        cannot let an obsolete worker overwrite the live planner.
        """
        if self.done and ego.position.y < p.STATIC_PREPLAN_ENTRY_Y and -65 < ego.position.x < -44:
            self.reset()
        if self.executor is None or self.active or self.done or self.error:
            return
        if ego.position.y >= p.STATIC_PREPLAN_ENTRY_Y and self.contains(ego):
            return  # A late start is planned from the measured pose by plan().
        if not obstacles:
            return
        key = self._obstacle_key(obstacles)
        if self.pending is not None:
            if key == self.pending_key:
                return
            if not self.pending.done() and not self.pending.cancel():
                return  # Discard this obsolete result on the next cycle.
            self.pending = None
        if time.monotonic()-self.last_attempt < 1.:
            return
        route = [point for point in global_path if -180 < point.y < -85 and -65 < point.x < -54]
        if len(route) < 10:
            return
        xy = np.array(route)
        xy = xy[np.r_[True, np.diff(xy[:, 1]) > .01]]
        y = p.STATIC_PREPLAN_ENTRY_Y
        if np.any(np.diff(xy[:, 1]) <= 0) or xy[0, 1] > y or xy[-1, 1] < GLOBAL_REJOIN_POINT[1]+10.:
            return
        curve = CubicSpline(xy[:, 1], xy[:, 0], bc_type='natural')
        entry = VehicleState(float(curve(y)), y, math.atan2(1., float(curve(y, 1))), 0.)
        self._submit(entry, route, obstacles)
        self.preplanned = True

    def _submit(self, ego, route, obstacles):
        worker = copy.copy(self)
        worker.pending = None
        worker.executor = None
        worker.reset()
        self.pending_key = self._obstacle_key(obstacles)
        self.last_attempt = time.monotonic()
        self.pending = self.executor.submit(worker._prepare_snapshot, copy.deepcopy(ego),
                                            list(route), copy.deepcopy(list(obstacles)))

    def approach_result(self, ego, reference_path):
        """Brake for a known entry bend before reaching the lateral handoff."""
        if (self.pending is None or not self.pending.done() or not self.preplanned
                or not -65 < ego.position.x < -44 or math.sin(ego.yaw) <= .5
                or not 0. <= p.STATIC_PREPLAN_ENTRY_Y-ego.position.y <= 100.):
            return None
        try:
            worker = self.pending.result()
        except Exception:
            return None  # plan() reports the failure if entry is reached.
        if not worker.path:
            return None
        entry_limit, _ = worker._curve_speed_limit(worker.path)
        distance = max(0., p.STATIC_PREPLAN_ENTRY_Y-ego.position.y-p.STATIC_SPEED_PREVIEW)
        speed = math.sqrt(entry_limit**2+2*p.STATIC_END_DECELERATION*distance)
        return LatticePlanningResult(reference_path, mode='static_obstacle',
            acc_target_velocity=speed, reason='static_recorded_approach',
            debug_details={'static_avoidance': dict(phase='approach', path_validated=False,
                preplanned_complete=True, entry_speed_limit=entry_limit, entry_distance_m=distance)})

    def committed_approach_result(self, ego, reference_path):
        """Follow the global route until a committed editor path starts.

        Interactive preview deliberately exposes the complete avoidance path
        while Ego is held at the scenario start. After control is released,
        that path may still start many metres ahead. Tracking it immediately
        would exceed the production cross-track guard and command a stop.
        """
        if (not self.live_drive_committed or not self.path
                or len(reference_path) < 2
                or ego.position.y >= self.path[0].y-.25):
            return None
        entry_limit, _ = self._curve_speed_limit(self.path)
        distance = max(
            0., self.path[0].y-ego.position.y-p.STATIC_SPEED_PREVIEW
        )
        speed = math.sqrt(
            entry_limit**2 + 2*p.STATIC_END_DECELERATION*distance
        )
        candidate = CandidatePath(layer="recorded")
        candidate.points = list(self.path)
        candidate.valid = True
        return LatticePlanningResult(
            reference_path,
            active=True,
            mode="static_obstacle",
            obstacle_detected=True,
            acc_override=True,
            acc_target_velocity=speed,
            candidates=[candidate],
            reason="static_recorded_drive_approach",
            debug_details={"static_avoidance": dict(
                phase="drive_approach",
                path_validated=True,
                planner="recorded_route_graph",
                live_preview=False,
                live_drive_committed=True,
                selected_routes=self.selected_names,
                entry_speed_limit=entry_limit,
                entry_distance_m=distance,
                preplanned_complete=True,
            )},
        )

    def _prepare_snapshot(self, ego, route, obstacles):
        complete = self._prepare(ego, route, obstacles)
        self.path = [Point(x, y) for x, y in complete] if complete is not None else []
        if self.path:
            self._curve_speed_limit(self.path)
        return self

    def _waiting_result(self, ego, reference_path, obstacles, phase, error=None):
        """Keep driving only as far as the currently validated reference permits."""
        points = [ego.position] + [v for v in reference_path if v.y > ego.position.y+.01]
        if len(points) < 3:
            return self._result(reference_path, False, phase, error)
        yaw = np.array([ego.yaw] + [self.planner._path_heading(points, i) for i in range(1, len(points))])
        # Dense samples avoid stepping over a small obstacle on a sparse route.
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.array(points), axis=0), axis=1))]
        sample_s = np.linspace(0., distance[-1], max(2, int(math.ceil(distance[-1]/.1))+1))
        xy = np.array(points)
        samples = np.column_stack([np.interp(sample_s, distance, xy[:, axis]) for axis in (0, 1)])
        headings = np.interp(sample_s, distance, np.unwrap(yaw))
        clear = self._collision_mask(samples, headings, obstacles, .12)
        blocked = np.flatnonzero(~clear)
        available = sample_s[blocked[0]] if len(blocked) else distance[-1]
        speed = min(self._curve_speed_limit(reference_path)[0],
                    math.sqrt(2*p.STATIC_END_DECELERATION*max(0., available-p.STATIC_END_BUFFER)))
        result = self._result(reference_path, False, phase, error)
        result.acc_target_velocity = speed
        details = result.debug_details['static_avoidance']
        details.update(reference_clear_distance_m=float(available), preplanned_entry=self.preplanned,
                       speed_policy_reason='planning_reference_clearance',
                       tracking_speed_limits={'planning_reference_clearance': speed})
        return result

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
        # 지도 점을 정확히 보간하며 생기는 작은 굴곡은 가짜 조향 변화율 제한을 만든다.
        # 최대 변위를 제한하고, 이후 모든 연결 경로를 기존 차체·장애물 기준으로 검사한다.
        smooth = UnivariateSpline(points[:, 1], points[:, 0],
                                  s=len(points)*p.STATIC_PATH_SMOOTHING_RMS**2)
        probe = np.linspace(points[0, 1], points[-1, 1],
                            int(math.ceil((points[-1, 1]-points[0, 1])/.05))+1)
        if np.max(np.abs(smooth(probe)-self.global_curve(probe))) <= p.STATIC_PATH_SMOOTHING_MAX_DEVIATION:
            self.global_curve = smooth
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
        return bool(np.all(self._envelope_mask(xy, yaw, body_margin)))

    def _envelope_mask(self, xy, yaw, body_margin=0.):
        centers = np.array([c(xy[:, 1]) for c in self.curves + [self._global_return_corridor]])
        valid = (xy[:, 0] >= centers.min(axis=0)-.15) & (xy[:, 0] <= centers.max(axis=0)+.15)
        for along in (-body_margin, self.planner.vehicle_length+body_margin):
            for side in (-self.planner.vehicle_half_width-body_margin, self.planner.vehicle_half_width+body_margin):
                x = xy[:, 0] + along*np.cos(yaw) - side*np.sin(yaw)
                y = xy[:, 1] + along*np.sin(yaw) + side*np.cos(yaw)
                valid &= ((x >= np.interp(y, self.envelope_y, self.envelope_low)-.15)
                          & (x <= np.interp(y, self.envelope_y, self.envelope_high)+.15))
        return valid

    def _global_return_corridor(self, y, derivative=0):
        # The original lane is available between obstacles too, not just at exit.
        return self.global_curve(y, derivative)

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
                    result = np.vstack((complete[complete[:,1] < start-1e-6], edge, tail[1:]))
                    options.append((self._tracking_delay(result), end, start, result))
        if not options:
            return complete
        # 복귀 종료 위치를 먼저 고르면 짧고 급한 복귀가 선택되어 다시 제동하게 된다.
        # 기준속도에서 필요한 감속을 먼저 최소화하고, 같을 때만 빠른 복귀를 선택한다.
        _, end, start, result = min(options, key=lambda item: (item[0], item[1]))
        points = [Point(x,y) for x,y in result]
        if self.planner._max_curvature(points) > MAX_CURVATURE:
            return complete
        self.return_start_y, self.handoff_y = float(start), float(end)
        return result

    def _tracking_delay(self, xy):
        """기준속도 대비 곡률·조향 변화·통로 여유 때문에 추가되는 주행시간 [s]."""
        station, _, profiles = self._speed_profile(xy)
        cruise = p.STATIC_PATH_DESIGN_SPEED_KPH/3.6
        limits = np.minimum(cruise, np.minimum.reduce(list(profiles.values())))
        # 속도 제한 직전 감속도 비용에 포함해, 짧고 급한 연결이 유리해지지 않게 한다.
        for i in range(len(limits)-2, -1, -1):
            limits[i] = min(limits[i], math.sqrt(limits[i+1]**2
                + 2*p.STATIC_END_DECELERATION*(station[i+2]-station[i+1])))
        return float(np.sum(np.diff(station)[1:]*(1/np.maximum(limits, .1)-1/cruise)))

    def _smooth_recorded_path(self, complete, obstacles):
        """기록 경로의 작은 흔들림만 제거하고 시작·끝 위치/방향/곡률은 유지한다."""
        y = complete[:, 1]
        original = CubicSpline(y, complete[:, 0])
        best, cost = complete, self._tracking_delay(complete)
        probe = np.linspace(y[0], y[-1], int(math.ceil((y[-1]-y[0])/.025))+1)
        for scale in (1., 2., 4.):
            fit = UnivariateSpline(y, complete[:, 0],
                                   s=len(y)*(scale*p.STATIC_PATH_SMOOTHING_RMS)**2)
            blend = min(4., (y[-1]-y[0])/3.)
            head_end, tail_start = y[0]+blend, y[-1]-blend
            head = connector(self._state(original, y[0]), self._state(fit, head_end))
            tail = connector(self._state(fit, tail_start), self._state(original, y[-1]))
            def curve(value, derivative=0):
                value = np.asarray(value)
                return np.where(value < head_end, head(value, derivative),
                    np.where(value > tail_start, tail(value, derivative), fit(value, derivative)))
            deviation = float(np.max(np.abs(curve(probe)-original(probe))))
            if deviation > p.STATIC_PATH_SMOOTHING_MAX_DEVIATION:
                continue
            candidate = self._edge(curve, y[0], y[-1], obstacles)
            if candidate is None:
                continue
            candidate_cost = self._tracking_delay(candidate)
            if candidate_cost < cost-1e-6:
                best, cost = candidate, candidate_cost
                self.path_smoothing_deviation = deviation
        return best

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

    def collision_free(self, xy, yaw, obstacles, extra=0.0, tracking_reserve=True):
        return bool(np.all(self._collision_mask(xy, yaw, obstacles, extra, tracking_reserve)))

    def _collision_mask(self, xy, yaw, obstacles, extra=0.0, tracking_reserve=True):
        xy, yaw = np.asarray(xy), np.asarray(yaw)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(yaw)
        if not obstacles:
            return valid
        forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
        left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        center = xy + .5*self.planner.vehicle_length*forward
        margin = self.planner.collision_margin + extra
        if tracking_reserve:
            margin += p.STATIC_TRACKING_MARGIN
        length, width = .5*self.planner.vehicle_length+margin, self.planner.vehicle_half_width+margin
        for obj in obstacles:
            if not all(math.isfinite(v) for v in (obj.position.x, obj.position.y, obj.heading_deg, obj.size_x, obj.size_y)):
                return np.zeros(len(xy), dtype=bool)
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
            valid &= ~overlap
        return valid

    def _edge(self, curve, y0, y1, obstacles, body_margin=0., tracking_reserve=True):
        spacing = .1 if tracking_reserve else .025
        ys = np.linspace(y0, y1, max(2, int(math.ceil((y1-y0)/spacing))+1))
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
        extra = max(.12, sweep) if tracking_reserve else sweep
        if not self.collision_free(xy, yaw, obstacles, extra, tracking_reserve):
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
        return search_corridor(self, ego, obstacles, connector, MAX_CURVATURE)

    def _remaining_clear(self, points, yaw, obstacles):
        xy = np.asarray(points)
        if self.recovery_until_y is None:
            return self.collision_free(xy, yaw, obstacles, .12)
        head = xy[:, 1] <= self.recovery_until_y+1e-6
        # Recovery edges are sampled every 2.5cm; retain translation/rotation
        # inflation as well as the full configured 0.5m obstacle clearance.
        return (self.collision_free(xy[head], yaw[head], obstacles, .03, tracking_reserve=False)
                and self.collision_free(xy[~head], yaw[~head], obstacles, .12))

    def _reconnect_measured_pose(self, ego, obstacles):
        """Repair a small tracking discrepancy, never a blocked nominal route."""
        xy = np.array(self.path)
        curve = CubicSpline(xy[:, 1], xy[:, 0])
        start = float(ego.position.y)
        a = (start, ego.position.x, math.cos(ego.yaw)/max(.5, math.sin(ego.yaw)), 0.)
        for distance in (3., 4., 5., 6., 8.):
            index = int(np.searchsorted(xy[:, 1], start+distance))
            if index >= len(xy)-2:
                continue
            end = float(xy[index, 1])
            edge = self._edge(connector(a, self._state(curve, end)), start, end,
                              obstacles, tracking_reserve=False)
            if edge is None:
                continue
            combined = [Point(x, y) for x, y in np.vstack((edge, xy[index+1:]))]
            yaw = np.array([self.planner._path_heading(combined, i) for i in range(len(combined))])
            split = len(edge)
            if (self.planner._max_curvature(combined) > MAX_CURVATURE
                    or not self._envelope_clear(np.array(combined), yaw)
                    or not self.collision_free(np.array(combined[:split]), yaw[:split], obstacles,
                                               .03, tracking_reserve=False)
                    or not self.collision_free(np.array(combined[split:]), yaw[split:], obstacles, .12)):
                continue
            self.path, self.progress = combined, 0
            self.recovery_until_y = end
            self.recovery_count += 1
            return True
        return False

    def _tracking_alignment(self, ego, nearest):
        xy = np.array(self.path[max(0, nearest-2):nearest+3])
        if len(xy) < 2:
            return float(np.linalg.norm(xy[0]-np.asarray(ego.position))), math.pi
        delta = np.diff(xy, axis=0)
        length2 = np.sum(delta*delta, axis=1)
        fraction = np.clip(np.sum((np.asarray(ego.position)-xy[:-1])*delta, axis=1)
                           / np.maximum(length2, 1e-9), 0., 1.)
        projected = xy[:-1]+fraction[:, None]*delta
        distance = np.linalg.norm(projected-np.asarray(ego.position), axis=1)
        index = int(np.argmin(distance))
        # Match the tangent at the projection, not the next sampled waypoint.
        slope = float(np.interp(projected[index, 1], xy[:, 1], np.gradient(xy[:, 0], xy[:, 1])))
        tangent = math.atan2(1., slope)
        heading = abs(math.atan2(math.sin(ego.yaw-tangent), math.cos(ego.yaw-tangent)))
        return float(distance[index]), heading

    def _curve_speed_limit(self, points):
        if len(points) < 3:
            self.curve_speed_limits = {}
            return TRACKING_SPEED, 0.
        start = len(self.path)-len(points)
        is_suffix = (start >= 0 and self.path and points[0] is self.path[start]
                     and points[-1] is self.path[-1])
        if is_suffix:
            if self._speed_profile_path is not self.path:
                self._cached_speed_profile = self._speed_profile(self.path)
                self._speed_profile_path = self.path
            station, curvature, profiles = self._cached_speed_profile
            distance = station[start+1:-1]-station[start]
            curvature = curvature[start:]
            profiles = {name: values[start:] for name, values in profiles.items()}
        else:
            station, curvature, profiles = self._speed_profile(points)
            distance = station[1:-1]
        # Cache the immutable full-path constraints so cropping the local path
        # cannot change a future steering-rate estimate and create a speed jump.
        self.curve_speed_limits = {name: min(TRACKING_SPEED, float(np.min(np.sqrt(
            targets*targets+2*p.STATIC_END_DECELERATION*np.maximum(0., distance-p.STATIC_SPEED_PREVIEW)))))
            for name, targets in profiles.items()}
        preview = curvature[distance <= ALIGNMENT_PREVIEW]
        return min(self.curve_speed_limits.values()), float(np.max(preview)) if len(preview) else 0.

    def _speed_profile(self, points):
        xy = np.asarray(points)
        delta = np.diff(xy, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        chord = np.linalg.norm(xy[2:]-xy[:-2], axis=1)
        twice_area = delta[:-1, 0]*delta[1:, 1]-delta[:-1, 1]*delta[1:, 0]
        signed_curvature = 2*twice_area / np.maximum(lengths[:-1]*lengths[1:]*chord, 1e-9)
        curvature = np.abs(signed_curvature)
        distance = np.cumsum(lengths)[:-1]
        # Apply the requested offset before braking propagation and the other
        # speed constraints. The reference acceleration is no longer a hard cap.
        profiles = {'lateral_acceleration': (
            np.sqrt(p.STATIC_LATERAL_ACCEL/np.maximum(curvature, 1e-6))
            + p.STATIC_CURVATURE_SPEED_OFFSET_KPH/3.6)}
        if self.curves:
            # Near a recorded road edge, extra speed causes the delayed yaw
            # response to swing the nose out of the validated body corridor.
            # Keep the validated low-speed baseline and spend only the measured
            # lateral headroom on faster tracking: extra_v * k * tau * length.
            tangent = np.arctan2(xy[2:, 1]-xy[:-2, 1], xy[2:, 0]-xy[:-2, 0])
            clearance = np.full(len(curvature), np.inf)
            for along in (0., self.planner.vehicle_length):
                for side in (-self.planner.vehicle_half_width, self.planner.vehicle_half_width):
                    x = xy[1:-1, 0]+along*np.cos(tangent)-side*np.sin(tangent)
                    y = xy[1:-1, 1]+along*np.sin(tangent)+side*np.cos(tangent)
                    clearance = np.minimum(clearance, np.minimum(
                        x-np.interp(y, self.envelope_y, self.envelope_low),
                        np.interp(y, self.envelope_y, self.envelope_high)-x))
            tracking = ALIGNMENT_SPEED + np.maximum(0., clearance)/(self.planner.vehicle_length
                        *p.STATIC_STEERING_RESPONSE_TIME*np.maximum(curvature, 1e-6))
            profiles['corridor_tracking'] = tracking
        if len(distance) > 1:
            steering = np.arctan(p.STATIC_WHEELBASE*signed_curvature)
            # Evaluate changes over the controller's minimum 1 m preview, not
            # recording/sample jitter at adjacent 0.1 m points.
            low, high = np.maximum(distance[0], distance-.5), np.minimum(distance[-1], distance+.5)
            slew_per_m = np.abs(np.interp(high, distance, steering)-np.interp(low, distance, steering)) / np.maximum(high-low, 1e-6)
            profiles['steering_rate'] = p.STATIC_STEERING_RATE/np.maximum(slew_per_m, 1e-6)
        return np.r_[0., np.cumsum(lengths)], curvature, profiles

    def _result(self, points, valid, phase, error=None):
        candidate = CandidatePath(layer="recorded")
        candidate.points, candidate.valid = points, valid
        curve_limit, preview_curvature = self._curve_speed_limit(points)
        limits = dict(tracking=min(TRACKING_SPEED, p.STATIC_MAX_SPEED_KPH/3.6), curvature=curve_limit)
        if self.path:
            remaining = sum(a.distance(b) for a,b in zip(points, points[1:]))
            limits['path_end'] = math.sqrt(2*p.STATIC_END_DECELERATION*max(0., remaining-1.))
            if self.recovery_until_y is not None:
                limits['recovery'] = ALIGNMENT_SPEED
            if self.alignment_slow:
                heading_severity = (self.heading_error-ALIGNMENT_HEADING_ERROR) / (
                    ALIGNMENT_FULL_HEADING_ERROR-ALIGNMENT_HEADING_ERROR)
                lateral_severity = (self.cross_track_error-ALIGNMENT_CROSS_TRACK_ERROR) / (
                    ALIGNMENT_FULL_CROSS_TRACK_ERROR-ALIGNMENT_CROSS_TRACK_ERROR)
                severity = float(np.clip(max(heading_severity, lateral_severity), 0., 1.))
                base = min(limits.values())
                limits['alignment'] = base-(max(0., base-ALIGNMENT_SPEED))*severity
        if not valid:
            limits['blocked'] = 0.
        speed_reason = min(limits, key=limits.get)
        speed = limits[speed_reason]
        return LatticePlanningResult(points, active=bool(self.path), mode="static_obstacle",
            obstacle_detected=True, acc_override=True, acc_target_velocity=speed,
            candidates=[candidate], reason="static_recorded_"+phase,
            debug_details={"static_avoidance": dict(phase=phase, path_validated=valid,
                planner="recorded_route_graph", boundary_source="recorded_swept_envelope_not_map",
                live_preview=self.interactive_preview,
                live_drive_committed=self.live_drive_committed,
                preview_ego_on_route=self.preview_ego_on_route,
                selected_routes=self.selected_names, recording_audit=self.audit, error=error,
                corridor_search=dict(self.search_details),
                heading_error_deg=math.degrees(self.heading_error), tracking_error_m=self.tracking_error,
                cross_track_error_m=self.cross_track_error, alignment_slow=self.alignment_slow,
                preview_curvature=preview_curvature, speed_policy_reason=speed_reason,
                tracking_speed_limits=dict(limits),
                curve_speed_limits=dict(self.curve_speed_limits),
                curvature_speed_offset_kph=p.STATIC_CURVATURE_SPEED_OFFSET_KPH,
                path_design_speed_kph=p.STATIC_PATH_DESIGN_SPEED_KPH,
                path_smoothing_deviation_m=self.path_smoothing_deviation,
                recovery_until_y=self.recovery_until_y, recovery_count=self.recovery_count,
                return_start_y=self.return_start_y, handoff_y=self.handoff_y,
                preplanned_complete=bool(self.path), plan_count=self.plan_count)})

    def plan(self, ego, reference_path, obstacles, prediction_path=None):
        if (not self.interactive_preview and (
                (not self.active and self.past_rejoin(ego) and len(reference_path) >= 2)
                or self._early_handoff_ready(ego, reference_path, obstacles))):
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
                    return self._waiting_result(ego, reference_path, obstacles, "planning")
                try:
                    worker = self.pending.result()
                except Exception as exc:
                    self.pending = None
                    return self._result(reference_path, False, "planning_error", str(exc))
                self.pending = None
                self.last_attempt = time.monotonic()
                if not worker.path:
                    return self._result(reference_path, False, "no_complete_route")
                # All adopted lambdas close over the isolated completed worker.
                for name, value in vars(worker).items():
                    if name not in ('executor', 'pending', 'pending_key', 'preplanned', 'last_attempt'):
                        setattr(self, name, value)
            elif self.executor is not None:
                if time.monotonic()-self.last_attempt >= 1.:
                    route = list(prediction_path if prediction_path is not None else reference_path)
                    self._submit(ego, route, obstacles)
                return self._waiting_result(ego, reference_path, obstacles, "planning")
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
        if self.interactive_preview and self.active:
            # Do not advance progress or reject a useful full-route preview
            # because the editor's stationary Ego was moved off the route.
            # The same path has already passed the production envelope,
            # curvature and obstacle checks in the isolated worker.
            return self._result(list(self.path), True, "preview")
        committed_approach = self.committed_approach_result(
            ego, reference_path
        )
        if committed_approach is not None:
            return committed_approach
        xy = np.array(self.path)
        nearest = int(np.argmin(np.linalg.norm(xy[max(0,self.progress-2):]-np.asarray(ego.position), axis=1)))+max(0,self.progress-2)
        error = ego.position.distance(self.path[nearest])
        if error > p.STATIC_RESET_DISTANCE or ego.position.y < self.path[0].y-3:
            self.reset()
            return self._result(reference_path, False, "reset_after_teleport")
        self.progress = max(self.progress, nearest)
        self.cross_track_error, self.heading_error = self._tracking_alignment(ego, nearest)
        self.tracking_error = error
        if self.heading_error > ALIGNMENT_HEADING_ERROR or self.cross_track_error > ALIGNMENT_CROSS_TRACK_ERROR:
            self.alignment_slow = True
        elif (self.heading_error < .5*ALIGNMENT_HEADING_ERROR
                and self.cross_track_error < .5*ALIGNMENT_CROSS_TRACK_ERROR):
            self.alignment_slow = False
        points = self.path[max(0, self.progress-1):]
        if obstacles:
            self.known = list(obstacles)
        yaw = np.array([self.planner._path_heading(points, i) for i in range(len(points))]) if len(points) >= 2 else np.array([])
        pose, heading = np.array([ego.position]), np.array([ego.yaw])
        nominal_pose_clear = self.collision_free(pose, heading, self.known)
        if (self.recovery_until_y is not None and ego.position.y > self.recovery_until_y
                and nominal_pose_clear):
            self.recovery_until_y = None
        failure = None
        if len(points) < 3 or error >= .75:
            failure = 'tracking_error'
        elif not self._envelope_clear(np.array([ego.position]), np.array([ego.yaw])):
            failure = 'recorded_envelope'
        elif not self.collision_free(pose, heading, self.known, tracking_reserve=False):
            failure = 'ego_collision'
        elif not self._remaining_clear(points, yaw, self.known):
            failure = 'remaining_path_collision'
        elif not nominal_pose_clear and self.recovery_until_y is None:
            if (abs(ego.velocity) > ALIGNMENT_SPEED + .2 or error > .25 or self.heading_error > math.radians(12.)
                    or not self._reconnect_measured_pose(ego, self.known)):
                failure = 'alignment_recovery_unavailable'
            else:
                points = self.path
        phase = 'recover_alignment' if self.recovery_until_y is not None else 'follow'
        return self._result(points, failure is None, phase if failure is None else 'blocked', failure)

    def _prepare(self, ego, route, obstacles):
        self.search_details = {}
        self.search_progress = 0
        self._setup(route, ego)
        self.known = list(obstacles)
        self.plan_count += 1
        complete = self._search(ego, obstacles)
        if complete is None:
            return None
        return self._smooth_return(self._smooth_recorded_path(complete, obstacles), obstacles)
