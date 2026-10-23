"""Hold a right-side lattice path and recover against a saved global route."""

import bisect
import math
import numpy as np

from ...localization.point import Point
from ...vehicle_state import VehicleState
from ...config.parameters import lattice as p
from ..lattice.structures import CandidatePath, LatticePlanningResult


class StaticAvoidance:
    def __init__(self, planner):
        self.planner = planner
        self.reset()

    def reset(self):
        self.path = []
        self.progress = 0
        self.return_index = 0
        self.rejoin_index = 0
        self.capture_index = 0
        self.offset = 0.0
        self.obstacles = {}
        self.route = []
        self._route_segments = None
        self.start_lateral = 0.0
        self.right_boundary = p.STATIC_RIGHT_BOUNDARY
        self.left_boundary = p.STATIC_LEFT_BOUNDARY
        self.diagnostics = {}
        self.anchor = None
        self._blocked_key = None
        self._blocked_result = None
        self._entry_path = []
        self._candidates = []
        self._selected_index = 0
        self._recovery_ready = False
        self._recovery_key = None

    @property
    def active(self):
        return bool(self.path)

    def _remember(self, obstacles):
        for obj in obstacles:
            key = (getattr(obj, "unique_id", None), getattr(obj, "name", ""),
                   round(obj.position.x, 2), round(obj.position.y, 2))
            self.obstacles[key] = obj

    @staticmethod
    def _distances(points):
        distances = [0.0]
        for start, end in zip(points, points[1:]):
            distances.append(distances[-1] + start.distance(end))
        return distances

    @staticmethod
    def _project(point, route, distances):
        best = (float("inf"), 0.0)
        for i, (a, b) in enumerate(zip(route, route[1:])):
            dx, dy = b.x - a.x, b.y - a.y
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-8:
                continue
            ratio = max(0.0, min(1.0, ((point.x - a.x) * dx + (point.y - a.y) * dy) / length_sq))
            error = math.hypot(point.x - a.x - ratio * dx, point.y - a.y - ratio * dy)
            if error < best[0]:
                best = (error, distances[i] + ratio * math.sqrt(length_sq))
        return best[1]

    @staticmethod
    def _pose(route, distances, distance):
        index = min(len(route) - 2, max(0, bisect.bisect_right(distances, distance) - 1))
        a, b = route[index:index + 2]
        length = max(1e-8, distances[index + 1] - distances[index])
        ratio = max(0.0, min(1.0, (distance - distances[index]) / length))
        before, after = route[max(0, index - 3)], route[min(len(route) - 1, index + 4)]
        return Point(a.x + ratio * (b.x - a.x), a.y + ratio * (b.y - a.y)), math.atan2(after.y - before.y, after.x - before.x)

    def _build_bypass(self, route, ego, obstacles):
        """Prefer a safe long leg inside the narrow envelope; retain it verbatim."""
        half = (self.planner.num_offsets - 1) // 2
        span = min(half * self.planner.lateral_offset_step, self._offset_limit())
        offsets = tuple(np.linspace(-span, span, self.planner.num_offsets))
        result = self.planner.plan(ego, route, obstacles, prefer_right=True, offsets=offsets)
        if not result.obstacle_detected:
            return ()
        self._candidates = result.candidates
        self.diagnostics['lateral_candidates'] = len(offsets)
        self.diagnostics.update(allowed_right_offset=self._offset_limit(),
                                right_body_boundary=self.right_boundary)
        distances = self._distances(route)
        start_s = self._project(ego.position, route, distances)
        for candidate in self._candidates:
            if (candidate.offset > 0.0 or candidate.offset < -self._offset_limit()
                    or not candidate.valid or not candidate.points):
                candidate.valid = False
                continue
            candidate.valid = (self.planner._max_curvature(candidate.points) <= p.STATIC_MAX_CURVATURE
                               and self._clear(candidate.points, obstacles, ego, p.STATIC_TRACKING_MARGIN))
            if candidate.valid and candidate.layer == 'short':
                # Check clearance beside the remaining visible obstacle cluster,
                # without adding that probe to the path the vehicle will follow.
                end_s = self._project(candidate.points[-1], route, distances)
                probe = list(candidate.points)
                for s in np.arange(end_s + p.STATIC_SAMPLE_SPACING,
                                   min(distances[-1], start_s + self.planner.obstacle_detection_distance),
                                   p.STATIC_SAMPLE_SPACING):
                    center, yaw = self._pose(route, distances, s)
                    probe.append(Point(center.x - candidate.offset * math.sin(yaw),
                                       center.y + candidate.offset * math.cos(yaw)))
                candidate.valid = self._clear(probe, obstacles, ego, p.STATIC_TRACKING_MARGIN)
        valid_long = [c for c in self._candidates if c.valid and c.layer == 'long']
        valid_short = [c for c in self._candidates if c.valid and c.layer == 'short']
        # A truncated preview is not a full long-horizon avoidance leg.
        full_long_preview = distances[-1] >= self.planner.long_lookahead_distance
        valid = valid_long if valid_long and full_long_preview else valid_short
        self.diagnostics.update(valid_long_candidates=len(valid_long),
                                valid_short_candidates=len(valid_short))
        if not valid:
            self.diagnostics['blocked_by'] = 'no_collision_free_right_lattice'
            return None
        selected = min(valid, key=lambda c: (abs(c.offset), c.cost))
        self._selected_index = next(i for i, c in enumerate(self._candidates) if c is selected)
        self.diagnostics.update(selected_layer=selected.layer,
                                selection_reason='long_preferred' if selected.layer == 'long' else 'short_fallback')
        self.planner.last_selected_offset = selected.offset
        self._entry_path = list(selected.points)
        self.diagnostics.pop('blocked_by', None)
        self.diagnostics.pop('blocker_id', None)
        self.capture_index = len(selected.points)
        return list(selected.points), len(selected.points), len(selected.points), selected.offset

    def _attach_return(self, ego, obstacles):
        """Connect from the untouched lattice endpoint to the nearest feasible route point."""
        prefix = self._entry_path
        endpoint = prefix[-1]
        endpoint_yaw = self.planner._path_heading(prefix, len(prefix) - 1)
        key = (round(ego.position.x, 1), round(ego.position.y, 1), round(ego.yaw, 2),
               tuple(self.obstacles))
        if key == self._recovery_key:
            return False
        self._recovery_key = key
        route = self.route
        distances = self._distances(route)
        end_s = self._project(endpoint, route, distances)
        cosine, sine = math.cos(endpoint_yaw), math.sin(endpoint_yaw)
        targets = []
        for index, point in enumerate(route[:-1]):
            if distances[index] <= end_s:
                continue
            dx, dy = point.x - endpoint.x, point.y - endpoint.y
            distance = math.hypot(dx, dy)
            forward = cosine * dx + sine * dy
            if forward >= 2.0 and distance <= p.STATIC_RETURN_MAX_DISTANCE:
                targets.append((distance, index))
        for distance, index in sorted(targets):
            target = route[index]
            target_yaw = math.atan2(route[index + 1].y - target.y, route[index + 1].x - target.x)
            for lead_distance in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
                lead = list(prefix)
                for s in np.arange(end_s + p.STATIC_SAMPLE_SPACING, end_s + lead_distance + 1e-6,
                                   p.STATIC_SAMPLE_SPACING):
                    center, yaw = self._pose(route, distances, s)
                    lead.append(Point(center.x - self.offset * math.sin(yaw),
                                      center.y + self.offset * math.cos(yaw)))
                origin = lead[-1]
                yaw = self.planner._path_heading(lead, len(lead) - 1)
                dx, dy = target.x - origin.x, target.y - origin.y
                forward = math.cos(yaw) * dx + math.sin(yaw) * dy
                if forward < 2.0:
                    continue
                relative_yaw = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
                if abs(relative_yaw) > math.pi / 3:
                    continue
                coefficients = self.planner._solve_polynomial(
                    forward, -math.sin(yaw) * dx + math.cos(yaw) * dy, relative_yaw)
                if coefficients is None:
                    continue
                pose = VehicleState(origin.x, origin.y, yaw, ego.velocity)
                recovery = self.planner._sample_global_path(pose, forward, coefficients)
                recovery[-1] = target
                tail_index = min(len(route), bisect.bisect_right(distances, distances[index] + 8.0))
                points = lead + recovery[1:] + route[index + 1:tail_index]
                join_index = len(lead) + len(recovery) - 2
                if self.planner._max_curvature(points[len(prefix) - 3:]) > p.STATIC_RECOVERY_MAX_CURVATURE:
                    continue
                remaining = points[max(0, self.progress - 1):]
                if not self._clear(remaining, obstacles, ego, p.STATIC_TRACKING_MARGIN):
                    continue
                self.path = points
                self.return_index = len(prefix) - 1
                self.capture_index = self.rejoin_index = join_index
                self._recovery_ready = True
                self.diagnostics.pop('recovery_wait', None)
                self.diagnostics.update(
                    return_distance=round(self._distances(points[self.return_index:join_index + 1])[-1], 2),
                    rejoin_distance=round(distance, 2),
                    return_start=[endpoint.x, endpoint.y],
                    rejoin_point=[target.x, target.y], rejoin_route_index=index)
                return True
        self.diagnostics['recovery_wait'] = 'no_collision_free_endpoint_return'
        return False

    def _offset_limit(self):
        return p.STATIC_MAX_RIGHT_OFFSET

    def _clear(self, points, obstacles, ego, tracking_margin=0.0):
        if len(points) < 2:
            return False
        if not self._right_only(points):
            self.diagnostics['blocked_by'] = 'forbidden_side'
            return False
        if not self._within_corridor(points, ego, tracking_margin):
            self.diagnostics['blocked_by'] = 'road_boundary'
            return False
        # Check the actual pose as well as the planned swept footprint.
        if any(self.planner._obb_collision(ego.position, ego.yaw, obj) for obj in obstacles):
            self.diagnostics['blocked_by'] = 'ego_collision'
            return False
        radius = math.hypot(self.planner.vehicle_length, self.planner.vehicle_half_width) + 2.0 * (self.planner.collision_margin + tracking_margin)
        for i, point in enumerate(points):
            yaw = self.planner._path_heading(points, i)
            for obj in obstacles:
                if point.distance(obj.position) > radius + self.planner._obstacle_radius(obj):
                    continue
                if self.planner._obb_collision(point, yaw, obj, extra_margin=tracking_margin):
                    self.diagnostics.update(blocked_by='obstacle_collision', blocker_id=getattr(obj, 'unique_id', None))
                    return False
        return True

    def _set_route(self, route, ego):
        self.route = list(route)
        starts = np.array([(point.x, point.y) for point in route[:-1]])
        ends = np.array([(point.x, point.y) for point in route[1:]])
        segments = ends - starts
        lengths_sq = np.einsum("ij,ij->i", segments, segments)
        valid = lengths_sq > 1e-8
        self._route_segments = (starts[valid], segments[valid], lengths_sq[valid])
        self.start_lateral = max(0.0, float(self._laterals([ego.position])[0]))
        self.left_boundary = max(p.STATIC_LEFT_BOUNDARY, max(self._laterals(
            self._corners(ego.position, ego.yaw, p.STATIC_TRACKING_MARGIN))))

    def _laterals(self, points):
        starts, segments, lengths_sq = self._route_segments
        xy = np.array([(point.x, point.y) for point in points])
        delta = xy[:, None, :] - starts[None, :, :]
        ratio = np.clip(np.einsum("ijk,jk->ij", delta, segments) / lengths_sq, 0.0, 1.0)
        residual = delta - ratio[:, :, None] * segments
        nearest = np.argmin(np.einsum("ijk,ijk->ij", residual, residual), axis=1)
        errors = residual[np.arange(len(points)), nearest]
        directions = segments[nearest] / np.sqrt(lengths_sq[nearest, None])
        return directions[:, 0] * errors[:, 1] - directions[:, 1] * errors[:, 0]

    def _right_only(self, points):
        if self._route_segments is None:
            return False
        # Allow recovery from an initial left tracking error, never a new left bypass.
        upper_bound = self.start_lateral
        for lateral in self._laterals(points):
            if (lateral > upper_bound + p.STATIC_SIDE_TOLERANCE
                    or lateral < -self._offset_limit() - p.STATIC_SIDE_TOLERANCE):
                return False
            upper_bound = max(0.0, min(upper_bound, lateral))
        return True

    def _corners(self, point, yaw, extra_margin=0.0):
        margin = self.planner.collision_margin + extra_margin
        half_width = self.planner.vehicle_half_width + margin
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return [Point(point.x + longitudinal * cosine - lateral * sine,
                      point.y + longitudinal * sine + lateral * cosine)
                for longitudinal in (-margin, self.planner.vehicle_length + margin)
                for lateral in (-half_width, half_width)]

    def _within_corridor(self, points, ego, extra_margin, boundary_margin=None):
        margin = p.STATIC_BOUNDARY_MARGIN if boundary_margin is None else boundary_margin
        boundary_extra = margin - self.planner.collision_margin
        corners = self._corners(ego.position, ego.yaw, boundary_extra)
        for i, point in enumerate(points):
            corners.extend(self._corners(point, self.planner._path_heading(points, i), boundary_extra + extra_margin))
        laterals = self._laterals(corners)
        return bool(np.all(laterals <= self.left_boundary + 1e-6)
                    and np.all(laterals >= self.right_boundary - 1e-6))

    def _result(self, points, valid, phase):
        layer = self._candidates[self._selected_index].layer if self._candidates else 'short'
        selected = CandidatePath(offset=self.offset, layer=layer)
        selected.points = points
        selected.valid = valid
        # Do not interpret the tiny tracking connector as a new sharp turn;
        # its length changes each tick while the retained route does not.
        curvature_path = self.path[self.progress:] if self.path else points
        selected.curvature_cost = self.planner._max_curvature(curvature_path)
        speed = min(p.STATIC_MAX_SPEED_KPH / 3.6,
                    math.sqrt(p.STATIC_LATERAL_ACCEL / max(1e-4, selected.curvature_cost))
                    + p.STATIC_CURVATURE_SPEED_OFFSET_KPH / 3.6) if valid else 0.0
        # An unavailable return is not permission to drive beyond the held leg.
        remaining = self._distances(self.path[self.progress:])[-1] if self.path else 0.0
        speed = min(speed, math.sqrt(2.0 * p.STATIC_END_DECELERATION *
                                    max(0.0, remaining - p.STATIC_END_BUFFER)))
        candidates = list(self._candidates)
        if candidates:
            candidates[self._selected_index] = selected
        else:
            candidates = [selected]
        return LatticePlanningResult(
            points, active=True, obstacle_detected=True, mode="static_obstacle",
            acc_override=True, acc_target_velocity=speed, selected_offset=self.offset,
            candidates=candidates,
            reason="static_{}".format(phase),
            debug_details={"static_avoidance": {"phase": phase, "path_validated": valid,
                                               **self.diagnostics}},
        )

    def _continue(self, ego, obstacles, reference_path):
        nearest = min(range(max(0, self.progress - 2), len(self.path)),
                      key=lambda i: ego.position.distance(self.path[i]))
        error = ego.position.distance(self.path[nearest])
        if error > p.STATIC_RESET_DISTANCE:
            self.reset()
            return None
        self.progress = max(self.progress, nearest)
        target_yaw = self.planner._path_heading(self.path, nearest)
        heading_error = abs(math.atan2(math.sin(ego.yaw - target_yaw), math.cos(ego.yaw - target_yaw)))
        if (self._recovery_ready and self.progress >= self.rejoin_index and error < 0.6
                and heading_error < math.radians(15.0) and len(reference_path) >= 2):
            self.reset()
            return LatticePlanningResult(reference_path, mode="static_obstacle", reason="static_rejoined")
        self._remember(obstacles)
        if not self._recovery_ready:
            entry_remaining = self._distances(self.path[nearest:len(self._entry_path)])[-1]
            if (entry_remaining <= p.STATIC_RECOVERY_LOOKAHEAD
                    and error <= p.STATIC_MAX_TRACKING_ERROR):
                self.diagnostics['recovery_trigger'] = 'lattice_endpoint'
                self._attach_return(ego, list(self.obstacles.values()))
        # Keep the original tangent, including one point behind the projection.
        # A tiny connector from the measured pose invents a sharp heading and
        # falsely swings the collision box outside the corridor. The actual
        # measured body is checked independently by _clear().
        points = self.path[max(0, nearest - 1):]
        valid = (nearest < len(self.path) - 3 and error <= p.STATIC_MAX_TRACKING_ERROR
                 and self._clear(points, list(self.obstacles.values()), ego))
        if valid:
            self.diagnostics.pop('blocked_by', None)
            self.diagnostics.pop('blocker_id', None)
        phase = "return" if self._recovery_ready and self.progress >= self.return_index else "avoid"
        if self._recovery_ready and self.progress >= self.capture_index:
            phase = "align"
        return self._result(points, valid, phase if valid else "blocked")

    def plan(self, ego, reference_path, obstacles, prediction_path=None):
        if self.anchor is not None and ego.position.distance(self.anchor) > p.STATIC_RESET_DISTANCE and not self.active:
            self.reset()
        if self.active:
            result = self._continue(ego, obstacles, reference_path)
            if result is not None:
                return result
            return LatticePlanningResult(reference_path, reason="static_reset_after_displacement")
        route = prediction_path if prediction_path is not None else reference_path
        if len(route) < 2 or all(a.distance(b) < 1e-6 for a, b in zip(route, route[1:])):
            return LatticePlanningResult(reference_path, reason="path_too_short")
        self._set_route(route, ego)
        if self.anchor is None:
            self.anchor = ego.position
        self.diagnostics = {'boundary_source': 'configured_offset', 'planner': 'right_lattice_cached_recovery'}
        self._remember(obstacles)
        known = list(self.obstacles.values())
        key = (round(ego.position.x, 1), round(ego.position.y, 1), round(ego.yaw, 2),
               len(route), round(route[-1].x, 1), round(route[-1].y, 1), tuple(self.obstacles))
        if key == self._blocked_key:
            return self._blocked_result
        complete = self._build_bypass(route, ego, known)
        if complete == ():
            return LatticePlanningResult(reference_path, mode="static_obstacle", reason="no_static_obstacle")
        if (complete is not None
                and self.planner._max_curvature(complete[0]) <= p.STATIC_MAX_CURVATURE
                and self._clear(complete[0], known, ego, tracking_margin=p.STATIC_TRACKING_MARGIN)):
            self.path, self.return_index, self.rejoin_index, self.offset = complete
            return self._result(self.path, True, "avoid")
        self.diagnostics.setdefault('blocked_by', 'curvature_limit')
        result = LatticePlanningResult(reference_path, mode="static_obstacle", reason="static_no_safe_lattice",
            obstacle_detected=True, acc_override=True, acc_target_velocity=0.0,
            debug_details={"static_avoidance": {"phase": "blocked", "path_validated": False, **self.diagnostics}})
        self._blocked_key, self._blocked_result = key, result
        return result
