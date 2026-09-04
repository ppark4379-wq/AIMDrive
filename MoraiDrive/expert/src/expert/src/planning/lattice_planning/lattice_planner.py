#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT-object based lattice planner ported from 26_Winter_Team2.

The source planner uses a LiDAR occupancy grid.  GT_BEV already receives
ground-truth objects on /Object_topic, so this port evaluates the same lateral
polynomial candidates directly against obstacle positions and sizes.
"""

import math

import numpy as np

from ...localization.point import Point
from . import parameters
from .structures import CandidatePath, LatticePlanningResult, PolynomialCoefficients


class LatticePlanner:
    def __init__(
        self,
        num_offsets=parameters.NUM_OFFSETS,
        lateral_offset_step=parameters.LATERAL_OFFSET_STEP,
        sample_spacing=parameters.SAMPLE_SPACING,
        short_lookahead_distance=parameters.SHORT_LOOKAHEAD_DISTANCE,
        long_lookahead_distance=parameters.LONG_LOOKAHEAD_DISTANCE,
        obstacle_detection_distance=parameters.OBSTACLE_DETECTION_DISTANCE,
        obstacle_path_threshold=parameters.OBSTACLE_PATH_THRESHOLD,
        vehicle_length=parameters.VEHICLE_LENGTH,
        vehicle_width=parameters.VEHICLE_WIDTH,
        collision_margin=parameters.COLLISION_MARGIN,
        right_preference_weight=parameters.RIGHT_PREFERENCE_WEIGHT,
    ):
        if num_offsets < 3 or num_offsets % 2 == 0:
            raise ValueError("num_offsets must be an odd integer >= 3")

        self.num_offsets = int(num_offsets)
        self.lateral_offset_step = float(lateral_offset_step)
        self.sample_spacing = float(sample_spacing)
        self.short_lookahead_distance = float(short_lookahead_distance)
        self.long_lookahead_distance = float(long_lookahead_distance)
        self.obstacle_detection_distance = float(obstacle_detection_distance)
        self.obstacle_path_threshold = float(obstacle_path_threshold)
        self.vehicle_length = float(vehicle_length)
        self.vehicle_half_width = float(vehicle_width) * 0.5
        self.collision_margin = float(collision_margin)
        self.right_preference_weight = float(right_preference_weight)
        self.last_selected_offset = 0.0

    def plan(
        self,
        vehicle_state,
        reference_path,
        obstacles,
        prefer_right=True,
        longitudinal_speed_scale=parameters.LONG_OBSTACLE_SPEED_SCALE,
        inactive_reason="no_static_obstacle",
        active_reason="avoid_static_obstacle_right",
        mode="static_obstacle",
    ):
        if len(reference_path) < 2:
            return LatticePlanningResult(reference_path, reason="path_too_short")

        relevant_obstacles = self._find_relevant_obstacles(
            vehicle_state, reference_path, obstacles
        )
        if not relevant_obstacles:
            return LatticePlanningResult(
                reference_path,
                mode=mode,
                reason=inactive_reason,
            )

        long_candidates = self._generate_candidates(
            vehicle_state,
            reference_path,
            layer="long",
            lookahead_distance=self.long_lookahead_distance,
        )
        short_candidates = self._generate_candidates(
            vehicle_state,
            reference_path,
            layer="short",
            lookahead_distance=self.short_lookahead_distance,
        )
        self._evaluate_candidates(long_candidates, relevant_obstacles, add_policy_cost=False)
        self._evaluate_candidates(
            short_candidates,
            relevant_obstacles,
            add_policy_cost=prefer_right,
        )
        candidates = long_candidates + short_candidates

        long_center = min(long_candidates, key=lambda candidate: abs(candidate.offset))
        long_obstacle_detected = not long_center.valid
        valid_candidates = [
            candidate for candidate in short_candidates if candidate.valid
        ]
        if not valid_candidates:
            return LatticePlanningResult(
                reference_path,
                obstacle_detected=True,
                long_obstacle_detected=long_obstacle_detected,
                longitudinal_speed_scale=longitudinal_speed_scale,
                mode=mode,
                candidates=candidates,
                reason="all_candidates_blocked",
            )

        best_path = min(valid_candidates, key=lambda candidate: candidate.cost)
        self.last_selected_offset = best_path.offset
        return LatticePlanningResult(
            best_path.points,
            active=True,
            obstacle_detected=True,
            long_obstacle_detected=long_obstacle_detected,
            longitudinal_speed_scale=longitudinal_speed_scale,
            mode=mode,
            selected_offset=best_path.offset,
            candidates=candidates,
            reason=active_reason,
        )

    def _find_relevant_obstacles(self, vehicle_state, reference_path, obstacles):
        relevant = []
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        ego_x = vehicle_state.position.x
        ego_y = vehicle_state.position.y

        for obstacle in obstacles:
            dx = obstacle.position.x - ego_x
            dy = obstacle.position.y - ego_y
            longitudinal = cos_yaw * dx + sin_yaw * dy
            if longitudinal <= 0.0 or longitudinal > self.obstacle_detection_distance:
                continue

            obstacle_radius = self._obstacle_radius(obstacle)
            path_distance = self._distance_to_path(obstacle.position, reference_path)
            if path_distance <= self.obstacle_path_threshold + obstacle_radius:
                relevant.append(obstacle)
        return relevant

    def _generate_candidates(
        self, vehicle_state, reference_path, layer, lookahead_distance
    ):
        goal_point, goal_heading = self._find_goal(
            reference_path, lookahead_distance
        )

        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        dx = goal_point.x - vehicle_state.position.x
        dy = goal_point.y - vehicle_state.position.y
        goal_x = cos_yaw * dx + sin_yaw * dy
        base_goal_y = -sin_yaw * dx + cos_yaw * dy
        relative_goal_yaw = self._normalize_angle(goal_heading - vehicle_state.yaw)

        candidates = []
        half_count = (self.num_offsets - 1) // 2
        for index in range(self.num_offsets):
            # The path-normal convention makes negative offset the route-right side.
            offset = (index - half_count) * self.lateral_offset_step
            goal_y = base_goal_y + offset
            candidate = CandidatePath(offset=offset, layer=layer)
            coefficients = self._solve_polynomial(
                goal_x, goal_y, relative_goal_yaw
            )
            if coefficients is None:
                candidate.valid = False
                candidate.cost = float("inf")
            else:
                candidate.points = self._sample_global_path(
                    vehicle_state, goal_x, coefficients
                )
                if len(candidate.points) < 2:
                    candidate.valid = False
                    candidate.cost = float("inf")
            candidates.append(candidate)
        return candidates

    def _find_goal(self, path, target_distance):
        accumulated = 0.0
        for index in range(1, len(path)):
            segment = path[index].distance(path[index - 1])
            accumulated += segment
            if accumulated >= target_distance:
                heading = math.atan2(
                    path[index].y - path[index - 1].y,
                    path[index].x - path[index - 1].x,
                )
                return path[index], heading

        heading = math.atan2(
            path[-1].y - path[-2].y,
            path[-1].x - path[-2].x,
        )
        return path[-1], heading

    @staticmethod
    def _solve_polynomial(goal_x, goal_y, goal_yaw):
        if goal_x < 2.0:
            return None

        yaw_limit = math.radians(85.0)
        goal_yaw = max(-yaw_limit, min(yaw_limit, goal_yaw))
        x2 = goal_x * goal_x
        x3 = x2 * goal_x
        x4 = x3 * goal_x
        x5 = x4 * goal_x
        goal_slope = math.tan(goal_yaw)

        # Closed-form solution for y(0)=y'(0)=y''(0)=0 and
        # y(X)=Y, y'(X)=goal_slope, y''(X)=0.  This is equivalent to
        # the Eigen solve in the source planner without a LAPACK dependency.
        a3 = (10.0 * goal_y - 4.0 * goal_slope * goal_x) / x3
        a4 = (-15.0 * goal_y + 7.0 * goal_slope * goal_x) / x4
        a5 = (6.0 * goal_y - 3.0 * goal_slope * goal_x) / x5
        return PolynomialCoefficients(a3=a3, a4=a4, a5=a5)

    def _sample_global_path(self, vehicle_state, goal_x, coefficients):
        sample_count = max(2, int(math.ceil(goal_x / self.sample_spacing)))
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        points = []
        for local_x in np.linspace(0.0, goal_x, sample_count + 1):
            local_y = (
                coefficients.a3 * local_x ** 3
                + coefficients.a4 * local_x ** 4
                + coefficients.a5 * local_x ** 5
            )
            global_x = vehicle_state.position.x + cos_yaw * local_x - sin_yaw * local_y
            global_y = vehicle_state.position.y + sin_yaw * local_x + cos_yaw * local_y
            points.append(Point(global_x, global_y))
        return points

    def _evaluate_candidates(self, candidates, obstacles, add_policy_cost):
        for candidate in candidates:
            if not candidate.valid:
                continue

            min_clearance = float("inf")
            for point_index, point in enumerate(candidate.points):
                path_heading = self._path_heading(candidate.points, point_index)
                for obstacle in obstacles:
                    vehicle_radius = math.hypot(
                        self.vehicle_length * 0.5,
                        self.vehicle_half_width,
                    )
                    clearance = (
                        point.distance(obstacle.position)
                        - vehicle_radius
                        - self._obstacle_radius(obstacle)
                    )
                    min_clearance = min(min_clearance, clearance)
                    if self._obb_collision(point, path_heading, obstacle):
                        candidate.valid = False
                        candidate.cost = float("inf")
                        break
                if not candidate.valid:
                    break

            if not candidate.valid:
                continue

            candidate.obstacle_cost = 1.0 / max(0.1, min_clearance)
            candidate.curvature_cost = self._max_curvature(candidate.points)
            candidate.offset_cost = abs(candidate.offset)
            candidate.offset_change_cost = abs(
                candidate.offset - self.last_selected_offset
            )

            # Added policy cost for the traffic-light/static-obstacle section.
            # Any collision-free right path beats center/left; collision validity
            # remains a hard safety constraint and is never overridden by this cost.
            if add_policy_cost:
                if candidate.offset < -1e-6:
                    candidate.right_preference_cost = 0.0
                elif candidate.offset > 1e-6:
                    candidate.right_preference_cost = 2.0
                else:
                    candidate.right_preference_cost = 1.0

        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        if not valid_candidates:
            return

        max_obstacle = max(candidate.obstacle_cost for candidate in valid_candidates) or 1.0
        max_curvature = max(
            candidate.curvature_cost for candidate in valid_candidates
        ) or 1.0
        max_offset = max(candidate.offset_cost for candidate in valid_candidates) or 1.0
        max_offset_change = max(
            candidate.offset_change_cost for candidate in valid_candidates
        ) or 1.0

        for candidate in valid_candidates:
            candidate.cost = (
                0.40 * candidate.obstacle_cost / max_obstacle
                + parameters.CURVATURE_COST_WEIGHT
                * candidate.curvature_cost / max_curvature
                + 0.15 * candidate.offset_cost / max_offset
                + 0.15 * candidate.offset_change_cost / max_offset_change
                + self.right_preference_weight * candidate.right_preference_cost
            )

    @staticmethod
    def _max_curvature(points):
        """Return maximum geometric curvature [1/m] along a sampled path."""
        max_curvature = 0.0
        for index in range(1, len(points) - 1):
            first = points[index - 1]
            middle = points[index]
            last = points[index + 1]
            side_a = middle.distance(first)
            side_b = last.distance(middle)
            side_c = last.distance(first)
            denominator = side_a * side_b * side_c
            if denominator < 1e-9:
                continue
            twice_area = abs(
                (middle.x - first.x) * (last.y - first.y)
                - (middle.y - first.y) * (last.x - first.x)
            )
            curvature = 2.0 * twice_area / denominator
            max_curvature = max(max_curvature, curvature)
        return max_curvature

    @staticmethod
    def _path_heading(points, index):
        if index + 1 < len(points):
            start, end = points[index], points[index + 1]
        else:
            start, end = points[index - 1], points[index]
        return math.atan2(end.y - start.y, end.x - start.x)

    def _obb_collision(self, rear_axle_point, vehicle_yaw, obstacle):
        """Separating-axis test between vehicle and static-obstacle OBBs."""
        vehicle_forward = (math.cos(vehicle_yaw), math.sin(vehicle_yaw))
        vehicle_left = (-vehicle_forward[1], vehicle_forward[0])
        vehicle_center = (
            rear_axle_point.x + 0.5 * self.vehicle_length * vehicle_forward[0],
            rear_axle_point.y + 0.5 * self.vehicle_length * vehicle_forward[1],
        )
        vehicle_extents = (
            0.5 * self.vehicle_length + self.collision_margin,
            self.vehicle_half_width + self.collision_margin,
        )

        obstacle_yaw = math.radians(float(getattr(obstacle, "heading_deg", 0.0)))
        obstacle_forward = (math.cos(obstacle_yaw), math.sin(obstacle_yaw))
        obstacle_left = (-obstacle_forward[1], obstacle_forward[0])
        size_x = max(1.0, float(getattr(obstacle, "size_x", 0.0)))
        size_y = max(1.0, float(getattr(obstacle, "size_y", 0.0)))
        obstacle_extents = (0.5 * size_x, 0.5 * size_y)
        obstacle_center = (obstacle.position.x, obstacle.position.y)

        center_delta = (
            obstacle_center[0] - vehicle_center[0],
            obstacle_center[1] - vehicle_center[1],
        )
        vehicle_axes = (vehicle_forward, vehicle_left)
        obstacle_axes = (obstacle_forward, obstacle_left)
        for axis in vehicle_axes + obstacle_axes:
            center_projection = abs(self._dot(center_delta, axis))
            vehicle_projection = sum(
                vehicle_extents[i] * abs(self._dot(vehicle_axes[i], axis))
                for i in range(2)
            )
            obstacle_projection = sum(
                obstacle_extents[i] * abs(self._dot(obstacle_axes[i], axis))
                for i in range(2)
            )
            if center_projection > vehicle_projection + obstacle_projection:
                return False
        return True

    @staticmethod
    def _dot(first, second):
        return first[0] * second[0] + first[1] * second[1]

    @staticmethod
    def _distance_to_path(point, path):
        min_distance = float("inf")
        for start, end in zip(path[:-1], path[1:]):
            segment_x = end.x - start.x
            segment_y = end.y - start.y
            segment_length_sq = segment_x ** 2 + segment_y ** 2
            if segment_length_sq < 1e-9:
                distance = point.distance(start)
            else:
                ratio = (
                    (point.x - start.x) * segment_x
                    + (point.y - start.y) * segment_y
                ) / segment_length_sq
                ratio = max(0.0, min(1.0, ratio))
                projection = Point(
                    start.x + ratio * segment_x,
                    start.y + ratio * segment_y,
                )
                distance = point.distance(projection)
            min_distance = min(min_distance, distance)
        return min_distance

    @staticmethod
    def _obstacle_radius(obstacle):
        size_x = max(0.0, float(getattr(obstacle, "size_x", 0.0)))
        size_y = max(0.0, float(getattr(obstacle, "size_y", 0.0)))
        if size_x <= 0.0 and size_y <= 0.0:
            return 0.75
        return max(0.5, 0.5 * max(size_x, size_y))

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))
