"""차선 후보 경로 생성·다항식·경로 기하 계산.

HighwayLanePlanner의 동일 인스턴스 상태를 사용하는 메서드 묶음이다.
생성자나 별도 상태를 추가하지 않으며, 기존 메서드 본문과 호출 순서를 유지한다.
"""
import math
import numpy as np
from ...interfaces.geometry import Point
from . import parameters
from .structures import HighwayCandidatePath


class TrajectoryMixin:
    def _build_candidate(
        self,
        vehicle_state,
        target_link,
        target_projection,
        direction,
        source_link=None,
        transition_distance=None,
        prediction_horizon=0.0,
    ):
        candidate = HighwayCandidatePath(layer="lane_{}".format(direction))
        candidate.direction = direction
        candidate.target_link_id = target_link.id
        candidate.source_link_id = source_link.id if source_link is not None else target_link.id
        candidate.target_lane = target_link.lane_number
        candidate.minimum_prediction_time = prediction_horizon
        source_lane = source_link.lane_number if source_link else target_link.lane_number
        candidate.offset = (
            source_lane - target_link.lane_number
        ) * parameters.LANE_WIDTH

        path_distance = max(
            parameters.PATH_LOOKAHEAD_MIN,
            parameters.LANE_CHANGE_DISTANCE
            + max(vehicle_state.velocity, self._requested_speed) * prediction_horizon + 20.0,
            vehicle_state.velocity * parameters.PREDICTION_HORIZON + 12.0,
            min(parameters.STATIC_LOOKAHEAD_MAX,
                max(parameters.STATIC_LOOKAHEAD_MIN,
                    vehicle_state.velocity * parameters.STATIC_LOOKAHEAD_TIME)),
            parameters.LANE_CHANGE_DISTANCE
            + max(vehicle_state.velocity, self._requested_speed) * parameters.LANE_CHANGE_SETTLE_TIME + 15.0,
        )
        if direction == "keep":
            # A short polynomial recentering segment can produce a steering spike
            # at highway speed. Follow the measured link centerline directly.
            candidate.points = [
                item[0]
                for item in self.lane_map.route_points(
                    target_projection,
                    path_distance,
                    parameters.PATH_SAMPLE_SPACING,
                )
            ]
            candidate.curvature_cost = self._max_curvature(candidate.points)
            candidate.cost = 4.0 * candidate.curvature_cost
            return candidate

        transition_distance = (parameters.LANE_CHANGE_DISTANCE
                               if transition_distance is None else transition_distance)
        candidate.transition_distance = transition_distance
        goal, goal_yaw, _ = self.lane_map.advance_default(
            target_projection, transition_distance
        )
        coefficients, goal_x = self._solve_goal(vehicle_state, goal, goal_yaw)
        if coefficients is None:
            candidate.valid = False
            candidate.cost = float("inf")
            return candidate

        candidate.points = self._sample_polynomial(
            vehicle_state, goal_x, coefficients
        )
        candidate.transition_end = candidate.points[-1]
        travelled = transition_distance + parameters.PATH_SAMPLE_SPACING
        while travelled <= path_distance:
            point, _, _ = self.lane_map.advance_default(target_projection, travelled)
            if not candidate.points or point.distance(candidate.points[-1]) > 0.1:
                candidate.points.append(point)
            travelled += parameters.PATH_SAMPLE_SPACING

        candidate.curvature_cost = self._max_curvature(candidate.points)
        lane_change_penalty = 0.0 if direction == "keep" else 0.25
        no_successor_penalty = (
            2.0
            if direction != "keep"
            and not self.lane_map.successors.get(target_link.id, ())
            else 0.0
        )
        candidate.cost = (
            4.0 * candidate.curvature_cost
            + lane_change_penalty
            + no_successor_penalty
        )
        return candidate

    @staticmethod
    def _solve_goal(vehicle_state, goal, goal_yaw):
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        dx = goal.x - vehicle_state.position.x
        dy = goal.y - vehicle_state.position.y
        goal_x = cos_yaw * dx + sin_yaw * dy
        goal_y = -sin_yaw * dx + cos_yaw * dy
        relative_yaw = math.atan2(
            math.sin(goal_yaw - vehicle_state.yaw),
            math.cos(goal_yaw - vehicle_state.yaw),
        )
        if goal_x < 2.0:
            return None, goal_x
        relative_yaw = max(math.radians(-70.0), min(math.radians(70.0), relative_yaw))
        slope = math.tan(relative_yaw)
        a3 = (10.0 * goal_y - 4.0 * slope * goal_x) / goal_x ** 3
        a4 = (-15.0 * goal_y + 7.0 * slope * goal_x) / goal_x ** 4
        a5 = (6.0 * goal_y - 3.0 * slope * goal_x) / goal_x ** 5
        return (a3, a4, a5), goal_x

    @staticmethod
    def _sample_polynomial(vehicle_state, goal_x, coefficients):
        a3, a4, a5 = coefficients
        count = max(2, int(math.ceil(goal_x / parameters.PATH_SAMPLE_SPACING)))
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        points = []
        for local_x in np.linspace(0.0, goal_x, count + 1):
            local_y = a3 * local_x ** 3 + a4 * local_x ** 4 + a5 * local_x ** 5
            points.append(
                Point(
                    vehicle_state.position.x + cos_yaw * local_x - sin_yaw * local_y,
                    vehicle_state.position.y + sin_yaw * local_x + cos_yaw * local_y,
                )
            )
        return points

    @staticmethod
    def _cumulative_distance(points):
        values = [0.0]
        for start, end in zip(points[:-1], points[1:]):
            values.append(values[-1] + math.hypot(end.x-start.x, end.y-start.y))
        return values

    @staticmethod
    def _pose_at_distance(points, cumulative, target):
        target = max(0.0, min(target, cumulative[-1]))
        for index in range(1, len(cumulative)):
            if cumulative[index] < target:
                continue
            start = points[index - 1]
            end = points[index]
            segment = cumulative[index] - cumulative[index - 1]
            ratio = 0.0 if segment < 1e-9 else (target - cumulative[index - 1]) / segment
            return (
                Point(
                    start.x + ratio * (end.x - start.x),
                    start.y + ratio * (end.y - start.y),
                ),
                math.atan2(end.y - start.y, end.x - start.x),
            )
        start, end = points[-2], points[-1]
        return end, math.atan2(end.y - start.y, end.x - start.x)

    @staticmethod
    def _max_curvature(points):
        maximum = 0.0
        for index in range(1, len(points) - 1):
            first, middle, last = points[index - 1], points[index], points[index + 1]
            a = math.hypot(middle.x-first.x, middle.y-first.y)
            b = math.hypot(last.x-middle.x, last.y-middle.y)
            c = math.hypot(last.x-first.x, last.y-first.y)
            denominator = a * b * c
            if denominator < 1e-9:
                continue
            twice_area = abs(
                (middle.x - first.x) * (last.y - first.y)
                - (middle.y - first.y) * (last.x - first.x)
            )
            maximum = max(maximum, 2.0 * twice_area / denominator)
        return maximum
