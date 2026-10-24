"""미래 객체 경로·예측 시각·차체 충돌 계산.

HighwayLanePlanner의 동일 인스턴스 상태를 사용하는 메서드 묶음이다.
생성자나 별도 상태를 추가하지 않으며, 기존 메서드 본문과 호출 순서를 유지한다.
"""
import math
from . import parameters


class CollisionPredictionMixin:
    def _evaluate_dynamic_collision(self, candidate, vehicle_state, object_matches):
        if not candidate.valid or len(candidate.points) < 2:
            return
        cumulative = self._cumulative_distance(candidate.points)
        ego_speed = max(parameters.PREDICTION_MIN_SPEED, vehicle_state.velocity)
        minimum_clearance = float("inf")
        collision_time = float("inf")

        for time_value in self._prediction_times(parameters.PREDICTION_TIME_STEP):
            ego_point, ego_yaw = self._pose_at_distance(
                candidate.points, cumulative, ego_speed * time_value
            )
            for entry, object_match in object_matches:
                object_info = entry["object_info"]
                local_position = entry.get("local_position")
                if (
                    candidate.direction == "keep"
                    and local_position is not None
                    and local_position.x <= 0.0
                    and object_match.lane_link.id == getattr(candidate, "target_link_id", None)
                    and abs(local_position.y) < 1.0
                ):
                    # A rear vehicle must never make Ego brake on a keep-lane
                    # path; braking increases the rear-end collision risk.
                    continue
                predicted_options = self.merge_safety._pose(object_info, object_match, time_value)
                for object_point, object_yaw in predicted_options:
                    clearance = ego_point.distance(object_point)
                    minimum_clearance = min(minimum_clearance, clearance)
                    if self._obb_collision(
                        ego_point, ego_yaw, object_point, object_yaw, object_info
                    ):
                        candidate.valid = False
                        candidate.cost = float("inf")
                        collision_time = time_value
                        break
                if not candidate.valid:
                    break
            if not candidate.valid:
                break
        candidate.collision_time = collision_time
        candidate.obstacle_cost = (
            0.0 if math.isinf(minimum_clearance) else 1.0 / max(0.1, minimum_clearance)
        )
        if candidate.valid:
            candidate.cost += 0.4 * candidate.obstacle_cost

    def _prediction_paths(self, object_matches):
        paths = []
        for entry, lane_match in object_matches:
            points = []
            velocity = max(0.0, float(entry["object_info"].velocity))
            # RViz shows one future OBB per vehicle at the configured horizon.
            # SAT collision checking still uses the finer fixed time step above.
            for time_value in (0.0, parameters.PREDICTION_HORIZON):
                point, _, _ = self.lane_map.advance_default(
                    lane_match, velocity * time_value
                )
                points.append(point)
            paths.append(points)
        return paths

    @staticmethod
    def _prediction_times(step):
        horizon = max(0.0, float(parameters.PREDICTION_HORIZON))
        step = max(1e-3, float(step))
        values = [0.0]
        time_value = step
        while time_value < horizon - 1e-9:
            values.append(time_value)
            time_value += step
        if horizon > 1e-9:
            values.append(horizon)
        return values

    @staticmethod
    def _obb_collision(ego_rear, ego_yaw, object_center, object_yaw, object_info):
        ego_forward = (math.cos(ego_yaw), math.sin(ego_yaw))
        ego_left = (-ego_forward[1], ego_forward[0])
        ego_center = (
            ego_rear.x + 0.5 * parameters.VEHICLE_LENGTH * ego_forward[0],
            ego_rear.y + 0.5 * parameters.VEHICLE_LENGTH * ego_forward[1],
        )
        ego_extents = (
            0.5 * parameters.VEHICLE_LENGTH + parameters.COLLISION_MARGIN,
            0.5 * parameters.VEHICLE_WIDTH + parameters.COLLISION_MARGIN,
        )
        object_forward = (math.cos(object_yaw), math.sin(object_yaw))
        object_left = (-object_forward[1], object_forward[0])
        object_extents = (
            0.5 * max(1.0, float(getattr(object_info, "size_x", 0.0))),
            0.5 * max(1.0, float(getattr(object_info, "size_y", 0.0))),
        )
        delta = (object_center.x - ego_center[0], object_center.y - ego_center[1])
        ego_axes = (ego_forward, ego_left)
        object_axes = (object_forward, object_left)
        for axis in ego_axes + object_axes:
            center_projection = abs(delta[0] * axis[0] + delta[1] * axis[1])
            ego_projection = sum(
                ego_extents[i]
                * abs(ego_axes[i][0] * axis[0] + ego_axes[i][1] * axis[1])
                for i in range(2)
            )
            object_projection = sum(
                object_extents[i]
                * abs(object_axes[i][0] * axis[0] + object_axes[i][1] * axis[1])
                for i in range(2)
            )
            if center_projection > ego_projection + object_projection:
                return False
        return True
