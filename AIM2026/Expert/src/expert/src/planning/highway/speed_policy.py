"""Existing highway speed adjustments, extracted without changing equations."""
import numpy as np
from ...config.parameters import highway as highway_parameters


def _second_merge_curvature_bypass(vehicle_state):
    start = np.asarray(highway_parameters.SECOND_MERGE_SLOWDOWN_POINT)
    end = np.asarray(highway_parameters.SECOND_MERGE_CURVATURE_BYPASS_END)
    direction = end - start
    length = np.linalg.norm(direction)
    unit = direction / length
    relative = np.asarray((vehicle_state.position.x, vehicle_state.position.y)) - start
    progress = np.dot(relative, unit)
    lateral = abs(relative[0] * unit[1] - relative[1] * unit[0])
    heading = np.asarray((np.cos(vehicle_state.yaw), np.sin(vehicle_state.yaw)))
    return bool(0.0 <= progress <= length
                and lateral <= highway_parameters.SECOND_MERGE_CURVATURE_BYPASS_HALF_WIDTH
                and np.dot(heading, unit) > 0.5)



def _second_merge_slowdown_band(planned_velocity, vehicle_state):
    """두 번째 합류를 지정 좌표 이후부터 35~45km/h 로 통과한다.

    원의 앞쪽 경계가 SECOND_MERGE_SLOWDOWN_POINT 이므로 그 좌표 이전에는
    감속하지 않는다. 하한은 경로 프로파일에만 적용되고, 이후
    target_velocity 의 min() 에서 ACC·신호·비상정지가 계속 우선하므로
    실제 장애물 앞에서 35km/h 로 버티는 일은 없다.
    """
    center_x, center_y = highway_parameters.SECOND_MERGE_SLOWDOWN_CENTER
    distance = np.hypot(
        vehicle_state.position.x - center_x,
        vehicle_state.position.y - center_y,
    )
    if distance > highway_parameters.SECOND_MERGE_SLOWDOWN_RADIUS:
        return float(planned_velocity)
    return float(
        np.clip(
            planned_velocity,
            highway_parameters.SECOND_MERGE_SLOWDOWN_MIN_SPEED,
            highway_parameters.SECOND_MERGE_SLOWDOWN_MAX_SPEED,
        )
    )



def _merge_ego_curvature_speed_floor(
    planned_velocity, current_velocity, lattice_result
):
    """Do not let route curvature brake either Ego-first merge."""
    details = getattr(lattice_result, "debug_details", {}) or {}
    safety = details.get("merge_safety", {}) or {}
    if safety.get("zone") == "SECOND_ENTRY" and "speed_floor" in safety:
        # This exact speed was checked on the current path. Signals, front
        # ACC and the final planner ceiling still take precedence below.
        planned_velocity = max(float(planned_velocity), float(safety["speed_floor"]))
    priorities = details.get("merge_zone_priorities", {}) or {}
    first_merge_ego = (
        highway_parameters.PREDICTIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR
        and priorities.get("FIRST_PREDICTIVE") == "EGO"
    )
    second_merge_ego = (
        highway_parameters.CONSERVATIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR
        and priorities.get("SECOND_ENTRY") == "EGO"
    )
    if first_merge_ego or second_merge_ego:
        return max(float(planned_velocity), float(current_velocity))
    return float(planned_velocity)



def _hold_merge_ego_curvature_speed(self, planned_velocity, lattice_result):
    """Latch speed while Ego owns either merge so curvature cannot bleed it down."""
    details = getattr(lattice_result, "debug_details", {}) or {}
    priorities = details.get("merge_zone_priorities", {}) or {}
    if priorities.get("FIRST_PREDICTIVE") == "EGO":
        zone = "FIRST_PREDICTIVE"
    elif priorities.get("SECOND_ENTRY") == "EGO":
        zone = "SECOND_ENTRY"
    else:
        self._merge_curvature_floor_zone = None
        self._merge_curvature_floor_speed = None
        return float(planned_velocity)

    planned_velocity = float(planned_velocity)
    if self._merge_curvature_floor_zone != zone:
        self._merge_curvature_floor_zone = zone
        self._merge_curvature_floor_speed = planned_velocity
    else:
        self._merge_curvature_floor_speed = max(
            float(self._merge_curvature_floor_speed), planned_velocity
        )
    return float(self._merge_curvature_floor_speed)

