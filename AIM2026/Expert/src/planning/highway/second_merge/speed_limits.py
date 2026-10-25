"""기존 경로 속도 보정식. 위험에 따른 최종 속도 제한은 별도로 적용된다."""
import numpy as np
from ....config.parameters import highway as highway_parameters


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
