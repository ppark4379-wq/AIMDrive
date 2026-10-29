"""기존 경로 속도 보정식. 위험에 따른 최종 속도 제한은 별도로 적용된다."""
import numpy as np
from ....config.parameters import highway as highway_parameters


def _merge_ego_curvature_speed_floor(
    planned_velocity, current_velocity, lattice_result
):
    """Do not let route curvature brake either Ego-first merge."""
    details = getattr(lattice_result, "debug_details", {}) or {}
    safety = details.get("merge_safety", {}) or {}
    if safety.get("zone") in ("FIRST_PREDICTIVE", "SECOND_ENTRY") and "speed_floor" in safety:
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
