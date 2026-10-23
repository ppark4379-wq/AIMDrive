"""Apply the original section, ACC and signal ceilings in their original order."""
import numpy as np
from ...config.parameters import lattice as lattice_parameters
from ..jamming.pedestrian_speed import pedestrian_speed


def target_speed(self, vehicle_state, local_path, planning_path, lattice_result,
                 planned_velocity, curvature_bypass, curvature_free_limit,
                 forward_object_info_dic_list, planning_object_info_dic_list,
                 current_traffic_light):
    static_valid = (lattice_result.mode == 'static_obstacle' and lattice_result.active
                    and lattice_result.debug_details.get('static_avoidance', {}).get('path_validated', False))
    if len(local_path) >= 2:
        self._static_reference_speed = planned_velocity if static_valid else None
    elif static_valid and len(planning_path) >= 2:
        cached_speed = getattr(self, '_static_reference_speed', None)
        if cached_speed is None and lattice_result.debug_details.get('static_avoidance', {}).get('preplanned_complete'):
            cached_speed = lattice_result.acc_target_velocity
        if cached_speed is not None:
            # Only a missing reference may use the saved profile. Signals,
            # traffic and the validated held-path speed cap still apply below.
            planned_velocity = min(cached_speed, lattice_parameters.STATIC_MAX_SPEED_KPH / 3.6)
    else:
        self._static_reference_speed = None
    planned_velocity = self._merge_ego_curvature_speed_floor(
        planned_velocity,
        vehicle_state.velocity,
        lattice_result,
    )
    planned_velocity = self._hold_merge_ego_curvature_speed(
        planned_velocity, lattice_result
    )
    planned_velocity = self._second_merge_slowdown_band(
        planned_velocity, vehicle_state
    )

    # adaptive cruise control를 활용한 속도 계획
    if curvature_bypass:
        planned_velocity = min(planned_velocity, curvature_free_limit)

    legacy_acc_entries = self._legacy_acc_entries(
        vehicle_state, lattice_result, forward_object_info_dic_list
    )
    person_check, pedestrian_target_velocity = pedestrian_speed(
        self, static_valid, lattice_result, planning_path, vehicle_state,
        planned_velocity, planning_object_info_dic_list)
    if person_check is None:
        self.adaptive_cruise_control.check_object(
            planning_path, legacy_acc_entries, current_traffic_light)
    else:
        self.adaptive_cruise_control.check_object(
            planning_path, legacy_acc_entries, current_traffic_light,
            pedestrian_conflict_check=person_check)
    target_velocity = self.adaptive_cruise_control.get_target_velocity(vehicle_state.velocity, planned_velocity)
    signal_target_velocity = self.traffic_light_stop_controller.get_target_velocity(
        current_traffic_light=current_traffic_light,
        current_waypoint=self.path_manager.current_waypoint,
        planned_velocity=planned_velocity,
    )
    # Long-layer center-path OBB detection applies the mode-specific speed scale.
    lattice_target_velocity = (
        planned_velocity * lattice_result.longitudinal_speed_scale
        if lattice_result.long_obstacle_detected
        else planned_velocity
    )
    legacy_target_velocity = target_velocity
    target_velocity = max(0., min(
        target_velocity,
        signal_target_velocity,
        lattice_target_velocity,
        getattr(lattice_result, "acc_target_velocity", float("inf")),
        pedestrian_target_velocity,
    ))
    unconstrained_static_acceleration = target_velocity
    target_velocity = self._smooth_static_target_speed(target_velocity, vehicle_state, lattice_result)
    if lattice_result.mode in ("roundabout_entry", "static_obstacle"):
        limits = {
            "legacy_acc": legacy_target_velocity,
            "signal": signal_target_velocity,
            "path_speed": lattice_target_velocity,
            ("roundabout" if lattice_result.mode == "roundabout_entry" else "static"): lattice_result.acc_target_velocity,
        }
        if target_velocity < unconstrained_static_acceleration:
            limits['static_acceleration'] = target_velocity
        if pedestrian_target_velocity != float('inf'):
            limits['pedestrian_obb'] = pedestrian_target_velocity
        key = "roundabout_entry" if lattice_result.mode == "roundabout_entry" else "static_avoidance"
        details = lattice_result.debug_details.setdefault(key, {})
        details.setdefault("phase", lattice_result.reason)
        details["speed_limits"] = limits
        details["speed_source"] = min(limits, key=limits.get)
        if (lattice_result.mode == "static_obstacle" and details.get("error")
                and limits["static"] <= limits[details["speed_source"]]):
            details["speed_source"] = "static"
        details["final_target_velocity"] = max(0.0, target_velocity)
    if lattice_result.mode == "highway_lane":
        state = (lattice_result.reason, target_velocity <= 0.05,
                 getattr(lattice_result, "highway_state", ""))
        previous = getattr(self, "_last_highway_control_state", None)
        changed_from_lane_change = (previous is not None
            and previous[2] in ("CHANGE_LEFT", "CHANGE_RIGHT") and previous[2] != state[2])
        if state != previous and (state[1] or (previous is not None and previous[1])
                                   or changed_from_lane_change):
            details = lattice_result.debug_details
            print("[HighwayDecision] pos=({:.2f},{:.2f}) yaw={:.2f} speed={:.2f} target={:.2f} "
                  "reason={} lane_conflict={} merge={}".format(
                      vehicle_state.position.x, vehicle_state.position.y,
                      np.degrees(vehicle_state.yaw), vehicle_state.velocity, target_velocity,
                      lattice_result.reason, details.get("lane_change_conflict"),
                      details.get("merge_safety")))
        self._last_highway_control_state = state
    else:
        self._last_highway_control_state = None
    return target_velocity, planned_velocity
