"""Apply the original section, ACC and signal ceilings in their original order."""
import numpy as np
from ...config.parameters import lattice as lattice_parameters
from ..jamming.pedestrian_speed import pedestrian_speed
from .dynamic_crossing import DynamicCrossingGuard


def target_speed(self, vehicle_state, local_path, planning_path, lattice_result,
                 planned_velocity, curvature_bypass, curvature_free_limit,
                 forward_object_info_dic_list, planning_object_info_dic_list,
                 current_traffic_light, crossing_path=None):
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
    entry_merge = lattice_result.debug_details.get('entry_merge', {})
    if entry_merge.get('phase') in ('MERGING', 'COMPLETE', 'FOLLOWING'):
        # Keep the verified first-merge speed instead of re-reading curvature
        # on the abandoned source path. Signals/pedestrians/ACC below still win.
        planned_velocity = max(planned_velocity, entry_merge['target_speed'])
    change_cruise = lattice_result.debug_details.get('lane_change_cruise')
    if change_cruise:
        # Ordinary lane changes keep their validated entry speed. The global
        # route still follows the abandoned lane; its curvature must not brake
        # this maneuver. Legal-zone/terminal limits and hazards below still win.
        planned_velocity = min(change_cruise['speed'], self.path_manager.get_velocity_without_curvature())

    # adaptive cruise control를 활용한 속도 계획
    if curvature_bypass:
        planned_velocity = min(planned_velocity, curvature_free_limit)

    legacy_acc_entries = self._legacy_acc_entries(
        vehicle_state, lattice_result, forward_object_info_dic_list
    )
    # Highway following/merging/lane changes retain their existing authorities,
    # including fallback and lane-match failure. Do not carry a previous road's
    # crossing hold into the highway, even if legacy ACC still has entries.
    if lattice_result.mode in ('highway', 'highway_lane'):
        if hasattr(self, 'dynamic_crossing_guard'):
            self.dynamic_crossing_guard.reset()
        crossing_velocity = float('inf')
        crossing_details = dict(state='disabled_highway')
    else:
        # The legacy entries also exclude objects owned by tunnel prediction.
        if not hasattr(self, 'dynamic_crossing_guard'):
            self.dynamic_crossing_guard = DynamicCrossingGuard()
        crossing_velocity, crossing_details = self.dynamic_crossing_guard.evaluate(
            planning_path if crossing_path is None else crossing_path,
            vehicle_state, legacy_acc_entries, planned_velocity)
        legacy_acc_entries = self.dynamic_crossing_guard.following_entries(legacy_acc_entries)
    lattice_result.debug_details['dynamic_crossing'] = crossing_details
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
    acc = self.adaptive_cruise_control
    lattice_result.debug_details['legacy_acc'] = dict(
        object_id=getattr(acc, 'object_id', None), object_type=getattr(acc, 'object_type', None),
        gap_m=getattr(acc, 'object_distance', None) if getattr(acc, 'object_type', None) is not None else None,
        time_gap_s=getattr(acc, 'object_time_gap', None),
        target_velocity=legacy_target_velocity)
    target_velocity = max(0., min(
        target_velocity,
        signal_target_velocity,
        lattice_target_velocity,
        getattr(lattice_result, "acc_target_velocity", float("inf")),
        pedestrian_target_velocity,
        crossing_velocity,
    ))
    # The explicitly enabled MORAI deadlock override is the final obstacle
    # authority, including duplicate legacy/person crossing ceilings. Keep
    # signals, road/path speed and the validated corridor's capped creep speed.
    # Freshness/manual termination still gate the command in runtime.py.
    resume = lattice_result.debug_details.get('jamming_traffic', {}).get('forced_resume', {})
    if (lattice_result.mode == 'static_obstacle' and lattice_result.active
            and lattice_result.debug_details.get('jamming_corridor', {}).get('standing_only')
            and resume.get('active') and resume.get('geometry_validated')):
        resume['before_final_override_mps'] = target_velocity
        protected = set(resume.get('protected_vehicle_ids', ()))
        target_velocity = max(0., min(
            signal_target_velocity, lattice_target_velocity,
            lattice_result.acc_target_velocity, resume['target_mps'],
            resume.get('vehicle_speed_cap_mps', float('inf')),
            legacy_target_velocity if (getattr(acc, 'object_type', None) == 3
                                       or getattr(acc, 'object_id', None) in protected) else float('inf'),
            crossing_velocity if crossing_details.get('object_id') in protected else float('inf')))
    unconstrained_static_acceleration = target_velocity
    target_velocity = self._smooth_static_target_speed(target_velocity, vehicle_state, lattice_result)
    lattice_result.debug_details['speed_limits'] = dict(
        path=planned_velocity, legacy_acc=legacy_target_velocity, signal=signal_target_velocity,
        section=getattr(lattice_result, 'acc_target_velocity', float('inf')),
        dynamic_crossing=crossing_velocity, final=target_velocity)
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
        if crossing_velocity != float('inf'):
            limits['dynamic_crossing'] = crossing_velocity
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
