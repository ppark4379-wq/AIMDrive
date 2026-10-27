"""Reference → objects → section planning → final target speed.

One persistent driver owns the original planner/controller state. Process
objects hold that same owner; no state is copied or reset between ticks.
"""
from ..interfaces.planning_result import DrivingPlan
from .common.reference_path import prepare_reference, prepare_prediction_path
from .common.object_selection import select_objects
from .common.target_speed import target_speed


class PlanningProcess:
    def __init__(self, driver):
        self.driver = driver

    def run(self, vehicle_state, dynamic_object_list, current_traffic_light,
            sync_mode=False, control_dt=None, ego_header_seq=None,
            ego_source_ns=None, perception_health=None):
        driver = self.driver
        driver._configure_runtime(sync_mode=sync_mode, control_dt=control_dt)
        if getattr(driver, 'perception_source', 'ros') == 've':
            return self._ve_plan(vehicle_state, dynamic_object_list,
                                 current_traffic_light, perception_health)

        local_path, planned_velocity, profile_speed, bypass, free_limit = prepare_reference(driver, vehicle_state)
        front_objects, planning_objects = select_objects(driver, vehicle_state, dynamic_object_list)
        prediction_path = prepare_prediction_path(driver, vehicle_state)
        section = driver.lattice_planning_process.run(
            vehicle_state, local_path, planning_objects,
            desired_velocity=planned_velocity, prediction_path=prediction_path,
            global_path=driver.path, ego_header_seq=ego_header_seq,
            ego_source_ns=ego_source_ns)
        path = section.path
        target, planned_velocity = target_speed(
            driver, vehicle_state, local_path, path, section,
            planned_velocity, bypass, free_limit, front_objects,
            planning_objects, current_traffic_light,
            crossing_path=prediction_path if path is local_path else path)
        return DrivingPlan(
            path, target, section,
            pedestrian_released=driver.adaptive_cruise_control.pedestrian_released,
            log_context=dict(planned_velocity=planned_velocity,
                             profile_planned_velocity=profile_speed,
                             curvature_bypass=bypass))

    def _ve_plan(self, vehicle_state, objects, traffic_light, health):
        return ve_plan(self.driver, vehicle_state, objects, traffic_light, health)


def ve_plan(self, vehicle_state, objects, traffic_light, health):
    local_path, planned_velocity = self.path_manager.get_local_path(vehicle_state)
    process = self.lattice_planning_process
    planned_velocity = min(max(0., planned_velocity), process.max_speed)
    result = process.run(vehicle_state, local_path, objects, planned_velocity, input_health=health)
    signal_velocity = self.traffic_light_stop_controller.get_target_velocity(
        current_traffic_light=traffic_light, current_waypoint=self.path_manager.current_waypoint,
        planned_velocity=planned_velocity,
    )
    target_velocity = max(0., min(planned_velocity, signal_velocity, result.acc_target_velocity))
    result.debug_details.setdefault("ve_perception", {}).update(
        final_target_velocity=target_velocity, signal_target_velocity=signal_velocity,
        signals_source="original_ros", objects_source="ve", ve_waypoints_used=False,
    )
    force_brake = len(result.path) < 2 or result.debug_details['ve_perception'].get('force_brake')
    return DrivingPlan(result.path, target_velocity, result, force_brake=force_brake)
