"""Existing PID/Pure Pursuit execution; transport encoding stays in network."""
from ..interfaces.control_command import ControlInput
from ..utils.driving_log import log_control


class ControlProcess:
    def __init__(self, driver):
        self.driver = driver

    def run(self, vehicle_state, plan):
        driver = self.driver
        if plan.force_brake:
            return driver.force_ve_brake(), []
        # The VE-only branch originally omitted the optional pedestrian flag.
        kwargs = ({'pedestrian_released': plan.pedestrian_released}
                  if plan.log_context else {})
        acc_cmd, raw_steering_rad, front_steer_rad = driver._calculate_control(
            vehicle_state, plan.path, plan.target_velocity, plan.section_result,
            **kwargs)
        if plan.log_context:
            log_control(driver, vehicle_state, plan.path, plan.section_result,
                        plan.target_velocity, acc_cmd=acc_cmd,
                        raw_steering_rad=raw_steering_rad,
                        front_steer_rad=front_steer_rad, **plan.log_context)
        return ControlInput(acc_cmd, front_steer_rad), plan.path
