"""Expert path-following control, shared by ROS expert and VLDrive UDP.

The control block and helper methods were moved verbatim from AutonomousDriving.
Planning remains upstream; this module consumes planning_path/target_velocity.
"""
import numpy as np
from .pure_pursuit import PurePursuit
from .pid import Pid
from .control_input import ControlInput
from ..planning.lattice_planning import parameters as lattice_parameters
from ..planning.highway_lane_planning import parameters as highway_parameters


class PathController:
    def __init__(self, config):
        self.lattice_planning_process = None
        self._init_control(config)

    def _init_control(self, config):
        common_config = config['common']
        self.realtime_sampling_rate = float(
            common_config.get('realtime_sampling_rate', common_config['sampling_rate'])
        )
        self.sync_fallback_sampling_rate = float(
            common_config.get('sync_fallback_sampling_rate', self.realtime_sampling_rate)
        )
        self.control_dt = 1.0 / self.realtime_sampling_rate
        self.pid = Pid(sampling_time=self.control_dt, **config['control']['pid'])
        self.sampling_rate = self.realtime_sampling_rate
        self.pure_pursuit = PurePursuit(
            wheelbase=config['common']['wheelbase'], **config['control']['pure_pursuit']
        )
        self.max_steering = config['common']['max_steering'] / 180.0 * np.pi
        legacy_steering_sign = float(common_config.get('steering_sign', -1.0))
        self.realtime_steering_sign = float(
            common_config.get('realtime_steering_sign', legacy_steering_sign)
        )
        self.sync_steering_sign = float(
            common_config.get('sync_steering_sign', self.realtime_steering_sign)
        )
        self.steering_sign = self.realtime_steering_sign
        self.sync_mode = False
        self.previous_front_steer_rad = 0.0
        self.previous_target_velocity = None
        self.previous_acc_cmd = 0.0

    def control_path(self, planning_path, vehicle_state, target_velocity,
                     lattice_result, *, sync_mode=False, control_dt=None):
        self._configure_runtime(sync_mode=sync_mode, control_dt=control_dt)
        acc_cmd, raw_steering_rad, front_steer_rad = self._calculate_control(
            vehicle_state, planning_path, target_velocity, lattice_result)
        self.last_control_debug = dict(raw_steering_rad=float(raw_steering_rad))
        return ControlInput(acc_cmd, front_steer_rad)

    def _calculate_control(self, vehicle_state, planning_path, target_velocity, lattice_result):
        if self.previous_target_velocity is not None:
            if (
                target_velocity
                < self.previous_target_velocity
                - highway_parameters.TARGET_SPEED_DROP_RESET_THRESHOLD
            ):
                self.pid.reset()
        fast_roundabout_release = self._roundabout_entry_fast_release(
            target_velocity, lattice_result
        )
        if fast_roundabout_release:
            self.pid.reset()
        # 속도 제어를 위한 PID control
        acc_cmd = self.pid.get_output(
            target_velocity, vehicle_state.velocity,
            derivative_on_measurement=lattice_result.mode == "roundabout_entry",
        )
        # 합류 양보/충돌 예측이 정지를 요구하면 현재 속도와 무관하게 즉시 제동합니다.
        if target_velocity <= 0.05:
            self.pid.reset()
            acc_cmd = -1.
        elif fast_roundabout_release:
            acc_cmd = max(0.0, acc_cmd)
        else:
            acc_cmd = self._smooth_longitudinal_command(acc_cmd)
        self.previous_target_velocity = float(target_velocity)
        self.previous_acc_cmd = float(acc_cmd)
        # 경로 추종을 위한 pure pursuit control
        self.pure_pursuit.path = planning_path
        self.pure_pursuit.vehicle_state = vehicle_state
        # CtrlCmd.front_steer in MORAI 26.R1 is a front-wheel angle in radians,
        # not a normalized [-1, 1] command. GT_BEV's pure-pursuit lateral sign
        # is selected through steering_sign and clamped to the physical limit.
        recorded_lookahead = self._recorded_lookahead_limits(vehicle_state, lattice_result)
        if recorded_lookahead is not None:
            raw_steering_rad = self.pure_pursuit.calculate_steering_angle(lookahead_limits=recorded_lookahead)
        elif lattice_result.mode == "static_obstacle" and lattice_result.active:
            raw_steering_rad = self.pure_pursuit.calculate_steering_angle(
                lookahead_limits=lattice_parameters.STATIC_TRACKING_LOOKAHEAD
            )
        else:
            raw_steering_rad = self.pure_pursuit.calculate_steering_angle()
        requested_steer_rad = self.steering_sign * raw_steering_rad
        highway_lane_change = getattr(lattice_result, "highway_state", "") in (
            "CHANGE_LEFT",
            "CHANGE_RIGHT",
        )
        if highway_lane_change:
            lane_change_limit = np.deg2rad(
                highway_parameters.LANE_CHANGE_STEERING_LIMIT_DEG
            )
            requested_steer_rad = np.clip(
                requested_steer_rad,
                -lane_change_limit,
                lane_change_limit,
            )

            max_steer_step = np.deg2rad(
                highway_parameters.STEERING_RATE_LIMIT_DEG_PER_SEC
            ) * self.control_dt
            requested_steer_rad = np.clip(
                requested_steer_rad,
                self.previous_front_steer_rad - max_steer_step,
                self.previous_front_steer_rad + max_steer_step,
            )
        front_steer_rad = np.clip(
            requested_steer_rad,
            -self.max_steering,
            self.max_steering,
        )
        self.previous_front_steer_rad = float(front_steer_rad)

        return acc_cmd, raw_steering_rad, front_steer_rad

    def _configure_runtime(self, sync_mode=False, control_dt=None):
        """Apply transport sign and simulation-time dt for the current frame."""
        sync_mode = bool(sync_mode)
        fallback_rate = (
            self.sync_fallback_sampling_rate
            if sync_mode
            else self.realtime_sampling_rate
        )
        fallback_dt = 1.0 / fallback_rate
        try:
            selected_dt = float(control_dt) if control_dt is not None else fallback_dt
        except (TypeError, ValueError):
            selected_dt = fallback_dt
        if not np.isfinite(selected_dt) or selected_dt <= 0.0:
            selected_dt = fallback_dt
        # Reject discontinuities caused by a reset/invalid ROS header. The next
        # valid frame will immediately restore the measured simulation dt.
        selected_dt = float(np.clip(selected_dt, 0.001, 0.5))

        mode_changed = sync_mode != self.sync_mode
        self.sync_mode = sync_mode
        self.control_dt = selected_dt
        self.sampling_rate = 1.0 / selected_dt
        self.steering_sign = (
            self.sync_steering_sign
            if sync_mode
            else self.realtime_steering_sign
        )
        self.pid.set_sampling_time(selected_dt)

        highway_planner = getattr(
            self.lattice_planning_process, 'highway_lane_planner', None
        )
        if highway_planner is not None:
            highway_planner.set_control_dt(selected_dt)

        if mode_changed:
            self.pid.reset()
            self.previous_acc_cmd = 0.0
            self.previous_front_steer_rad = 0.0
            print(
                "[ExpertMode] mode={} steering_sign={:+.1f} dt={:.3f}s".format(
                    "sync" if sync_mode else "realtime",
                    self.steering_sign,
                    self.control_dt,
                )
            )


    def _recorded_lookahead_limits(self, vehicle_state, lattice_result):
        details = getattr(lattice_result, 'debug_details', {}).get('static_avoidance', {})
        if getattr(lattice_result, 'active', False) and details.get('planner') == 'recorded_route_graph':
            value = float(np.clip(self.pure_pursuit.lfd_gain*vehicle_state.velocity,
                                  *lattice_parameters.STATIC_TRACKING_LOOKAHEAD))
            self._recorded_last_lookahead = value
            return (value, value)
        previous = getattr(self, '_recorded_last_lookahead', None)
        if previous is None:
            return None
        if getattr(lattice_result, 'reason', '') != 'static_recorded_rejoined':
            self._recorded_last_lookahead = None
            return None
        nominal = float(np.clip(self.pure_pursuit.lfd_gain*vehicle_state.velocity,
                                self.pure_pursuit.min_lfd, self.pure_pursuit.max_lfd))
        dt = min(.2, max(.001, getattr(self, 'control_dt', 1./30.)))
        value = min(nominal, previous+dt)
        self._recorded_last_lookahead = value if value < nominal else None
        return (value, value)


    def _smooth_longitudinal_command(self, requested_command):
        """Release braking promptly, then build positive throttle smoothly."""
        requested_command = float(np.clip(requested_command, -1.0, 1.0))
        previous = float(self.previous_acc_cmd)
        if requested_command <= previous:
            # Safety-related braking must never be delayed.
            return requested_command
        rate = (
            highway_parameters.BRAKE_RELEASE_RATE
            if previous < 0.0
            else highway_parameters.ACCEL_COMMAND_RISE_RATE
        )
        maximum_step = rate * self.control_dt
        return min(requested_command, previous + maximum_step)


    def _roundabout_entry_fast_release(self, target_velocity, lattice_result):
        if self.previous_target_velocity is None:
            return False
        if self.previous_target_velocity > 0.05 or target_velocity <= 0.05:
            return False
        return getattr(lattice_result, "reason", "") in (
            "roundabout_entry_cruise",
            "roundabout_entry_follow_lead",
        )


