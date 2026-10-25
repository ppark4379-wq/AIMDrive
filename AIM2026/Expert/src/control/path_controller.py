"""Expert path-following control, shared by ROS expert and VLDrive UDP.

Shared control retains upstream expert improvements; planner events are explicit inputs.
Planning remains upstream; this module consumes planning_path/target_velocity.
"""
import numpy as np
from .pure_pursuit import PurePursuit
from .pid import Pid
from .control_input import ControlInput
from ..planning.lattice import parameters as lattice_parameters
from ..planning.roundabout import diagnostics as roundabout_debug
from ..planning.highway import parameters as highway_parameters


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

    def _calculate_control(self, vehicle_state, planning_path, target_velocity, lattice_result,
                           *, pedestrian_released=False):
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
        fast_pedestrian_release = (
            pedestrian_released
            and self.previous_target_velocity is not None
            and self.previous_target_velocity <= .05 and target_velocity > .05
        )
        second_merge_launch_gain = self._second_merge_launch_acceleration_gain(
            target_velocity, vehicle_state, lattice_result)
        entry_merge_launch_gain = self._entry_merge_launch_acceleration_gain(
            target_velocity, vehicle_state, lattice_result)
        second_merge_release = (second_merge_launch_gain is not None
                                and self.previous_target_velocity is not None
                                and self.previous_target_velocity <= .05)
        entry_merge_release = (entry_merge_launch_gain is not None
                               and self.previous_target_velocity is not None
                               and self.previous_target_velocity <= .05)
        fast_stop_release = fast_roundabout_release or fast_pedestrian_release or second_merge_release or entry_merge_release
        if fast_stop_release:
            self.pid.reset()
        # 속도 제어를 위한 PID control
        acc_cmd = self.pid.get_output(
            target_velocity, vehicle_state.velocity,
            derivative_on_measurement=(lattice_result.mode in ("roundabout_entry", "highway_lane")
                                       or fast_pedestrian_release),
            acceleration_p_gain=(entry_merge_launch_gain or second_merge_launch_gain or self._highway_escape_acceleration_gain(
                target_velocity, vehicle_state, lattice_result
            ) or self._roundabout_entry_acceleration_gain(
                target_velocity, vehicle_state, lattice_result
            )),
        )
        # 합류 양보/충돌 예측이 정지를 요구하면 현재 속도와 무관하게 즉시 제동합니다.
        if target_velocity <= 0.05:
            self.pid.reset()
            acc_cmd = -1.
        elif (lattice_result.debug_details.get('lane_change_cruise', {}).get('maintain_speed')
              and target_velocity >= lattice_result.debug_details['lane_change_cruise']['speed'] - 1e-6
              and vehicle_state.velocity <= target_velocity):
            # A derivative kick or a previous ACC brake must not slow a verified
            # constant-speed lane change while we are still below its setpoint.
            # Overspeed, lowered legal/signal limits and new hazards retain PID braking.
            acc_cmd = max(0., self._smooth_longitudinal_command(acc_cmd))
        elif fast_stop_release:
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
            compact_escape = ((lattice_result.debug_details.get("highway_static_escape", {}).get("compact", False)
                               or lattice_result.debug_details.get("entry_merge", {}).get("compact", False))
                              and vehicle_state.velocity <= highway_parameters.STATIC_CHANGE_MAX_SPEED)
            lane_change_limit = np.deg2rad(
                highway_parameters.STATIC_CHANGE_STEERING_DEG if compact_escape
                else highway_parameters.LANE_CHANGE_STEERING_LIMIT_DEG
            )
            requested_steer_rad = np.clip(
                requested_steer_rad,
                -lane_change_limit,
                lane_change_limit,
            )

            max_steer_step = np.deg2rad(
                highway_parameters.STATIC_CHANGE_STEERING_RATE if compact_escape
                else highway_parameters.STEERING_RATE_LIMIT_DEG_PER_SEC
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
        roundabout_debug.record_control(
            lattice_result, target_velocity, acc_cmd, self.control_dt, front_steer_rad)

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
            highway_planner.search_time_budget = (None if sync_mode else
                highway_parameters.LANE_CHANGE_SEARCH_TIME_BUDGET)

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
            # Longer preview stabilizes fast gentle avoidance; retain the short
            # preview through tight bends so the body does not cut a corner.
            nominal = max(lattice_parameters.STATIC_TRACKING_LOOKAHEAD[1],
                          self.pure_pursuit.lfd_gain*vehicle_state.velocity)
            upper = float(np.interp(details.get('preview_curvature', .08),
                                    [.02, .08], [nominal, 1.5]))
            value = float(np.clip(self.pure_pursuit.lfd_gain*vehicle_state.velocity,
                                  lattice_parameters.STATIC_TRACKING_LOOKAHEAD[0], upper))
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

    @staticmethod
    def _entry_merge_launch_acceleration_gain(target_velocity, vehicle_state, lattice_result):
        details = getattr(lattice_result, 'debug_details', {}).get('entry_merge', {})
        if (details.get('phase') != 'MERGING' or not details.get('verified')
                or vehicle_state.velocity >= 3.0
                or target_velocity <= max(.05, vehicle_state.velocity)
                or target_velocity < details.get('target_speed', float('inf'))-1e-6):
            return None
        return highway_parameters.ENTRY_MERGE_LAUNCH_P_GAIN

    @staticmethod
    def _second_merge_launch_acceleration_gain(target_velocity, vehicle_state, lattice_result):
        safety = getattr(lattice_result, "debug_details", {}).get("merge_safety", {})
        if (getattr(lattice_result, "mode", "") != "highway_lane"
                or safety.get("zone") != "SECOND_ENTRY"
                or safety.get("decision") != "GO" or not safety.get("launch_verified")
                or target_velocity <= max(.05, vehicle_state.velocity)
                or target_velocity < safety.get("launch_target", float("inf")) - 1e-6):
            return None
        # Lower ACC, pedestrian/signal limits and every STOP retain normal braking.
        return highway_parameters.SECOND_MERGE_LAUNCH_P_GAIN

    @staticmethod
    def _highway_escape_acceleration_gain(target_velocity, vehicle_state, lattice_result):
        details = getattr(lattice_result, "debug_details", {}).get("highway_static_escape", {})
        if (getattr(lattice_result, "highway_state", "") not in ("CHANGE_LEFT", "CHANGE_RIGHT")
                or not details.get("compact") or not details.get("verified")
                or target_velocity <= max(0.05, vehicle_state.velocity)
                or target_velocity < details.get("speed_limit", float("inf")) - 1e-6):
            return None
        # Only boost an approved escape target. A lower ACC/signal limit and
        # every braking command retain the ordinary controller response.
        return highway_parameters.STATIC_CHANGE_ACCEL_P_GAIN

    @staticmethod
    def _roundabout_entry_acceleration_gain(target_velocity, vehicle_state, lattice_result):
        """Raise P only for permitted entry acceleration, keeping other limits."""
        if (getattr(lattice_result, "mode", "") != "roundabout_entry"
                or target_velocity <= 0.05 or target_velocity <= vehicle_state.velocity):
            return None
        details = lattice_result.debug_details.get("roundabout_entry", {})
        if details.get("phase") not in ("cruise", "rear_pressure") or details.get("front_gap") is not None:
            return None
        entry_started = (details.get("entering", False) or details.get("committed", False)
                         or details.get("hold_gap", float("inf")) <= lattice_parameters.ROUNDABOUT_HOLD_TOLERANCE)
        if not entry_started:
            return None
        limits = details.get("speed_limits", {})
        entry_target = min(limits.get("path_speed", float("inf")), lattice_result.acc_target_velocity)
        # A pedestrian/signal or another lower speed limit keeps the base gain.
        if target_velocity < entry_target - 1e-6:
            return None
        return lattice_parameters.ROUNDABOUT_ENTRY_ACCEL_P_GAIN
