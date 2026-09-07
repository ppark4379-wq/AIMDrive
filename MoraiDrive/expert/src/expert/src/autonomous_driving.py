#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .obstacle.forward_object_detector import ForwardObjectDetector
from .localization.path_manager import PathManager
from .planning.adaptive_cruise_control import AdaptiveCruiseControl
from .planning.traffic_light_stop import TrafficLightStopController
from .planning.lattice_planning import LatticePlanningProcess
from .planning.highway_lane_planning import parameters as highway_parameters
from .control.pure_pursuit import PurePursuit
from .control.pid import Pid
from .control.control_input import ControlInput
from .config.config import Config

import numpy as np

class AutonomousDriving:
    def __init__(self):
        config = Config()

        if config["map"]["use_mgeo_path"]:
            # pyproj를 포함한 MGeo loader는 내부 경로 생성 모드에서만 필요하다.
            from .path.calc_path import mgeo_dijkstra_path

            mgeo_path = mgeo_dijkstra_path(config["map"]["name"])
            self.path = mgeo_path.calc_dijkstra_path(config["map"]["mgeo"]["start_node"], config["map"]["mgeo"]["end_node"])
            self.path_manager = PathManager(
                self.path, config["map"]["is_closed_path"], config["map"]["local_path_size"]
            )
        else:
            self.path = config["map"]["path"]
            self.path_manager = PathManager(
                self.path, config["map"]["is_closed_path"], config["map"]["local_path_size"]
            )
        self.path_manager.set_velocity_profile(**config['planning']['velocity_profile'])

        self.traffic_light_stop_controller = TrafficLightStopController(
            path=self.path,
            stopline_mapper=config["map"]["stopline_mapper"],
            is_closed_path=config["map"]["is_closed_path"],
            **config['planning'].get('traffic_light_stop', {})
        )

        self.forward_object_detector = ForwardObjectDetector(config["map"]["traffic_light_list"])

        self.adaptive_cruise_control = AdaptiveCruiseControl(
            vehicle_length=config['common']['vehicle_length'], **config['planning']['adaptive_cruise_control']
        )
        # Uses the measured static-obstacle and highway section centerlines.
        self.lattice_planning_process = LatticePlanningProcess(
            map_data_dir=config["map"]["data_dir"]
        )
        self.pid = Pid(sampling_time=1/float(config['common']['sampling_rate']), **config['control']['pid'])
        self.sampling_rate = float(config['common']['sampling_rate'])
        self.pure_pursuit = PurePursuit(
            wheelbase=config['common']['wheelbase'], **config['control']['pure_pursuit']
        )
        self.max_steering = config['common']['max_steering'] / 180.0 * np.pi
        self.steering_sign = float(config['common'].get('steering_sign', 1.0))
        self.previous_front_steer_rad = 0.0
        self.previous_target_velocity = None
        self.previous_acc_cmd = 0.0
        self._merge_curvature_floor_zone = None
        self._merge_curvature_floor_speed = None

    def execute(self, vehicle_state, dynamic_object_list, current_traffic_light):
        # 현재 위치 기반으로 local path과 planned velocity 추출
        local_path, planned_velocity = self.path_manager.get_local_path(vehicle_state)
        profile_planned_velocity = planned_velocity

        # 전방 장애물 인지
        self.forward_object_detector._dynamic_object_list = dynamic_object_list
        forward_object_info_dic_list = self.forward_object_detector.detect_object(
            vehicle_state, forward_only=True
        )
        # 목표 차선 뒤에서 접근하는 차량도 미래 OBB 검사에 포함합니다.
        planning_object_info_dic_list = self.forward_object_detector.detect_object(
            vehicle_state, forward_only=False
        )

        # Static zone: right-biased obstacle avoidance.
        # Highway zone: cost-based vehicle avoidance, or base-path ACC priority
        # when the distance/TTC risk threshold is crossed.
        lattice_result = self.lattice_planning_process.run(
            vehicle_state, local_path, planning_object_info_dic_list
        )
        planning_path = lattice_result.path
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
        legacy_acc_entries = self._legacy_acc_entries(
            lattice_result, forward_object_info_dic_list
        )
        self.adaptive_cruise_control.check_object(
            planning_path, legacy_acc_entries, current_traffic_light
        )
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
        target_velocity = min(
            target_velocity,
            signal_target_velocity,
            lattice_target_velocity,
            getattr(lattice_result, "acc_target_velocity", float("inf")),
        )
        if self.previous_target_velocity is not None:
            if (
                target_velocity
                < self.previous_target_velocity
                - highway_parameters.TARGET_SPEED_DROP_RESET_THRESHOLD
            ):
                self.pid.reset()
        self.previous_target_velocity = float(target_velocity)
        # 속도 제어를 위한 PID control
        acc_cmd = self.pid.get_output(target_velocity, vehicle_state.velocity)
        # 합류 양보/충돌 예측이 정지를 요구하면 현재 속도와 무관하게 즉시 제동합니다.
        if target_velocity <= 0.05:
            self.pid.reset()
            acc_cmd = -1.
        else:
            acc_cmd = self._smooth_longitudinal_command(acc_cmd)
        self.previous_acc_cmd = float(acc_cmd)
        # 경로 추종을 위한 pure pursuit control
        self.pure_pursuit.path = planning_path
        self.pure_pursuit.vehicle_state = vehicle_state
        # CtrlCmd.front_steer in MORAI 26.R1 is a front-wheel angle in radians,
        # not a normalized [-1, 1] command. GT_BEV's pure-pursuit lateral sign
        # is selected through steering_sign and clamped to the physical limit.
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
            ) / self.sampling_rate
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

        self._debug_counter = getattr(self, "_debug_counter", 0) + 1
        if (
            highway_parameters.HIGHWAY_DEBUG_ENABLED
            and self._debug_counter
            >= highway_parameters.HIGHWAY_DEBUG_INTERVAL_CYCLES
        ):
            self._debug_counter = 0
            print(
                f"[CTRL] raw_steering={np.degrees(raw_steering_rad):+.2f}deg "
                f"front_steer={np.degrees(front_steer_rad):+.2f}deg "
                f"({front_steer_rad:+.3f}rad) acc_cmd={acc_cmd:+.3f} "
                f"target_v={target_velocity:.2f} planned_v={planned_velocity:.2f} "
                f"profile_v={profile_planned_velocity:.2f} "
                f"cur_v={vehicle_state.velocity:.2f}"
            )
            details = getattr(lattice_result, "debug_details", {})
            if details:
                def debug_value(value, suffix=""):
                    return "inf" if np.isinf(value) else f"{value:.2f}{suffix}"

                print(
                    "[PLAN] reason={} state={} link={} lane={} s={:.1f} -> target_link={} "
                    "objects={}/{} path={} active={}".format(
                        lattice_result.reason,
                        getattr(lattice_result, "highway_state", "OFF"),
                        getattr(lattice_result, "current_link_id", ""),
                        details.get("ego_lane", "?"),
                        details.get("ego_s", 0.0),
                        getattr(lattice_result, "target_link_id", ""),
                        details.get("matched_vehicle_objects", 0),
                        details.get("total_vehicle_objects", 0),
                        len(planning_path),
                        lattice_result.active,
                    )
                )
                print(
                    "[PLAN] risk gap={} TTC={} front_acc={} merge_raw={} "
                    "merge_filtered={} final_acc={} keep_valid={} keep_collision={}".format(
                        debug_value(lattice_result.nearest_vehicle_distance, "m"),
                        debug_value(lattice_result.ttc, "s"),
                        debug_value(details.get("front_acc_limit", float("inf")), "m/s"),
                        debug_value(details.get("raw_merge_limit", float("inf")), "m/s"),
                        debug_value(details.get("filtered_merge_limit", float("inf")), "m/s"),
                        debug_value(getattr(lattice_result, "acc_target_velocity", float("inf")), "m/s"),
                        details.get("keep_valid", True),
                        debug_value(details.get("keep_collision_time", float("inf")), "s"),
                    )
                )
                for merge_check in details.get("merge_checks", ())[:6]:
                    print("[PLAN] merge: {}".format(merge_check))
                for lane_check in details.get("lane_checks", ())[:6]:
                    print("[PLAN] lane: {}".format(lane_check))
                for candidate in getattr(lattice_result, "candidates", ()):
                    print(
                        "[PLAN] candidate dir={} link={} valid={} cost={} "
                        "collision={} curvature={:.4f}".format(
                            getattr(candidate, "direction", "?"),
                            getattr(candidate, "target_link_id", ""),
                            candidate.valid,
                            debug_value(candidate.cost),
                            debug_value(getattr(candidate, "collision_time", float("inf")), "s"),
                            getattr(candidate, "curvature_cost", 0.0),
                        )
                    )

        return ControlInput(acc_cmd, front_steer_rad), planning_path

    @staticmethod
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

    @staticmethod
    def _merge_ego_curvature_speed_floor(
        planned_velocity, current_velocity, lattice_result
    ):
        """Do not let route curvature brake either Ego-first merge."""
        details = getattr(lattice_result, "debug_details", {}) or {}
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
        maximum_step = rate / self.sampling_rate
        return min(requested_command, previous + maximum_step)

    @staticmethod
    def _legacy_acc_entries(lattice_result, forward_entries):
        """Avoid applying path-distance vehicle ACC twice on mapped highways.

        The highway lane planner already associates vehicles with lane links,
        checks fallback collisions, and returns an ACC speed ceiling.  The
        legacy path-distance detector can mistake an adjacent/old-lane vehicle
        for a lead vehicle immediately after a lane change because the two
        paths remain close.  Keep pedestrians and traffic lights in the legacy
        controller, but let the lane planner be the sole vehicle authority.
        """
        if getattr(lattice_result, "mode", "") != "highway_lane":
            return forward_entries
        return [
            entry
            for entry in forward_entries
            if getattr(entry["object_info"], "type", -1) not in (1, 2)
        ]
