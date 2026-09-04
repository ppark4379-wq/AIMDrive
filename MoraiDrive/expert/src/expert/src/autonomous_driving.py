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

    def execute(self, vehicle_state, dynamic_object_list, current_traffic_light):
        # 현재 위치 기반으로 local path과 planned velocity 추출
        local_path, planned_velocity = self.path_manager.get_local_path(vehicle_state)

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

        # adaptive cruise control를 활용한 속도 계획
        self.adaptive_cruise_control.check_object(
            planning_path, forward_object_info_dic_list, current_traffic_light
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
        # 속도 제어를 위한 PID control
        acc_cmd = self.pid.get_output(target_velocity, vehicle_state.velocity)
        # target velocity가 0이고, 일정 속도 이하일 경우 full brake를 하여 차량을 멈추도록 함.
        if round(target_velocity) == 0 and vehicle_state.velocity < 2:
            acc_cmd = -1.
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
