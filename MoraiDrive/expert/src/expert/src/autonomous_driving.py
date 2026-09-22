#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .obstacle.forward_object_detector import ForwardObjectDetector
from .localization.path_manager import PathManager
from .planning.adaptive_cruise_control import AdaptiveCruiseControl
from .planning.traffic_light_stop import TrafficLightStopController
from .planning.lattice_planning import LatticePlanningProcess
from .planning.lattice_planning import parameters as lattice_parameters
from .planning.highway_lane_planning import parameters as highway_parameters
from .control.path_controller import PathController
from .control.control_input import ControlInput
from .config.config import Config

import numpy as np

class AutonomousDriving(PathController):
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
            vehicle_length=config['common']['vehicle_length'],
            time_gap_zones=config['planning']['velocity_profile'].get('speed_limit_zones', []),
            **config['planning']['adaptive_cruise_control']
        )
        # Uses the measured static-obstacle and highway section centerlines.
        self.lattice_planning_process = LatticePlanningProcess(
            map_data_dir=config["map"]["data_dir"]
        )
        self._init_control(config)
        self._merge_curvature_floor_zone = None
        self._merge_curvature_floor_speed = None
        self._static_reference_speed = None
        self._static_speed_ramp = False

    def execute(
        self,
        vehicle_state,
        dynamic_object_list,
        current_traffic_light,
        sync_mode=False,
        control_dt=None,
    ):
        self._configure_runtime(sync_mode=sync_mode, control_dt=control_dt)
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
        preview_start = self.path_manager.current_waypoint
        static_maneuver = getattr(self.lattice_planning_process, "static_avoidance", None)
        static_preview = static_maneuver is not None and (
            static_maneuver.active or self.lattice_planning_process._is_inside_static_zone(vehicle_state)
        )
        recorded = getattr(self.lattice_planning_process, "recorded_static", None)
        static_preview = static_preview or (recorded is not None and (recorded.active or recorded.contains(vehicle_state)))
        preview_count = lattice_parameters.STATIC_PREVIEW_POINTS if static_preview else 120
        roundabout = self.lattice_planning_process.roundabout_planner
        if roundabout.active or self.lattice_planning_process._roundabout_entry_state(vehicle_state)["in_approach"]:
            preview_count = max(preview_count, lattice_parameters.ROUNDABOUT_PREVIEW_POINTS)
        prediction_path = self.path[preview_start:preview_start + preview_count]
        if self.path_manager.is_closed_path and len(prediction_path) < preview_count:
            prediction_path = prediction_path + self.path[:preview_count - len(prediction_path)]
        lattice_result = self.lattice_planning_process.run(
            vehicle_state, local_path, planning_object_info_dic_list,
            desired_velocity=planned_velocity,
            prediction_path=prediction_path,
        )
        planning_path = lattice_result.path
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
        legacy_acc_entries = self._legacy_acc_entries(
            vehicle_state, lattice_result, forward_object_info_dic_list
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
        legacy_target_velocity = target_velocity
        target_velocity = min(
            target_velocity,
            signal_target_velocity,
            lattice_target_velocity,
            getattr(lattice_result, "acc_target_velocity", float("inf")),
        )
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
            key = "roundabout_entry" if lattice_result.mode == "roundabout_entry" else "static_avoidance"
            details = lattice_result.debug_details.setdefault(key, {})
            details.setdefault("phase", lattice_result.reason)
            details["speed_limits"] = limits
            details["speed_source"] = min(limits, key=limits.get)
            details["final_target_velocity"] = max(0.0, target_velocity)
        acc_cmd, raw_steering_rad, front_steer_rad = self._calculate_control(
            vehicle_state, planning_path, target_velocity, lattice_result)

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
            if "static_avoidance" in details:
                print("[STATIC] {}".format(details["static_avoidance"]))
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



    def _smooth_static_target_speed(self, target_velocity, vehicle_state, lattice_result):
        """Carry a rise-only speed ramp through the recorded-to-global handoff."""
        details = getattr(lattice_result, 'debug_details', {}).get('static_avoidance', {})
        recorded = details.get('planner') == 'recorded_route_graph'
        if recorded and details.get('path_validated', False):
            self._static_speed_ramp = True
        if not getattr(self, '_static_speed_ramp', False):
            return target_velocity
        if getattr(lattice_result, 'mode', '') in ('highway', 'highway_lane', 'roundabout_entry'):
            self._static_speed_ramp = False
            return target_velocity
        previous = getattr(self, 'previous_target_velocity', None)
        if previous is None:
            previous = max(0., vehicle_state.velocity)
        # Brake requests are never delayed or lifted by acceleration smoothing.
        if target_velocity <= previous:
            return target_velocity
        dt = min(.2, max(.001, getattr(self, 'control_dt', 1./30.)))
        ceiling = previous + lattice_parameters.STATIC_TARGET_ACCEL*dt
        target = min(target_velocity, ceiling)
        if not recorded and target >= target_velocity:
            self._static_speed_ramp = False
        return target

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



    def _legacy_acc_entries(self, vehicle_state, lattice_result, forward_entries):
        """Avoid applying path-distance vehicle ACC twice on mapped highways.

        The highway lane planner already associates vehicles with lane links,
        checks fallback collisions, and returns an ACC speed ceiling.  The
        legacy path-distance detector can mistake an adjacent/old-lane vehicle
        for a lead vehicle immediately after a lane change because the two
        paths remain close.  Keep pedestrians and traffic lights in the legacy
        controller, but let the lane planner be the sole vehicle authority.
        """
        static_details = getattr(lattice_result, "debug_details", {}).get("static_avoidance", {})
        if (getattr(lattice_result, "mode", "") == "static_obstacle"
                and getattr(lattice_result, "active", False)
                and static_details.get("path_validated", False)):
            # Only bypass static vehicle ACC after validating the complete swept
            # path. Moving vehicles, people and traffic signals keep their stops.
            return [entry for entry in forward_entries if not (
                getattr(entry["object_info"], "is_static", False)
                and getattr(entry["object_info"], "type", -1) in (1, 2)
                and abs(getattr(entry["object_info"], "velocity", 0.0)) < 0.2
            )]
        if getattr(lattice_result, "mode", "") != "highway_lane":
            return [
                entry
                for entry in forward_entries
                if not self.lattice_planning_process.roundabout_entry_should_ignore_legacy_acc(
                    vehicle_state, entry["object_info"]
                )
            ]
        return [
            entry
            for entry in forward_entries
            if getattr(entry["object_info"], "type", -1) not in (1, 2)
        ]
