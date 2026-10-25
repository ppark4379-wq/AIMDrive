"""Persistent Expert state and backwards-compatible plan/control entry point.

The algorithms live in planning/ and control/. The same instances and mutable
state survive every tick, including callers of the legacy execute() API.
"""
from .planning.common.object_selection import ForwardObjectDetector, _legacy_acc_entries
from .planning.common.reference_path import PathManager
from .planning.common.adaptive_cruise_control import AdaptiveCruiseControl
from .planning.traffic.stop_controller import TrafficLightStopController
from .planning.section_planning import LatticePlanningProcess
from .planning.ve_planning import VePerceptionPlanningProcess, perception_source
from .planning.planning_process import PlanningProcess, ve_plan
from .planning.highway.second_merge.speed_limits import _second_merge_curvature_bypass
from .planning.highway.second_merge.speed_limits import _second_merge_slowdown_band
from .planning.highway.common.speed_limits import _merge_ego_curvature_speed_floor
from .planning.highway.common.speed_limits import _hold_merge_ego_curvature_speed
from .planning.static_obstacle.speed_policy import _smooth_static_target_speed
from .control.path_controller import PathController
from .control.control_process import ControlProcess
from .interfaces.control_command import ControlInput
from .config.config import Config


class AutonomousDriving(PathController):
    def __init__(self):
        config = Config()
        self.perception_source = perception_source(config["common"])

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
            route_signal_targets=config["map"].get("route_signal_targets"),
            **config['planning'].get('traffic_light_stop', {})
        )

        self.forward_object_detector = ForwardObjectDetector(config["map"]["traffic_light_list"])

        self.adaptive_cruise_control = AdaptiveCruiseControl(
            vehicle_length=config['common']['vehicle_length'],
            time_gap_zones=config['planning']['velocity_profile'].get('speed_limit_zones', []),
            **config['planning']['adaptive_cruise_control']
        )
        # Uses the measured static-obstacle and highway section centerlines.
        if self.perception_source == "ve":
            self.lattice_planning_process = VePerceptionPlanningProcess(
                config["planning"].get("ve_perception", {}),
                vehicle_length=config["common"]["vehicle_length"],
                vehicle_width=config["common"].get("vehicle_width", 2.0),
            )
        else:
            self.lattice_planning_process = LatticePlanningProcess(
                map_data_dir=config["map"]["data_dir"],
                jamming_enabled=config["planning"].get("jamming", {}).get("enabled", True),
            )
        self._init_control(config)
        self._merge_curvature_floor_zone = None
        self._merge_curvature_floor_speed = None
        self._static_reference_speed = None
        self._static_speed_ramp = False


    # Compatibility method bindings preserve callers and original state ownership.
    _legacy_acc_entries = _legacy_acc_entries
    _smooth_static_target_speed = _smooth_static_target_speed
    _second_merge_curvature_bypass = staticmethod(_second_merge_curvature_bypass)
    _second_merge_slowdown_band = staticmethod(_second_merge_slowdown_band)
    _merge_ego_curvature_speed_floor = staticmethod(_merge_ego_curvature_speed_floor)
    _hold_merge_ego_curvature_speed = _hold_merge_ego_curvature_speed

    @property
    def planning(self):
        if not hasattr(self, '_planning_process'):
            self._planning_process = PlanningProcess(self)
        return self._planning_process

    @property
    def control(self):
        if not hasattr(self, '_control_process'):
            self._control_process = ControlProcess(self)
        return self._control_process

    def execute(self, vehicle_state, dynamic_object_list, current_traffic_light,
                sync_mode=False, control_dt=None, ego_header_seq=None,
                ego_source_ns=None, perception_health=None):
        plan = self.planning.run(
            vehicle_state, dynamic_object_list, current_traffic_light,
            sync_mode=sync_mode, control_dt=control_dt,
            ego_header_seq=ego_header_seq, ego_source_ns=ego_source_ns,
            perception_health=perception_health)
        return self.control.run(vehicle_state, plan)

    def _execute_ve(self, vehicle_state, objects, traffic_light, health):
        return self.control.run(vehicle_state,
                                ve_plan(self, vehicle_state, objects, traffic_light, health))

    def force_ve_brake(self):
        """No path is passed to the path follower on missing/blocked perception."""
        self.pid.reset()
        self.previous_target_velocity = 0.
        self.previous_acc_cmd = -1.
        return ControlInput(-1., getattr(self, "previous_front_steer_rad", 0.))
