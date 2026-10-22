import json
import math
import os
import random
import time

from ..adapters.object_ghost_audit import ObjectGhostAudit
from ..utils import safe_print as print
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..utils.transform_utils import polyline_length
from .full_loop_parts import (
    actor_lifecycle,
    dynamic_obstacle,
    highway,
    roundabout,
    route_profile,
    static_blockage_release,
    signal_truth_log,
    static_obstacle,
    traffic_light_override,
    tunnel,
    tunnel_dynamic,
    tunnel_vehicle_dynamic,
    vehicle_common,
)


class FullLoopScenario(FullLoopDriveBase):
    """One official MOLIT lap with run-scoped environment and fixed NPCs."""

    def _clock_domain(self):
        return getattr(getattr(self, "clock", None), "domain", None)

    def _scenario_now(self):
        """Use the injected scenario clock, with a host-monotonic fallback."""
        clock = getattr(self, "clock", None)
        return float(clock.now()) if clock is not None else time.monotonic()

    zone_name = "full_loop"
    scenario_name = "full_loop"

    def setup(self):
        self.clean_road_mode = bool(self.cfg.get("clean_road_mode", True))
        self.randomize_links = False
        self.route_configured = False
        self.route_setup_mode = None
        self.zone_allowed_links = set()
        self.recent_start_links = []
        self.recent_route_keys = []
        self.random_route_pool = None
        self.scenario_actors = []
        self.highway_npcs = []
        self.highway_spawn_points = []
        self.highway_spawn_points_active = False
        self.highway_spawn_point_activation_started = False
        self.highway_spawn_point_next_retry_at = 0.0
        self.highway_traffic_cleared = False
        self.highway_npc_planned_count = 0
        self.highway_merge_static_vehicle = None
        self.highway_merge_static_vehicle_info = None
        self.tunnel_npc = None
        self.tunnel_event_phase = "uninitialized"
        self.tunnel_event_result = {}
        self.tunnel_event_last_print_sec = -1
        self.roundabout_npcs = []
        self.dynamic_obstacle = None
        self.dynamic_obstacle_catalog = []
        self.dynamic_obstacle_events = []
        self.dynamic_obstacle_event_index = 0
        self.dynamic_obstacle_owned_ids = set()
        self.dynamic_obstacle_phase = "uninitialized"
        self.dynamic_obstacle_started_at = None
        self.dynamic_obstacle_started_scenario_time = None
        self.dynamic_obstacle_min_ego_distance_m = float("inf")
        self.static_obstacle = None
        self.static_obstacles = []
        self.static_obstacle_expected = None
        self.static_obstacle_expectations = []
        self.static_obstacle_models_selected = []
        self.static_obstacle_scales = []
        self.static_obstacle_route_s = None
        self.static_obstacle_last_verify_elapsed = float("-inf")
        self.static_obstacle_pass_verified = False
        traffic_light_override.initialize(self)
        static_blockage_release.initialize(self)
        signal_truth_log.initialize(self)
        actor_lifecycle.initialize(self)

        configured_seed = self.cfg.get("random_seed")
        self.run_seed = (
            int(configured_seed)
            if configured_seed is not None
            else int.from_bytes(os.urandom(8), "big")
        )
        self.random = random.Random(self.run_seed)
        self.run_instance_id = (
            f"{int(time.time() * 1000)}_{self.run_seed & 0xFFFFFFFF:08x}"
        )
        print(
            f"[FullLoop] run_seed={self.run_seed}, "
            f"run_instance_id={self.run_instance_id}"
        )

        self.route_definition = self.load_route_definition()
        self.prepare_full_loop_route()
        tunnel_dynamic.prepare_tunnel_dynamic(self)
        tunnel_vehicle_dynamic.prepare_tunnel_vehicle(self)
        traffic_light_override.prepare(self)
        self.profile_index, self.profile, self.previous_profile_state = self.select_profile()
        self.scenario_hour = self.select_scenario_hour()
        self.scenario_time_last_verify_elapsed = float("-inf")

        self.grpc.start_world(self.start_tf)
        if hasattr(self.grpc, "resume_world"):
            self.grpc.resume_world()
        try:
            self.apply_weather()
            self.apply_time()
            if self.clean_road_mode:
                self.static_obstacle = None
                self.static_obstacles = []
                self.static_obstacle_model = None
                self.static_obstacle_scale = None
                self.dynamic_obstacle_model = None
                self.dynamic_obstacle_phase = "disabled"
                self.tunnel_event_phase = "disabled"
                print("[FullLoop] clean_road_mode: obstacles, NPCs and tunnel event disabled")
            else:
                self.resolve_profile_models()
                self.spawn_static_obstacle()
                self.spawn_dynamic_obstacle()
                self.spawn_roundabout_npcs()
                self.spawn_highway_npcs()
                if self.tunnel_vehicle_handler is not None:
                    self.tunnel_vehicle_handler.arm(self)
                elif self.tunnel_dynamic_handler is None:
                    self.arm_tunnel_sudden_brake()
                else:
                    tunnel_dynamic.arm_tunnel_dynamic(self)
            self.configure_drive_with_retries()
            self.persist_profile_state()
        except BaseException as error:
            if self.tunnel_vehicle_handler is not None and hasattr(self, "tunnel_vehicle_result"):
                self.tunnel_vehicle_handler.record_interruption(self, error)
            self.cleanup_spawned_actors()
            raise

        print(
            f"[FullLoop] ready profile={self.profile['name']}, "
            f"weather={self.profile['weather']}, "
            f"time={self.scenario_hour:02d}:00, "
            f"static={self.static_obstacle_models_selected or self.static_obstacle_model}, "
            f"dynamic={self.dynamic_obstacle_model}, "
            f"roundabout_npcs={len(self.roundabout_npcs)}, "
            f"highway_npcs_planned={self.highway_npc_planned_count}"
        )


    # Public scenario interface kept stable; implementations live by zone.
    load_route_definition = route_profile.load_route_definition
    load_global_path = route_profile.load_global_path
    prepare_full_loop_route = route_profile.prepare_full_loop_route
    path_s_at_index = route_profile.path_s_at_index
    profile_state_path = route_profile.profile_state_path
    read_profile_state = route_profile.read_profile_state
    validate_profiles = route_profile.validate_profiles
    select_profile = route_profile.select_profile
    filter_models = route_profile.filter_models
    resolve_profile_models = route_profile.resolve_profile_models
    apply_weather = route_profile.apply_weather
    select_scenario_hour = route_profile.select_scenario_hour
    apply_time = route_profile.apply_time

    make_actor_scale = static_obstacle.make_actor_scale
    _read_static_obstacle_state = static_obstacle.read_static_obstacle_state
    select_static_obstacle_placements = (
        static_obstacle.select_static_obstacle_placements
    )
    spawn_static_obstacle = static_obstacle.spawn_static_obstacle
    maintain_static_obstacles_for_editor = (
        static_obstacle.maintain_static_obstacles_for_editor
    )
    verify_static_obstacle = static_obstacle.verify_static_obstacle

    spawn_dynamic_obstacle = dynamic_obstacle.spawn_dynamic_obstacle
    _spawn_active_dynamic_obstacle = (
        dynamic_obstacle._spawn_active_dynamic_obstacle
    )
    _destroy_dynamic_obstacle_actor = (
        dynamic_obstacle._destroy_dynamic_obstacle_actor
    )
    _finish_dynamic_obstacle = dynamic_obstacle._finish_dynamic_obstacle
    update_dynamic_obstacle = dynamic_obstacle.update_dynamic_obstacle
    dynamic_obstacle_reports = dynamic_obstacle.dynamic_obstacle_reports
    verify_dynamic_obstacles_removed = (
        dynamic_obstacle.verify_dynamic_obstacles_removed
    )

    vehicle_models = vehicle_common.vehicle_models
    spawn_vehicle_with_retry = vehicle_common.spawn_vehicle_with_retry

    rotate_loop_route = roundabout.rotate_loop_route
    spawn_roundabout_npcs = roundabout.spawn_roundabout_npcs
    _spawn_roundabout_npcs_now = roundabout.spawn_roundabout_npcs_now
    _retire_roundabout_npcs = roundabout.retire_roundabout_npcs
    update_roundabout_npcs = roundabout.update_roundabout_npcs

    spawn_highway_npcs = highway.spawn_highway_npcs
    spawn_highway_merge_static_vehicle = (
        highway.spawn_highway_merge_static_vehicle
    )
    _route_geometry = highway._route_geometry
    _nearest_waypoint = staticmethod(vehicle_common._nearest_waypoint)
    _nearest_official_link = highway._nearest_official_link
    _spawn_highway_vehicle = vehicle_common._spawn_highway_vehicle
    activate_highway_spawn_points = highway.activate_highway_spawn_points
    _set_highway_spawn_points_enabled = (
        highway._set_highway_spawn_points_enabled
    )
    deactivate_highway_npcs = highway.deactivate_highway_npcs
    update_highway_actor_lifecycle = highway.update_highway_actor_lifecycle

    arm_tunnel_sudden_brake = tunnel.arm_tunnel_sudden_brake
    _actor_speed = staticmethod(tunnel._actor_speed)
    _tunnel_route_start = tunnel._tunnel_route_start
    spawn_tunnel_sudden_brake_npc = tunnel.spawn_tunnel_sudden_brake_npc
    _write_tunnel_event_report = tunnel._write_tunnel_event_report
    _destroy_tunnel_npc = tunnel._destroy_tunnel_npc
    _command_tunnel_hard_brake = tunnel._command_tunnel_hard_brake
    _resume_tunnel_npc_after_event = tunnel._resume_tunnel_npc_after_event
    update_tunnel_sudden_brake = tunnel.update_tunnel_sudden_brake

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        if (
            not self.clean_road_mode
            and actor_lifecycle.ready_for_update(self, elapsed)
        ):
            self.update_roundabout_npcs(elapsed, ego_state, current_s)
            self.update_highway_actor_lifecycle(elapsed, ego_state, current_s)
        verify_period = float(
            self.cfg.get("scenario_time_verify_period_sec", 30.0)
        )
        if (
            verify_period > 0.0
            and elapsed - self.scenario_time_last_verify_elapsed >= verify_period
        ):
            current_hour = self.grpc.get_time()
            if current_hour != self.scenario_hour:
                print(
                    f"[FullLoop] scenario time drift detected: "
                    f"{current_hour:02d}:00 -> {self.scenario_hour:02d}:00"
                )
                self.apply_time()
            self.scenario_time_last_verify_elapsed = float(elapsed)
        if self.clean_road_mode:
            traffic_light_override.update(self, elapsed, ego_state, current_s)
            return
        self.verify_static_obstacle(elapsed, current_s)
        static_blockage_release.update(self, elapsed, ego_state, current_s)
        signal_truth_log.update(self, elapsed, ego_state)
        self.update_dynamic_obstacle(ego_state, current_s)
        traffic_light_override.update(self, elapsed, ego_state, current_s)
        self.activate_highway_spawn_points(elapsed, current_s)
        if (
            self.highway_spawn_point_activation_started
            and not self.highway_traffic_cleared
            and current_s >= self.highway_exit_s
        ):
            self.deactivate_highway_npcs(current_s)
        if getattr(self, "tunnel_vehicle_handler", None) is not None:
            self.tunnel_vehicle_handler.update(self, elapsed, ego_state, current_s)
        elif self.tunnel_dynamic_handler is None:
            self.update_tunnel_sudden_brake(elapsed, ego_state, current_s)
        else:
            tunnel_dynamic.update_tunnel_dynamic(self, elapsed, ego_state, current_s)

    def run_timeline(self):
        try:
            super().run_timeline()
            if getattr(self, "tunnel_vehicle_handler", None) is not None:
                self.tunnel_vehicle_handler.finish_at_ego_goal(self)
            if (self.tunnel_dynamic_handler is not None
                    and self.tunnel_dynamic_phase != "completed"):
                raise RuntimeError("Route ended before every tunnel pedestrian event completed")
            if getattr(self, "dynamic_obstacle_events", []):
                if self.dynamic_obstacle_event_index != len(self.dynamic_obstacle_events):
                    raise RuntimeError("Route ended before every dynamic crossing was completed")
                if self.dynamic_validation.get("corridor_observations", 0) == 0:
                    raise RuntimeError("Dynamic run has no verified Ego corridor observations")
                self.dynamic_validation["goal_reached"] = True
                dynamic_obstacle._persist_progress(self)
        except BaseException as error:
            vehicle_handler = getattr(self, "tunnel_vehicle_handler", None)
            if vehicle_handler is not None:
                vehicle_handler.record_interruption(self, error)
            handler = getattr(self, "tunnel_dynamic_handler", None)
            if handler is not None and callable(getattr(handler, "record_interruption", None)):
                handler.record_interruption(self, error)
            if getattr(self, "dynamic_obstacle_events", []):
                self.dynamic_validation["safety_validation_passed"] = False
                self.dynamic_validation["failure_reason"] = str(error) or type(error).__name__
                dynamic_obstacle._persist_progress(self, during_cleanup=True)
            raise

    def persist_profile_state(self):
        state = {
            "profile_index": self.profile_index,
            "profile_name": self.profile["name"],
            "weather": str(self.profile["weather"]).upper(),
            "time_hour": self.scenario_hour,
            "static_obstacle_model": self.static_obstacle_model,
            "static_obstacle_scale": self.static_obstacle_scale,
            "static_obstacle_models": self.static_obstacle_models_selected,
            "static_obstacle_scales": self.static_obstacle_scales,
            "static_obstacle_count": len(self.static_obstacles),
            "dynamic_obstacle_model": self.dynamic_obstacle_model,
            "dynamic_obstacle_count": len(self.dynamic_obstacle_events),
            "dynamic_obstacle_events": self.dynamic_obstacle_reports(),
            "dynamic_validation": dict(getattr(self, "dynamic_validation", {})),
            "run_seed": self.run_seed,
            "saved_at_unix": time.time(),
        }
        previous = self.previous_profile_state
        for key in ("weather", "time_hour"):
            forced_key = {
                "weather": "forced_weather",
                "time_hour": "forced_time_hour",
            }[key]
            if (
                self.cfg.get(forced_key) is None
                and previous.get(key) == state[key]
            ):
                raise RuntimeError(
                    f"Run-scoped profile did not change for {key}: {state[key]}"
                )

        path = self.profile_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        os.replace(temporary, path)
        print(f"[FullLoop] profile state saved: {path}")

    def restart_to_start_and_drive(self):
        raise RuntimeError(
            "Full-loop automatic restart is disabled so run-scoped NPC actors "
            "and environment remain fixed. Start a new scenario process instead."
        )

    def cleanup_spawned_actors(self):
        # A preflight failure can occur before setup() initializes any
        # scenario-owned actor state. In that case there is nothing to clean.
        if not hasattr(self, "scenario_actors"):
            return
        try:
            self._set_highway_spawn_points_enabled(False)
        except Exception as e:
            print(f'[FullLoop] SpawnPoint disable warning: {e}')
        for actor in reversed(getattr(self, "scenario_actors", [])):
            try:
                actor.destroy()
            except Exception as e:
                print(f"[FullLoop] actor cleanup warning: {e}")
        self.scenario_actors = []
        reset_result = False
        try:
            # Native SpawnPoints and every vehicle they generated share this
            # client's key. Wait for MORAI to apply run-owned cleanup before
            # the next sequence creates new runtime SpawnPoint IDs.
            lifecycle = self.cfg.get('highway_npc', {}).get(
                'spawn_point_lifecycle',
                {},
            )
            cleanup_settle_sec = max(
                0.0,
                float(lifecycle.get('cleanup_settle_sec', 1.5)),
            )
            reset_result = bool(
                self.grpc.reset_actors(settle_sec=cleanup_settle_sec)
            )
        except Exception as e:
            print(f'[FullLoop] run-owned SpawnPoint cleanup warning: {e}')
        self.highway_spawn_points = []
        self.highway_spawn_points_active = False
        if getattr(self, "dynamic_obstacle_owned_ids", set()) or any(
            event.get("actor") is not None or event.get("spawn_attempts")
            for event in getattr(self, "dynamic_obstacle_events", [])
        ):
            if not reset_result:
                raise RuntimeError(
                    "Runner-owned actor reset failed after dynamic scenario"
                )
            self.verify_dynamic_obstacles_removed()

    def cleanup(self):
        vehicle_handler = getattr(self, "tunnel_vehicle_handler", None)
        if (vehicle_handler is not None and hasattr(self, "tunnel_vehicle_result")
                and self.tunnel_vehicle_phase not in ("completed", "failed")):
            vehicle_handler.record_interruption(
                self, RuntimeError("Scenario teardown before vehicle interaction completed"))
        base_cleanup_ok = False
        try:
            super().cleanup()
            base_cleanup_ok = True
        finally:
            try:
                try:
                    if vehicle_handler is not None and hasattr(self, "tunnel_vehicle_result"):
                        vehicle_handler.teardown(self)
                finally:
                    self.cleanup_spawned_actors()
            except BaseException as error:
                if getattr(self, "dynamic_obstacle_events", []):
                    self.dynamic_validation["cleanup_verified"] = False
                    self.dynamic_validation["safety_validation_passed"] = False
                    self.dynamic_validation["failure_reason"] = str(error)
                    dynamic_obstacle._persist_progress(self, during_cleanup=True)
                raise
            if getattr(self, "dynamic_obstacle_events", []):
                self.dynamic_validation["cleanup_verified"] = True
                # Completing diagnostic collection is not safety approval.
                # Also include collection contacts outside the dynamic zone.
                metrics = self.dynamic_validation
                metrics["safety_validation_passed"] = bool(
                    base_cleanup_ok
                    and metrics.get("goal_reached")
                    and self.dynamic_obstacle_event_index == len(self.dynamic_obstacle_events)
                    and metrics.get("corridor_observations", 0) > 0
                    and not metrics.get("collision_objects")
                    and not getattr(self, "collision_events", 0)
                    and not metrics.get("failure_reason")
                    and not metrics.get("assessment_issues")
                    and not metrics.get("recording_errors")
                )
                dynamic_obstacle._persist_progress(self, during_cleanup=True)

class FullLoopAfterStaticScenario(FullLoopScenario):
    """The same full-loop scenario, starting just after the static obstacle."""

    scenario_name = "full_loop_after_static"

    def path_s_at_index(self, index):
        if hasattr(self, "_full_route_points"):
            index = max(0, min(int(index), len(self._full_route_points) - 1))
            global_s = polyline_length(self._full_route_points[: index + 1])
            return max(0.0, global_s - self._route_prefix_s)
        return super().path_s_at_index(index)

    def prepare_full_loop_route(self):
        route_debug_enabled = bool(self.cfg.get("route_debug_enabled", False))
        self.cfg["route_debug_enabled"] = False
        try:
            super().prepare_full_loop_route()
        finally:
            self.cfg["route_debug_enabled"] = route_debug_enabled

        start_index = int(self.cfg.get("after_static_start_path_index", 703))
        full_points = list(self.route_points)
        if start_index <= 0 or start_index >= len(full_points) - 1:
            raise RuntimeError(
                f"Invalid after-static start path index: {start_index}"
            )

        global_highway_activation_s = self.highway_activation_s
        global_highway_exit_s = self.highway_exit_s
        global_tunnel_entry_s = self.tunnel_entry_s
        global_tunnel_exit_s = self.tunnel_exit_s
        self._full_route_points = full_points
        self._route_prefix_s = polyline_length(full_points[: start_index + 1])
        self.route_points = full_points[start_index:]
        self.route_length_m = polyline_length(self.route_points)
        self.pure_pursuit_last_s = 0.0

        first = self.route_points[0]
        second = next(
            point for point in self.route_points[1:]
            if math.hypot(point[0] - first[0], point[1] - first[1]) > 1e-3
        )
        start_yaw = math.degrees(
            math.atan2(second[1] - first[1], second[0] - first[0])
        )
        self.start_tf = self.grpc.make_transform(
            first[0], first[1], first[2], start_yaw
        )

        last = self.route_points[-1]
        previous = next(
            point for point in reversed(self.route_points[:-1])
            if math.hypot(point[0] - last[0], point[1] - last[1]) > 1e-3
        )
        self.goal_x, self.goal_y, self.goal_z = last[:3]
        self.goal_yaw = math.degrees(
            math.atan2(last[1] - previous[1], last[0] - previous[0])
        )

        self.highway_activation_s = (
            global_highway_activation_s - self._route_prefix_s
        )
        self.highway_exit_s = global_highway_exit_s - self._route_prefix_s
        self.tunnel_entry_s = global_tunnel_entry_s - self._route_prefix_s
        self.tunnel_exit_s = global_tunnel_exit_s - self._route_prefix_s
        self.cfg["expert_is_closed_path"] = False

        print(
            f"[FullLoopAfterStatic] start_path_index={start_index}, "
            f"remaining_points={len(self.route_points)}, "
            f"remaining_length={self.route_length_m:.1f}m, "
            f"destination=original START/END"
        )
        if route_debug_enabled:
            self.dump_route_debug()

class FullLoopHighwayTunnelTestScenario(FullLoopAfterStaticScenario):
    """Focused run from checkpoint 10 through the highway exit and tunnel."""

    scenario_name = "full_loop_highway_tunnel_test"

    def highway_object_audit_enabled(self):
        return bool(
            self.cfg.get("highway_object_audit", {}).get("enabled", False)
        )

    def setup(self):
        self.object_ghost_audit = None
        super().setup()
        if not self.highway_object_audit_enabled():
            return

        audit_cfg = dict(self.cfg.get("highway_object_audit", {}))
        runtime_root = os.path.abspath(
            self.global_cfg["paths"]["runtime_root"]
        )
        output_dir = os.path.abspath(
            os.path.join(
                runtime_root,
                str(audit_cfg.get("output_subdir", "object_audit")),
            )
        )
        if os.path.commonpath([runtime_root, output_dir]) != runtime_root:
            raise RuntimeError(
                f"Object audit output escapes runtime root: {output_dir}"
            )
        self.object_ghost_audit = ObjectGhostAudit(
            grpc_client=self.grpc,
            output_dir=output_dir,
            run_instance_id=self.run_instance_id,
            sample_period_sec=audit_cfg.get("sample_period_sec", 0.5),
            match_distance_m=audit_cfg.get("match_distance_m", 5.0),
            ros_max_age_sec=audit_cfg.get("ros_max_age_sec", 1.0),
            candidate_samples=audit_cfg.get("candidate_samples", 3),
        )
        self.object_ghost_audit.start()

    def prepare_full_loop_route(self):
        audit_enabled = self.highway_object_audit_enabled()
        route_debug_enabled = bool(
            self.cfg.get("route_debug_enabled", False)
        )
        if audit_enabled:
            self.cfg["route_debug_enabled"] = False
        start_index = int(
            self.cfg.get("highway_tunnel_test_start_path_index", 2275)
        )
        self.cfg["after_static_start_path_index"] = start_index
        try:
            super().prepare_full_loop_route()
        finally:
            if audit_enabled:
                self.cfg["route_debug_enabled"] = route_debug_enabled

        if audit_enabled:
            end_index = int(
                self.cfg["highway_object_audit"].get(
                    "end_path_index",
                    3529,
                )
            )
            if (
                end_index <= start_index
                or end_index >= len(self._full_route_points)
            ):
                raise RuntimeError(
                    "Invalid highway object audit path range: "
                    f"{start_index}..{end_index}"
                )
            local_end_index = end_index - start_index
            self.route_points = self.route_points[: local_end_index + 1]
            self.route_length_m = polyline_length(self.route_points)
            last = self.route_points[-1]
            previous = next(
                point for point in reversed(self.route_points[:-1])
                if math.hypot(
                    point[0] - last[0],
                    point[1] - last[1],
                ) > 1e-3
            )
            self.goal_x, self.goal_y, self.goal_z = last[:3]
            self.goal_yaw = math.degrees(
                math.atan2(
                    last[1] - previous[1],
                    last[0] - previous[0],
                )
            )
            endpoint_match = self._nearest_official_link(
                self.goal_x,
                self.goal_y,
            )
            if endpoint_match is not None:
                self.end_link = self.route_links[endpoint_match[1]]
            self.cfg["expert_is_closed_path"] = False
            print(
                "[FullLoopHighwayObjectAudit] route "
                f"global_index={start_index}..{end_index}, "
                f"length={self.route_length_m:.1f}m, "
                f"goal=({self.goal_x:.2f},{self.goal_y:.2f}), "
                f"end_link={self.end_link}; tunnel excluded"
            )
            if route_debug_enabled:
                self.dump_route_debug()

        print(
            f"[FullLoopHighwayTunnelTest] start_path_index={start_index}, "
            f"start_yaw={self.start_tf.rotation.z:.1f}deg, "
            f"highway_exit_s={self.highway_exit_s:.1f}m, "
            f"tunnel_s={self.tunnel_entry_s:.1f}-{self.tunnel_exit_s:.1f}m"
        )

    def spawn_static_obstacle(self):
        self.static_obstacle_model = self.static_obstacle_models[0]
        scale_overrides = self.cfg.get("static_obstacle_model_scales", {})
        spec = self.route_definition["zones"]["static_obstacle"]
        values = scale_overrides.get(
            self.static_obstacle_model,
            spec.get("scale", [1.0, 1.0, 1.0]),
        )
        self.static_obstacle_scale = [float(value) for value in values]
        self.static_obstacle = None
        print("[FullLoopHighwayTunnelTest] static obstacle skipped")

    def spawn_dynamic_obstacle(self):
        self.dynamic_obstacle_model = self.dynamic_obstacle_models[0]
        self.dynamic_obstacle = None
        self.dynamic_obstacle_phase = "disabled"
        print("[FullLoopHighwayTunnelTest] pedestrian obstacle skipped")

    def spawn_roundabout_npcs(self):
        self.roundabout_npcs = []
        print("[FullLoopHighwayTunnelTest] roundabout traffic skipped")

    def spawn_highway_npcs(self):
        FullLoopScenario.spawn_highway_npcs(self)
        event_text = (
            "object audit ends before the tunnel"
            if self.highway_object_audit_enabled()
            else "tunnel event remains enabled"
        )
        print(
            f"[FullLoopHighwayTunnelTest] native highway SpawnPoints included; "
            f"{event_text}"
        )

    def arm_tunnel_sudden_brake(self):
        if not self.highway_object_audit_enabled():
            return super().arm_tunnel_sudden_brake()
        self.tunnel_npc = None
        self.tunnel_event_phase = "disabled"
        self.tunnel_event_result = {
            "disabled_reason": "highway object audit ends before tunnel"
        }
        print("[FullLoopHighwayObjectAudit] tunnel event skipped")

    def on_expert_timeline_tick(
        self,
        elapsed,
        ego_state,
        current_s,
        remaining_s,
    ):
        super().on_expert_timeline_tick(
            elapsed=elapsed,
            ego_state=ego_state,
            current_s=current_s,
            remaining_s=remaining_s,
        )
        audit = getattr(self, "object_ghost_audit", None)
        if audit is not None:
            audit.sample(
                elapsed=elapsed,
                ego_state=ego_state,
                current_s=current_s,
            )

    def cleanup(self):
        audit = getattr(self, "object_ghost_audit", None)
        if audit is not None:
            try:
                audit.stop()
            except Exception as error:
                print(f"[ObjectAudit] cleanup warning: {error}")
            self.object_ghost_audit = None
        super().cleanup()
