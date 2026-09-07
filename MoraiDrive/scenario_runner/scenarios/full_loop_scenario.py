import json
import math
import os
import random
import time

import yaml
from ..adapters.object_ghost_audit import ObjectGhostAudit

from ..utils.link_graph import LinkGraph
from ..utils.transform_utils import (
    interpolate_on_polyline,
    polyline_length,
    project_distance_on_polyline,
)
from ..core.full_loop_drive_base import FullLoopDriveBase


class FullLoopScenario(FullLoopDriveBase):
    """One official MOLIT lap with run-scoped environment and fixed NPCs."""

    zone_name = "full_loop"
    scenario_name = "full_loop"

    def setup(self):
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
        self.tunnel_npc = None
        self.tunnel_event_phase = "uninitialized"
        self.tunnel_event_result = {}
        self.tunnel_event_last_print_sec = -1
        self.roundabout_npcs = []
        self.dynamic_obstacle = None
        self.dynamic_obstacle_phase = "uninitialized"
        self.dynamic_obstacle_started_at = None
        self.dynamic_obstacle_min_ego_distance_m = float("inf")

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
        self.profile_index, self.profile, self.previous_profile_state = self.select_profile()
        self.scenario_hour = self.select_scenario_hour()
        self.scenario_time_last_verify_elapsed = float("-inf")

        self.grpc.start_world(self.start_tf)
        if hasattr(self.grpc, "resume_world"):
            self.grpc.resume_world()
        try:
            self.apply_weather()
            self.apply_time()
            self.resolve_profile_models()
            self.spawn_static_obstacle()
            self.spawn_dynamic_obstacle()
            self.spawn_roundabout_npcs()
            self.spawn_highway_npcs()
            self.arm_tunnel_sudden_brake()
            self.configure_drive_with_retries()
            self.persist_profile_state()
        except Exception:
            self.cleanup_spawned_actors()
            raise

        print(
            f"[FullLoop] ready profile={self.profile['name']}, "
            f"weather={self.profile['weather']}, "
            f"time={self.scenario_hour:02d}:00, "
            f"static={self.static_obstacle_model}, "
            f"dynamic={self.dynamic_obstacle_model}, "
            f"roundabout_npcs={len(self.roundabout_npcs)}, "
            f"highway_npcs_planned={self.highway_npc_planned_count}"
        )

    def load_route_definition(self):
        path = self.cfg.get(
            "route_definition_path",
            "config/full_loop_links.yaml",
        )
        with open(path, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        key = self.cfg.get("route_definition_key", "full_loop")
        route_definition = data.get(key)
        if not isinstance(route_definition, dict):
            raise RuntimeError(f"Missing route definition '{key}' in {path}")
        return route_definition

    def load_global_path(self):
        path = self.cfg["global_path"]
        points = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                fields = text.split()
                if len(fields) < 2:
                    raise ValueError(f"{path}:{line_no}: expected x y [z]")
                z = float(fields[2]) if len(fields) >= 3 else 0.0
                points.append((float(fields[0]), float(fields[1]), z))
        if len(points) < 2:
            raise RuntimeError(f"Official global path is empty: {path}")
        return points

    def prepare_full_loop_route(self):
        route_links = list(self.route_definition.get("route_links", []))
        if len(route_links) < 2:
            raise RuntimeError("full_loop route needs at least two link ids")

        missing = [
            link_id
            for link_id in route_links
            if link_id not in self.map_loader.link_set
        ]
        if missing:
            raise RuntimeError(f"Unknown full-loop link ids: {missing}")

        graph = LinkGraph(self.map_loader)
        disconnected = [
            (current, following)
            for current, following in zip(route_links[:-1], route_links[1:])
            if following not in graph.adj.get(current, [])
        ]
        if disconnected:
            raise RuntimeError(f"Disconnected full-loop links: {disconnected}")

        self.start_link = route_links[0]
        self.end_link = route_links[-1]
        self.route_links = route_links
        self.zone_allowed_links = set(route_links)
        self.route_points = self.load_global_path()
        self.route_length_m = polyline_length(self.route_points)
        self.pure_pursuit_last_s = 0.0

        first = self.route_points[0]
        second = next(
            point for point in self.route_points[1:]
            if math.hypot(point[0] - first[0], point[1] - first[1]) > 1e-3
        )
        yaw = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
        self.start_tf = self.grpc.make_transform(first[0], first[1], first[2], yaw)

        last = self.route_points[-1]
        self.goal_x, self.goal_y, self.goal_z = last[:3]
        self.goal_yaw = yaw
        self.route_waypoint_indices = self.build_route_waypoint_indices()

        closure = math.sqrt(
            (last[0] - first[0]) ** 2
            + (last[1] - first[1]) ** 2
            + (last[2] - first[2]) ** 2
        )
        if closure > float(self.cfg.get("max_loop_closure_error_m", 0.5)):
            raise RuntimeError(
                f"Official path is not a closed loop: closure={closure:.3f}m"
            )

        checkpoints = self.route_definition.get("checkpoints", [])
        for checkpoint in checkpoints:
            index = int(checkpoint["path_index"])
            if index < 0 or index >= len(self.route_points):
                raise RuntimeError(
                    f"Checkpoint {checkpoint['id']} has invalid path_index={index}"
                )

        highway_zone = self.route_definition["zones"]["highway"]
        self.highway_activation_s = self.path_s_at_index(
            int(highway_zone["activation_path_index"])
        )
        exit_checkpoint_id = str(highway_zone["ego_exit_checkpoint"])
        exit_checkpoint = next(
            checkpoint
            for checkpoint in checkpoints
            if str(checkpoint["id"]) == exit_checkpoint_id
        )
        self.highway_exit_s = self.path_s_at_index(
            int(exit_checkpoint["path_index"])
        )

        tunnel_zone = self.route_definition["zones"]["tunnel"]
        tunnel_entry_id = str(tunnel_zone["entry_checkpoint"])
        tunnel_exit_id = str(tunnel_zone["exit_checkpoint"])
        tunnel_entry = next(
            checkpoint for checkpoint in checkpoints
            if str(checkpoint["id"]) == tunnel_entry_id
        )
        tunnel_exit = next(
            checkpoint for checkpoint in checkpoints
            if str(checkpoint["id"]) == tunnel_exit_id
        )
        self.tunnel_entry_s = self.path_s_at_index(
            int(tunnel_entry["path_index"])
        )
        self.tunnel_exit_s = self.path_s_at_index(
            int(tunnel_exit["path_index"])
        )
        tunnel_route = list(
            tunnel_zone.get("npc_route_links", tunnel_zone["route_links"])
        )
        missing_tunnel_links = [
            link_id for link_id in tunnel_route
            if link_id not in self.map_loader.link_set
        ]
        if missing_tunnel_links:
            raise RuntimeError(
                f"Unknown tunnel NPC link ids: {missing_tunnel_links}"
            )
        disconnected_tunnel = [
            (current, following)
            for current, following in zip(tunnel_route[:-1], tunnel_route[1:])
            if following not in graph.adj.get(current, [])
        ]
        if disconnected_tunnel:
            raise RuntimeError(
                f"Disconnected tunnel NPC links: {disconnected_tunnel}"
            )
        print(
            f"[FullLoop] official route links={len(self.route_links)}, "
            f"points={len(self.route_points)}, length={self.route_length_m:.1f}m, "
            f"closure={closure:.6f}m"
        )
        if self.cfg.get("route_debug_enabled", False):
            self.dump_route_debug()

    def path_s_at_index(self, index):
        index = max(0, min(int(index), len(self.route_points) - 1))
        return polyline_length(self.route_points[: index + 1])

    def profile_state_path(self):
        path = self.cfg.get(
            "profile_state_path",
            ".runtime/full_loop_profile.json",
        )
        return os.path.abspath(path)

    def read_profile_state(self):
        path = self.profile_state_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state if isinstance(state, dict) else {}
        except Exception as e:
            print(f"[FullLoop] profile state ignored: {e}")
            return {}

    def validate_profiles(self, profiles):
        if len(profiles) < 2:
            raise RuntimeError("At least two full-loop profiles are required")
        keys = ("weather",)
        for index, current in enumerate(profiles):
            previous = profiles[index - 1]
            for key in keys:
                if current.get(key) == previous.get(key):
                    raise RuntimeError(
                        f"Adjacent profiles must differ for {key}: "
                        f"{previous.get('name')} -> {current.get('name')}"
                    )

    def select_profile(self):
        profiles = list(self.cfg.get("profiles", []))
        self.validate_profiles(profiles)
        previous_state = self.read_profile_state()
        previous_index = int(previous_state.get("profile_index", -1))
        profile_index = (previous_index + 1) % len(profiles)
        profile = dict(profiles[profile_index])
        print(
            f"[FullLoop] profile selected index={profile_index}, "
            f"name={profile.get('name')}, previous={previous_index}"
        )
        return profile_index, profile, previous_state

    def filter_models(self, configured, available, label, minimum_count=1):
        available = list(dict.fromkeys(str(item) for item in available if item))
        configured = list(dict.fromkeys(str(item) for item in (configured or []) if item))
        if configured:
            models = [model for model in configured if model in available]
            missing = [model for model in configured if model not in available]
            if missing:
                print(f"[FullLoop] unavailable configured {label} models: {missing}")
        else:
            models = available
        if len(models) < int(minimum_count):
            raise RuntimeError(
                f"Need at least {minimum_count} available {label} models; "
                f"got {models}"
            )
        return models

    def resolve_profile_models(self):
        obstacle_models = self.filter_models(
            self.cfg.get("static_obstacle_models"),
            self.grpc.get_available_obstacle_models(),
            "static obstacle",
        )
        pedestrian_models = self.filter_models(
            self.cfg.get("dynamic_obstacle_models"),
            self.grpc.get_available_pedestrian_models(),
            "dynamic pedestrian",
        )

        self.static_obstacle_models = list(obstacle_models)
        self.dynamic_obstacle_models = list(pedestrian_models)
        self.random.shuffle(self.static_obstacle_models)
        self.random.shuffle(self.dynamic_obstacle_models)

        self.static_obstacle_model = self.static_obstacle_models[0]
        self.dynamic_obstacle_model = self.dynamic_obstacle_models[0]

    def apply_weather(self):
        weather = str(self.profile["weather"]).upper()
        previous_weather = self.previous_profile_state.get("weather")
        if previous_weather and weather == str(previous_weather).upper():
            raise RuntimeError(
                f"Weather must differ from previous run: {previous_weather}"
            )

        for attempt in range(1, 4):
            if self.grpc.set_weather(weather):
                return
            print(f"[FullLoop] weather verification retry {attempt}/3")
            time.sleep(0.2)
        raise RuntimeError(f"Failed to apply weather={weather}")

    def select_scenario_hour(self):
        hours = [int(value) for value in self.cfg.get("scenario_times", [])]
        if not hours:
            raise RuntimeError("scenario_times must contain at least one hour")
        invalid = [hour for hour in hours if hour < 0 or hour > 23]
        if invalid:
            raise RuntimeError(f"Invalid scenario_times hours: {invalid}")
        selected = self.random.choice(hours)
        print(f"[FullLoop] scenario time selected={selected:02d}:00 from {hours}")
        return selected

    def apply_time(self):
        for attempt in range(1, 4):
            if self.grpc.set_time(
                self.scenario_hour,
                data_only=False,
                instantly=True,
            ):
                return
            print(f"[FullLoop] time verification retry {attempt}/3")
            time.sleep(0.2)
        raise RuntimeError(
            f"Failed to apply scenario time={self.scenario_hour:02d}:00"
        )

    def make_actor_scale(self, values):
        from proto.morai.common.type_pb2 import Vector3

        scale = Vector3()
        scale.x = float(values[0])
        scale.y = float(values[1])
        scale.z = float(values[2])
        return scale

    def spawn_static_obstacle(self):
        spec = self.route_definition["zones"]["static_obstacle"]
        spawn_x = float(spec["position"]["x"])
        spawn_y = float(spec["position"]["y"])
        transform = self.grpc.make_transform(
            spawn_x,
            spawn_y,
            spec["position"]["z"],
            spec["yaw_deg"],
        )

        stale_ids = self.grpc.destroy_obstacles_near(
            spawn_x,
            spawn_y,
            max_dist_m=3.0,
        )
        if stale_ids:
            time.sleep(0.2)
        remaining_ids = self.grpc.obstacle_ids_near(
            spawn_x,
            spawn_y,
            max_dist_m=3.0,
        )
        if remaining_ids:
            raise RuntimeError(
                "Failed to clear stale static obstacle at spawn: "
                f"actor_ids={remaining_ids}"
            )

        actor = None
        scale_overrides = self.cfg.get("static_obstacle_model_scales", {})
        for model in self.static_obstacle_models:
            scale_values = scale_overrides.get(
                model,
                spec.get("scale", [1.0, 1.0, 1.0]),
            )
            actor = self.grpc.spawn_obstacle(
                transform=transform,
                model_name=model,
                label=f"full_loop_static_{self.profile_index}",
                scale=self.make_actor_scale(scale_values),
            )
            if actor is not None:
                self.static_obstacle_model = model
                self.static_obstacle_scale = [float(value) for value in scale_values]
                print(
                    f"[FullLoop] static obstacle selected model={model}, "
                    f"scale={self.static_obstacle_scale}"
                )
                break
            print(
                f"[FullLoop] static obstacle model unavailable at spawn: {model}"
            )
        if actor is None:
            raise RuntimeError(
                "Failed to spawn every randomized static obstacle candidate: "
                f"{self.static_obstacle_models}"
            )

        spawned_id = str(actor.get_id()).strip()
        deadline = time.monotonic() + 1.5
        nearby_ids = []
        while time.monotonic() < deadline:
            nearby_ids = self.grpc.obstacle_ids_near(
                spawn_x,
                spawn_y,
                max_dist_m=3.0,
            )
            if spawned_id in nearby_ids:
                break
            time.sleep(0.1)
        if nearby_ids != [spawned_id]:
            for actor_id in nearby_ids:
                self.grpc.destroy_obstacle_by_id(actor_id)
            raise RuntimeError(
                "Static obstacle singleton verification failed: "
                f"expected={spawned_id}, nearby={nearby_ids}"
            )

        self.static_obstacle = actor
        self.scenario_actors.append(actor)

    def spawn_dynamic_obstacle(self):
        spec = self.route_definition["zones"]["dynamic_obstacle"]
        start = spec["start"]
        end = spec["end"]
        dx = float(end["x"]) - float(start["x"])
        dy = float(end["y"]) - float(start["y"])
        distance = math.hypot(dx, dy)
        if distance < 1e-3:
            raise RuntimeError("Dynamic obstacle path has zero length")

        self.dynamic_obstacle_start = dict(start)
        self.dynamic_obstacle_end = dict(end)
        self.dynamic_obstacle_path_length_m = distance
        self.dynamic_obstacle_yaw_deg = math.degrees(math.atan2(dy, dx))
        self.dynamic_obstacle_speed_mps = float(spec.get("speed_mps", 2.5))
        self.dynamic_obstacle_api_speed_kph = float(
            spec.get("speed_kph", self.dynamic_obstacle_speed_mps * 3.6)
        )
        self.dynamic_obstacle_trigger_distance_m = float(
            spec.get("trigger_distance_m", 25.0)
        )
        self.dynamic_obstacle_arrival_tolerance_m = float(
            spec.get("arrival_tolerance_m", 0.8)
        )
        self.dynamic_obstacle_max_crossing_sec = float(
            spec.get("max_crossing_sec", 12.0)
        )
        self.dynamic_obstacle_route_s = self.path_s_at_index(
            int(spec["route_path_index"])
        )
        self.dynamic_obstacle_model = self.dynamic_obstacle_models[0]
        self.dynamic_obstacle_phase = "waiting"
        self.dynamic_obstacle_direction_checked = False
        self.dynamic_obstacle_direction_respawned = False
        self.dynamic_obstacle_last_print_sec = -1

        print(
            f"[FullLoop] dynamic obstacle armed model={self.dynamic_obstacle_model}, "
            f"start=({start['x']:.2f},{start['y']:.2f}), "
            f"end=({end['x']:.2f},{end['y']:.2f}), "
            f"route_s={self.dynamic_obstacle_route_s:.1f}m, "
            f"trigger={self.dynamic_obstacle_trigger_distance_m:.1f}m, "
            f"speed={self.dynamic_obstacle_speed_mps:.2f}m/s"
        )

    def _spawn_active_dynamic_obstacle(self, yaw_offset_deg=0.0):
        start = self.dynamic_obstacle_start
        stale_ids = self.grpc.destroy_pedestrians_near(
            start["x"],
            start["y"],
            max_dist_m=3.0,
        )
        if stale_ids:
            time.sleep(0.2)
            remaining = self.grpc.find_spawned_pedestrian_near(
                start["x"],
                start["y"],
                max_dist_m=3.0,
            )
            if remaining is not None:
                raise RuntimeError(
                    "Failed to clear stale pedestrian at dynamic obstacle "
                    f"start: actor_id={remaining.get_id()}"
                )

        actor = None
        selected_model = None
        models = [self.dynamic_obstacle_model] + [
            model
            for model in self.dynamic_obstacle_models
            if model != self.dynamic_obstacle_model
        ]
        for model in models:
            actor = self.grpc.spawn_pedestrian(
                transform=self.grpc.make_transform(
                    start["x"],
                    start["y"],
                    start["z"],
                    self.dynamic_obstacle_yaw_deg + float(yaw_offset_deg),
                ),
                model_name=model,
                label=f"full_loop_dynamic_{self.profile_index}",
                velocity=self.dynamic_obstacle_api_speed_kph,
                active_dist=0.0,
                move_dist=self.dynamic_obstacle_path_length_m + 2.0,
                start_action=True,
            )
            if actor is not None:
                selected_model = model
                break
            print(f"[FullLoop] dynamic obstacle spawn fallback from model={model}")

        if actor is None:
            raise RuntimeError(
                "Failed to spawn every randomized dynamic obstacle candidate: "
                f"{self.dynamic_obstacle_models}"
            )

        self.dynamic_obstacle = actor
        self.dynamic_obstacle_model = selected_model
        self.scenario_actors.append(actor)
        self.dynamic_obstacle_started_at = time.time()
        self.dynamic_obstacle_phase = "crossing"
        self.dynamic_obstacle_spawn_yaw_offset_deg = float(yaw_offset_deg)
        print(
            f"[FullLoop] dynamic obstacle START model={selected_model}, "
            f"yaw={self.dynamic_obstacle_yaw_deg + yaw_offset_deg:.1f}deg"
        )

    def _destroy_dynamic_obstacle_actor(self):
        actor = self.dynamic_obstacle
        if actor is None:
            return
        try:
            actor.destroy()
        finally:
            if actor in self.scenario_actors:
                self.scenario_actors.remove(actor)
            self.dynamic_obstacle = None

    def _finish_dynamic_obstacle(self, reason):
        end = self.dynamic_obstacle_end
        actor = self.dynamic_obstacle
        if actor is not None:
            try:
                # MORAI 26.R1 SetTransform adds the map elevation again for a
                # spawned pedestrian.  Pause at the naturally reached endpoint
                # instead of teleporting the actor upward.
                actor.set_pause(True)
            except Exception as e:
                print(f"[FullLoop] dynamic obstacle final stop warning: {e}")
        self.dynamic_obstacle_phase = "cleared"
        elapsed = time.time() - float(self.dynamic_obstacle_started_at or time.time())
        min_ego = self.dynamic_obstacle_min_ego_distance_m
        min_ego_text = "unknown" if not math.isfinite(min_ego) else f"{min_ego:.2f}m"
        print(
            f"[FullLoop] dynamic obstacle DONE reason={reason}, "
            f"elapsed={elapsed:.1f}s, min_ego_distance={min_ego_text}"
        )

    def update_dynamic_obstacle(self, ego_state, current_s):
        phase = getattr(self, "dynamic_obstacle_phase", "waiting")
        if phase == "waiting":
            remaining = self.dynamic_obstacle_route_s - float(current_s)
            if -2.0 <= remaining <= self.dynamic_obstacle_trigger_distance_m:
                ego_x = float(ego_state["x"])
                ego_y = float(ego_state["y"])
                start = self.dynamic_obstacle_start
                self.dynamic_obstacle_trigger_remaining_m = remaining
                self.dynamic_obstacle_trigger_ego_distance_m = math.hypot(
                    ego_x - float(start["x"]), ego_y - float(start["y"])
                )
                print(
                    f"[FullLoop] dynamic obstacle TRIGGER "
                    f"route_remaining={remaining:.2f}m, "
                    f"ego_to_start={self.dynamic_obstacle_trigger_ego_distance_m:.2f}m, "
                    f"ego_speed={float(ego_state.get('speed', 0.0)):.2f}"
                )
                self._spawn_active_dynamic_obstacle()
            return

        if phase != "crossing" or self.dynamic_obstacle is None:
            return

        state = self.dynamic_obstacle.get_actor_state()
        if state is None:
            raise RuntimeError("Failed to read active dynamic obstacle state")
        x = float(state.transform.location.x)
        y = float(state.transform.location.y)
        start = self.dynamic_obstacle_start
        end = self.dynamic_obstacle_end
        path_dx = float(end["x"]) - float(start["x"])
        path_dy = float(end["y"]) - float(start["y"])
        along = (
            (x - float(start["x"])) * path_dx
            + (y - float(start["y"])) * path_dy
        ) / self.dynamic_obstacle_path_length_m
        lateral_error = abs(
            (x - float(start["x"])) * path_dy
            - (y - float(start["y"])) * path_dx
        ) / self.dynamic_obstacle_path_length_m
        distance_to_end = math.hypot(x - float(end["x"]), y - float(end["y"]))
        ego_distance = math.hypot(
            x - float(ego_state["x"]), y - float(ego_state["y"])
        )
        self.dynamic_obstacle_min_ego_distance_m = min(
            self.dynamic_obstacle_min_ego_distance_m, ego_distance
        )
        elapsed = time.time() - float(self.dynamic_obstacle_started_at)

        print_second = int(elapsed)
        if print_second != self.dynamic_obstacle_last_print_sec:
            self.dynamic_obstacle_last_print_sec = print_second
            print(
                f"[FullLoop] dynamic crossing t={elapsed:.1f}s "
                f"pos=({x:.2f},{y:.2f}) along={along:.2f}m "
                f"lateral={lateral_error:.2f}m to_end={distance_to_end:.2f}m "
                f"ego_distance={ego_distance:.2f}m"
            )

        displacement = math.hypot(
            x - float(start["x"]),
            y - float(start["y"]),
        )
        if elapsed >= 2.0 and displacement < 0.4:
            details = (
                "Dynamic obstacle did not leave spawn B: "
                f"model={self.dynamic_obstacle_model}, "
                f"displacement={displacement:.2f}m, elapsed={elapsed:.1f}s"
            )
            self._destroy_dynamic_obstacle_actor()
            raise RuntimeError(details)

        if not self.dynamic_obstacle_direction_checked and displacement >= 0.4:
            self.dynamic_obstacle_direction_checked = True
            if along < -0.2 or lateral_error > 1.0:
                details = (
                    "Dynamic obstacle left the required B->A crossing: "
                    f"model={self.dynamic_obstacle_model}, "
                    f"along={along:.2f}m, lateral={lateral_error:.2f}m"
                )
                self._destroy_dynamic_obstacle_actor()
                raise RuntimeError(details)

        if distance_to_end <= self.dynamic_obstacle_arrival_tolerance_m:
            self._finish_dynamic_obstacle("reached_dest")
            return
        if along >= self.dynamic_obstacle_path_length_m:
            self._finish_dynamic_obstacle("passed_dest")
            return
        if elapsed >= self.dynamic_obstacle_max_crossing_sec:
            raise RuntimeError(
                f"Dynamic obstacle crossing timeout: model={self.dynamic_obstacle_model}, "
                f"along={along:.2f}m, lateral={lateral_error:.2f}m, "
                f"to_end={distance_to_end:.2f}m"
            )

    def vehicle_models(self):
        if not hasattr(self, "_npc_vehicle_model_pool"):
            self._npc_vehicle_model_pool = self.filter_models(
                self.cfg.get("npc_vehicle_models"),
                self.grpc.get_available_surround_vehicle_models(),
                "NPC vehicle",
                minimum_count=1,
            )
        return list(self._npc_vehicle_model_pool)

    def spawn_vehicle_with_retry(self, transform, label, speed):
        models = list(self.vehicle_models())
        self.random.shuffle(models)
        for model in models:
            actor = self.grpc.spawn_vehicle(
                transform=transform,
                model_name=model,
                label=label,
                velocity=float(speed),
                multi_ego=False,
            )
            if actor is not None:
                return actor, model
        return None, None

    def rotate_loop_route(self, route_links, start_link, laps):
        index = route_links.index(start_link)
        rotated = route_links[index:] + route_links[:index]
        route = []
        for _ in range(max(1, int(laps))):
            route.extend(rotated)
        route.append(rotated[0])
        return route

    def spawn_roundabout_npcs(self):
        spec = self.route_definition["zones"]["roundabout"]
        route_links = list(spec["internal_loop_links"])
        graph = LinkGraph(self.map_loader)
        for current, following in zip(route_links, route_links[1:] + route_links[:1]):
            if following not in graph.adj.get(current, []):
                raise RuntimeError(
                    f"Disconnected roundabout cycle: {current} -> {following}"
                )

        segments = []
        total_length = 0.0
        for link_id in route_links:
            points = self.map_loader.get_link_points(link_id)
            length = polyline_length(points)
            segments.append((link_id, points, total_length, total_length + length))
            total_length += length

        count = int(self.cfg["roundabout_npc"]["count"])
        if count < 1:
            print("[FullLoop] roundabout traffic skipped")
            return
        vehicle_spacing_m = total_length / count
        phase = vehicle_spacing_m * 0.5
        route_laps = int(self.cfg["roundabout_npc"].get("route_laps", 30))
        traffic_speed = float(self.cfg["roundabout_npc"]["speed_mps"])
        if traffic_speed <= 0.0:
            raise RuntimeError("roundabout_npc speed_mps must be greater than 0")
        model = str(
            self.cfg["roundabout_npc"].get("model", "DEFAULT_SEDAN")
        )
        if model not in self.vehicle_models():
            raise RuntimeError(
                f"Configured roundabout NPC model is unavailable: {model}"
            )
        print(
            f"[FullLoop] roundabout traffic model={model}, count={count}, "
            f"spacing={vehicle_spacing_m:.1f}m, "
            f"speed={traffic_speed:.1f}m/s"
        )

        for index in range(count):
            loop_s = (phase + index * vehicle_spacing_m) % total_length
            for link_id, points, start_s, end_s in segments:
                if loop_s <= end_s:
                    local_s = loop_s - start_s
                    x, y, z, yaw = interpolate_on_polyline(points, local_s)
                    break
            speed = traffic_speed
            actor = self.grpc.spawn_vehicle(
                transform=self.grpc.make_transform(x, y, z, yaw),
                model_name=model,
                label=f"roundabout_loop_{index}",
                velocity=float(speed),
                multi_ego=False,
            )
            if actor is None:
                raise RuntimeError(f"Failed to spawn roundabout NPC {index}")
            route = self.rotate_loop_route(route_links, link_id, route_laps)
            if not self.grpc.set_vehicle_route(
                actor,
                route,
                decision_range=float(self.cfg.get("npc_decision_range_m", 30.0)),
                label=f"RoundaboutNPC{index}",
            ):
                actor.destroy()
                raise RuntimeError(f"Failed to set roundabout NPC route {index}")
            actor.set_pause(False)
            self.grpc.set_vehicle_speed_limit(actor, speed, enabled=True)
            self.grpc.set_vehicle_ai(actor, True)
            self.grpc.set_vehicle_velocity(actor, speed)
            self.roundabout_npcs.append(
                {"actor": actor, "model": model, "speed_mps": speed, "route": route}
            )
            self.scenario_actors.append(actor)


    def spawn_highway_npcs(self):
        """Create native MORAI SpawnPoints and let them settle while disabled."""
        cfg = self.cfg["highway_npc"]
        model = str(cfg.get("model", "DEFAULT_SEDAN"))
        if model not in self.vehicle_models():
            raise RuntimeError(f"Configured highway NPC model is unavailable: {model}")
        self.highway_fixed_model = model

        specs = list(cfg.get("spawn_points", []))
        if not specs:
            raise RuntimeError("No native highway SpawnPoint definitions configured")

        lifecycle = dict(cfg.get("spawn_point_lifecycle", {}))
        create_attempts = max(1, int(lifecycle.get("create_attempts", 3)))
        create_settle_sec = max(
            0.0,
            float(lifecycle.get("create_settle_sec", 0.5)),
        )
        recreate_delay_sec = max(
            0.0,
            float(lifecycle.get("recreate_delay_sec", 1.0)),
        )

        for spec in specs:
            spawn_name = spec.get("name", spec.get("id"))
            created_point = None
            for create_attempt in range(1, create_attempts + 1):
                spawn_point_id = self.grpc.create_vehicle_spawn_point(spec)
                if spawn_point_id is not None:
                    if create_settle_sec > 0.0:
                        time.sleep(create_settle_sec)
                    created_point = {
                        "id": str(spawn_point_id),
                        "name": str(spawn_name),
                        "enabled": False,
                        "retired": False,
                        "generation": 0,
                        "activation_failures": 0,
                        "spec": dict(spec),
                    }
                    self.highway_spawn_points.append(created_point)
                    break

                print(
                    f"[FullLoop] SpawnPoint create recovery: name={spawn_name}, "
                    f"attempt={create_attempt}/{create_attempts}"
                )
                if create_attempt < create_attempts and recreate_delay_sec > 0.0:
                    time.sleep(recreate_delay_sec)

            if created_point is None:
                raise RuntimeError(
                    f"Failed to create native SpawnPoint {spawn_name} "
                    f"after {create_attempts} attempts"
                )

        self.highway_spawn_points_active = False
        self.highway_npc_planned_count = len(self.highway_spawn_points)
        reference_safety = float(cfg.get("reference_safety_distance_m", 5.0))
        reference_ttc = float(cfg.get("reference_ttc_sec", 1.0))
        print(
            f"[FullLoop] native highway SpawnPoints created inactive: "
            f"count={self.highway_npc_planned_count}, lane_change=False, "
            f"reference_safety={reference_safety:.1f}m, "
            f"reference_ttc={reference_ttc:.1f}s; "
            f"activation is delayed until Ego approaches the highway"
        )

    def _route_geometry(self, route):
        points, ranges, total = [], [], 0.0
        for index, link_id in enumerate(route):
            link_points = list(self.map_loader.get_link_points(link_id))
            if len(link_points) < 2:
                continue
            length = polyline_length(link_points)
            ranges.append((total, total + length, index, link_points))
            points.extend(
                link_points[1:]
                if points and points[-1][:2] == link_points[0][:2]
                else link_points
            )
            total += length
        if len(points) < 2:
            raise RuntimeError(f"Highway route has no geometry: {route}")
        return points, ranges, total

    @staticmethod
    def _nearest_waypoint(points, x, y):
        return min(
            range(len(points)),
            key=lambda i: ((float(points[i][0])-x)**2+(float(points[i][1])-y)**2),
        )

    def _nearest_official_link(self, x, y):
        best = None
        for index, link_id in enumerate(self.route_links):
            points = self.map_loader.get_link_points(link_id)
            waypoint = self._nearest_waypoint(points, x, y)
            point = points[waypoint]
            dist = math.hypot(float(point[0])-x, float(point[1])-y)
            if best is None or dist < best[0]:
                best = (dist, index, waypoint)
        return best

    def _spawn_highway_vehicle(
        self,
        transform,
        route,
        waypoint,
        label,
        speed,
        end_waypoint=None,
        request_id="",
        recover_ambiguous=False,
    ):
        actor = self.grpc.spawn_vehicle(
            transform=transform,
            model_name=self.highway_fixed_model,
            label=label,
            velocity=0.0,
            multi_ego=False,
            request_id=request_id,
            recover_ambiguous=recover_ambiguous,
        )
        if actor is None:
            return None
        waypoint_indices = {route[0]: int(waypoint)}
        if end_waypoint is not None:
            waypoint_indices[route[-1]] = int(end_waypoint)
        ok = self.grpc.set_vehicle_route(
            actor, route,
            decision_range=float(self.cfg.get("npc_decision_range_m", 30.0)),
            route_waypoint_indices=waypoint_indices, label=label,
        )
        if not ok:
            actor.destroy()
            return None
        actor.set_pause(False)
        self.grpc.set_vehicle_speed_limit(actor, speed, enabled=True)
        self.grpc.set_vehicle_ai(actor, True)
        self.grpc.set_vehicle_velocity(actor, speed)
        self.scenario_actors.append(actor)
        return actor

    def activate_highway_spawn_points(self, elapsed, current_s):
        active_points = [
            point
            for point in self.highway_spawn_points
            if not point.get("retired", False)
        ]
        if self.highway_spawn_points_active or not active_points:
            return self.highway_spawn_points_active

        cfg = self.cfg["highway_npc"]
        lifecycle = dict(cfg.get("spawn_point_lifecycle", {}))
        warmup_distance = max(
            0.0,
            float(cfg.get("background_warmup_distance_m", 300.0)),
        )
        activation_s = max(0.0, self.highway_activation_s - warmup_distance)
        if float(current_s) < activation_s:
            return False
        if float(elapsed) < self.highway_spawn_point_next_retry_at:
            return False

        self.highway_spawn_point_activation_started = True
        changed = self._set_highway_spawn_points_enabled(True)
        pending = [
            point
            for point in self.highway_spawn_points
            if not point.get("retired", False)
            and not point.get("enabled", False)
        ]

        # MORAI 26.R1 can allocate a runtime ID (observed repeatedly as ID 5)
        # that never accepts EnableSpawnPoint. Retrying the same ID cannot heal
        # that state, so replace only the failed runtime object and preserve the
        # three healthy SpawnPoints.
        max_replacements = max(
            1,
            int(lifecycle.get("max_runtime_replacements", 3)),
        )
        create_attempts = max(1, int(lifecycle.get("create_attempts", 3)))
        create_settle_sec = max(
            0.0,
            float(lifecycle.get("create_settle_sec", 0.5)),
        )
        recreate_delay_sec = max(
            0.0,
            float(lifecycle.get("recreate_delay_sec", 1.0)),
        )
        replacements = 0
        for point in list(pending):
            point["activation_failures"] = (
                int(point.get("activation_failures", 0)) + 1
            )
            generation = int(point.get("generation", 0))
            if generation >= max_replacements:
                continue

            replacement_id = None
            for create_attempt in range(1, create_attempts + 1):
                replacement_id = self.grpc.create_vehicle_spawn_point(
                    point["spec"]
                )
                if replacement_id is not None:
                    if create_settle_sec > 0.0:
                        time.sleep(create_settle_sec)
                    break
                print(
                    f"[FullLoop] SpawnPoint replacement create retry: "
                    f"name={point['name']}, "
                    f"attempt={create_attempt}/{create_attempts}"
                )
                if create_attempt < create_attempts and recreate_delay_sec > 0.0:
                    time.sleep(recreate_delay_sec)

            if replacement_id is None:
                continue

            point["retired"] = True
            replacement = {
                "id": str(replacement_id),
                "name": str(point["name"]).split("#runtime-")[0],
                "enabled": False,
                "retired": False,
                "generation": generation + 1,
                "activation_failures": 0,
                "spec": dict(point["spec"]),
            }
            self.highway_spawn_points.append(replacement)
            replacements += 1
            print(
                f"[FullLoop] SpawnPoint runtime replacement: "
                f"name={replacement['name']}, old_id={point['id']}, "
                f"new_id={replacement['id']}, "
                f"generation={replacement['generation']}/{max_replacements}"
            )

        if replacements:
            changed += self._set_highway_spawn_points_enabled(True)

        active_points = [
            point
            for point in self.highway_spawn_points
            if not point.get("retired", False)
        ]
        pending = [
            point["name"]
            for point in active_points
            if not point.get("enabled", False)
        ]
        self.highway_spawn_points_active = bool(active_points) and not pending
        if self.highway_spawn_points_active:
            print(
                f"[FullLoop] native highway SpawnPoints ACTIVE: "
                f"count={len(active_points)}, "
                f"warmup_distance={warmup_distance:.1f}m, "
                f"runtime_replacements={replacements}"
            )
            return True

        retry_period = max(
            0.2,
            float(cfg.get("spawn_point_activation_retry_period_sec", 1.0)),
        )
        self.highway_spawn_point_next_retry_at = float(elapsed) + retry_period
        print(
            f"[FullLoop] SpawnPoint activation pending: "
            f"enabled_now={changed}, pending={pending}, "
            f"retry_in={retry_period:.1f}s"
        )
        return False

    def _set_highway_spawn_points_enabled(self, enabled):
        changed = 0
        lifecycle = self.cfg.get('highway_npc', {}).get(
            'spawn_point_lifecycle',
            {},
        )
        if enabled:
            attempts = max(1, int(lifecycle.get('enable_attempts', 5)))
            retry_delay_sec = max(
                0.0,
                float(lifecycle.get('enable_retry_delay_sec', 0.5)),
            )
        else:
            attempts = max(1, int(lifecycle.get('disable_attempts', 3)))
            retry_delay_sec = max(
                0.0,
                float(lifecycle.get('disable_retry_delay_sec', 0.5)),
            )

        for point in self.highway_spawn_points:
            if point.get("retired", False):
                continue
            if point.get('enabled') == bool(enabled):
                continue
            point_name = point['name']
            ok = self.grpc.enable_vehicle_spawn_point(
                point['id'],
                enabled,
                attempts=attempts,
                retry_delay_sec=retry_delay_sec,
            )
            if not ok:
                print(
                    f'[FullLoop] SpawnPoint state warning: '
                    f'name={point_name}, active={bool(enabled)}'
                )
                continue
            point['enabled'] = bool(enabled)
            changed += 1
        active_points = [
            point
            for point in self.highway_spawn_points
            if not point.get("retired", False)
        ]
        self.highway_spawn_points_active = bool(active_points) and all(
            point.get("enabled", False) for point in active_points
        )
        return changed

    def deactivate_highway_npcs(self, current_s):
        disabled = self._set_highway_spawn_points_enabled(False)
        # Native SpawnPoint-generated vehicles do not have Python actor
        # handles. Remove every object owned by this scenario client at the
        # highway boundary so none can enter or block the tunnel event.
        self.grpc.reset_actors()
        self.scenario_actors = []
        self.roundabout_npcs = []
        self.highway_npcs = []
        self.static_obstacle = None
        self.dynamic_obstacle = None
        self.highway_spawn_points = []
        self.highway_spawn_points_active = False
        self.highway_traffic_cleared = True
        print(
            f"[FullLoop] highway traffic CLEARED at exit "
            f"s={current_s:.1f}m, spawn_points_disabled={disabled}, "
            f"run_owned_reset=True"
        )

    def arm_tunnel_sudden_brake(self):
        cfg = dict(self.cfg.get("tunnel_sudden_brake", {}))
        if not bool(cfg.get("enabled", False)):
            self.tunnel_event_phase = "disabled"
            print("[FullLoop] tunnel sudden-brake event disabled")
            return

        model = str(
            cfg.get(
                "model",
                getattr(
                    self,
                    "highway_fixed_model",
                    self.cfg["highway_npc"].get("model", "DEFAULT_SEDAN"),
                ),
            )
        )
        if model != getattr(self, "highway_fixed_model", model):
            raise RuntimeError(
                "Tunnel NPC must use the same model as highway traffic: "
                f"tunnel={model}, highway={self.highway_fixed_model}"
            )
        if model not in self.vehicle_models():
            raise RuntimeError(f"Configured tunnel NPC model is unavailable: {model}")

        tunnel_zone = self.route_definition["zones"]["tunnel"]
        route = list(
            tunnel_zone.get("npc_route_links", tunnel_zone["route_links"])
        )
        trigger_before = float(cfg.get("spawn_trigger_before_entry_m", 60.0))
        spawn_inside = float(cfg.get("spawn_offset_inside_m", 5.0))
        brake_start_fraction = float(
            cfg.get("brake_zone_start_fraction", 0.62)
        )
        brake_end_fraction = float(
            cfg.get("brake_zone_end_fraction", 0.82)
        )
        if not 0.0 < brake_start_fraction < brake_end_fraction < 1.0:
            raise RuntimeError(
                "Tunnel brake fractions must satisfy "
                f"0 < start < end < 1: {brake_start_fraction}, "
                f"{brake_end_fraction}"
            )

        self.tunnel_npc_model = model
        self.tunnel_npc_route = route
        self.tunnel_spawn_trigger_s = max(
            0.0, self.tunnel_entry_s - trigger_before
        )
        self.tunnel_spawn_s = self.tunnel_entry_s + spawn_inside
        tunnel_length = self.tunnel_exit_s - self.tunnel_entry_s
        self.tunnel_brake_zone_start_s = (
            self.tunnel_entry_s + tunnel_length * brake_start_fraction
        )
        self.tunnel_brake_latest_s = (
            self.tunnel_entry_s + tunnel_length * brake_end_fraction
        )
        goal_margin = float(cfg.get("despawn_goal_margin_m", 30.0))
        self.tunnel_cleanup_s = min(
            self.tunnel_exit_s + float(cfg.get("despawn_after_exit_m", 100.0)),
            self.route_length_m - goal_margin,
        )
        if self.tunnel_spawn_s >= self.tunnel_exit_s:
            raise RuntimeError(
                f"Tunnel NPC spawn is outside tunnel: "
                f"spawn_s={self.tunnel_spawn_s:.1f}, exit_s={self.tunnel_exit_s:.1f}"
            )
        if self.tunnel_cleanup_s <= self.tunnel_exit_s:
            raise RuntimeError(
                f"Tunnel NPC cleanup must be after the tunnel: "
                f"cleanup_s={self.tunnel_cleanup_s:.1f}, "
                f"exit_s={self.tunnel_exit_s:.1f}"
            )
        cleanup_x, cleanup_y, _, _ = interpolate_on_polyline(
            self.route_points, self.tunnel_cleanup_s
        )
        cleanup_error, cleanup_route_index, cleanup_waypoint = (
            self._tunnel_route_start(cleanup_x, cleanup_y)
        )
        self.tunnel_npc_route = route[: cleanup_route_index + 1]
        self.tunnel_cleanup_link_id = self.tunnel_npc_route[-1]
        self.tunnel_cleanup_waypoint = cleanup_waypoint

        planned_initial_gap = self.tunnel_spawn_s - self.tunnel_spawn_trigger_s
        self.tunnel_event_phase = "armed"
        self.tunnel_event_result = {
            "model": model,
            "route_links": route,
            "tunnel_entry_s": self.tunnel_entry_s,
            "tunnel_exit_s": self.tunnel_exit_s,
            "spawn_trigger_s": self.tunnel_spawn_trigger_s,
            "planned_spawn_s": self.tunnel_spawn_s,
            "brake_zone_start_s": self.tunnel_brake_zone_start_s,
            "brake_latest_s": self.tunnel_brake_latest_s,
            "planned_cleanup_s": self.tunnel_cleanup_s,
            "cleanup_link": self.tunnel_cleanup_link_id,
            "cleanup_link_distance_m": float(cleanup_error),
            "planned_initial_gap_m": planned_initial_gap,
            "spawned": False,
            "spawn_verified": False,
            "brake_commanded": False,
            "stop_confirmed": False,
            "ego_reaction_detected": False,
            "destroyed": False,
            "resumed_after_event": False,
            "observed_links": [],
        }
        print(
            f"[FullLoop] tunnel sudden-brake ARMED model={model}, "
            f"tunnel_s={self.tunnel_entry_s:.1f}-{self.tunnel_exit_s:.1f}m, "
            f"spawn_trigger_s={self.tunnel_spawn_trigger_s:.1f}m, "
            f"spawn_s={self.tunnel_spawn_s:.1f}m, "
            f"brake_zone={self.tunnel_brake_zone_start_s:.1f}-"
            f"{self.tunnel_brake_latest_s:.1f}m, "
            f"planned_initial_gap={planned_initial_gap:.1f}m, "
            f"cleanup_s={self.tunnel_cleanup_s:.1f}m"
        )

    @staticmethod
    def _actor_speed(state):
        velocity = state.velocity
        return math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def _tunnel_route_start(self, x, y):
        best = None
        for route_index, link_id in enumerate(self.tunnel_npc_route):
            points = self.map_loader.get_link_points(link_id)
            waypoint = self._nearest_waypoint(points, x, y)
            point = points[waypoint]
            distance = math.hypot(float(point[0]) - x, float(point[1]) - y)
            if best is None or distance < best[0]:
                best = (distance, route_index, waypoint)
        return best

    def spawn_tunnel_sudden_brake_npc(self, current_s, elapsed):
        cfg = self.cfg["tunnel_sudden_brake"]
        x, y, z, yaw = interpolate_on_polyline(
            self.route_points, self.tunnel_spawn_s
        )
        nearest, route_index, waypoint = self._tunnel_route_start(x, y)
        route = list(self.tunnel_npc_route[route_index:])
        end_points = self.map_loader.get_link_points(route[-1])
        cleanup_x, cleanup_y, _, _ = interpolate_on_polyline(
            self.route_points, self.tunnel_cleanup_s
        )
        cleanup_waypoint = self._nearest_waypoint(
            end_points, cleanup_x, cleanup_y
        )
        speed = float(cfg.get("cruise_speed_mps", 12.0))

        actor = self._spawn_highway_vehicle(
            self.grpc.make_transform(x, y, z, yaw),
            route,
            waypoint,
            "tunnel_sudden_brake",
            speed,
            end_waypoint=cleanup_waypoint,
        )
        if actor is None:
            raise RuntimeError("Failed to spawn tunnel sudden-brake NPC")

        initial_gap = self.tunnel_spawn_s - float(current_s)
        minimum_spawn_gap = float(cfg.get("minimum_spawn_gap_m", 55.0))
        if initial_gap < minimum_spawn_gap:
            try:
                actor.destroy()
            finally:
                if actor in self.scenario_actors:
                    self.scenario_actors.remove(actor)
            self.tunnel_event_phase = "failed"
            raise RuntimeError(
                f"Tunnel NPC unsafe spawn gap: {initial_gap:.1f}m "
                f"< {minimum_spawn_gap:.1f}m"
            )

        self.tunnel_npc = actor
        self.tunnel_npc_last_xy = None
        self.tunnel_npc_spawn_elapsed = float(elapsed)
        self.tunnel_npc_brake_wall_time = None
        self.tunnel_npc_stopped_wall_time = None
        self.tunnel_npc_forced_stop = False
        self.tunnel_npc_approach_slowed = False
        self.tunnel_npc_resumed_wall_time = None
        self.tunnel_event_phase = "cruising"
        self.tunnel_event_result.update({
            "spawned": True,
            "spawn_elapsed_sec": float(elapsed),
            "spawn_target": {"x": x, "y": y, "z": z, "yaw_deg": yaw},
            "spawn_route_link_distance_m": float(nearest),
            "initial_route_gap_m": initial_gap,
            "minimum_route_gap_m": initial_gap,
        })
        print(
            f"[FullLoop] tunnel NPC SPAWN model={self.tunnel_npc_model}, "
            f"route={route}, target=({x:.2f},{y:.2f}), "
            f"initial_gap={initial_gap:.1f}m"
        )

    def _write_tunnel_event_report(self):
        path = self.cfg.get("tunnel_sudden_brake", {}).get(
            "verification_report_path",
            ".runtime/verification_reports/tunnel_sudden_brake_latest.json",
        )
        if not path:
            return
        path = os.path.abspath(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        payload = dict(self.tunnel_event_result)
        payload["phase"] = self.tunnel_event_phase
        payload["saved_at_unix"] = time.time()
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
        print(f"[FullLoop] tunnel event report: {path}")

    def _destroy_tunnel_npc(self, reason, current_s):
        actor = self.tunnel_npc
        destroy_ok = False
        if actor is not None:
            try:
                actor.destroy()
                destroy_ok = True
            except Exception as exc:
                print(f"[FullLoop] tunnel NPC destroy warning: {exc}")
            finally:
                if actor in self.scenario_actors:
                    self.scenario_actors.remove(actor)
        self.tunnel_npc = None
        self.tunnel_event_phase = "completed"
        self.tunnel_event_result.update({
            "destroy_reason": str(reason),
            "destroyed": destroy_ok,
            "destroy_ego_s": float(current_s),
            "event_success": bool(
                self.tunnel_event_result.get("spawn_verified")
                and self.tunnel_event_result.get("brake_commanded")
                and self.tunnel_event_result.get("stop_confirmed")
                and self.tunnel_event_result.get("ego_reaction_detected")
                and self.tunnel_event_result.get("resumed_after_event")
                and reason == "post_tunnel_cleanup"
                and destroy_ok
            ),
        })
        min_gap = self.tunnel_event_result.get("minimum_route_gap_m")
        print(
            f"[FullLoop] tunnel NPC REMOVED reason={reason}, "
            f"event_success={self.tunnel_event_result['event_success']}, "
            f"ego_reaction={self.tunnel_event_result.get('ego_reaction_detected')}, "
            f"min_route_gap={min_gap}"
        )
        self._write_tunnel_event_report()

    def _command_tunnel_hard_brake(
        self, elapsed, ego_state, npc_s, route_gap, x, y, link_id, speed, reason
    ):
        cfg = self.cfg["tunnel_sudden_brake"]
        minimum_brake_gap = float(cfg.get("minimum_brake_gap_m", 20.0))
        if route_gap < minimum_brake_gap:
            self.tunnel_event_result["brake_gap_below_preferred"] = True
            print(
                f"[FullLoop] tunnel brake gap warning: "
                f"{route_gap:.1f}m < preferred {minimum_brake_gap:.1f}m"
            )
        limit_ok = self.grpc.set_vehicle_speed_limit(
            self.tunnel_npc, 0.0, enabled=True
        )
        brake_ok = self.grpc.set_vehicle_speed(self.tunnel_npc, 0.0)
        self.tunnel_npc_brake_wall_time = time.time()
        self.tunnel_event_phase = "braking"
        ego_speed = float(ego_state.get("speed", 0.0))
        self.tunnel_event_result.update({
            "brake_commanded": bool(limit_ok or brake_ok),
            "brake_trigger_reason": str(reason),
            "brake_command_elapsed_sec": float(elapsed),
            "brake_position": {"x": x, "y": y, "link": link_id},
            "brake_npc_s": npc_s,
            "brake_inside_tunnel_m": npc_s - self.tunnel_entry_s,
            "brake_route_gap_m": route_gap,
            "brake_speed": speed,
            "brake_ego_speed": ego_speed,
            "post_brake_peak_ego_speed": ego_speed,
        })
        print(
            f"[FullLoop] tunnel NPC HARD BRAKE reason={reason}, "
            f"inside={npc_s-self.tunnel_entry_s:.1f}m, "
            f"route_gap={route_gap:.1f}m, speed_before={speed:.1f}, "
            f"command_ok={bool(limit_ok or brake_ok)}"
        )

    def _resume_tunnel_npc_after_event(self, elapsed, npc_s, route_gap, reason):
        cfg = self.cfg["tunnel_sudden_brake"]
        resume_speed = float(cfg.get("resume_speed_mps", 10.0))
        pause_ok = self.tunnel_npc.set_pause(False)
        limit_ok = self.grpc.set_vehicle_speed_limit(
            self.tunnel_npc, resume_speed, enabled=True
        )
        speed_ok = self.grpc.set_vehicle_speed(
            self.tunnel_npc, resume_speed
        )
        velocity_ok = self.grpc.set_vehicle_velocity(
            self.tunnel_npc, resume_speed
        )
        resumed = bool(pause_ok or limit_ok or speed_ok or velocity_ok)
        self.tunnel_npc_resumed_wall_time = time.time()
        self.tunnel_event_phase = "departing"
        self.tunnel_event_result.update({
            "resumed_after_event": resumed,
            "resume_reason": str(reason),
            "resume_elapsed_sec": float(elapsed),
            "resume_npc_s": float(npc_s),
            "resume_route_gap_m": float(route_gap),
            "resume_speed_mps": resume_speed,
        })
        print(
            f"[FullLoop] tunnel NPC RESUME reason={reason}, "
            f"route_gap={route_gap:.1f}m, target_speed={resume_speed:.1f}, "
            f"command_ok={resumed}"
        )

    def update_tunnel_sudden_brake(self, elapsed, ego_state, current_s):
        phase = getattr(self, "tunnel_event_phase", "disabled")
        if phase in ("disabled", "completed", "failed", "uninitialized"):
            return
        if phase == "armed":
            if float(current_s) >= self.tunnel_spawn_trigger_s:
                self.spawn_tunnel_sudden_brake_npc(current_s, elapsed)
            return
        if self.tunnel_npc is None:
            return

        state = self.tunnel_npc.get_actor_state()
        if state is None:
            self.tunnel_event_result["actor_state_unavailable"] = True
            self._destroy_tunnel_npc("actor_state_unavailable", current_s)
            return

        x = float(state.transform.location.x)
        y = float(state.transform.location.y)
        npc_s = project_distance_on_polyline(self.route_points, x, y)
        route_gap = npc_s - float(current_s)
        euclidean_gap = math.hypot(
            x - float(ego_state["x"]), y - float(ego_state["y"])
        )
        speed = self._actor_speed(state)
        travelled = max(0.0, npc_s - self.tunnel_spawn_s)

        try:
            link_id = state.vehicle_state.current_link_info.id.value
        except Exception:
            link_id = ""
        observed_links = self.tunnel_event_result.setdefault("observed_links", [])
        if link_id and link_id not in observed_links:
            observed_links.append(link_id)

        if not self.tunnel_event_result.get("spawn_verified"):
            target = self.tunnel_event_result["spawn_target"]
            spawn_error = math.hypot(x - target["x"], y - target["y"])
            tolerance = float(
                self.cfg["tunnel_sudden_brake"].get(
                    "spawn_position_tolerance_m", 3.0
                )
            )
            self.tunnel_event_result["actual_spawn"] = {
                "x": x, "y": y, "link": link_id,
            }
            self.tunnel_event_result["spawn_error_m"] = spawn_error
            self.tunnel_event_result["spawn_verified"] = bool(
                spawn_error <= tolerance
                and (not link_id or link_id in self.tunnel_npc_route)
            )
            print(
                f"[FullLoop] tunnel NPC SPAWN VERIFY "
                f"error={spawn_error:.2f}m, link={link_id}, "
                f"verified={self.tunnel_event_result['spawn_verified']}"
            )
            if not self.tunnel_event_result["spawn_verified"]:
                self._destroy_tunnel_npc("spawn_verification_failed", current_s)
                raise RuntimeError(
                    f"Tunnel NPC spawn verification failed: "
                    f"error={spawn_error:.2f}m, link={link_id}"
                )

        if route_gap > 0.0:
            previous_min = float(
                self.tunnel_event_result.get("minimum_route_gap_m", route_gap)
            )
            self.tunnel_event_result["minimum_route_gap_m"] = min(
                previous_min, route_gap
            )
        self.tunnel_event_result.update({
            "last_position": {"x": x, "y": y, "link": link_id},
            "last_npc_s": npc_s,
            "last_route_gap_m": route_gap,
            "last_euclidean_gap_m": euclidean_gap,
            "last_speed": speed,
            "travelled_m": travelled,
        })

        second = int(float(elapsed) - float(self.tunnel_npc_spawn_elapsed))
        if second != self.tunnel_event_last_print_sec:
            self.tunnel_event_last_print_sec = second
            print(
                f"[FullLoop] tunnel event phase={phase}, "
                f"npc_s={npc_s:.1f}m, inside={npc_s-self.tunnel_entry_s:.1f}m, "
                f"travelled={travelled:.1f}m, route_gap={route_gap:.1f}m, "
                f"ego_speed={float(ego_state.get('speed', 0.0)):.1f}, "
                f"npc_speed={speed:.1f}"
            )

        cfg = self.cfg["tunnel_sudden_brake"]
        if phase == "departing":
            if npc_s >= self.tunnel_cleanup_s - 2.0:
                self.tunnel_event_result.update({
                    "actual_cleanup_s": float(npc_s),
                    "cleanup_after_tunnel_m": float(
                        npc_s - self.tunnel_exit_s
                    ),
                    "cleanup_position": {
                        "x": x, "y": y, "link": link_id,
                    },
                })
                self._destroy_tunnel_npc(
                    "post_tunnel_cleanup", current_s
                )
            return

        if phase in ("cruising", "closing_gap"):
            trigger_gap = float(cfg.get("brake_trigger_gap_m", 35.0))
            inside_brake_zone = npc_s >= self.tunnel_brake_zone_start_s
            reached_latest = npc_s >= self.tunnel_brake_latest_s
            gap_ready = 0.0 < route_gap <= trigger_gap
            if inside_brake_zone and (gap_ready or reached_latest):
                reason = (
                    "ego_gap_ready" if gap_ready
                    else "latest_tunnel_position"
                )
                self._command_tunnel_hard_brake(
                    elapsed, ego_state, npc_s, route_gap,
                    x, y, link_id, speed, reason,
                )
                return

            if inside_brake_zone and phase == "cruising":
                approach_speed = float(
                    cfg.get("approach_speed_mps", 7.0)
                )
                limit_ok = self.grpc.set_vehicle_speed_limit(
                    self.tunnel_npc, approach_speed, enabled=True
                )
                velocity_ok = self.grpc.set_vehicle_velocity(
                    self.tunnel_npc, approach_speed
                )
                self.tunnel_npc_approach_slowed = True
                self.tunnel_event_phase = "closing_gap"
                self.tunnel_event_result.update({
                    "closing_gap_started": True,
                    "closing_gap_npc_s": float(npc_s),
                    "closing_gap_route_gap_m": float(route_gap),
                    "closing_gap_speed_mps": approach_speed,
                    "closing_gap_command_ok": bool(limit_ok or velocity_ok),
                })
                print(
                    f"[FullLoop] tunnel NPC CLOSING GAP "
                    f"inside={npc_s-self.tunnel_entry_s:.1f}m, "
                    f"route_gap={route_gap:.1f}m, "
                    f"target_speed={approach_speed:.1f}"
                )
            return

        if phase != "braking":
            return

        ego_speed = float(ego_state.get("speed", 0.0))
        peak_speed = max(
            float(
                self.tunnel_event_result.get(
                    "post_brake_peak_ego_speed", ego_speed
                )
            ),
            ego_speed,
        )
        self.tunnel_event_result["post_brake_peak_ego_speed"] = peak_speed
        speed_drop = peak_speed - ego_speed
        reaction_gap = float(cfg.get("reaction_max_gap_m", 40.0))
        if (
            0.0 < route_gap <= reaction_gap
            and speed_drop >= float(cfg.get("ego_reaction_speed_drop", 5.0))
            and not self.tunnel_event_result.get("ego_reaction_detected")
        ):
            self.tunnel_event_result.update({
                "ego_reaction_detected": True,
                "ego_reaction_elapsed_sec": float(elapsed),
                "ego_speed_drop": speed_drop,
                "ego_reaction_route_gap_m": float(route_gap),
            })
            print(
                f"[FullLoop] tunnel Ego reaction detected "
                f"speed_drop={speed_drop:.1f}, route_gap={route_gap:.1f}m"
            )

        now = time.time()
        if (
            not self.tunnel_event_result.get("stop_confirmed")
            and speed <= float(cfg.get("stopped_speed_threshold_mps", 0.5))
        ):
            self.tunnel_event_result["stop_confirmed"] = True
            self.tunnel_event_result["stopped_position"] = {
                "x": x, "y": y, "link": link_id,
            }
            self.tunnel_event_result["stopped_npc_s"] = npc_s
            self.tunnel_event_result["stopped_inside_tunnel_m"] = (
                npc_s - self.tunnel_entry_s
            )
            self.tunnel_event_result["stopped_route_gap_m"] = route_gap
            self.tunnel_npc_stopped_wall_time = now
            print(
                f"[FullLoop] tunnel NPC STOP CONFIRMED "
                f"inside={npc_s-self.tunnel_entry_s:.1f}m, "
                f"route_gap={route_gap:.1f}m"
            )

        if (
            not self.tunnel_event_result.get("stop_confirmed")
            and now - float(self.tunnel_npc_brake_wall_time)
                >= float(cfg.get("stop_confirm_timeout_sec", 1.0))
            and not self.tunnel_npc_forced_stop
        ):
            self.tunnel_npc_forced_stop = True
            self.tunnel_event_result["forced_stop"] = True
            self.grpc.stop_vehicle(self.tunnel_npc)
            print("[FullLoop] tunnel NPC forced-stop fallback requested")

        emergency_gap = float(cfg.get("emergency_destroy_gap_m", 8.0))
        if (
            route_gap <= emergency_gap
            and not self.tunnel_event_result.get("ego_reaction_detected")
        ):
            self.tunnel_event_result["emergency_resume"] = True
            self._resume_tunnel_npc_after_event(
                elapsed, npc_s, route_gap, "ego_safety_gap_guard"
            )
            return

        if (
            self.tunnel_event_result.get("stop_confirmed")
            and self.tunnel_event_result.get("ego_reaction_detected")
            and self.tunnel_npc_stopped_wall_time is not None
            and now - self.tunnel_npc_stopped_wall_time
                >= float(cfg.get("stop_hold_sec", 3.0))
        ):
            self._resume_tunnel_npc_after_event(
                elapsed, npc_s, route_gap, "ego_reaction_complete"
            )

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
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
        self.update_dynamic_obstacle(ego_state,current_s)
        self.activate_highway_spawn_points(elapsed, current_s)
        if (
            self.highway_spawn_point_activation_started
            and not self.highway_traffic_cleared
            and current_s >= self.highway_exit_s
        ):
            self.deactivate_highway_npcs(current_s)
        self.update_tunnel_sudden_brake(elapsed, ego_state, current_s)

    def persist_profile_state(self):
        state = {
            "profile_index": self.profile_index,
            "profile_name": self.profile["name"],
            "weather": str(self.profile["weather"]).upper(),
            "time_hour": self.scenario_hour,
            "static_obstacle_model": self.static_obstacle_model,
            "static_obstacle_scale": self.static_obstacle_scale,
            "dynamic_obstacle_model": self.dynamic_obstacle_model,
            "run_seed": self.run_seed,
            "saved_at_unix": time.time(),
        }
        previous = self.previous_profile_state
        for key in ("weather",):
            if previous.get(key) == state[key]:
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
            self.grpc.reset_actors(settle_sec=cleanup_settle_sec)
        except Exception as e:
            print(f'[FullLoop] run-owned SpawnPoint cleanup warning: {e}')
        self.highway_spawn_points = []
        self.highway_spawn_points_active = False

    def cleanup(self):
        super().cleanup()
        self.cleanup_spawned_actors()

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
