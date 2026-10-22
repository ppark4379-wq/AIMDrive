"""Route, profile, weather, and time functions for the full-loop scenario.

Each function receives the scenario instance as one context object. Function
bodies are kept equivalent to the pre-refactor FullLoopScenario methods.
"""

import json
import math
import os
import time

import yaml

from ...utils.link_graph import LinkGraph
from ...utils.transform_utils import polyline_length


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
    if self.cfg.get('ve_stop_before_tunnel', False):
        from .before_tunnel import apply
        apply(self)
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
    forced_weather = self.cfg.get("forced_weather")
    if forced_weather is not None:
        forced_weather = str(forced_weather).upper()
        matches = [
            index for index, profile in enumerate(profiles)
            if str(profile.get("weather", "")).upper() == forced_weather
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "forced_weather must match exactly one profile: {}".format(
                    forced_weather
                )
            )
        profile_index = matches[0]
    elif previous_index < 0:
        profile_index = self.random.randrange(len(profiles))
    else:
        previous_weather = str(previous_state.get("weather", "")).upper()
        candidates = [
            index
            for index, profile in enumerate(profiles)
            if str(profile.get("weather", "")).upper() != previous_weather
        ]
        if not candidates:
            raise RuntimeError("No weather profile differs from the previous run")
        profile_index = self.random.choice(candidates)
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

    # Keep the existing static-model rotation. Dynamic scenes intentionally do
    # not read the previous scene: every pedestrian/model/path combination is
    # selected only from this run's seed and may repeat on consecutive runs.
    previous_static = self.previous_profile_state.get("static_obstacle_model")
    if previous_static in self.static_obstacle_models and len(
        self.static_obstacle_models
    ) > 1:
        self.static_obstacle_models.append(
            self.static_obstacle_models.pop(
                self.static_obstacle_models.index(previous_static)
            )
        )

    self.static_obstacle_model = self.static_obstacle_models[0]
    self.dynamic_obstacle_model = self.dynamic_obstacle_models[0]

def apply_weather(self):
    weather = str(self.profile["weather"]).upper()
    previous_weather = self.previous_profile_state.get("weather")
    if (
        self.cfg.get("forced_weather") is None
        and previous_weather
        and weather == str(previous_weather).upper()
    ):
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
    forced_hour = self.cfg.get("forced_time_hour")
    if forced_hour is not None:
        forced_hour = int(forced_hour)
        if forced_hour not in hours:
            raise RuntimeError(
                "forced_time_hour={} is not in scenario_times {}".format(
                    forced_hour, hours
                )
            )
        print(
            f"[FullLoop] scenario time forced={forced_hour:02d}:00 "
            f"from collection plan"
        )
        return forced_hour
    previous_hour = self.previous_profile_state.get("time_hour")
    candidates = [hour for hour in hours if hour != previous_hour]
    if not candidates:
        raise RuntimeError("No scenario time differs from the previous run")
    selected = self.random.choice(candidates)
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
