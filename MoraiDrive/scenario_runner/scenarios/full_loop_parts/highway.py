"""Highway native-SpawnPoint functions for the full-loop scenario."""

import math
import random
import time

from ...utils.transform_utils import interpolate_on_polyline, polyline_length
from . import actor_lifecycle


_MERGE_STATIC_MODEL_SEED_XOR = 0x48574D45524745


def _merge_static_vehicle_model_candidates(self):
    """Return one reproducibly shuffled, fully available passenger-car pool."""
    cfg = dict(self.cfg.get("highway_merge_static_vehicle", {}))
    models = [str(value).strip() for value in cfg.get("models", [])]
    if not models or any(not value for value in models):
        raise RuntimeError(
            "Highway merge static vehicle requires a non-empty model pool"
        )
    if len(models) != len(set(models)):
        raise RuntimeError(
            "Highway merge static vehicle model pool contains duplicates"
        )

    available = set(self.vehicle_models())
    missing = [model for model in models if model not in available]
    if missing:
        raise RuntimeError(
            "Configured highway merge static vehicle models are unavailable: "
            + ", ".join(missing)
        )

    model_random = random.Random(
        int(self.run_seed) ^ _MERGE_STATIC_MODEL_SEED_XOR
    )
    candidates = list(models)
    model_random.shuffle(candidates)
    return candidates


def _merge_static_vehicle_pose(self):
    """Validate and resolve the configured merge-ramp stationary pose."""
    cfg = dict(self.cfg.get("highway_merge_static_vehicle", {}))
    if not bool(cfg.get("enabled", False)):
        return None

    model_candidates = _merge_static_vehicle_model_candidates(self)

    link_id = str(cfg.get("route_link", "")).strip()
    if link_id not in self.route_links:
        raise RuntimeError(
            f"Highway merge static vehicle link is outside this route: {link_id}"
        )
    link = self.map_loader.get_link(link_id)
    if not bool(link.get("can_move_left_lane")) or not link.get(
        "left_lane_change_dst_link_idx"
    ):
        raise RuntimeError(
            "Highway merge static vehicle must be placed only after a legal "
            f"left lane change becomes available: {link_id}"
        )

    points = list(link.get("points", []))
    link_length = polyline_length(points)
    offset = float(cfg.get("distance_from_link_start_m", -1.0))
    minimum_start = max(
        0.0, float(cfg.get("minimum_distance_from_link_start_m", 0.0))
    )
    minimum_end = max(
        0.0, float(cfg.get("minimum_distance_to_link_end_m", 0.0))
    )
    if offset < minimum_start or link_length - offset < minimum_end:
        raise RuntimeError(
            "Unsafe highway merge static vehicle offset: "
            f"offset={offset:.3f}m, link_length={link_length:.3f}m, "
            f"minimum_start={minimum_start:.3f}m, minimum_end={minimum_end:.3f}m"
        )

    x, y, z, yaw = interpolate_on_polyline(points, offset)
    return {
        "model_candidates": model_candidates,
        "route_link": link_id,
        "distance_from_link_start_m": offset,
        "left_lane_change_dst_link": str(
            link["left_lane_change_dst_link_idx"]
        ),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw_deg": float(yaw),
        "waypoint": int(self._nearest_waypoint(points, x, y)),
    }


def spawn_highway_merge_static_vehicle(self):
    """Spawn one stopped sedan when native highway SpawnPoints are active."""
    existing = getattr(self, "highway_merge_static_vehicle", None)
    if existing is not None:
        return existing
    if not getattr(self, "highway_spawn_points_active", False):
        raise RuntimeError(
            "Highway merge static vehicle cannot spawn before native "
            "SpawnPoint activation"
        )

    pose = _merge_static_vehicle_pose(self)
    if pose is None:
        self.highway_merge_static_vehicle = None
        self.highway_merge_static_vehicle_info = None
        print("[FullLoop] highway merge static vehicle OFF")
        return None

    label = "highway_merge_static_vehicle"
    actor = None
    selected_model = None
    spawn_attempts = []
    transform = self.grpc.make_transform(
        pose["x"], pose["y"], pose["z"], pose["yaw_deg"]
    )
    for attempt, model in enumerate(pose["model_candidates"], 1):
        actor = self.grpc.spawn_vehicle(
            transform=transform,
            model_name=model,
            label=label,
            velocity=0.0,
            multi_ego=False,
            request_id=f"{label}_{self.run_instance_id}_{attempt}",
            recover_ambiguous=True,
        )
        spawn_attempts.append(
            {"attempt": attempt, "model": model, "spawned": actor is not None}
        )
        if actor is not None:
            selected_model = model
            break
        print(
            "[FullLoop] highway merge static vehicle model retry: "
            f"failed={model}, next_attempt={attempt + 1}"
        )
    if actor is None:
        raise RuntimeError(
            "Failed to spawn highway merge static vehicle with every "
            "configured model"
        )

    speed_limit_ok = self.grpc.set_vehicle_speed_limit(
        actor, 0.0, enabled=True
    )
    velocity_ok = self.grpc.set_vehicle_velocity(actor, 0.0)
    route_start = self.route_links.index(pose["route_link"])
    actor_route = list(self.route_links[route_start:])
    route_ok = self.grpc.set_vehicle_route(
        actor,
        actor_route,
        decision_range=float(self.cfg.get("npc_decision_range_m", 30.0)),
        route_waypoint_indices={pose["route_link"]: pose["waypoint"]},
        label=label,
    )
    if not (speed_limit_ok and velocity_ok and route_ok):
        actor.destroy()
        raise RuntimeError(
            "Failed to establish highway merge stationary contract: "
            f"velocity_zero={velocity_ok}, speed_limit_zero={speed_limit_ok}, "
            f"route={route_ok}"
        )

    self.highway_merge_static_vehicle = actor
    self.highway_merge_static_vehicle_info = dict(
        pose,
        model=selected_model,
        spawn_attempts=spawn_attempts,
        actor_id=str(actor.get_id()),
        stationary_contract={
            "default_ai_unchanged": True,
            "velocity_mps": 0.0,
            "speed_limit_mps": 0.0,
            "physics_enabled": True,
        },
        actor_route=actor_route,
    )
    self.scenario_actors.append(actor)
    print(
        "[FullLoop] highway merge static vehicle ready: "
        f"model={selected_model}, link={pose['route_link']}, "
        f"offset={pose['distance_from_link_start_m']:.1f}m, "
        f"position=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}), "
        f"yaw={pose['yaw_deg']:.2f}deg"
    )
    return actor


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

    total_spawn_capacity = sum(
        int(spec.get("maximum_spawn_vehicle", 0)) for spec in specs
    )
    maximum_total = int(
        cfg.get("collection_maximum_total_vehicles", len(specs))
    )
    if total_spawn_capacity <= 0 or total_spawn_capacity > maximum_total:
        raise RuntimeError(
            "Highway SpawnPoint population exceeds the scenario cap: "
            f"capacity={total_spawn_capacity}, cap={maximum_total}"
        )

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
        # MORAI may begin spawning immediately when pause is false. Create in
        # a paused state and require an explicit disabled acknowledgement.
        spec = dict(spec, pause=True)
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
                if not self.grpc.enable_vehicle_spawn_point(
                    spawn_point_id,
                    False,
                    attempts=int(lifecycle.get("disable_attempts", 3)),
                    retry_delay_sec=float(
                        lifecycle.get("disable_retry_delay_sec", 0.5)
                    ),
                ):
                    raise RuntimeError(
                        "Cannot confirm initial SpawnPoint disable: "
                        + str(spawn_point_id)
                    )
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
        f"count={self.highway_npc_planned_count}, "
        f"total_vehicle_cap={total_spawn_capacity}, lane_change=False, "
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
def activate_highway_spawn_points(self, elapsed, current_s):
    if getattr(self, "nearby_highway_stopped", False):
        return False
    active_points = [
        point
        for point in self.highway_spawn_points
        if not point.get("retired", False)
    ]
    if not active_points:
        return False
    if self.highway_spawn_points_active:
        self.spawn_highway_merge_static_vehicle()
        return True

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
        self.spawn_highway_merge_static_vehicle()
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

def update_highway_actor_lifecycle(self, elapsed, ego_state, current_s):
    if (
        getattr(self, "nearby_highway_stopped", False)
        or self.highway_traffic_cleared
        or not self.highway_spawn_points_active
    ):
        return

    lifecycle = actor_lifecycle.policy(self)
    highway_zone = self.route_definition["zones"]["highway"]
    entry_s = actor_lifecycle.checkpoint_s(
        self, highway_zone["ego_entry_checkpoint"]
    )
    stop_s = entry_s + float(lifecycle["highway_stop_after_entry_m"])
    if float(current_s) <= stop_s:
        return

    points = [
        point
        for point in self.highway_spawn_points
        if not point.get("retired", False)
    ]
    distances = [
        math.hypot(
            float(point["spec"]["position"]["x"]) - float(ego_state["x"]),
            float(point["spec"]["position"]["y"]) - float(ego_state["y"]),
        )
        for point in points
    ]
    minimum_distance = float(lifecycle["highway_spawn_distance_m"])
    if not distances or min(distances) <= minimum_distance:
        return

    def stop_spawning():
        self._set_highway_spawn_points_enabled(False)
        if any(point["enabled"] for point in points):
            raise RuntimeError("Cannot confirm distant highway spawner disable")

    actor_lifecycle.mutate(
        self,
        "highway_stop_new_spawns",
        current_s,
        stop_spawning,
    )
    self.nearby_highway_stopped = True


def deactivate_highway_npcs(self, current_s):
    disabled = self._set_highway_spawn_points_enabled(False)
    # Native SpawnPoint-generated vehicles do not have Python actor
    # handles. Remove every object owned by this scenario client at the
    # highway boundary so none can enter or block the tunnel event.
    self.grpc.reset_actors()
    self.scenario_actors = []
    self.roundabout_npcs = []
    self.highway_npcs = []
    self.highway_merge_static_vehicle = None
    self.highway_merge_static_vehicle_info = None
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
