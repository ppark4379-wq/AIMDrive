"""Highway native-SpawnPoint functions for the full-loop scenario."""

import math
import time

from ...utils.transform_utils import polyline_length


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
