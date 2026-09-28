"""Static-obstacle creation and exact run-owned state verification."""

import math
import time
import copy
from pathlib import Path

import yaml

from . import actor_lifecycle


def make_actor_scale(self, values):
    from proto.morai.common.type_pb2 import Vector3

    scale = Vector3()
    scale.x = float(values[0])
    scale.y = float(values[1])
    scale.z = float(values[2])
    return scale


def read_static_obstacle_state(self, actor_id, attempts=6, retry_seconds=0.05):
    """Read one exact obstacle while tolerating a brief empty inventory reply."""
    observations = []
    matches = []
    for attempt in range(1, max(1, int(attempts)) + 1):
        states = list(self.grpc.get_all_obstacle_states())
        matches = [
            state
            for state in states
            if str(state.actor_info.id.value).strip() == str(actor_id).strip()
        ]
        observations.append(
            {
                "attempt": attempt,
                "inventory_size": len(states),
                "exact_matches": len(matches),
            }
        )
        if len(matches) == 1:
            return matches[0], observations
        if attempt < attempts:
            time.sleep(max(0.0, float(retry_seconds)))
    return None, observations


def lock_static_obstacle_transform(
    self,
    actor,
    actor_id,
    transform,
    expected_xyz,
    attempts=8,
    stable_samples=2,
    retry_seconds=0.1,
):
    """Freeze an obstacle at its requested pose and prove that it stays there.

    MORAI can advance physics between SpawnObstacle and SetPhysics.  Some
    obstacle models therefore reach the inventory after they have already
    fallen below the road.  Disabling physics alone preserves that bad pose,
    so every repair also reapplies the requested transform.
    """
    expected_x, expected_y, expected_z = map(float, expected_xyz)
    diagnostics = []
    stable = 0
    repair_required = True
    last_state = None

    for attempt in range(1, max(1, int(attempts)) + 1):
        repair = None
        if repair_required:
            physics_ok = bool(actor.set_physics(False))
            transform_ok = bool(actor.set_transform(transform))
            repair = {
                "physics_disabled": physics_ok,
                "transform_reapplied": transform_ok,
            }
            if not physics_ok or not transform_ok:
                diagnostics.append({
                    "attempt": attempt,
                    "repair": repair,
                    "state": "repair_failed",
                })
                stable = 0
                if attempt < attempts:
                    time.sleep(max(0.0, float(retry_seconds)))
                continue

        state, inventory = read_static_obstacle_state(
            self, actor_id, attempts=1, retry_seconds=0.0,
        )
        row = {
            "attempt": attempt,
            "repair": repair,
            "inventory": inventory,
        }
        if state is None:
            row["state"] = "missing"
            diagnostics.append(row)
            stable = 0
            repair_required = True
        else:
            location = state.transform.location
            xy_error = math.hypot(
                float(location.x) - expected_x,
                float(location.y) - expected_y,
            )
            z_error = abs(float(location.z) - expected_z)
            row.update({
                "state": "stable" if xy_error <= 1.0 and z_error <= 1.0 else "moved",
                "observed_position": {
                    "x": float(location.x),
                    "y": float(location.y),
                    "z": float(location.z),
                },
                "xy_error_m": float(xy_error),
                "z_error_m": float(z_error),
            })
            diagnostics.append(row)
            if xy_error <= 1.0 and z_error <= 1.0:
                stable += 1
                last_state = state
                repair_required = False
                if stable >= max(1, int(stable_samples)):
                    return last_state, diagnostics
            else:
                stable = 0
                repair_required = True

        if attempt < attempts:
            time.sleep(max(0.0, float(retry_seconds)))

    return None, diagnostics


def select_static_obstacle_placements(self, spec, count):
    if count == 1:
        return [copy.deepcopy(spec)]

    placements = copy.deepcopy(list(spec.get("multi_positions", [])))
    if len(placements) < count:
        raise RuntimeError(
            "Static obstacle multi_positions does not cover requested "
            f"count={count}"
        )
    placements = placements[:count]
    policy = dict(self.cfg.get("static_obstacle_multi_randomization", {}))
    if not bool(policy.get("enabled", True)):
        return placements

    longitudinal = self.random.uniform(
        -float(policy.get("longitudinal_margin_m", 2.0)),
        float(policy.get("longitudinal_margin_m", 2.0)),
    )
    lateral = self.random.uniform(
        -float(policy.get("lateral_margin_m", 0.35)),
        float(policy.get("lateral_margin_m", 0.35)),
    )
    spacing_jitter = self.random.uniform(
        -float(policy.get("spacing_jitter_m", 1.0)),
        float(policy.get("spacing_jitter_m", 1.0)),
    )
    yaw_delta = self.random.uniform(
        -float(policy.get("yaw_margin_deg", 3.0)),
        float(policy.get("yaw_margin_deg", 3.0)),
    )
    center = 0.5 * (count - 1)
    for index, placement in enumerate(placements):
        signed_spacing = (float(index) - center) * spacing_jitter
        delta_y = longitudinal + signed_spacing
        placement["position"]["x"] = (
            float(placement["position"]["x"]) + lateral
        )
        placement["position"]["y"] = (
            float(placement["position"]["y"]) + delta_y
        )
        placement["yaw_deg"] = (
            float(placement.get("yaw_deg", spec["yaw_deg"])) + yaw_delta
        )
        # The official path is sampled at approximately 0.5 m intervals.
        placement["route_path_index"] = int(
            placement["route_path_index"] + round(delta_y / 0.5)
        )
        placement["random_offset"] = {
            "longitudinal_m": delta_y,
            "lateral_m": lateral,
            "yaw_deg": yaw_delta,
        }
    print(
        "[FullLoop] static pair randomization "
        f"longitudinal={longitudinal:+.2f}m, lateral={lateral:+.2f}m, "
        f"spacing_jitter={spacing_jitter:+.2f}m, yaw={yaw_delta:+.2f}deg"
    )
    return placements


def load_static_obstacle_preset(path, expected_map_name, route_points,
                                route_index_offset=0):
    """Validate an editor preset and bind every obstacle to this route.

    ``route_points`` may be a mission window of the full route; the nearest
    point is searched only there, and ``route_index_offset`` (the window's
    first full-route index) makes ``route_path_index`` a full-route index as
    ``path_s_at_index`` expects.
    """
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict):
        raise ValueError("Static obstacle preset must be a YAML mapping")
    if document.get("schema") != "morai_static_obstacle_preset_v1":
        raise ValueError("Unsupported static obstacle preset schema")
    if str(document.get("map_name")) != str(expected_map_name):
        raise ValueError(
            "Static obstacle preset map mismatch: {} != {}".format(
                document.get("map_name"), expected_map_name
            )
        )
    entries = document.get("obstacles")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 8:
        raise ValueError("Static obstacle preset must contain 1..8 obstacles")
    if len(route_points) < 2:
        raise ValueError("Cannot bind a static preset to an empty route")

    placements = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError("Preset obstacle {} must be a mapping".format(index))
        model = str(entry.get("model", "")).strip()
        position = entry.get("position")
        scale = entry.get("scale")
        if not model:
            raise ValueError("Preset obstacle {} has no model".format(index))
        if not isinstance(position, dict) or any(key not in position for key in ("x", "y", "z")):
            raise ValueError("Preset obstacle {} has an invalid position".format(index))
        if not isinstance(scale, list) or len(scale) != 3 or any(float(value) <= 0 for value in scale):
            raise ValueError("Preset obstacle {} has an invalid scale".format(index))
        x, y, z = (float(position[key]) for key in ("x", "y", "z"))
        yaw = float(entry.get("yaw_deg", 0.0))
        values = (x, y, z, yaw, *(float(value) for value in scale))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Preset obstacle {} contains a non-finite value".format(index))
        route_index = min(
            range(len(route_points)),
            key=lambda point_index: math.hypot(
                float(route_points[point_index][0]) - x,
                float(route_points[point_index][1]) - y,
            ),
        )
        route_distance = math.hypot(
            float(route_points[route_index][0]) - x,
            float(route_points[route_index][1]) - y,
        )
        placements.append({
            "model": model,
            "position": {"x": x, "y": y, "z": z},
            "yaw_deg": yaw,
            "scale": [float(value) for value in scale],
            "route_path_index": int(route_index_offset) + route_index,
            "route_distance_m": route_distance,
            "source_actor_id": str(entry.get("source_actor_id", "")),
        })
    return {
        "path": str(path),
        "name": str(document.get("name") or path.stem),
        "placements": placements,
    }


def _spawn_one_static_obstacle(
    self, spec, placement, model_offset, requested_model=None
):
    spawn_x = float(placement["position"]["x"])
    spawn_y = float(placement["position"]["y"])
    transform = self.grpc.make_transform(
        spawn_x,
        spawn_y,
        placement["position"]["z"],
        placement.get("yaw_deg", spec["yaw_deg"]),
    )

    # The editor may intentionally place two run-owned obstacles less than
    # three metres apart. Preserve obstacles spawned earlier in this layout
    # while still deleting stale actors left by an interrupted prior run.
    preserved_ids = {
        str(item.get_id()).strip()
        for item in getattr(self, "static_obstacles", [])
    }
    if preserved_ids:
        nearby_before = self.grpc.obstacle_ids_near(
            spawn_x, spawn_y, max_dist_m=3.0,
        )
        stale_ids = [
            actor_id for actor_id in nearby_before
            if str(actor_id).strip() not in preserved_ids
        ]
        for actor_id in stale_ids:
            self.grpc.destroy_obstacle_by_id(actor_id)
        if stale_ids:
            time.sleep(0.2)
        remaining_ids = self.grpc.obstacle_ids_near(
            spawn_x, spawn_y, max_dist_m=3.0,
        )
        unexpected = [
            actor_id for actor_id in remaining_ids
            if str(actor_id).strip() not in preserved_ids
        ]
        if unexpected:
            raise RuntimeError(
                "Failed to clear stale static obstacle at spawn: "
                f"actor_ids={unexpected}"
            )
        preserved_nearby = [str(value).strip() for value in remaining_ids]
    else:
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
        preserved_nearby = []

    actor = None
    if requested_model is not None:
        if requested_model not in self.static_obstacle_models:
            raise RuntimeError(
                "Preset static obstacle model is unavailable: {}".format(
                    requested_model
                )
            )
        models = [requested_model]
    else:
        models = list(self.static_obstacle_models)
        models = models[model_offset:] + models[:model_offset]
    selected_scale = None
    for model in models:
        if requested_model is not None:
            # Explicit historical presets replay exactly as verified.
            scale_values = placement.get("scale") or [1.0, 1.0, 1.0]
        else:
            # Fresh layouts use native asset size, including the single-object
            # path whose route placement can still contain a legacy scale.
            scale_values = [1.0, 1.0, 1.0]
        actor = self.grpc.spawn_obstacle(
            transform=transform,
            model_name=model,
            label=f"full_loop_static_{self.profile_index}_{model_offset + 1}",
            scale=self.make_actor_scale(scale_values),
        )
        if actor is not None:
            selected_scale = [float(value) for value in scale_values]
            print(
                f"[FullLoop] static obstacle {model_offset + 1} selected "
                f"model={model}, position=({spawn_x:.2f},{spawn_y:.2f}), "
                f"scale={selected_scale}"
            )
            break
        print(f"[FullLoop] static obstacle model unavailable at spawn: {model}")
    if actor is None:
        raise RuntimeError(
            "Failed to spawn every randomized static obstacle candidate: "
            f"{models}"
        )

    spawned_id = str(actor.get_id()).strip()
    state, observations = lock_static_obstacle_transform(
        self,
        actor,
        spawned_id,
        transform,
        (spawn_x, spawn_y, float(placement["position"]["z"])),
    )
    if state is None:
        actor.destroy()
        raise RuntimeError(
            "Static obstacle could not be locked at its requested position: "
            f"actor_id={spawned_id}, observations={observations}"
        )

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
    expected_nearby = set(preserved_nearby) | {spawned_id}
    if set(str(value).strip() for value in nearby_ids) != expected_nearby:
        for actor_id in nearby_ids:
            if str(actor_id).strip() not in preserved_ids:
                self.grpc.destroy_obstacle_by_id(actor_id)
        raise RuntimeError(
            "Static obstacle inventory verification failed: "
            f"expected={sorted(expected_nearby)}, nearby={nearby_ids}"
        )

    location = state.transform.location
    xy_error = math.hypot(
        float(location.x) - spawn_x,
        float(location.y) - spawn_y,
    )
    z_error = abs(float(location.z) - float(placement["position"]["z"]))
    expected = {
        "actor_id": spawned_id,
        "x": spawn_x,
        "y": spawn_y,
        "z": float(placement["position"]["z"]),
        "model": model,
        "scale": selected_scale,
        "route_s": self.path_s_at_index(int(placement["route_path_index"])),
    }
    self.scenario_actors.append(actor)
    actor_lifecycle.emit(
        self,
        "static_obstacle_locked",
        actor_id=spawned_id,
        model=model,
        physics_enabled=False,
        expected_position={"x": spawn_x, "y": spawn_y, "z": expected["z"]},
        observed_position={
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        },
        inventory_observations=observations,
    )
    return actor, expected


def spawn_static_obstacle(self):
    spec = self.route_definition["zones"]["static_obstacle"]
    preset_path = self.cfg.get("static_obstacle_preset")
    if preset_path:
        # In a mission mode route_points is only the segment window, while
        # path_s_at_index() takes a full-route index.
        window = getattr(self, "mission_route", None)
        preset = load_static_obstacle_preset(
            preset_path,
            self.global_cfg["morai"]["map_name"],
            self.route_points,
            route_index_offset=window.first_index if window is not None else 0,
        )
        placements = preset["placements"]
        count = len(placements)
        print(
            "[FullLoop] static editor preset loaded: name={} count={} path={}".format(
                preset["name"], count, preset["path"]
            )
        )
    else:
        count = int(self.cfg.get("static_obstacle_count", 1))
        if count not in (1, 2):
            raise ValueError(f"static_obstacle_count must be 1 or 2; got {count}")
        placements = select_static_obstacle_placements(self, spec, count)

    self.static_obstacles = []
    self.static_obstacle_expectations = []
    for index, placement in enumerate(placements):
        actor, expected = _spawn_one_static_obstacle(
            self,
            spec,
            placement,
            index,
            requested_model=placement.get("model") if preset_path else None,
        )
        self.static_obstacles.append(actor)
        self.static_obstacle_expectations.append(expected)

    # Keep the existing singular fields for the proven one-obstacle collector.
    self.static_obstacle = self.static_obstacles[0]
    self.static_obstacle_expected = self.static_obstacle_expectations[0]
    self.static_obstacle_model = self.static_obstacle_expected["model"]
    self.static_obstacle_scale = self.static_obstacle_expected["scale"]
    self.static_obstacle_models_selected = [
        item["model"] for item in self.static_obstacle_expectations
    ]
    self.static_obstacle_scales = [
        item["scale"] for item in self.static_obstacle_expectations
    ]
    self.static_obstacle_route_s = max(
        item["route_s"] for item in self.static_obstacle_expectations
    )
    print(
        f"[FullLoop] static obstacle layout ready: count={count}, "
        f"models={self.static_obstacle_models_selected}"
    )


def maintain_static_obstacles_for_editor(self):
    """Keep movable editor obstacles on the road without changing x/y/yaw."""
    if not bool(self.cfg.get("static_obstacle_editor_mode", False)):
        return
    now = time.monotonic()
    if now - float(getattr(self, "_static_editor_obstacle_check_time", 0.0)) < 0.25:
        return
    self._static_editor_obstacle_check_time = now

    actors = list(getattr(self, "static_obstacles", []))
    expectations = list(getattr(self, "static_obstacle_expectations", []))
    if not actors or len(actors) != len(expectations):
        return

    states = list(self.grpc.get_all_obstacle_states())
    states_by_id = {}
    for state in states:
        actor_id = str(state.actor_info.id.value).strip()
        states_by_id.setdefault(actor_id, []).append(state)

    for actor, expected in zip(actors, expectations):
        actor_id = str(expected["actor_id"]).strip()
        matches = states_by_id.get(actor_id, [])
        if len(matches) != 1:
            raise RuntimeError(
                "Static editor obstacle inventory became ambiguous: "
                f"actor_id={actor_id}, exact_matches={len(matches)}"
            )
        state = matches[0]
        location = state.transform.location
        rotation = state.transform.rotation
        x = float(location.x)
        y = float(location.y)
        z = float(location.z)
        yaw = float(rotation.z)
        target_z = float(expected["z"])
        if not all(math.isfinite(value) for value in (x, y, z, yaw)):
            raise RuntimeError(
                "Static editor obstacle has a non-finite transform: "
                f"actor_id={actor_id}, xyz_yaw={(x, y, z, yaw)}"
            )

        if abs(z - target_z) <= 0.25:
            if not bool(actor.set_physics(False)):
                raise RuntimeError(
                    "Static editor could not disable obstacle physics: "
                    f"actor_id={actor_id}"
                )
            continue

        restored = self.grpc.make_transform(x, y, target_z, yaw)
        repaired, observations = lock_static_obstacle_transform(
            self,
            actor,
            actor_id,
            restored,
            (x, y, target_z),
        )
        repaired_z = (
            None if repaired is None
            else float(repaired.transform.location.z)
        )
        if repaired_z is None or abs(repaired_z - target_z) > 0.25:
            raise RuntimeError(
                "Static editor obstacle height repair did not hold: "
                f"actor_id={actor_id}, repaired_z={repaired_z}, "
                f"target_z={target_z:.3f}, observations={observations}"
            )
        print(
            "[STATIC EDITOR] restored obstacle road height: "
            f"actor_id={actor_id}, model={expected.get('model')}, "
            f"z={z:.3f}->{repaired_z:.3f}, x={x:.3f}, y={y:.3f}, "
            f"yaw={yaw:.2f}"
        )


def verify_static_obstacle(self, elapsed, current_s):
    """Fail closed if a supposedly static obstacle falls or vanishes."""
    if bool(self.cfg.get("static_obstacle_editor_mode", False)):
        # Live editor mode is intentionally mutable and never used by capture.
        return
    expectations = list(
        getattr(self, "static_obstacle_expectations", [])
        or ([self.static_obstacle_expected] if self.static_obstacle_expected else [])
    )
    if not expectations:
        return
    passed_zone = float(current_s) > float(self.static_obstacle_route_s) + 30.0
    if passed_zone and self.static_obstacle_pass_verified:
        return
    if (
        not passed_zone
        and float(elapsed) - self.static_obstacle_last_verify_elapsed < 1.0
    ):
        return
    self.static_obstacle_last_verify_elapsed = float(elapsed)
    observed = []
    inventory = []
    for expected in expectations:
        state, observations = read_static_obstacle_state(
            self, expected["actor_id"]
        )
        inventory.append(observations)
        if state is None:
            raise RuntimeError(
                "Static obstacle disappeared before its route zone: "
                f"actor_id={expected['actor_id']}, observations={observations}"
            )
        location = state.transform.location
        xy_error = math.hypot(
            float(location.x) - expected["x"],
            float(location.y) - expected["y"],
        )
        z_error = abs(float(location.z) - expected["z"])
        if xy_error > 1.0 or z_error > 1.0:
            raise RuntimeError(
                "Static obstacle left its verified route position: "
                f"actor_id={expected['actor_id']}, xy_error={xy_error:.3f}m, "
                f"z_error={z_error:.3f}m"
            )
        observed.append({
            "actor_id": expected["actor_id"],
            "model": expected.get("model"),
            "position": {
                "x": float(location.x), "y": float(location.y),
                "z": float(location.z),
            },
            "xy_error_m": float(xy_error),
            "z_error_m": float(z_error),
        })
    if passed_zone:
        self.static_obstacle_pass_verified = True
        actor_lifecycle.emit(
            self,
            "static_obstacle_pass_verified",
            actor_ids=[item["actor_id"] for item in expectations],
            models=[item.get("model") for item in expectations],
            route_s=float(current_s),
            obstacle_route_s=float(self.static_obstacle_route_s),
            observed_obstacles=observed,
            inventory_observations=inventory,
        )
