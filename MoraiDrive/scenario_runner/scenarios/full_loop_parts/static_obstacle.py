"""Static-obstacle creation and exact run-owned state verification."""

import math
import time

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
            physics_ok = bool(actor.set_physics(False))
            if not physics_ok:
                actor.destroy()
                raise RuntimeError(
                    "Failed to disable physics for static obstacle "
                    f"model={model}"
                )
            self.static_obstacle_model = model
            self.static_obstacle_scale = [float(value) for value in scale_values]
            print(
                f"[FullLoop] static obstacle selected model={model}, "
                f"scale={self.static_obstacle_scale}"
            )
            break
        print(f"[FullLoop] static obstacle model unavailable at spawn: {model}")
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

    state, observations = read_static_obstacle_state(self, spawned_id)
    if state is None:
        actor.destroy()
        raise RuntimeError(
            "Static obstacle state unavailable after physics lock: "
            f"actor_id={spawned_id}, observations={observations}"
        )
    location = state.transform.location
    xy_error = math.hypot(
        float(location.x) - spawn_x,
        float(location.y) - spawn_y,
    )
    z_error = abs(float(location.z) - float(spec["position"]["z"]))
    if xy_error > 1.0 or z_error > 1.0:
        actor.destroy()
        raise RuntimeError(
            "Static obstacle moved away before run start: "
            f"actor_id={spawned_id}, xy_error={xy_error:.3f}m, "
            f"z_error={z_error:.3f}m"
        )

    self.static_obstacle = actor
    self.static_obstacle_expected = {
        "actor_id": spawned_id,
        "x": spawn_x,
        "y": spawn_y,
        "z": float(spec["position"]["z"]),
    }
    self.static_obstacle_route_s = self.path_s_at_index(
        int(spec["route_path_index"])
    )
    self.scenario_actors.append(actor)
    actor_lifecycle.emit(
        self,
        "static_obstacle_locked",
        actor_id=spawned_id,
        model=self.static_obstacle_model,
        physics_enabled=False,
        expected_position={"x": spawn_x, "y": spawn_y, "z": self.static_obstacle_expected["z"]},
        observed_position={
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        },
        inventory_observations=observations,
    )


def verify_static_obstacle(self, elapsed, current_s):
    """Fail closed if a supposedly static obstacle falls or vanishes."""
    expected = self.static_obstacle_expected
    if expected is None or self.static_obstacle is None:
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
    state, observations = read_static_obstacle_state(self, expected["actor_id"])
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
    if passed_zone:
        self.static_obstacle_pass_verified = True
        actor_lifecycle.emit(
            self,
            "static_obstacle_pass_verified",
            actor_id=expected["actor_id"],
            model=self.static_obstacle_model,
            route_s=float(current_s),
            obstacle_route_s=float(self.static_obstacle_route_s),
            observed_position={
                "x": float(location.x),
                "y": float(location.y),
                "z": float(location.z),
            },
            xy_error_m=float(xy_error),
            z_error_m=float(z_error),
            inventory_observations=observations,
        )
