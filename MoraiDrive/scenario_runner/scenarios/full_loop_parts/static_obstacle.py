"""Static-obstacle zone functions for the full-loop scenario."""

import time
from . import static_obstacle_watch


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
    static_obstacle_watch.arm(self)
