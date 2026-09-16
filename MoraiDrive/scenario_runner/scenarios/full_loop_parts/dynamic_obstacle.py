"""Dynamic pedestrian-obstacle zone functions for the full-loop scenario."""

import math
import time


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
    self.dynamic_obstacle_started_scenario_time = self._scenario_now()
    self.dynamic_obstacle_started_at = self.dynamic_obstacle_started_scenario_time
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
    actor = self.dynamic_obstacle
    if actor is not None:
        start = self.dynamic_obstacle_start
        end = self.dynamic_obstacle_end
        dx = float(end["x"]) - float(start["x"])
        dy = float(end["y"]) - float(start["y"])
        length = math.hypot(dx, dy)
        if length < 1e-3:
            raise RuntimeError("Dynamic obstacle stop direction is undefined")
        # MORAI 26.R1 SetPause can restore a completed pedestrian to its
        # original spawn transform. Keep the actor at the naturally reached
        # endpoint by issuing a zero-speed control in its crossing direction.
        stopped = self.grpc.control_pedestrian(
            actor,
            dx / length,
            dy / length,
            0.0,
            verbose=True,
        )
        if not stopped:
            raise RuntimeError(
                "Failed to stop dynamic obstacle at the reached endpoint"
            )
    self.dynamic_obstacle_phase = "cleared"
    now = self._scenario_now()
    started = getattr(
        self,
        "dynamic_obstacle_started_scenario_time",
        getattr(self, "dynamic_obstacle_started_at", None),
    )
    elapsed = now - float(started if started is not None else now)
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
    now = self._scenario_now()
    started = getattr(
        self,
        "dynamic_obstacle_started_scenario_time",
        getattr(self, "dynamic_obstacle_started_at", None),
    )
    elapsed = now - float(started if started is not None else now)

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
