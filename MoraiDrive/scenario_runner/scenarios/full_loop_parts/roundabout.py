"""Roundabout traffic-zone functions for the full-loop scenario."""

import math

from ...utils.link_graph import LinkGraph
from ...utils.transform_utils import interpolate_on_polyline, polyline_length
from . import actor_lifecycle


def rotate_loop_route(self, route_links, start_link, laps):
    index = route_links.index(start_link)
    rotated = route_links[index:] + route_links[:index]
    route = []
    for _ in range(max(1, int(laps))):
        route.extend(rotated)
    route.append(rotated[0])
    return route


def arm_roundabout_npcs(self):
    lifecycle = actor_lifecycle.policy(self)
    self.roundabout_lifecycle_state = "armed"
    self.nearby_roundabout_state = "armed"
    self.roundabout_spawn_elapsed = None
    self.roundabout_progress_anchor_elapsed = None
    self.roundabout_progress_anchor_s = None
    self.nearby_roundabout_spawn_elapsed = None
    self.nearby_roundabout_progress_anchor_elapsed = None
    self.nearby_roundabout_progress_anchor_s = None
    actor_lifecycle.emit(
        self,
        "nearby_policy",
        policy=lifecycle,
        visibility_verified=False,
        mission_distribution_changed=True,
    )
    print(
        "[FullLoop] roundabout traffic ARMED: "
        f"lead={float(lifecycle['roundabout_lead_m']):.1f}m"
    )


def spawn_roundabout_npcs(self):
    """Public setup hook: arm delayed creation owned by this scenario."""
    return arm_roundabout_npcs(self)


def spawn_roundabout_npcs_now(self):
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
    model = str(self.cfg["roundabout_npc"].get("model", "DEFAULT_SEDAN"))
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
        actor = self.grpc.spawn_vehicle(
            transform=self.grpc.make_transform(x, y, z, yaw),
            model_name=model,
            label=f"roundabout_loop_{index}",
            velocity=float(traffic_speed),
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

        pause_ok = actor.set_pause(False)
        ai_ok = self.grpc.set_vehicle_ai(actor, True)
        limit_ok = self.grpc.set_vehicle_speed_limit(
            actor, traffic_speed, enabled=True
        )
        velocity_ok = self.grpc.set_vehicle_velocity(actor, traffic_speed)
        if not (pause_ok and limit_ok and velocity_ok):
            actor.destroy()
            raise RuntimeError(
                "Failed to establish roundabout NPC motion contract "
                f"{index}: pause={pause_ok}, ai={ai_ok}, "
                f"limit={limit_ok}, velocity={velocity_ok}"
            )
        self.roundabout_npcs.append(
            {
                "actor": actor,
                "model": model,
                "speed_mps": traffic_speed,
                "route": route,
                "ai_enable_acknowledged": bool(ai_ok),
                "speed_limit_acknowledged": bool(limit_ok),
                "initial_velocity_acknowledged": bool(velocity_ok),
            }
        )
        self.scenario_actors.append(actor)


def retire_roundabout_npcs(
    self,
    elapsed,
    ego_state,
    current_s,
    reason,
    **reason_fields,
):
    owned = {str(item["actor"].get_id()) for item in self.roundabout_npcs}
    states = self.grpc.get_all_vehicle_states()
    matches = [
        state
        for state in states
        if str(state.actor_info.id.value) in owned
    ]
    if (
        not owned
        or len(matches) != len(owned)
        or {str(state.actor_info.id.value) for state in matches} != owned
    ):
        return False

    distances = [
        math.hypot(
            state.transform.location.x - ego_state["x"],
            state.transform.location.y - ego_state["y"],
        )
        for state in matches
    ]

    def retire():
        for item in list(self.roundabout_npcs):
            actor = item["actor"]
            if not self.grpc.destroy_vehicle_by_id(str(actor.get_id())):
                raise RuntimeError(
                    "Cannot confirm owned roundabout actor retirement"
                )
            self.scenario_actors.remove(actor)
            self.roundabout_npcs.remove(item)

    lifecycle = actor_lifecycle.policy(self)
    actor_lifecycle.mutate(
        self,
        "roundabout_retire",
        current_s,
        retire,
        actor_ids=sorted(owned),
        ego_actor_distances_m=sorted(float(value) for value in distances),
        retire_reason=reason,
        active_scenario_s=(
            float(elapsed) - float(self.roundabout_spawn_elapsed)
            if self.roundabout_spawn_elapsed is not None
            else None
        ),
        mutation_boundary_requires_dataset_mask=True,
        mutation_boundary_quarantine_s=float(
            lifecycle["roundabout_mutation_quarantine_s"]
        ),
        **reason_fields,
    )
    self.roundabout_lifecycle_state = "retired"
    self.nearby_roundabout_state = "retired"
    return True


def update_roundabout_npcs(self, elapsed, ego_state, current_s):
    state = getattr(self, "roundabout_lifecycle_state", "disabled")
    if state not in ("armed", "spawned"):
        return

    lifecycle = actor_lifecycle.policy(self)
    zone = self.route_definition["zones"]["roundabout"]
    entry_s = actor_lifecycle.checkpoint_s(self, zone["ego_entry_checkpoint"])
    exit_s = actor_lifecycle.checkpoint_s(self, zone["ego_exit_checkpoint"])

    if (
        state == "armed"
        and float(current_s)
        >= entry_s - float(lifecycle["roundabout_lead_m"])
    ):
        actor_lifecycle.mutate(
            self,
            "roundabout_spawn",
            current_s,
            lambda: spawn_roundabout_npcs_now(self),
        )
        self.roundabout_lifecycle_state = "spawned"
        self.nearby_roundabout_state = "spawned"
        self.roundabout_spawn_elapsed = float(elapsed)
        self.roundabout_progress_anchor_elapsed = float(elapsed)
        self.roundabout_progress_anchor_s = float(current_s)
        self.nearby_roundabout_spawn_elapsed = float(elapsed)
        self.nearby_roundabout_progress_anchor_elapsed = float(elapsed)
        self.nearby_roundabout_progress_anchor_s = float(current_s)
        actor_lifecycle.emit(
            self,
            "roundabout_spawn_contract",
            route_s=float(current_s),
            elapsed_seconds=float(elapsed),
            actor_ids=sorted(
                str(item["actor"].get_id()) for item in self.roundabout_npcs
            ),
            actor_motion_contracts=[
                {
                    "actor_id": str(item["actor"].get_id()),
                    "ai_enable_acknowledged": bool(
                        item.get("ai_enable_acknowledged", False)
                    ),
                    "speed_limit_acknowledged": bool(
                        item.get("speed_limit_acknowledged", False)
                    ),
                    "initial_velocity_acknowledged": bool(
                        item.get("initial_velocity_acknowledged", False)
                    ),
                }
                for item in self.roundabout_npcs
            ],
            target_speed_mps=float(
                self.cfg.get("roundabout_npc", {}).get("speed_mps", 0.0)
            ),
            stall_window_s=float(lifecycle["roundabout_stall_window_s"]),
            stall_min_progress_m=float(
                lifecycle["roundabout_stall_min_progress_m"]
            ),
            training_visible=False,
        )
        state = "spawned"

    if state != "spawned":
        return

    retire_s = exit_s + float(lifecycle["roundabout_retire_after_m"])
    if float(current_s) > retire_s:
        retire_roundabout_npcs(
            self,
            elapsed,
            ego_state,
            current_s,
            "route_exit",
            route_exit_contract_s=float(retire_s),
        )
        return

    if self.roundabout_progress_anchor_s is None:
        self.roundabout_progress_anchor_s = float(current_s)
    if self.roundabout_progress_anchor_elapsed is None:
        self.roundabout_progress_anchor_elapsed = float(elapsed)
    if self.roundabout_spawn_elapsed is None:
        self.roundabout_spawn_elapsed = float(elapsed)

    progress_since_anchor = (
        float(current_s) - float(self.roundabout_progress_anchor_s)
    )
    if progress_since_anchor >= float(
        lifecycle["roundabout_stall_min_progress_m"]
    ):
        self.roundabout_progress_anchor_s = float(current_s)
        self.roundabout_progress_anchor_elapsed = float(elapsed)
        self.nearby_roundabout_progress_anchor_s = float(current_s)
        self.nearby_roundabout_progress_anchor_elapsed = float(elapsed)
        progress_since_anchor = 0.0

    active_scenario_s = float(elapsed) - float(self.roundabout_spawn_elapsed)
    stalled_scenario_s = (
        float(elapsed) - float(self.roundabout_progress_anchor_elapsed)
    )
    if (
        active_scenario_s
        >= float(lifecycle["roundabout_min_active_scenario_s"])
        and stalled_scenario_s
        >= float(lifecycle["roundabout_stall_window_s"])
    ):
        retire_roundabout_npcs(
            self,
            elapsed,
            ego_state,
            current_s,
            "progress_stall",
            stall_scenario_s=float(stalled_scenario_s),
            progress_anchor_route_s=float(
                self.roundabout_progress_anchor_s
            ),
            progress_since_anchor_m=float(progress_since_anchor),
            route_exit_contract_s=float(retire_s),
        )
