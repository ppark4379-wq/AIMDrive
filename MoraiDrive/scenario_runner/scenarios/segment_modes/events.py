"""Zone gates for new modes. Enabled events delegate to existing functions."""
from dataclasses import asdict
from pathlib import Path

from .modes import get_mission_mode
from ..full_loop_parts import (
    dynamic_obstacle,
    highway,
    roundabout,
    route_profile,
    static_obstacle,
    tunnel,
)


def prepare_mission(context):
    context.mission_mode = get_mission_mode(context.cfg["mission_mode"])
    context.scenario_name = context.mission_mode.name
    context.mission_route = None
    if context.scenario_name == "full_loop_tunnel_dynamic":
        context.cfg.setdefault("tunnel_dynamic", {})["enabled"] = True
    if context.scenario_name == "full_loop_tunnel_vehicle":
        context.cfg.setdefault("tunnel_vehicle_dynamic", {})["enabled"] = True
        context.cfg.setdefault("tunnel_dynamic", {})["enabled"] = False
        context.cfg.setdefault("tunnel_sudden_brake", {})["enabled"] = False
    # Each mode alternates its own weather without consuming full_loop history.
    profile_path = Path(context.cfg["profile_state_path"])
    context.cfg["profile_state_path"] = str(
        profile_path.with_name(context.scenario_name + "_profile.json")
    )
    print(
        f"[Mission] mode={context.scenario_name} "
        f"zones={asdict(context.mission_mode.zones)}"
    )


def clear_previous_run(context):
    lifecycle = context.cfg["highway_npc"].get("spawn_point_lifecycle", {})
    settle = max(0.0, float(lifecycle.get("cleanup_settle_sec", 1.5)))
    # Only this runner's client-key actors, never foreign/manual scene actors.
    if not context.grpc.reset_actors(settle_sec=settle):
        raise RuntimeError("Mission preflight failed to clear runner-owned actors")


def select_models(context, enabled, key, available_getter, label):
    if not enabled:
        return []
    models = route_profile.filter_models(
        context, context.cfg.get(key), available_getter(), label,
    )
    context.random.shuffle(models)
    return models


def resolve_profile_models(context):
    clear_previous_run(context)
    zones = context.mission_mode.zones
    context.static_obstacle_models = select_models(
        context, zones.static, "static_obstacle_models",
        context.grpc.get_available_obstacle_models, "static obstacle",
    )
    context.dynamic_obstacle_models = select_models(
        context, zones.dynamic, "dynamic_obstacle_models",
        context.grpc.get_available_pedestrian_models, "dynamic pedestrian",
    )
    previous_static = context.previous_profile_state.get(
        "static_obstacle_model"
    )
    if previous_static in context.static_obstacle_models and len(
        context.static_obstacle_models
    ) > 1:
        context.static_obstacle_models.append(
            context.static_obstacle_models.pop(
                context.static_obstacle_models.index(previous_static)
            )
        )
    context.static_obstacle_model = (
        context.static_obstacle_models[0] if zones.static else "OFF"
    )
    context.dynamic_obstacle_model = (
        context.dynamic_obstacle_models[0] if zones.dynamic else "OFF"
    )
    context.static_obstacle_scale = []
    context.static_obstacle = None
    context.static_obstacles = []
    context.static_obstacle_expectations = []
    context.static_obstacle_models_selected = []
    context.static_obstacle_scales = []


def spawn_static_obstacle(context):
    if context.mission_mode.zones.static:
        return static_obstacle.spawn_static_obstacle(context)
    print("[Mission] static obstacle OFF")


def spawn_dynamic_obstacle(context):
    if context.mission_mode.zones.dynamic:
        return dynamic_obstacle.spawn_dynamic_obstacle(context)
    context.dynamic_obstacle_phase = "disabled"
    print("[Mission] dynamic obstacle OFF")


def spawn_roundabout_npcs(context):
    if context.mission_mode.zones.roundabout:
        return roundabout.spawn_roundabout_npcs(context)
    print("[Mission] roundabout NPCs OFF")


def spawn_highway_npcs(context):
    if context.mission_mode.zones.highway:
        return highway.spawn_highway_npcs(context)
    context.highway_traffic_cleared = True
    print("[Mission] highway SpawnPoints OFF")


def arm_tunnel_sudden_brake(context):
    if context.mission_mode.zones.tunnel:
        return tunnel.arm_tunnel_sudden_brake(context)
    context.tunnel_event_phase = "disabled"
    context.tunnel_event_result = {"disabled_reason": context.scenario_name}
    print("[Mission] tunnel NPC/event OFF")


def update_dynamic_obstacle(context, ego_state, current_s):
    if context.mission_mode.zones.dynamic:
        return dynamic_obstacle.update_dynamic_obstacle(context, ego_state, current_s)


def activate_highway_spawn_points(context, elapsed, current_s):
    if context.mission_mode.zones.highway:
        return highway.activate_highway_spawn_points(context, elapsed, current_s)
    return False


def update_tunnel_sudden_brake(context, elapsed, ego_state, current_s):
    if context.mission_mode.zones.tunnel:
        return tunnel.update_tunnel_sudden_brake(context, elapsed, ego_state, current_s)
