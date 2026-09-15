"""One thin compatibility class for the existing runner lifecycle.

New mode data are frozen dataclasses; operations live in ordinary functions.
The existing FullLoopScenario and its five zone implementations are unchanged.
"""
from ..full_loop_scenario import FullLoopScenario
from . import events, expert, route


class MissionScenario(FullLoopScenario):
    def setup(self):
        events.prepare_mission(self)
        super().setup()

    def setup_expert_controller(self):
        mission_points = self.route_points
        self.route_points = route.build_expert_route_points(self)
        try:
            super().setup_expert_controller()
        finally:
            self.route_points = mission_points
        expert.prepare_speed_zones(self)

    prepare_full_loop_route = route.prepare_full_loop_route
    path_s_at_index = route.path_s_at_index
    resolve_profile_models = events.resolve_profile_models
    spawn_static_obstacle = events.spawn_static_obstacle
    spawn_dynamic_obstacle = events.spawn_dynamic_obstacle
    spawn_roundabout_npcs = events.spawn_roundabout_npcs
    spawn_highway_npcs = events.spawn_highway_npcs
    arm_tunnel_sudden_brake = events.arm_tunnel_sudden_brake
    update_dynamic_obstacle = events.update_dynamic_obstacle
    activate_highway_spawn_points = events.activate_highway_spawn_points
    update_tunnel_sudden_brake = events.update_tunnel_sudden_brake
