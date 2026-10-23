"""One thin compatibility class for the existing runner lifecycle.

New mode data are frozen dataclasses; operations live in ordinary functions.
The existing FullLoopScenario and its five zone implementations are unchanged.
"""
from ..full_loop_scenario import FullLoopScenario
from . import events, expert, route, traffic_light, tunnel_baseline, tunnel_static_trial, tunnel_static_startup


class MissionScenario(FullLoopScenario):
    def stop_drive_control(self):
        if tunnel_static_trial.enabled(self):
            tunnel_static_trial.immediate_hold(self)
        return super().stop_drive_control()

    def cleanup(self):
        if tunnel_static_trial.enabled(self):
            tunnel_static_trial.immediate_hold(self)
        return super().cleanup()

    def cleanup_spawned_actors(self):
        if tunnel_static_trial.enabled(self):
            tunnel_static_trial.immediate_hold(self)
        return super().cleanup_spawned_actors()

    def run(self):
        if tunnel_static_trial.enabled(self):
            tunnel_static_trial.required(self)
        try:
            try:
                return super().run()
            finally:
                tunnel_baseline.park(self)
                tunnel_static_trial.park(self)
                traffic_light.restore(self)
        except (Exception, KeyboardInterrupt) as error:
            tunnel_baseline.finish(self, "failed", str(error) or type(error).__name__)
            traffic_light.finish(self, "failed", str(error) or type(error).__name__)
            if tunnel_static_trial.enabled(self):
                tunnel_static_trial.required(self).fail(error)
            raise

    def setup(self):
        if tunnel_static_trial.enabled(self):
            tunnel_static_trial.required(self)
        events.prepare_mission(self)
        try:
            super().setup()
        except (Exception, KeyboardInterrupt) as error:
            tunnel_baseline.finish(self, "failed", str(error) or type(error).__name__)
            traffic_light.finish(self, "failed", str(error) or type(error).__name__)
            raise

    def prepare_full_loop_route(self):
        route.prepare_full_loop_route(self)
        tunnel_baseline.prepare(self)
        traffic_light.prepare(self)
        if tunnel_static_trial.enabled(self) and getattr(self, "tunnel_static_trial_context", None) is not None:
            tunnel_static_trial.required(self).prepare(self)

    def configure_drive_with_retries(self):
        tunnel_baseline.before_drive(self)
        traffic_light.before_drive(self)
        if tunnel_static_trial.enabled(self):
            trial = tunnel_static_trial.required(self)
            trial.bind_static_surfaces(self)
            trial.bind_vehicle_specs(self)
            trial.inspect(self, 0.)
            trial.report["status"] = "running"
            trial.persist()
        return super().configure_drive_with_retries()

    def configure_drive_from_start(self):
        if tunnel_static_trial.enabled(self):
            return tunnel_static_startup.configure_drive(self)
        return super().configure_drive_from_start()

    def activate_expert_drive(self, *, before_expert_start=None, after_expert_setup=None,
                             after_expert_start=None):
        if traffic_light.enabled(self):
            previous_setup_hook = after_expert_setup

            def after_expert_setup():
                if previous_setup_hook is not None:
                    previous_setup_hook()
                # RawScenario.setup_expert_controller includes native readiness.
                traffic_light.record_start(self)

        return super().activate_expert_drive(
            before_expert_start=before_expert_start, after_expert_setup=after_expert_setup,
            after_expert_start=after_expert_start)

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        tunnel_baseline.observe(self, elapsed, ego_state, current_s)
        traffic_light.observe(self, elapsed, ego_state, current_s)
        if tunnel_static_trial.enabled(self):
            tunnel_static_trial.required(self).inspect(self, elapsed, ego_state)
        return super().on_expert_timeline_tick(
            elapsed, ego_state, current_s, remaining_s,
        )

    def on_goal_reached_before_stop(self, *, elapsed, ego_state, current_s, remaining_s):
        if tunnel_static_trial.enabled(self):
            tunnel_static_trial.request_normal_goal_stop(
                self, elapsed=elapsed, ego_state=ego_state,
                current_s=current_s, remaining_s=remaining_s,
            )
        return super().on_goal_reached_before_stop(
            elapsed=elapsed, ego_state=ego_state,
            current_s=current_s, remaining_s=remaining_s,
        )

    def run_timeline(self):
        try:
            super().run_timeline()
            tunnel_baseline.finish(self, "passed")
            traffic_light.finish(self, "passed")
            if tunnel_static_trial.enabled(self):
                tunnel_static_trial.required(self).finish(self)
        except (Exception, KeyboardInterrupt) as error:
            tunnel_static_trial.cancel_normal_goal_stop(self, error)
            tunnel_baseline.finish(self, "failed", str(error) or type(error).__name__)
            traffic_light.finish(self, "failed", str(error) or type(error).__name__)
            raise

    def setup_expert_controller(self):
        mission_points = self.route_points
        self.route_points = route.build_expert_route_points(self)
        try:
            super().setup_expert_controller()
        finally:
            self.route_points = mission_points
        expert.prepare_speed_zones(self)
        traffic_light.verify_runtime(self)
        if tunnel_static_trial.enabled(self):
            expert.prepare_tunnel_static_inputs(self)

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
