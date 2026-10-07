"""Single slow-vehicle trials, isolated from the legacy tunnel brake event."""
import json
import math
from pathlib import Path

from . import tunnel
from .tunnel_dynamic_catalog import _finite
from .tunnel_vehicle_catalog import generate_tunnel_vehicle_candidates
from ...utils.transform_utils import interpolate_on_polyline, project_distance_on_polyline


class TunnelVehicleHandler:
    def validate(self, context):
        cfg = context.cfg["tunnel_vehicle_dynamic"]
        settings = {}
        for key in ("slow_speed_mps", "spawn_trigger_gap_m", "minimum_spawn_gap_m",
                    "spawn_position_tolerance_m", "arrival_tolerance_m",
                    "minimum_observed_travel_m", "max_run_sec"):
            settings[key] = _finite(cfg.get(key), key)
            if settings[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if settings["spawn_trigger_gap_m"] <= settings["minimum_spawn_gap_m"]:
            raise ValueError("Vehicle trigger gap must exceed minimum spawn gap")
        zone = context.route_definition["zones"]["tunnel"]
        candidates = generate_tunnel_vehicle_candidates(
            context.route_points, context.tunnel_entry_s, context.tunnel_exit_s,
            zone["candidate_catalog"], zone["vehicle_candidate_catalog"])
        candidate_id = cfg.get("trial_candidate_id", "vehicle_slow_01")
        selected = next((p for p in candidates if p["id"] == candidate_id), None)
        if selected is None or selected["status"] == "disabled":
            raise ValueError(f"Unknown or disabled tunnel vehicle candidate: {candidate_id}")
        travel = selected["end_s_m"] - selected["start_s_m"]
        if (settings["minimum_observed_travel_m"] >= travel - settings["arrival_tolerance_m"]
                or settings["arrival_tolerance_m"] >= travel / 2):
            raise ValueError("Vehicle travel verification tolerances exceed candidate length")
        if settings["max_run_sec"] <= travel / settings["slow_speed_mps"]:
            raise ValueError("Vehicle timeout must exceed nominal travel time")
        trigger_s = selected["start_s_m"] - settings["spawn_trigger_gap_m"]
        if trigger_s < 0:
            raise ValueError("Vehicle trial needs an earlier Ego start to provide spawn gap")
        links = list(zone.get("npc_route_links") or zone["route_links"])
        if not links or any(link not in context.map_loader.link_set for link in links):
            raise ValueError("Tunnel vehicle route contains unavailable links")
        # Reuse the legacy link/waypoint lookup and route assignment. Keep the
        # remaining route for AI; the handler removes the actor at the trial end.
        context.tunnel_npc_route = links
        start = selected["start_xyz"]
        error, route_index, waypoint = tunnel._tunnel_route_start(context, start["x"], start["y"])
        if error > settings["spawn_position_tolerance_m"]:
            raise ValueError("Tunnel vehicle spawn is too far from its MGeo route")
        context.tunnel_vehicle_plan = {
            **selected, **settings, "trigger_s_m": trigger_s,
            "route_links": links[route_index:], "start_waypoint": int(waypoint),
            "model": str(cfg.get("model") or context.cfg["highway_npc"]["model"]),
        }
        context.tunnel_vehicle_actor = None
        context.tunnel_vehicle_phase = "validated"
        context.tunnel_vehicle_report_path = None

    def arm(self, context):
        plan = context.tunnel_vehicle_plan
        if plan["model"] not in context.vehicle_models():
            raise RuntimeError(f"Tunnel vehicle model unavailable: {plan['model']}")
        context.highway_fixed_model = plan["model"]
        context.tunnel_event_phase = "disabled"
        context.tunnel_event_result = {"disabled_reason": "tunnel_vehicle_dynamic"}
        context.tunnel_vehicle_phase = "waiting"
        context.tunnel_vehicle_result = {
            "run_seed": context.run_seed, "plan": dict(plan),
            "spawned": False, "spawn_verified": False, "observed_moving": False,
            "completed": False, "destroyed": False, "observations": [],
        }
        context.tunnel_vehicle_last_log_sec = -1
        context.tunnel_vehicle_report_path = (
            Path(context.cfg["profile_state_path"]).parent / "tunnel_vehicle_runs"
            / f"{context.run_instance_id}.json")
        self._save(context)
        print(f"[TunnelVehicle] armed candidate={plan['id']} model={plan['model']} "
              f"speed={plan['slow_speed_mps']:.1f}m/s trigger_s={plan['trigger_s_m']:.1f} "
              f"spawn_gap={plan['spawn_trigger_gap_m']:.1f}m report={context.tunnel_vehicle_report_path}")

    def _save(self, context):
        path = context.tunnel_vehicle_report_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**context.tunnel_vehicle_result, "phase": context.tunnel_vehicle_phase}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _remove(self, context):
        actor = context.tunnel_vehicle_actor
        if actor is None:
            return
        if actor.destroy() is False:
            raise RuntimeError("Tunnel vehicle removal rejected")
        if actor in context.scenario_actors:
            context.scenario_actors.remove(actor)
        context.tunnel_vehicle_actor = None
        context.tunnel_vehicle_result["destroyed"] = True

    def record_interruption(self, context, error):
        if context.tunnel_vehicle_phase == "completed":
            return
        context.tunnel_vehicle_phase = "failed"
        context.tunnel_vehicle_result["failure_reason"] = str(error) or type(error).__name__
        try:
            self._remove(context)
        except Exception as cleanup_error:
            context.tunnel_vehicle_result["cleanup_error"] = str(cleanup_error)
        self._save(context)

    def _fail(self, context, message):
        error = RuntimeError(message)
        self.record_interruption(context, error)
        raise error

    def _read_state(self, context):
        state = context.tunnel_vehicle_actor.get_actor_state()
        if state is None:
            self._fail(context, "Tunnel vehicle state unavailable")
        location = state.transform.location
        x, y = float(location.x), float(location.y)
        speed = tunnel._actor_speed(state)
        if not all(math.isfinite(value) for value in (x, y, speed)):
            self._fail(context, "Tunnel vehicle state is not finite")
        npc_s = project_distance_on_polyline(context.route_points, x, y)
        try:
            link = str(state.vehicle_state.current_link_info.id.value)
        except AttributeError:
            link = ""
        return x, y, npc_s, speed, link

    def _spawn(self, context, elapsed, current_s):
        plan = context.tunnel_vehicle_plan
        planned_gap = plan["start_s_m"] - current_s
        if planned_gap < plan["minimum_spawn_gap_m"]:
            self._fail(context, f"Tunnel vehicle unsafe spawn gap: {planned_gap:.1f}m")
        x, y, z, yaw = interpolate_on_polyline(context.route_points, plan["start_s_m"])
        context.tunnel_vehicle_actor = context._spawn_highway_vehicle(
            context.grpc.make_transform(x, y, z, yaw), plan["route_links"],
            plan["start_waypoint"], plan["id"], plan["slow_speed_mps"])
        actor = context.tunnel_vehicle_actor
        if actor is None:
            self._fail(context, "Failed to spawn tunnel slow vehicle or assign its route")
        context.tunnel_vehicle_spawn_elapsed = elapsed
        context.tunnel_vehicle_result.update({"spawned": True, "spawn_elapsed_sec": elapsed,
                                               "planned_initial_gap_m": planned_gap})
        ax, ay, npc_s, speed, link = self._read_state(context)
        error = math.hypot(ax - x, ay - y)
        actual_gap = npc_s - current_s
        context.tunnel_vehicle_result.update({
            "spawn_error_m": error, "initial_route_gap_m": actual_gap,
            "actual_spawn": {"x": ax, "y": ay, "link": link, "speed_mps": speed},
        })
        if (error > plan["spawn_position_tolerance_m"]
                or actual_gap < plan["minimum_spawn_gap_m"]
                or (link and link not in plan["route_links"])):
            self._fail(context, f"Tunnel vehicle spawn verification failed: error={error:.2f}m gap={actual_gap:.1f}m link={link}")
        # The shared helper starts route AI; require acknowledgement of this
        # trial's speed commands too, rather than accepting a silent failure.
        if not context.grpc.set_vehicle_speed_limit(actor, plan["slow_speed_mps"], enabled=True):
            self._fail(context, "Tunnel vehicle speed limit rejected")
        if not context.grpc.set_vehicle_velocity(actor, plan["slow_speed_mps"]):
            self._fail(context, "Tunnel vehicle initial velocity rejected")
        if not context.grpc.set_vehicle_ai(actor, True):
            self._fail(context, "Tunnel vehicle route AI rejected")
        context.tunnel_vehicle_actual_start_s = npc_s
        context.tunnel_vehicle_result["spawn_verified"] = True
        context.tunnel_vehicle_phase = "moving"
        self._save(context)
        print(f"[TunnelVehicle] SPAWN candidate={plan['id']} error={error:.2f}m "
              f"gap={actual_gap:.1f}m target_speed={plan['slow_speed_mps']:.1f}m/s")

    def update(self, context, elapsed, ego_state, current_s):
        phase = context.tunnel_vehicle_phase
        if phase in ("completed", "failed"):
            return
        plan = context.tunnel_vehicle_plan
        if phase == "waiting":
            if current_s >= plan["trigger_s_m"]:
                self._spawn(context, float(elapsed), float(current_s))
            return
        age = float(elapsed) - context.tunnel_vehicle_spawn_elapsed
        if age > plan["max_run_sec"]:
            self._fail(context, "Tunnel slow vehicle timed out")
        x, y, npc_s, speed, link = self._read_state(context)
        if link and link not in plan["route_links"]:
            self._fail(context, f"Tunnel vehicle left its assigned route: {link}")
        route_x, route_y, _, _ = interpolate_on_polyline(context.route_points, npc_s)
        if math.hypot(x - route_x, y - route_y) > plan["spawn_position_tolerance_m"]:
            self._fail(context, "Tunnel vehicle left the trial lane")
        result = context.tunnel_vehicle_result
        travel = max(0.0, npc_s - context.tunnel_vehicle_actual_start_s)
        result["observed_moving"] |= speed > 0.5
        observation = {"elapsed_sec": float(elapsed), "x": x, "y": y, "link": link,
                       "npc_s_m": npc_s, "route_gap_m": npc_s - current_s,
                       "speed_mps": speed, "travelled_m": travel}
        result["last_observation"] = observation
        second = int(age)
        if second != context.tunnel_vehicle_last_log_sec:
            context.tunnel_vehicle_last_log_sec = second
            result["observations"].append(observation)
            self._save(context)
            print(f"[TunnelVehicle] moving candidate={plan['id']} travelled={travel:.1f}m "
                  f"speed={speed:.1f}m/s gap={npc_s-current_s:.1f}m")
        if npc_s >= plan["end_s_m"] - plan["arrival_tolerance_m"]:
            if (not result["observed_moving"] or travel < plan["minimum_observed_travel_m"]
                    or age < plan["minimum_observed_travel_m"] / plan["slow_speed_mps"]):
                self._fail(context, "Tunnel vehicle reached end without verified travel")
            self._remove(context)
            result["completed"] = True
            result["completion_elapsed_sec"] = float(elapsed)
            context.tunnel_vehicle_phase = "completed"
            self._save(context)
            print(f"[TunnelVehicle] completed candidate={plan['id']} travelled={travel:.1f}m")


def prepare_tunnel_vehicle(context):
    context.tunnel_vehicle_handler = None
    # Enable only this dedicated trial mode, even if another local YAML has
    # enabled the setting. Full-loop's legacy brake event keeps its own lifecycle.
    if getattr(context, "scenario_name", None) != "full_loop_tunnel_vehicle":
        return False
    if not context.cfg.get("tunnel_vehicle_dynamic", {}).get("enabled", False):
        return False
    if context.clean_road_mode:
        raise ValueError("tunnel_vehicle_dynamic cannot run with clean_road_mode")
    handler = TunnelVehicleHandler()
    handler.validate(context)
    context.tunnel_vehicle_handler = handler
    return True
