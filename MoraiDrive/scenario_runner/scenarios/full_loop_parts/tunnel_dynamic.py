"""One fixed A -> B -> C tunnel pedestrian for closed-loop development.

The path is chosen by the scenario, while the gRPC adapter only executes
spawn, direction, speed, and destroy commands. No random choices live here.
"""

import math

from ...utils import safe_print as print
from ...utils.transform_utils import (
    interpolate_on_polyline,
    project_distance_on_polyline,
)
from .tunnel_dynamic_catalog import generate_tunnel_candidates


def _number(value, name):
    if isinstance(value, bool):
        raise RuntimeError(f"tunnel_dynamic.{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"tunnel_dynamic.{name} must be finite") from error
    if not math.isfinite(result):
        raise RuntimeError(f"tunnel_dynamic.{name} must be finite")
    return result


def _point(raw, name):
    if not isinstance(raw, dict):
        raise RuntimeError(f"tunnel_dynamic waypoint {name} must be an XYZ mapping")
    return {axis: _number(raw.get(axis), f"{name}.{axis}") for axis in ("x", "y", "z")}


def _distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _direction(a, b):
    distance = _distance(a, b)
    if distance < 1.0:
        raise RuntimeError("Tunnel pedestrian waypoints must be at least 1 m apart")
    return ((b["x"] - a["x"]) / distance,
            (b["y"] - a["y"]) / distance)


class TunnelDynamicHandler:
    """Stateless handler; all run state belongs to the scenario context."""

    def validate(self, context):
        cfg = context.cfg["tunnel_dynamic"]
        zone = context.route_definition["zones"]["tunnel"]
        if cfg.get("trial_candidate_id"):
            self._validate_candidate(context, cfg, zone)
            return
        path_id = str(cfg.get("path_id", "")).strip()
        path = (zone.get("dynamic_paths") or {}).get(path_id)
        if not isinstance(path, dict):
            raise RuntimeError(f"Unknown tunnel_dynamic path_id: {path_id!r}")
        raw_points = path.get("waypoints")
        if not isinstance(raw_points, list) or len(raw_points) != 3:
            raise RuntimeError("Tunnel dynamic path must contain exactly A, B, C")
        points = [_point(raw, name) for raw, name in zip(raw_points, "ABC")]
        segment_lengths = [_distance(a, b) for a, b in zip(points[:-1], points[1:])]
        for a, b in zip(points[:-1], points[1:]):
            _direction(a, b)

        speed = _number(cfg.get("speed_mps"), "speed_mps")
        trigger_before = _number(
            cfg.get("spawn_trigger_before_crossing_m"),
            "spawn_trigger_before_crossing_m",
        )
        tolerance = _number(cfg.get("arrival_tolerance_m"), "arrival_tolerance_m")
        timeout = _number(cfg.get("max_run_sec"), "max_run_sec")
        if not 0.0 < speed <= 5.0 or trigger_before <= 0.0:
            raise RuntimeError("Tunnel dynamic speed and trigger distance must be positive")
        if not 0.0 < tolerance < min(segment_lengths) / 2:
            raise RuntimeError("Tunnel dynamic arrival tolerance is outside path limits")
        if timeout <= sum(segment_lengths) / speed:
            raise RuntimeError("Tunnel dynamic timeout is shorter than planned travel")
        model = str(cfg.get("model", "")).strip()
        if not model:
            raise RuntimeError("tunnel_dynamic.model is required")

        index = int(path["route_path_index"])
        full_points = getattr(getattr(context, "mission_route", None),
                              "full_points", context.route_points)
        if not 1 <= index < len(full_points) - 1:
            raise RuntimeError("Tunnel dynamic route anchor is outside official path")
        anchor = full_points[index]
        before, after = full_points[index - 1], full_points[index + 1]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            raise RuntimeError("Tunnel dynamic route anchor has no tangent")
        tx, ty = dx / length, dy / length
        lateral = [tx * (p["y"] - anchor[1]) - ty * (p["x"] - anchor[0])
                   for p in points]
        if not (lateral[0] > 0.5 and lateral[1] > 0.5 and lateral[2] < -0.5):
            raise RuntimeError("Tunnel dynamic A/B/C must cross the official route")

        crossing_s = context.path_s_at_index(index)
        if not context.tunnel_entry_s + 10.0 < crossing_s < context.tunnel_exit_s - 10.0:
            raise RuntimeError("Tunnel dynamic crossing must be inside tunnel")
        trigger_s = crossing_s - trigger_before
        if trigger_s < 0.0:
            raise RuntimeError("Tunnel dynamic trigger precedes scenario START")
        for name, point in zip("ABC", points):
            s = project_distance_on_polyline(
                context.route_points, point["x"], point["y"])
            route_x, route_y, route_z, _ = interpolate_on_polyline(
                context.route_points, s)
            if (not context.tunnel_entry_s + 5.0 < s < context.tunnel_exit_s - 5.0
                    or math.hypot(point["x"] - route_x, point["y"] - route_y) > 5.0
                    or abs(point["z"] - route_z) > 1.0):
                raise RuntimeError(f"Tunnel dynamic waypoint {name} is outside route bounds")

        context.tunnel_dynamic_plan = {
            "path_id": path_id,
            "waypoints": points,
            "model": model,
            "speed_mps": speed,
            "speed_kmh": speed * 3.6,
            "trigger_s": trigger_s,
            "crossing_s": crossing_s,
            "arrival_tolerance_m": tolerance,
            "max_run_sec": timeout,
            "path_length_m": sum(segment_lengths),
        }

    def _validate_candidate(self, context, cfg, zone):
        path_id = str(cfg["trial_candidate_id"]).strip()
        candidates = generate_tunnel_candidates(
            context.route_points, context.tunnel_entry_s, context.tunnel_exit_s,
            zone.get("candidate_catalog"),
        )
        path = next((item for item in candidates if item["id"] == path_id), None)
        if path is None:
            raise RuntimeError(f"Unknown tunnel candidate ID: {path_id!r}")
        if path["status"] == "disabled":
            raise RuntimeError(f"Tunnel candidate is disabled: {path_id}")
        points = [path["start_xyz"], path["end_xyz"]]
        distance = _distance(*points)
        _direction(*points)
        speed = _number(cfg.get("speed_mps"), "speed_mps")
        tolerance = _number(cfg.get("arrival_tolerance_m"), "arrival_tolerance_m")
        timeout = _number(cfg.get("max_run_sec"), "max_run_sec")
        trigger_before = _number(
            cfg.get("trial_trigger_before_start_m"), "trial_trigger_before_start_m")
        model = str(cfg.get("model", "")).strip()
        if not model or not 0.0 < speed <= 5.0:
            raise RuntimeError("Tunnel candidate model and speed must be valid")
        if not 0.0 < tolerance < distance / 2 or timeout <= distance / speed:
            raise RuntimeError("Tunnel candidate tolerance or timeout is outside path limits")
        start_s = path["start_sd"]["s_m"]
        trigger_s = start_s - trigger_before
        if trigger_before <= 0 or trigger_s < 0:
            raise RuntimeError("Tunnel candidate trigger precedes scenario START")
        context.tunnel_dynamic_plan = {
            "path_id": path_id,
            "waypoints": points,
            "model": model,
            "speed_mps": speed,
            "speed_kmh": speed * 3.6,
            "trigger_s": trigger_s,
            "crossing_s": start_s,
            "arrival_tolerance_m": tolerance,
            "max_run_sec": timeout,
            "path_length_m": distance,
            "trial_candidate": True,
            "review_status": path["status"],
        }

    def arm(self, context):
        plan = context.tunnel_dynamic_plan
        if plan["model"] not in context.grpc.get_available_pedestrian_models():
            raise RuntimeError(f"Tunnel pedestrian model unavailable: {plan['model']}")
        context.tunnel_dynamic_actor = None
        context.tunnel_dynamic_phase = "waiting"
        context.tunnel_dynamic_target_index = None
        context.tunnel_dynamic_state_misses = 0
        context.tunnel_dynamic_result = {
            "run_seed": context.run_seed,
            "path_id": plan["path_id"],
            "waypoints": plan["waypoints"],
            "model": plan["model"],
            "speed_mps": plan["speed_mps"],
            "spawned": False,
            "completed": False,
        }
        print(f"[TunnelDynamic] armed seed={context.run_seed} "
              f"path={plan['path_id']} model={plan['model']} "
              f"trigger_s={plan['trigger_s']:.1f}")
        if plan.get("trial_candidate"):
            start, end = plan["waypoints"]
            print(f"[TunnelDynamic] trial candidate={plan['path_id']} "
                  f"status={plan['review_status']} "
                  f"start=({start['x']:.2f},{start['y']:.2f},{start['z']:.2f}) "
                  f"end=({end['x']:.2f},{end['y']:.2f},{end['z']:.2f})")

    def _command(self, context, target_index):
        plan = context.tunnel_dynamic_plan
        target = plan["waypoints"][target_index]
        if not context.grpc.set_pedestrian_waypoint(
                context.tunnel_dynamic_actor,
                target["x"], target["y"], target["z"], plan["speed_mps"]):
            raise RuntimeError(f"Tunnel pedestrian waypoint failed: {target_index}")
        context.tunnel_dynamic_target_index = target_index

    def _destroy(self, context):
        actor = context.tunnel_dynamic_actor
        if actor is None:
            return
        result = actor.destroy()
        if result is False:
            raise RuntimeError("Failed to destroy tunnel pedestrian")
        if actor in context.scenario_actors:
            context.scenario_actors.remove(actor)
        context.tunnel_dynamic_actor = None

    def _fail(self, context, reason):
        context.tunnel_dynamic_phase = "failed"
        context.tunnel_dynamic_result["failure_reason"] = str(reason)
        try:
            self._destroy(context)
        except Exception as error:
            raise RuntimeError(f"{reason}; cleanup failed: {error}") from error
        raise RuntimeError(str(reason))

    def _spawn(self, context, elapsed):
        plan = context.tunnel_dynamic_plan
        a, b = plan["waypoints"][:2]
        dx, dy = _direction(a, b)
        yaw = math.degrees(math.atan2(dy, dx))
        transform = context.grpc.make_transform(a["x"], a["y"], a["z"], yaw)
        label = f"tunnel_dynamic_{context.run_instance_id}"
        # Straight candidates use the same native yaw/velocity movement as
        # dynamic_obstacle. In the live tunnel trial, both waypoint APIs
        # reported the destination immediately instead of observed traversal.
        actor = context.grpc.spawn_pedestrian(
            transform=transform, model_name=plan["model"], label=label,
            velocity=plan["speed_kmh"], active_dist=0.0,
            move_dist=plan["path_length_m"] + 2.0, start_action=True,
        )
        if actor is None:
            self._fail(context, "Failed to spawn tunnel pedestrian")
        context.tunnel_dynamic_actor = actor
        context.scenario_actors.append(actor)
        try:
            actor_id = str(actor.get_id() or "").strip()
        except Exception as error:
            self._fail(context, f"Tunnel pedestrian ID unavailable: {error}")
        if not actor_id:
            self._fail(context, "Tunnel pedestrian has no actor ID")
        context.tunnel_dynamic_result.update(
            spawned=True, actor_id=actor_id, spawn_elapsed_sec=float(elapsed))
        context.tunnel_dynamic_spawn_elapsed = float(elapsed)
        context.tunnel_dynamic_last_position = {"x": a["x"], "y": a["y"]}
        context.tunnel_dynamic_last_observed_elapsed = float(elapsed)
        context.tunnel_dynamic_last_log_elapsed = float("-inf")
        if plan.get("trial_candidate"):
            context.tunnel_dynamic_target_index = 1
        else:
            try:
                self._command(context, 1)
            except RuntimeError as error:
                self._fail(context, error)
        context.tunnel_dynamic_phase = "moving"
        print(f"[TunnelDynamic] spawned actor={actor_id} A->B "
              f"method={'native_straight' if plan.get('trial_candidate') else 'set_waypoint'} "
              f"yaw={yaw:.1f} speed={plan['speed_mps']:.2f}m/s")

    def update(self, context, elapsed, ego_state, current_s):
        phase = context.tunnel_dynamic_phase
        plan = context.tunnel_dynamic_plan
        if phase == "waiting":
            if current_s > plan["crossing_s"]:
                self._fail(context, "Ego passed tunnel crossing before spawn")
            if current_s >= plan["trigger_s"]:
                self._spawn(context, elapsed)
            return
        if phase in ("completed", "disabled", "diagnostic_hold"):
            return
        if phase == "failed":
            raise RuntimeError("Tunnel dynamic event already failed")
        if float(elapsed) - context.tunnel_dynamic_spawn_elapsed > plan["max_run_sec"]:
            self._fail(context, "Tunnel pedestrian timed out")
        actor = context.tunnel_dynamic_actor
        try:
            state = actor.get_actor_state()
            x = float(state.transform.location.x)
            y = float(state.transform.location.y)
        except Exception as error:
            context.tunnel_dynamic_state_misses += 1
            if context.tunnel_dynamic_state_misses >= 3:
                self._fail(context, f"Tunnel pedestrian state unavailable: {error}")
            return
        context.tunnel_dynamic_state_misses = 0
        if not math.isfinite(x) or not math.isfinite(y):
            self._fail(context, "Tunnel pedestrian position is not finite")
        previous = context.tunnel_dynamic_last_position
        dt = max(0.0, float(elapsed) - context.tunnel_dynamic_last_observed_elapsed)
        step_m = math.hypot(x - previous["x"], y - previous["y"])
        max_plausible_step_m = max(5.0, plan["speed_mps"] * 3.0) * dt + 1.0
        if plan.get("trial_candidate") and step_m > max_plausible_step_m:
            context.tunnel_dynamic_phase = "diagnostic_hold"
            context.tunnel_dynamic_result["failure_reason"] = "implausible_pedestrian_jump"
            context.tunnel_dynamic_result["last_position"] = {"x": x, "y": y}
            print(f"[TunnelDynamic] implausible jump: "
                  f"({previous['x']:.2f},{previous['y']:.2f})->({x:.2f},{y:.2f}) "
                  f"step={step_m:.2f}m dt={dt:.2f}s; actor held until shutdown")
            return
        context.tunnel_dynamic_last_position = {"x": x, "y": y}
        context.tunnel_dynamic_last_observed_elapsed = float(elapsed)
        target_index = context.tunnel_dynamic_target_index
        target = plan["waypoints"][target_index]
        remaining = math.hypot(x - target["x"], y - target["y"])
        context.tunnel_dynamic_result["last_position"] = {"x": x, "y": y}
        if float(elapsed) - context.tunnel_dynamic_last_log_elapsed >= 1.0:
            print(f"[TunnelDynamic] actor=({x:.2f},{y:.2f}) "
                  f"remaining={remaining:.2f}m elapsed_since_spawn="
                  f"{float(elapsed) - context.tunnel_dynamic_spawn_elapsed:.2f}s")
            context.tunnel_dynamic_last_log_elapsed = float(elapsed)
        if remaining > plan["arrival_tolerance_m"]:
            return
        if target_index < len(plan["waypoints"]) - 1:
            try:
                self._command(context, target_index + 1)
            except RuntimeError as error:
                self._fail(context, error)
            print("[TunnelDynamic] reached B; B->C")
            return
        if not context.grpc.control_pedestrian(actor, 0.0, 0.0, 0.0):
            self._fail(context, "Failed to stop tunnel pedestrian at C")
        try:
            self._destroy(context)
        except RuntimeError as error:
            self._fail(context, error)
        context.tunnel_dynamic_phase = "completed"
        context.tunnel_dynamic_result.update(
            completed=True, completed_elapsed_sec=float(elapsed))
        print(f"[TunnelDynamic] reached final waypoint; pedestrian removed "
              f"path={plan['path_id']}")


TUNNEL_DYNAMIC_HANDLER = TunnelDynamicHandler()


def prepare_tunnel_dynamic(context, handler=None):
    """Validate an enabled tunnel crossing before starting the MORAI world."""
    context.tunnel_dynamic_handler = None
    mission_mode = getattr(context, "mission_mode", None)
    if mission_mode is not None and not mission_mode.zones.tunnel:
        return False
    if not (context.cfg.get("tunnel_dynamic") or {}).get("enabled", False):
        return False
    if getattr(context, "clean_road_mode", False):
        raise ValueError("tunnel_dynamic cannot run with clean_road_mode")
    if handler is None:
        handler = TUNNEL_DYNAMIC_HANDLER
    if handler is None or not all(
        callable(getattr(handler, method, None))
        for method in ("validate", "arm", "update")
    ):
        raise RuntimeError(
            "Tunnel dynamic obstacle has no validated placement and lifecycle handler"
        )
    handler.validate(context)
    context.tunnel_dynamic_handler = handler
    return True


def arm_tunnel_dynamic(context):
    """Arm the tunnel crossing after the simulator world starts."""
    context.tunnel_event_phase = "disabled"
    context.tunnel_event_result = {"disabled_reason": "tunnel_dynamic"}
    context.tunnel_dynamic_handler.arm(context)


def update_tunnel_dynamic(context, elapsed, ego_state, current_s):
    """Advance the tunnel crossing on each driving tick."""
    context.tunnel_dynamic_handler.update(context, elapsed, ego_state, current_s)
