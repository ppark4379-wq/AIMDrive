"""Tunnel pedestrian trials and seeded, sequential section mixes."""

import json
import math
import random
from pathlib import Path

from ...utils import safe_print as print
from ...utils.transform_utils import (
    interpolate_on_polyline,
    project_distance_on_polyline,
)
from .tunnel_dynamic_catalog import generate_tunnel_candidates, select_tunnel_mix
from .dynamic_obstacle import _route_intersections
from .pedestrian_timing import interaction_timing


TRIGGER_DEFAULTS = {
    "interaction_lead_sec": 1.5,
    "minimum_trigger_distance_m": 12.0,
    "maximum_trigger_distance_m": 60.0,
    "minimum_eta_delta_sec": 0.0,
    "maximum_eta_delta_sec": 3.0,
    "reaction_time_sec": 1.0,
    "comfortable_deceleration_mps2": 3.5,
    "stopping_safety_margin_m": 3.0,
    "minimum_ego_speed_mps": 2.0,
    "missed_trigger_tolerance_m": 1.0,
}


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
        context.tunnel_dynamic_trigger_cfg = self._trigger_config(context)
        policy = cfg.get("candidate_policy", "trial" if context.cfg.get(
            "drive_control_mode") == "morai_keyboard" else "approved")
        if policy not in ("trial", "approved"):
            raise ValueError("tunnel_dynamic.candidate_policy must be trial or approved")
        context.tunnel_dynamic_candidate_policy = policy
        context.tunnel_dynamic_plans = None
        context.tunnel_dynamic_random_mix = False
        if cfg.get("trial_candidate_id"):
            self._validate_candidate(context, cfg, zone)
            return
        if (cfg.get("random_mix") or {}).get("enabled", False):
            self._validate_random_mix(context, cfg, zone)
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
        tolerance = _number(cfg.get("arrival_tolerance_m"), "arrival_tolerance_m")
        timeout = _number(cfg.get("max_run_sec"), "max_run_sec")
        if not 0.0 < speed <= 5.0:
            raise RuntimeError("Tunnel dynamic speed must be positive")
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
            "arrival_tolerance_m": tolerance,
            "max_run_sec": timeout,
            "path_length_m": sum(segment_lengths),
            **self._interaction(context, points),
        }

    def _trigger_config(self, context):
        inherited = context.cfg.get("dynamic_obstacle") or {}
        overrides = context.cfg["tunnel_dynamic"].get("interaction_trigger") or {}
        if not isinstance(inherited, dict) or not isinstance(overrides, dict):
            raise ValueError("Tunnel interaction timing settings must be mappings")
        cfg = {}
        for key, default in TRIGGER_DEFAULTS.items():
            value = _number(overrides.get(key, inherited.get(key, default)),
                            f"interaction_trigger.{key}")
            if value < 0.0 or (key in ("comfortable_deceleration_mps2",
                                      "minimum_ego_speed_mps",
                                      "minimum_trigger_distance_m") and value == 0.0):
                raise ValueError(f"Invalid tunnel interaction_trigger.{key}")
            cfg[key] = value
        if (cfg["maximum_trigger_distance_m"] < cfg["minimum_trigger_distance_m"]
                or cfg["maximum_eta_delta_sec"] < cfg["minimum_eta_delta_sec"]):
            raise ValueError("Tunnel interaction trigger bounds must be ordered")
        return cfg

    def _interaction(self, context, points, anchor_s=None):
        if anchor_s is not None:
            a, b = points
            point = {axis: (a[axis] + b[axis]) / 2.0 for axis in ("x", "y", "z")}
            route_s = float(anchor_s)
            distance = _distance(a, b) / 2.0
            kind = "anchor"
        else:
            # Intersect the commanded straight segments with the actual route,
            # rather than assuming the midpoint crosses a curved centreline.
            route = context.route_points
            cumulative = [0.0]
            for a, b in zip(route[:-1], route[1:]):
                cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
            hits = []
            travelled = 0.0
            for a, b in zip(points[:-1], points[1:]):
                for x, y, index, ratio in _route_intersections(route, a, b, 0, len(route) - 1):
                    if any(math.hypot(x - hit[0]["x"], y - hit[0]["y"]) < 1e-5 for hit in hits):
                        continue
                    segment_length = _distance(a, b)
                    npc_ratio = math.hypot(x - a["x"], y - a["y"]) / segment_length
                    point = {"x": x, "y": y, "z": a["z"] + npc_ratio * (b["z"] - a["z"])}
                    route_s = cumulative[index] + ratio * (cumulative[index + 1] - cumulative[index])
                    hits.append((point, route_s, travelled + npc_ratio * segment_length))
                travelled += _distance(a, b)
            if len(hits) != 1:
                raise RuntimeError("Tunnel crossing must intersect Ego route exactly once")
            point, route_s, distance = hits[0]
            kind = "route_crossing"
        if not context.tunnel_entry_s < route_s < context.tunnel_exit_s or distance <= 0.0:
            raise RuntimeError("Tunnel interaction must lie inside the tunnel path")
        return {
            "interaction_s": route_s,
            "crossing_s": route_s,
            "interaction_point": point,
            "interaction_kind": kind,
            "start_to_interaction_m": distance,
            "trigger_s": None,  # Actual Ego s is recorded when the timing gates pass.
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
        if context.tunnel_dynamic_candidate_policy == "approved" and path["status"] != "approved":
            raise RuntimeError(f"Tunnel candidate requires trial policy: {path_id}")
        context.tunnel_dynamic_plan = self._candidate_plan(context, cfg, path)

    def _validate_random_mix(self, context, cfg, zone):
        mix = cfg["random_mix"]
        if isinstance(mix.get("events_per_section"), bool) or mix.get("events_per_section", 1) != 1:
            raise ValueError("random_mix.events_per_section must be 1")
        repeat = mix.get("allow_behavior_repeat", False)
        if not isinstance(repeat, bool):
            raise ValueError("random_mix.allow_behavior_repeat must be boolean")
        rng = random.Random(context.run_seed)
        candidates = generate_tunnel_candidates(
            context.route_points, context.tunnel_entry_s, context.tunnel_exit_s,
            zone.get("candidate_catalog"),
        )
        selected = select_tunnel_mix(
            candidates, rng, mix.get("behaviors"), allow_repeat=repeat,
            approved_only=context.tunnel_dynamic_candidate_policy == "approved",
        )
        context.tunnel_dynamic_plans = [self._candidate_plan(context, cfg, path)
                                        for path in selected]
        context.tunnel_dynamic_plan = context.tunnel_dynamic_plans[0]
        context.tunnel_dynamic_random_mix = True
        context.tunnel_dynamic_random = rng

    def _candidate_plan(self, context, cfg, path):
        path_id = path["id"]
        points = [path["start_xyz"], path["end_xyz"]]
        distance = _distance(*points)
        _direction(*points)
        speed = _number(cfg.get("speed_mps"), "speed_mps")
        tolerance = _number(cfg.get("arrival_tolerance_m"), "arrival_tolerance_m")
        timeout = _number(cfg.get("max_run_sec"), "max_run_sec")
        model = str(cfg.get("model", "")).strip()
        if not model or not 0.0 < speed <= 5.0:
            raise RuntimeError("Tunnel candidate model and speed must be valid")
        if not 0.0 < tolerance < distance / 2 or timeout <= distance / speed:
            raise RuntimeError("Tunnel candidate tolerance or timeout is outside path limits")
        anchor_s = path["anchor_s_m"] if path["behavior"].startswith("straight_") else None
        return {
            "path_id": path_id,
            "waypoints": points,
            "model": model,
            "speed_mps": speed,
            "speed_kmh": speed * 3.6,
            "arrival_tolerance_m": tolerance,
            "max_run_sec": timeout,
            "path_length_m": distance,
            "trial_candidate": True,
            "review_status": path["status"],
            "section_index": path["section_index"],
            "anchor_from_entry_m": path["anchor_from_entry_m"],
            "behavior": path["behavior"],
            "lane_index": path.get("lane_index"),
            **self._interaction(context, points, anchor_s),
        }

    def arm(self, context):
        plans = context.tunnel_dynamic_plans or [context.tunnel_dynamic_plan]
        context.tunnel_dynamic_plans = plans
        available = set(context.grpc.get_available_pedestrian_models())
        if context.tunnel_dynamic_random_mix:
            models = (context.cfg["tunnel_dynamic"]["random_mix"].get("models")
                      or context.cfg.get("dynamic_obstacle_models"))
            if (not isinstance(models, list) or not models
                    or any(not isinstance(model, str) or not model.strip() for model in models)):
                raise ValueError("Tunnel random mix needs a nonempty pedestrian model list")
            pool = sorted(set(models) & available)
            if not pool:
                raise RuntimeError("No configured tunnel pedestrian model is available in MORAI")
            context.tunnel_dynamic_model_pool = pool
            for plan in plans:
                plan["model"] = context.tunnel_dynamic_random.choice(pool)
        else:
            for plan in plans:
                if plan["model"] not in available:
                    raise RuntimeError(f"Tunnel pedestrian model unavailable: {plan['model']}")
        context.tunnel_dynamic_results = [self._event_result(context, plan) for plan in plans]
        context.tunnel_dynamic_report_path = None
        if context.cfg.get("profile_state_path"):
            context.tunnel_dynamic_report_path = (
                Path(context.cfg["profile_state_path"]).parent / "tunnel_dynamic_runs"
                / f"{context.run_instance_id}.json")
        if context.tunnel_dynamic_random_mix:
            schedule = ", ".join(f"{p['section_index']}:{p['path_id']}/{p['model']}"
                                 for p in plans)
            print(f"[TunnelDynamic] random mix seed={context.run_seed} schedule=[{schedule}]")
        print(f"[TunnelDynamic] controller={context.cfg.get('drive_control_mode', 'aimdrive_expert')} "
              f"candidate_policy={context.tunnel_dynamic_candidate_policy}")
        self._activate_event(context, 0)
        self._save_report(context)
        if context.tunnel_dynamic_report_path:
            print(f"[TunnelDynamic] report={context.tunnel_dynamic_report_path}")

    def _event_result(self, context, plan):
        return {
            "run_seed": context.run_seed,
            "path_id": plan["path_id"],
            "waypoints": plan["waypoints"],
            "model": plan["model"],
            "speed_mps": plan["speed_mps"],
            "trigger_s": plan["trigger_s"],
            "interaction_s": plan["interaction_s"],
            "interaction_point": plan["interaction_point"],
            "interaction_kind": plan["interaction_kind"],
            "start_to_interaction_m": plan["start_to_interaction_m"],
            "section_index": plan.get("section_index"),
            "behavior": plan.get("behavior"),
            "review_status": plan.get("review_status", "candidate"),
            "spawned": False,
            "completed": False,
        }

    def _activate_event(self, context, index):
        context.tunnel_dynamic_event_index = index
        plan = context.tunnel_dynamic_plans[index]
        context.tunnel_dynamic_plan = plan
        context.tunnel_dynamic_result = context.tunnel_dynamic_results[index]
        context.tunnel_dynamic_actor = None
        context.tunnel_dynamic_phase = "waiting"
        context.tunnel_dynamic_target_index = None
        context.tunnel_dynamic_state_misses = 0
        context.tunnel_dynamic_last_trigger_log_elapsed = float("-inf")
        print(f"[TunnelDynamic] armed seed={context.run_seed} "
              f"path={plan['path_id']} model={plan['model']} "
              f"trigger=eta interaction_s={plan['interaction_s']:.1f} "
              f"npc_eta={plan['start_to_interaction_m'] / plan['speed_mps']:.2f}s")
        if plan.get("trial_candidate"):
            start, end = plan["waypoints"]
            print(f"[TunnelDynamic] trial candidate={plan['path_id']} "
                  f"status={plan['review_status']} "
                  f"section={plan['section_index']} "
                  f"anchor_from_entry={plan['anchor_from_entry_m']:.1f}m "
                  f"start=({start['x']:.2f},{start['y']:.2f},{start['z']:.2f}) "
                  f"end=({end['x']:.2f},{end['y']:.2f},{end['z']:.2f})")

    def _save_report(self, context):
        path = context.tunnel_dynamic_report_path
        if path is None:
            return
        report = {
            "run_seed": context.run_seed,
            "run_instance_id": context.run_instance_id,
            "random_mix": context.tunnel_dynamic_random_mix,
            "model_pool": getattr(context, "tunnel_dynamic_model_pool", []),
            "trigger_policy": "interaction_eta",
            "candidate_policy": context.tunnel_dynamic_candidate_policy,
            "trigger_settings": context.tunnel_dynamic_trigger_cfg,
            "phase": context.tunnel_dynamic_phase,
            "completed": all(item["completed"] for item in context.tunnel_dynamic_results),
            "events": context.tunnel_dynamic_results,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def record_interruption(self, context, error):
        if not getattr(context, "tunnel_dynamic_results", None):
            return
        if not all(item["completed"] for item in context.tunnel_dynamic_results):
            context.tunnel_dynamic_result.setdefault(
                "failure_reason", str(error) or type(error).__name__)
            context.tunnel_dynamic_phase = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        self._save_report(context)

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
        finally:
            self._save_report(context)
        raise RuntimeError(str(reason))

    def _spawn(self, context, elapsed):
        plan = context.tunnel_dynamic_plan
        a, b = plan["waypoints"][:2]
        dx, dy = _direction(a, b)
        yaw = math.degrees(math.atan2(dy, dx))
        transform = context.grpc.make_transform(a["x"], a["y"], a["z"], yaw)
        label = f"tunnel_dynamic_{context.run_instance_id}_{context.tunnel_dynamic_event_index + 1}"
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
              f"path={plan['path_id']} model={plan['model']} "
              f"method={'native_straight' if plan.get('trial_candidate') else 'set_waypoint'} "
              f"yaw={yaw:.1f} speed={plan['speed_mps']:.2f}m/s")
        self._save_report(context)

    def update(self, context, elapsed, ego_state, current_s):
        phase = context.tunnel_dynamic_phase
        plan = context.tunnel_dynamic_plan
        if phase == "waiting":
            self._wait_for_interaction(context, elapsed, ego_state, current_s)
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
            if context.tunnel_dynamic_random_mix:
                self._fail(context, f"Implausible pedestrian jump {step_m:.2f}m in {dt:.2f}s")
            context.tunnel_dynamic_phase = "diagnostic_hold"
            context.tunnel_dynamic_result["failure_reason"] = "implausible_pedestrian_jump"
            context.tunnel_dynamic_result["last_position"] = {"x": x, "y": y}
            print(f"[TunnelDynamic] implausible jump: "
                  f"({previous['x']:.2f},{previous['y']:.2f})->({x:.2f},{y:.2f}) "
                  f"step={step_m:.2f}m dt={dt:.2f}s; actor held until shutdown")
            self._save_report(context)
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
        next_index = context.tunnel_dynamic_event_index + 1
        if next_index < len(context.tunnel_dynamic_plans):
            self._activate_event(context, next_index)
        self._save_report(context)

    def _wait_for_interaction(self, context, elapsed, ego_state, current_s):
        plan = context.tunnel_dynamic_plan
        cfg = context.tunnel_dynamic_trigger_cfg
        speed = _number(ego_state.get("speed_mps", ego_state.get("speed", 0.0)), "ego_speed_mps")
        timing = interaction_timing(
            cfg, plan["interaction_s"], plan["start_to_interaction_m"],
            plan["speed_mps"], {"speed_mps": speed}, current_s,
        )
        context.tunnel_dynamic_result["last_timing"] = timing
        if float(elapsed) - context.tunnel_dynamic_last_trigger_log_elapsed >= 1.0:
            print(f"[TunnelDynamic] waiting path={plan['path_id']} "
                  f"remaining={timing['remaining_m']:.2f}m "
                  f"ego_speed={timing['ego_speed_mps']:.2f}m/s "
                  f"ego_eta={timing['ego_eta_sec']:.2f}s "
                  f"npc_eta={timing['npc_eta_sec']:.2f}s "
                  f"eta_delta={timing['eta_delta_sec']:.2f}s "
                  f"trigger_distance={timing['trigger_distance_m']:.2f}m "
                  f"safe_stop={timing['safe_stop_distance_m']:.2f}m")
            context.tunnel_dynamic_last_trigger_log_elapsed = float(elapsed)
        if timing["remaining_m"] < -cfg["missed_trigger_tolerance_m"]:
            self._fail(context, f"missed_trigger: Ego passed tunnel interaction {plan['path_id']}")
        if timing["ego_speed_mps"] < cfg["minimum_ego_speed_mps"]:
            return
        if timing["remaining_m"] > timing["trigger_distance_m"]:
            return
        if timing["remaining_m"] < timing["safe_stop_distance_m"]:
            self._fail(context, f"unsafe_trigger_window: remaining={timing['remaining_m']:.2f}m "
                       f"safe_stop={timing['safe_stop_distance_m']:.2f}m")
        if timing["eta_delta_sec"] > cfg["maximum_eta_delta_sec"]:
            return
        if timing["eta_delta_sec"] < cfg["minimum_eta_delta_sec"]:
            self._fail(context, "unsafe_trigger_window: NPC would arrive after Ego")
        context.tunnel_dynamic_result.update(
            trigger_s=float(current_s), trigger=dict(timing, elapsed_sec=float(elapsed)),
        )
        print(f"[TunnelDynamic] TRIGGER path={plan['path_id']} "
              f"remaining={timing['remaining_m']:.2f}m "
              f"ego_eta={timing['ego_eta_sec']:.2f}s npc_eta={timing['npc_eta_sec']:.2f}s "
              f"safe_stop={timing['safe_stop_distance_m']:.2f}m")
        self._spawn(context, elapsed)


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
