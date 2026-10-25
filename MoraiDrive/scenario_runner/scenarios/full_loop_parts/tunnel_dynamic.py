"""Tunnel pedestrian trials and seeded section mixes with bounded overlap."""

import json
import math
import random
import time
from pathlib import Path

from ...utils import safe_print as print
from ...utils.transform_utils import (
    interpolate_on_polyline,
    project_distance_on_polyline,
)
from .tunnel_dynamic_catalog import generate_tunnel_candidates, select_tunnel_mix
from .dynamic_obstacle import _route_intersections
from .pedestrian_timing import interaction_timing
from . import tunnel_dynamic_timing as trajectory_timing
from .tunnel_static_parts.screening import POLICY as TUNNEL_GEOMETRY_POLICY


TRIGGER_DEFAULTS = {
    "interaction_lead_sec": 0.3,
    "minimum_trigger_distance_m": 0.0,
    "maximum_trigger_distance_m": 60.0,
    "minimum_eta_delta_sec": 0.0,
    "maximum_eta_delta_sec": 3.0,
    "reaction_time_sec": 1.0,
    "comfortable_deceleration_mps2": 3.5,
    "stopping_safety_margin_m": 3.0,
    "minimum_ego_speed_mps": 2.0,
    "missed_trigger_tolerance_m": 1.0,
}

# The native spawn readback probe observed UNKNOWN/empty inventories immediately
# after a successful ACK, followed by a valid actor at 0.11--0.15 wall seconds.
# Bound initialization in host time rather than treating three fast ticks as
# proof of a lost moving actor. This never verifies an unobserved spawn.
INITIAL_STATE_TIMEOUT_WALL_SEC = 1.0

# An oncoming pedestrian keeps walking toward its endpoint after Ego brakes.
# Preserve the original lead for these conflicting paths; the shorter crossing
# lead leaves too little space between a stopped Ego and that endpoint.
ONCOMING_INTERACTION_LEAD_SEC = 1.5


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


class _TunnelEventContext:
    """Own one pedestrian's lifecycle; share the original scenario and Expert."""

    LOCAL_FIELDS = frozenset("tunnel_dynamic_" + name for name in (
        "event_index", "plan", "result", "actor", "phase", "target_index",
        "state_misses", "last_trigger_log_elapsed", "spawn_ack_monotonic",
        "spawn_elapsed", "last_position", "last_observed_elapsed", "last_log_elapsed",
    ))

    def __init__(self, root):
        object.__setattr__(self, "root", root)

    def __getattr__(self, name):
        if name in self.LOCAL_FIELDS:
            raise AttributeError(name)
        return getattr(self.root, name)

    def __setattr__(self, name, value):
        if name in self.LOCAL_FIELDS:
            object.__setattr__(self, name, value)
        else:
            setattr(self.root, name, value)


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
            # Short tunnel paths must not inherit the general zone's early
            # 1.5 s lead / 12 m floor. Keep its stopping-distance safeguards.
            fallback = (default if key in ("interaction_lead_sec",
                                          "minimum_trigger_distance_m")
                        else inherited.get(key, default))
            value = _number(overrides.get(key, fallback),
                            f"interaction_trigger.{key}")
            if value < 0.0 or (key in ("comfortable_deceleration_mps2",
                                      "minimum_ego_speed_mps") and value == 0.0):
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
        context.tunnel_dynamic_motion_samples = [
            trajectory_timing.motion_samples(context.route_points, plan) for plan in plans]
        context.tunnel_dynamic_ego_history = []
        context.tunnel_dynamic_peak_active = 0
        route_length = getattr(context, "route_length_m", None)
        if route_length is None:
            route_length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                               for a, b in zip(context.route_points[:-1], context.route_points[1:]))
        context.tunnel_dynamic_goal_s = trajectory_timing.goal_progress_s(route_length, context.cfg)
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
        context.tunnel_dynamic_events = []
        if len(plans) > 1:
            for index in range(len(plans)):
                event = _TunnelEventContext(context)
                self._activate_event(event, index)
                context.tunnel_dynamic_events.append(event)
            self._sync_events(context)
        else:
            self._activate_event(context, 0)
        self._save_report(context)
        if context.tunnel_dynamic_report_path:
            print(f"[TunnelDynamic] report={context.tunnel_dynamic_report_path}")

    def _event_result(self, context, plan):
        # Classify paths for spawn timing and response telemetry only.
        # Ego response does not determine whether an event completes.
        corridor = 0.5 * float((context.cfg.get("dynamic_obstacle") or {}).get(
            "ego_body_width_m", 2.17)) + 0.15
        relevant = False
        for a, b in zip(plan["waypoints"][:-1], plan["waypoints"][1:]):
            count = max(1, int(math.ceil(_distance(a, b) / 0.25)))
            for index in range(count + 1):
                ratio = index / count
                x, y = a["x"] + ratio * (b["x"] - a["x"]), a["y"] + ratio * (b["y"] - a["y"])
                s = project_distance_on_polyline(context.route_points, x, y)
                rx, ry, _, _ = interpolate_on_polyline(context.route_points, s)
                if math.hypot(x - rx, y - ry) <= corridor:
                    relevant = True
                    break
            if relevant:
                break
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
            # Retain the report key used by spawn timing; it is not a
            # requirement for scenario completion.
            "interaction_required": relevant,
            "interaction_verified": False,
            "npc_motion_observed": False,
            "ego_response": {"observations": 0, "baseline_speed_mps": None,
                             "minimum_speed_mps": None, "maximum_drop_mps": 0.0,
                             "slowed": False, "stopped": False},
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
        if isinstance(context, _TunnelEventContext):
            context = context.root
        self._sync_events(context)
        path = context.tunnel_dynamic_report_path
        if path is None:
            return
        event_contexts = getattr(context, "tunnel_dynamic_events", []) or [context]
        report = {
            "run_seed": context.run_seed,
            "run_instance_id": context.run_instance_id,
            "random_mix": context.tunnel_dynamic_random_mix,
            "model_pool": getattr(context, "tunnel_dynamic_model_pool", []),
            "trigger_policy": "tunnel_interaction_eta",
            "candidate_policy": context.tunnel_dynamic_candidate_policy,
            "trigger_settings": context.tunnel_dynamic_trigger_cfg,
            "phase": context.tunnel_dynamic_phase,
            "completed": all(item["completed"] for item in context.tunnel_dynamic_results),
            "ego_response_affects_completion": False,
            "overlap_policy": "only_when_waiting_would_miss_next_start",
            "active_event_indices": [event.tunnel_dynamic_event_index
                for event in event_contexts
                if event.tunnel_dynamic_actor is not None],
            "peak_active_npcs": context.tunnel_dynamic_peak_active,
            "interaction_verified": any(item["interaction_required"]
                                        for item in context.tunnel_dynamic_results)
                and all(item["interaction_verified"] for item in context.tunnel_dynamic_results
                        if item["interaction_required"]),
            "events": context.tunnel_dynamic_results,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def record_interruption(self, context, error):
        if not getattr(context, "tunnel_dynamic_results", None):
            return
        events = getattr(context, "tunnel_dynamic_events", [])
        if events:
            for event in events:
                if event.tunnel_dynamic_phase != "completed":
                    event.tunnel_dynamic_result.setdefault(
                        "failure_reason", str(error) or type(error).__name__)
                    event.tunnel_dynamic_phase = (
                        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
            self._cleanup_events(context)
            self._save_report(context)
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
        previous = context.tunnel_dynamic_last_position
        yaw = math.degrees(math.atan2(target["y"] - previous["y"],
                                      target["x"] - previous["x"]))
        # SetPedestrianWaypoint(is_create=True) teleports to the destination on
        # the live server. ControlPedestrian instead walks continuously: its
        # direction is Euler rotation (yaw in z), not an XY unit vector.
        # Live readback also confirms km/h here, just like native spawning,
        # despite the SDK's m/s comment. Preserve the configured walking speed.
        rotation = context.grpc.make_transform(0.0, 0.0, 0.0, yaw).rotation
        if not context.tunnel_dynamic_actor.control(rotation, plan["speed_kmh"]):
            raise RuntimeError(f"Tunnel pedestrian heading command failed: {target_index}")
        context.tunnel_dynamic_target_index = target_index

    def _destroy(self, context):
        actor = context.tunnel_dynamic_actor
        if actor is None:
            return
        actor_id = context.tunnel_dynamic_result.get("actor_id")
        result = None
        errors = []
        try:
            result = actor.destroy()
        except Exception as error:
            errors.append(f"DestroyActor: {error}")
        if not actor_id:
            # A returned spawn handle must still be cleaned up when get_id
            # failed. Keep it owned for outer teardown: without an exact ID,
            # neither a destroy ACK nor empty inventory verifies removal.
            errors.append("Actor ID unavailable; deletion cannot be verified")
            context.tunnel_dynamic_result["removal_observation"] = {
                "actor_id": None, "destroy_result": result,
                "absence_confirmations": 0, "polls": [],
                "errors": errors, "verified": False,
            }
            raise RuntimeError("Tunnel pedestrian removal unverified: actor ID unavailable")
        # Native Actor.destroy returns None and can hide a server RPC error.
        # Do not spawn the next event with a reused ID until deletion is
        # observed. Use the normal dynamic handler's bounded poll interval,
        # with two independent absence checks even when the ACK says success.
        absent = 0
        polls = []
        for attempt in range(3):
            try:
                states = context.grpc.get_all_pedestrian_states(strict=True)
                matches = [state for state in states
                           if str(state.actor_info.id.value).strip() == actor_id]
                absent = absent + 1 if not matches else 0
                polls.append({"poll": attempt + 1, "exact_id_matches": len(matches)})
            except Exception as error:
                absent = 0
                errors.append(f"GetAllActorsState: {error}")
                polls.append({"poll": attempt + 1, "error": str(error)})
            if absent >= 2:
                break
            if attempt < 2:
                time.sleep(0.1)
        context.tunnel_dynamic_result["removal_observation"] = {
            "actor_id": actor_id, "destroy_result": result,
            "absence_confirmations": absent, "polls": polls,
            "errors": errors, "verified": absent >= 2,
        }
        if absent < 2:
            raise RuntimeError(f"Tunnel pedestrian removal unverified actor_id={actor_id}: "
                               f"absence_checks={absent}, errors={errors}")
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
        try:
            actor = context.grpc.spawn_pedestrian(
                transform=transform, model_name=plan["model"], label=label,
                velocity=plan["speed_kmh"], active_dist=0.0,
                move_dist=plan["path_length_m"] + 2.0, start_action=True,
            )
        except Exception as error:
            self._fail(context, f"Tunnel pedestrian spawn RPC failed: {error}")
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
            spawned=True, actor_id=actor_id, spawn_elapsed_sec=float(elapsed),
            spawn_state_verified=False)
        others = [event for event in getattr(context, "tunnel_dynamic_events", [])
                  if event is not context and event.tunnel_dynamic_actor is not None]
        other_ids = [event.tunnel_dynamic_result.get("actor_id") for event in others]
        context.tunnel_dynamic_result["overlapping_actor_ids_at_spawn"] = other_ids
        if actor_id in other_ids:
            self._fail(context, f"Tunnel pedestrian duplicate active actor ID: {actor_id}")
        context.tunnel_dynamic_peak_active = max(
            context.tunnel_dynamic_peak_active, len(others) + 1)
        context.tunnel_dynamic_spawn_ack_monotonic = time.monotonic()
        context.tunnel_dynamic_result.update(
            spawn_ack_monotonic_sec=context.tunnel_dynamic_spawn_ack_monotonic,
            initial_state_policy={"clock": "host_monotonic",
                                  "timeout_wall_sec": INITIAL_STATE_TIMEOUT_WALL_SEC},
            initial_state_read_misses=0, initial_state_read_history=[],
        )
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
        # A successful spawn response is not an observed actor. The first
        # actual, plausible pose below is required before confirming motion.
        context.tunnel_dynamic_phase = "spawned_unobserved"
        print(f"[TunnelDynamic] spawned actor={actor_id} A->B "
              f"path={plan['path_id']} model={plan['model']} "
              f"method={'native_straight' if plan.get('trial_candidate') else 'native_heading'} "
              f"yaw={yaw:.1f} speed={plan['speed_mps']:.2f}m/s")
        self._save_report(context)

    def _read_state(self, context, actor):
        """Read this spawned pedestrian, never an old pose or a nearby actor.

        MORAI can return UNKNOWN for GetActorState while GetAllActorsState
        still exposes the actor. The SDK converts that RPC error to None.
        Query a fresh, client-scoped inventory only on a direct read failure;
        require one exact ID and pedestrian type, as the normal dynamic
        obstacle handler does. Strict mode keeps RPC failure distinct from an
        observed empty inventory. Neither is a completed pedestrian event.
        """
        errors = []
        try:
            state = actor.get_actor_state()
            if state is not None:
                return state, "GetActorState", errors
            errors.append("GetActorState returned no state")
        except Exception as error:
            errors.append(f"GetActorState: {error}")
        actor_id = context.tunnel_dynamic_result["actor_id"]
        try:
            states = context.grpc.get_all_pedestrian_states(strict=True)
            matches = [state for state in states
                       if str(state.actor_info.id.value).strip() == actor_id
                       and state.actor_info.object_type == 2]
            if len(matches) == 1:
                return matches[0], "GetAllActorsState", errors
            errors.append(f"GetAllActorsState expected one exact pedestrian {actor_id}, "
                          f"found {len(matches)}")
        except Exception as error:
            errors.append(f"GetAllActorsState: {error}")
        raise RuntimeError("; ".join(errors))

    def _observe_response(self, context, elapsed, ego_state, current_s, x, y):
        result = context.tunnel_dynamic_result
        start = context.tunnel_dynamic_plan["waypoints"][0]
        result["npc_motion_observed"] |= math.hypot(x - start["x"], y - start["y"]) >= 0.5
        raw_speed = ego_state.get("speed_mps", ego_state.get("speed"))
        # Missing telemetry is not a stopped Ego. Only measured responses near
        # the moving actor count; no Ego commands or Expert overrides are used.
        if raw_speed is None:
            return
        speed = _number(raw_speed, "ego_speed_mps")
        npc_s = project_distance_on_polyline(context.route_points, x, y)
        distance = npc_s - float(current_s)
        if not result["npc_motion_observed"] or not -3.0 <= distance <= 30.0:
            return
        response = result["ego_response"]
        baseline = response["baseline_speed_mps"]
        if baseline is None:
            return
        response["observations"] += 1
        minimum = response["minimum_speed_mps"]
        response["minimum_speed_mps"] = speed if minimum is None else min(minimum, speed)
        response["maximum_drop_mps"] = max(response["maximum_drop_mps"], baseline - speed)
        response["slowed"] |= baseline - speed >= 1.0
        response["stopped"] |= speed <= 0.5
        result["interaction_verified"] = bool(result["interaction_required"]
            and (response["slowed"] or response["stopped"]))
        if result["interaction_verified"]:
            response.setdefault("first_response_elapsed_sec", float(elapsed))
        # Persist observations even if the run is interrupted before arrival.
        self._save_report(context)

    def update(self, context, elapsed, ego_state, current_s):
        self._record_ego_motion(context, elapsed, ego_state)
        events = getattr(context, "tunnel_dynamic_events", [])
        if not events:
            return self._update_event(context, elapsed, ego_state, current_s)
        try:
            for event in events:
                if event.tunnel_dynamic_phase not in ("waiting", "completed"):
                    self._update_event(event, elapsed, ego_state, current_s)
            # Preserve section order and at most one new spawn per tick. Each
            # pending event must pass its own live approach gates, even when
            # the preceding pedestrian is still walking.
            pending = next((event for event in events
                            if event.tunnel_dynamic_phase == "waiting"), None)
            if pending is not None:
                self._wait_for_interaction(pending, elapsed, ego_state, current_s)
        except BaseException:
            self._cleanup_events(context)
            self._save_report(context)
            raise
        finally:
            self._sync_events(context)

    def _sync_events(self, context):
        events = getattr(context, "tunnel_dynamic_events", [])
        if not events:
            return
        focus = next((event for event in events
                      if event.tunnel_dynamic_phase in ("failed", "interrupted")), None)
        if focus is None:
            focus = next((event for event in events
                          if event.tunnel_dynamic_phase != "completed"), events[-1])
        # Retain the public single-event view used by scenario status and
        # existing callers. These aliases never own another actor's state.
        for name in _TunnelEventContext.LOCAL_FIELDS:
            if name in focus.__dict__:
                setattr(context, name, focus.__dict__[name])
            elif name in context.__dict__:
                delattr(context, name)
        for event in events:
            event.tunnel_dynamic_result["phase"] = event.tunnel_dynamic_phase

    def _cleanup_events(self, context):
        for event in context.tunnel_dynamic_events:
            if event.tunnel_dynamic_actor is not None:
                try:
                    self._destroy(event)
                except Exception as error:
                    event.tunnel_dynamic_result["cleanup_error"] = str(error)
                    # Keep unverified handles in scenario_actors for the
                    # existing outer teardown, and still clean other actors.

    def _update_event(self, context, elapsed, ego_state, current_s):
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
        initializing = phase == "spawned_unobserved"
        if initializing:
            wait_wall = time.monotonic() - context.tunnel_dynamic_spawn_ack_monotonic
            context.tunnel_dynamic_result["initial_state_wait_wall_seconds"] = wait_wall
            if wait_wall >= INITIAL_STATE_TIMEOUT_WALL_SEC:
                self._fail(context, "Tunnel pedestrian initial state unavailable after "
                           f"{INITIAL_STATE_TIMEOUT_WALL_SEC:.1f}s wall deadline")
            # Do not allow the initialization deadline to carry an absent
            # obstacle into an unsafe or already missed interaction window.
            timing = interaction_timing(
                context.tunnel_dynamic_trigger_cfg, plan["interaction_s"],
                plan["start_to_interaction_m"], plan["speed_mps"], ego_state, current_s)
            context.tunnel_dynamic_result["initial_state_interaction_timing"] = timing
        actor = context.tunnel_dynamic_actor
        try:
            state, source, read_errors = self._read_state(context, actor)
            x = float(state.transform.location.x)
            y = float(state.transform.location.y)
        except Exception as error:
            context.tunnel_dynamic_state_misses += 1
            context.tunnel_dynamic_result.update(
                state_read_failures=context.tunnel_dynamic_state_misses,
                state_read_error=str(error),
                state_failure_phase="initialization" if initializing else "moving",
            )
            if initializing:
                if timing["remaining_m"] < timing["safe_stop_distance_m"]:
                    self._fail(context, "Tunnel pedestrian initial state unobserved in unsafe "
                               f"interaction window: remaining={timing['remaining_m']:.2f}m, "
                               f"safe_stop={timing['safe_stop_distance_m']:.2f}m")
                context.tunnel_dynamic_result["initial_state_read_misses"] += 1
                observation = {"wait_wall_seconds": time.monotonic() -
                               context.tunnel_dynamic_spawn_ack_monotonic,
                               "error": str(error)}
                context.tunnel_dynamic_result["last_initial_state_miss"] = observation
                history = context.tunnel_dynamic_result["initial_state_read_history"]
                if len(history) < 8:
                    history.append(observation)
                if observation["wait_wall_seconds"] >= INITIAL_STATE_TIMEOUT_WALL_SEC:
                    self._fail(context, "Tunnel pedestrian initial state unavailable after "
                               f"{INITIAL_STATE_TIMEOUT_WALL_SEC:.1f}s wall deadline: {error}")
            elif context.tunnel_dynamic_state_misses >= 3:
                self._fail(context, f"Tunnel pedestrian state unavailable: {error}")
            self._save_report(context)
            return
        if (initializing and time.monotonic() - context.tunnel_dynamic_spawn_ack_monotonic
                >= INITIAL_STATE_TIMEOUT_WALL_SEC):
            self._fail(context, "Tunnel pedestrian first state arrived after "
                       f"{INITIAL_STATE_TIMEOUT_WALL_SEC:.1f}s wall deadline")
        context.tunnel_dynamic_state_misses = 0
        if not math.isfinite(x) or not math.isfinite(y):
            self._fail(context, "Tunnel pedestrian position is not finite")
        previous = context.tunnel_dynamic_last_position
        dt = max(0.0, float(elapsed) - context.tunnel_dynamic_last_observed_elapsed)
        step_m = math.hypot(x - previous["x"], y - previous["y"])
        max_plausible_step_m = max(5.0, plan["speed_mps"] * 3.0) * dt + 1.0
        if step_m > max_plausible_step_m:
            if context.tunnel_dynamic_random_mix or not plan.get("trial_candidate"):
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
        context.tunnel_dynamic_phase = "moving"
        observations = context.tunnel_dynamic_result.setdefault("state_observations", {})
        observations[source] = observations.get(source, 0) + 1
        if source == "GetAllActorsState" and observations[source] == 1:
            print(f"[TunnelDynamic] actor={context.tunnel_dynamic_result['actor_id']} "
                  f"state_source={source} direct_read_error={'; '.join(read_errors)}")
        context.tunnel_dynamic_result.update(
            state_read_failures=0, state_read_error=None,
            last_state_source=source, last_state_elapsed_sec=float(elapsed),
            spawn_state_verified=True,
        )
        self._observe_path(context, elapsed, ego_state, current_s, x, y)
        self._observe_response(context, elapsed, ego_state, current_s, x, y)
        if initializing:
            context.tunnel_dynamic_result.update(
                first_verified_state_source=source,
                first_verified_state_delay_wall_seconds=time.monotonic() -
                context.tunnel_dynamic_spawn_ack_monotonic,
                first_verified_position={"x": x, "y": y},
                first_verified_displacement_from_start_m=_distance(
                    {"x": x, "y": y}, plan["waypoints"][0]),
            )
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
        context.tunnel_dynamic_result.update(
            endpoint_reached=True, endpoint_elapsed_sec=float(elapsed),
            ego_s_at_endpoint=float(current_s),
            remaining_to_interaction_at_endpoint_m=plan["interaction_s"] - float(current_s))
        # This scenario supplies obstacles for controller development. Record
        # Ego response without requiring slowdown or stopping for completion.
        context.tunnel_dynamic_phase = "completed"
        context.tunnel_dynamic_result.update(
            completed=True, completed_elapsed_sec=float(elapsed),
            interaction_assessment=("verified" if context.tunnel_dynamic_result["interaction_verified"]
                                    else "not_observed" if context.tunnel_dynamic_result["interaction_required"]
                                    else "non_conflicting_path"))
        print(f"[TunnelDynamic] reached final waypoint; pedestrian removed "
              f"path={plan['path_id']}")
        next_index = context.tunnel_dynamic_event_index + 1
        if (not isinstance(context, _TunnelEventContext)
                and next_index < len(context.tunnel_dynamic_plans)):
            self._activate_event(context, next_index)
        self._save_report(context)

    def _observe_path(self, context, elapsed, ego_state, current_s, x, y):
        """Read-only trajectory/encounter evidence, never a completion gate."""
        result = context.tunnel_dynamic_result
        points = context.tunnel_dynamic_plan["waypoints"]
        best = (float("inf"), 0.0)
        cumulative = 0.0
        for a, b in zip(points[:-1], points[1:]):
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            length = math.hypot(dx, dy)
            ratio = max(0.0, min(1.0, ((x-a["x"])*dx + (y-a["y"])*dy) / length**2))
            deviation = math.hypot(x-a["x"]-ratio*dx, y-a["y"]-ratio*dy)
            best = min(best, (deviation, cumulative + ratio * length))
            cumulative += length
        npc_s = project_distance_on_polyline(context.route_points, x, y)
        gap = npc_s - float(current_s)
        observed = result.setdefault("trajectory_observation", {
            "samples": 0, "max_path_deviation_m": 0.0,
            "minimum_abs_longitudinal_gap_m": abs(gap),
        })
        observed["samples"] += 1
        observed["max_path_deviation_m"] = max(observed["max_path_deviation_m"], best[0])
        observed["minimum_abs_longitudinal_gap_m"] = min(
            observed["minimum_abs_longitudinal_gap_m"], abs(gap))
        observed["last_path_progress_m"] = best[1]
        observed["last_longitudinal_gap_m"] = gap
        trace = result.setdefault("motion_trace", [])
        if not trace or float(elapsed) - trace[-1]["elapsed_sec"] >= 0.1:
            trace.append({"elapsed_sec": float(elapsed), "x": x, "y": y,
                          "ego_s": float(current_s), "npc_s": npc_s,
                          "ego_speed_mps": ego_state.get("speed_mps", ego_state.get("speed")),
                          "path_progress_m": best[1], "path_deviation_m": best[0]})

    def _overlap_needed(self, context, elapsed, timing, current_s):
        if not isinstance(context, _TunnelEventContext):
            return True
        active = [event for event in context.root.tunnel_dynamic_events
                  if event is not context and event.tunnel_dynamic_actor is not None]
        if not active:
            return True
        # Ego can still brake for an earlier crossing/oncoming pedestrian.
        # Extrapolating its current approach speed through that encounter can
        # send a later side pedestrian past long before Ego arrives. Keep the
        # previous ordering until this conflict finishes or Ego has passed it.
        blocking = [event.tunnel_dynamic_event_index for event in active
                    if event.tunnel_dynamic_result["interaction_required"]
                    and current_s < event.tunnel_dynamic_plan["interaction_s"]]
        if blocking:
            context.tunnel_dynamic_result["overlap_decision"] = {
                "elapsed_sec": float(elapsed), "allowed": False,
                "reason": "prior_conflicting_event_still_ahead",
                "active_event_indices": [event.tunnel_dynamic_event_index for event in active],
                "blocking_event_indices": blocking,
            }
            return False
        acceleration, ceiling = self._motion_forecast(context)
        speed = timing["ego_speed_mps"]
        eta = lambda distance: trajectory_timing.travel_time(distance, speed, acceleration, ceiling)
        plan = context.tunnel_dynamic_plan
        # Keep a metre before the existing safety boundary. Do not wait until
        # the next tick could turn a valid start into an unsafe one.
        latest = eta(timing["remaining_m"] - timing["safe_stop_distance_m"] - 1.0)
        if context.tunnel_dynamic_result["interaction_required"]:
            eta_bound = speed * (timing["npc_eta_sec"] +
                                 context.tunnel_dynamic_trigger_cfg["minimum_eta_delta_sec"])
            latest = min(latest, eta(timing["remaining_m"] - eta_bound))
            if plan.get("behavior", "").startswith("straight_opposite"):
                # Waiting for an earlier side walk must not consume the lead
                # restored to leave room beyond an oncoming NPC's endpoint.
                latest = min(latest, eta(timing["remaining_m"]
                                         - timing["trigger_distance_m"]))
        else:
            _, encounter_latest = trajectory_timing.encounter_start_window(
                context.tunnel_dynamic_motion_samples[context.tunnel_dynamic_event_index],
                current_s, speed, acceleration, ceiling,
                context.tunnel_dynamic_trigger_cfg["stopping_safety_margin_m"])
            latest = min(latest, encounter_latest)
        cleanup_budget = 0.2 + float(context.cfg.get("check_period_sec", 0.05))
        latest = min(latest, eta(context.tunnel_dynamic_goal_s - current_s)
                     - plan["path_length_m"] / plan["speed_mps"] - cleanup_budget)
        remaining_times = []
        for event in active:
            previous_plan = event.tunnel_dynamic_plan
            points = previous_plan["waypoints"]
            target = event.tunnel_dynamic_target_index
            remaining = _distance(event.tunnel_dynamic_last_position, points[target])
            remaining += sum(_distance(a, b) for a, b in zip(points[target:-1], points[target+1:]))
            remaining_times.append(remaining / previous_plan["speed_mps"] + cleanup_budget)
        wait = max(remaining_times)
        needed = wait >= latest
        context.tunnel_dynamic_result["overlap_decision"] = {
            "elapsed_sec": float(elapsed), "allowed": needed,
            "active_event_indices": [event.tunnel_dynamic_event_index for event in active],
            "predicted_wait_for_deletion_sec": wait,
            "latest_start_delay_sec": latest,
        }
        return needed

    def _record_ego_motion(self, context, elapsed, ego_state):
        raw_speed = ego_state.get("speed_mps", ego_state.get("speed"))
        if raw_speed is None:
            return
        speed = max(0.0, _number(raw_speed, "ego_speed_mps"))
        history = context.tunnel_dynamic_ego_history
        now = float(elapsed)
        if history and now < history[-1][0]:
            history.clear()
        history[:] = [item for item in history if now - item[0] <= 1.0]
        if history and now == history[-1][0]:
            history[-1] = (now, speed)
        else:
            history.append((now, speed))

    def _motion_forecast(self, context):
        history = context.tunnel_dynamic_ego_history
        acceleration = 0.0
        if len(history) >= 3 and history[-1][0] - history[0][0] >= 0.25:
            acceleration = max(0.0, (history[-1][1] - history[0][1]) /
                               (history[-1][0] - history[0][0]))
        # Read the existing controller's ceiling for prediction only. Never
        # change its configuration, target speed, or controls.
        controller = getattr(context, "expert_controller", None)
        ceiling = getattr(controller, "max_velocity_kmh", None)
        ceiling = float(ceiling) / 3.6 if isinstance(ceiling, (int, float)) and ceiling > 0 else None
        return acceleration, ceiling

    def _side_motion_state(self, context):
        """Observe recent braking separately from the optimistic deadline ETA."""
        history = context.tunnel_dynamic_ego_history
        if not history:
            return {"decelerating": False, "observation_span_sec": 0.0}
        recent = [item for item in history if history[-1][0] - item[0] <= 0.3]
        span = recent[-1][0] - recent[0][0]
        drop = recent[0][1] - recent[-1][1]
        acceleration = -drop / span if span > 0.0 else 0.0
        return {
            "decelerating": len(recent) >= 3 and span >= 0.15
                and drop >= 0.1 and acceleration < -0.25,
            "observation_span_sec": span,
            "speed_drop_mps": drop,
            "acceleration_mps2": acceleration,
        }

    def _event_trigger_config(self, context):
        cfg = context.tunnel_dynamic_trigger_cfg
        if (not context.tunnel_dynamic_plan.get("behavior", "").startswith("straight_opposite")
                or not context.tunnel_dynamic_result["interaction_required"]):
            return cfg
        overrides = context.cfg["tunnel_dynamic"].get("interaction_trigger") or {}
        cfg = dict(cfg)
        if "interaction_lead_sec" not in overrides:
            cfg["interaction_lead_sec"] = ONCOMING_INTERACTION_LEAD_SEC
        if "minimum_trigger_distance_m" not in overrides:
            cfg["minimum_trigger_distance_m"] = 12.0
        return cfg

    def _side_schedule(self, context, current_s, timing):
        """Choose a start inside an encounter window and available time budget.

        The no-early bound uses only the current measured speed. Positive
        acceleration may tighten a deadline, but cannot open the encounter
        window early. Infeasible predictions remain
        visible; they never justify a distant spawn, a reroll, or Ego control.
        """
        speed = timing["ego_speed_mps"]
        cfg = context.tunnel_dynamic_trigger_cfg
        index = context.tunnel_dynamic_event_index
        plans = context.tunnel_dynamic_plans
        acceleration, ceiling = self._motion_forecast(context)
        margin = cfg["stopping_safety_margin_m"]
        # Route progress refers to Ego's pose, not its front bumper. Reuse
        # the existing tunnel body geometry; an NPC alongside the front is
        # already an encounter. This reads geometry only, never a planner.
        front_margin = max(margin, TUNNEL_GEOMETRY_POLICY["minimum_ego_body"]["front_m"])
        earliest, own_latest = trajectory_timing.encounter_start_window(
            context.tunnel_dynamic_motion_samples[index], current_s, speed,
            0.0, speed, margin, front_margin_m=front_margin)
        eta = lambda s: trajectory_timing.travel_time(
            s - current_s, speed, acceleration, ceiling)
        goal_eta = eta(context.tunnel_dynamic_goal_s)
        # Budget the entire unchanged path, plus the existing removal polling
        # and a driving tick. Anchor ETA alone accounts for only half a walk.
        cleanup_budget = 0.2 + float(context.cfg.get("check_period_sec", 0.05))
        deadline = goal_eta
        for future in range(len(plans) - 1, index - 1, -1):
            plan = plans[future]
            duration = plan["path_length_m"] / plan["speed_mps"] + cleanup_budget
            safe_start = eta(plan["interaction_s"] - timing["safe_stop_distance_m"] - 1.0)
            if context.tunnel_dynamic_results[future]["interaction_required"]:
                latest = safe_start
            else:
                _, latest = trajectory_timing.encounter_start_window(
                    context.tunnel_dynamic_motion_samples[future], current_s, speed,
                    acceleration, ceiling, margin)
                latest = min(latest, safe_start)
            deadline = min(latest, deadline - duration)
        preferred = (timing["ego_eta_sec"] - timing["npc_eta_sec"]
                     - cfg["interaction_lead_sec"])
        target = max(earliest, min(preferred, deadline))
        return {
            "policy": "tunnel_trajectory_schedule",
            "earliest_spawn_delay_sec": earliest,
            "latest_spawn_delay_sec": min(own_latest, deadline),
            "target_spawn_delay_sec": target,
            "schedule_feasible": earliest <= min(own_latest, deadline),
            "predicted_goal_eta_sec": goal_eta,
            "forecast_acceleration_mps2": acceleration,
            "approach_forecast_speed_ceiling_mps": speed,
            "approach_acceleration_horizon_sec": 0.0,
            "encounter_front_margin_m": front_margin,
            "npc_full_duration_sec": plans[index]["path_length_m"] / plans[index]["speed_mps"],
            "cleanup_budget_sec": cleanup_budget,
        }

    def _wait_for_interaction(self, context, elapsed, ego_state, current_s):
        plan = context.tunnel_dynamic_plan
        cfg = self._event_trigger_config(context)
        speed = _number(ego_state.get("speed_mps", ego_state.get("speed", 0.0)), "ego_speed_mps")
        timing = interaction_timing(
            cfg, plan["interaction_s"], plan["start_to_interaction_m"],
            plan["speed_mps"], {"speed_mps": speed}, current_s,
        )
        side_path = not context.tunnel_dynamic_result["interaction_required"]
        timing["policy"] = "tunnel_interaction_eta"
        timing["interaction_lead_sec"] = cfg["interaction_lead_sec"]
        if side_path and speed >= cfg["minimum_ego_speed_mps"]:
            timing.update(self._side_schedule(context, current_s, timing))
            timing["side_motion_state"] = self._side_motion_state(context)
            timing["side_motion_hold"] = timing["side_motion_state"]["decelerating"]
        context.tunnel_dynamic_result["last_timing"] = {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in timing.items()}
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
        if side_path:
            if (timing["side_motion_hold"]
                    or timing["remaining_m"] > cfg["maximum_trigger_distance_m"]
                    or timing["target_spawn_delay_sec"] > 0.0):
                return
        elif timing["remaining_m"] > timing["trigger_distance_m"]:
            return
        if timing["remaining_m"] < timing["safe_stop_distance_m"]:
            self._fail(context, f"unsafe_trigger_window: remaining={timing['remaining_m']:.2f}m "
                       f"safe_stop={timing['safe_stop_distance_m']:.2f}m")
        if not side_path and timing["eta_delta_sec"] > cfg["maximum_eta_delta_sec"]:
            return
        if not side_path and timing["eta_delta_sec"] < cfg["minimum_eta_delta_sec"]:
            self._fail(context, "unsafe_trigger_window: NPC would arrive after Ego")
        if not self._overlap_needed(context, elapsed, timing, current_s):
            return
        context.tunnel_dynamic_result.update(
            trigger_s=float(current_s), trigger=dict(timing, elapsed_sec=float(elapsed)),
        )
        context.tunnel_dynamic_result["ego_response"]["baseline_speed_mps"] = speed
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
