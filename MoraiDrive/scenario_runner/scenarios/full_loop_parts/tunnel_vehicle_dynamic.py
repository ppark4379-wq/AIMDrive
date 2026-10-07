"""One actor per whole-tunnel behavior; no section spawn/despawn cycles."""

import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from . import tunnel
from .tunnel_dynamic_catalog import _finite
from .tunnel_vehicle_catalog import (
    BEHAVIORS,
    build_behavior_catalog,
    build_vehicle_plan,
    nearest_route_pose,
    UnsupportedVehicleTopology,
)
from ...utils.transform_utils import (
    interpolate_on_polyline,
    project_distance_on_polyline,
)


class TunnelVehicleHandler:
    def validate(self, context):
        cfg = context.cfg["tunnel_vehicle_dynamic"]
        for key in (
            "initial_spawn_gap_m",
            "minimum_initial_gap_m",
            "slow_speed_mps",
            "cruise_speed_mps",
            "oncoming_speed_mps",
            "cut_in_speed_mps",
            "minimum_event_run_in_m",
            "pass_clearance_m",
            "spawn_position_tolerance_m",
            "lane_position_tolerance_m",
            "minimum_observed_travel_m",
            "max_run_sec",
            "maximum_step_speed_mps",
            "step_slack_m",
            "cut_in_timeout_sec",
        ):
            if _finite(cfg[key], key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not cfg.get("event_sections") or any(
            type(i) is not int or i not in (2, 3) for i in cfg["event_sections"]
        ):
            raise ValueError(
                "Vehicle event_sections must be a nonempty subset of [2, 3]"
            )
        brake = dict(context.cfg.get("tunnel_sudden_brake", {}))
        brake.update(cfg.get("brake_overrides") or {})
        for key in (
            "brake_trigger_gap_m",
            "minimum_brake_gap_m",
            "reaction_max_gap_m",
            "emergency_destroy_gap_m",
            "stopped_speed_threshold_mps",
            "stop_confirm_timeout_sec",
            "stop_hold_sec",
            "ego_reaction_speed_drop",
            "resume_speed_mps",
            "approach_speed_mps",
        ):
            if _finite(brake[key], key) <= 0:
                raise ValueError(f"{key} must be positive")
        if (
            not brake["emergency_destroy_gap_m"]
            < brake["minimum_brake_gap_m"]
            <= brake["brake_trigger_gap_m"]
            <= brake["reaction_max_gap_m"]
        ):
            raise ValueError("Vehicle braking safety gap settings are inconsistent")
        window = getattr(context, "mission_route", None)
        context.tunnel_vehicle_route_points = (
            window.full_points if window is not None else context.route_points
        )
        context.tunnel_vehicle_route_prefix_s = (
            window.prefix_s if window is not None else 0.0
        )
        prefix = context.tunnel_vehicle_route_prefix_s
        zone = context.route_definition["zones"]["tunnel"]
        args = (
            context.map_loader,
            context.tunnel_vehicle_route_points,
            context.tunnel_entry_s + prefix,
            context.tunnel_exit_s + prefix,
            zone,
            cfg,
        )
        catalog = build_behavior_catalog(*args)
        context.tunnel_vehicle_catalog = catalog
        behavior = cfg.get("behavior", "vehicle_slow")
        if cfg.get("random_behavior"):
            requested = cfg.get("behaviors") or list(BEHAVIORS)
            if len(set(requested)) != len(requested) or any(
                b not in BEHAVIORS for b in requested
            ):
                raise ValueError("Unknown or duplicate random vehicle behavior")
            choices = [b for b in requested if b in catalog["behaviors"]]
            if not choices:
                raise ValueError(
                    "No supported tunnel vehicle behavior available for random selection"
                )
            behavior = context.random.choice(choices)
        if behavior in catalog["unavailable_behaviors"]:
            raise UnsupportedVehicleTopology(catalog["unavailable_behaviors"][behavior])
        if behavior not in catalog["behaviors"]:
            raise ValueError(f"Unknown tunnel vehicle behavior: {behavior}")
        base = catalog["behaviors"][behavior]
        section = (
            context.random.choice(base["safe_event_sections"])
            if base["event_section"] is not None
            else None
        )
        plan = build_vehicle_plan(*args, behavior, event_section=section)
        plan["brake_settings"] = brake
        if cfg["max_run_sec"] <= (
            (plan["exit_s_m"] - plan["spawn_s_m"]) / max(plan["speed_mps"], 1.0)
            if behavior != "vehicle_oncoming"
            else (plan["spawn_s_m"] - plan["entry_s_m"]) / plan["speed_mps"]
        ):
            raise ValueError("Vehicle timeout must exceed nominal tunnel travel time")
        # Freeze the scenario RNG after behavior/section selection. Weather
        # history must not change the later live model draw.
        context.tunnel_vehicle_random_state = context.random.getstate()
        context.tunnel_vehicle_plan = plan
        context.tunnel_vehicle_actor = None
        context.tunnel_vehicle_phase = "validated"
        context.tunnel_vehicle_report_path = None

    def _now(self, context):
        return (
            float(context._scenario_now())
            if callable(getattr(context, "_scenario_now", None))
            else time.monotonic()
        )

    def arm(self, context):
        plan, cfg = context.tunnel_vehicle_plan, context.cfg["tunnel_vehicle_dynamic"]
        context.tunnel_vehicle_result = {
            "run_seed": context.run_seed,
            "plan": plan,
            "spawned": False,
            "spawn_verified": False,
            "observed_moving": False,
            "completed": False,
            "destroyed": False,
            "actor_id": None,
            "same_actor_continuity": True,
            "observations": [],
            "transitions": [],
            "unavailable_behaviors": context.tunnel_vehicle_catalog[
                "unavailable_behaviors"
            ],
            "default_ai_unchanged": True,
            "travelled_m": 0.0,
            "brake": {},
        }
        context.tunnel_vehicle_report_path = (
            Path(context.cfg["profile_state_path"]).parent
            / "tunnel_vehicle_runs"
            / f"{context.run_instance_id}.json"
        )
        context.tunnel_vehicle_started_at = self._now(context)
        context.tunnel_vehicle_last_sample = None
        context.tunnel_vehicle_last_log_sec = -1
        context.tunnel_vehicle_elapsed = 0.0
        context.tunnel_event_phase = "disabled"
        context.tunnel_event_result = {"disabled_reason": "tunnel_vehicle_dynamic"}
        try:
            available = sorted(set(context.vehicle_models()))
            configured = cfg.get("models") or available
            if not configured or any(m not in available for m in configured):
                raise RuntimeError(
                    "Tunnel vehicle model unavailable in configured pool"
                )
            model = cfg.get("model")
            if model is None:
                model_rng = random.Random()
                model_rng.setstate(context.tunnel_vehicle_random_state)
                model = (
                    model_rng.choice(list(configured))
                    if cfg.get("random_model", True)
                    else context.cfg["highway_npc"]["model"]
                )
            if model not in available:
                raise RuntimeError(f"Tunnel vehicle model unavailable: {model}")
            plan["model"] = model
            context.tunnel_vehicle_result["selection"] = {
                "behavior": plan["behavior"],
                "event_section": plan["event_section"],
                "model": model,
            }
            print(
                f"[TunnelVehicle] SELECT seed={context.run_seed} behavior={plan['behavior']} event_section={plan['event_section']} model={model}"
            )
            context.highway_fixed_model = model
            for behavior, reason in context.tunnel_vehicle_catalog[
                "unavailable_behaviors"
            ].items():
                print(f"[TunnelVehicle] excluded {behavior}: {reason}")
            self._spawn(context)
        except BaseException as error:
            self.record_interruption(context, error)
            raise
        print(
            f"[TunnelVehicle] armed seed={context.run_seed} behavior={plan['behavior']} event_section={plan['event_section']} model={model} actor={context.tunnel_vehicle_result['actor_id']} report={context.tunnel_vehicle_report_path}"
        )

    def _save(self, context):
        path = context.tunnel_vehicle_report_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    **context.tunnel_vehicle_result,
                    "phase": context.tunnel_vehicle_phase,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def _phase(self, context, phase, reason):
        if phase == context.tunnel_vehicle_phase:
            return
        context.tunnel_vehicle_result["transitions"].append(
            {
                "from": context.tunnel_vehicle_phase,
                "to": phase,
                "elapsed_sec": self._now(context) - context.tunnel_vehicle_started_at,
                "reason": reason,
                "actor_id": context.tunnel_vehicle_result["actor_id"],
            }
        )
        context.tunnel_vehicle_phase = phase
        self._save(context)
        print(
            f"[TunnelVehicle] phase={phase} behavior={context.tunnel_vehicle_plan['behavior']} reason={reason}"
        )

    def _remove(self, context):
        actor = context.tunnel_vehicle_actor
        if actor is None:
            return
        if not actor.destroy():
            raise RuntimeError("Tunnel vehicle removal rejected")
        if actor in context.scenario_actors:
            context.scenario_actors.remove(actor)
        context.tunnel_vehicle_actor = None
        context.tunnel_vehicle_result["destroyed"] = True
        context.tunnel_vehicle_result["cleanup_verified"] = True

    def record_interruption(self, context, error):
        if (
            not hasattr(context, "tunnel_vehicle_result")
            or context.tunnel_vehicle_phase == "completed"
        ):
            return
        result = context.tunnel_vehicle_result
        result["failure_reason"] = str(error) or type(error).__name__
        self._phase(context, "failed", result["failure_reason"])
        try:
            self._remove(context)
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            result["cleanup_verified"] = False
        self._save(context)

    def _fail(self, context, message):
        error = RuntimeError(message)
        self.record_interruption(context, error)
        raise error

    def _read_state(self, context):
        actor = context.tunnel_vehicle_actor
        state = actor.get_actor_state()
        if state is None:
            self._fail(context, "Tunnel vehicle state unavailable")
        try:
            p = state.transform.location
            x, y, z, yaw = (
                float(p.x),
                float(p.y),
                float(p.z),
                float(state.transform.rotation.z),
            )
            speed = tunnel._actor_speed(state)
            link = str(state.vehicle_state.current_link_info.id.value).strip()
            actor_id = str(state.actor_info.id.value).strip()
            handle_id = str(actor.get_id()).strip()
        except (AttributeError, TypeError, ValueError):
            self._fail(context, "Tunnel vehicle state/identity malformed")
        if not all(math.isfinite(v) for v in (x, y, z, yaw, speed)):
            self._fail(context, "Tunnel vehicle state is not finite")
        result, plan = context.tunnel_vehicle_result, context.tunnel_vehicle_plan
        if (
            not actor_id
            or actor_id != handle_id
            or (result["actor_id"] is not None and actor_id != result["actor_id"])
        ):
            result["same_actor_continuity"] = False
            self._fail(context, "Tunnel vehicle actor identity changed")
        result["actor_id"] = actor_id
        if not link or link not in plan["route_links"]:
            self._fail(context, f"Tunnel vehicle left its assigned route: {link}")
        now = self._now(context)
        previous = context.tunnel_vehicle_last_sample
        if previous is not None:
            dt = max(0.0, now - previous["time"])
            step = math.dist((x, y, z), (previous["x"], previous["y"], previous["z"]))
            cfg = context.cfg["tunnel_vehicle_dynamic"]
            if step > cfg["maximum_step_speed_mps"] * dt + cfg["step_slack_m"]:
                self._fail(
                    context,
                    f"Tunnel vehicle implausible step/teleport: {step:.2f}m in {dt:.3f}s",
                )
            result["travelled_m"] += step
        sample = {
            "time": now,
            "x": x,
            "y": y,
            "z": z,
            "yaw_deg": yaw,
            "speed_mps": speed,
            "link": link,
            "actor_id": actor_id,
        }
        full_s = project_distance_on_polyline(context.tunnel_vehicle_route_points, x, y)
        sample["npc_s_m"] = full_s - context.tunnel_vehicle_route_prefix_s
        rx, ry, _, ego_yaw = interpolate_on_polyline(
            context.tunnel_vehicle_route_points, full_s
        )
        angle = math.radians(ego_yaw)
        sample["lateral_m"] = -(x - rx) * math.sin(angle) + (y - ry) * math.cos(angle)
        sample["tangent_dot"] = math.cos(math.radians(yaw - ego_yaw))
        if previous is not None and "npc_s_m" in previous:
            delta_s = sample["npc_s_m"] - previous["npc_s_m"]
            opposite = plan["behavior"] == "vehicle_oncoming"
            if plan["entry_s_m"] + 0.5 < full_s < plan["exit_s_m"] - 0.5 and (
                delta_s > 1.0 if opposite else delta_s < -1.0
            ):
                self._fail(
                    context,
                    "Tunnel vehicle observed reverse progress on assigned route",
                )
            sample["progress_delta_m"] = delta_s
        context.tunnel_vehicle_last_sample = sample
        # Compare against its real MGeo lane, including the exit continuation.
        error, _ = nearest_route_pose(context.map_loader, [link], x, y)
        changing = context.tunnel_vehicle_phase == "changing_lane"
        tolerance = (
            4.0
            if changing
            else context.cfg["tunnel_vehicle_dynamic"]["lane_position_tolerance_m"]
        )
        if error > tolerance:
            self._fail(context, "Tunnel vehicle left the intended MGeo lane")
        entry, exit = plan["entry_s_m"], plan["exit_s_m"]
        if entry + 0.5 < full_s < exit - 0.5:
            opposite = plan["behavior"] == "vehicle_oncoming"
            if (
                sample["tangent_dot"] >= -0.5
                if opposite
                else sample["tangent_dot"] <= 0.5
            ):
                self._fail(context, "Tunnel vehicle observed wrong travel direction")
            if opposite and abs(sample["lateral_m"]) < 2:
                self._fail(context, "Oncoming vehicle entered Ego lane")
        return sample

    def _spawn(self, context):
        plan = context.tunnel_vehicle_plan
        p = plan["spawn_target"]
        ego = context.grpc.get_ego_motion_state()
        if not isinstance(ego, dict) or not all(
            k in ego and math.isfinite(float(ego[k])) for k in ("x", "y")
        ):
            self._fail(context, "Ego state unavailable at vehicle setup")
        start_x, start_y, _, _ = interpolate_on_polyline(
            context.tunnel_vehicle_route_points, plan["entry_s_m"]
        )
        if (
            math.hypot(ego["x"] - start_x, ego["y"] - start_y)
            > context.cfg["tunnel_vehicle_dynamic"]["spawn_position_tolerance_m"]
        ):
            self._fail(context, "Ego is outside the common CP14 start at vehicle setup")
        ego_s = project_distance_on_polyline(
            context.tunnel_vehicle_route_points, ego["x"], ego["y"]
        )
        initial_gap = plan["spawn_s_m"] - ego_s
        context.tunnel_vehicle_result["spawn_ego"] = {
            "x": ego["x"],
            "y": ego["y"],
            "s_m": ego_s,
        }
        context.tunnel_vehicle_result["planned_initial_gap_m"] = initial_gap
        if initial_gap < context.cfg["tunnel_vehicle_dynamic"]["minimum_initial_gap_m"]:
            self._fail(context, "Tunnel vehicle unsafe planned initial gap")

        def track(actor):
            context.tunnel_vehicle_actor = actor
            context.tunnel_vehicle_result["spawned"] = True

        # Same route/waypoint/speed operations as highway. Do not toggle SetAI:
        # live MORAI rejects that call even when default NPC AI is moving.
        actor = context._spawn_highway_vehicle(
            context.grpc.make_transform(p["x"], p["y"], p["z"], p["yaw_deg"]),
            (
                [plan["route_links"][0]]
                if plan["behavior"] == "vehicle_cut_in"
                else plan["route_links"][p["route_index"] :]
            ),
            p["waypoint"],
            plan["behavior"],
            plan["speed_mps"],
            configure_ai=False,
            on_spawn=track,
        )
        if actor is None:
            self._fail(context, "Failed to spawn tunnel vehicle or assign route")
        sample = self._read_state(context)
        error = math.dist(
            (sample["x"], sample["y"], sample["z"]), (p["x"], p["y"], p["z"])
        )
        gap = sample["npc_s_m"] + context.tunnel_vehicle_route_prefix_s - ego_s
        result = context.tunnel_vehicle_result
        result.update(
            {
                "spawn_target": dict(p),
                "actual_spawn": dict(sample),
                "spawn_error_m": error,
                "initial_route_gap_m": gap,
            }
        )
        if (
            error > context.cfg["tunnel_vehicle_dynamic"]["spawn_position_tolerance_m"]
            or gap < context.cfg["tunnel_vehicle_dynamic"]["minimum_initial_gap_m"]
            or sample["link"] != p["link"]
        ):
            self._fail(context, "Tunnel vehicle spawn verification failed")
        result["spawn_verified"] = True
        print(
            f"[TunnelVehicle] SPAWN actor={sample['actor_id']} link={sample['link']} gap={gap:.1f}m error={error:.2f}m"
        )
        context.tunnel_vehicle_spawn_xy = (sample["x"], sample["y"])
        phases = {
            "vehicle_slow": "moving",
            "vehicle_sudden_brake": "cruising",
            "vehicle_stopped": "stopped",
            "vehicle_oncoming": "approaching",
            "vehicle_cut_in": "adjacent_cruise",
        }
        self._phase(context, phases[plan["behavior"]], "spawn_verified")
        if plan["behavior"] == "vehicle_stopped":
            if not context.grpc.stop_vehicle(actor):
                self._fail(context, "Stopped vehicle stop command rejected")
            sample = self._read_state(context)
            if (
                sample["speed_mps"]
                > plan["brake_settings"]["stopped_speed_threshold_mps"]
            ):
                self._fail(context, "Stopped vehicle is moving at spawn")
            result["stopped_verified"] = True
        if plan["behavior"] == "vehicle_sudden_brake":
            proxy = SimpleNamespace(
                cfg={"tunnel_sudden_brake": plan["brake_settings"]},
                grpc=context.grpc,
                tunnel_npc=actor,
                tunnel_event_result=result["brake"],
                tunnel_entry_s=context.tunnel_entry_s,
                tunnel_event_phase="cruising",
                tunnel_npc_brake_scenario_time=None,
                tunnel_npc_stopped_scenario_time=None,
                tunnel_npc_forced_stop=False,
                _scenario_now=lambda: self._now(context),
            )
            proxy._resume_tunnel_npc_after_event = (
                lambda *args: tunnel._resume_tunnel_npc_after_event(proxy, *args)
            )
            context.tunnel_vehicle_brake_context = proxy
        self._save(context)

    def _complete(self, context, reason):
        self._remove(context)
        context.tunnel_vehicle_result.update(
            {"completed": True, "cleanup_reason": reason}
        )
        self._phase(context, "completed", reason)

    def update(self, context, elapsed, ego_state, current_s):
        if context.tunnel_vehicle_phase in ("completed", "failed"):
            return
        try:
            self._update(context, float(elapsed), ego_state, float(current_s))
        except BaseException as error:
            if context.tunnel_vehicle_phase != "failed":
                self.record_interruption(context, error)
            raise

    def _update(self, context, elapsed, ego_state, current_s):
        plan, cfg, result = (
            context.tunnel_vehicle_plan,
            context.cfg["tunnel_vehicle_dynamic"],
            context.tunnel_vehicle_result,
        )
        age = self._now(context) - context.tunnel_vehicle_started_at
        if age > cfg["max_run_sec"]:
            self._fail(context, "Tunnel vehicle timed out")
        context.tunnel_vehicle_last_ego = dict(ego_state)
        context.tunnel_vehicle_last_ego_s = current_s
        sample = self._read_state(context)
        x, y, npc_s, speed, link = (
            sample[k] for k in ("x", "y", "npc_s_m", "speed_mps", "link")
        )
        gap = npc_s - current_s
        sample.update(
            {
                "elapsed_sec": elapsed,
                "route_gap_m": gap,
                "travelled_m": result["travelled_m"],
            }
        )
        result["observed_moving"] |= speed > 0.5
        result["last_observation"] = sample
        if int(age) != context.tunnel_vehicle_last_log_sec:
            context.tunnel_vehicle_last_log_sec = int(age)
            result["observations"].append(sample)
            self._save(context)
            print(
                f"[TunnelVehicle] {context.tunnel_vehicle_phase} actor={sample['actor_id']} speed={speed:.1f}m/s gap={gap:.1f}m travel={result['travelled_m']:.1f}m"
            )
        behavior = plan["behavior"]
        if behavior == "vehicle_stopped":
            if (
                speed > plan["brake_settings"]["stopped_speed_threshold_mps"]
                or math.hypot(
                    x - context.tunnel_vehicle_spawn_xy[0],
                    y - context.tunnel_vehicle_spawn_xy[1],
                )
                > 1.0
            ):
                self._fail(context, "Stopped vehicle moved before Ego passed")
            if (
                -gap >= cfg["pass_clearance_m"]
                and math.hypot(x - ego_state["x"], y - ego_state["y"])
                >= cfg["pass_clearance_m"]
            ):
                result["ego_passed"] = True
                self._phase(context, "ego_passed", "pass_clearance_verified")
                self._complete(context, "ego_passed_stopped_vehicle")
        elif behavior == "vehicle_oncoming":
            if (
                result["travelled_m"] >= cfg["minimum_observed_travel_m"]
                and -gap >= cfg["pass_clearance_m"]
                and math.hypot(x - ego_state["x"], y - ego_state["y"])
                >= cfg["pass_clearance_m"]
            ):
                result["passed"] = True
                self._phase(context, "passed", "oncoming_pass_clearance_verified")
                self._complete(context, "oncoming_passed")
        elif behavior == "vehicle_sudden_brake":
            self._brake(context, elapsed, ego_state, npc_s, gap, x, y, link, speed)
        elif behavior == "vehicle_cut_in":
            self._cut_in(context, sample, gap)

    def _brake(self, context, elapsed, ego_state, npc_s, gap, x, y, link, speed):
        plan = context.tunnel_vehicle_plan
        proxy = context.tunnel_vehicle_brake_context
        cfg = plan["brake_settings"]
        region = plan["event_region"]
        full_s = npc_s + context.tunnel_vehicle_route_prefix_s
        if proxy.tunnel_event_phase in ("cruising", "closing_gap"):
            if full_s > region["end_s_m"]:
                self._fail(context, "Vehicle missed safe brake region/gap")
            if full_s >= region["start_s_m"]:
                if cfg["minimum_brake_gap_m"] <= gap <= cfg["brake_trigger_gap_m"]:
                    tunnel._command_tunnel_hard_brake(
                        proxy,
                        elapsed,
                        ego_state,
                        npc_s,
                        gap,
                        x,
                        y,
                        link,
                        speed,
                        "safe_event_region_and_gap",
                    )
                    if not proxy.tunnel_event_result["brake_commanded"]:
                        self._fail(context, "Tunnel brake commands rejected")
                    self._phase(context, "braking", "safe_event_region_and_gap")
                elif proxy.tunnel_event_phase == "cruising":
                    a = context.grpc.set_vehicle_speed_limit(
                        proxy.tunnel_npc, cfg["approach_speed_mps"], enabled=True
                    )
                    b = context.grpc.set_vehicle_velocity(
                        proxy.tunnel_npc, cfg["approach_speed_mps"]
                    )
                    if not (a or b):
                        self._fail(context, "Tunnel closing-gap commands rejected")
                    proxy.tunnel_event_phase = "closing_gap"
                    self._phase(context, "closing_gap", "waiting_for_safe_brake_gap")
        elif proxy.tunnel_event_phase == "braking":
            tunnel.observe_tunnel_braking(
                proxy, elapsed, ego_state, npc_s, gap, x, y, link, speed
            )
            if proxy.tunnel_event_phase == "departing":
                self._phase(
                    context, "departing", proxy.tunnel_event_result["resume_reason"]
                )
            elif proxy.tunnel_event_result.get("stop_confirmed"):
                self._phase(context, "stopped", "stop_confirmed")
        elif proxy.tunnel_event_phase == "departing" and speed > 0.5:
            proxy.tunnel_event_result["resume_motion_verified"] = True
        self._save(context)

    def _cut_in(self, context, sample, gap):
        plan, cfg = context.tunnel_vehicle_plan, context.cfg["tunnel_vehicle_dynamic"]
        phase = context.tunnel_vehicle_phase
        full_s = sample["npc_s_m"] + context.tunnel_vehicle_route_prefix_s
        if phase == "adjacent_cruise":
            if sample["link"] != plan["route_links"][0]:
                self._fail(context, "Cut-in changed lanes before the trigger")
            region = plan["event_region"]
            if full_s > region["end_s_m"]:
                self._fail(context, "Vehicle missed safe cut-in region/gap")
            if (
                full_s >= region["start_s_m"]
                and plan["brake_settings"]["minimum_brake_gap_m"]
                <= gap
                <= plan["brake_settings"]["brake_trigger_gap_m"]
            ):
                target_route = plan["route_links"][1:]
                _, p = nearest_route_pose(
                    context.map_loader, [sample["link"]], sample["x"], sample["y"]
                )
                _, q = nearest_route_pose(
                    context.map_loader, [target_route[0]], sample["x"], sample["y"]
                )
                indices = {
                    sample["link"]: p["waypoint"],
                    target_route[0]: q["waypoint"],
                }
                if not context.grpc.set_vehicle_route(
                    context.tunnel_vehicle_actor,
                    plan["route_links"],
                    decision_range=float(context.cfg.get("npc_decision_range_m", 30.0)),
                    route_waypoint_indices=indices,
                    label="vehicle_cut_in",
                ):
                    self._fail(context, "Cut-in route change rejected")
                context.tunnel_vehicle_cut_in_lateral = sample["lateral_m"]
                context.tunnel_vehicle_cut_in_started = self._now(context)
                self._phase(
                    context, "changing_lane", "legal_mgeo_lane_change_route_assigned"
                )
        elif phase == "changing_lane":
            displacement = abs(
                sample["lateral_m"] - context.tunnel_vehicle_cut_in_lateral
            )
            context.tunnel_vehicle_result["lane_change_lateral_displacement_m"] = (
                displacement
            )
            if (
                sample["link"] in plan["route_links"][1:]
                and abs(sample["lateral_m"]) <= 1.0
                and displacement >= 1.5
            ):
                context.tunnel_vehicle_result["lane_change_verified"] = True
                self._phase(
                    context, "target_lane_cruise", "link_and_lateral_motion_verified"
                )
            elif (
                self._now(context) - context.tunnel_vehicle_cut_in_started
                > cfg["cut_in_timeout_sec"]
            ):
                self._fail(context, "Cut-in lateral motion/link transition timed out")

    def finish_at_ego_goal(self, context):
        """Ego is stopped by the runner; let this same NPC finish its route."""
        if context.tunnel_vehicle_phase == "completed":
            return
        result, plan = context.tunnel_vehicle_result, context.tunnel_vehicle_plan
        result["ego_goal_reached"] = True
        if plan["behavior"] in ("vehicle_stopped", "vehicle_oncoming"):
            return self.finish(context)
        if not result["observed_moving"]:
            return self.finish(context)
        if plan["behavior"] == "vehicle_sudden_brake" and not all(
            result["brake"].get(k)
            for k in (
                "brake_commanded",
                "stop_confirmed",
                "ego_reaction_detected",
                "resumed_after_event",
                "resume_motion_verified",
            )
        ):
            return self.finish(context)
        if plan["behavior"] == "vehicle_cut_in" and not result.get(
            "lane_change_verified"
        ):
            return self.finish(context)
        self._save(context)
        deadline = (
            time.monotonic() + context.cfg["tunnel_vehicle_dynamic"]["max_run_sec"]
        )
        while context.tunnel_vehicle_phase not in ("completed", "failed"):
            sample = self._read_state(context)
            if (
                sample["npc_s_m"] + context.tunnel_vehicle_route_prefix_s
                >= plan["exit_s_m"] - 2.0
            ):
                return self.finish(context)
            if time.monotonic() > deadline:
                self._fail(context, "Tunnel vehicle exit timed out after Ego goal")
            self.update(
                context,
                self._now(context) - context.tunnel_vehicle_started_at,
                context.tunnel_vehicle_last_ego,
                context.tunnel_vehicle_last_ego_s,
            )
            time.sleep(float(context.cfg.get("check_period_sec", 0.05)))

    def finish(self, context):
        if context.tunnel_vehicle_phase == "completed":
            return
        try:
            # Take a fresh observation; never infer arrival from a command ACK.
            result, plan = context.tunnel_vehicle_result, context.tunnel_vehicle_plan
            sample = self._read_state(context)
            cfg = context.cfg["tunnel_vehicle_dynamic"]
            if plan["behavior"] in ("vehicle_stopped", "vehicle_oncoming"):
                self._fail(context, "Route ended before vehicle pass was verified")
            if (
                not result["observed_moving"]
                or result["travelled_m"] < cfg["minimum_observed_travel_m"]
                or sample["npc_s_m"] + context.tunnel_vehicle_route_prefix_s
                < plan["exit_s_m"] - 2.0
            ):
                self._fail(
                    context, "Route ended before vehicle exit/travel was verified"
                )
            if plan["behavior"] == "vehicle_sudden_brake" and not all(
                result["brake"].get(k)
                for k in (
                    "brake_commanded",
                    "stop_confirmed",
                    "ego_reaction_detected",
                    "resumed_after_event",
                    "resume_motion_verified",
                )
            ):
                self._fail(
                    context, "Route ended before complete brake/stop/resume interaction"
                )
            if plan["behavior"] == "vehicle_cut_in" and not result.get(
                "lane_change_verified"
            ):
                self._fail(context, "Route ended before actual lane change")
            self._complete(context, "ego_goal_and_npc_exit_verified")
        except BaseException as error:
            if context.tunnel_vehicle_phase != "failed":
                self.record_interruption(context, error)
            raise


def prepare_tunnel_vehicle(context):
    context.tunnel_vehicle_handler = None
    if getattr(
        context, "scenario_name", None
    ) != "full_loop_tunnel_vehicle" or not context.cfg.get(
        "tunnel_vehicle_dynamic", {}
    ).get(
        "enabled", False
    ):
        return False
    if context.clean_road_mode:
        raise ValueError("tunnel_vehicle_dynamic cannot run with clean_road_mode")
    handler = TunnelVehicleHandler()
    handler.validate(context)
    context.tunnel_vehicle_handler = handler
    return True
