"""Runtime tunnel traffic; each behavior keeps its actor through continuation."""

import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from . import tunnel, vehicle_common
from .highway import _route_geometry
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
            "minimum_spawn_ego_speed_mps",
            "minimum_spawn_progress_m",
            "minimum_npc_speed_mps",
            "encounter_gap_m",
            "encounter_exit_margin_m",
            "spawn_reaction_time_sec",
            "spawn_deceleration_mps2",
            "spawn_safety_margin_m",
            "oncoming_eta_tolerance_sec",
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
            "cleanup_after_boundary_m",
            "cleanup_ego_clearance_m",
            "motion_startup_timeout_sec",
            "motion_startup_minimum_travel_m",
        ):
            if _finite(cfg[key], key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < _finite(cfg["slow_ego_speed_ratio"], "slow_ego_speed_ratio") < 1:
            raise ValueError("slow_ego_speed_ratio must be between zero and one")
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
        context.tunnel_vehicle_cleanup_points, _, _ = _route_geometry(
            context, plan["cleanup_route_links"]
        )
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
            "interaction_complete": False,
            "actor_active": False,
            "destroyed": False,
            "actor_id": None,
            "same_actor_continuity": True,
            "observations": [],
            "transitions": [],
            "unavailable_behaviors": context.tunnel_vehicle_catalog[
                "unavailable_behaviors"
            ],
            "motion_commands": {},
            "startup_motion_verified": False,
            "travelled_m": 0.0,
            "brake": {},
            "encounter_verified": False,
            "spawn_policy": "ego_motion_and_runtime_geometry",
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
        context.tunnel_vehicle_spawned_at = None
        context.tunnel_vehicle_wait_log_at = None
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
            self._phase(context, "waiting_for_ego", "runtime_spawn_pending")
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
        # SDK Actor.destroy() discards the RPC result and returns None.
        # The existing adapter returns the actual DestroyActor acknowledgement.
        if not context.grpc.destroy_vehicle_by_id(str(actor.get_id())):
            raise RuntimeError("Tunnel vehicle removal rejected")
        context.tunnel_vehicle_result["cleanup_method"] = "destroy_vehicle_by_id"
        if actor in context.scenario_actors:
            context.scenario_actors.remove(actor)
        context.tunnel_vehicle_actor = None
        context.tunnel_vehicle_result["actor_active"] = False
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
        sample["npc_route_s_m"] = project_distance_on_polyline(
            context.tunnel_vehicle_cleanup_points, x, y
        )
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

    def _spawn(self, context, ego):
        plan = context.tunnel_vehicle_plan
        p = plan["spawn_target"]
        ego_s = project_distance_on_polyline(
            context.tunnel_vehicle_route_points, ego["x"], ego["y"]
        )
        context.tunnel_vehicle_last_ego = dict(ego)
        context.tunnel_vehicle_last_ego_s = (
            ego_s - context.tunnel_vehicle_route_prefix_s
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
            context.tunnel_vehicle_result["actor_active"] = True

        # SetAI is unsupported in the live world. Unpause is independent and
        # movement is verified from subsequent runtime states, without blocking ticks.
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
            unpause=True,
            on_spawn=track,
            spawn_velocity=plan["speed_mps"],
            command_report=context.tunnel_vehicle_result["motion_commands"],
        )
        if actor is None:
            self._fail(context, "Failed to spawn tunnel vehicle or assign route")
        context.tunnel_vehicle_spawned_at = self._now(context)
        context.tunnel_vehicle_startup_wall_at = time.monotonic()
        context.tunnel_vehicle_command_speed = plan["speed_mps"]
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
                "spawn_elapsed_sec": self._now(context)
                - context.tunnel_vehicle_started_at,
                "runtime_timeout_sec": context.tunnel_vehicle_runtime_timeout,
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
            "vehicle_stopped": "cruising_to_stop",
            "vehicle_oncoming": "approaching",
            "vehicle_cut_in": "adjacent_cruise",
        }
        context.tunnel_vehicle_motion_phase = phases[plan["behavior"]]
        context.tunnel_vehicle_motion_monitor_started = self._now(context)
        self._phase(
            context,
            "starting" if plan["speed_mps"] > 0.0 else "stopped",
            "spawn_verified",
        )
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

    def _complete_interaction(self, context, reason):
        result = context.tunnel_vehicle_result
        if not result["interaction_complete"]:
            result.update(interaction_complete=True, interaction_reason=reason)
            self._save(context)

    def _complete(self, context, reason):
        # Interaction completion never implies actor removal.
        self._complete_interaction(context, reason)
        context.tunnel_vehicle_result["completed"] = True
        self._phase(context, "completed", reason)

    def teardown(self, context):
        if context.tunnel_vehicle_actor is not None:
            try:
                self._remove(context)
                context.tunnel_vehicle_result["cleanup_reason"] = "scenario_teardown"
            except Exception as error:
                context.tunnel_vehicle_result["cleanup_error"] = str(error)
                context.tunnel_vehicle_result["cleanup_verified"] = False
                self._save(context)
                raise
            self._save(context)

    def _waiting(self, context, observation, reason):
        result = context.tunnel_vehicle_result
        result["spawn_wait_reason"] = reason
        result["spawn_timing"] = observation
        now = self._now(context)
        last = context.tunnel_vehicle_wait_log_at
        if last is None or now - last >= 1.0:
            context.tunnel_vehicle_wait_log_at = now
            self._save(context)
            print(
                f"[TunnelVehicle] waiting reason={reason} ego_speed={observation['ego_speed_mps']:.2f}m/s progress={observation['progress_m']:.1f}m"
            )
        return False

    def _runtime_spawn(self, context, ego, current_s):
        plan, cfg = context.tunnel_vehicle_plan, context.cfg["tunnel_vehicle_dynamic"]
        values = [ego.get(k) for k in ("x", "y", "speed")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            self._fail(context, "Runtime Ego state is malformed")
        full_s = current_s + context.tunnel_vehicle_route_prefix_s
        projected_s = project_distance_on_polyline(
            context.tunnel_vehicle_route_points, ego["x"], ego["y"]
        )
        rx, ry, _, _ = interpolate_on_polyline(
            context.tunnel_vehicle_route_points, full_s
        )
        if (
            abs(projected_s - full_s) > cfg["spawn_position_tolerance_m"]
            or math.hypot(rx - ego["x"], ry - ego["y"])
            > cfg["spawn_position_tolerance_m"]
        ):
            self._fail(
                context, "Runtime Ego position/progress is outside the tunnel route"
            )
        v = max(
            0.0, ego["speed"] / 3.6
        )  # Runner Ego speed is km/h; NPC commands are m/s.
        progress = full_s - plan["entry_s_m"]
        observation = {"ego_speed_mps": v, "progress_m": progress, "ego_s_m": full_s}
        if full_s >= plan["exit_s_m"] - cfg["encounter_exit_margin_m"]:
            self._fail(context, "Ego passed the available runtime vehicle spawn window")
        # Progress alone is insufficient while stopped: the new actor must be moving.
        if v <= cfg["minimum_npc_speed_mps"] or not (
            v >= cfg["minimum_spawn_ego_speed_mps"]
            or progress >= cfg["minimum_spawn_progress_m"]
        ):
            return self._waiting(context, observation, "ego_not_ready")
        behavior = plan["behavior"]
        if behavior == "vehicle_oncoming":
            pose = plan["spawn_target"]
            spawn_route_s = project_distance_on_polyline(
                context.tunnel_vehicle_cleanup_points, pose["x"], pose["y"]
            )
            options = []
            for section in plan["sections"]:
                if section["section_index"] not in cfg["event_sections"]:
                    continue
                anchor = section["anchor_s_m"]
                if anchor <= full_s:
                    continue
                ax, ay, _, _ = interpolate_on_polyline(
                    context.tunnel_vehicle_route_points, anchor
                )
                _, opposite = nearest_route_pose(
                    context.map_loader, plan["route_links"], ax, ay
                )
                conflict_s = project_distance_on_polyline(
                    context.tunnel_vehicle_cleanup_points, opposite["x"], opposite["y"]
                )
                npc_eta = (conflict_s - spawn_route_s) / cfg["oncoming_speed_mps"]
                ego_eta = (anchor - full_s) / v
                if npc_eta > 0:
                    options.append((abs(ego_eta - npc_eta), anchor, ego_eta, npc_eta))
            if not options:
                self._fail(context, "Oncoming ETA encounter window was missed")
            delta, anchor, ego_eta, npc_eta = min(options)
            observation.update(
                interaction_s_m=anchor,
                ego_eta_sec=ego_eta,
                npc_eta_sec=npc_eta,
                eta_delta_sec=ego_eta - npc_eta,
            )
            if delta > cfg["oncoming_eta_tolerance_sec"]:
                return self._waiting(context, observation, "oncoming_eta_not_aligned")
            plan["interaction_s_m"] = anchor
            gap = plan["spawn_s_m"] - full_s
            speed = cfg["oncoming_speed_mps"]
        else:
            stopping = (
                v * cfg["spawn_reaction_time_sec"]
                + v * v / (2 * cfg["spawn_deceleration_mps2"])
                + cfg["spawn_safety_margin_m"]
            )
            gap = max(
                cfg["initial_spawn_gap_m"], cfg["minimum_initial_gap_m"], stopping
            )
            target_s = full_s + gap
            if target_s >= plan["exit_s_m"] - cfg["encounter_exit_margin_m"]:
                self._fail(
                    context, "Insufficient tunnel distance for runtime safe spawn gap"
                )
            x, y, _, _ = interpolate_on_polyline(
                context.tunnel_vehicle_route_points, target_s
            )
            links = (
                [plan["route_links"][0]]
                if behavior == "vehicle_cut_in"
                else plan["route_links"]
            )
            _, pose = nearest_route_pose(context.map_loader, links, x, y)
            actual_s = project_distance_on_polyline(
                context.tunnel_vehicle_route_points, pose["x"], pose["y"]
            )
            if abs(actual_s - target_s) > cfg["spawn_position_tolerance_m"]:
                self._fail(context, "No MGeo waypoint at runtime spawn position")
            gap = actual_s - full_s
            if behavior == "vehicle_slow":
                available = plan["exit_s_m"] - cfg["encounter_exit_margin_m"] - full_s
                speed = min(
                    cfg["slow_speed_mps"],
                    v * cfg["slow_ego_speed_ratio"],
                    v * (1 - gap / available) * 0.85,
                )
            else:
                speed = min(
                    (
                        cfg["cut_in_speed_mps"]
                        if behavior == "vehicle_cut_in"
                        else cfg["cruise_speed_mps"]
                    ),
                    v,
                )
            if speed < cfg["minimum_npc_speed_mps"]:
                return self._waiting(context, observation, "ego_too_slow_for_encounter")
            region = plan["event_region"]
            if region:
                run_in = (
                    5.0
                    if behavior == "vehicle_stopped"
                    else cfg["minimum_event_run_in_m"]
                )
                event_start = max(region["start_s_m"], actual_s + run_in)
                if event_start >= region["end_s_m"]:
                    self._fail(
                        context, "Runtime spawn missed safe vehicle event run-in"
                    )
                plan["event_region"] = {
                    **region,
                    "start_s_m": event_start,
                    "anchor_s_m": max(region["anchor_s_m"], event_start),
                }
            plan.update(spawn_target=pose, spawn_s_m=actual_s)
        if gap < cfg["minimum_initial_gap_m"]:
            return self._waiting(context, observation, "insufficient_initial_gap")
        # Independently scheduled traffic must not spawn on an existing actor.
        for other in context.scenario_actors:
            other_state = other.get_actor_state()
            if other_state is not None:
                loc = other_state.transform.location
                if (
                    math.hypot(pose["x"] - loc.x, pose["y"] - loc.y)
                    < cfg["minimum_initial_gap_m"]
                ):
                    return self._waiting(
                        context, observation, "existing_traffic_at_spawn"
                    )
        plan["spawn_preview_only"] = False
        plan["speed_mps"] = speed
        observation.update(
            spawn_s_m=plan["spawn_s_m"], initial_gap_m=gap, npc_speed_mps=speed
        )
        context.tunnel_vehicle_result["spawn_timing"] = observation
        context.tunnel_vehicle_runtime_timeout = max(
            cfg["max_run_sec"],
            (
                plan["cleanup_route_s_m"]
                - project_distance_on_polyline(
                    context.tunnel_vehicle_cleanup_points, pose["x"], pose["y"]
                )
            )
            / speed
            + 30.0,
        )
        self._spawn(context, ego)
        return True

    def _approach_speed(self, context, sample, ego, gap):
        plan, result, cfg = (
            context.tunnel_vehicle_plan,
            context.tunnel_vehicle_result,
            context.cfg["tunnel_vehicle_dynamic"],
        )
        behavior = plan["behavior"]
        if behavior == "vehicle_slow":
            if (
                0 <= gap <= cfg["encounter_gap_m"]
                and math.hypot(sample["x"] - ego["x"], sample["y"] - ego["y"])
                <= cfg["encounter_gap_m"] + cfg["lane_position_tolerance_m"]
            ):
                result["encounter_verified"] = True
                result.setdefault("encounter_observation", dict(sample))
            target = (
                plan["speed_mps"]
                if result["encounter_verified"]
                else min(
                    plan["speed_mps"],
                    max(0.0, ego["speed"] / 3.6) * cfg["slow_ego_speed_ratio"],
                )
            )
        elif behavior == "vehicle_sudden_brake" and context.tunnel_vehicle_phase in (
            "cruising",
            "closing_gap",
        ):
            cap = (
                cfg["cruise_speed_mps"]
                if context.tunnel_vehicle_phase == "cruising"
                else plan["brake_settings"]["approach_speed_mps"]
            )
            ratio = 1.0 if context.tunnel_vehicle_phase == "cruising" else 0.75
            target = min(cap, max(0.0, ego["speed"] / 3.6) * ratio)
        else:
            return
        if abs(target - context.tunnel_vehicle_command_speed) < 0.2:
            return
        a = context.grpc.set_vehicle_speed_limit(
            context.tunnel_vehicle_actor, target, enabled=True
        )
        b = context.grpc.set_vehicle_velocity(context.tunnel_vehicle_actor, target)
        if not (a and b):
            self._fail(context, "Runtime approach speed command rejected")
        context.tunnel_vehicle_command_speed = target
        result["approach_speed_mps"] = target

    def _cleanup_if_ready(self, context, sample, ego_state):
        plan, cfg, result = (
            context.tunnel_vehicle_plan,
            context.cfg["tunnel_vehicle_dynamic"],
            context.tunnel_vehicle_result,
        )
        if plan["behavior"] == "vehicle_stopped" or not result["interaction_complete"]:
            return False
        if (
            sample["npc_route_s_m"] >= plan["cleanup_route_s_m"]
            and math.hypot(sample["x"] - ego_state["x"], sample["y"] - ego_state["y"])
            >= cfg["cleanup_ego_clearance_m"]
        ):
            result["cleanup_observation"] = dict(sample)
            self._remove(context)
            result["cleanup_reason"] = "outside_tunnel_continuation_and_ego_clearance"
            self._complete(context, result["cleanup_reason"])
            return True
        return False

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
        context.tunnel_vehicle_last_ego = dict(ego_state)
        context.tunnel_vehicle_last_ego_s = current_s
        if context.tunnel_vehicle_actor is None:
            if not self._runtime_spawn(context, ego_state, current_s):
                return
        age = self._now(context) - context.tunnel_vehicle_spawned_at
        if age > context.tunnel_vehicle_runtime_timeout:
            self._fail(context, "Tunnel vehicle timed out")
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
        if not self._check_startup_motion(context, sample):
            return
        self._approach_speed(context, sample, ego_state, gap)
        if behavior == "vehicle_stopped":
            self._stopped(context, sample, ego_state, gap)
        elif behavior == "vehicle_oncoming":
            if (
                result["travelled_m"] >= cfg["minimum_observed_travel_m"]
                and -gap >= cfg["pass_clearance_m"]
                and math.hypot(x - ego_state["x"], y - ego_state["y"])
                >= cfg["pass_clearance_m"]
            ):
                result["passed"] = True
                self._complete_interaction(context, "oncoming_pass_clearance_verified")
                self._phase(context, "passed", "continuing_to_entrance_side_cleanup")
        elif behavior == "vehicle_sudden_brake":
            self._brake(context, elapsed, ego_state, npc_s, gap, x, y, link, speed)
            if self._brake_verified(result):
                self._complete_interaction(context, "brake_stop_hold_resume_verified")
        elif behavior == "vehicle_cut_in":
            self._cut_in(context, sample, gap)
            if result.get("lane_change_verified"):
                self._complete_interaction(context, "lane_change_verified")
        elif (
            result["travelled_m"] >= cfg["minimum_observed_travel_m"]
            and sample["npc_route_s_m"] >= plan["cleanup_boundary_route_s_m"]
        ):
            if not result["encounter_verified"]:
                self._fail(context, "Slow vehicle reached exit before Ego encounter")
            self._complete_interaction(
                context, "continuous_travel_and_ego_encounter_verified"
            )
        self._cleanup_if_ready(context, sample, ego_state)

    def _stopped(self, context, sample, ego_state, gap):
        plan, result = context.tunnel_vehicle_plan, context.tunnel_vehicle_result
        settings = plan["brake_settings"]
        now = self._now(context)
        if context.tunnel_vehicle_phase == "cruising_to_stop":
            full_s = sample["npc_s_m"] + context.tunnel_vehicle_route_prefix_s
            region = plan["event_region"]
            if full_s > region["end_s_m"]:
                self._fail(context, "Vehicle missed stationary event region")
            if full_s < region["anchor_s_m"]:
                return
            if not context.grpc.stop_vehicle(context.tunnel_vehicle_actor):
                self._fail(context, "Stopped vehicle stop command rejected")
            context.tunnel_vehicle_stop_requested_at = now
            self._phase(context, "stopping", "selected_event_anchor_reached")
            return
        if context.tunnel_vehicle_phase == "stopping":
            if sample["speed_mps"] <= settings["stopped_speed_threshold_mps"]:
                full_s = sample["npc_s_m"] + context.tunnel_vehicle_route_prefix_s
                region = plan["event_region"]
                if not region["start_s_m"] <= full_s <= region["end_s_m"]:
                    self._fail(context, "Vehicle stopped outside selected event region")
                result["stopped_verified"] = True
                result["stationary_position"] = {k: sample[k] for k in ("x", "y", "z")}
                self._phase(context, "stopped", "actual_stationary_state_verified")
            elif (
                now - context.tunnel_vehicle_stop_requested_at
                >= settings["stop_confirm_timeout_sec"]
            ):
                self._fail(context, "Stopped vehicle failed to stop at event anchor")
            else:
                return
        position = result["stationary_position"]
        if (
            sample["speed_mps"] > settings["stopped_speed_threshold_mps"]
            or math.hypot(sample["x"] - position["x"], sample["y"] - position["y"])
            > 1.0
        ):
            self._fail(context, "Stopped vehicle moved after stationary verification")
        clearance = context.cfg["tunnel_vehicle_dynamic"]["pass_clearance_m"]
        if (
            -gap >= clearance
            and math.hypot(sample["x"] - ego_state["x"], sample["y"] - ego_state["y"])
            >= clearance
        ):
            result["ego_passed"] = True
            self._complete_interaction(context, "ego_passed_stopped_vehicle")
            self._phase(context, "parked_after_pass", "actor_retained_until_teardown")

    @staticmethod
    def _brake_verified(result):
        return all(
            result["brake"].get(k)
            for k in (
                "brake_commanded",
                "stop_confirmed",
                "ego_reaction_detected",
                "resumed_after_event",
                "resume_motion_verified",
            )
        )

    def _check_startup_motion(self, context, sample):
        result = context.tunnel_vehicle_result
        if result["startup_motion_verified"]:
            return True
        cfg = context.cfg["tunnel_vehicle_dynamic"]
        now = self._now(context)
        if context.tunnel_vehicle_motion_monitor_started is None:
            context.tunnel_vehicle_motion_monitor_started = now
        spawn = result["actual_spawn"]
        displacement = math.hypot(sample["x"] - spawn["x"], sample["y"] - spawn["y"])
        result["startup_displacement_m"] = displacement
        if (
            displacement >= cfg["motion_startup_minimum_travel_m"]
            or sample["speed_mps"] > 0.5
        ):
            result["startup_motion_verified"] = True
            result["observed_moving"] = True
            result["startup_observation"] = dict(sample)
            self._phase(
                context,
                context.tunnel_vehicle_motion_phase,
                "actual_startup_motion_verified",
            )
            return True
        if (
            now - context.tunnel_vehicle_motion_monitor_started
            >= cfg["motion_startup_timeout_sec"]
            or time.monotonic() - context.tunnel_vehicle_startup_wall_at
            >= cfg["motion_startup_timeout_sec"]
        ):
            self._fail(
                context,
                "Tunnel vehicle startup motion not observed: "
                f"speed={sample['speed_mps']:.2f}m/s displacement={displacement:.2f}m "
                f"commands={result['motion_commands']}",
            )
        return False

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
                        proxy.tunnel_npc,
                        min(
                            cfg["approach_speed_mps"],
                            max(0.0, ego_state["speed"] / 3.6) * 0.75,
                        ),
                        enabled=True,
                    )
                    b = context.grpc.set_vehicle_velocity(
                        proxy.tunnel_npc,
                        min(
                            cfg["approach_speed_mps"],
                            max(0.0, ego_state["speed"] / 3.6) * 0.75,
                        ),
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
                context.tunnel_vehicle_resume_xy = (x, y)
                context.tunnel_vehicle_resume_at = self._now(context)
                self._phase(
                    context, "departing", proxy.tunnel_event_result["resume_reason"]
                )
            elif proxy.tunnel_event_result.get("stop_confirmed"):
                self._phase(context, "stopped", "stop_confirmed")
        elif proxy.tunnel_event_phase == "departing":
            if not proxy.tunnel_event_result.get("resume_motion_verified"):
                displacement = math.dist((x, y), context.tunnel_vehicle_resume_xy)
                proxy.tunnel_event_result["resume_displacement_m"] = displacement
                if (
                    speed > 0.5
                    or displacement
                    >= context.cfg["tunnel_vehicle_dynamic"][
                        "motion_startup_minimum_travel_m"
                    ]
                ):
                    proxy.tunnel_event_result["resume_motion_verified"] = True
                elif (
                    self._now(context) - context.tunnel_vehicle_resume_at
                    >= context.cfg["tunnel_vehicle_dynamic"][
                        "motion_startup_timeout_sec"
                    ]
                ):
                    self._fail(context, "Tunnel vehicle resumed movement not observed")
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
        """Keep the same moving NPC alive through its outside-tunnel continuation."""
        if context.tunnel_vehicle_phase == "completed":
            return
        result, plan = context.tunnel_vehicle_result, context.tunnel_vehicle_plan
        result["ego_goal_reached"] = True
        if not result["spawned"]:
            self._fail(
                context, "Ego goal reached before runtime traffic spawn/interaction"
            )
        if plan["behavior"] == "vehicle_stopped":
            return self.finish(context)
        if (
            plan["behavior"] == "vehicle_sudden_brake"
            and not self._brake_verified(result)
        ) or (
            plan["behavior"] == "vehicle_cut_in"
            and not result.get("lane_change_verified")
        ):
            return self.finish(context)
        deadline = time.monotonic() + context.tunnel_vehicle_runtime_timeout
        while context.tunnel_vehicle_phase not in ("completed", "failed"):
            if time.monotonic() > deadline:
                self._fail(
                    context,
                    "Tunnel vehicle continuation cleanup timed out after Ego goal",
                )
            self.update(
                context,
                self._now(context) - context.tunnel_vehicle_started_at,
                context.tunnel_vehicle_last_ego,
                context.tunnel_vehicle_last_ego_s,
            )
            if context.tunnel_vehicle_phase != "completed":
                time.sleep(float(context.cfg.get("check_period_sec", 0.05)))

    def finish(self, context):
        if context.tunnel_vehicle_phase == "completed":
            return
        try:
            result, plan = context.tunnel_vehicle_result, context.tunnel_vehicle_plan
            sample = self._read_state(context)
            if plan["behavior"] == "vehicle_stopped":
                if (
                    not result.get("stopped_verified")
                    or not result["interaction_complete"]
                ):
                    self._fail(
                        context,
                        "Route ended before stationary vehicle/pass was verified",
                    )
                position = result["stationary_position"]
                if (
                    sample["speed_mps"]
                    > plan["brake_settings"]["stopped_speed_threshold_mps"]
                    or math.hypot(
                        sample["x"] - position["x"], sample["y"] - position["y"]
                    )
                    > 1.0
                ):
                    self._fail(context, "Stopped vehicle moved before teardown")
                self._complete(context, "stationary_vehicle_retained_until_teardown")
                return
            if plan["behavior"] == "vehicle_sudden_brake" and not self._brake_verified(
                result
            ):
                self._fail(
                    context, "Route ended before complete brake/stop/resume interaction"
                )
            if plan["behavior"] == "vehicle_cut_in" and not result.get(
                "lane_change_verified"
            ):
                self._fail(context, "Route ended before actual lane change")
            if not self._cleanup_if_ready(
                context, sample, context.tunnel_vehicle_last_ego
            ):
                self._fail(
                    context,
                    "Route ended before vehicle exit/travel and outside-tunnel cleanup were verified",
                )
        except BaseException as error:
            if context.tunnel_vehicle_phase != "failed":
                self.record_interruption(context, error)
            raise


class _TrafficContext:
    """Isolate each actor's lifecycle while sharing the world, clock and ownership."""

    def __init__(self, parent, behavior, index):
        self._parent = parent
        vehicle_cfg = {
            **parent.cfg["tunnel_vehicle_dynamic"],
            "behavior": behavior,
            "random_behavior": False,
            "additional_behaviors": [],
        }
        self.cfg = {**parent.cfg, "tunnel_vehicle_dynamic": vehicle_cfg}
        self.run_instance_id = f"{parent.run_instance_id}_traffic_{index}"
        self._spawn_highway_vehicle = (
            lambda *args, **kw: vehicle_common._spawn_highway_vehicle(self, *args, **kw)
        )

    def __getattr__(self, name):
        return getattr(self._parent, name)


class TunnelVehicleTraffic:
    """Independent actors may coexist; no actor is replaced at section boundaries."""

    def __init__(self, entries):
        self.entries = entries

    def _call(self, name, *args):
        try:
            for handler, context in self.entries:
                getattr(handler, name)(context, *args)
        except BaseException as error:
            self.record_interruption(None, error)
            raise

    def arm(self, context):
        self._call("arm")

    def update(self, context, elapsed, ego, current_s):
        self._call("update", elapsed, ego, current_s)

    def finish_at_ego_goal(self, context):
        # Continue observing all live traffic while any one actor finishes.
        # Serial per-actor waits would miss another actor's stop/resume event.
        try:
            for handler, owner in self.entries:
                result, plan = owner.tunnel_vehicle_result, owner.tunnel_vehicle_plan
                result["ego_goal_reached"] = True
                if owner.tunnel_vehicle_phase == "completed":
                    continue
                if not result["spawned"]:
                    handler._fail(
                        owner,
                        "Ego goal reached before runtime traffic spawn/interaction",
                    )
                if (
                    plan["behavior"] == "vehicle_stopped"
                    or (
                        plan["behavior"] == "vehicle_sudden_brake"
                        and not handler._brake_verified(result)
                    )
                    or (
                        plan["behavior"] == "vehicle_cut_in"
                        and not result.get("lane_change_verified")
                    )
                ):
                    handler.finish(owner)
            deadline = time.monotonic() + max(
                owner.tunnel_vehicle_runtime_timeout for _, owner in self.entries
            )
            while any(
                owner.tunnel_vehicle_phase != "completed" for _, owner in self.entries
            ):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Tunnel traffic continuation cleanup timed out")
                for handler, owner in self.entries:
                    handler.update(
                        owner,
                        handler._now(owner) - owner.tunnel_vehicle_started_at,
                        owner.tunnel_vehicle_last_ego,
                        owner.tunnel_vehicle_last_ego_s,
                    )
                if any(
                    owner.tunnel_vehicle_phase != "completed"
                    for _, owner in self.entries
                ):
                    time.sleep(float(context.cfg.get("check_period_sec", 0.05)))
        except BaseException as error:
            self.record_interruption(context, error)
            raise

    def record_interruption(self, context, error):
        for handler, owner in self.entries:
            handler.record_interruption(owner, error)

    def teardown(self, context):
        first_error = None
        for handler, owner in self.entries:
            try:
                handler.teardown(owner)
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error


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
    extra = context.cfg["tunnel_vehicle_dynamic"].get("additional_behaviors", [])
    if not isinstance(extra, list) or any(b not in BEHAVIORS for b in extra):
        raise ValueError(
            "additional_behaviors must be a list of supported behavior names"
        )
    entries = [(handler, context)]
    for index, behavior in enumerate(extra, 1):
        owner = _TrafficContext(context, behavior, index)
        additional = TunnelVehicleHandler()
        additional.validate(owner)
        entries.append((additional, owner))
    context.tunnel_vehicle_handler = TunnelVehicleTraffic(entries) if extra else handler
    return True
