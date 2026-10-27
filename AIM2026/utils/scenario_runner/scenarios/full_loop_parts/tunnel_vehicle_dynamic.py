"""Runtime tunnel traffic; each behavior keeps its actor through continuation."""

import json
from fnmatch import fnmatchcase
import math
import random
import time
from pathlib import Path
from . import tunnel, vehicle_common
from .highway import _route_geometry
from .tunnel_vehicle_cut_in import TunnelVehicleCutIn, source_pose, validate_cut_in
from .tunnel_vehicle_brake import TunnelVehicleBrakeEvent, required_brake_space
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
        offset_range = cfg.get("spawn_forward_offset_range_m", [0.0, 0.0])
        if not isinstance(offset_range, (list, tuple)) or len(offset_range) != 2:
            raise ValueError("spawn_forward_offset_range_m must contain two distances")
        offset_lo, offset_hi = (
            _finite(value, "spawn_forward_offset_range_m") for value in offset_range
        )
        if not 0 <= offset_lo <= offset_hi:
            raise ValueError(
                "spawn_forward_offset_range_m must satisfy 0 <= min <= max"
            )
        patterns = cfg.get("excluded_model_patterns", [])
        if not isinstance(patterns, list) or any(
            not isinstance(pattern, str) or not pattern.strip() for pattern in patterns
        ):
            raise ValueError(
                "excluded_model_patterns must be a list of nonempty strings"
            )
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
        cfg["spawn_link_timeout_sec"] = _finite(
            cfg.get("spawn_link_timeout_sec", 2.0), "spawn_link_timeout_sec"
        )
        if cfg["spawn_link_timeout_sec"] <= 0:
            raise ValueError("spawn_link_timeout_sec must be positive")
        if not 0 < _finite(cfg["slow_ego_speed_ratio"], "slow_ego_speed_ratio") < 1:
            raise ValueError("slow_ego_speed_ratio must be between zero and one")
        if not cfg.get("event_sections") or any(
            type(i) is not int or i not in (2, 3) for i in cfg["event_sections"]
        ):
            raise ValueError(
                "Vehicle event_sections must be a nonempty subset of [2, 3]"
            )
        if cfg.get("npc_state_velocity_unit", "kmh") not in ("kmh", "mps"):
            raise ValueError("npc_state_velocity_unit must be kmh or mps")
        cfg["cut_in"] = validate_cut_in(cfg["cut_in"])
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
            extras = cfg.get("additional_behaviors") or []
            forbidden = {"vehicle_cut_in"} if "vehicle_oncoming" in extras else set()
            if "vehicle_cut_in" in extras:
                forbidden.add("vehicle_oncoming")
            choices = [
                b for b in requested if b in catalog["behaviors"] and b not in forbidden
            ]
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
        # Draw once before weather/model selection; never redraw on waiting ticks.
        offset = (
            context.random.uniform(offset_lo, offset_hi)
            if offset_hi > offset_lo
            else offset_lo
        )
        plan["spawn_forward_offset_m"] = offset
        plan["spawn_offset_direction"] = "npc_travel_direction"
        if behavior == "vehicle_oncoming":
            nominal_s = plan["spawn_s_m"]
            margin = cfg["spawn_position_tolerance_m"]
            target_s = max(
                nominal_s - offset,
                plan["entry_s_m"] + cfg["minimum_initial_gap_m"] + margin,
            )
            x, y, _, _ = interpolate_on_polyline(
                context.tunnel_vehicle_route_points, target_s
            )
            _, pose = nearest_route_pose(context.map_loader, plan["route_links"], x, y)
            actual_s = project_distance_on_polyline(
                context.tunnel_vehicle_route_points, pose["x"], pose["y"]
            )
            if abs(actual_s - target_s) > margin or not (
                plan["entry_s_m"] < actual_s <= nominal_s + margin
            ):
                raise ValueError(
                    "Oncoming randomized spawn is outside the tunnel corridor"
                )
            plan.update(
                nominal_spawn_s_m=nominal_s,
                spawn_s_m=actual_s,
                spawn_target=pose,
                applied_spawn_forward_offset_m=nominal_s - actual_s,
            )
        if behavior != "vehicle_sudden_brake":
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
            geometry_selection = cfg.get("collection_geometry_selection")
            unavailable_reviewed = []
            if geometry_selection is not None:
                # Collection intersects reviewed geometry with this simulator's
                # model inventory. Ordinary CLI trials keep their old policy.
                approved = geometry_selection.get("approved_models")
                if (geometry_selection.get("schema") != "morai_tunnel_collection_vehicle_geometry_v1"
                        or geometry_selection.get("policy") != "approved_default3d_geometry_intersect_live_available_models"
                        or not isinstance(approved, list) or not approved
                        or not isinstance(configured, list)
                        or set(configured) != set(approved)
                        or set(geometry_selection.get("sources", {})) != set(approved)):
                    raise RuntimeError("Invalid tunnel collection geometry model policy")
                unavailable_reviewed = sorted(set(approved) - set(available))
                configured = sorted(set(approved) & set(available))
            if not configured or any(m not in available for m in configured):
                raise RuntimeError(
                    "Tunnel vehicle model unavailable in configured pool"
                )
            patterns = [
                pattern.casefold() for pattern in cfg.get("excluded_model_patterns", [])
            ]
            excluded = [
                name
                for name in available
                if any(fnmatchcase(name.casefold(), pattern) for pattern in patterns)
            ]
            configured = [name for name in configured if name not in excluded]
            context.tunnel_vehicle_result["model_pool"] = {
                "candidates": list(configured),
                "excluded": excluded,
            }
            if geometry_selection is not None:
                context.tunnel_vehicle_result["model_pool"].update(
                    selection_policy=geometry_selection["policy"],
                    unavailable_reviewed_models=unavailable_reviewed,
                    unreviewed_available_models=sorted(set(available) - set(approved)),
                    geometry_sources={name: geometry_selection["sources"][name] for name in configured},
                )
            if excluded:
                print(f"[TunnelVehicle] excluded non-road models={excluded}")
            if not configured:
                raise RuntimeError("No tunnel vehicle models remain after exclusion")
            model = cfg.get("model")
            if model is None:
                model_rng = random.Random()
                model_rng.setstate(context.tunnel_vehicle_random_state)
                model = (
                    model_rng.choice(list(configured))
                    if cfg.get("random_model", True)
                    else context.cfg["highway_npc"]["model"]
                )
            if model in excluded:
                raise RuntimeError(f"Tunnel vehicle model excluded: {model}")
            if model not in available:
                raise RuntimeError(f"Tunnel vehicle model unavailable: {model}")
            if geometry_selection is not None and model not in configured:
                raise RuntimeError(f"Tunnel collection vehicle has no approved geometry: {model}")
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
        reason = str(error) or type(error).__name__
        brake = result.get("brake", {})
        primary = brake.get("failure_reason") if brake.get("failure_pending") else None
        if primary and reason != primary:
            result["secondary_failure_reason"] = reason
        result["failure_reason"] = primary or reason
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
            raw_speed = tunnel._actor_speed(state)
            velocity_unit = context.cfg["tunnel_vehicle_dynamic"].get(
                "npc_state_velocity_unit", "kmh"
            )
            speed = raw_speed / 3.6 if velocity_unit == "kmh" else raw_speed
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
        direct = plan[
            "behavior"
        ] == "vehicle_cut_in" and context.tunnel_vehicle_phase not in (
            "target_lane_cruise",
            "completed",
            "failed",
        )
        reported_link = link
        now = self._now(context)
        if not direct and not link:
            timeout = context.cfg["tunnel_vehicle_dynamic"]["spawn_link_timeout_sec"]
            if (result["spawn_verified"]
                    or now - context.tunnel_vehicle_spawned_at >= timeout
                    or time.monotonic() - context.tunnel_vehicle_startup_wall_at >= timeout):
                self._fail(context, "Tunnel vehicle reported link unavailable after spawn")
            # The first ActorState may precede native route initialization.
            # Geometry still has to pass the same lane, heading and spawn checks.
            _, pose = nearest_route_pose(context.map_loader, plan["route_links"], x, y)
            link = pose["link"]
        if not direct and link not in plan["route_links"]:
            self._fail(context, f"Tunnel vehicle left its assigned route: {link}")
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
            "raw_velocity_magnitude": raw_speed,
            "state_velocity_unit": velocity_unit,
            "link": link,
            "reported_link": reported_link,
            "link_pending": not direct and not reported_link,
            "actor_id": actor_id,
        }
        if previous is not None and now > previous["time"]:
            sample["position_speed_mps"] = math.dist(
                (x, y), (previous["x"], previous["y"])
            ) / (now - previous["time"])
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
        if direct:
            cfg = context.cfg["tunnel_vehicle_dynamic"]
            source = source_pose(context, full_s)
            source_d = -(source["x"] - rx) * math.sin(angle) + (
                source["y"] - ry
            ) * math.cos(angle)
            lateral = sample["lateral_m"]
            if sample["tangent_dot"] <= 0.5:
                self._fail(context, "Cut-in observed wrong travel direction")
            if context.tunnel_vehicle_phase in (
                "waiting_for_ego",
                "starting",
                "source_corridor_cruise",
            ):
                valid = (
                    math.hypot(x - source["x"], y - source["y"])
                    <= cfg["lane_position_tolerance_m"]
                )
            else:
                valid = (
                    min(0.0, source_d) - cfg["cut_in"]["target_lateral_tolerance_m"]
                    <= lateral
                    <= max(0.0, source_d) + cfg["cut_in"]["target_lateral_tolerance_m"]
                )
            if not valid or not plan["entry_s_m"] <= full_s <= plan["exit_s_m"]:
                self._fail(context, "Cut-in left source-to-target drivable corridor")
            return sample
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

    def _native_route_endpoint(self, context, route):
        """Resolve the native destination on the already planned final link."""
        plan = context.tunnel_vehicle_plan
        try:
            cleanup_links = plan["cleanup_route_links"]
            if not route or list(route) != cleanup_links[-len(route):]:
                raise ValueError("assigned links differ from the planned continuation")
            points = context.map_loader.get_link_points(route[-1])
            if len(points) < 2:
                raise ValueError("final link has insufficient geometry")
            waypoint = len(points) - 1
            x, y, z = (
                _finite(points[waypoint][i], "native_route_endpoint_xyz")
                for i in range(3)
            )
            endpoint_s = _finite(project_distance_on_polyline(
                context.tunnel_vehicle_cleanup_points, x, y
            ), "native_route_endpoint_s_m")
            cleanup_s = _finite(plan["cleanup_route_s_m"], "cleanup_route_s_m")
            if endpoint_s <= cleanup_s:
                raise ValueError("destination does not extend beyond planned cleanup")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            self._fail(context, f"Tunnel vehicle native route endpoint invalid: {error}")
        return {
            "policy": "explicit_last_waypoint_after_cleanup",
            "link": route[-1],
            "waypoint": waypoint,
            "x": x, "y": y, "z": z,
            "npc_route_s_m": endpoint_s,
            "cleanup_route_s_m": cleanup_s,
            "continuation_margin_m": endpoint_s - cleanup_s,
            "assigned_route_links": list(route),
        }

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
        wait_departure = (
            plan["behavior"] == "vehicle_sudden_brake"
            and plan["sudden_brake_settings"]["departure_wait_enabled"]
        )
        spawn_speed = 0.0 if wait_departure else plan["speed_mps"]
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
        if plan["behavior"] == "vehicle_cut_in":
            actor = context.grpc.spawn_vehicle(
                transform=context.grpc.make_transform(
                    p["x"], p["y"], p["z"], p["yaw_deg"]
                ),
                model_name=plan["model"],
                label="vehicle_cut_in",
                velocity=0.0,
                # External steering requires a multi-Ego vehicle, not native NPC AI.
                multi_ego=True,
            )
            if actor is not None:
                context.scenario_actors.append(actor)
                track(actor)
                context.tunnel_vehicle_cut_in = TunnelVehicleCutIn(self, context)
                if not context.grpc.set_vehicle_direct_control(actor, True):
                    self._fail(
                        context,
                        "Cut-in multi-Ego AUTO/cruise/Drive setup unavailable or rejected",
                    )
                if not actor.set_pause(False):
                    self._fail(context, "Cut-in unpause rejected")
                if not context.grpc.set_vehicle_speed_limit(
                    actor, plan["speed_mps"], enabled=True
                ):
                    self._fail(context, "Cut-in speed limit rejected")
                if not context.grpc.control_vehicle_direct(
                    actor, 0.0, plan["speed_mps"]
                ):
                    self._fail(
                        context, "Cut-in initial direct movement command rejected"
                    )
                context.tunnel_vehicle_result["motion_commands"].update(
                    source_route_assigned=False,
                    spawn_multi_ego=True,
                    direct_control_accepted=True,
                    unpause_accepted=True,
                    speed_limit_accepted=True,
                    velocity_accepted=True,
                )
        else:
            route = plan["route_links"][p["route_index"] :]
            endpoint = self._native_route_endpoint(context, route)
            context.tunnel_vehicle_result["motion_commands"]["route_endpoint"] = endpoint
            actor = context._spawn_highway_vehicle(
                context.grpc.make_transform(p["x"], p["y"], p["z"], p["yaw_deg"]),
                route,
                p["waypoint"],
                plan["behavior"],
                spawn_speed,
                end_waypoint=endpoint["waypoint"],
                configure_ai=False,
                unpause=not wait_departure,
                spawn_paused=wait_departure,
                on_spawn=track,
                spawn_velocity=spawn_speed,
                command_report=context.tunnel_vehicle_result["motion_commands"],
            )
        if actor is None:
            self._fail(context, "Failed to spawn tunnel vehicle or assign route")
        context.tunnel_vehicle_spawned_at = self._now(context)
        context.tunnel_vehicle_startup_wall_at = time.monotonic()
        context.tunnel_vehicle_command_speed = spawn_speed
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
        result["spawn_verified"] = not sample["link_pending"]
        print(
            f"[TunnelVehicle] SPAWN actor={sample['actor_id']} link={sample['link']} gap={gap:.1f}m error={error:.2f}m"
        )
        context.tunnel_vehicle_spawn_xy = (sample["x"], sample["y"])
        phases = {
            "vehicle_slow": "moving",
            "vehicle_sudden_brake": "cruising",
            "vehicle_oncoming": "approaching",
            "vehicle_cut_in": "source_corridor_cruise",
        }
        context.tunnel_vehicle_motion_phase = phases[plan["behavior"]]
        context.tunnel_vehicle_motion_monitor_started = self._now(context)
        self._phase(
            context,
            "waiting_for_departure" if wait_departure else "starting",
            "spawn_verified" if result["spawn_verified"] else "waiting_for_reported_spawn_link",
        )
        if plan["behavior"] == "vehicle_sudden_brake":
            context.tunnel_vehicle_brake_event = TunnelVehicleBrakeEvent(self, context)
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
        if context.cfg.get("competition_tunnel_scene") == "vehicle":
            competition_start_s = _finite(
                context.competition_tunnel_start_s, "competition_tunnel_start_s")
            if current_s < competition_start_s:
                return self._waiting(context, {
                    "ego_speed_mps": max(0.0, ego["speed"] / 3.6),
                    "progress_m": full_s - plan["entry_s_m"],
                    "ego_s_m": full_s,
                }, "before_competition_tunnel_start")
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
        if (
            behavior == "vehicle_sudden_brake"
            and v
            < plan["sudden_brake_settings"]["cruise_speed_mps"]
            + plan["sudden_brake_settings"]["minimum_catchup_relative_speed_mps"]
        ):
            return self._waiting(
                context, observation, "ego_not_ready_to_catch_cruising_npc"
            )
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
            nominal_s = full_s + gap
            offset = plan["spawn_forward_offset_m"]
            # Randomization cannot consume the space required for the interaction.
            limit = plan["exit_s_m"] - cfg["encounter_exit_margin_m"]
            if behavior == "vehicle_sudden_brake":
                limit = min(
                    limit,
                    max(region["end_s_m"] for region in plan["brake_regions"]),
                    plan["exit_s_m"]
                    - required_brake_space(
                        plan["sudden_brake_settings"],
                        plan["sudden_brake_settings"]["cruise_speed_mps"],
                    ),
                )
            elif plan["event_region"]:
                limit = min(
                    limit,
                    plan["event_region"]["end_s_m"] - cfg["minimum_event_run_in_m"],
                )
            allowed_offset = max(
                0.0, limit - nominal_s - cfg["spawn_position_tolerance_m"]
            )
            target_s = nominal_s + min(offset, allowed_offset)
            plan["nominal_spawn_s_m"] = nominal_s
            if target_s >= plan["exit_s_m"] - cfg["encounter_exit_margin_m"]:
                self._fail(
                    context, "Insufficient tunnel distance for runtime safe spawn gap"
                )
            x, y, _, _ = interpolate_on_polyline(
                context.tunnel_vehicle_route_points, target_s
            )
            links = (
                plan["source_corridor_links"]
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
            if behavior == "vehicle_cut_in":
                _, _, _, pose["yaw_deg"] = interpolate_on_polyline(
                    context.tunnel_vehicle_route_points, actual_s
                )
            if behavior == "vehicle_slow":
                available = plan["exit_s_m"] - cfg["encounter_exit_margin_m"] - full_s
                speed = min(
                    cfg["slow_speed_mps"],
                    v * cfg["slow_ego_speed_ratio"],
                    v * (1 - gap / available) * 0.85,
                )
            elif behavior == "vehicle_sudden_brake":
                speed = plan["sudden_brake_settings"]["cruise_speed_mps"]
                if actual_s >= max(r["end_s_m"] for r in plan["brake_regions"]) or plan[
                    "exit_s_m"
                ] - actual_s < required_brake_space(
                    plan["sudden_brake_settings"], speed
                ):
                    self._fail(
                        context,
                        "Runtime spawn leaves no stable-follow brake/hold/resume space",
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
                run_in = cfg["minimum_event_run_in_m"]
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
            plan.update(
                spawn_target=pose,
                spawn_s_m=actual_s,
                applied_spawn_forward_offset_m=actual_s - nominal_s,
            )
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
            spawn_s_m=plan["spawn_s_m"],
            initial_gap_m=gap,
            npc_speed_mps=speed,
            nominal_spawn_s_m=plan["nominal_spawn_s_m"],
            requested_spawn_forward_offset_m=plan["spawn_forward_offset_m"],
            applied_spawn_forward_offset_m=plan["applied_spawn_forward_offset_m"],
        )
        context.tunnel_vehicle_result["spawn_timing"] = observation
        print(
            f"[TunnelVehicle] spawn offset_forward={plan['applied_spawn_forward_offset_m']:.2f}m "
            f"requested={plan['spawn_forward_offset_m']:.2f}m gap={gap:.2f}m"
        )
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
        recovery = result["brake"].get("failure_pending")
        if not result["interaction_complete"] and not (
            recovery and result["brake"].get("recovery_motion_verified")
        ):
            return False
        if sample["npc_route_s_m"] < plan["cleanup_route_s_m"]:
            return False
        clearance = math.hypot(
            sample["x"] - ego_state["x"], sample["y"] - ego_state["y"]
        )
        clearance_met = clearance >= cfg["cleanup_ego_clearance_m"]
        scene_boundary = None
        if (
            not clearance_met
            and (not recovery or result["brake"].get("recovery_motion_verified"))
            and getattr(context, "scenario_name", None) == "full_loop_competition"
            and context.cfg.get("competition_tunnel_scene") == "vehicle"
        ):
            goal_s = _finite(
                context.competition_tunnel_goal_s, "competition_tunnel_goal_s"
            )
            ego_s = context.tunnel_vehicle_last_ego_s + (
                context.tunnel_vehicle_route_prefix_s
            )
            if ego_s >= goal_s:
                # The full-course Ego keeps driving after the dedicated tunnel
                # mission's goal. A following Ego may therefore never fall 30 m
                # behind this NPC before its native route ends. End only this
                # scene after the original outside-tunnel continuation. A failed
                # recovery is removed here with its failure outcome preserved;
                # keep actual clearance distinct from the original criterion.
                scene_boundary = {
                    "ego_s_m": ego_s,
                    "ego_goal_s_m": goal_s,
                    "npc_route_s_m": sample["npc_route_s_m"],
                    "npc_cleanup_route_s_m": plan["cleanup_route_s_m"],
                    "outcome": "failed_recovery" if recovery else "verified_interaction",
                }
        if clearance_met or scene_boundary is not None:
            result["cleanup_observation"] = dict(sample)
            result["cleanup_actual_ego_clearance_m"] = clearance
            result["cleanup_required_ego_clearance_m"] = cfg["cleanup_ego_clearance_m"]
            result["cleanup_ego_clearance_met"] = clearance_met
            if scene_boundary is not None:
                result["cleanup_scene_boundary"] = scene_boundary
            self._remove(context)
            result["cleanup_reason"] = (
                ("competition_tunnel_failed_recovery_boundary" if recovery
                 else "competition_tunnel_scene_boundary")
                if scene_boundary is not None
                else "outside_tunnel_continuation_and_ego_clearance"
            )
            if recovery:
                result["completed"] = False
                self._phase(context, "failed", result["failure_reason"])
                self._fail(context, result["failure_reason"])
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
        if not result["spawn_verified"]:
            if sample["link_pending"]:
                return
            result["spawn_verified"] = True
            result["spawn_link_confirmation"] = dict(sample)
            self._save(context)
        behavior = plan["behavior"]
        if context.tunnel_vehicle_phase == "waiting_for_departure":
            context.tunnel_vehicle_brake_event.wait_for_departure(
                elapsed, ego_state, sample, gap
            )
            return
        if not self._check_startup_motion(context, sample):
            if behavior == "vehicle_cut_in":
                # Multi-Ego control needs periodic commands while accelerating.
                context.tunnel_vehicle_cut_in.steer(sample)
            return
        self._approach_speed(context, sample, ego_state, gap)
        if behavior == "vehicle_oncoming":
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
            context.tunnel_vehicle_brake_event.update(elapsed, ego_state, sample, gap)
            if self._brake_verified(result):
                self._complete_interaction(
                    context,
                    "stable_follow_sudden_drop_eight_second_hold_resume_verified",
                )
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

    @staticmethod
    def _brake_verified(result):
        return not result["brake"].get("failure_pending") and all(
            result["brake"].get(k)
            for k in (
                "brake_commanded",
                "stable_following_verified",
                "deceleration_verified",
                "hold_verified",
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

    def _cut_in(self, context, sample, gap):
        context.tunnel_vehicle_cut_in.update(sample, gap)

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
        if (
            plan["behavior"] == "vehicle_sudden_brake"
            and not self._brake_verified(result)
            and not result["brake"].get("failure_pending")
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
            if (
                plan["behavior"] == "vehicle_sudden_brake"
                and not result["brake"].get("failure_pending")
                and not self._brake_verified(result)
            ):
                self._fail(
                    context,
                    "Route ended before complete stable-follow/brake/hold/resume interaction",
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
                    plan["behavior"] == "vehicle_sudden_brake"
                    and not handler._brake_verified(result)
                    and not result["brake"].get("failure_pending")
                ) or (
                    plan["behavior"] == "vehicle_cut_in"
                    and not result.get("lane_change_verified")
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
    scenario_name = getattr(context, "scenario_name", None)
    competition_vehicle = (
        scenario_name == "full_loop_competition"
        and context.cfg.get("competition_tunnel_scene") == "vehicle")
    enabled = context.cfg.get("tunnel_vehicle_dynamic", {}).get("enabled", False)
    if (scenario_name != "full_loop_tunnel_vehicle" and not competition_vehicle) or not enabled:
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
    active_behaviors = {context.tunnel_vehicle_plan["behavior"], *extra}
    if {"vehicle_cut_in", "vehicle_oncoming"} <= active_behaviors:
        raise ValueError(
            "vehicle_cut_in and vehicle_oncoming are mutually exclusive in tunnel traffic"
        )
    entries = [(handler, context)]
    for index, behavior in enumerate(extra, 1):
        owner = _TrafficContext(context, behavior, index)
        additional = TunnelVehicleHandler()
        additional.validate(owner)
        entries.append((additional, owner))
    context.tunnel_vehicle_handler = TunnelVehicleTraffic(entries) if extra else handler
    return True
