"""Tunnel sudden-brake event functions for the full-loop scenario."""

import json
import math
import os
import time

from ...utils.transform_utils import (
    interpolate_on_polyline,
    polyline_length,
    project_distance_on_polyline,
)


def arm_tunnel_sudden_brake(self):
    cfg = dict(self.cfg.get("tunnel_sudden_brake", {}))
    if not bool(cfg.get("enabled", False)):
        self.tunnel_event_phase = "disabled"
        print("[FullLoop] tunnel sudden-brake event disabled")
        return

    model = str(
        cfg.get(
            "model",
            getattr(
                self,
                "highway_fixed_model",
                self.cfg["highway_npc"].get("model", "DEFAULT_SEDAN"),
            ),
        )
    )
    if model != getattr(self, "highway_fixed_model", model):
        raise RuntimeError(
            "Tunnel NPC must use the same model as highway traffic: "
            f"tunnel={model}, highway={self.highway_fixed_model}"
        )
    if model not in self.vehicle_models():
        raise RuntimeError(f"Configured tunnel NPC model is unavailable: {model}")

    tunnel_zone = self.route_definition["zones"]["tunnel"]
    route = list(
        tunnel_zone.get("npc_route_links", tunnel_zone["route_links"])
    )
    trigger_before = float(cfg.get("spawn_trigger_before_entry_m", 60.0))
    spawn_inside = float(cfg.get("spawn_offset_inside_m", 5.0))
    brake_start_fraction = float(
        cfg.get("brake_zone_start_fraction", 0.62)
    )
    brake_end_fraction = float(
        cfg.get("brake_zone_end_fraction", 0.82)
    )
    if not 0.0 < brake_start_fraction < brake_end_fraction < 1.0:
        raise RuntimeError(
            "Tunnel brake fractions must satisfy "
            f"0 < start < end < 1: {brake_start_fraction}, "
            f"{brake_end_fraction}"
        )

    self.tunnel_npc_model = model
    self.tunnel_npc_route = route
    self.tunnel_spawn_trigger_s = max(
        0.0, self.tunnel_entry_s - trigger_before
    )
    self.tunnel_spawn_s = self.tunnel_entry_s + spawn_inside
    tunnel_length = self.tunnel_exit_s - self.tunnel_entry_s
    self.tunnel_brake_zone_start_s = (
        self.tunnel_entry_s + tunnel_length * brake_start_fraction
    )
    self.tunnel_brake_latest_s = (
        self.tunnel_entry_s + tunnel_length * brake_end_fraction
    )
    goal_margin = float(cfg.get("despawn_goal_margin_m", 30.0))
    self.tunnel_cleanup_s = min(
        self.tunnel_exit_s + float(cfg.get("despawn_after_exit_m", 100.0)),
        self.route_length_m - goal_margin,
    )
    if self.tunnel_spawn_s >= self.tunnel_exit_s:
        raise RuntimeError(
            f"Tunnel NPC spawn is outside tunnel: "
            f"spawn_s={self.tunnel_spawn_s:.1f}, exit_s={self.tunnel_exit_s:.1f}"
        )
    if self.tunnel_cleanup_s <= self.tunnel_exit_s:
        raise RuntimeError(
            f"Tunnel NPC cleanup must be after the tunnel: "
            f"cleanup_s={self.tunnel_cleanup_s:.1f}, "
            f"exit_s={self.tunnel_exit_s:.1f}"
        )
    cleanup_x, cleanup_y, _, _ = interpolate_on_polyline(
        self.route_points, self.tunnel_cleanup_s
    )
    cleanup_error, cleanup_route_index, cleanup_waypoint = (
        self._tunnel_route_start(cleanup_x, cleanup_y)
    )
    self.tunnel_npc_route = route[: cleanup_route_index + 1]
    self.tunnel_cleanup_link_id = self.tunnel_npc_route[-1]
    self.tunnel_cleanup_waypoint = cleanup_waypoint

    planned_initial_gap = self.tunnel_spawn_s - self.tunnel_spawn_trigger_s
    self.tunnel_event_phase = "armed"
    self.tunnel_event_result = {
        "model": model,
        "route_links": route,
        "tunnel_entry_s": self.tunnel_entry_s,
        "tunnel_exit_s": self.tunnel_exit_s,
        "spawn_trigger_s": self.tunnel_spawn_trigger_s,
        "planned_spawn_s": self.tunnel_spawn_s,
        "brake_zone_start_s": self.tunnel_brake_zone_start_s,
        "brake_latest_s": self.tunnel_brake_latest_s,
        "planned_cleanup_s": self.tunnel_cleanup_s,
        "cleanup_link": self.tunnel_cleanup_link_id,
        "cleanup_link_distance_m": float(cleanup_error),
        "planned_initial_gap_m": planned_initial_gap,
        "spawned": False,
        "spawn_verified": False,
        "brake_commanded": False,
        "stop_confirmed": False,
        "ego_reaction_detected": False,
        "destroyed": False,
        "resumed_after_event": False,
        "observed_links": [],
    }
    print(
        f"[FullLoop] tunnel sudden-brake ARMED model={model}, "
        f"tunnel_s={self.tunnel_entry_s:.1f}-{self.tunnel_exit_s:.1f}m, "
        f"spawn_trigger_s={self.tunnel_spawn_trigger_s:.1f}m, "
        f"spawn_s={self.tunnel_spawn_s:.1f}m, "
        f"brake_zone={self.tunnel_brake_zone_start_s:.1f}-"
        f"{self.tunnel_brake_latest_s:.1f}m, "
        f"planned_initial_gap={planned_initial_gap:.1f}m, "
        f"cleanup_s={self.tunnel_cleanup_s:.1f}m"
    )

def _actor_speed(state):
    velocity = state.velocity
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )

def _tunnel_route_start(self, x, y):
    best = None
    for route_index, link_id in enumerate(self.tunnel_npc_route):
        points = self.map_loader.get_link_points(link_id)
        waypoint = self._nearest_waypoint(points, x, y)
        point = points[waypoint]
        distance = math.hypot(float(point[0]) - x, float(point[1]) - y)
        if best is None or distance < best[0]:
            best = (distance, route_index, waypoint)
    return best

def spawn_tunnel_sudden_brake_npc(self, current_s, elapsed):
    cfg = self.cfg["tunnel_sudden_brake"]
    x, y, z, yaw = interpolate_on_polyline(
        self.route_points, self.tunnel_spawn_s
    )
    nearest, route_index, waypoint = self._tunnel_route_start(x, y)
    route = list(self.tunnel_npc_route[route_index:])
    end_points = self.map_loader.get_link_points(route[-1])
    cleanup_x, cleanup_y, _, _ = interpolate_on_polyline(
        self.route_points, self.tunnel_cleanup_s
    )
    cleanup_waypoint = self._nearest_waypoint(
        end_points, cleanup_x, cleanup_y
    )
    speed = float(cfg.get("cruise_speed_mps", 12.0))

    actor = self._spawn_highway_vehicle(
        self.grpc.make_transform(x, y, z, yaw),
        route,
        waypoint,
        "tunnel_sudden_brake",
        speed,
        end_waypoint=cleanup_waypoint,
    )
    if actor is None:
        raise RuntimeError("Failed to spawn tunnel sudden-brake NPC")

    initial_gap = self.tunnel_spawn_s - float(current_s)
    minimum_spawn_gap = float(cfg.get("minimum_spawn_gap_m", 55.0))
    if initial_gap < minimum_spawn_gap:
        try:
            actor.destroy()
        finally:
            if actor in self.scenario_actors:
                self.scenario_actors.remove(actor)
        self.tunnel_event_phase = "failed"
        raise RuntimeError(
            f"Tunnel NPC unsafe spawn gap: {initial_gap:.1f}m "
            f"< {minimum_spawn_gap:.1f}m"
        )

    self.tunnel_npc = actor
    self.tunnel_npc_last_xy = None
    self.tunnel_npc_spawn_elapsed = float(elapsed)
    self.tunnel_npc_brake_wall_time = None
    self.tunnel_npc_stopped_wall_time = None
    self.tunnel_npc_forced_stop = False
    self.tunnel_npc_approach_slowed = False
    self.tunnel_npc_resumed_wall_time = None
    self.tunnel_event_phase = "cruising"
    self.tunnel_event_result.update({
        "spawned": True,
        "spawn_elapsed_sec": float(elapsed),
        "spawn_target": {"x": x, "y": y, "z": z, "yaw_deg": yaw},
        "spawn_route_link_distance_m": float(nearest),
        "initial_route_gap_m": initial_gap,
        "minimum_route_gap_m": initial_gap,
    })
    print(
        f"[FullLoop] tunnel NPC SPAWN model={self.tunnel_npc_model}, "
        f"route={route}, target=({x:.2f},{y:.2f}), "
        f"initial_gap={initial_gap:.1f}m"
    )

def _write_tunnel_event_report(self):
    path = self.cfg.get("tunnel_sudden_brake", {}).get(
        "verification_report_path",
        ".runtime/verification_reports/tunnel_sudden_brake_latest.json",
    )
    if not path:
        return
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    payload = dict(self.tunnel_event_result)
    payload["phase"] = self.tunnel_event_phase
    payload["saved_at_unix"] = time.time()
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)
    print(f"[FullLoop] tunnel event report: {path}")

def _destroy_tunnel_npc(self, reason, current_s):
    actor = self.tunnel_npc
    destroy_ok = False
    if actor is not None:
        try:
            actor.destroy()
            destroy_ok = True
        except Exception as exc:
            print(f"[FullLoop] tunnel NPC destroy warning: {exc}")
        finally:
            if actor in self.scenario_actors:
                self.scenario_actors.remove(actor)
    self.tunnel_npc = None
    self.tunnel_event_phase = "completed"
    self.tunnel_event_result.update({
        "destroy_reason": str(reason),
        "destroyed": destroy_ok,
        "destroy_ego_s": float(current_s),
        "event_success": bool(
            self.tunnel_event_result.get("spawn_verified")
            and self.tunnel_event_result.get("brake_commanded")
            and self.tunnel_event_result.get("stop_confirmed")
            and self.tunnel_event_result.get("ego_reaction_detected")
            and self.tunnel_event_result.get("resumed_after_event")
            and reason == "post_tunnel_cleanup"
            and destroy_ok
        ),
    })
    min_gap = self.tunnel_event_result.get("minimum_route_gap_m")
    print(
        f"[FullLoop] tunnel NPC REMOVED reason={reason}, "
        f"event_success={self.tunnel_event_result['event_success']}, "
        f"ego_reaction={self.tunnel_event_result.get('ego_reaction_detected')}, "
        f"min_route_gap={min_gap}"
    )
    self._write_tunnel_event_report()

def _command_tunnel_hard_brake(
    self, elapsed, ego_state, npc_s, route_gap, x, y, link_id, speed, reason
):
    cfg = self.cfg["tunnel_sudden_brake"]
    minimum_brake_gap = float(cfg.get("minimum_brake_gap_m", 20.0))
    if route_gap < minimum_brake_gap:
        self.tunnel_event_result["brake_gap_below_preferred"] = True
        print(
            f"[FullLoop] tunnel brake gap warning: "
            f"{route_gap:.1f}m < preferred {minimum_brake_gap:.1f}m"
        )
    limit_ok = self.grpc.set_vehicle_speed_limit(
        self.tunnel_npc, 0.0, enabled=True
    )
    brake_ok = self.grpc.set_vehicle_speed(self.tunnel_npc, 0.0)
    self.tunnel_npc_brake_wall_time = time.time()
    self.tunnel_event_phase = "braking"
    ego_speed = float(ego_state.get("speed", 0.0))
    self.tunnel_event_result.update({
        "brake_commanded": bool(limit_ok or brake_ok),
        "brake_trigger_reason": str(reason),
        "brake_command_elapsed_sec": float(elapsed),
        "brake_position": {"x": x, "y": y, "link": link_id},
        "brake_npc_s": npc_s,
        "brake_inside_tunnel_m": npc_s - self.tunnel_entry_s,
        "brake_route_gap_m": route_gap,
        "brake_speed": speed,
        "brake_ego_speed": ego_speed,
        "post_brake_peak_ego_speed": ego_speed,
    })
    print(
        f"[FullLoop] tunnel NPC HARD BRAKE reason={reason}, "
        f"inside={npc_s-self.tunnel_entry_s:.1f}m, "
        f"route_gap={route_gap:.1f}m, speed_before={speed:.1f}, "
        f"command_ok={bool(limit_ok or brake_ok)}"
    )

def _resume_tunnel_npc_after_event(self, elapsed, npc_s, route_gap, reason):
    cfg = self.cfg["tunnel_sudden_brake"]
    resume_speed = float(cfg.get("resume_speed_mps", 10.0))
    pause_ok = self.tunnel_npc.set_pause(False)
    limit_ok = self.grpc.set_vehicle_speed_limit(
        self.tunnel_npc, resume_speed, enabled=True
    )
    speed_ok = self.grpc.set_vehicle_speed(
        self.tunnel_npc, resume_speed
    )
    velocity_ok = self.grpc.set_vehicle_velocity(
        self.tunnel_npc, resume_speed
    )
    resumed = bool(pause_ok or limit_ok or speed_ok or velocity_ok)
    self.tunnel_npc_resumed_wall_time = time.time()
    self.tunnel_event_phase = "departing"
    self.tunnel_event_result.update({
        "resumed_after_event": resumed,
        "resume_reason": str(reason),
        "resume_elapsed_sec": float(elapsed),
        "resume_npc_s": float(npc_s),
        "resume_route_gap_m": float(route_gap),
        "resume_speed_mps": resume_speed,
    })
    print(
        f"[FullLoop] tunnel NPC RESUME reason={reason}, "
        f"route_gap={route_gap:.1f}m, target_speed={resume_speed:.1f}, "
        f"command_ok={resumed}"
    )

def update_tunnel_sudden_brake(self, elapsed, ego_state, current_s):
    phase = getattr(self, "tunnel_event_phase", "disabled")
    if phase in ("disabled", "completed", "failed", "uninitialized"):
        return
    if phase == "armed":
        if float(current_s) >= self.tunnel_spawn_trigger_s:
            self.spawn_tunnel_sudden_brake_npc(current_s, elapsed)
        return
    if self.tunnel_npc is None:
        return

    state = self.tunnel_npc.get_actor_state()
    if state is None:
        self.tunnel_event_result["actor_state_unavailable"] = True
        self._destroy_tunnel_npc("actor_state_unavailable", current_s)
        return

    x = float(state.transform.location.x)
    y = float(state.transform.location.y)
    npc_s = project_distance_on_polyline(self.route_points, x, y)
    route_gap = npc_s - float(current_s)
    euclidean_gap = math.hypot(
        x - float(ego_state["x"]), y - float(ego_state["y"])
    )
    speed = self._actor_speed(state)
    travelled = max(0.0, npc_s - self.tunnel_spawn_s)

    try:
        link_id = state.vehicle_state.current_link_info.id.value
    except Exception:
        link_id = ""
    observed_links = self.tunnel_event_result.setdefault("observed_links", [])
    if link_id and link_id not in observed_links:
        observed_links.append(link_id)

    if not self.tunnel_event_result.get("spawn_verified"):
        target = self.tunnel_event_result["spawn_target"]
        spawn_error = math.hypot(x - target["x"], y - target["y"])
        tolerance = float(
            self.cfg["tunnel_sudden_brake"].get(
                "spawn_position_tolerance_m", 3.0
            )
        )
        self.tunnel_event_result["actual_spawn"] = {
            "x": x, "y": y, "link": link_id,
        }
        self.tunnel_event_result["spawn_error_m"] = spawn_error
        self.tunnel_event_result["spawn_verified"] = bool(
            spawn_error <= tolerance
            and (not link_id or link_id in self.tunnel_npc_route)
        )
        print(
            f"[FullLoop] tunnel NPC SPAWN VERIFY "
            f"error={spawn_error:.2f}m, link={link_id}, "
            f"verified={self.tunnel_event_result['spawn_verified']}"
        )
        if not self.tunnel_event_result["spawn_verified"]:
            self._destroy_tunnel_npc("spawn_verification_failed", current_s)
            raise RuntimeError(
                f"Tunnel NPC spawn verification failed: "
                f"error={spawn_error:.2f}m, link={link_id}"
            )

    if route_gap > 0.0:
        previous_min = float(
            self.tunnel_event_result.get("minimum_route_gap_m", route_gap)
        )
        self.tunnel_event_result["minimum_route_gap_m"] = min(
            previous_min, route_gap
        )
    self.tunnel_event_result.update({
        "last_position": {"x": x, "y": y, "link": link_id},
        "last_npc_s": npc_s,
        "last_route_gap_m": route_gap,
        "last_euclidean_gap_m": euclidean_gap,
        "last_speed": speed,
        "travelled_m": travelled,
    })

    second = int(float(elapsed) - float(self.tunnel_npc_spawn_elapsed))
    if second != self.tunnel_event_last_print_sec:
        self.tunnel_event_last_print_sec = second
        print(
            f"[FullLoop] tunnel event phase={phase}, "
            f"npc_s={npc_s:.1f}m, inside={npc_s-self.tunnel_entry_s:.1f}m, "
            f"travelled={travelled:.1f}m, route_gap={route_gap:.1f}m, "
            f"ego_speed={float(ego_state.get('speed', 0.0)):.1f}, "
            f"npc_speed={speed:.1f}"
        )

    cfg = self.cfg["tunnel_sudden_brake"]
    if phase == "departing":
        if npc_s >= self.tunnel_cleanup_s - 2.0:
            self.tunnel_event_result.update({
                "actual_cleanup_s": float(npc_s),
                "cleanup_after_tunnel_m": float(
                    npc_s - self.tunnel_exit_s
                ),
                "cleanup_position": {
                    "x": x, "y": y, "link": link_id,
                },
            })
            self._destroy_tunnel_npc(
                "post_tunnel_cleanup", current_s
            )
        return

    if phase in ("cruising", "closing_gap"):
        trigger_gap = float(cfg.get("brake_trigger_gap_m", 35.0))
        inside_brake_zone = npc_s >= self.tunnel_brake_zone_start_s
        reached_latest = npc_s >= self.tunnel_brake_latest_s
        gap_ready = 0.0 < route_gap <= trigger_gap
        if inside_brake_zone and (gap_ready or reached_latest):
            reason = (
                "ego_gap_ready" if gap_ready
                else "latest_tunnel_position"
            )
            self._command_tunnel_hard_brake(
                elapsed, ego_state, npc_s, route_gap,
                x, y, link_id, speed, reason,
            )
            return

        if inside_brake_zone and phase == "cruising":
            approach_speed = float(
                cfg.get("approach_speed_mps", 7.0)
            )
            limit_ok = self.grpc.set_vehicle_speed_limit(
                self.tunnel_npc, approach_speed, enabled=True
            )
            velocity_ok = self.grpc.set_vehicle_velocity(
                self.tunnel_npc, approach_speed
            )
            self.tunnel_npc_approach_slowed = True
            self.tunnel_event_phase = "closing_gap"
            self.tunnel_event_result.update({
                "closing_gap_started": True,
                "closing_gap_npc_s": float(npc_s),
                "closing_gap_route_gap_m": float(route_gap),
                "closing_gap_speed_mps": approach_speed,
                "closing_gap_command_ok": bool(limit_ok or velocity_ok),
            })
            print(
                f"[FullLoop] tunnel NPC CLOSING GAP "
                f"inside={npc_s-self.tunnel_entry_s:.1f}m, "
                f"route_gap={route_gap:.1f}m, "
                f"target_speed={approach_speed:.1f}"
            )
        return

    if phase != "braking":
        return

    ego_speed = float(ego_state.get("speed", 0.0))
    peak_speed = max(
        float(
            self.tunnel_event_result.get(
                "post_brake_peak_ego_speed", ego_speed
            )
        ),
        ego_speed,
    )
    self.tunnel_event_result["post_brake_peak_ego_speed"] = peak_speed
    speed_drop = peak_speed - ego_speed
    reaction_gap = float(cfg.get("reaction_max_gap_m", 40.0))
    if (
        0.0 < route_gap <= reaction_gap
        and speed_drop >= float(cfg.get("ego_reaction_speed_drop", 5.0))
        and not self.tunnel_event_result.get("ego_reaction_detected")
    ):
        self.tunnel_event_result.update({
            "ego_reaction_detected": True,
            "ego_reaction_elapsed_sec": float(elapsed),
            "ego_speed_drop": speed_drop,
            "ego_reaction_route_gap_m": float(route_gap),
        })
        print(
            f"[FullLoop] tunnel Ego reaction detected "
            f"speed_drop={speed_drop:.1f}, route_gap={route_gap:.1f}m"
        )

    now = time.time()
    if (
        not self.tunnel_event_result.get("stop_confirmed")
        and speed <= float(cfg.get("stopped_speed_threshold_mps", 0.5))
    ):
        self.tunnel_event_result["stop_confirmed"] = True
        self.tunnel_event_result["stopped_position"] = {
            "x": x, "y": y, "link": link_id,
        }
        self.tunnel_event_result["stopped_npc_s"] = npc_s
        self.tunnel_event_result["stopped_inside_tunnel_m"] = (
            npc_s - self.tunnel_entry_s
        )
        self.tunnel_event_result["stopped_route_gap_m"] = route_gap
        self.tunnel_npc_stopped_wall_time = now
        print(
            f"[FullLoop] tunnel NPC STOP CONFIRMED "
            f"inside={npc_s-self.tunnel_entry_s:.1f}m, "
            f"route_gap={route_gap:.1f}m"
        )

    if (
        not self.tunnel_event_result.get("stop_confirmed")
        and now - float(self.tunnel_npc_brake_wall_time)
            >= float(cfg.get("stop_confirm_timeout_sec", 1.0))
        and not self.tunnel_npc_forced_stop
    ):
        self.tunnel_npc_forced_stop = True
        self.tunnel_event_result["forced_stop"] = True
        self.grpc.stop_vehicle(self.tunnel_npc)
        print("[FullLoop] tunnel NPC forced-stop fallback requested")

    emergency_gap = float(cfg.get("emergency_destroy_gap_m", 8.0))
    if (
        route_gap <= emergency_gap
        and not self.tunnel_event_result.get("ego_reaction_detected")
    ):
        self.tunnel_event_result["emergency_resume"] = True
        self._resume_tunnel_npc_after_event(
            elapsed, npc_s, route_gap, "ego_safety_gap_guard"
        )
        return

    if (
        self.tunnel_event_result.get("stop_confirmed")
        and self.tunnel_event_result.get("ego_reaction_detected")
        and self.tunnel_npc_stopped_wall_time is not None
        and now - self.tunnel_npc_stopped_wall_time
            >= float(cfg.get("stop_hold_sec", 3.0))
    ):
        self._resume_tunnel_npc_after_event(
            elapsed, npc_s, route_gap, "ego_reaction_complete"
        )
