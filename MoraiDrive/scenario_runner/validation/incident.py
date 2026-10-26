"""Bounded pre-failure evidence and portable incident bundles.

This module observes caller-supplied state. It never controls MORAI or a controller.
"""

from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from .paths import repo_root
from .pipeline_log import read_window, read_ve_audit_window

# Scenario launchers that drive the integrated AIM2026 VE -> Gate -> Planning -> Control stack.
AIM2026_CONTROLLERS = ("aim2026_highway", "aim2026_roundabout", "aim2026_static")
ROUNDABOUT_PREFIX = "full_loop_roundabout"


def _field(value, *names):
    for name in names:
        value = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        if value is None:
            return None
    return value


def snapshot_nearby_actors(grpc, ego_state, radius_m=120.0):
    """Read and normalize nearby live MORAI actor states through existing APIs."""
    actors, errors = [], []
    for kind, method_name in (("vehicle", "get_all_vehicle_states"),
                              ("pedestrian", "get_all_pedestrian_states")):
        method = getattr(grpc, method_name, None)
        if not callable(method):
            errors.append(f"{method_name}_unavailable")
            continue
        try:
            states = list(method())
        except Exception as error:
            errors.append(f"{method_name}: {type(error).__name__}: {error}")
            continue
        for state in states:
            try:
                actor_id = str(_field(state, "actor_info", "id", "value"))
                x = float(_field(state, "transform", "location", "x"))
                y = float(_field(state, "transform", "location", "y"))
                z = float(_field(state, "transform", "location", "z"))
                yaw = float(_field(state, "transform", "rotation", "z"))
                vx = float(_field(state, "velocity", "x"))
                vy = float(_field(state, "velocity", "y"))
                vz = float(_field(state, "velocity", "z"))
                if not actor_id or actor_id == "None" or not all(map(math.isfinite, (x, y, z, yaw, vx, vy, vz))):
                    raise ValueError("invalid actor identity or numeric state")
                if actor_id.lower() == "ego":
                    continue
                ego_x, ego_y = ego_state.get("x"), ego_state.get("y")
                distance = (math.hypot(x - ego_x, y - ego_y)
                            if isinstance(ego_x, (int, float)) and isinstance(ego_y, (int, float))
                            else None)
                if (distance is not None and distance > radius_m
                        and actor_id not in (ego_state.get("collision_objects") or ())):
                    continue
                actors.append({
                    "actor_id": actor_id, "kind": kind,
                    "position_xyz": [x, y, z], "yaw_deg": yaw,
                    "velocity_native": [vx, vy, vz], "velocity_unit": "kmh",
                    "speed_mps": math.sqrt(vx * vx + vy * vy + vz * vz) / 3.6,
                    "distance_to_ego_m": distance,
                    "current_link": _field(state, "vehicle_state", "current_link_info", "id", "value")
                    if kind == "vehicle" else None,
                })
            except (TypeError, ValueError, AttributeError) as error:
                errors.append(f"{kind}_state_unreadable: {error}")
    return {"actors": actors, "errors": errors}


def _code_version():
    package_root = Path(__file__).resolve().parents[1]
    root = repo_root()
    sources = (
        "scenario_runner/main_runner.py",
        "scenario_runner/core/full_loop_drive_base.py",
        "scenario_runner/scenarios/full_loop_scenario.py",
        "scenario_runner/scenarios/full_loop_parts/tunnel_vehicle_dynamic.py",
        "scenario_runner/validation/event_monitor.py",
        "scenario_runner/validation/run_log.py",
        "scenario_runner/validation/incident.py",
        "scenario_runner/validation/incident_rerun.py",
    )
    source_hashes = {}
    for name in sources:
        path = package_root.parent / name
        if path.is_file():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=1, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"commit": commit, "source_sha256": source_hashes}


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def scenario_evidence(scenario):
    """Preserve existing scenario selections without re-evaluating them."""
    result = getattr(scenario, "tunnel_vehicle_result", None)
    if not isinstance(result, dict):
        return {"npc_placement": None, "npc_placement_status": "unavailable"}
    return {
        "npc_placement": {
            "actor_id": result.get("actor_id"),
            "behavior": (result.get("plan") or {}).get("behavior"),
            "model": (result.get("plan") or {}).get("model"),
            "spawn_target": result.get("spawn_target"),
            "actual_spawn": result.get("actual_spawn"),
            "selection": result.get("selection"),
        },
        "npc_placement_status": "observed" if result.get("actual_spawn") else "not_observed",
    }


class IncidentRecorder:
    """Keep 30 seconds of observations for collision and scenario timeout failures."""

    def __init__(self, root, scenario, run_options=(), window_sec=30.0,
                 actor_sample_period_sec=0.2):
        self.root = Path(root)
        self.scenario = scenario
        self.run_options = list(run_options)
        self.window_sec = float(window_sec)
        self.actor_sample_period_sec = float(actor_sample_period_sec)
        self.ego = deque()
        self.actors = deque()
        self.planner = deque()
        self.events = deque()
        self.last_actor_sample_at = float("-inf")
        self.last_route_s_m = None
        self.bundles = []
        self.code_version = _code_version()
        self.pipeline_path = None
        self.latest_observation = None
        self.latest_sequence = None
        self.timeout_recorded = False

    def connect_pipeline(self, path):
        """Read the driver-owned journal on collision; no control-side dependency."""
        self.pipeline_path = Path(path)

    def note_route_progress(self, route_s_m):
        self.last_route_s_m = float(route_s_m)
        if self.ego:
            self.ego[-1]["route_progress_s_m"] = self.last_route_s_m
            self.ego[-1]["route_progress_source"] = "current_observation"

    def record_planner_sample(self, sample, *, captured_at=None):
        """Future VE/Planner adapter: input, trajectory, and command are caller data."""
        if not isinstance(sample, dict):
            raise TypeError("planner sample must be a mapping")
        now = time.monotonic() if captured_at is None else float(captured_at)
        self.planner.append({**sample, "captured_monotonic_sec": now})
        self._trim(now)

    def _trim(self, now):
        limit = now - self.window_sec
        for queue in (self.ego, self.actors, self.planner, self.events):
            while queue and queue[0]["captured_monotonic_sec"] < limit:
                queue.popleft()

    def observe(self, sequence, observation, events, *, ego_state,
                actor_provider=None, route_progress_provider=None, captured_at=None):
        now = time.monotonic() if captured_at is None else float(captured_at)
        self.ego.append({
            "captured_monotonic_sec": now, "sequence": sequence,
            "route_progress_s_m": self.last_route_s_m,
            "route_progress_source": "previous_observation" if self.last_route_s_m is not None else "unavailable",
            **asdict(observation),
        })
        self.latest_observation = observation
        self.latest_sequence = sequence
        collision = [event for event in events if event.event_type == "collision_started"]
        if collision and route_progress_provider is not None:
            try:
                route_s = float(route_progress_provider())
                if not math.isfinite(route_s):
                    raise ValueError("nonfinite route progress")
            except Exception:
                pass
            else:
                self.note_route_progress(route_s)
        if actor_provider is not None and (collision or now - self.last_actor_sample_at >= self.actor_sample_period_sec):
            try:
                snapshot = actor_provider()
                if not isinstance(snapshot, dict) or not isinstance(snapshot.get("actors"), list):
                    raise ValueError("actor provider returned malformed data")
            except Exception as error:
                snapshot = {"actors": [], "errors": [f"actor_provider: {type(error).__name__}: {error}"]}
            self.actors.append({"captured_monotonic_sec": now, "sequence": sequence,
                                "timestamp": observation.timestamp, **snapshot})
            self.last_actor_sample_at = now
        for event in events:
            self.events.append({"captured_monotonic_sec": now, **asdict(event)})
        self._trim(now)
        if collision:
            self._save(sequence, observation, collision, now)

    def capture_timeout(self, run_id, error, *, captured_at=None):
        """Capture known Runner timeout/stall aborts before finalization."""
        if (self.timeout_recorded or type(error) is not RuntimeError or
                self.latest_observation is None or
                self.latest_observation.run_id != run_id):
            return None
        reason = str(error)
        controller = getattr(self.scenario, "_validation_controller", None)
        tunnel_timeout = (reason == "Tunnel vehicle timed out" and
                          getattr(self.scenario, "tunnel_vehicle_result", {}).get("failure_reason") == reason)
        highway_timeout = (controller in AIM2026_CONTROLLERS and
                           reason.startswith("K-City full-loop timeout:"))
        highway_stall = (controller in AIM2026_CONTROLLERS and
                         reason.startswith("EGO progress watchdog abort: reason=no_route_progress "))
        if not (tunnel_timeout or highway_timeout or highway_stall):
            return None
        now = time.monotonic() if captured_at is None else float(captured_at)
        last_ego = self.ego[-1]
        self._trim(now)
        if not self.ego:
            self.ego.append(last_ego)
        self._save(self.latest_sequence, self.latest_observation, [], now,
                   failure_type="scenario_stall" if highway_stall else "scenario_timeout", reason=reason)
        self.timeout_recorded = True
        return self.bundles[-1]

    def _save(self, sequence, observation, collision, now, *,
              failure_type="collision_started", reason=None):
        suffix = ("_timeout" if failure_type == "scenario_timeout" else
                  "_stall" if failure_type == "scenario_stall" else "")
        incident_id = f"{observation.run_id}_{sequence:06d}{suffix}"
        folder = self.root / f"incident_{incident_id}"
        folder.mkdir(parents=True, exist_ok=False)
        scenario = self.scenario
        profile = getattr(scenario, "profile", None)
        cfg = getattr(scenario, "cfg", {})
        runtime_paths = getattr(scenario, "global_cfg", {}).get("paths", {})
        rerun_cfg = dict(cfg)
        rerun_cfg["random_seed"] = getattr(scenario, "run_seed", None)
        actor_errors = [error for row in self.actors for error in row.get("errors", ())]
        actor_count = sum(len(row.get("actors", ())) for row in self.actors)
        pipeline_rows, pipeline_error = (read_window(
            self.pipeline_path, observation.run_id, int(now * 1e9), self.window_sec,
            until_wall_ns=time.time_ns())
            if self.pipeline_path is not None else ([], "not_connected"))
        pipeline_by_stage = {stage: [row for row in pipeline_rows if row.get("stage") == stage]
                             for stage in ("perception", "planning", "control")}
        ve_generation = next((row.get("ve_generation") for row in pipeline_rows
                              if row.get("ve_generation")), None)
        # The VE publisher and controller share one host monotonic clock. The
        # Runner may be in WSL while both run on Windows, so anchor the audit
        # window to the verified controller journal, never to Runner monotonic.
        publisher_until_ns = (pipeline_rows[-1]["timestamp_ns"] if pipeline_rows else int(now * 1e9))
        ve_audit, ve_audit_error = (read_ve_audit_window(
            self.pipeline_path.parent / "performance.jsonl", observation.run_id,
            ve_generation, publisher_until_ns, self.window_sec)
            if self.pipeline_path is not None and ve_generation else ([], "not_connected"))
        if actor_errors:
            actor_status = "partial" if actor_count else "unavailable"
        elif not self.actors:
            actor_status = "not_connected"
        elif actor_count:
            actor_status = "observed"
        else:
            actor_status = "queried_no_actors_observed"
        scenario_result_path = getattr(scenario, "tunnel_vehicle_report_path", None)
        scenario_result_bytes = None
        if scenario_result_path is not None:
            source = Path(scenario_result_path)
            if source.is_file() and source.name == f"{observation.run_id}.json":
                scenario_result_bytes = source.read_bytes()
        manifest = {
            "schema_version": 1,
            "incident_id": incident_id,
            "source_run_id": observation.run_id,
            "scenario_name": getattr(scenario, "_runner_scenario_name", getattr(scenario, "scenario_name", None)),
            "seed": getattr(scenario, "run_seed", None),
            "controller": getattr(scenario, "_validation_controller",
                                  cfg.get("drive_control_mode", "aimdrive_expert")),
            "run_options": self.run_options,
            "effective_scenario_config": rerun_cfg,
            "previous_profile_state": getattr(scenario, "previous_profile_state", None),
            "profile": profile,
            "weather": profile.get("weather") if isinstance(profile, dict) else None,
            "time_hour": getattr(scenario, "scenario_hour", None),
            "code_version": self.code_version,
            "runner_package_root": str(Path(__file__).resolve().parents[1]),
            "controller_code_version": (pipeline_rows[-1].get("controller_code_sha256")
                                        if pipeline_rows else "not_connected"),
            "environment": {
                "mgeo_root": runtime_paths.get("mgeo_root"),
                "grpc_sdk_root": runtime_paths.get("grpc_src"),
                "expert_root": runtime_paths.get("expert_root"),
                "morai_version": "not_recorded",
                "map_content_fingerprint": "not_recorded",
            },
            "failure": {
                "event_type": failure_type,
                "actor_ids": [event.actor_id for event in collision],
                "timestamp": observation.timestamp,
                "time_basis": observation.time_basis,
                "captured_monotonic_sec": now,
                "pipeline_anchor_ns": (pipeline_rows[-1].get("timestamp_ns") if pipeline_rows else None),
                "pipeline_clock_host": (pipeline_rows[-1].get("clock_host") if pipeline_rows else None),
                "pipeline_window_basis": pipeline_error,
                "position_xyz": observation.position_xyz,
                "route_progress_s_m": self.last_route_s_m,
                "route_progress_source": self.ego[-1]["route_progress_source"],
                "observation_sequence": sequence,
                **({"reason": reason,
                    "observation_age_sec": max(0.0, now - self.ego[-1]["captured_monotonic_sec"])}
                   if failure_type in ("scenario_timeout", "scenario_stall") else {}),
            },
            "scenario_evidence": scenario_evidence(scenario),
            "scenario_result_sha256": (hashlib.sha256(scenario_result_bytes).hexdigest()
                                       if scenario_result_bytes is not None else None),
            "data_sources": {
                "ego": "connected",
                "actors": actor_status,
                "actor_sample_count": len(self.actors),
                "actor_observed_count": actor_count,
                "actor_errors": actor_errors,
                "planner": "connected" if self.planner else "not_connected",
                "ve_input": "connected" if any("ve_input" in row for row in self.planner) else "not_connected",
                "planned_trajectory": "connected" if any("planned_trajectory" in row for row in self.planner) else "not_connected",
                "control_command": "connected" if any("control_command" in row for row in self.planner) else "not_connected",
                "pipeline": {stage: "connected" if rows else "unavailable"
                             for stage, rows in pipeline_by_stage.items()},
                "pipeline_error": pipeline_error,
                "ve_publisher_audit": "connected" if ve_audit else "unavailable",
                "ve_publisher_audit_error": ve_audit_error,
                "scenario_result": "copied" if scenario_result_bytes is not None else "not_connected",
            },
            "reproduction": {
                "start_from_seed": "manual_same_command" if getattr(scenario, "_validation_controller", None) in ("ve_gps_udp", *AIM2026_CONTROLLERS) else "supported",
                "restore_failure_instant": "not_implemented",
                "reason": "MORAI snapshot restoration of every actor and controller state is unverified",
            },
        }
        _write_jsonl(folder / "ego_trace.jsonl", self.ego)
        _write_jsonl(folder / "actors_trace.jsonl", self.actors)
        if self.planner:
            _write_jsonl(folder / "planner_trace.jsonl", self.planner)
        for stage, rows in pipeline_by_stage.items():
            if rows:
                _write_jsonl(folder / (stage + "_trace.jsonl"), rows)
        if ve_audit:
            _write_jsonl(folder / "ve_audit_trace.jsonl", ve_audit)
        _write_jsonl(folder / "validation_events.jsonl", self.events)
        if scenario_result_bytes is not None:
            with (folder / "scenario_result.json").open("wb") as stream:
                stream.write(scenario_result_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        if manifest["controller"] == "aim2026_highway":
            rerun_command = shlex.join(["bash", "AIM2026/Expert/run_highway.sh",
                                        "--seed", str(manifest["seed"]), "--validation-events"])
        elif manifest["controller"] == "aim2026_static":
            static_options = list(self.run_options) or ["--seed", str(manifest["seed"])]
            rerun_command = shlex.join(["bash", "AIM2026/Expert/run_static.sh", *static_options])
        elif manifest["controller"] == "aim2026_roundabout":
            variant = str(manifest["scenario_name"] or ROUNDABOUT_PREFIX)[len(ROUNDABOUT_PREFIX):].lstrip("_") or "normal"
            rerun_command = shlex.join(["bash", "AIM2026/Expert/run_roundabout.sh", "--variant", variant,
                                        "--seed", str(manifest["seed"]), "--validation-events"])
        elif manifest["controller"] == "ve_gps_udp":
            gps_options = list(self.run_options)
            if "--seed" not in gps_options and not any(value.startswith("--seed=") for value in gps_options):
                gps_options.extend(["--seed", str(manifest["seed"])])
            if "--validation-events" not in gps_options:
                gps_options.append("--validation-events")
            package = Path(__file__).resolve().parents[1].relative_to(repo_root()).as_posix()
            rerun_command = shlex.join(["bash", package + "/tools/run_ve_full_route.sh", *gps_options])
        else:
            rerun_command = "python3 -m scenario_runner.main_runner --incident-rerun <INCIDENT_DIR> --once --validation-events"
        _write_json(folder / "rerun.json", {
            "status": "not_run",
            "command": rerun_command,
            "comparison_path": (None if manifest["controller"] in ("ve_gps_udp", *AIM2026_CONTROLLERS) else "rerun_result.json"),
            "comparison_status": ("not_implemented_for_gps_route" if manifest["controller"] in ("ve_gps_udp", *AIM2026_CONTROLLERS)
                                  else "available_after_rerun"),
        })
        (folder / "README.txt").write_text(
            f"Incident {incident_id}: {failure_type} with {manifest['failure']['actor_ids']}\n"
            f"Source run: {observation.run_id}; scenario: {manifest['scenario_name']}; seed: {manifest['seed']}\n"
            "manifest.json contains configuration, weather, time, code fingerprints and source status.\n"
            "ego_trace.jsonl and actors_trace.jsonl contain up to 30 seconds before failure.\n"
            "validation_events.jsonl contains Event Monitor events in that window.\n"
            "scenario_result.json is copied when the scenario has a matching run result.\n"
            "planner_trace.jsonl exists only after a VE/Planner adapter supplies samples.\n"
            "From the AIMDrive repository root, with MORAI and matching code/map installed, run:\n"
            + rerun_command + "\n"
            "Replace <INCIDENT_DIR> with this folder's path. Read rerun_result.json for setup comparison.\n"
            "This restarts from the scenario seed and configuration; it does not restore the failure instant.\n",
            encoding="utf-8",
        )
        _write_json(folder / "manifest.json", manifest)
        self.bundles.append(folder)
        # Diagnosis is best-effort: it must never prevent the existing failure
        # abort or invalidate the durable incident bundle.
        try:
            from .diagnose import diagnose_incident
            diagnose_incident(folder)
        except Exception as error:
            print("[Incident] automatic diagnosis failed: " + str(error))
