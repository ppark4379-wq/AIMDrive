"""Per-run storage adapter for the controller-independent Event Monitor."""

import json
import hashlib
from pathlib import Path
import re
import time

from .event_monitor import EgoObservation, EventMonitor
from .jsonl import JsonlEventWriter, JsonlObservationWriter
from .incident import IncidentRecorder


class ValidationRunLog:
    def __init__(self, root, scenario_name):
        self.root = Path(root)
        self.scenario_name = scenario_name
        self.monitor = EventMonitor()
        self.run_id = None
        self.output_dir = None
        self.events_path = None
        self.observations_path = None
        self.writer = None
        self.observation_writer = None
        self.observation_count = 0
        self.event_count = 0
        self.closed = False
        self.incident_recorder = None

    def enable_incidents(self, root, scenario, run_options=()):
        self.incident_recorder = IncidentRecorder(root, scenario, run_options)

    def note_route_progress(self, route_s_m):
        if self.incident_recorder is not None:
            self.incident_recorder.note_route_progress(route_s_m)

    def record_planner_sample(self, sample, *, captured_at=None):
        """Optional future adapter hook; the Runner supplies no planner data today."""
        if self.incident_recorder is None:
            raise RuntimeError("Incident recording is not enabled")
        self.incident_recorder.record_planner_sample(sample, captured_at=captured_at)

    def _bind(self, run_id):
        if self.run_id is not None:
            if run_id != self.run_id:
                raise ValueError("Validation run_id changed during one run")
            return
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
            raise ValueError("Validation run_id must be a safe nonempty path name")
        output_dir = self.root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        events_path = output_dir / "validation_events.jsonl"
        events_path.touch(exist_ok=False)
        observations_path = output_dir / "ego_observations.jsonl"
        observations_path.touch(exist_ok=False)
        self.run_id = run_id
        self.output_dir = output_dir
        self.events_path = events_path
        self.observations_path = observations_path
        self.writer = JsonlEventWriter(events_path)
        self.observation_writer = JsonlObservationWriter(observations_path)

    def observe(self, run_id, timestamp, ego_state, *, clock_domain="unknown", lap_index=0,
                actor_provider=None, route_progress_provider=None):
        if self.closed:
            raise RuntimeError("Validation run is already closed")
        self._bind(run_id)
        position = None
        if all(axis in ego_state for axis in ("x", "y", "z")):
            position = (ego_state["x"], ego_state["y"], ego_state["z"])
        collisions = ego_state.get("collision_objects")
        observation = EgoObservation(
            run_id=run_id,
            timestamp=timestamp,
            position_xyz=position,
            speed_mps=ego_state.get("speed_mps"),
            collision_objects=None if collisions is None else tuple(collisions),
            yaw_deg=ego_state.get("yaw_deg"),
            time_basis="lap_elapsed_sec",
            clock_domain=clock_domain,
            lap_index=lap_index,
        )
        sequence = self.observation_count
        write_error = None
        try:
            self.observation_writer.write(sequence, observation)
            self.observation_count += 1
        except OSError as error:
            write_error = error
        events = self.monitor.observe(observation)
        for event in events:
            try:
                self.writer.write(event)
                self.event_count += 1
            except OSError as error:
                write_error = error
        if self.incident_recorder is not None:
            self.incident_recorder.observe(
                sequence, observation, events,
                ego_state=ego_state, actor_provider=actor_provider,
                route_progress_provider=route_progress_provider,
            )
        if write_error is not None:
            raise write_error

    def finish(self, run_id, status, error=None):
        if self.closed:
            return self.output_dir / "summary.json"
        if self.run_id is None:
            self._bind(run_id or "uninitialized_" + str(time.time_ns()))
        summary = {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "status": status,
            "observation_count": self.observation_count,
            "event_count": self.event_count,
            "validation_events_path": str(self.events_path.resolve()),
            "ego_observations_path": str(self.observations_path.resolve()),
            "observation_sha256": hashlib.sha256(self.observations_path.read_bytes()).hexdigest(),
            "event_sha256": hashlib.sha256(self.events_path.read_bytes()).hexdigest(),
        }
        if error is not None:
            summary["error_type"] = type(error).__name__
            summary["error_message"] = str(error)
        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        self.closed = True
        return summary_path
