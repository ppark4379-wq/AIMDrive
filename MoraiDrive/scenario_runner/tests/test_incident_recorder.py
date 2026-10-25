"""Incident capture and seed/config rerun preparation without MORAI."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys

from ..validation.event_monitor import EgoObservation, ValidationEvent
from ..validation.incident import IncidentRecorder, snapshot_nearby_actors
from ..validation.incident_rerun import compare_rerun, load_incident, prepare_config
from ..validation.finish import finish_validation_run
from ..validation.run_log import ValidationRunLog
from .validation_fakes import FakeDrive, ego


RUN_ID = "123456789_0000002a"


def fake_scenario():
    return SimpleNamespace(
        run_instance_id=RUN_ID, run_seed=42, scenario_name="full_loop_tunnel_vehicle",
        cfg={"drive_control_mode": "aimdrive_expert", "profile_state_path": "/old/profile.json",
             "max_laps": 1, "progress_watchdog": {"enabled": False}},
        profile={"name": "scenario_sunny", "weather": "SUNNY"},
        scenario_hour=11, previous_profile_state={"weather": "FOGGY", "time_hour": 13},
        tunnel_vehicle_result={
            "plan": {"behavior": "vehicle_slow", "model": "Sedan"},
            "actual_spawn": {"x": 1.0, "y": 2.0},
            "spawn_target": {"x": 1.0, "y": 2.0},
            "actor_id": "npc-7", "selection": {"model": "Sedan"},
        },
    )


def observation(timestamp, collisions=()):
    return EgoObservation(
        RUN_ID, timestamp, position_xyz=(1.0, 2.0, 3.0), speed_mps=4.0,
        collision_objects=tuple(collisions), yaw_deg=45.0,
        time_basis="lap_elapsed_sec", clock_domain="host_monotonic", lap_index=0,
    )


class IncidentRecorderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "incidents"
        self.scenario = fake_scenario()

    def capture(self, *, planner=False):
        recorder = IncidentRecorder(self.root, self.scenario, ["--scenario", "full_loop_tunnel_vehicle"],
                                    window_sec=30.0)
        provider = lambda: {"actors": [{"actor_id": "npc-7", "position_xyz": [4, 5, 0]}],
                            "errors": []}
        for sequence, tick in enumerate((0.0, 10.0, 29.0)):
            recorder.observe(sequence, observation(tick), [], ego_state={},
                             actor_provider=provider, captured_at=tick)
            recorder.note_route_progress(tick * 2)
        if planner:
            recorder.record_planner_sample({
                "ve_input": {"objects": []},
                "planned_trajectory": [[1.0, 2.0]],
                "control_command": {"steer": 0.1},
            }, captured_at=30.0)
        collision = ValidationEvent(31.0, "collision_started", RUN_ID, "npc-7")
        recorder.observe(3, observation(31.0, ["npc-7"]), [collision], ego_state={},
                         actor_provider=provider, captured_at=31.0)
        self.assertEqual(len(recorder.bundles), 1)
        return recorder.bundles[0]

    def test_collision_bundle_keeps_preceding_window_and_metadata(self):
        bundle = self.capture()
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        ego = [json.loads(row) for row in (bundle / "ego_trace.jsonl").read_text().splitlines()]
        actors = [json.loads(row) for row in (bundle / "actors_trace.jsonl").read_text().splitlines()]
        events = [json.loads(row) for row in (bundle / "validation_events.jsonl").read_text().splitlines()]
        self.assertEqual([row["timestamp"] for row in ego], [10.0, 29.0, 31.0])
        self.assertEqual(len(actors), 3)
        self.assertEqual(events[0]["actor_id"], "npc-7")
        self.assertEqual(manifest["failure"]["route_progress_s_m"], 58.0)
        self.assertEqual(manifest["failure"]["route_progress_source"], "previous_observation")
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["weather"], "SUNNY")
        self.assertEqual(manifest["time_hour"], 11)
        self.assertEqual(manifest["effective_scenario_config"]["random_seed"], 42)
        self.assertEqual(manifest["data_sources"]["planner"], "not_connected")
        self.assertFalse((bundle / "planner_trace.jsonl").exists())
        self.assertEqual(manifest["reproduction"]["restore_failure_instant"], "not_implemented")
        self.assertIn("--incident-rerun", (bundle / "README.txt").read_text())

    def test_planner_adapter_records_only_supplied_samples(self):
        bundle = self.capture(planner=True)
        manifest = load_incident(bundle)
        planner = [json.loads(row) for row in (bundle / "planner_trace.jsonl").read_text().splitlines()]
        self.assertEqual(planner[0]["control_command"], {"steer": 0.1})
        for name in ("planner", "ve_input", "planned_trajectory", "control_command"):
            self.assertEqual(manifest["data_sources"][name], "connected")

    def test_matching_scenario_result_is_copied_into_bundle(self):
        source = self.root.parent / "tunnel_vehicle_runs" / f"{RUN_ID}.json"
        source.parent.mkdir()
        source.write_text('{"phase":"failed","actor_id":"npc-7"}\n', encoding="utf-8")
        self.scenario.tunnel_vehicle_report_path = source
        bundle = self.capture()
        self.assertEqual((bundle / "scenario_result.json").read_bytes(), source.read_bytes())
        manifest = load_incident(bundle)
        self.assertEqual(manifest["data_sources"]["scenario_result"], "copied")

    def test_timeout_creates_separate_bundle_and_final_diagnosis(self):
        reason = "Tunnel vehicle timed out"
        self.scenario.tunnel_vehicle_result["failure_reason"] = reason
        source = self.root.parent / "tunnel_vehicle_runs" / f"{RUN_ID}.json"
        source.parent.mkdir()
        source.write_text(json.dumps({"run_seed": 42, "phase": "failed",
            "failure_reason": reason, "encounter_verified": False,
            "plan": {"behavior": "vehicle_slow", "speed_mps": 1.2},
            "last_observation": {"route_gap_m": 28.0},
            "approach_speed_mps": 0.1}) + "\n", encoding="utf-8")
        self.scenario.tunnel_vehicle_report_path = source
        log = ValidationRunLog(self.root.parent / "validation_runs", "full_loop_tunnel_vehicle")
        log.enable_incidents(self.root, self.scenario)
        for tick in (0.0, 10.0, 20.0, 29.0):
            log.observe(RUN_ID, tick, {"x": tick, "y": 2.0, "z": 3.0,
                "speed_mps": 0.1, "collision_objects": []}, clock_domain="host_monotonic")
            log.note_route_progress(tick)
        finish_validation_run(log, self.scenario, "exception", RuntimeError(reason))
        self.assertEqual(len(log.incident_recorder.bundles), 1)
        bundle = log.incident_recorder.bundles[0]
        manifest = load_incident(bundle)
        self.assertEqual(manifest["failure"]["event_type"], "scenario_timeout")
        self.assertEqual(manifest["failure"]["actor_ids"], [])
        self.assertTrue(bundle.name.endswith("_timeout"))
        self.assertEqual((bundle / "validation_events.jsonl").read_text(), "")
        self.assertEqual(len((bundle / "ego_trace.jsonl").read_text().splitlines()), 4)
        self.assertEqual(json.loads((bundle / "summary.json").read_text())["error_message"], reason)
        checks = {row["check_id"]: row for row in json.loads((bundle / "diagnosis.json").read_text())["checks"]}
        self.assertEqual(checks["timeout.event"]["status"], "OBSERVED")
        self.assertEqual(checks["collision.event"]["status"], "NOT_APPLICABLE")
        finish_validation_run(log, self.scenario, "exception", RuntimeError(reason))
        self.assertEqual(len(log.incident_recorder.bundles), 1)

    def test_non_timeout_exception_does_not_create_incident(self):
        recorder = IncidentRecorder(self.root, self.scenario)
        recorder.observe(0, observation(1.0), [], ego_state={}, captured_at=1.0)
        self.assertIsNone(recorder.capture_timeout(RUN_ID, RuntimeError("other failure"), captured_at=2.0))
        self.assertEqual(recorder.bundles, [])

    def test_timeout_capture_failure_does_not_replace_original_failure(self):
        log = ValidationRunLog(self.root.parent / "validation_runs", "full_loop_tunnel_vehicle")
        log.enable_incidents(self.root, self.scenario)
        log.observe(RUN_ID, 1.0, {"x": 1.0, "y": 2.0, "z": 3.0,
            "speed_mps": 0.0, "collision_objects": []})
        with patch.object(log.incident_recorder, "capture_timeout", side_effect=OSError("disk full")):
            finish_validation_run(log, self.scenario, "exception",
                                              RuntimeError("Tunnel vehicle timed out"))
        summary = json.loads((log.output_dir / "summary.json").read_text())
        self.assertEqual(summary["error_message"], "Tunnel vehicle timed out")

    def test_timeout_keeps_only_preceding_30_seconds_and_pipeline(self):
        reason = "Tunnel vehicle timed out"
        self.scenario.tunnel_vehicle_result["failure_reason"] = reason
        recorder = IncidentRecorder(self.root, self.scenario)
        pipeline = self.root.parent / "pipeline.jsonl"
        recorder.connect_pipeline(pipeline)
        rows = []
        for index, tick in enumerate((100.0, 120.0, 130.0)):
            for stage in ("planning", "control"):
                rows.append({"schema": "aimdrive_pipeline_v1", "run_id": RUN_ID,
                    "clock_domain": "host_monotonic_ns", "timestamp_ns": int(tick * 1e9),
                    "sequence": len(rows), "stage": stage, "cycle_id": index})
        pipeline.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        provider = lambda: {"actors": [{"actor_id": "npc-7", "position_xyz": [4, 5, 0]}],
                            "errors": []}
        for sequence, tick in enumerate((95.0, 101.0, 120.0, 130.0)):
            recorder.observe(sequence, observation(tick), [], ego_state={},
                             actor_provider=provider, captured_at=tick)
            recorder.note_route_progress(tick - 80.0)
        recorder.capture_timeout(RUN_ID, RuntimeError(reason), captured_at=131.0)
        bundle = recorder.bundles[0]
        ego_rows = [json.loads(line) for line in (bundle / "ego_trace.jsonl").read_text().splitlines()]
        self.assertEqual([row["timestamp"] for row in ego_rows], [101.0, 120.0, 130.0])
        self.assertEqual(len((bundle / "actors_trace.jsonl").read_text().splitlines()), 3)
        self.assertEqual(len((bundle / "planning_trace.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len((bundle / "control_trace.jsonl").read_text().splitlines()), 2)

    def test_rerun_preserves_seed_and_previous_profile_without_overwriting_original(self):
        bundle = self.capture()
        manifest = load_incident(bundle)
        cfg = prepare_config(bundle, manifest)
        self.assertEqual(cfg["random_seed"], 42)
        self.assertNotEqual(cfg["profile_state_path"], "/old/profile.json")
        prior = json.loads(Path(cfg["profile_state_path"]).read_text(encoding="utf-8"))
        self.assertEqual(prior, self.scenario.previous_profile_state)
        rerun = fake_scenario()
        rerun.run_instance_id = "new_run"
        rerun.cfg = cfg
        result_path = compare_rerun(bundle, manifest, rerun, "run_returned")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertTrue(result["matches"]["seed"])
        self.assertTrue(result["matches"]["scenario_config"])
        self.assertTrue(result["matches"]["weather"])
        self.assertTrue(result["matches"]["time_hour"])
        self.assertTrue(result["matches"]["npc_placement"]["spawn_target_match"])
        self.assertEqual(result["reproduction_level"], "setup_and_npc_target_matched")
        self.assertFalse(result["failure_instant_restored"])

    def test_mission_rerun_seeds_the_profile_file_that_setup_actually_reads(self):
        self.scenario.cfg["mission_mode"] = "full_loop_tunnel_vehicle"
        self.scenario.cfg["profile_state_path"] = "/old/full_loop_tunnel_vehicle_profile.json"
        bundle = self.capture()
        manifest = load_incident(bundle)
        cfg = prepare_config(bundle, manifest)
        target = Path(cfg["profile_state_path"])
        self.assertEqual(target.name, "full_loop_tunnel_vehicle_profile.json")
        self.assertEqual(json.loads(target.read_text()), manifest["previous_profile_state"])
        # prepare_mission() replaces the basename with this same name.
        self.assertEqual(target.with_name("full_loop_tunnel_vehicle_profile.json"), target)
        rerun = fake_scenario()
        rerun.cfg = cfg
        result = json.loads(compare_rerun(bundle, manifest, rerun, "run_returned").read_text())
        self.assertTrue(result["matches"]["scenario_config"])
        self.assertTrue(result["matches"]["profile_state_path"])
        self.assertTrue(result["matches"]["previous_profile_state"])
        self.assertEqual(len(result["configuration_differences"]), 1)
        difference = result["configuration_differences"][0]
        self.assertEqual(difference["key"], "profile_state_path")
        self.assertEqual(difference["original"], "/old/full_loop_tunnel_vehicle_profile.json")
        self.assertEqual(difference["rerun"], str(target))
        self.assertFalse(difference["affects_reproduction"])

    def test_profile_storage_path_is_not_excused_when_prior_state_was_not_loaded(self):
        self.scenario.cfg["mission_mode"] = "full_loop_tunnel_vehicle"
        bundle = self.capture()
        manifest = load_incident(bundle)
        rerun = fake_scenario()
        rerun.cfg = prepare_config(bundle, manifest)
        rerun.previous_profile_state = {}
        result = json.loads(compare_rerun(bundle, manifest, rerun, "run_returned").read_text())
        self.assertFalse(result["matches"]["scenario_config"])
        self.assertFalse(result["matches"]["previous_profile_state"])
        self.assertTrue(result["configuration_differences"][0]["affects_reproduction"])

    def test_real_scenario_setting_difference_remains_a_mismatch(self):
        bundle = self.capture()
        manifest = load_incident(bundle)
        rerun = fake_scenario()
        rerun.cfg = prepare_config(bundle, manifest)
        rerun.cfg["drive_control_mode"] = "morai_keyboard"
        result = json.loads(compare_rerun(bundle, manifest, rerun, "run_returned").read_text())
        self.assertFalse(result["matches"]["scenario_config"])
        difference = next(row for row in result["configuration_differences"]
                          if row["key"] == "drive_control_mode")
        self.assertEqual(difference["original"], "aimdrive_expert")
        self.assertEqual(difference["rerun"], "morai_keyboard")
        self.assertEqual(difference["category"], "scenario_setting")
        self.assertTrue(difference["affects_reproduction"])

    def test_actual_spawn_compares_pose_instead_of_per_run_sample_time(self):
        bundle = self.capture()
        manifest = load_incident(bundle)
        rerun = fake_scenario()
        rerun.cfg = prepare_config(bundle, manifest)
        rerun.tunnel_vehicle_result["actual_spawn"]["time"] = 99999.0
        rerun.tunnel_vehicle_result["actual_spawn"]["speed_mps"] = 0.5
        result = json.loads(compare_rerun(bundle, manifest, rerun, "run_returned").read_text())
        self.assertTrue(result["matches"]["npc_placement"]["actual_spawn_match"])
        self.assertEqual(result["npc_placement_delta_xy_m"]["actual_spawn"], 0.0)
        rerun.tunnel_vehicle_result["actual_spawn"]["x"] += 1.5
        result = json.loads(compare_rerun(bundle, manifest, rerun, "run_returned").read_text())
        self.assertFalse(result["matches"]["npc_placement"]["actual_spawn_match"])
        self.assertEqual(result["npc_placement_delta_xy_m"]["actual_spawn"], 1.5)

    def test_mismatched_weather_and_unsupported_controller_are_explicit(self):
        bundle = self.capture()
        manifest = load_incident(bundle)
        rerun = fake_scenario()
        rerun.profile = {"name": "scenario_foggy", "weather": "FOGGY"}
        result = json.loads(compare_rerun(bundle, manifest, rerun, "run_returned").read_text())
        self.assertFalse(result["matches"]["weather"])
        self.assertEqual(result["reproduction_level"], "setup_mismatch_or_unverified")
        manifest["controller"] = "ve_planning"
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "controller"):
            load_incident(bundle)

    def test_actor_snapshot_uses_existing_state_api_and_radius(self):
        def state(actor_id, x):
            return SimpleNamespace(
                actor_info=SimpleNamespace(id=SimpleNamespace(value=actor_id)),
                transform=SimpleNamespace(location=SimpleNamespace(x=x, y=0.0, z=0.0),
                                          rotation=SimpleNamespace(z=90.0)),
                velocity=SimpleNamespace(x=3.6, y=0.0, z=0.0),
                vehicle_state=SimpleNamespace(current_link_info=SimpleNamespace(
                    id=SimpleNamespace(value="link-1"))),
            )
        grpc = SimpleNamespace(get_all_vehicle_states=lambda: [state("near", 10.0), state("far", 200.0)],
                               get_all_pedestrian_states=lambda: [])
        result = snapshot_nearby_actors(grpc, {"x": 0.0, "y": 0.0, "collision_objects": []})
        self.assertEqual([row["actor_id"] for row in result["actors"]], ["near"])
        self.assertEqual(result["actors"][0]["speed_mps"], 1.0)
        self.assertEqual(result["errors"], [])

    def test_incident_survives_online_event_writer_failure(self):
        log = ValidationRunLog(self.root.parent / "validation_runs", "full_loop_tunnel_vehicle")
        log.enable_incidents(self.root, self.scenario)
        log._bind(RUN_ID)
        log.writer.write = Mock(side_effect=OSError("event file unavailable"))
        with self.assertRaisesRegex(OSError, "event file unavailable"):
            log.observe(RUN_ID, 1.0, {"x": 1.0, "y": 2.0, "z": 3.0,
                                       "collision_objects": ["npc-7"]})
        bundles = list(self.root.glob("incident_*"))
        self.assertEqual(len(bundles), 1)
        self.assertEqual(len((bundles[0] / "validation_events.jsonl").read_text().splitlines()), 1)

        other_root = self.root.parent / "observation_failure_incidents"
        other = ValidationRunLog(self.root.parent / "other_validation_runs", "full_loop_tunnel_vehicle")
        other.enable_incidents(other_root, self.scenario)
        other._bind(RUN_ID)
        other.observation_writer.write = Mock(side_effect=OSError("observation file unavailable"))
        with self.assertRaisesRegex(OSError, "observation file unavailable"):
            other.observe(RUN_ID, 1.0, {"x": 1.0, "y": 2.0, "z": 3.0,
                                         "collision_objects": ["npc-7"]})
        self.assertEqual(len(list(other_root.glob("incident_*"))), 1)


if __name__ == "__main__":
    unittest.main()
