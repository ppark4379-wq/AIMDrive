"""Tunnel capture wiring using real mission routes and a mocked MORAI boundary."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / "data_collection/internal_sensor_record"
sys.path[:0] = [str(PROJECT), str(INTERNAL), str(INTERNAL / "tools")]
from scenario_collection import (TUNNEL_SCENARIOS, TUNNEL_NORMAL_SCENARIO, configure_collection_scenario,
    tunnel_collection_evidence, collection_tunnel_phase)
from dataset_layout import collecting_campaign, final_campaign
from nas_publish import scenario_folder as nas_scenario_folder
from audit_scenario_interventions import scenario_zone_contract, zone_observation_checks
from async_capture.scenario_catalog import COLLECTION_SCENARIOS, TUNNEL_STATIC_SCENARIO
from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.segment_modes import events
from scenario_runner.scenarios.full_loop_parts import tunnel_dynamic, tunnel_vehicle_dynamic
from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.tests.test_segment_modes import make_grpc


class TunnelCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, _, cls.cfg = load_configuration()
        cls.map_loader = MGeoMapLoader(cls.global_cfg["paths"]["mgeo_root"])
        spec = importlib.util.spec_from_file_location("tunnel_collection_runner",
            INTERNAL / "tools/run_proven_three_lap_scenario.py")
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def scenario(self, name, seed):
        cfg = copy.deepcopy(self.cfg)
        configure_collection_scenario(cfg, name)
        temporary = tempfile.TemporaryDirectory(dir=PROJECT / "scenario_runner/.runtime")
        self.addCleanup(temporary.cleanup)
        gc = copy.deepcopy(self.global_cfg)
        gc["paths"]["runtime_root"] = temporary.name
        cfg["route_debug_enabled"] = False
        cfg["profile_state_path"] = str(Path(temporary.name) / "profile.json")
        grpc = make_grpc(cfg)
        scenario = self.runner.collection_scenario_class(name)(
            grpc, self.map_loader, gc, cfg,
            output=Path(temporary.name) / "episode", contract_path=Path(temporary.name) / "contract.json",
            seconds=3000, full_lap=True)
        scenario.clean_road_mode = False
        scenario.run_seed = seed
        scenario.run_instance_id = "collection_%d" % seed
        scenario.random = random.Random(seed)
        with redirect_stdout(io.StringIO()):
            events.prepare_mission(scenario)
            scenario.route_definition = scenario.load_route_definition()
            scenario.prepare_full_loop_route()
            tunnel_dynamic.prepare_tunnel_dynamic(scenario)
            tunnel_vehicle_dynamic.prepare_tunnel_vehicle(scenario)
        return scenario

    def test_real_capture_mro_and_seeded_tunnel_plans(self):
        for name in TUNNEL_SCENARIOS:
            with self.subTest(name=name):
                first, repeat = self.scenario(name, 123), self.scenario(name, 123)
                self.assertEqual(first.start_link, "A2256W000448")
                self.assertAlmostEqual(first.route_length_m, 137.0892132912, places=5)
                self.assertTrue(first.full_lap)
                first.grpc.spawn_vehicle.assert_not_called()
                first.grpc.spawn_pedestrian.assert_not_called()
                self.assertFalse(first.cfg["sync_runtime"]["enabled"])
                if name.endswith("vehicle"):
                    self.assertEqual(first.tunnel_vehicle_plan, repeat.tunnel_vehicle_plan)
                    self.assertIn(first.tunnel_vehicle_plan["behavior"],
                        ("vehicle_slow", "vehicle_sudden_brake", "vehicle_oncoming"))
                else:
                    self.assertEqual(first.tunnel_dynamic_plans, repeat.tunnel_dynamic_plans)
                    self.assertEqual(len(first.tunnel_dynamic_plans), 4)
                    self.assertEqual(len({p["behavior"] for p in first.tunnel_dynamic_plans}), 4)
                    self.assertEqual([p["section_index"] for p in first.tunnel_dynamic_plans], [1,2,3,4])
                # Verify collection dispatch reaches the main MissionScenario setup.
                self.assertIn(self.runner.MissionScenario, type(first).__mro__)
                self.assertIn(self.runner.RawScenario, type(first).__mro__)

    def test_evidence_snapshot_and_active_phase_do_not_use_legacy_disabled_event(self):
        for name in TUNNEL_SCENARIOS:
            context = SimpleNamespace(cfg={"mission_mode": name}, run_seed=12,
                tunnel_event_phase="disabled", tunnel_vehicle_phase="waiting_for_ego",
                tunnel_vehicle_plan={"model": "Sedan", "behavior": "vehicle_slow"},
                tunnel_vehicle_result={"startup_motion_verified": False},
                tunnel_dynamic_phase="waiting", tunnel_dynamic_plans=[{"model": "Man1"}],
                tunnel_dynamic_results=[{"completed": False}])
            evidence = tunnel_collection_evidence(context)
            self.assertFalse(evidence["completed"])
            self.assertNotEqual(collection_tunnel_phase(context), "disabled")
            if name.endswith("vehicle"):
                context.tunnel_vehicle_phase = "completed"
                context.tunnel_vehicle_result["startup_motion_verified"] = True
            else:
                context.tunnel_dynamic_phase = "completed"
                context.tunnel_dynamic_results[0]["completed"] = True
            complete = tunnel_collection_evidence(context)
            self.assertTrue(complete["completed"])
            self.assertNotEqual(evidence["results"], complete["results"])
            json.dumps(complete)

    def test_existing_scenario_config_is_unchanged(self):
        for name in ("full_loop", "full_loop_dynamic", "full_loop_static", TUNNEL_STATIC_SCENARIO):
            cfg = copy.deepcopy(self.cfg)
            original = copy.deepcopy(cfg)
            configure_collection_scenario(cfg, name)
            self.assertEqual(cfg, original)

    def test_guarded_static_collection_stays_separate_from_dynamic_mission_policy(self):
        self.assertIn(TUNNEL_STATIC_SCENARIO, COLLECTION_SCENARIOS)
        self.assertNotIn(TUNNEL_STATIC_SCENARIO, TUNNEL_SCENARIOS)
        context = SimpleNamespace(cfg={"mission_mode": TUNNEL_STATIC_SCENARIO},
                                  run_seed=12, tunnel_event_phase="disabled")
        self.assertIsNone(tunnel_collection_evidence(context))
        self.assertEqual(collection_tunnel_phase(context), "disabled")
        self.assertEqual(nas_scenario_folder({"scenario_name": TUNNEL_STATIC_SCENARIO}),
                         "tunnel_static_obstacle")
        self.assertTrue(scenario_zone_contract(TUNNEL_STATIC_SCENARIO)["tunnel_static"])
        self.assertFalse(scenario_zone_contract(TUNNEL_STATIC_SCENARIO)["tunnel"])

    def test_empty_tunnel_uses_baseline_route_and_disabled_event_contract(self):
        scenario = self.scenario(TUNNEL_NORMAL_SCENARIO, 123)
        self.assertEqual(scenario.start_link, "A2256W000446")
        self.assertAlmostEqual(scenario.route_length_m, 189.7, delta=0.1)
        self.assertEqual(scenario.tunnel_baseline_report["status"], "prepared")
        self.assertIn("_tunnel_normal_collection", scenario.cfg)
        self.assertFalse(any(scenario_zone_contract(TUNNEL_NORMAL_SCENARIO).values()))
        self.assertIn(TUNNEL_NORMAL_SCENARIO, COLLECTION_SCENARIOS)
        scenario.grpc.spawn_vehicle.assert_not_called()
        scenario.grpc.spawn_pedestrian.assert_not_called()

    def test_collection_vehicle_pool_is_reviewed_and_runtime_available(self):
        scenario = self.scenario("full_loop_tunnel_vehicle", 321)
        policy = scenario.cfg["tunnel_vehicle_dynamic"]["collection_geometry_selection"]
        self.assertEqual(len(policy["approved_models"]), 36)
        self.assertFalse(policy["physical_simultaneity_verified"])
        with redirect_stdout(io.StringIO()):
            scenario.tunnel_vehicle_handler.arm(scenario)
        pool = scenario.tunnel_vehicle_result["model_pool"]
        self.assertTrue(pool["candidates"])
        self.assertTrue(set(pool["candidates"]) <= set(policy["approved_models"]))
        self.assertTrue(set(pool["candidates"]) <= set(scenario.vehicle_models()))
        self.assertEqual(set(policy["approved_models"]) - set(scenario.vehicle_models()),
                         set(pool["unavailable_reviewed_models"]))

    def test_collection_rejects_unreviewed_fixed_or_empty_vehicle_pool(self):
        for vehicle in ({"models": ["Unreviewed"]}, {"models": []},
                        {"model": "Unreviewed"}):
            cfg = copy.deepcopy(self.cfg)
            cfg["tunnel_vehicle_dynamic"].update(vehicle)
            with self.subTest(vehicle=vehicle), self.assertRaises(ValueError):
                configure_collection_scenario(cfg, "full_loop_tunnel_vehicle")

    def test_collection_requires_an_available_approved_vehicle_before_spawn(self):
        scenario = self.scenario("full_loop_tunnel_vehicle", 322)
        with patch.object(scenario, "vehicle_models", return_value=["Unreviewed"]):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                scenario.tunnel_vehicle_handler.arm(scenario)
        scenario.grpc.spawn_vehicle.assert_not_called()
        self.assertEqual(scenario.tunnel_vehicle_phase, "failed")

    def test_normal_capture_run_verifies_parking_after_raw_cleanup(self):
        scenario = self.scenario(TUNNEL_NORMAL_SCENARIO, 124)
        result = dict(status="mission_completed", full_mission_completed=True)
        calls = []
        def raw_run(context):
            calls.append("raw_cleanup_complete")
            return result
        with patch.object(self.runner.RawScenario, "run", raw_run), patch(
                "scenario_runner.scenarios.segment_modes.tunnel_baseline.park",
                side_effect=lambda context: calls.append("parking_verified")):
            self.assertIs(scenario.run(), result)
        self.assertEqual(calls, ["raw_cleanup_complete", "parking_verified"])

    def test_normal_parking_failure_cannot_keep_a_completed_capture(self):
        scenario = self.scenario(TUNNEL_NORMAL_SCENARIO, 125)
        result = dict(status="mission_completed", full_mission_completed=True)
        with patch.object(self.runner.RawScenario, "run", return_value=result), patch(
                "scenario_runner.scenarios.segment_modes.tunnel_baseline.park",
                side_effect=RuntimeError("Ego gear verification failed")), patch(
                "scenario_runner.scenarios.segment_modes.tunnel_baseline.finish") as finish:
            self.assertEqual(scenario.run()["status"], "failed")
        finish.assert_called_once_with(scenario, "failed", "Ego gear verification failed")

    def test_other_collection_modes_do_not_acquire_baseline_parking(self):
        scenario = self.scenario("full_loop_tunnel_vehicle", 126)
        result = dict(status="mission_completed")
        with patch.object(self.runner.RawScenario, "run", return_value=result), patch(
                "scenario_runner.scenarios.segment_modes.tunnel_baseline.park") as park:
            self.assertIs(scenario.run(), result)
        park.assert_not_called()

    def test_unknown_collection_mode_is_not_silently_selected(self):
        with self.assertRaises(ValueError):
            self.runner.collection_scenario_class("full_loop_tunnel_typo")
        with self.assertRaises(ValueError):
            scenario_zone_contract("full_loop_tunnel_typo")

    def test_tunnel_dataset_and_export_paths_agree(self):
        for name, folder in list(zip(TUNNEL_SCENARIOS, ("tunnel_dynamic", "tunnel_vehicle"))) + [
                (TUNNEL_STATIC_SCENARIO, "tunnel_static_obstacle"),
                (TUNNEL_NORMAL_SCENARIO, "tunnel_normal")]:
            self.assertEqual(nas_scenario_folder({"scenario_name": name}), folder)
            self.assertEqual(collecting_campaign(PROJECT, name, "campaign_01").parts[-3:],
                (".collecting", folder, "campaign_01"))
            self.assertEqual(final_campaign(PROJECT, name, "campaign_01").parts[-2:],
                (folder, "campaign_01"))

    def test_waiting_or_failed_tunnel_is_not_admitted_as_completed(self):
        for name in TUNNEL_SCENARIOS:
            for phases, passed in ((["disabled"], False), (["waiting_for_ego"], False),
                                   (["waiting", "failed"], False), (["moving", "completed"], True)):
                checks = zone_observation_checks(name, scenario_zone_contract(name),
                    selected_static="OFF", selected_dynamic="OFF", static_models=[], dynamic_models=[],
                    dynamic_phases=["disabled"], roundabout_observation={
                        "spawn_events": 0, "spawn_lifecycle_successes": 0, "retire_lifecycle_successes": 0},
                    highway_samples=0, tunnel_phases=phases)
                self.assertEqual(checks["tunnel_observed"], passed)

    def test_shell_scenario_options_and_folder_selection(self):
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash is required for collection script argument tests")
        source = (INTERNAL / "tools/collect_scenario_laps.sh").read_text(encoding="utf-8")
        parser = source[source.index("usage() {"):source.index("export MORAI_ANALYSIS_PYTHON=")]
        for name, folder in list(zip(TUNNEL_SCENARIOS, ("tunnel_dynamic", "tunnel_vehicle"))) + [
                (TUNNEL_STATIC_SCENARIO, "tunnel_static_obstacle"),
                (TUNNEL_NORMAL_SCENARIO, "tunnel_normal")]:
            process = subprocess.run([bash, "-s", "--", "3", "campaign_01", "--scenario", name],
                input=parser + '\nprintf "%s %s %s %s\\n" "$SCENARIO" "$SCENARIO_FOLDER" "$LAPS" "$CAMPAIGN_ID"\n',
                env=dict(os.environ, MORAI_ANALYSIS_PYTHON=sys.executable),
                text=True, capture_output=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertIn(name + " " + folder + " 3 campaign_01", process.stdout)
        syntax = subprocess.run([bash, "-n"], input=source, text=True, capture_output=True, timeout=15)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)


if __name__ == "__main__":
    unittest.main()
