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

PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / "data_collection/internal_sensor_record"
sys.path[:0] = [str(PROJECT), str(INTERNAL), str(INTERNAL / "tools")]
from scenario_collection import (TUNNEL_SCENARIOS, configure_collection_scenario,
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

    def test_tunnel_dataset_and_export_paths_agree(self):
        for name, folder in list(zip(TUNNEL_SCENARIOS, ("tunnel_dynamic", "tunnel_vehicle"))) + [
                (TUNNEL_STATIC_SCENARIO, "tunnel_static_obstacle")]:
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
                (TUNNEL_STATIC_SCENARIO, "tunnel_static_obstacle")]:
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
