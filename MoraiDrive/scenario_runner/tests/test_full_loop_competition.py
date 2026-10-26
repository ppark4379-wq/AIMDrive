"""Competition sequencing survives process boundaries without changing Expert."""

import copy
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..main_runner import load_configuration
from ..scenarios import full_loop_competition as competition_module
from ..scenarios.full_loop_scenario import FullLoopScenario
from ..scenarios.full_loop_competition import FullLoopCompetitionScenario
from ..scenarios.full_loop_parts import roundabout
from ..scenarios.full_loop_parts import tunnel_dynamic, tunnel_vehicle_dynamic
from . import test_tunnel_interaction as pedestrian_fixture
from . import test_tunnel_vehicle_dynamic as vehicle_fixture
from . import test_tunnel_static_live as static_fixture


class FullLoopCompetitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, _, cls.base_cfg = load_configuration()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = copy.deepcopy(self.base_cfg)
        self.cfg.update(profile_state_path=str(self.root / "full_loop_profile.json"),
                        forced_weather="SUNNY", forced_time_hour=11,
                        random_seed=123, route_debug_enabled=False)
        self.global_config = copy.deepcopy(self.global_cfg)
        self.global_config["paths"]["runtime_root"] = str(self.root / "runtime")
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def scenario(self, scenario_type=FullLoopCompetitionScenario):
        return scenario_type(Mock(), None, copy.deepcopy(self.global_config), self.cfg)

    @staticmethod
    def ready_setup(scenario):
        """Keep the real profile writer while replacing MORAI-only setup work."""
        scenario.profile_index = 0
        scenario.profile = {"name": "sunny-test", "weather": "SUNNY"}
        scenario.scenario_hour = 11
        scenario.previous_profile_state = scenario.read_profile_state()
        scenario.static_obstacle_model = None
        scenario.static_obstacle_scale = None
        scenario.static_obstacle_models_selected = []
        scenario.static_obstacle_scales = []
        scenario.static_obstacles = []
        scenario.dynamic_obstacle_model = None
        scenario.dynamic_obstacle_events = []
        scenario.dynamic_obstacle_reports = lambda: []
        scenario.run_seed = 123
        scenario.persist_profile_state()

    def setup_ready(self, scenario):
        with patch.object(FullLoopScenario, "setup", self.ready_setup):
            scenario.setup()

    def cleanup_without_rpc(self, scenario):
        with patch.object(FullLoopScenario, "cleanup"), patch.object(
                competition_module.competition_static, "cleanup"):
            scenario.cleanup()

    def test_new_instances_continue_the_four_scene_cycle_from_persisted_state(self):
        for sequence, expected in enumerate(
                ("legacy", "static", "pedestrian", "vehicle", "legacy"), 1):
            # Recreate all scenario state as a fresh --once process would.
            scenario = self.scenario()
            self.setup_ready(scenario)
            saved = json.loads(Path(scenario.profile_state_path()).read_text())
            self.assertEqual(scenario.competition_sequence, sequence)
            self.assertEqual(scenario.competition_tunnel_scene, expected)
            self.assertEqual(scenario.cfg["competition_tunnel_scene"], expected)
            self.assertEqual(saved["competition"], {
                "schema": 1, "sequence": sequence, "tunnel_scene": expected})
            self.assertEqual(scenario.cfg["tunnel_sudden_brake"]["enabled"], expected == "legacy")
            self.assertEqual(scenario.cfg["tunnel_dynamic"]["enabled"], expected == "pedestrian")
            self.assertEqual(scenario.cfg["tunnel_vehicle_dynamic"]["enabled"], expected == "vehicle")
            if expected == "pedestrian":
                self.assertTrue(scenario.cfg["tunnel_dynamic"]["random_mix"]["enabled"])
            if expected == "vehicle":
                self.assertTrue(scenario.cfg["tunnel_vehicle_dynamic"]["random_behavior"])
            self.cleanup_without_rpc(scenario)

    def test_competition_history_and_configuration_are_isolated_from_full_loop(self):
        ordinary = Path(self.cfg["profile_state_path"])
        incoming = b'{"ordinary_full_loop_history":"preserve exactly"}\n'
        ordinary.write_bytes(incoming)
        original_cfg = copy.deepcopy(self.cfg)
        scenario = self.scenario()
        self.assertEqual(Path(scenario.profile_state_path()),
                         ordinary.with_name("full_loop_competition_profile.json"))
        self.setup_ready(scenario)
        self.cleanup_without_rpc(scenario)
        self.assertEqual(self.cfg, original_cfg)
        self.assertEqual(ordinary.read_bytes(), incoming)
        self.assertNotIn("competition_tunnel_scene", self.cfg)
        ordinary_scenario = self.scenario(FullLoopScenario)
        self.assertEqual(Path(ordinary_scenario.profile_state_path()), ordinary)
        self.assertEqual(ordinary_scenario.profile_state_metadata(), {})

    def test_setup_failure_does_not_consume_first_scene_or_leave_lock_held(self):
        failed = self.scenario()
        with patch.object(FullLoopScenario, "setup", side_effect=RuntimeError("setup RPC failed")):
            with self.assertRaisesRegex(RuntimeError, "setup RPC failed"):
                failed.setup()
        self.assertFalse(Path(failed.profile_state_path()).exists())
        self.cleanup_without_rpc(failed)
        retry = self.scenario()
        self.setup_ready(retry)
        self.assertEqual(retry.competition_sequence, 1)
        self.assertEqual(retry.competition_tunnel_scene, "legacy")
        self.cleanup_without_rpc(retry)

    def test_setup_failure_preserves_previous_sequence_and_incoming_bytes(self):
        path = self.root / "full_loop_competition_profile.json"
        incoming = (b'{ "weather":"SUNNY","time_hour":11, '
                    b'"competition":{"schema":1,"sequence":3,"tunnel_scene":"pedestrian"} }\n')
        path.write_bytes(incoming)
        failed = self.scenario()
        with patch.object(FullLoopScenario, "setup", side_effect=RuntimeError("drive export failed")):
            with self.assertRaisesRegex(RuntimeError, "drive export failed"):
                failed.setup()
        self.assertEqual(path.read_bytes(), incoming)
        self.cleanup_without_rpc(failed)
        retry = self.scenario()
        self.setup_ready(retry)
        self.assertEqual(retry.competition_sequence, 4)
        self.assertEqual(retry.competition_tunnel_scene, "vehicle")
        self.cleanup_without_rpc(retry)

    def test_corrupt_history_fails_before_base_setup_instead_of_resetting_cycle(self):
        path = self.root / "full_loop_competition_profile.json"
        incoming = b'{ invalid competition history\n'
        path.write_bytes(incoming)
        scenario = self.scenario()
        with patch.object(FullLoopScenario, "setup") as parent:
            with self.assertRaises((RuntimeError, ValueError)):
                scenario.setup()
        parent.assert_not_called()
        self.assertEqual(path.read_bytes(), incoming)
        self.cleanup_without_rpc(scenario)

    def test_concurrent_process_cannot_consume_the_same_or_next_rotation_slot(self):
        owner = self.scenario()
        self.setup_ready(owner)
        before = Path(owner.profile_state_path()).read_bytes()
        contender = self.scenario()
        try:
            # Reject the second command before inherited run can start its
            # runtime bridge or perform any simulator setup.
            with patch.object(FullLoopScenario, "run") as parent:
                with self.assertRaisesRegex(RuntimeError, "owns the rotation state"):
                    contender.run()
            parent.assert_not_called()
            self.assertEqual(Path(owner.profile_state_path()).read_bytes(), before)
        finally:
            self.cleanup_without_rpc(contender)
            self.cleanup_without_rpc(owner)
        next_process = self.scenario()
        self.setup_ready(next_process)
        self.assertEqual(next_process.competition_sequence, 2)
        self.cleanup_without_rpc(next_process)

    def test_eight_fresh_runs_select_different_weather_and_hour_using_real_history(self):
        self.cfg.pop("forced_weather", None)
        self.cfg.pop("forced_time_hour", None)
        ordinary = Path(self.cfg["profile_state_path"])
        ordinary_bytes = b'{"profile_index":0,"weather":"SUNNY","time_hour":13,"run_seed":99}\n'
        ordinary.write_bytes(ordinary_bytes)
        selections = []

        def environment_setup(scenario):
            scenario.run_seed = 1000 + scenario.competition_sequence
            scenario.random = random.Random(scenario.run_seed)
            (scenario.profile_index, scenario.profile,
             scenario.previous_profile_state) = scenario.select_profile()
            scenario.scenario_hour = scenario.select_scenario_hour()
            for name, value in (
                    ("static_obstacle_model", None), ("static_obstacle_scale", None),
                    ("static_obstacle_models_selected", []), ("static_obstacle_scales", []),
                    ("static_obstacles", []), ("dynamic_obstacle_model", None),
                    ("dynamic_obstacle_events", [])):
                setattr(scenario, name, value)
            scenario.dynamic_obstacle_reports = lambda: []
            scenario.persist_profile_state()

        for sequence in range(1, 9):
            scenario = self.scenario()
            with patch.object(FullLoopScenario, "setup", environment_setup):
                scenario.setup()
            saved = json.loads(Path(scenario.profile_state_path()).read_text())
            current = saved["weather"], saved["time_hour"]
            if selections:
                self.assertNotEqual(current[0], selections[-1][0])
                self.assertNotEqual(current[1], selections[-1][1])
            self.assertIn(current[0], ("SUNNY", "FOGGY"))
            self.assertIn(current[1], self.cfg["scenario_times"])
            self.assertEqual(saved["competition"]["sequence"], sequence)
            self.assertEqual(saved["competition"]["tunnel_scene"],
                             ("legacy", "static", "pedestrian", "vehicle")[(sequence - 1) % 4])
            selections.append(current)
            self.cleanup_without_rpc(scenario)
        self.assertEqual(ordinary.read_bytes(), ordinary_bytes)

    def test_run_report_records_scene_environment_and_failure_without_advancing_again(self):
        scenario = self.scenario()
        scenario.run_instance_id = "unit-competition-run"
        self.setup_ready(scenario)
        report_path = self.root / "competition_runs/unit-competition-run.json"
        running = json.loads(report_path.read_text())
        self.assertEqual(running["status"], "running")
        self.assertEqual((running["sequence"], running["run_seed"], running["tunnel_scene"]),
                         (1, 123, "legacy"))
        self.assertEqual((running["weather"], running["time_hour"]), ("SUNNY", 11))
        committed = Path(scenario.profile_state_path()).read_bytes()
        with patch.object(FullLoopScenario, "run_timeline", side_effect=RuntimeError("observed collision")):
            with self.assertRaisesRegex(RuntimeError, "observed collision"):
                scenario.run_timeline()
        self.cleanup_without_rpc(scenario)
        failed = json.loads(report_path.read_text())
        self.assertEqual(failed["status"], "interrupted")
        self.assertEqual(failed["error"], "observed collision")
        self.assertFalse(failed["route_finished"])
        self.assertEqual(Path(scenario.profile_state_path()).read_bytes(), committed)

    def test_expert_runtime_planning_matches_ordinary_full_loop_and_original_source(self):
        source = Path(self.global_cfg["paths"]["expert_root"]) / "src/config/config.json"
        original = source.read_bytes()
        exports = []
        for label, scenario_type in (("ordinary", FullLoopScenario),
                                     ("competition", FullLoopCompetitionScenario)):
            scenario = self.scenario(scenario_type)
            scenario.global_cfg["paths"]["runtime_root"] = str(self.root / label)
            scenario.route_points = [(0., 0., 0.), (10., 0., 0.), (20., 0., 0.)]
            scenario.route_definition = {}
            scenario.start_link, scenario.end_link = "start", "end"
            scenario.setup_expert_controller()
            controller = scenario.expert_controller
            exports.append(json.loads(Path(controller.runtime_config_path).read_text()))
            self.assertTrue(controller.speed_contract["inherited"])
            self.assertIsNone(controller.speed_cap_kmh)
            self.assertIsNone(controller.process)
        self.assertEqual(exports[0]["planning"], exports[1]["planning"])
        self.assertEqual(exports[1]["planning"], json.loads(original)["planning"])
        self.assertEqual(source.read_bytes(), original)
        self.assertIs(FullLoopCompetitionScenario.setup_expert_controller,
                      FullLoopScenario.setup_expert_controller)

    def test_ordinary_zone_creation_and_lifecycle_use_the_existing_shared_hooks(self):
        for name in ("spawn_static_obstacle", "spawn_dynamic_obstacle",
                     "update_dynamic_obstacle", "spawn_roundabout_npcs",
                     "update_roundabout_npcs", "spawn_highway_npcs",
                     "update_highway_actor_lifecycle"):
            with self.subTest(zone_hook=name):
                self.assertIs(getattr(FullLoopCompetitionScenario, name),
                              getattr(FullLoopScenario, name))

    def test_competition_still_aborts_on_real_morai_collision_before_scene_update(self):
        scenario = self.scenario()
        scenario.competition_tunnel_scene = "static"
        scenario.clock = SimpleNamespace(domain="wall", now=lambda: 1.)
        scenario.expert_controller = Mock()
        scenario.expert_controller.is_running.return_value = True
        scenario.get_ego_motion_state = Mock(return_value={
            "x": 50., "y": 2., "speed": 0., "collision_objects": ["static-actor"]})
        scenario.record_ego_trace = Mock()
        scenario.stop_drive_control = Mock()
        with patch.object(competition_module.competition_static, "update") as scene_update:
            with self.assertRaisesRegex(RuntimeError, "expert collision.*static-actor"):
                scenario.run_expert_timeline()
        scenario.stop_drive_control.assert_called_once_with()
        scene_update.assert_not_called()

    def test_roundabout_chooses_one_common_speed_and_ordinary_full_loop_keeps_its_behavior(self):
        choices = [10., 15., 20., 25.]
        for selected in choices:
            with self.subTest(speed_kph=selected):
                scenario = self.scenario()
                scenario.random = Mock()
                scenario.random.choice.return_value = selected
                settings = {"common_speed_choices_kph": choices}
                self.assertEqual(roundabout._select_common_speed_kph(scenario, settings), selected)
                self.assertEqual(roundabout._select_common_speed_kph(scenario, settings), selected)
                scenario.random.choice.assert_called_once_with(choices)
        ordinary = self.scenario(FullLoopScenario)
        ordinary.random = Mock()
        self.assertIsNone(roundabout._select_common_speed_kph(
            ordinary, {"common_speed_choices_kph": choices}))
        ordinary.random.choice.assert_not_called()

    def test_static_tunnel_is_prepared_after_route_and_armed_before_expert_configuration_only(self):
        for scene in ("legacy", "static", "pedestrian", "vehicle"):
            with self.subTest(tunnel_scene=scene):
                scenario = self.scenario()
                scenario.competition_tunnel_scene = scene
                scenario.path_s_at_index = lambda index: float(index)
                calls = []
                with patch.object(FullLoopScenario, "prepare_full_loop_route",
                                  side_effect=lambda: calls.append("route")), patch.object(
                        competition_module.competition_static, "prepare",
                        side_effect=lambda _: calls.append("static_prepare")), patch.object(
                        competition_module.competition_static, "arm",
                        side_effect=lambda _: calls.append("static_arm")), patch.object(
                        FullLoopScenario, "configure_drive_with_retries",
                        side_effect=lambda: calls.append("expert_configure")):
                    scenario.prepare_full_loop_route()
                    scenario.configure_drive_with_retries()
                expected = (["route", "static_prepare", "static_arm", "expert_configure"]
                            if scene == "static" else ["route", "expert_configure"])
                self.assertEqual(calls, expected)
                self.assertGreater(scenario.competition_tunnel_goal_s,
                                   scenario.competition_tunnel_start_s)

    def test_early_cleanup_closes_started_runtime_ve_expert_and_trace_without_setup(self):
        scenario = self.scenario()
        bridge, ve_input, expert = Mock(), Mock(), Mock()
        trace = io.StringIO()
        scenario.runtime_bridge = bridge
        scenario.ve_object_input = ve_input
        scenario.expert_controller = expert
        scenario._ego_trace_stream = trace
        self.assertFalse(scenario._competition_setup_entered)
        with patch.object(FullLoopScenario, "cleanup") as full_cleanup:
            scenario.cleanup()
        full_cleanup.assert_not_called()
        bridge.close.assert_called_once_with()
        ve_input.close.assert_called_once_with()
        expert.stop_process.assert_called_once_with()
        self.assertTrue(trace.closed)
        self.assertIsNone(scenario.ve_object_input)
        self.assertIsNone(scenario.expert_controller)
        self.assertIsNone(scenario._ego_trace_stream)
        self.assertTrue(scenario._competition_cleanup_verified)
        self.assertEqual(scenario.grpc.method_calls, [])

    def test_static_fixture_namespace_shares_transport_without_mutating_parent_world_or_expert(self):
        controller = SimpleNamespace(planning={"maximum_speed": 37., "gain": [1., 2.]})
        parent_actors = {"Ego": object(), "dynamic-pedestrian": object()}
        parent_world = SimpleNamespace(_actors=parent_actors, _client_key="full-loop-owner",
                                       _sim_adapter=object())
        parent = SimpleNamespace(client_key="full-loop-owner", world=parent_world,
                                 client=object(), expert_controller=controller)
        before_planning = copy.deepcopy(controller.planning)
        fixture = competition_module.competition_static._fixture_client(parent)
        second = competition_module.competition_static._fixture_client(parent)
        self.assertNotEqual(fixture.client_key, parent.client_key)
        self.assertNotEqual(second.client_key, fixture.client_key)
        self.assertEqual(fixture.world._client_key, fixture.client_key)
        self.assertIsNot(fixture.world, parent_world)
        self.assertIsNot(fixture.world._actors, parent_actors)
        self.assertEqual(fixture.world._actors, {})
        self.assertIs(fixture.world._sim_adapter, parent_world._sim_adapter)
        self.assertIs(fixture.client, parent.client)
        fixture.world._actors["fixture-only"] = object()
        fixture.world._actors.clear()
        self.assertIs(parent.world, parent_world)
        self.assertIs(parent.world._actors, parent_actors)
        self.assertEqual(set(parent_actors), {"Ego", "dynamic-pedestrian"})
        self.assertEqual((parent.client_key, parent.world._client_key),
                         ("full-loop-owner", "full-loop-owner"))
        self.assertIs(parent.expert_controller, controller)
        self.assertEqual(controller.planning, before_planning)

    def test_static_cleanup_refuses_baseline_actor_binding_before_any_delete(self):
        native = SimpleNamespace(DestroyActor=Mock(), DestroyAllActors=Mock())
        actors = {"Ego": object(), "foreign": object()}
        parent = SimpleNamespace(client_key="full-loop-owner", world=SimpleNamespace(
            _client_key="full-loop-owner", _actors=actors,
            _sim_adapter=SimpleNamespace(_actor_stub=native)))
        fixture = competition_module.competition_static._fixture_client(parent)
        handle = Mock()
        handle.get_object_info.return_value = SimpleNamespace(
            client_key=fixture.client_key, id=SimpleNamespace(value="fixture-request"))
        owned = {"handle": handle, "request_id": "fixture-request", "raw_id": "foreign",
                 "source_client_key": fixture.client_key, "source_handle_id": "fixture-request"}
        state = SimpleNamespace(grpc=fixture, owned=[owned], baseline={"foreign": object()}, pins={})
        result = {"phase": "armed", "cleanup": {"verified": False, "actors": []},
                  "tunnel_passed": False, "fresh_placement_verified": True}
        context = SimpleNamespace(grpc=parent, competition_static_state=state,
                                  competition_static_result=result)
        modules = {
            "proto.morai.common.enum_pb2": SimpleNamespace(STATUS_CODE_SUCCESS=1),
            "proto.morai.common.type_pb2": SimpleNamespace(StringValue=lambda **kw: SimpleNamespace(**kw)),
        }
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "raw ID overlaps the pre-spawn baseline"):
                competition_module.competition_static.cleanup(context)
        native.DestroyActor.assert_not_called()
        native.DestroyAllActors.assert_not_called()
        self.assertFalse(result["cleanup"]["verified"])
        self.assertEqual(result["phase"], "failed")
        self.assertIs(parent.world._actors, actors)
        self.assertEqual(set(actors), {"Ego", "foreign"})


class CompetitionStaticRuntimeTest(unittest.TestCase):
    """Driving clearance is not the stationary preview's placement margin."""

    def setUp(self):
        fixture = static_fixture.LivePreviewTest("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.sim = fixture.sim
        self.module = competition_module.competition_static
        self.live = self.module.live
        self.sim.states["Ego"] = static_fixture.state(
            "Ego", 50., 2., width=2., length=4.5, model="EgoCar")
        self.sim.states["Ego"].global_velocity.x = 5.
        geometry = {
            "own_points": [(0., 0., 28.), (120., 0., 28.)],
            "opposing_points": [(120., 3.5, 28.), (0., 3.5, 28.)],
            "own_lane_width_m": 3.5, "opposing_lane_width_m": 3.5,
            "length_m": 120.,
        }
        self.state = SimpleNamespace(
            grpc=self.sim.grpc, ego_id="Ego", geometry=geometry,
            candidate={"obstacles": [{}]}, owned=[], references={}, pins={},
            baseline=copy.deepcopy(self.sim.states), last_check=float("-inf"))
        self.report = dict(phase="active", inventory_checks=0, observed=[],
                           stationary_samples=0, tunnel_passed=False, completed=False,
                           fresh_placement_verified=True,
                           cleanup={"verified": False, "actors": []})
        self.context = SimpleNamespace(
            grpc=self.sim.grpc, competition_static_state=self.state,
            competition_static_result=self.report, tunnel_exit_s=120.,
            cfg={"drive_control_mode": "aimdrive_expert"},
            expert_controller=SimpleNamespace(process=None, planning={"speed": 37.}))
        self.add_actor()
        sleeper = patch.object(self.live.time, "sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def add_actor(self, *_):
        actor = static_fixture.state("101", 50., 0.)
        actor.actor_info.client_key = "preview-client"
        self.sim.states["101"] = actor
        self.sim.aliases["fixture-request"] = "101"
        self.sim.owners["101"] = "preview-client"
        observed = self.live._observation(actor)
        self.state.references["101"] = copy.deepcopy(observed)
        self.report["observed"].append(copy.deepcopy(observed["raw_bbox_normalized"]))
        self.state.owned.append(dict(
            handle=static_fixture.Actor(self.sim, "fixture-request"),
            raw_id="101", request_id="fixture-request", actor_kind="obstacle",
            requested_model="WoodBox", source_client_key="preview-client",
            source_handle_id="fixture-request", raw_object_info=copy.deepcopy(actor.actor_info),
            raw_observation=copy.deepcopy(observed), binding_phase="frozen"))

    def test_real_arm_keeps_parked_ego_margin_rejection_for_point_four_metre_gap(self):
        self.report.update(phase="prepared", fresh_placement_verified=False)
        self.report["observed"].clear()
        self.state.owned.clear()
        self.state.references.clear()
        self.sim.states.pop("101")
        with patch.object(self.module, "_fixture_client", return_value=self.sim.grpc), patch.object(
                self.module, "_spawn_one", side_effect=self.add_actor):
            with self.assertRaisesRegex(RuntimeError, "overlaps the parked Ego"):
                self.module.arm(self.context)
        self.assertEqual(self.report["phase"], "failed")
        self.assertTrue(self.report["cleanup"]["verified"])
        self.assertEqual(set(self.sim.states), {"Ego"})

    def test_moving_ego_can_pass_with_point_four_metre_gap_without_control_changes(self):
        box = self.report["observed"][0]
        ego_box = self.live._observation(self.sim.states["Ego"])["raw_bbox_normalized"]
        gap = ego_box["center_xyz"][1] - .5 * (ego_box["width_m"] + box["width_m"])
        self.assertAlmostEqual(gap, .4)
        self.assertFalse(self.live.boxes_overlap(box, ego_box))
        with self.assertRaisesRegex(RuntimeError, "overlaps the parked Ego"):
            self.live._geometry_checks([box], ego_box, self.state.geometry)
        cfg = copy.deepcopy(self.context.cfg)
        planning = copy.deepcopy(self.context.expert_controller.planning)
        before_actor = copy.deepcopy(self.sim.states)
        self.module.update(self.context, 1., {"speed": 18.}, 50.)
        self.assertEqual(self.report["phase"], "active")
        self.assertEqual(self.report["stationary_samples"], 1)
        self.assertGreater(self.report["inventory_checks"], 0)
        self.assertEqual(self.state.owned[0]["lookup_evidence"]["status"], "verified")
        self.assertEqual(self.context.cfg, cfg)
        self.assertEqual(self.context.expert_controller.planning, planning)
        self.assertEqual(self.sim.states, before_actor)
        self.assertEqual(self.sim.commands, [])

    def test_owned_actor_identity_and_stationarity_failures_still_abort_and_attempt_cleanup(self):
        for fault, expected in (("missing", "absent"), ("model", "model or type changed"),
                                ("moved", "pose moved"), ("moving", "is moving")):
            with self.subTest(fault=fault):
                original = copy.deepcopy(self.sim.states["101"])
                if fault == "missing":
                    self.sim.states.pop("101")
                elif fault == "model":
                    self.sim.states["101"].actor_info.model = "different-model"
                elif fault == "moved":
                    self.sim.states["101"].transform.location.x += 1.
                else:
                    self.sim.states["101"].global_velocity.x = 1.
                with patch.object(self.module, "cleanup", wraps=self.module.cleanup) as cleanup:
                    with self.assertRaisesRegex(RuntimeError, expected):
                        self.module.update(self.context, self.state.last_check + 1.
                                           if self.state.last_check > 0 else 1., {}, 50.)
                cleanup.assert_called_once_with(self.context)
                self.assertEqual(self.report["phase"], "failed")
                self.assertFalse(self.report["completed"])
                # Unproven identity/body cannot be deleted as though verified.
                self.assertFalse(self.report["cleanup"]["verified"])
                self.assertEqual(self.sim.commands, [])
                self.sim.states["101"] = original
                self.report.update(phase="active", failure_reason=None)

    def test_missing_ego_still_aborts_with_unverified_cleanup(self):
        self.sim.states.pop("Ego")
        with self.assertRaisesRegex(RuntimeError, "lost the existing Ego"):
            self.module.update(self.context, 1., {}, 50.)
        self.assertEqual(self.report["phase"], "failed")
        self.assertFalse(self.report["completed"])
        self.assertFalse(self.report["cleanup"]["verified"])
        self.assertIn("baseline actors disappeared", self.report["cleanup_error"])

    def test_foreign_corridor_occupancy_still_aborts_and_cleans_only_owned_actor(self):
        foreign = static_fixture.state("foreign", 70., 0.)
        self.sim.states["foreign"] = foreign
        with self.assertRaisesRegex(RuntimeError, "Foreign actors occupy"):
            self.module.update(self.context, 1., {}, 50.)
        self.assertEqual(self.report["phase"], "failed")
        self.assertFalse(self.report["completed"])
        self.assertTrue(self.report["cleanup"]["verified"])
        self.assertEqual(set(self.sim.states), {"Ego", "foreign"})
        self.assertIs(self.sim.states["foreign"], foreign)
        self.assertFalse(any(command[0] == "Ego" for command in self.sim.commands))

    def test_original_exit_threshold_still_removes_fixture_after_close_pass(self):
        self.module.update(self.context, 1., {}, 149.9)
        self.assertEqual(self.report["phase"], "active")
        self.assertIn("101", self.sim.states)
        self.module.update(self.context, 2., {}, 150.)
        self.assertEqual(self.report["phase"], "completed")
        self.assertTrue(self.report["cleanup"]["verified"])
        self.assertTrue(self.report["tunnel_passed"])
        self.assertEqual(set(self.sim.states), {"Ego"})


class CompetitionTunnelReuseTest(unittest.TestCase):
    """Standalone tunnel handlers retain their gates on the full-course route."""

    def setUp(self):
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def vehicle(self, scene="vehicle"):
        fixture = vehicle_fixture.TunnelVehicleTest("runTest")
        self.addCleanup(fixture.doCleanups)
        context, actor = fixture.make_context("vehicle_slow")
        context.scenario_name = "full_loop_competition"
        context.cfg["competition_tunnel_scene"] = scene
        context.competition_tunnel_start_s = 10.
        return fixture, context, actor

    def test_vehicle_competition_scene_prepares_without_enabling_other_scenes(self):
        _, context, _ = self.vehicle()
        self.assertTrue(tunnel_vehicle_dynamic.prepare_tunnel_vehicle(context))
        _, other, _ = self.vehicle("static")
        self.assertFalse(tunnel_vehicle_dynamic.prepare_tunnel_vehicle(other))

    def test_vehicle_does_not_spawn_or_validate_previous_zone_lane_before_start(self):
        _, context, _ = self.vehicle()
        tunnel_vehicle_dynamic.prepare_tunnel_vehicle(context)
        handler = context.tunnel_vehicle_handler
        handler.arm(context)
        handler.update(context, 300., dict(x=9., y=10., speed=36.), 9.)
        context.grpc.spawn_vehicle.assert_not_called()
        self.assertEqual(context.tunnel_vehicle_result["spawn_wait_reason"],
                         "before_competition_tunnel_start")
        handler.update(context, 301., dict(x=10., y=0., speed=36.), 10.)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, 1)
        self.assertEqual(context.tunnel_vehicle_result["initial_route_gap_m"], 40.)

    def test_vehicle_competition_preserves_original_mapped_entry_progress(self):
        _, context, _ = self.vehicle()
        tunnel_vehicle_dynamic.prepare_tunnel_vehicle(context)
        handler = context.tunnel_vehicle_handler
        handler.arm(context)
        handler.update(context, 300., dict(x=9., y=0., speed=5.4), 9.)
        context.grpc.spawn_vehicle.assert_not_called()
        handler.update(context, 301., dict(x=10., y=0., speed=5.4), 10.)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, 1)
        self.assertAlmostEqual(context.tunnel_vehicle_result["spawn_timing"]["progress_m"],
                               10. - context.tunnel_vehicle_plan["entry_s_m"])
        self.assertAlmostEqual(context.tunnel_vehicle_plan["spawn_s_m"], 50.)

    def test_pedestrian_distance_gate_and_pre_start_eta_keep_tunnel_goal_budget(self):
        fixture = pedestrian_fixture._TunnelFixture(self)
        context, _ = fixture.context("straight_opposite_01")
        context.cfg["competition_tunnel_scene"] = "pedestrian"
        context.scenario_name = "full_loop_competition"
        context.competition_tunnel_start_s = 530.
        context.competition_tunnel_goal_s = 688.
        context.route_points = [(float(i), 0., 0.) for i in range(1201)]
        context.tunnel_entry_s, context.tunnel_exit_s = 510., 690.
        context.route_length_m = 1200.
        tunnel_dynamic.prepare_tunnel_dynamic(context)
        before = copy.deepcopy(context.tunnel_dynamic_plan)
        tunnel_dynamic.arm_tunnel_dynamic(context)
        self.assertEqual(context.tunnel_dynamic_plan, before)
        self.assertEqual(context.tunnel_dynamic_goal_s, 688.)
        self.assertEqual(context.tunnel_dynamic_plan["interaction_s"], 540.)
        fixture.tick(context, 300., 40., 4.)
        context.grpc.spawn_pedestrian.assert_not_called()
        fixture.tick(context, 301., 529., 4.)
        self.assertLess(529., context.competition_tunnel_start_s)
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)


class CompetitionVehicleCleanupTest(unittest.TestCase):
    """A continuing full-course Ego must not chase traffic past its native route."""

    def setUp(self):
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def vehicle_scene(self, competition=True, goal=135.):
        fixture = vehicle_fixture.TunnelVehicleTest("runTest")
        self.addCleanup(fixture.doCleanups)
        context, actor = fixture.make_context("vehicle_slow")
        if competition:
            context.scenario_name = "full_loop_competition"
            context.cfg["competition_tunnel_scene"] = "vehicle"
            context.competition_tunnel_start_s = 0.
            context.competition_tunnel_goal_s = goal
        tunnel_vehicle_dynamic.prepare_tunnel_vehicle(context)
        handler = context.tunnel_vehicle_handler
        handler.arm(context)
        fixture.start(context, handler)
        fixture.tick(context, actor, handler, 5., 60., 45.)
        self.assertTrue(context.tunnel_vehicle_result["encounter_verified"])
        return fixture, context, actor, handler

    def test_moving_ego_completes_only_tunnel_scene_after_verified_continuation(self):
        fixture, context, actor, handler = self.vehicle_scene()
        fixture.tick(context, actor, handler, 35., 180., 167.)
        result = context.tunnel_vehicle_result
        self.assertTrue(result["interaction_complete"])
        self.assertTrue(result["completed"])
        self.assertEqual(context.tunnel_vehicle_phase, "completed")
        context.grpc.destroy_vehicle_by_id.assert_called_once_with("npc-1")
        self.assertEqual(context.scenario_actors, [])
        self.assertEqual(result["cleanup_reason"], "competition_tunnel_scene_boundary")
        self.assertAlmostEqual(result["cleanup_actual_ego_clearance_m"], 13.)
        self.assertEqual(result["cleanup_required_ego_clearance_m"], 30.)
        self.assertFalse(result["cleanup_ego_clearance_met"])
        self.assertEqual(result["cleanup_scene_boundary"]["ego_s_m"], 167.)
        self.assertEqual(result["cleanup_scene_boundary"]["ego_goal_s_m"], 135.)
        context.grpc.stop_vehicle.assert_not_called()
        actor.set_pause.assert_called_once_with(False)
        # Later Ego ticks continue; the removed native actor is never queried again.
        actor.get_actor_state.return_value = None
        handler.update(context, 36., {"x": 171., "y": 0., "speed": 14.4}, 171.)
        self.assertTrue(result["completed"])

    def test_standalone_retains_original_thirty_metre_clearance_gate(self):
        fixture, context, actor, handler = self.vehicle_scene(competition=False)
        fixture.tick(context, actor, handler, 35., 180., 167.)
        self.assertTrue(context.tunnel_vehicle_result["interaction_complete"])
        self.assertFalse(context.tunnel_vehicle_result["completed"])
        context.grpc.destroy_vehicle_by_id.assert_not_called()
        self.assertIs(context.tunnel_vehicle_actor, actor)

    def test_original_clearance_cleanup_still_works_before_competition_scene_goal(self):
        fixture, context, actor, handler = self.vehicle_scene()
        fixture.tick(context, actor, handler, 35., 180., 120.)
        self.assertTrue(context.tunnel_vehicle_result["completed"])
        self.assertEqual(context.tunnel_vehicle_result["cleanup_reason"],
                         "outside_tunnel_continuation_and_ego_clearance")

    def test_scene_cleanup_requires_ego_to_reach_dedicated_tunnel_goal(self):
        fixture, context, actor, handler = self.vehicle_scene(goal=200.)
        fixture.tick(context, actor, handler, 35., 180., 167.)
        self.assertFalse(context.tunnel_vehicle_result["completed"])
        context.grpc.destroy_vehicle_by_id.assert_not_called()

    def test_scene_cleanup_requires_npc_to_reach_original_continuation_boundary(self):
        fixture, context, actor, handler = self.vehicle_scene()
        fixture.tick(context, actor, handler, 30., 170., 157.)
        self.assertTrue(context.tunnel_vehicle_result["interaction_complete"])
        self.assertFalse(context.tunnel_vehicle_result["completed"])
        context.grpc.destroy_vehicle_by_id.assert_not_called()

    def test_scene_boundary_cannot_replace_unverified_interaction(self):
        _, context, _, handler = self.vehicle_scene()
        context.tunnel_vehicle_last_ego_s = 167.
        context.tunnel_vehicle_result["interaction_complete"] = False
        ready = handler._cleanup_if_ready(
            context, {"x": 180., "y": 0., "npc_route_s_m": 180.},
            {"x": 167., "y": 0., "speed": 14.4})
        self.assertFalse(ready)
        self.assertFalse(context.tunnel_vehicle_result["completed"])
        context.grpc.destroy_vehicle_by_id.assert_not_called()

    def test_scene_boundary_does_not_turn_failed_brake_recovery_into_success(self):
        fixture = vehicle_fixture.TunnelVehicleTest("runTest")
        self.addCleanup(fixture.doCleanups)
        context, actor, handler = fixture.stable_brake()
        context.scenario_name = "full_loop_competition"
        context.cfg["competition_tunnel_scene"] = "vehicle"
        context.competition_tunnel_goal_s = 135.
        fixture.tick_after_brake(context, actor, handler, .25, .125, gap=6, ego_speed=3.6)
        result = context.tunnel_vehicle_result
        original_failure = result["failure_reason"]
        self.assertIn("emergency gap", original_failure)
        self.assertFalse(result["interaction_complete"])
        fixture.tick_after_brake(context, actor, handler, .5, 1.125, gap=7, speed=4.)
        self.assertTrue(result["brake"]["recovery_motion_verified"])
        with self.assertRaisesRegex(RuntimeError, "emergency gap"):
            fixture.tick(context, actor, handler, 35., 180., 167., speed=4.)
        context.grpc.destroy_vehicle_by_id.assert_called_once_with("npc-1")
        self.assertTrue(result["cleanup_verified"])
        self.assertFalse(result["cleanup_ego_clearance_met"])
        self.assertAlmostEqual(result["cleanup_actual_ego_clearance_m"], 13.)
        self.assertEqual(result["failure_reason"], original_failure)
        self.assertEqual(result["brake"]["failure_reason"], original_failure)
        self.assertTrue(result["failed"])
        self.assertFalse(result["completed"])
        self.assertFalse(result["interaction_complete"])
        self.assertEqual(context.tunnel_vehicle_phase, "failed")
        self.assertEqual(context.scenario_actors, [])
        saved = json.loads(context.tunnel_vehicle_report_path.read_text())
        self.assertEqual(saved["failure_reason"], original_failure)
        self.assertFalse(saved["completed"])
        # The native route can disappear afterwards; cleanup has already ended
        # observation and retains the original failure instead of "unavailable".
        actor.get_actor_state.return_value = None
        handler.update(context, 36., {"x": 171., "y": 0., "speed": 14.4}, 171.)
        self.assertEqual(result["failure_reason"], original_failure)

    def test_scene_boundary_cannot_replace_unobserved_failed_brake_recovery(self):
        _, context, _, handler = self.vehicle_scene()
        context.tunnel_vehicle_last_ego_s = 167.
        result = context.tunnel_vehicle_result
        result["failure_reason"] = "NPC did not maintain trough speed"
        result["brake"].update(failure_pending=True, recovery_motion_verified=False)
        for interaction_complete in (False, True):
            with self.subTest(stale_interaction_complete=interaction_complete):
                result["interaction_complete"] = interaction_complete
                self.assertFalse(handler._cleanup_if_ready(
                    context, {"x": 180., "y": 0., "npc_route_s_m": 180.},
                    {"x": 167., "y": 0., "speed": 14.4}))
                context.grpc.destroy_vehicle_by_id.assert_not_called()
                self.assertFalse(result["completed"])

    def test_missing_actor_state_still_fails_without_verified_scene_cleanup(self):
        _, context, actor, handler = self.vehicle_scene()
        actor.get_actor_state.return_value = None
        context.now = 35.
        with self.assertRaisesRegex(RuntimeError, "state unavailable"):
            handler.update(context, 35., {"x": 167., "y": 0., "speed": 14.4}, 167.)
        self.assertFalse(context.tunnel_vehicle_result["completed"])
        self.assertEqual(context.tunnel_vehicle_phase, "failed")


if __name__ == "__main__":
    unittest.main()
