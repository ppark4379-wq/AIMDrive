"""Tunnel regression checks: commands."""

import contextlib, copy, io, json, subprocess, sys, tempfile, unittest, ast
from pathlib import Path
from unittest.mock import patch, Mock
from ..main_runner import load_configuration
from ..tools import tunnel_static_candidates as cli
from types import ModuleType
from .. import main_runner


# CLI preparation must remain distinct from live Expert validation.

class TunnelStaticCliTest(unittest.TestCase):
    def test_expanded_policy_passes_pattern_rotation_and_domain_without_mutating_config(self):
        config = dict(layout=dict(max_width_m=6.7, lateral_limit_m=.25),
            selected_pattern='short_detour', footprint_domain='two_lane_physical_road',
            static_yaw_offsets_deg={'Steel_Barricade': [90.]},
            station_window_m=[20., 105.], lateral_regions=None)
        original = copy.deepcopy(config)
        options = cli.generation_options(config, 'alternating_three')
        self.assertEqual(options['pattern'], 'alternating_three')
        self.assertEqual(options['footprint_domain'], 'two_lane_physical_road')
        self.assertEqual(options['static_yaw_offsets_deg'], {'Steel_Barricade': [90.]})
        options['static_yaw_offsets_deg']['Steel_Barricade'][0] = 0.
        self.assertEqual(config, original)

    def test_unknown_pattern_and_missing_explicit_two_lane_domain_fail_closed(self):
        for config, pattern in [(dict(layout={}), 'alternating_three'),
                                (dict(layout={}, footprint_domain='two_lane_physical_road'), 'invented')]:
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                cli.generation_options(config, pattern)

    def test_legacy_generator_kwargs_remain_identical_without_new_options(self):
        layout = dict(max_width_m=2.8, lateral_limit_m=.25, min_pair_gap_m=18.)
        self.assertEqual(cli.generation_options(dict(layout=layout)), layout)

    def test_three_requires_explicit_opt_in_before_configuration_or_connection(self):
        args = cli.build_parser().parse_args(["--count", "3"])
        with patch.object(cli, "load_configuration") as load, self.assertRaisesRegex(ValueError, "experimental-three"):
            cli.run(args)
        load.assert_not_called()

    def test_opt_in_three_keeps_geometry_and_expert_claims_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            args = cli.build_parser().parse_args(["--seed", "42", "--count", "3", "--experimental-three",
                "--actor-kind", "obstacle", "--obstacle-model", "WoodBox", "--output", directory])
            with patch.object(cli, "screen_candidate", return_value={"status": "nominal_path_found", "path_found": True}), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli.run(args)
            candidate = json.loads((Path(directory)/"candidate_000/candidate.json").read_text())
            self.assertEqual(candidate["count"], 3)
            self.assertIs(candidate["experimental_three_actor_trial"], True)
            self.assertGreaterEqual(candidate["placement"]["geometry"]["minimum_center_gap_m"], 18.)
            self.assertFalse(candidate["expert_verified"])
            self.assertFalse(candidate["live_avoidance_verified"])

    def test_importing_zone_entrypoint_does_not_start_simulator_or_expert(self):
        # A fresh interpreter detects import side effects that a shared test
        # process could hide after another test has loaded Expert or the SDK.
        result = subprocess.run(
            [sys.executable, "-c", "\n".join((
                "import sys",
                "paths_before = list(sys.path)",
                "from scenario_runner.scenarios.full_loop_parts import tunnel_static",
                "assert sys.path == paths_before",
                "assert not any(name == 'expert' or name.startswith('expert.') for name in sys.modules)",
                "assert not any(name.startswith(('proto.morai', 'rospy', 'rosgraph')) for name in sys.modules)",
            ))],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_offline_preparation_never_connects_or_claims_expert_success(self):
        with contextlib.redirect_stdout(io.StringIO()):
            global_cfg, zone_cfg, scenario_cfg = load_configuration()
        with tempfile.TemporaryDirectory() as temporary:
            cfg = copy.deepcopy(global_cfg)
            cfg["paths"]["runtime_root"] = temporary
            args = cli.build_parser().parse_args([
                "--seed", "42", "--count", "2", "--candidates", "2",
                "--output", temporary,
            ])
            screened = {"status": "nominal_path_found", "path_found": True,
                        "geometry_checks": {"passed": True}, "expert_verified": False,
                        "live_avoidance_verified": False}
            with patch.object(cli, "load_configuration", return_value=(cfg, zone_cfg, scenario_cfg)), \
                    patch.object(cli, "screen_candidate", return_value=screened), \
                    patch.object(cli, "_controller_publishers", side_effect=AssertionError("No ROS in offline mode")), \
                    patch("scenario_runner.adapters.morai_grpc.MoraiGrpcClient.connect",
                          side_effect=AssertionError("No simulator in offline mode")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = cli.run(args)
            self.assertEqual(result["status"], "prepared")
            self.assertFalse(result["expert_verified"])
            self.assertFalse(result["live_avoidance_verified"])
            for row in result["results"]:
                candidate = json.loads(Path(row["candidate_path"]).read_text())
                self.assertEqual(candidate["count"], 2)
                self.assertFalse(candidate["expert_verified"])
                self.assertEqual(candidate["seed"], 42)
                self.assertTrue(all(item["geometry_source"] for item in candidate["obstacles"]))

    def test_search_timeout_is_a_rejected_candidate_not_an_approved_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = cli.build_parser().parse_args(["--seed", "42", "--count", "1", "--output", temporary])
            with patch.object(cli, "screen_candidate", return_value={
                    "status": "search_timeout", "path_found": False,
                    "geometric_impossibility_proven": False}), contextlib.redirect_stdout(io.StringIO()):
                result = cli.run(args)
            self.assertEqual(result["status"], "screening_rejected")
            self.assertFalse(result["live_avoidance_verified"])
            stored = json.loads((Path(temporary) / "candidate_000/screening.json").read_text())
            self.assertFalse(stored["geometric_impossibility_proven"])

    def test_preview_batch_is_rejected_before_any_connection(self):
        args = cli.build_parser().parse_args(["--preview", "--candidates", "2"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            cli.run(args)

    def test_live_preview_refuses_an_active_expert_before_world_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = cli.build_parser().parse_args(["--preview", "--output", temporary])
            with patch.object(cli, "_controller_publishers", return_value=["/active_expert"]), \
                    patch("scenario_runner.adapters.morai_grpc.MoraiGrpcClient.connect") as connect, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "active Ego controller"):
                    cli.run(args)
            connect.assert_not_called()
            report = json.loads((Path(temporary) / "manifest.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["expert_verified"])

    def test_current_live_model_acceptance_limits_offline_random_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "acceptance.json"
            receipt.write_text('{"schema":"test_receipt"}')
            args = cli.build_parser().parse_args([
                "--seed", "44", "--count", "2", "--output", temporary,
                "--model-acceptance", str(receipt),
            ])
            def only_tested(path, statics, vehicles, **kwargs):
                self.assertTrue(kwargs["source_geometry_sha256"])
                return {"WoodBox": statics["WoodBox"]}, {"DEFAULT_SEDAN": vehicles["DEFAULT_SEDAN"]}
            with patch("scenario_runner.tools.verify_tunnel_models.accepted_model_pools", side_effect=only_tested), \
                    patch.object(cli, "screen_candidate", return_value={
                        "status": "nominal_path_found", "path_found": True}), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = cli.run(args)
            self.assertEqual(result["catalog_model_pools"], {
                "obstacle": ["WoodBox"], "vehicle": ["DEFAULT_SEDAN"],
            })
            self.assertTrue(result["model_acceptance"]["sha256"])


# Main-command tunnel replay dispatch must precede the generic live runner.

REPLAY_MODULE = "scenario_runner.scenarios.segment_modes.tunnel_static_replay"
MODE = "full_loop_tunnel_static"


class TunnelStaticMainCommandTests(unittest.TestCase):
    def setUp(self):
        self.global_cfg = {"paths": {"mgeo_root": "/mock/mgeo", "runtime_root": "/mock/runtime"}}
        self.zone_cfg = {"zone": "full_loop"}
        self.scenario_cfg = {"sync_runtime": {"enabled": True}, "auto_repeat": True,
                             "repeat_delay_sec": 0.}
        self.dispatch = Mock(return_value=0)
        replay_module = ModuleType(REPLAY_MODULE)
        replay_module.run_tunnel_static_scenario = self.dispatch
        self.patch_module = patch.dict(sys.modules, {REPLAY_MODULE: replay_module})
        self.patch_module.start()
        self.addCleanup(self.patch_module.stop)
        self.load = Mock(side_effect=lambda: (copy.deepcopy(self.global_cfg), copy.deepcopy(self.zone_cfg),
                                             copy.deepcopy(self.scenario_cfg)))
        self.grpc = Mock(side_effect=AssertionError("Generic SDK constructor must not precede tunnel dispatch"))
        self.loader = Mock(side_effect=AssertionError("Generic map constructor must not precede tunnel dispatch"))
        self.offline = Mock()
        for name, replacement in (("load_configuration", self.load), ("MoraiGrpcClient", self.grpc),
                                  ("MGeoMapLoader", self.loader), ("run_offline_check", self.offline)):
            patcher = patch.object(main_runner, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def invoke(self, arguments):
        with patch.object(sys, "argv", ["scenario_runner.main_runner", *arguments]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main_runner.main()

    def assert_generic_runner_unused(self):
        self.grpc.assert_not_called()
        self.loader.assert_not_called()
        self.offline.assert_not_called()

    def test_default_tunnel_command_dispatches_once_before_any_generic_live_runner(self):
        self.assertEqual(self.invoke(["--zone", "full_loop", "--scenario", MODE]), 0)
        self.load.assert_called_once_with()
        self.dispatch.assert_called_once()
        args, global_cfg, scenario_cfg = self.dispatch.call_args.args
        self.assertEqual(args.scenario, MODE)
        self.assertFalse(args.once)
        self.assertFalse(args.check)
        self.assertFalse(args.no_sync_runtime)
        self.assertIsNone(args.max_runs)
        self.assertIsNone(args.tunnel_static_library)
        self.assertIsNone(args.tunnel_static_seed)
        self.assertIsNone(args.tunnel_static_output)
        self.assertEqual(global_cfg, self.global_cfg)
        self.assertEqual(scenario_cfg["mission_mode"], MODE)
        self.assert_generic_runner_unused()

    def test_once_and_no_sync_are_passed_to_guarded_dispatch_after_option_validation(self):
        self.assertEqual(self.invoke(["--zone", "full_loop", "--scenario", MODE,
                                      "--once", "--no-sync-runtime"]), 0)
        args, _global_cfg, scenario_cfg = self.dispatch.call_args.args
        self.assertTrue(args.once)
        self.assertTrue(args.no_sync_runtime)
        self.assertFalse(scenario_cfg["sync_runtime"]["enabled"])
        self.assert_generic_runner_unused()

    def test_explicit_library_seed_output_and_max_runs_keep_their_typed_values(self):
        arguments = ["--scenario", MODE, "--max-runs", "4", "--tunnel-static-library", "fixture_library.json",
                     "--tunnel-static-seed", "73", "--tunnel-static-output", "fixture_replays"]
        self.assertEqual(self.invoke(arguments), 0)
        args = self.dispatch.call_args.args[0]
        self.assertEqual(args.max_runs, 4)
        self.assertFalse(args.once)
        self.assertEqual(args.tunnel_static_seed, 73)
        self.assertIsInstance(args.tunnel_static_seed, int)
        self.assertEqual(args.tunnel_static_library, Path("fixture_library.json"))
        self.assertEqual(args.tunnel_static_output, Path("fixture_replays"))
        self.assert_generic_runner_unused()

    def test_tunnel_check_routes_library_validation_through_replay_helper(self):
        self.assertEqual(self.invoke(["--scenario", MODE, "--check",
                                      "--tunnel-static-library", "fixture_library.json"]), 0)
        self.assertTrue(self.dispatch.call_args.args[0].check)
        self.dispatch.assert_called_once()
        self.assert_generic_runner_unused()

    def test_invalid_generic_run_limit_rejects_before_configuration_or_dispatch(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(SystemExit) as raised:
                self.invoke(["--scenario", MODE, "--max-runs", value])
            self.assertEqual(raised.exception.code, 2)
        self.load.assert_not_called()
        self.dispatch.assert_not_called()
        self.assert_generic_runner_unused()

    def test_bad_seed_type_rejects_before_configuration_or_dispatch(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke(["--scenario", MODE, "--tunnel-static-seed", "not_an_integer"])
        self.assertEqual(raised.exception.code, 2)
        self.load.assert_not_called()
        self.dispatch.assert_not_called()
        self.assert_generic_runner_unused()

    def test_options_for_other_zones_cannot_enter_the_tunnel_helper(self):
        selections = (["--static-obstacle-count", "2"], ["--static-preset", "fixture"],
                      ["--static-editor", "--no-sync-runtime"],
                      ["--dynamic-review", "crossing:actor:2", "--once", "--no-sync-runtime"],
                      ["--highway-object-audit"], ["--ego-trace", "fixture.csv"])
        for extra in selections:
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as raised:
                self.invoke(["--scenario", MODE, *extra])
            self.assertEqual(raised.exception.code, 2)
        self.dispatch.assert_not_called()
        self.assert_generic_runner_unused()

    def test_tunnel_options_reject_other_scenarios_even_when_the_seed_is_zero(self):
        for extra in (["--tunnel-static-seed", "0"], ["--tunnel-static-library", "fixture.json"],
                      ["--tunnel-static-output", "fixture_output"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as raised:
                self.invoke(["--scenario", "full_loop_normal", "--check", *extra])
            self.assertEqual(raised.exception.code, 2)
        self.load.assert_not_called()
        self.dispatch.assert_not_called()
        self.assert_generic_runner_unused()

    def test_helper_exit_code_is_returned_by_main(self):
        for code in (0, 1, 2, 17):
            self.dispatch.return_value = code
            with self.subTest(code=code):
                self.assertEqual(self.invoke(["--scenario", MODE, "--once"]), code)
        self.assert_generic_runner_unused()

    def test_helper_validation_failure_exits_before_generic_sdk_construction(self):
        for error in (ValueError("Invalid library"), RuntimeError("Stale proof"), OSError("Missing library")):
            self.dispatch.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(SystemExit) as raised:
                self.invoke(["--scenario", MODE, "--check"])
            self.assertEqual(raised.exception.code, 2)
        self.assert_generic_runner_unused()

    def test_other_scenario_check_keeps_existing_offline_path(self):
        self.invoke(["--scenario", "full_loop_normal", "--check"])
        self.offline.assert_called_once()
        self.assertEqual(self.offline.call_args.args[2], "full_loop_normal")
        self.dispatch.assert_not_called()
        self.grpc.assert_not_called()
        self.loader.assert_not_called()

    def test_other_scenario_once_keeps_existing_main_runner_path(self):
        self.grpc.side_effect = None
        self.loader.side_effect = None
        scenario_class = Mock()
        with patch.dict(main_runner.SUPPORTED_SCENARIOS, {"full_loop_normal": scenario_class}):
            self.invoke(["--scenario", "full_loop_normal", "--once", "--no-sync-runtime"])
        self.grpc.assert_called_once_with(self.global_cfg)
        self.grpc.return_value.connect.assert_called_once_with()
        self.grpc.return_value.stop.assert_called_once_with()
        scenario_class.return_value.run.assert_called_once_with()
        self.assertFalse(scenario_class.call_args.kwargs["scenario_cfg"]["sync_runtime"]["enabled"])
        self.dispatch.assert_not_called()
        self.offline.assert_not_called()

    def test_module_entrypoint_preserves_main_exit_code(self):
        # Execute the actual entrypoint guard with a mocked main: this checks
        # process exit semantics without importing a second live-run namespace.
        tree = ast.parse(Path(main_runner.__file__).read_text(encoding="utf-8"))
        guard = next(statement for statement in reversed(tree.body) if isinstance(statement, ast.If))
        executable = compile(ast.Module(body=[guard], type_ignores=[]), str(main_runner.__file__), "exec")
        for code in (0, 1, 17):
            main = Mock(return_value=code)
            with self.subTest(code=code), self.assertRaises(SystemExit) as raised:
                exec(executable, {"__name__": "__main__", "main": main, "sys": sys})
            self.assertEqual(raised.exception.code, code)
            main.assert_called_once_with()
        self.assert_generic_runner_unused()


if __name__ == "__main__":
    unittest.main()
