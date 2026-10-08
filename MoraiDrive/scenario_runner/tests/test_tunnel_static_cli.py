"""CLI preparation must remain distinct from live Expert validation."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ..main_runner import load_configuration
from ..tools import tunnel_static_candidates as cli


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


if __name__ == "__main__":
    unittest.main()
