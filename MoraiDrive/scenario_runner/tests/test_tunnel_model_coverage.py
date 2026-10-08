"""Coverage audit distinguishes geometry, live acceptance, and infrastructure failure."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..main_runner import load_configuration
from ..tools import verify_tunnel_models as coverage


class TunnelModelCoverageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "coverage"
        with contextlib.redirect_stdout(io.StringIO()):
            self.global_cfg, self.zones, self.scenarios = load_configuration()
        self.grpc = NS(client=NS(disconnect=Mock()), get_ego=Mock())
        self.screen = {"status": "nominal_path_found", "path_found": True,
                       "geometry_checks": {"passed": True}, "geometric_impossibility_proven": False}
        self.screen.update(coverage.screening_provenance())

    def args(self, *extra):
        return coverage.build_parser().parse_args(["--output", str(self.output), "--model", "WoodBox", *extra])

    def run_audit(self, args, **patches):
        changes = {"screen_candidate": Mock(return_value=self.screen),
                   "_connect_parked_world": Mock(return_value=self.grpc),
                   "_available_models": Mock(return_value={"obstacle": {"WoodBox"}, "vehicle": {"DEFAULT_SEDAN"}}),
                   "_simulator_version": Mock(return_value="test simulator"),
                   "_controller_publishers": Mock(return_value=[]), "_inventory": Mock(return_value={}),
                   "_park": Mock(), **patches}
        with contextlib.ExitStack() as stack:
            for key, value in changes.items():
                stack.enter_context(patch.object(coverage, key, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = coverage.run(args)
        return result, changes

    def preview(self, grpc, loader, candidate, *, hold_seconds, output_dir, screen_fn):
        report = {"status": "valid_preview", "schema": "aimdrive_tunnel_static_preview_v1",
                  "hold_seconds": hold_seconds, "stationary_observed_seconds": hold_seconds + .01,
                  "stationary_samples": 26, "planned": copy.deepcopy(candidate),
                  "cleanup": {"verified": True}, "route_screened": True,
                  "measured_route_screen": self.screen, "observed": [{"model": candidate["obstacles"][0]["model"],
                        "actor_kind": candidate["obstacles"][0]["actor_kind"]}],
                  "expert_verified": False, "live_avoidance_verified": False}
        coverage._write(output_dir / "preview.json", report)
        return report

    def test_offline_geometry_does_not_connect_or_approve_live_models(self):
        result, changes = self.run_audit(self.args())
        self.assertEqual(result["status"], "completed_offline")
        self.assertEqual(result["results"][0]["status"], "nominal_geometry_passed")
        self.assertFalse(result["results"][0]["live_tested"])
        self.assertEqual(result["accepted_models"], [])
        changes["_connect_parked_world"].assert_not_called()
        changes["_controller_publishers"].assert_not_called()
        candidate = json.loads(next(self.output.glob("model_*/candidate.json")).read_text())
        self.assertEqual(candidate["count"], 1)
        self.assertAlmostEqual(candidate["obstacles"][0]["station_m"], 45., places=5)
        self.assertEqual(candidate["obstacles"][0]["lateral_m"], 0.)
        self.assertEqual(candidate["obstacles"][0]["scale"], [1., 1., 1.])

    def test_timeout_retains_failure_evidence_and_does_not_call_model_unsuitable(self):
        timeout = {"status": "search_timeout", "path_found": False, "geometric_impossibility_proven": False}
        result, changes = self.run_audit(self.args("--live"), screen_candidate=Mock(return_value=timeout),
                                         preview_candidate=Mock(side_effect=AssertionError("Must not spawn")))
        row = result["results"][0]
        self.assertEqual(row["status"], "nominal_screening_inconclusive")
        self.assertEqual(row["screening_status"], "search_timeout")
        self.assertFalse(row["geometric_impossibility_proven"])
        self.assertFalse(row["live_tested"])
        changes["preview_candidate"].assert_not_called()
        self.assertTrue(next(self.output.glob("model_*/screening.json")).is_file())

    def test_unavailable_model_is_not_silently_removed_from_coverage(self):
        result, changes = self.run_audit(self.args("--live"),
            _available_models=Mock(return_value={"obstacle": set(), "vehicle": set()}))
        self.assertEqual(result["results"][0]["status"], "skipped_unavailable")
        self.assertFalse(result["results"][0]["live_tested"])
        changes["screen_candidate"].assert_not_called()

    def test_controller_preflight_precedes_connection_and_ego_mutation(self):
        with self.assertRaisesRegex(RuntimeError, "active Ego controller"):
            self.run_audit(self.args("--live"), _controller_publishers=Mock(return_value=["/expert"]))
        self.grpc.get_ego.assert_not_called()
        report = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["not_run_models"]), 1)

    def test_hold_less_than_five_seconds_is_rejected_before_connection(self):
        with self.assertRaisesRegex(ValueError, "5..600"):
            coverage.run(self.args("--live", "--hold-seconds", "4.99"))
        self.assertFalse(self.output.exists())

    def test_existing_output_refused_without_overwriting_evidence(self):
        self.output.mkdir()
        retained = self.output / "manifest.json"
        retained.write_text("existing evidence")
        with self.assertRaises(FileExistsError):
            self.run_audit(self.args())
        self.assertEqual(retained.read_text(), "existing evidence")

    def test_live_acceptance_is_hash_linked_and_source_catalog_is_unchanged(self):
        statics = coverage.load_static_models()
        catalog = Path(coverage.package_path("config/static_obstacle_footprints_v1.json"))
        before = catalog.read_bytes()
        result, _ = self.run_audit(self.args("--live"), preview_candidate=Mock(side_effect=self.preview))
        row = result["results"][0]
        self.assertTrue(row["live_tested"])
        self.assertTrue(row["live_accepted"])
        self.assertFalse(result["expert_verified"])
        acceptance_path = self.output / "acceptance.json"
        accepted_static, accepted_vehicle = coverage.accepted_model_pools(acceptance_path, statics, {},
            source_geometry_sha256=result["source_geometry_sha256"])
        self.assertEqual(list(accepted_static), ["WoodBox"])
        self.assertFalse(accepted_vehicle)
        self.assertEqual(catalog.read_bytes(), before)
        self.grpc.client.disconnect.assert_called_once()
        changed = copy.deepcopy(statics)
        changed["WoodBox"]["geometry_source"]["catalog_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            coverage.accepted_model_pools(acceptance_path, changed, {})
        preview_path = next(self.output.glob("model_*/preview.json"))
        preview_path.write_text(preview_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "preview hash mismatch"):
            coverage.accepted_model_pools(acceptance_path, statics, {})

    def test_cleanup_failure_stops_before_next_model_and_retains_failure(self):
        def failed(grpc, loader, candidate, **kwargs):
            return {"status": "failed", "failure_reason": "Owned actors remain after cleanup",
                    "cleanup": {"verified": False, "errors": ["leaked"]}}
        args = self.args("--live", "--model", "NCAP_GVT")
        available = {"obstacle": {"WoodBox", "NCAP_GVT"}, "vehicle": set()}
        result, changes = self.run_audit(args, _available_models=Mock(return_value=available),
                                         preview_candidate=Mock(side_effect=failed))
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(len(result["not_run_models"]), 1)
        self.assertFalse(result["results"][0]["live_accepted"])
        changes["preview_candidate"].assert_called_once()
        self.assertEqual(json.loads((self.output / "acceptance.json").read_text())["models"], [])

    def test_ego_command_guard_checks_new_competing_controller(self):
        actor = Mock()
        guarded = coverage._GuardedEgo(actor)
        with patch.object(coverage, "_controller_publishers", return_value=["/new_controller"]):
            with self.assertRaisesRegex(RuntimeError, "active Ego controller"):
                guarded.set_velocity(0.)
            with self.assertRaisesRegex(RuntimeError, "active Ego controller"):
                guarded.set_vehicle_gear(1)
        actor.set_velocity.assert_not_called()
        actor.set_vehicle_gear.assert_not_called()

    def test_owned_sdk_calls_receive_deadlines_including_spawn_and_cleanup(self):
        stub = Mock()
        bounded = coverage._BoundedStub(stub)
        request = object()
        bounded.SpawnVehicle(request)
        bounded.DestroyActor(request)
        bounded.GetActorState(request, timeout=.3)
        stub.SpawnVehicle.assert_called_once_with(request, timeout=1.)
        stub.DestroyActor.assert_called_once_with(request, timeout=1.)
        stub.GetActorState.assert_called_once_with(request, timeout=.3)

    def test_configured_vehicle_without_reviewed_geometry_stays_in_report(self):
        args = coverage.build_parser().parse_args(["--output", str(self.output), "--model", "2015_Toyota_Prius(HEV)"])
        result, changes = self.run_audit(args)
        self.assertEqual(result["requested_model_count"], 1)
        self.assertEqual(result["results"][0]["status"], "skipped_no_reviewed_geometry")
        self.assertFalse(result["results"][0]["geometry_verified"])
        self.assertFalse(result["results"][0]["live_tested"])
        changes["screen_candidate"].assert_not_called()

    def test_offline_cache_rejects_changed_configuration_before_live_connection(self):
        manifest = Path(self.temp.name) / "old_manifest.json"
        manifest.write_text(json.dumps({"schema": "aimdrive_tunnel_model_coverage_v1",
            "status": "completed_offline", "config_sha256": "a" * 64}))
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            self.run_audit(self.args("--live", "--offline-manifest", str(manifest)))
        self.grpc.get_ego.assert_not_called()

    def test_offline_cache_does_not_reuse_changed_planner(self):
        summary = {"config_sha256": "cfg", "source_geometry_sha256": "map", "seed": 42,
                   "station_m": 45., "time_budget_sec": 8.}
        directory = Path(self.temp.name) / "cached"
        coverage._write(directory / "candidate.json", {"candidate": "data"})
        coverage._write(directory / "screening.json", {"planner_source_sha256": {
            "expert/src/expert/src/planning/highway_lane_planning/static_corridor.py": "f" * 64}})
        manifest = Path(self.temp.name) / "manifest.json"
        coverage._write(manifest, {"schema": "aimdrive_tunnel_model_coverage_v1", "status": "completed_offline",
            **summary, "results": [{"actor_kind": "obstacle", "model": "WoodBox", "report_dir": str(directory)}]})
        self.assertEqual(coverage._offline_cache(manifest, summary), {})

    def test_live_attach_never_calls_sdk_world_constructor_or_global_cleanup(self):
        class World:
            def __init__(self, *args, **kwargs):
                raise AssertionError("SDK constructor performs broad cleanup")

        actor = Mock()
        actor.set_transform.return_value = True
        client = NS(_sim_adapter=NS(_actor_stub=Mock(), _simulator_stub=Mock()), disconnect=Mock())
        grpc = NS(client=client, client_key="unique-client", connect=Mock(),
                  _patch_spawn_diagnostics=Mock(), get_ego=lambda: actor,
                  make_transform=Mock(), start_world=Mock(), resume_world=Mock(), reset_actors=Mock(), stop=Mock())
        modules = {"api.simulation_world": NS(SimulationWorld=World),
                   "proto.morai.simulation.simulation_enum_pb2": NS(SYNC_MODE_TYPE_UNSPECIFIED=0)}
        with patch.dict(sys.modules, modules), \
                patch("scenario_runner.adapters.morai_grpc.MoraiGrpcClient", return_value=grpc), \
                patch.object(coverage, "_park", return_value=(NS(actor_info=NS(id=NS(value="Ego"))), {})), \
                patch.object(coverage, "_inventory", return_value={}), \
                patch.object(coverage, "_occupancy"), \
                patch.object(coverage, "_controller_publishers", return_value=[]), \
                patch("scenario_runner.scenarios.segment_modes.tunnel_static_startup.block_and_park") as block, \
                patch("scenario_runner.scenarios.segment_modes.tunnel_static_startup.align_parked_ego") as align, \
                patch("scenario_runner.tools.run_tunnel_static_trial.settle_initial_ego") as settle:
            attached = coverage._connect_parked_world(self.global_cfg, {})
        self.assertIs(attached, grpc)
        self.assertIsInstance(grpc.world, World)
        self.assertIsInstance(grpc.world._sim_adapter._actor_stub, coverage._BoundedStub)
        grpc.start_world.assert_not_called()
        grpc.resume_world.assert_not_called()
        grpc.reset_actors.assert_not_called()
        grpc.stop.assert_not_called()
        client.disconnect.assert_not_called()
        self.assertEqual(block.call_count, 3)
        settle.assert_called_once_with(grpc)
        align.assert_called_once_with(grpc, {}, settle_fn=settle)


if __name__ == "__main__":
    unittest.main()
