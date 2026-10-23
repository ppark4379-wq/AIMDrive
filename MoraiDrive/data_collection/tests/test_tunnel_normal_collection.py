"""Negative tunnel examples require capture-era empty-road and drive evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / "data_collection/internal_sensor_record"
sys.path[:0] = [str(PROJECT), str(INTERNAL), str(INTERNAL / "tools")]
from async_capture import tunnel_normal_collection as normal
from scenario_collection import configure_collection_scenario, tunnel_collection_evidence
from build_sensor_record_training_dataset import required_training_coverage
from morai_pretraining.sensor_record_manifest import detection_for_entry, coverage_failed_splits


class EmptyTunnelEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.episode = self.root / "episode_000"
        self.episode.mkdir()
        cfg = {}
        configure_collection_scenario(cfg, normal.MODE)
        self.report = dict(
            schema="aimdrive_tunnel_baseline_v1", scenario=normal.MODE,
            status="passed", failure_reason="", baseline_drive_passed=True,
            started_before_entry=True, mapped_entry_observed=True, mapped_exit_observed=True,
            inside_observations=100, inventory_checks=25, last_blocking_actors=[],
            inventory_scope="all clients, vehicle/pedestrian/obstacle",
            ego_stopped_and_parked=False, stop_policy=normal.STOP_POLICY, final_stop_verified=True,
            test_route=dict(length_m=190.0, mapped_entry_route_s_m=40.0,
                            mapped_exit_route_s_m=160.0, approach_length_m=40.0,
                            after_exit_length_m=30.0),
            mapped_tunnel=dict(ego_link="A2256W000126", opposing_link="A2256W000131"),
        )
        observed = dict(control_mode=3, gear_mode=4, brake=1., throttle=0., brake_hold_verified=True,
                        speed_norm_raw=.001, position_xyz=[1., 2., 3.],
                        global_velocity_raw=[.001, 0., 0.], angular_velocity_raw=[0., 0., 0.])
        self.report["final_parking"] = dict(
            stop_policy=normal.STOP_POLICY, status="verified", velocity_reset_used=False,
            physical_stop_verified=True, owned_expert_stopped=True, controller_publishers_absent=True,
            publisher_released_after_verified=True, post_release_verified=True, final_verified=True,
            external_input_disabled=False, gear_p_verified=False,
            actual_stop=copy.deepcopy(observed), actual_final=copy.deepcopy(observed))
        self.context = SimpleNamespace(cfg=cfg, output=self.episode, run_seed=123,
                                       tunnel_baseline_report=self.report)
        self.outcome = dict(status="mission_completed", full_mission_completed=True)
        self.write("episode.json", dict(scenario_name=normal.MODE, seed=123,
                                        scenario_config=copy.deepcopy(cfg)))
        self.write("outcome.json", self.outcome)
        (self.episode / "raw").mkdir()
        self.write("raw/RAW_COMPLETE.json", dict(status="completed", training_visible=False))

    def write(self, relative, document):
        (self.episode / relative).write_text(json.dumps(document), encoding="utf-8")

    def finalize(self):
        normal.finalize_normal_collection(self.context, self.outcome)

    def validate(self):
        return normal.validate_normal_collection(self.episode, normal.MODE)

    def rehash(self, role):
        proof = json.loads((self.episode / normal.PROOF_RELATIVE).read_text())
        entry = proof["files"][role]
        entry["sha256"] = hashlib.sha256((self.episode / entry["path"]).read_bytes()).hexdigest()
        self.write(normal.PROOF_RELATIVE, proof)

    def test_missing_final_evidence_is_not_admitted_from_completed_outcome(self):
        self.assertFalse(self.validate()["valid"])

    def test_empty_object_target_remains_supervised_without_positive_or_light_coverage(self):
        target, valid, reason = detection_for_entry(dict(
            object_valid=True, objects=[], ego_pose=dict(x=0.0, y=0.0, yaw=0.0)))
        self.assertTrue(valid, reason)
        self.assertEqual(target.shape, (2500, 8))
        self.assertEqual(float(target.sum()), 0.0)
        required = required_training_coverage(normal.MODE)
        self.assertEqual(required, ("waypoint_frames",))
        counts = dict(waypoint_frames=20, object_frames=20, object_positive_frames=0,
                      traffic_light_0=0, traffic_light_1=0)
        self.assertTrue(all(counts[key] > 0 for key in required))
        self.assertEqual(coverage_failed_splits(dict(
            training_ready=False, required_coverage=required,
            coverage=dict(train=counts, val=counts))), [])

    def test_finalized_parked_drive_and_snapshots_are_admitted_without_executing_them(self):
        self.finalize()
        with patch.object(normal, "PROJECT", self.root / "unavailable_source_checkout"):
            result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["historical_source_evidence_only"])
        self.assertFalse(result["manual_all_frame_camera_certification"])
        self.assertFalse(result["physical_semantic_validation_completed"])
        self.assertEqual(self.outcome["tunnel_mission"]["baseline"], self.report)

    def test_historical_normal_source_set_remains_capture_owned(self):
        self.finalize()
        metadata = json.loads((self.episode / "episode.json").read_text())
        proof = json.loads((self.episode / normal.PROOF_RELATIVE).read_text())
        legacy_contract = metadata["scenario_config"]["_tunnel_normal_collection"]
        legacy_contract["schema"] = normal.LEGACY_CONTRACT_SCHEMA
        legacy_contract.pop("final_stop_policy")
        legacy = copy.deepcopy(self.report)
        for field in ("stop_policy", "final_stop_verified", "final_parking"):
            legacy.pop(field)
        legacy["ego_stopped_and_parked"] = True
        outcome = json.loads((self.episode / "outcome.json").read_text())
        outcome["tunnel_mission"]["baseline"] = legacy
        self.write("tunnel_normal/baseline.json", legacy)
        self.write("outcome.json", outcome)
        for role in ("baseline", "outcome"):
            entry = proof["files"][role]
            entry["sha256"] = hashlib.sha256((self.episode / entry["path"]).read_bytes()).hexdigest()
        for relative in set(normal.SOURCES) - set(normal.LEGACY_SOURCES):
            metadata["scenario_config"]["_tunnel_normal_collection"]["source_sha256"].pop(relative)
            proof["source_sha256"].pop(relative)
            proof["files"].pop("source:" + relative)
        self.write("episode.json", metadata)
        proof["files"]["episode"]["sha256"] = hashlib.sha256((self.episode / "episode.json").read_bytes()).hexdigest()
        self.write(normal.PROOF_RELATIVE, proof)
        with patch.object(normal, "PROJECT", self.root / "unavailable_checkout"):
            self.assertTrue(self.validate()["valid"])

    def test_final_proof_waits_for_final_parking_and_actual_traversal(self):
        for field in ("final_stop_verified", "mapped_entry_observed", "mapped_exit_observed"):
            with self.subTest(field=field):
                self.report[field] = False
                self.finalize()
                self.assertFalse(self.validate()["valid"])
                shutil.rmtree(self.episode / "tunnel_normal")
                self.report[field] = True

    def test_brake_stop_requires_actual_post_release_motion_and_actuator_evidence(self):
        for field, value in (("gear_mode", 1), ("brake", .5), ("throttle", .1),
                             ("speed_norm_raw", .5), ("global_velocity_raw", [.1, 0., 0.]),
                             ("angular_velocity_raw", [0., 0., .1]), ("position_xyz", [2., 2., 3.])):
            with self.subTest(field=field):
                damaged = copy.deepcopy(self.report)
                damaged["final_parking"]["actual_final"][field] = value
                self.assertFalse(normal._baseline_checks(damaged, self.context.cfg["_tunnel_normal_collection"])["final_stop"])

    def test_current_contract_cannot_fall_back_to_the_historical_parked_boolean(self):
        damaged = copy.deepcopy(self.report)
        damaged.pop("stop_policy")
        damaged["ego_stopped_and_parked"] = True
        self.assertFalse(normal._baseline_checks(damaged, self.context.cfg["_tunnel_normal_collection"])["final_stop"])
        damaged["stop_policy"] = "unknown_policy"
        self.assertFalse(normal._baseline_checks(damaged, self.context.cfg["_tunnel_normal_collection"])["final_stop"])

    def test_current_contract_requires_both_brake_implementation_and_control_source_snapshots(self):
        self.finalize()
        metadata = json.loads((self.episode / "episode.json").read_text())
        proof = json.loads((self.episode / normal.PROOF_RELATIVE).read_text())
        for relative in set(normal.SOURCES) - set(normal.LEGACY_SOURCES):
            metadata["scenario_config"]["_tunnel_normal_collection"]["source_sha256"].pop(relative)
            proof["source_sha256"].pop(relative)
            proof["files"].pop("source:" + relative)
        self.write("episode.json", metadata)
        proof["files"]["episode"]["sha256"] = hashlib.sha256((self.episode / "episode.json").read_bytes()).hexdigest()
        self.write(normal.PROOF_RELATIVE, proof)
        self.assertFalse(self.validate()["valid"])

    def test_hash_or_semantic_tampering_does_not_pass(self):
        self.finalize()
        modified = copy.deepcopy(self.report)
        modified["last_blocking_actors"] = [{"actor_id": "foreign_npc"}]
        self.write("tunnel_normal/baseline.json", modified)
        self.assertFalse(self.validate()["valid"])
        self.rehash("baseline")
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["empty_road_inventory"])

    def test_capture_source_snapshot_change_is_rejected(self):
        self.finalize()
        proof = json.loads((self.episode / normal.PROOF_RELATIVE).read_text())
        role = "source:" + normal.SOURCES[0]
        self.write(proof["files"][role]["path"], {"changed": True})
        self.rehash(role)
        self.assertFalse(self.validate()["valid"])

    def test_staging_rename_preserves_relative_evidence_and_original_hashes(self):
        self.finalize()
        before = (self.episode / normal.PROOF_RELATIVE).read_bytes()
        transferred = self.root / "final/tunnel_normal/campaign/lap/episode_000"
        transferred.parent.mkdir(parents=True)
        self.episode.rename(transferred)
        self.episode = transferred
        self.assertTrue(self.validate()["valid"])
        self.assertEqual((self.episode / normal.PROOF_RELATIVE).read_bytes(), before)

    def test_source_change_during_recording_is_rejected_and_raw_preserved(self):
        self.context.cfg["_tunnel_normal_collection"]["source_sha256"][normal.SOURCES[0]] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed during capture"):
            self.finalize()
        self.assertFalse((self.episode / normal.PROOF_RELATIVE).exists())
        self.assertTrue((self.episode / "raw/RAW_COMPLETE.json").exists())

    def test_external_or_symlink_evidence_path_is_rejected(self):
        self.finalize()
        proof = json.loads((self.episode / normal.PROOF_RELATIVE).read_text())
        baseline = self.episode / proof["files"]["baseline"]["path"]
        external = self.root / "external.json"
        baseline.rename(external)
        baseline.symlink_to(external)
        self.assertFalse(self.validate()["valid"])

    def test_old_nontunnel_capture_does_not_acquire_new_required_gate(self):
        self.assertEqual(normal.validate_normal_collection(self.episode, "full_loop_static"),
                         dict(required=False, valid=True))
        cfg = {"mission_mode": "full_loop_static"}
        before = copy.deepcopy(cfg)
        configure_collection_scenario(cfg, "full_loop_static")
        self.assertEqual(cfg, before)

    def test_initial_metadata_snapshot_does_not_claim_completed_baseline(self):
        self.context.tunnel_baseline_report = dict(self.report, status="running", baseline_drive_passed=False)
        evidence = tunnel_collection_evidence(self.context)
        self.assertFalse(evidence["completed"])


class EmptyTunnelShutdownResultTests(unittest.TestCase):
    """A reached route goal must not hide a failed final parking check."""

    def setUp(self):
        source = INTERNAL / "tools/run_proven_three_lap_scenario.py"
        tree = ast.parse(source.read_text())
        wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "RawMissionScenario")

        class CaptureBase:
            def run(self):
                return self.outcome

        namespace = dict(RawScenario=CaptureBase, MissionScenario=object)
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(source), "exec"), namespace)
        self.scenario = namespace["RawMissionScenario"]()
        self.scenario.cfg = dict(mission_mode=normal.MODE)
        self.scenario.outcome = dict(status="mission_completed", reason="runner_goal_reached",
                                     full_mission_completed=True, training_visible=False)
        self.scenario.event = Mock()

    def test_parking_failure_reaches_capture_supervisor_as_the_actual_failure(self):
        from scenario_runner.scenarios.segment_modes import tunnel_baseline

        with patch.object(tunnel_baseline, "park", side_effect=RuntimeError("P gear not observed")), \
                patch.object(tunnel_baseline, "finish") as finish:
            result = self.scenario.run()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "tunnel_normal_parking_failed: P gear not observed")
        self.assertTrue(result["full_mission_completed"])
        finish.assert_called_once_with(self.scenario, "failed", "P gear not observed")
        self.scenario.event.assert_called_once_with("tunnel_normal_parking_failed",
                                                   error="P gear not observed")

    def test_verified_parking_preserves_the_drive_result(self):
        from scenario_runner.scenarios.segment_modes import tunnel_baseline

        with patch.object(tunnel_baseline, "park") as park:
            result = self.scenario.run()
        self.assertEqual(result["status"], "mission_completed")
        park.assert_called_once_with(self.scenario)
        self.scenario.event.assert_called_once_with("tunnel_normal_parking_verified")

    def test_other_missions_keep_their_existing_shutdown_path(self):
        from scenario_runner.scenarios.segment_modes import tunnel_baseline

        self.scenario.cfg["mission_mode"] = "full_loop_tunnel_dynamic"
        with patch.object(tunnel_baseline, "park") as park:
            result = self.scenario.run()
        self.assertEqual(result["status"], "mission_completed")
        park.assert_not_called()
        self.scenario.event.assert_not_called()


if __name__ == "__main__":
    unittest.main()
