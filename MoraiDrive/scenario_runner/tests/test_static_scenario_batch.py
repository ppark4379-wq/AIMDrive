import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scenario_runner" / "tools" / "generate_static_scenario_suite.py"
SPEC = importlib.util.spec_from_file_location("static_scenario_batch", MODULE_PATH)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)

REVIEW_PATH = ROOT / "scenario_runner" / "tools" / "review_static_scenario_candidate.py"
REVIEW_SPEC = importlib.util.spec_from_file_location("static_scenario_review", REVIEW_PATH)
review = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(review)


class StaticScenarioBatchTest(unittest.TestCase):
    def policy(self):
        return {
            "models": ["A", "B", "C", "D"],
            "scale_overrides": {"C": [1.0, 1.0, 1.0]},
            "fallback_scale": [2.0, 3.0, 2.0],
            "anchor_start": {
                "route_path_index": 100,
                "position": {"x": 0.0, "y": 0.0, "z": 2.0},
                "yaw_deg": 10.0,
            },
            "anchor_end": {
                "route_path_index": 120,
                "position": {"x": 0.0, "y": 10.0, "z": 4.0},
                "yaw_deg": 10.0,
            },
            "map_name": "MAP",
        }

    def test_hundred_requested_layouts_are_unique_and_bounded(self):
        rng = random.Random(20260920)
        signatures = set()
        counts = set()
        selected_models = set()
        for _ in range(100):
            count = rng.randint(1, 3)
            obstacles = batch.generate_layout(
                self.policy(), rng, count, 0.6, 12.0, 0.1
            )
            signature = batch.layout_signature(obstacles)
            self.assertNotIn(signature, signatures)
            signatures.add(signature)
            counts.add(count)
            selected_models.update(item["model"] for item in obstacles)
            self.assertEqual(len({item["model"] for item in obstacles}), count)
            for item in obstacles:
                self.assertGreaterEqual(item["position"]["y"], 0.0)
                self.assertLessEqual(item["position"]["y"], 10.0)
                self.assertGreaterEqual(item["position"]["x"], -0.6)
                self.assertLessEqual(item["position"]["x"], 0.6)
                self.assertGreaterEqual(item["yaw_deg"], -2.0)
                self.assertLessEqual(item["yaw_deg"], 22.0)
                self.assertTrue(all(value > 0 for value in item["scale"]))
        self.assertEqual(counts, {1, 2, 3})
        self.assertEqual(selected_models, {"A", "B", "C", "D"})

    def test_candidate_is_not_marked_verified(self):
        obstacles = batch.generate_layout(
            self.policy(), random.Random(1), 2, 0.5, 5.0, 0.0
        )
        document = batch.build_candidate_document(
            "trial", 1, 7, obstacles, self.policy()
        )
        self.assertEqual(document["schema"], "morai_static_obstacle_preset_v1")
        self.assertTrue(
            document["generation"]["candidate_only_until_expert_passes"]
        )
        self.assertNotIn("verification", document)

    def test_repository_paths_are_stored_portably(self):
        repository_file = batch.ROOT / "scenario_runner" / "README.md"
        self.assertEqual(
            batch.portable_path(repository_file),
            "scenario_runner/README.md",
        )
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "external.yaml"
            self.assertTrue(Path(batch.portable_path(external)).is_absolute())

    def test_atomic_yaml_is_replayable_by_existing_schema(self):
        obstacles = batch.generate_layout(
            self.policy(), random.Random(2), 3, 0.5, 5.0, 0.1
        )
        document = batch.build_candidate_document(
            "trial", 2, 1, obstacles, self.policy()
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trial.yaml"
            batch.write_yaml_atomic(output, document)
            loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["obstacles"], obstacles)
        self.assertEqual(loaded["map_name"], "MAP")

    def test_scenario_only_bridge_mode_skips_capture_sensor_gate(self):
        source = (
            ROOT / "data_collection" / "tools" /
            "01_internal_sensor_record_preflight.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--scenario-only", source)
        self.assertIn("rosparam set /use_sim_time false", source)
        self.assertIn("시나리오 검증용 bridge를 계속 유지", source)
        self.assertIn('if [[ "$RUN_MODE" == "capture" ]]', source)

        launcher = (
            ROOT / "scenario_runner" / "tools" /
            "run_static_scenario_batch.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("01_internal_sensor_record_preflight.sh --scenario-only", launcher)

    def test_closed_loop_evidence_requires_goal_plan_and_tracks_max_cte(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[PLAN] candidate dir=L valid=True cost=1\n"
                "[KCityFullLoop] lap=1 cte=0.42 remain=10\n"
                "[KCityFullLoop] lap=1 cte=2.31 remain=2\n"
                "[KCityFullLoop] GOAL REACHED lap=1\n",
                encoding="utf-8",
            )
            evidence = batch.read_closed_loop_evidence(log)
        self.assertTrue(evidence["goal_reached"])
        self.assertEqual(evidence["valid_plan_samples"], 1)
        self.assertEqual(evidence["cross_track_samples"], 2)
        self.assertAlmostEqual(evidence["max_cross_track_error_m"], 2.31)

    def test_live_planner_health_detects_latest_tracking_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'driving', 'error': None}\n"
                "[PLAN] candidate dir=R valid=True cost=1\n"
                "[STATIC] {'phase': 'blocked', "
                "'error': 'tracking_error'}\n"
                "[PLAN] candidate dir=? valid=False cost=0\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertTrue(health["persistent_invalid_plan"])
        self.assertIn("tracking_error", health["static_status"])

    def test_live_planner_health_detects_recorded_envelope_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'blocked', "
                "'error': 'recorded_envelope'}\n"
                "[PLAN] candidate dir=? valid=False cost=0\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertTrue(health["persistent_invalid_plan"])

    def test_live_planner_health_detects_persistent_signal_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'follow', 'path_validated': True, "
                "'speed_source': 'signal', "
                "'final_target_velocity': 0.0}\n"
                "[PLAN] candidate dir=? valid=True cost=0\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertTrue(health["persistent_signal_stop"])
        self.assertFalse(health["persistent_invalid_plan"])

    def test_live_planner_health_clears_after_valid_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trial.log"
            log.write_text(
                "[STATIC] {'phase': 'blocked', "
                "'error': 'tracking_error'}\n"
                "[PLAN] candidate dir=? valid=False cost=0\n"
                "[STATIC] {'phase': 'driving', 'error': None}\n"
                "[PLAN] candidate dir=L valid=True cost=1\n",
                encoding="utf-8",
            )
            health = batch.read_latest_planner_health(log)
        self.assertFalse(health["persistent_invalid_plan"])
        self.assertFalse(health["persistent_signal_stop"])

    def test_manual_review_can_promote_only_cross_track_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            candidate = suite / "runtime" / "candidate.yaml"
            candidate.parent.mkdir()
            candidate.write_text(yaml.safe_dump({
                "schema": "morai_static_obstacle_preset_v1",
                "name": "candidate",
                "generation": {"candidate_only_until_expert_passes": True},
                "verification": {"expert_completed": True},
                "obstacles": [],
            }), encoding="utf-8")
            manifest_path = suite / "suite_manifest.json"
            manifest_path.write_text(__import__("json").dumps({
                "attempts": [{
                    "attempt": 1,
                    "accepted": False,
                    "candidate": str(candidate),
                    "return_code": 0,
                    "timed_out": False,
                    "rejection_reasons": ["cross_track_limit_exceeded"],
                    "closed_loop_evidence": {
                        "goal_reached": True,
                        "valid_plan_samples": 4,
                        "max_cross_track_error_m": 4.2,
                    },
                }],
                "accepted_count": 0,
                "rejected_count": 1,
                "accepted_presets": [],
            }), encoding="utf-8")
            output = review.promote_candidate(
                manifest_path, 1, "tester", "MORAI에서 실선 침범 없음 확인"
            )
            promoted = yaml.safe_load(output.read_text(encoding="utf-8"))
            manifest = __import__("json").loads(
                manifest_path.read_text(encoding="utf-8")
            )
        self.assertEqual(
            promoted["source"], "manually_reviewed_static_expert_verified"
        )
        self.assertFalse(promoted["verification"]["automatic_gate_passed"])
        self.assertEqual(manifest["manual_accepted_count"], 1)
        self.assertEqual(manifest["rejected_count"], 0)

    def test_manual_review_cannot_override_collision_or_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            candidate = suite / "runtime" / "candidate.yaml"
            candidate.parent.mkdir()
            candidate.write_text("name: candidate\n", encoding="utf-8")
            manifest_path = suite / "suite_manifest.json"
            manifest_path.write_text(__import__("json").dumps({
                "attempts": [{
                    "attempt": 1,
                    "accepted": False,
                    "candidate": str(candidate),
                    "return_code": 1,
                    "timed_out": False,
                    "rejection_reasons": ["runner_return_code"],
                    "closed_loop_evidence": {
                        "goal_reached": False,
                        "valid_plan_samples": 0,
                    },
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot manually override"):
                review.promote_candidate(
                    manifest_path, 1, "tester", "보기에 괜찮음"
                )


if __name__ == "__main__":
    unittest.main()
