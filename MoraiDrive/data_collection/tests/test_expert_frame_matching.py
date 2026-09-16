import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/04_evaluate_expert_driving.py"
SPEC = importlib.util.spec_from_file_location("evaluation_with_frames", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
from expert_frame_matching import load_frame_sources, match_collision


class ExpertFrameMatchingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.episode = self.base / "run/episode_000"
        self.associations = [
            {"index": 1415 + n, "gps": {"receive_monotonic_ns": 1000000000 + n * 200000000,
                                        "source_timestamp_ns": 172241000000 + n * 100000000}}
            for n in range(3)]
        self.frames = [
            {"index": 1415 + n, "ego_xy": [n, 0], "lidar_source_ns": 172241000000 + n * 100000000}
            for n in range(3)]
        self.collision = {"receive_monotonic_ns": 1269670000, "ego_xy": [1.19, 0]}
        self.write_sources()

    def write_json(self, relative, value):
        path = self.episode / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_sources(self):
        # Deliberately shuffled and sparse: array offsets are not bundle indices.
        self.write_json("native_associations/associations.json", list(reversed(self.associations)))
        self.write_json("gt_timeline_v1/frames.json", list(reversed(self.frames)))

    def match(self):
        return match_collision(self.collision, load_frame_sources(self.episode))

    def test_agreement_indices_units_and_projection(self):
        result = self.match()
        self.assertEqual(result["match_confidence"], "HIGH")
        self.assertEqual(result["match_status"], "CANDIDATES_AGREE")
        self.assertEqual(result["sensor_bundle_index"], 1416)
        self.assertEqual(result["sensor_bundle_number"], 1417)
        self.assertEqual(result["sensor_bundle_index_base"], 0)
        self.assertEqual(result["receive_time_candidate_index"], 1416)
        self.assertEqual(result["position_candidate_index"], 1416)
        self.assertEqual(result["nearest_sensor_frame_number"], 1417)
        self.assertAlmostEqual(result["receive_time_gap_ms"], 69.67)
        self.assertAlmostEqual(result["position_error_m"], .19)
        self.assertEqual(result["collision_between_frames"], [1416, 1417])
        self.assertAlmostEqual(result["estimated_fraction_after_frame"], .19)
        self.assertEqual(result["estimated_collision_source_time_ns"], 172360000000)
        self.assertEqual(result["estimated_collision_source_time_seconds"], 172.36)
        self.assertFalse(result["physical_collision_time_verified"])

    def test_disagreement_is_low_and_has_no_interpolation(self):
        self.collision["ego_xy"] = [.1, 0]
        result = self.match()
        self.assertEqual(result["match_confidence"], "LOW")
        self.assertEqual(result["position_candidate_index"], 1415)
        self.assertAlmostEqual(result["position_error_m"], .9)
        self.assertAlmostEqual(result["nearest_position_error_m"], .1)
        self.assertIsNone(result["collision_between_frames"])

    def test_adjacent_candidates_report_possible_interval_without_selecting_bundle(self):
        for x, interval in ((.1, [1415, 1416]), (1.9, [1416, 1417])):
            with self.subTest(x=x):
                self.collision["ego_xy"] = [x, 0]
                result = self.match()
                self.assertEqual(result["match_confidence"], "LOW")
                self.assertEqual(result["match_status"], "BETWEEN_FRAMES")
                self.assertEqual(result["candidate_index_distance"], 1)
                self.assertEqual(result["possible_collision_between_bundle_indices"], interval)
                self.assertEqual(result["candidate_source_time_gap_ms"], 100)
                self.assertIsNone(result["sensor_bundle_index"])
                self.assertIsNone(result["sensor_bundle_number"])
                self.assertIsNone(result["estimated_collision_source_time_ns"])
                self.assertIsNone(result["collision_between_frames"])

    def test_large_candidate_gap_is_distinct_from_adjacent_interval(self):
        self.collision.update(receive_monotonic_ns=1010000000, ego_xy=[1.9, 0])
        result = self.match()
        self.assertEqual(result["match_status"], "CANDIDATE_MISMATCH")
        self.assertEqual(result["match_confidence"], "LOW")
        self.assertEqual(result["candidate_index_distance"], 2)
        self.assertIsNone(result["possible_collision_between_bundle_indices"])
        self.assertIsNone(result["sensor_bundle_index"])

    def test_adjacent_candidates_with_bad_source_time_are_not_between_frames(self):
        self.collision["ego_xy"] = [1.9, 0]
        self.frames[2]["lidar_source_ns"] += 1
        self.write_sources()
        result = self.match()
        self.assertEqual(result["match_status"], "INCONSISTENT_EVIDENCE")
        self.assertIsNone(result["possible_collision_between_bundle_indices"])

    def test_adjacent_candidates_outside_receipt_range_are_not_between_frames(self):
        self.collision.update(receive_monotonic_ns=900000000, ego_xy=[1.1, 0])
        result = self.match()
        self.assertEqual(result["match_status"], "INCONSISTENT_EVIDENCE")
        self.assertEqual(result["candidate_index_distance"], 1)
        self.assertIsNone(result["possible_collision_between_bundle_indices"])

    def test_interval_duration_comes_from_source_time_not_assumed_rate(self):
        self.collision["ego_xy"] = [1.9, 0]
        self.frames[2]["lidar_source_ns"] += 50000000
        self.associations[2]["gps"]["source_timestamp_ns"] += 50000000
        self.write_sources()
        result = self.match()
        self.assertEqual(result["match_status"], "BETWEEN_FRAMES")
        self.assertEqual(result["candidate_source_time_gap_ms"], 150)

    def test_missing_gt_keeps_time_candidate_without_high_confidence(self):
        (self.episode / "gt_timeline_v1/frames.json").unlink()
        result = self.match()
        self.assertEqual(result["match_confidence"], "UNAVAILABLE")
        self.assertEqual(result["nearest_sensor_bundle_index"], 1416)
        self.assertTrue(result["match_reasons"])

    def test_tied_positions_and_tied_receipts_are_ambiguous(self):
        self.collision["ego_xy"] = [1.5, 0]
        self.assertEqual(self.match()["match_confidence"], "AMBIGUOUS")
        self.collision["ego_xy"] = [1.19, 0]
        self.collision["receive_monotonic_ns"] = 1300000000
        result = self.match()
        self.assertEqual(result["match_confidence"], "AMBIGUOUS")
        self.assertEqual(result["receive_time_candidate_indices"], [1416, 1417])
        self.assertIsNone(result["nearest_sensor_bundle_index"])

    def test_outside_receipt_range_is_not_high(self):
        self.collision.update(receive_monotonic_ns=2000000000, ego_xy=[2.1, 0])
        result = self.match()
        self.assertEqual(result["match_confidence"], "LOW")
        self.assertIsNone(result["estimated_fraction_after_frame"])

    def test_source_mismatch_blocks_high_confidence(self):
        self.frames[1]["lidar_source_ns"] += 1
        self.write_sources()
        self.assertEqual(self.match()["match_confidence"], "LOW")

    def test_sparse_indices_do_not_create_a_fake_adjacent_segment(self):
        self.frames[2]["index"] = 1418
        self.write_sources()
        result = self.match()
        self.assertEqual(result["match_confidence"], "HIGH")
        self.assertIsNone(result["collision_between_frames"])

    def test_interpolation_before_nearest_uses_previous_segment(self):
        self.collision["ego_xy"] = [.81, 0]
        result = self.match()
        self.assertEqual(result["collision_between_frames"], [1415, 1416])
        self.assertAlmostEqual(result["estimated_fraction_after_frame"], .81)

    def test_duplicate_indices_and_corrupt_sources_are_unavailable(self):
        self.associations.append(self.associations[0])
        self.write_sources()
        self.assertEqual(self.match()["match_confidence"], "UNAVAILABLE")
        (self.episode / "native_associations/associations.json").write_text("{", encoding="utf-8")
        self.assertEqual(self.match()["match_confidence"], "UNAVAILABLE")

    def test_missing_event_fields_or_null_gt_are_unavailable(self):
        for value in (None, [None, 0], [float("nan"), 0]):
            self.collision["ego_xy"] = value
            self.assertEqual(self.match()["match_confidence"], "UNAVAILABLE")
        self.collision["ego_xy"] = [1.19, 0]
        for frame in self.frames:
            frame["ego_xy"] = None
        self.write_sources()
        self.assertEqual(self.match()["match_confidence"], "UNAVAILABLE")

    def test_stationary_positions_are_ambiguous(self):
        for frame in self.frames:
            frame["ego_xy"] = [1, 0]
        self.write_sources()
        result = self.match()
        self.assertEqual(result["match_confidence"], "AMBIGUOUS")
        self.assertIsNone(result["collision_between_frames"])

    def test_campaign_integration_preserves_inputs_and_unverified_sensor_frame(self):
        campaign = self.base / "campaigns/test"
        campaign.mkdir(parents=True)
        (campaign / "runs.tsv").write_text("lap\trun_name\tstatus\n1\trun\tcaptured\n", encoding="utf-8")
        self.write_json("outcome.json", {"full_mission_completed": True})
        event = {"event": "collision", "receive_monotonic_ns": self.collision["receive_monotonic_ns"],
                 "ego": {"x": 1.19, "y": 0, "current_link": "LINK", "collision_objects": ["13"]}}
        (self.episode / "events.jsonl").write_text(json.dumps(event), encoding="utf-8")
        before = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        report = evaluation.evaluate_campaign(self.base, "test")
        collision = report["laps"][0]["collisions"][0]
        self.assertEqual(collision["current_link"], "LINK")
        self.assertEqual(collision["nearest_sensor_bundle_index"], 1416)
        self.assertEqual(collision["match_confidence"], "HIGH")
        self.assertIsNone(collision["sensor_frame"])
        self.assertEqual(report["summary"]["collision_events"], 1)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
