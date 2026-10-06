"""Dependency-free checks for preprocessing orchestration and frame/asset joins."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from MoraiDrive.data_collection.situation_labeling import label_episode as pipeline


class LabelEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def seal(self, folder):
        folder.mkdir(parents=True)
        pipeline.write_json(folder / "report.json", {"structural_valid": True})
        pipeline.write_json(folder / "COMPLETE.json", {"files": {
            "report.json": pipeline.digest(folder / "report.json")}})

    def test_preprocessing_order_and_v2_wiring(self):
        episode = self.root / "episode_000"
        stages = pipeline.preprocessing_stages(episode, self.root)
        folders = [folder.name for _, folder, _ in stages]
        self.assertLess(folders.index("gt_timeline_v1"), folders.index(pipeline.GEOMETRY))
        self.assertLess(folders.index(pipeline.GEOMETRY), folders.index(pipeline.REVIEW))
        self.assertEqual(folders[-1], pipeline.ADMISSION)
        self.assertIn(pipeline.GEOMETRY, stages[-1][2])
        self.assertIn(pipeline.REVIEW, stages[-1][2])

    def test_v5_evidence_set_is_reused_without_generating_v2(self):
        episode = self.root / "episode_000"
        evidence = pipeline.evidence_dirs("d4a53be1fe67")
        self.assertEqual(evidence["admission"], "admission_v5_d4a53be1fe67")
        self.assertEqual(evidence["review"], "review_closeout_v5_d4a53be1fe67")
        folders = [folder.name for _, folder, _ in
                   pipeline.preprocessing_stages(episode, self.root, evidence=evidence)]
        for generated in (pipeline.GEOMETRY, pipeline.REVIEW, pipeline.ADMISSION):
            self.assertNotIn(generated, folders)
        self.assertIn("gt_timeline_v1", folders)
        with self.assertRaises(ValueError):
            pipeline.evidence_dirs("../v5")

    def test_completed_stages_reused_and_missing_stage_executed(self):
        existing, missing = self.root / "existing", self.root / "missing"
        self.seal(existing)
        calls = []
        def execute(command):
            calls.append(command)
            self.seal(missing)
        rows = pipeline.run_stages([
            ("first.py", existing, []), ("second.py", missing, ["--output", str(missing)])
        ], execute)
        self.assertEqual([row["status"] for row in rows], ["reused", "created"])
        self.assertEqual(len(calls), 1)

    def test_changed_stage_stops_before_downstream(self):
        folder = self.root / "stage"
        self.seal(folder)
        (folder / "report.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Changed"):
            pipeline.run_stages([("tool.py", folder, [])], lambda _: self.fail("must not run"))

    def test_incomplete_stage_is_not_overwritten(self):
        folder = self.root / "partial"
        folder.mkdir()
        with self.assertRaises(FileNotFoundError):
            pipeline.run_stages([("tool.py", folder, [])], lambda _: self.fail("must not run"))

    def test_failed_stage_stops_before_next(self):
        calls = []
        def execute(command):
            calls.append(command)
            raise RuntimeError("failed audit")
        with self.assertRaisesRegex(RuntimeError, "failed audit"):
            pipeline.run_stages([("first.py", self.root / "a", []),
                                 ("second.py", self.root / "b", [])], execute)
        self.assertEqual(len(calls), 1)

    def fixture(self):
        episode, native, features = (self.root / name for name in ("episode", "native", "features"))
        (episode / pipeline.ADMISSION).mkdir(parents=True)
        native.mkdir()
        features.mkdir()
        (native / "frame.png").write_bytes(b"original sensor bytes")
        frames = [{"index": i, "lidar_source_ns": 100 + i, "zone_id": "static_obstacle",
                   "valid": True} for i in range(2)]
        rules = [{"index": 0, "instruction": "example instruction", "status": "matched"},
                 {"index": 1, "instruction": None, "status": "unknown"}]
        for name, rows in (("frame_features.jsonl", frames), ("situation_rules.jsonl", rules)):
            (features / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        admissions = [{"index": i, "lidar_source_time_ns": 100 + i,
                       "assets": {"camera_1": {"relative_path": "frame.png"}}} for i in range(2)]
        pipeline.write_json(episode / pipeline.ADMISSION / "frames.json", admissions)
        return episode, native, features, admissions

    def test_export_keeps_unlabeled_frames_and_original_sensor_references(self):
        episode, native, features, _ = self.fixture()
        destination = self.root / "labeled.jsonl"
        counts = pipeline.export_labeled_frames(episode, native, features, destination)
        rows = pipeline.read_jsonl(destination)
        self.assertEqual(counts, {"labeled": 1, "unlabeled": 1})
        self.assertEqual(rows[1]["frame_id"], 1)
        self.assertIsNone(rows[1]["instruction"])
        self.assertEqual(rows[0]["assets"]["camera_1"]["path"], str(native / "frame.png"))
        self.assertEqual([row["zone_instruction"] for row in rows], [1, 1])
        self.assertEqual((native / "frame.png").read_bytes(), b"original sensor bytes")

    def test_export_joins_the_selected_admission(self):
        episode, native, features, admissions = self.fixture()
        v5 = pipeline.evidence_dirs("abc123")["admission"]
        (episode / v5).mkdir()
        pipeline.write_json(episode / v5 / "frames.json", admissions)
        (episode / pipeline.ADMISSION / "frames.json").unlink()
        counts = pipeline.export_labeled_frames(episode, native, features,
                                                self.root / "v5.jsonl", v5)
        self.assertEqual(counts, {"labeled": 1, "unlabeled": 1})

    def test_timestamp_mismatch_rejected(self):
        episode, native, features, admissions = self.fixture()
        admissions[0]["lidar_source_time_ns"] = 999
        pipeline.write_json(episode / pipeline.ADMISSION / "frames.json", admissions)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            pipeline.export_labeled_frames(episode, native, features, self.root / "labels.jsonl")

    def test_duplicate_indices_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pipeline.indexed([{"index": 0}, {"index": 0}], "fixture")

    def test_asset_outside_native_root_rejected(self):
        episode, native, features, admissions = self.fixture()
        admissions[0]["assets"]["camera_1"]["relative_path"] = "../outside.png"
        pipeline.write_json(episode / pipeline.ADMISSION / "frames.json", admissions)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            pipeline.export_labeled_frames(episode, native, features, self.root / "labels.jsonl")

    def test_reused_admission_must_reference_current_inputs(self):
        episode, _, _, _ = self.fixture()
        self.seal(episode / "native_audit")
        pipeline.write_json(episode / pipeline.ADMISSION / "report.json",
                            {"input_seals": {"native_audit": "old digest"}})
        with self.assertRaisesRegex(ValueError, "different input evidence"):
            pipeline.verify_admission_inputs(episode, self.root)

    def test_single_command_exports_and_seals_after_labeling(self):
        episode, native, features, _ = self.fixture()
        output = self.root / "result"
        pipeline.write_json(features / "report.json", {})
        pipeline.write_json(self.root / "sensor_contract.json", {})
        (self.root / "supervisor.log").write_text("static log", encoding="utf-8")
        modules = {
            "python_environment": SimpleNamespace(
                build_analysis_environment=lambda _: {"PYTHONPATH": ""},
                probe_analysis_environment=lambda *args: None),
            "async_capture.route_sources": SimpleNamespace(resolve_route_sources=lambda *args: {
                "route_definition": "route.yaml", "global_path": "route.txt",
                "route_key": "full_loop", "provenance": {}}),
            "async_capture.native_session": SimpleNamespace(recorded_native_root=lambda *args: native),
            "ssd_storage": SimpleNamespace(SSD_ROOT=self.root),
        }
        calls = []
        def execute(command, **kwargs):
            calls.append(command)
            if "-m" in command:
                self.assertIn("--static-supervisor-log", command)
                self.assertIn(pipeline.ADMISSION, command)
                features.rename(output / "features")
        with patch.dict("sys.modules", modules), \
                patch.object(pipeline, "resolve_episode", return_value=(episode, self.root)), \
                patch.object(pipeline, "run_stages", return_value=[]), \
                patch.object(pipeline, "verify_admission_inputs"), \
                patch.object(pipeline.subprocess, "run", side_effect=execute):
            result = pipeline.main(["--episode", str(episode), "--output", str(output)])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["frame_counts"], {"labeled": 1, "unlabeled": 1})
        pipeline.verify_stage(output)
        self.assertEqual(len(calls), 2)  # dependency probe, then build_labels


if __name__ == "__main__":
    unittest.main()
