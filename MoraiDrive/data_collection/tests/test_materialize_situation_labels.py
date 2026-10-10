"""Full-row masking, immutable pins, and interrupted-output recovery contracts."""
from collections import Counter
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from MoraiDrive.data_collection.tools import materialize_situation_labels as materializer


class MaterializeSituationLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.profile, self.dataset, self.output = [self.root / name for name in ("base", "profile", "dataset", "applied")]
        for path in (self.base, self.profile, self.dataset):
            path.mkdir()
        self.source_root = "/old/ssd/dataset"
        self.build_sources()

    def write(self, root, relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(materializer.json_bytes(value))
        return materializer.sha256(path)

    def seal(self, root, names, **metadata):
        return self.write(root, "COMPLETE.json", dict(files={name: materializer.sha256(root / name) for name in names}, **metadata))

    def build_sources(self):
        records, frames = [], []
        for episode_number in (1, 2):
            key = "run/lap_{:03d}".format(episode_number)
            native = "dynamic_obstacle/" + key
            label_relative = "source_labels/" + native + "/labeled_frames.jsonl"
            folder = (self.base / label_relative).parent
            folder.mkdir(parents=True)
            full_rows = []
            for frame_id in range(2):
                valid = not (episode_number == 2 and frame_id == 1)
                instruction = "Continue driving." if valid else None
                relative_asset = "CAMERA_1/frame_{}.png".format(frame_id)
                row = dict(episode_path=self.source_root + "/" + native + "/episode_000", frame_id=frame_id,
                           sensor_time_ns=episode_number * 100000 + frame_id * 100, instruction=instruction,
                           instruction_label_valid=valid, frame_valid=True, zone="dynamic_obstacle", zone_instruction=2,
                           frame_invalid_reasons=[], dataset_build_candidate=True, object_training_label_valid=True,
                           assets={"camera_1": dict(relative_path=relative_asset, path=self.source_root + "/" + native + "/" + relative_asset,
                                    sha256="a" * 64, size_bytes=20)}, waypoint=[1, 2],
                           labeling=dict(index=frame_id, action_valid=valid, observation_valid=True,
                                         instruction=instruction, evidence={"actors": [{"id": 42}]},
                                         observation="A pedestrian is present.", unknown_reasons=[]))
                full_rows.append(row)
            with (folder / "labeled_frames.jsonl").open("w") as stream:
                for row in full_rows:
                    stream.write(json.dumps(row) + "\n")
            self.write(folder, "report.json", dict(episode=full_rows[0]["episode_path"]))
            self.write(folder, "features/frame_features.jsonl", {"index": 0})
            feature_sha = self.seal(folder / "features", ["frame_features.jsonl"])
            label_sha = materializer.sha256(folder / "labeled_frames.jsonl")
            complete_sha = self.seal(folder, ["labeled_frames.jsonl", "report.json"], features_seal_sha256=feature_sha,
                                     labeling_code_sha256="b" * 64)
            record = dict(episode_key=key, category="dynamic_obstacle", labeled_relative=label_relative,
                          labels_sha256=label_sha, report_sha256=materializer.sha256(folder / "report.json"),
                          complete_sha256=complete_sha, features_relative=str(Path(label_relative).parent / "features"),
                          features_complete_sha256=feature_sha, labeling_code_sha256="b" * 64,
                          native_episode_relative=native + "/episode_000", native_root_relative=native,
                          source_record={"labels_sha256": label_sha})
            valid_count = sum(r["instruction_label_valid"] for r in full_rows)
            record["counts"] = dict(rows=2, target_valid=valid_count, handoff_target_valid=valid_count - int(episode_number == 1),
                                    review_excluded=int(episode_number == 1), observation_valid=2, frame_valid=2)
            records.append(record)
            for row in full_rows:
                frame = dict(schema="situation_ssd_review_candidate_v1_frame", episode_key=key, category="dynamic_obstacle",
                             labeling_code_sha256=record["labeling_code_sha256"], source_label_relative=label_relative,
                             original_row=materializer.compact_projection(row, record),
                             portable=dict(native_episode_relative=record["native_episode_relative"], native_root_relative=native,
                                           assets={"camera_1": dict(dataset_relative_path=native + "/" + row["assets"]["camera_1"]["relative_path"],
                                                    sha256="a" * 64, size_bytes=20)}),
                             handoff_target_valid=row["instruction_label_valid"] and not (episode_number == 1 and row["frame_id"] == 0),
                             review_exclusion_reasons=["ambiguous"] if episode_number == 1 and row["frame_id"] == 0 else [])
                frames.append(frame)
        self.frames = frames
        source_index_sha = self.write(self.base, "source_bundle/version_index.json", {"records": records})
        source_complete_sha = self.seal(self.base / "source_bundle", ["version_index.json"])
        review_sha = self.write(self.base, "review_exclusions.json", dict(source_index_sha256=source_index_sha,
                               exclusions=[dict(episode_key="run/lap_001", frame_id=0, reason="ambiguous", basis="review")]))
        self.write(self.base, "portable_index.json", dict(source_index_sha256=source_index_sha, source_complete_sha256=source_complete_sha,
                   source_dataset_root_historical=self.source_root, review_exclusions_sha256=review_sha,
                   selected_episode_count=2, review_exclusions_source_count=1, records=records))
        with gzip.open(self.base / "portable_frames.jsonl.gz", "wt") as stream:
            for frame in frames:
                stream.write(json.dumps(frame) + "\n")
        self.seal_base(source_index_sha, source_complete_sha)
        pins = {name: materializer.sha256(self.base / name) for name in ("COMPLETE.json", "portable_index.json", "portable_frames.jsonl.gz")}
        self.write(self.profile, "profile.json", dict(base_package_pins=pins, source_episodes=2, source_frames=4,
                   source_instruction_targets=2, additional_instruction_exclusions=1, expected_instruction_targets=1))
        row = frames[1]["original_row"]
        self.write(self.profile, "review_exclusions.json", dict(source_index_sha256=source_index_sha,
                   base_portable_index_sha256=pins["portable_index.json"], exclusions=[dict(episode_key="run/lap_001", frame_id=1,
                   source_signature_sha256=materializer.source_signature(row, "run/lap_001"),
                   source_label_file_sha256=records[0]["labels_sha256"], category="dynamic_obstacle", reasons=["visual conflict"])]))
        self.write(self.profile, "verification.json", {"instruction_targets": 1})
        self.write(self.profile, "tools/loader.py", {"fixture": "not executed"})
        self.seal_profile()

    def seal_base(self, source_index_sha=None, source_complete_sha=None):
        if source_index_sha is None:
            old = materializer.read_json(self.base / "COMPLETE.json")
            source_index_sha, source_complete_sha = old["source_index_sha256"], old["source_complete_sha256"]
        files = [p.relative_to(self.base).as_posix() for p in self.base.rglob("*") if p.is_file() and p != self.base / "COMPLETE.json"]
        self.seal(self.base, files, schema="situation_ssd_review_candidate_v1_complete", status="REVIEW_CANDIDATE_MECHANICAL_EXPORT",
                  source_index_sha256=source_index_sha, source_complete_sha256=source_complete_sha)

    def seal_profile(self):
        self.seal(self.profile, materializer.PROFILE_METADATA, schema="situation_visual_instruction_profile_complete_v1", status="COMPLETE")

    def call(self, **kwargs):
        return materializer.materialize(self.base, self.profile, self.dataset, self.output, **kwargs)

    def test_full_rows_apply_all_masks_without_filter_loader(self):
        report = self.call()
        self.assertEqual((report["rows"], report["instruction_targets"], report["review_excluded"], report["visual_excluded"]), (4, 1, 1, 1))
        index = materializer.read_json(self.output / "applied_index.json")
        rows = []
        for record in index["records"]:
            rows.extend(json.loads(line) for line in (self.output / record["labeled_relative"]).read_text().splitlines())
        for row in rows:
            valid = row["instruction_label_valid"]
            self.assertTrue(all(row[field] == valid for field in materializer.MASK_FIELDS))
            self.assertEqual(row["labeling"]["action_valid"], valid)
            self.assertEqual(row["labeling"]["instruction"], row["instruction"])
            if not valid:
                self.assertIsNone(row["instruction"])
            self.assertTrue(row["labeling"]["observation_valid"])
            self.assertEqual(row["labeling"]["evidence"], {"actors": [{"id": 42}]})
            self.assertEqual(row["waypoint"], [1, 2])
            self.assertTrue(row["object_training_label_valid"])
            self.assertTrue(row["assets"]["camera_1"]["path"].startswith(str(self.dataset)))
        self.assertEqual(rows[0]["source_instruction"], "Continue driving.")
        self.assertTrue(rows[0]["source_labeling"]["action_valid"])
        self.assertFalse(rows[0]["labeling"]["action_valid"])
        self.assertTrue((self.output / "source_provenance/profile.json").exists())
        materializer.verify_seal(self.output)
        self.assertEqual(list(self.dataset.iterdir()), [])

    def test_preflight_verifies_without_creating_output(self):
        report = self.call(preflight_only=True)
        self.assertEqual(report["status"], "PREFLIGHT_VERIFIED")
        self.assertFalse(self.output.exists())

    def test_wrapper_materializes_and_installs_the_same_filtered_cohort(self):
        """The shipped command must bridge the two package contracts on disk."""
        index = materializer.read_json(self.base / "portable_index.json")
        for record in index["records"]:
            episode = self.dataset / record["native_episode_relative"]
            (episode / "raw").mkdir(parents=True)
            self.write(episode, "episode.json", {"scenario_id": record["category"]})
            for frame_id in range(2):
                image = self.dataset / record["native_root_relative"] / "CAMERA_1" / ("frame_{}.png".format(frame_id))
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"x" * 20)
        command = Path(materializer.__file__).with_name("apply_latest_situation_labels.sh")
        environment = dict(os.environ, SITUATION_PYTHON=sys.executable,
                           SITUATION_DATASET_ROOT=str(self.dataset), SITUATION_LABEL_BASE=str(self.base),
                           SITUATION_VISUAL_PROFILE=str(self.profile), SITUATION_APPLIED_ROOT=str(self.output))
        subprocess.check_output(["bash", str(command), "--preflight-only"], env=environment)
        self.assertFalse(self.output.exists())
        for _ in range(2):
            subprocess.check_output(["bash", str(command)], env=environment)
        applied = materializer.read_json(self.output / "applied_index.json")
        total = 0
        for record in applied["records"]:
            installed = self.dataset / record["native_episode_relative"] / "situation_labels"
            self.assertEqual((installed / "labeled_frames.jsonl").read_bytes(),
                             (self.output / record["labeled_relative"]).read_bytes())
            materializer.verify_seal(installed)
            for row in map(json.loads, (installed / "labeled_frames.jsonl").read_text().splitlines()):
                self.assertEqual(row["instruction_label_valid"], row["labeling"]["action_valid"])
                total += int(row["instruction_label_valid"])
        self.assertEqual(total, 1)

    def test_resume_reuses_episodes_and_rejects_changed_outputs(self):
        first = self.call()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.call()
        self.assertEqual(self.call(resume=True), first)
        (self.output / "COMPLETE.json").unlink()
        self.assertEqual(self.call(resume=True), first)
        record = materializer.read_json(self.output / "applied_index.json")["records"][0]
        (self.output / record["labeled_relative"]).write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Sealed file changed"):
            self.call(resume=True)

    def test_changed_source_and_visual_signature_fail_closed(self):
        exclusions = materializer.read_json(self.profile / "review_exclusions.json")
        exclusions["exclusions"][0]["source_signature_sha256"] = "0" * 64
        self.write(self.profile, "review_exclusions.json", exclusions)
        self.seal_profile()
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            self.call(preflight_only=True)
        self.assertFalse(self.output.exists())
        (self.base / "portable_index.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "base pin mismatch"):
            self.call(preflight_only=True)

    def test_safe_paths_duplicate_exclusions_and_overlap(self):
        for value in ("../elsewhere", "/absolute", "a//b", "a/./b", "C:/data", "a\\b"):
            with self.assertRaises(ValueError):
                materializer.safe_relative(value)
        with self.assertRaisesRegex(ValueError, "Duplicate exclusion"):
            materializer.exclusions_by_identity({"exclusions": [dict(episode_key="run/lap", frame_id=1)] * 2})
        with self.assertRaisesRegex(ValueError, "overlaps"):
            materializer.materialize(self.base, self.profile, self.dataset, self.dataset / "output", preflight_only=True)


if __name__ == "__main__":
    unittest.main()
