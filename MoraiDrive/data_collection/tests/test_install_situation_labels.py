"""Atomic native metadata installation and interrupted-generation recovery."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from MoraiDrive.data_collection.tools import install_situation_labels as installer


class InstallSituationLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.labels = self.root / "labels"
        self.native = "dynamic_obstacle/run/lap_001/episode_000"
        self.episode = self.dataset / self.native
        self.source = self.labels / "episodes/dynamic_obstacle/run/lap_001"
        self.episode.mkdir(parents=True)
        self.source.mkdir(parents=True)
        for name in ("raw", "quality", "geometry"):
            (self.episode / name).mkdir()
            (self.episode / name / "untouched.bin").write_bytes(b"immutable evidence")
        self.write(self.episode / "episode.json", {"scenario_id": "dynamic_obstacle"})
        self.write(self.episode / "raw/raw_manifest.json", {"run": "lap_001"})
        native_root = str(Path(self.native).parent)
        identity = {"schema": installer.SCHEMA, "dataset_root": str(self.dataset)}
        self.rows = []
        for index in range(2):
            asset = self.dataset / native_root / "CAMERA_1" / (str(index) + ".png")
            asset.parent.mkdir(exist_ok=True)
            asset.write_bytes(b"a camera image")
            asset_hash = installer.sha256(asset)
            source_labeling = {"index": index, "instruction": "Continue.", "action_valid": True,
                               "observation_valid": True, "instruction_id": "CONTINUE", "label": "proceed",
                               "evidence": {"pedestrian": 42}, "instruction_state": "normal"}
            row = {"frame_id": index, "sensor_time_ns": (index + 1) * 100,
                   "episode_key": "run/lap_001", "category": "dynamic_obstacle", "episode_path": str(self.episode),
                   "instruction": "Continue." if index else None,
                   "labeling": dict(source_labeling, instruction="Continue." if index else None, action_valid=bool(index)),
                   "source_labeling": source_labeling, "source_instruction": "Continue.", "source_instruction_label_valid": True,
                   "source_situation_target_valid": True, "source_handoff_target_valid": True,
                   "zone": "dynamic_obstacle", "zone_instruction": 2, "frame_valid": True,
                   "object_training_label_valid": True,
                   "assets": {"camera_1": {"relative_path": "CAMERA_1/" + asset.name, "path": str(asset),
                                            "sha256": asset_hash, "size_bytes": asset.stat().st_size}},
                   "portable": {"native_episode_relative": self.native, "native_root_relative": native_root,
                                "assets": {"camera_1": {"dataset_relative_path": native_root + "/CAMERA_1/" + asset.name,
                                                         "sha256": asset_hash, "size_bytes": asset.stat().st_size}}}}
            row.update({mask: bool(index) for mask in installer.MASKS})
            self.rows.append(row)
        counts = {"rows": 2, "instruction_targets": 1, "observation_valid": 2, "frame_valid": 2}
        self.record = {"episode_key": "run/lap_001", "category": "dynamic_obstacle", "native_episode_relative": self.native,
                       "counts": counts, "source_labels_sha256": "b" * 64,
                       "labeled_relative": "episodes/dynamic_obstacle/run/lap_001/labeled_frames.jsonl",
                       "manifest_relative": "episodes/dynamic_obstacle/run/lap_001/manifest.json",
                       "complete_relative": "episodes/dynamic_obstacle/run/lap_001/COMPLETE.json"}
        self.write_rows(self.source / "labeled_frames.jsonl", self.rows)
        self.write(self.source / "manifest.json", dict(schema=installer.SCHEMA, episode_key="run/lap_001", category="dynamic_obstacle",
                   native_episode_relative=self.native, counts=counts, source_labels_sha256="b" * 64, input_identity=identity))
        self.write(self.labels / "applied_index.json", dict(schema=installer.SCHEMA + "_index", status="COMPLETE", episodes=1,
                   records=[self.record], counts=counts, input_identity=identity))
        self.write(self.labels / "report.json", {"status": "COMPLETE"})
        self.seal_package()

    def write(self, path, value):
        path.write_text(json.dumps(value) + "\n")

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def seal(self, folder, names, **metadata):
        self.write(folder / "COMPLETE.json", dict(files={name: installer.sha256(folder / name) for name in names}, **metadata))

    def seal_package(self):
        self.seal(self.source, ["labeled_frames.jsonl", "manifest.json"], schema=installer.SCHEMA + "_episode_complete", status="COMPLETE")
        index = installer.read_json(self.labels / "applied_index.json")
        for name, key in (("labeled_frames.jsonl", "labels_sha256"), ("manifest.json", "manifest_sha256"), ("COMPLETE.json", "complete_sha256")):
            index["records"][0][key] = installer.sha256(self.source / name)
        self.write(self.labels / "applied_index.json", index)
        names = [p.relative_to(self.labels).as_posix() for p in self.labels.rglob("*") if p.is_file() and p != self.labels / "COMPLETE.json"]
        self.seal(self.labels, names, schema=installer.SCHEMA + "_complete", status="COMPLETE", input_identity=index["input_identity"])

    def old_labels(self, subset=False):
        folder = self.episode / "situation_labels"
        folder.mkdir()
        rows = copy.deepcopy(self.rows[:1] if subset else self.rows)
        for row in rows:
            row["instruction"] = "Old instruction."
            row["labeling"]["instruction_id"] = "OLDER_CLASS_REVISION"
        self.write_rows(folder / "labeled_frames.jsonl", rows)
        self.write(folder / "report.json", {"generation": "old"})
        self.seal(folder, ["labeled_frames.jsonl", "report.json"])
        return {p.name: p.read_bytes() for p in folder.iterdir()}

    def immutable_files(self):
        return {p.relative_to(self.dataset).as_posix(): p.read_bytes() for p in self.dataset.rglob("*") if p.is_file()
                and "situation_labels" not in p.parts and not any(part.startswith(".situation") for part in p.parts)}

    def call(self, **kwargs):
        return installer.install(self.labels, self.dataset, **kwargs)

    def test_fresh_install_has_physical_masks_and_leaves_native_data_untouched(self):
        before = self.immutable_files()
        report = self.call()
        target = self.episode / "situation_labels"
        self.assertEqual(report["instruction_targets"], 1)
        self.assertEqual(report["backup_paths"], [])
        self.assertEqual(installer.file_pins(target), installer.file_pins(self.source))
        rows = installer.jsonl(target / "labeled_frames.jsonl")
        self.assertIsNone(rows[0]["instruction"])
        self.assertTrue(all(rows[0][name] is False for name in installer.MASKS))
        self.assertEqual(rows[0]["labeling"]["instruction_id"], "CONTINUE")
        self.assertEqual(before, self.immutable_files())
        self.assertEqual(self.call(resume=True)["episode_results"], {"reused": 1})

    def test_existing_old_cohort_subset_is_preserved_in_backup(self):
        old = self.old_labels(subset=True)
        report = self.call()
        backup = Path(report["backup_paths"][0])
        self.assertEqual(old, {p.name: p.read_bytes() for p in backup.iterdir()})
        self.assertNotIn("situation_labels", backup.parts)
        self.assertEqual(len(list(self.dataset.glob("**/situation_labels/labeled_frames.jsonl"))), 1)
        self.call(resume=True)
        self.assertEqual(old, {p.name: p.read_bytes() for p in backup.iterdir()})

    def test_preflight_and_invalid_native_mapping_write_nothing(self):
        before = self.immutable_files()
        self.assertEqual(self.call(preflight_only=True)["status"], "PREFLIGHT_VERIFIED")
        self.assertFalse((self.dataset / ".situation_label_installs").exists())
        self.assertFalse((self.episode / "situation_labels").exists())
        (self.episode / "episode.json").unlink()
        with self.assertRaisesRegex(ValueError, "metadata missing"):
            self.call()
        self.assertFalse((self.dataset / ".situation_label_installs").exists())
        self.assertEqual(before["dynamic_obstacle/run/lap_001/episode_000/raw/untouched.bin"],
                         (self.episode / "raw/untouched.bin").read_bytes())

    def test_bad_masks_and_asset_symlink_fail_before_output(self):
        self.rows[0]["instruction_target_valid"] = True
        self.write_rows(self.source / "labeled_frames.jsonl", self.rows)
        self.seal_package()
        with self.assertRaisesRegex(ValueError, "masks differ"):
            self.call()
        self.assertFalse((self.dataset / ".situation_label_installs").exists())
        self.rows[0]["instruction_target_valid"] = False
        self.write_rows(self.source / "labeled_frames.jsonl", self.rows)
        self.seal_package()
        asset = Path(self.rows[0]["assets"]["camera_1"]["path"])
        outside = self.root / "outside.png"
        outside.write_bytes(asset.read_bytes())
        asset.unlink()
        asset.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.call(preflight_only=True)
        self.assertFalse((self.dataset / ".situation_label_installs").exists())

    def test_resume_after_interruption_between_backup_and_replacement(self):
        old = self.old_labels()
        save_state = installer.save_state
        def interrupt(plan, state, value):
            save_state(plan, state, value)
            if value == "backed_up":
                raise RuntimeError("power loss")
        with mock.patch.object(installer, "save_state", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "power loss"):
                self.call()
        self.assertFalse((self.episode / "situation_labels").exists())
        self.call(resume=True)
        backup = next(self.episode.glob(".situation_label_backups/*/previous"))
        self.assertEqual(old, {p.name: p.read_bytes() for p in backup.iterdir()})
        installer.verify_installed(self.episode / "situation_labels", installer.file_pins(self.source))

    def test_replacement_failure_restores_original_and_can_resume(self):
        old = self.old_labels()
        rename = installer.os.rename
        def fail_stage(source, target):
            if Path(source).name.startswith(".situation_label_stage_"):
                raise OSError("replacement failed")
            return rename(source, target)
        with mock.patch.object(installer.os, "rename", side_effect=fail_stage):
            with self.assertRaisesRegex(OSError, "replacement failed"):
                self.call()
        target = self.episode / "situation_labels"
        self.assertEqual(old, {p.name: p.read_bytes() for p in target.iterdir()})
        self.call(resume=True)
        installer.verify_installed(target, installer.file_pins(self.source))

    def test_partial_staging_resume_and_completed_three_file_hash_checks(self):
        copyfile = installer.shutil.copyfile
        def interrupt_copy(source, target):
            if Path(source).name == "manifest.json":
                raise RuntimeError("copy interruption")
            return copyfile(source, target)
        with mock.patch.object(installer.shutil, "copyfile", side_effect=interrupt_copy):
            with self.assertRaisesRegex(RuntimeError, "copy interruption"):
                self.call()
        self.call(resume=True)
        for name in installer.FILES:
            with self.subTest(changed=name):
                path = self.episode / "situation_labels" / name
                original = path.read_bytes()
                path.write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "hashes differ"):
                    self.call(resume=True)
                path.write_bytes(original)

    def test_overlapping_native_frame_identity_collision_fails_without_journal(self):
        self.old_labels(subset=True)
        folder = self.episode / "situation_labels"
        old = installer.jsonl(folder / "labeled_frames.jsonl")
        old[0]["sensor_time_ns"] += 10
        self.write_rows(folder / "labeled_frames.jsonl", old)
        self.seal(folder, ["labeled_frames.jsonl", "report.json"])
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with self.assertRaisesRegex(ValueError, "frame identity differs"):
            self.call()
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
        self.assertFalse((self.dataset / ".situation_label_installs").exists())

    def test_changed_class_provenance_is_rejected_even_for_excluded_instructions(self):
        self.rows[0]["labeling"]["instruction_id"] = "ALTERED_CLASS"
        self.write_rows(self.source / "labeled_frames.jsonl", self.rows)
        self.seal_package()
        with self.assertRaisesRegex(ValueError, "source class fields changed"):
            self.call()
        self.assertFalse((self.dataset / ".situation_label_installs").exists())

    def test_resume_binds_native_episode_metadata(self):
        self.call()
        self.write(self.episode / "episode.json", {"scenario_id": "different"})
        with self.assertRaisesRegex(ValueError, "journal differs"):
            self.call(resume=True)


if __name__ == "__main__":
    unittest.main()
