"""Old and revised publications must coexist with independently pinned evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from data.loaders.morai_sensor_record_dataset import digest, verify_admission_recovery


INTERNAL = (Path(__file__).resolve().parents[2] / "MoraiDrive/data_collection/"
            "internal_sensor_record")
sys.path[:0] = [str(INTERNAL), str(INTERNAL.parents[1])]
from morai_pretraining.sensor_record_manifest import (
    effective_admission_directory, effective_object_geometry_directory,
)

spec = importlib.util.spec_from_file_location(
    "scale_publication_builder", INTERNAL / "tools/build_sensor_record_training_dataset.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def sealed(folder, report):
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps(report))
    (folder / "frames.json").write_text("[]")
    (folder / "COMPLETE.json").write_text(json.dumps({"files": {
        name: digest(folder / name) for name in ("report.json", "frames.json")}}))


class VersionedRecoveryTests(unittest.TestCase):
    def make_recovery(self, run, version, name):
        episode = run / "episode_000"
        original = episode / "admission_v1"
        if not original.exists():
            sealed(original, {"raw_manifest_sha256": "raw"})
        geometry = episode / ("object_geometry_audit_v%d" % version)
        sealed(geometry, {"schema": "morai_sensor_record_object_geometry_audit_v%d" % version,
                          "raw_manifest_sha256": "raw", "structural_valid": True})
        corrected = episode / ("admission_v%d" % version)
        sealed(corrected, {
            "raw_manifest_sha256": "raw", "capture_contract_valid": True,
            "runtime_streams_valid": True, "sensor_capture_status": "admitted",
            "teacher_transport_status": "admitted", "source_timestamps_modified": False,
            "training_visible": False,
            "input_seals": {"object_geometry_audit": digest(geometry / "COMPLETE.json")},
        })
        def record(folder):
            return {"path": str((folder / "report.json").relative_to(run)),
                    "sha256": digest(folder / "report.json"),
                    "seal_sha256": digest(folder / "COMPLETE.json")}
        recovery = {
            "schema": "morai_sensor_record_admission_recovery_v1",
            "status": "completed", "recovery_is_additive": True,
            "source_timestamps_modified": False, "training_visible": False,
            "original_admission": record(original),
            "corrected_admission": record(corrected),
            "corrected_object_geometry": record(geometry),
        }
        (run / name).write_text(json.dumps(recovery))
        return {"admission_recovery": digest(run / name),
                "effective_admission": digest(corrected / "COMPLETE.json"),
                "effective_object_geometry": digest(geometry / "COMPLETE.json")}

    def test_legacy_and_scale_publications_coexist_and_bind_own_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            old = self.make_recovery(run, 2, "ADMISSION_RECOVERY.json")
            old_bytes = (run / "ADMISSION_RECOVERY.json").read_bytes()
            name = "ADMISSION_RECOVERY_SCALE_V1.json"
            new = self.make_recovery(run, 3, name)
            episode = run / "episode_000"
            self.assertEqual(verify_admission_recovery(episode, old), episode / "admission_v2")
            self.assertEqual(verify_admission_recovery(episode, new, name), episode / "admission_v3")
            self.assertEqual(effective_admission_directory(episode)[1], old)
            self.assertEqual(effective_admission_directory(episode, name)[1], new)
            self.assertEqual(effective_object_geometry_directory(episode, name),
                             episode / "object_geometry_audit_v3")
            self.assertEqual((run / "ADMISSION_RECOVERY.json").read_bytes(), old_bytes)
            with self.assertRaisesRegex(ValueError, "Changed admission recovery"):
                verify_admission_recovery(episode, old, name)
            with self.assertRaisesRegex(ValueError, "Changed admission recovery"):
                verify_admission_recovery(episode, new)
            (episode / "object_geometry_audit_v3/frames.json").write_text('[{"tampered":true}]')
            with self.assertRaises(ValueError):
                verify_admission_recovery(episode, new, name)
            # A modified NEW chain must not break the untouched legacy chain.
            self.assertEqual(verify_admission_recovery(episode, old), episode / "admission_v2")

    def test_recovery_cannot_escape_or_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            old = self.make_recovery(run, 2, "ADMISSION_RECOVERY.json")
            episode = run / "episode_000"
            for name in ("../ADMISSION_RECOVERY.json", "/tmp/ADMISSION_RECOVERY.json"):
                with self.assertRaises(ValueError):
                    verify_admission_recovery(episode, old, name)
                with self.assertRaises(ValueError):
                    effective_admission_directory(episode, name)
            with self.assertRaises(ValueError):
                verify_admission_recovery(episode, {}, "ADMISSION_RECOVERY_SCALE_V1.json")
            with self.assertRaisesRegex(ValueError, "missing"):
                effective_admission_directory(episode, "ADMISSION_RECOVERY_SCALE_V1.json")

    def test_geometry_revision_preserves_frames_splits_and_nonobject_targets(self):
        old = {"split": "test", "entries": [{"frame_id": 3, "traffic_light": 0,
                 "waypoints": [[0., -1.]], "assets": {"a.png": "hash"},
                 "objects": [], "object_valid": False, "object_positive": False}]}
        new = json.loads(json.dumps(old))
        new["entries"][0].update(object_valid=True, object_positive=True,
                                  objects=[{"id": "actor"}])
        builder.verify_reprocessed_manifest(old, new)
        new["entries"][0]["traffic_light"] = 1
        with self.assertRaisesRegex(ValueError, "non-object"):
            builder.verify_reprocessed_manifest(old, new)
        new = json.loads(json.dumps(old))
        new["split"] = "train"
        with self.assertRaisesRegex(ValueError, "split"):
            builder.verify_reprocessed_manifest(old, new)


if __name__ == "__main__":
    unittest.main()
