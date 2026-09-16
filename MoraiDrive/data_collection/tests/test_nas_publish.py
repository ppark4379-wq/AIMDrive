import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "internal_sensor_record"))
from nas_publish import campaign_folder, component, inventory, make_job, SENSORS, validate_target, publish_validated, sha256

class NasPlanningTests(unittest.TestCase):
    def test_future_campaign_layout(self):
        self.assertEqual(campaign_folder({"campaign_id": "scenario_campaign_20261001_123456"}), "full loop_123456")

    def test_windows_names(self):
        for value in ("..", "a/b", "a\\b", "NUL.json", "x:", "x.", "x "):
            with self.assertRaises(ValueError):
                component(value)
        for value in ("relative", "Z:\\", "../nas"):
            with self.assertRaises(ValueError):
                validate_target(value)
        self.assertEqual(validate_target("Z:\\DCM\\dataset"), "Z:\\DCM\\dataset")

    def test_merge_inventory_identity_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            name = "scenario_campaign_20261001_123456_lap_001"
            run = base / name
            native = base / "native_sessions" / name
            run.mkdir()
            native.mkdir(parents=True)
            for sensor in SENSORS:
                (native / sensor).mkdir()
            (native / "CAMERA_1" / "frame.bin").write_bytes(b"sensor")
            (native / "SESSION.json").write_text("{}")
            (run / "native_setup.json").write_text(json.dumps({"native_root": str(native)}))
            (run / "episode_000").mkdir()
            (run / "episode_000" / "result.json").write_text("{}")
            for kind in ("apply", "plan"):
                (base / (name + "_native_session_" + kind + ".json")).write_text(
                    json.dumps({"session_name": name, "kind": kind}))
            plan = {"campaign_id": "scenario_campaign_20261001_123456"}
            path = make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str)
            job = json.loads(path.read_text())
            self.assertEqual(job["files"], 6)
            self.assertEqual(job["destination"], "Z:\\DCM\\dataset\\full loop_123456\\" + name)
            self.assertEqual(path, make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str))
            self.assertEqual((native / "CAMERA_1" / "frame.bin").read_bytes(), b"sensor")
            (native / "CAMERA_1" / "frame.bin").write_bytes(b"changed")
            self.assertNotEqual(path, make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str))

    def test_each_lap_exports_only_its_session_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            plan = {"campaign_id": "scenario_campaign_20261001_123456"}
            for lap in range(1, 4):
                name = plan["campaign_id"] + "_lap_%03d" % lap
                run = base / name
                native = base / "native_sessions" / name
                run.mkdir()
                native.mkdir(parents=True)
                for sensor in SENSORS:
                    (native / sensor).mkdir()
                (native / "SESSION.json").write_text("{}")
                (run / "native_setup.json").write_text(json.dumps({"native_root": str(native)}))
                for kind in ("apply", "plan"):
                    (base / (name + "_native_session_" + kind + ".json")).write_text(
                        json.dumps({"session_name": name, "kind": kind}))
            (base / "unrelated.json").write_text("must not export")
            for lap in range(1, 4):
                name = plan["campaign_id"] + "_lap_%03d" % lap
                job_path = make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str)
                job = json.loads(job_path.read_text())
                rows = [json.loads(line) for line in Path(job["inventory"]).read_text().splitlines()]
                evidence = [row for row in rows if "_native_session_" in row["relative"]]
                self.assertEqual({row["relative"] for row in evidence}, {
                    name + "_native_session_apply.json", name + "_native_session_plan.json"})
                self.assertEqual(job["files"], 4)
                self.assertNotIn("unrelated.json", {row["relative"] for row in rows})
                for row in evidence:
                    self.assertEqual(row["source"], str(base / row["relative"]))
                    self.assertEqual(row["kind"], "file")
                    self.assertEqual(row["size"], Path(row["source"]).stat().st_size)
                    self.assertIn(row["source"], job["sources"])
                # Missing/linked evidence must fail, not silently publish an incomplete lap.
                for kind in ("apply", "plan"):
                    source = base / (name + "_native_session_" + kind + ".json")
                    original = source.read_bytes()
                    source.unlink()
                    with self.assertRaisesRegex(ValueError, "native session evidence"):
                        make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str)
                    source.symlink_to(base / "unrelated.json")
                    with self.assertRaisesRegex(ValueError, "native session evidence"):
                        make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str)
                    source.unlink()
                    source.write_bytes(original)
                source = base / (name + "_native_session_plan.json")
                previous = make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str)
                source.write_text('{"changed": true}')
                self.assertNotEqual(previous, make_job(
                    base, name, plan, "Z:\\DCM\\dataset", base / "state", str))
                # A duplicate filename inside the run cannot shadow the sidecar.
                (base / name / source.name).write_text("collision")
                with self.assertRaisesRegex(ValueError, "Colliding"):
                    make_job(base, name, plan, "Z:\\DCM\\dataset", base / "state", str)

    def test_publication_requires_unchanged_sealed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            output = campaign / "validation_test"
            output.mkdir(parents=True)
            plan = campaign / "campaign_plan.json"
            ledger = campaign / "runs.tsv"
            completion = campaign / "COLLECTION_COMPLETE.json"
            plan.write_text(json.dumps({"campaign_id": "scenario_campaign_20261001_123456",
                                        "requested_laps": 1}))
            ledger.write_text("unchanged")
            completion.write_text("{}")
            report_path = output / "report.json"
            report = dict(valid=True, campaign=str(campaign), campaign_plan_sha256=sha256(plan),
                          runs_ledger_sha256=sha256(ledger),
                          collection_completion_sha256=sha256(completion), laps=[])
            report_path.write_text(json.dumps(report))
            seal = output / "COMPLETE.json"
            seal.write_text(json.dumps({"status": "passed", "files": {"report.json": sha256(report_path)}}))
            modules = {
                "runtime_paths": SimpleNamespace(LOCAL_MACHINE={}, POWERSHELL_EXE="never-execute",
                                                  windows_path_from_wsl=str),
                "ssd_storage": SimpleNamespace(SSD_ROOT=root)}
            with patch.dict(sys.modules, modules), patch("nas_publish.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError, "complete campaign"):
                    publish_validated(report_path, "Z:\\DCM\\dataset", dry_run=True)
                ledger.write_text("changed")
                with self.assertRaisesRegex(ValueError, "ledger changed"):
                    publish_validated(report_path, "Z:\\DCM\\dataset")
                report_path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "passed, sealed"):
                    publish_validated(report_path, "Z:\\DCM\\dataset")
                launch.assert_not_called()
            self.assertFalse((root / "sensor_record_auto_v1").exists())

    def test_collision_link_and_reserved_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            a, b = base / "a", base / "b"
            a.mkdir()
            b.mkdir()
            (a / "FRAME").write_text("a")
            (b / "frame").write_text("b")
            with self.assertRaisesRegex(ValueError, "Colliding"):
                inventory([a, b], base / "inventory", str)
            (b / "frame").unlink()
            (b / ".nas-temp-example").write_text("reserved")
            with self.assertRaisesRegex(ValueError, "reserved"):
                inventory([b], base / "inventory", str)
            (b / ".nas-temp-example").unlink()
            (b / "link").symlink_to(a / "FRAME")
            with self.assertRaisesRegex(ValueError, "symbolic"):
                inventory([b], base / "inventory", str)

if __name__ == "__main__":
    unittest.main()
