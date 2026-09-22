import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "internal_sensor_record"))
from nas_publish import (campaign_folder, component, dataset_campaign_id,
    dataset_run_name, inventory, make_job, make_tree_job, materialize_validated,
    scenario_folder, SENSORS,
    validate_target, publish_validated, sha256)

class NasPlanningTests(unittest.TestCase):
    def test_future_campaign_layout(self):
        self.assertEqual(campaign_folder({"campaign_id": "scenario_campaign_20261001_123456"}), "full loop_123456")
        self.assertEqual(scenario_folder({"scenario_name": "full_loop_normal"}), "basic_driving")
        self.assertEqual(scenario_folder({"scenario_name": "full_loop_dynamic"}), "dynamic_obstacle")
        self.assertEqual(scenario_folder({"scenario_name": "full_loop_static"}), "static_obstacle")
        self.assertEqual(scenario_folder({"scenario_name": "full_loop_roundabout"}), "roundabout")
        self.assertEqual(scenario_folder({"scenario_name": "full_loop_highway"}), "highway")

    def test_public_dataset_alias_preserves_raw_identity(self):
        plan = {
            "campaign_id": "normal_10runs_retry2_20260917",
            "dataset_campaign_id": "basic_driving_lap10_20260917_091633",
        }
        self.assertEqual(dataset_campaign_id(plan),
                         "basic_driving_lap10_20260917_091633")
        self.assertEqual(dataset_run_name(plan, {
            "lap": 7,
            "run_name": "normal_10runs_retry2_20260917_lap_007",
        }), "basic_driving_lap10_20260917_091633_lap_007")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "public_run"
            source.mkdir()
            (source / "frame").write_text("data")
            plan["scenario_name"] = "full_loop_normal"
            public_run = "basic_driving_lap10_20260917_091633_lap_007"
            job = make_tree_job(source, public_run, plan, "Z:\\DCM\\dataset",
                                root / "state", str)
            payload = json.loads(job.read_text())
            self.assertEqual(payload["destination"],
                             "Z:\\DCM\\dataset\\basic_driving\\"
                             "basic_driving_lap10_20260917_091633\\" + public_run)

    def test_local_materialization_merges_run_without_duplicate_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "sensor_record_auto_v1"
            campaign_id = "scenario_campaign_20261001_123456"
            run_name = campaign_id + "_lap_001"
            campaign = base / "campaigns" / campaign_id
            run = base / run_name
            native = base / "native_sessions" / run_name
            output = campaign / "validation_test"
            output.mkdir(parents=True)
            run.mkdir(parents=True)
            native.mkdir(parents=True)
            for sensor in SENSORS:
                (native / sensor).mkdir()
            sensor_file = native / "CAMERA_1/frame.png"
            sensor_file.write_bytes(b"sensor")
            (native / "SESSION.json").write_text("{}")
            (run / "episode_000/raw").mkdir(parents=True)
            topic_file = run / "episode_000/raw/records.jsonl"
            topic_file.write_text("topic")
            (run / "native_setup.json").write_text(json.dumps({"native_root": str(native)}))
            for kind in ("apply", "plan"):
                (base / (run_name + "_native_session_" + kind + ".json")).write_text("{}")
            plan = campaign / "campaign_plan.json"
            ledger = campaign / "runs.tsv"
            completion = campaign / "COLLECTION_COMPLETE.json"
            plan.write_text(json.dumps({"campaign_id": campaign_id,
                                        "scenario_name": "full_loop_dynamic",
                                        "requested_laps": 1}))
            ledger.write_text("lap\trun_name\tstatus\n1\t%s\tcaptured\n" % run_name)
            completion.write_text("{}")
            report_path = output / "report.json"
            report_path.write_text(json.dumps({
                "schema": "morai_sensor_record_campaign_validation_v1",
                "valid": True, "campaign": str(campaign),
                "campaign_plan_sha256": sha256(plan),
                "runs_ledger_sha256": sha256(ledger),
                "collection_completion_sha256": sha256(completion),
                "laps": [{"run_name": run_name, "valid": True}],
            }))
            (output / "COMPLETE.json").write_text(json.dumps({
                "status": "passed", "files": {"report.json": sha256(report_path)}}))
            modules = {"ssd_storage": SimpleNamespace(SSD_ROOT=root)}
            with patch.dict(sys.modules, modules):
                result = materialize_validated(report_path)
            canonical = root / "dataset/dynamic_obstacle" / campaign_id / run_name
            self.assertEqual(Path(result["campaign"]), canonical.parent)
            self.assertTrue((canonical / "CAMERA_1/frame.png").is_file())
            self.assertTrue((canonical / "episode_000/raw/records.jsonl").is_file())
            self.assertTrue(sensor_file.samefile(canonical / "CAMERA_1/frame.png"))
            self.assertTrue(topic_file.samefile(canonical / "episode_000/raw/records.jsonl"))
            self.assertTrue((canonical.parent / "_campaign/metadata/campaign_plan.json").is_file())
            self.assertTrue((canonical.parent / "_campaign/validations/validation_test/report.json").is_file())

            job = make_tree_job(canonical, run_name,
                                {"campaign_id": campaign_id,
                                 "scenario_name": "full_loop_dynamic"},
                                "Z:\\DCM\\dataset", root / "state", str)
            payload = json.loads(job.read_text())
            self.assertEqual(payload["destination"],
                             "Z:\\DCM\\dataset\\dynamic_obstacle\\%s\\%s" %
                             (campaign_id, run_name))

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
            campaign_id = "scenario_campaign_20261001_123456"
            campaign = root / campaign_id
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

    def test_sealed_valid_subset_plans_only_selected_laps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_id = "scenario_campaign_20261001_123456"
            campaign = root / campaign_id
            output = campaign / "validation_test"
            output.mkdir(parents=True)
            plan = campaign / "campaign_plan.json"
            ledger = campaign / "runs.tsv"
            completion = campaign / "COLLECTION_COMPLETE.json"
            plan.write_text(json.dumps({
                "campaign_id": campaign_id,
                "scenario_name": "full_loop_dynamic",
                "requested_laps": 3,
            }))
            ledger.write_text("unchanged")
            completion.write_text("{}")
            report_path = output / "report.json"
            report = {
                "schema": "morai_sensor_record_campaign_validation_subset_v1",
                "valid": True,
                "campaign": str(campaign),
                "campaign_plan_sha256": sha256(plan),
                "runs_ledger_sha256": sha256(ledger),
                "collection_completion_sha256": sha256(completion),
                "selected_laps": 2,
                "laps": [
                    {"run_name": "lap_001", "valid": True},
                    {"run_name": "lap_003", "valid": True},
                ],
                "excluded_laps": [{"run_name": "lap_002", "valid": False}],
                "subset_coverage_preserved": True,
                "subset_policy": (
                    "whole_lap_fail_closed_exclusion_with_full_variation_preservation_v1"),
            }
            report_path.write_text(json.dumps(report))
            (output / "COMPLETE.json").write_text(json.dumps({
                "status": "passed", "files": {"report.json": sha256(report_path)}}))
            modules = {
                "runtime_paths": SimpleNamespace(
                    LOCAL_MACHINE={}, POWERSHELL_EXE="never-execute",
                    windows_path_from_wsl=str),
                "ssd_storage": SimpleNamespace(SSD_ROOT=root),
            }

            local = {
                "schema": "morai_local_dataset_campaign_v1",
                "campaign_id": campaign_id,
                "scenario_folder": "dynamic_obstacle",
                "runs": [
                    {"path": str(root / "local/lap_001"), "relative": "lap_001",
                     "run_name": "lap_001"},
                    {"path": str(root / "local/lap_003"), "relative": "lap_003",
                     "run_name": "lap_003"},
                ],
                "evidence": [],
            }
            for row in local["runs"]:
                Path(row["path"]).mkdir(parents=True)

            def fake_job(_source, relative, _plan, _target, state_root, _translate):
                path = Path(state_root) / (relative + ".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "files": 1, "bytes": 1, "destination": "Z:\\DCM\\" + relative,
                    "identity": relative,
                }))
                return path

            with patch.dict(sys.modules, modules), patch(
                    "nas_publish.materialize_validated", return_value=local), patch(
                    "nas_publish.make_tree_job", side_effect=fake_job) as make:
                result = publish_validated(
                    report_path, "Z:\\DCM\\dataset", dry_run=True)
            self.assertEqual([row["publication"] for row in result], ["lap_001", "lap_003"])
            self.assertEqual(make.call_count, 2)

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
