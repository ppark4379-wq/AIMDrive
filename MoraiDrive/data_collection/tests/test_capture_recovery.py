import importlib.util
import os
import json
import csv
import time
from pathlib import Path
import tempfile
import unittest

INTEGRATION = Path(__file__).resolve().parents[1] / "internal_sensor_record"
spec = importlib.util.spec_from_file_location(
    "recover_stale_capture", INTEGRATION / "tools/recover_stale_capture.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class CaptureRecoveryTests(unittest.TestCase):
    def direct_fixture(self, root):
        base = root / "sensor_record_auto_v1"
        campaign = root / "dataset/.collecting/tunnel_vehicle/campaign_03"
        metadata = campaign / "_campaign"
        metadata.mkdir(parents=True)
        (metadata / "campaign_plan.json").write_text(json.dumps({
            "schema": "morai_sensor_record_campaign_v1",
            "campaign_id": "campaign_03", "dataset_scenario_folder": "tunnel_vehicle",
        }))
        ledger = metadata / "runs.tsv"
        ledger.write_text("run_name\tstatus\ncampaign_03_lap_002\tstarted\n")
        return base, campaign, ledger

    def test_direct_staging_uses_exact_campaign_ledger_and_same_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            base, campaign, ledger = self.direct_fixture(Path(directory))
            run = campaign / "campaign_03_lap_002"
            self.assertEqual(recovery.recovery_run_location(run, base), run.resolve())
            self.assertEqual(list(recovery.campaign_ledgers(base)), [(ledger, campaign)])
            other_base = Path(directory) / "other/sensor_record_auto_v1"
            self.assertIsNone(recovery.recovery_run_location(run, other_base))
            self.assertIsNone(recovery.recovery_run_location(campaign / "_campaign", base))
            self.assertIsNone(recovery.recovery_run_location(run / "extra", base))
            (campaign / "_campaign/campaign_plan.json").write_text("{}")
            self.assertIsNone(recovery.recovery_run_location(run, base))

    def test_direct_orphan_keeps_interruption_identity_and_parent_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            base, campaign, _ = self.direct_fixture(Path(directory))
            request, row, processes = self.orphan_fixture(campaign)
            row["run_root"] = str(campaign / row["run"])
            proof = recovery.orphaned_stop_evidence(row, base, processes)
            self.assertEqual(proof["reason"], "campaign_interrupted")
            processes[10]["args"] = ["python3", "active_parent"]
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))

    def test_both_supported_runtimes_are_recognized(self):
        for name, kind in (
            ("collect_async_raw.py", "raw_recorder"),
            ("collect_async_raw_proven.py", "raw_recorder"),
            ("run_async_scenario_review.py", "scenario_runner"),
            ("run_proven_three_lap_scenario.py", "scenario_runner"),
        ):
            script = INTEGRATION / "tools" / name
            self.assertEqual(recovery.recovery_reference(script, INTEGRATION), (script, kind))

    def test_same_named_external_script_and_unrelated_worker_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "collect_async_raw_proven.py"
            clone.write_bytes((INTEGRATION / "tools" / clone.name).read_bytes())
            self.assertEqual(recovery.recovery_reference(clone, INTEGRATION), (None, None))
        validator = INTEGRATION / "tools/validate_sensor_record_campaign.py"
        self.assertEqual(recovery.recovery_reference(validator, INTEGRATION), (None, None))

    def test_only_terminal_interrupted_or_failed_processes_are_recoverable(self):
        rows = [
            dict(pid=i, safe_code_identity=True, process_identity={"start_ticks": "1"}, status=status)
            for i, status in enumerate(("interrupted", "failed_2", "started", "captured",
                                       "completed", None, "different_storage_root"))
        ]
        rows.extend([
            dict(pid=20, safe_code_identity=False, process_identity={"start_ticks": "1"}, status="interrupted"),
            dict(pid=21, safe_code_identity=True, process_identity=None, status="interrupted"),
        ])
        stale, active = recovery.partition_candidates(rows)
        self.assertEqual([row["pid"] for row in stale], [0, 1])
        self.assertEqual(len(active), 7)

    def test_process_identity_is_stable_and_missing_pid_is_not_recovered(self):
        identity = recovery.process_identity(os.getpid())
        self.assertEqual(identity, recovery.process_identity(os.getpid()))
        self.assertIsNotNone(identity)
        self.assertIsNone(recovery.process_identity(2147483647))


    def orphan_fixture(self, base):
        request = base / "run_lap_001/episode_000/sensor_record_coordination/stop.request"
        request.parent.mkdir(parents=True)
        request.write_text(json.dumps({"reason": "campaign_interrupted", "request_wall_ns": time.time_ns()}))
        identity = {"start_ticks": "99", "cmdline_sha256": "child"}
        row = dict(pid=50, run="run_lap_001", kind="raw_recorder", status="started",
                   safe_code_identity=True, process_identity=identity)
        processes = {
            50: dict(ppid=10, uid=1000, identity=identity, args=["python3", "collector"]),
            10: dict(ppid=1, uid=0, identity={"start_ticks": "1", "cmdline_sha256": "init"},
                     args=["/init"]),
        }
        return request, row, processes

    def test_started_orphan_requires_explicit_stop_and_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            request, row, processes = self.orphan_fixture(base)
            proof = recovery.orphaned_stop_evidence(row, base, processes)
            self.assertEqual(proof["reason"], "campaign_interrupted")
            row["orphaned_stop_evidence"] = proof
            self.assertEqual(recovery.partition_candidates([row])[0], [row])
            request.unlink()
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))

    def test_live_parent_or_supervisor_blocks_started_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            request, row, processes = self.orphan_fixture(base)
            processes[10]["args"] = ["python3", "runner"]
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))
            processes[10]["args"] = ["/init"]
            processes[60] = dict(args=["python3", "run_sensor_record_scenario.py", "--name", row["run"]])
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))
            processes[60] = dict(args=["python3", "runner", "--output=" + str(base / row["run"])])
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))

    def test_malformed_stop_request_and_reused_pid_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            request, row, processes = self.orphan_fixture(base)
            processes[50]["identity"] = {"start_ticks": "different", "cmdline_sha256": "child"}
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))
            processes[50]["identity"] = row["process_identity"]
            request.write_text('{"reason":"completed","request_wall_ns":1}')
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))
            request.write_text("{")
            self.assertIsNone(recovery.orphaned_stop_evidence(row, base, processes))

    def test_recovery_appends_interruption_without_overwriting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "runs.tsv"
            original = "lap\trun_name\tstatus\tfinished_utc\n1\trun_lap_001\tstarted\t-\n"
            ledger.write_text(original)
            self.assertTrue(recovery.append_recovered_interruption(ledger, "run_lap_001"))
            self.assertTrue(ledger.read_text().startswith(original))
            self.assertFalse(recovery.append_recovered_interruption(ledger, "run_lap_001"))
            rows = list(csv.DictReader(ledger.read_text().splitlines(), delimiter="\t"))
            self.assertEqual([r["status"] for r in rows], ["started", "interrupted"])
            ledger.write_text(original.replace("started", "captured"))
            before = ledger.read_bytes()
            self.assertFalse(recovery.append_recovered_interruption(ledger, "run_lap_001"))
            self.assertEqual(ledger.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
