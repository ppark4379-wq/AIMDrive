"""Failure analysis + report.html: offline, no MORAI/ROS/VE."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..validation.failure_analysis import analyze, detect_stall, DEFAULTS
from ..validation.failure_report import build_report

def _historical_run(name):
    """Drive logs were not moved: look in the new .runtime first, then the old MoraiDrive one."""
    package = Path(__file__).resolve().parents[1]
    for root in (package / ".runtime", package.parents[2] / "MoraiDrive" / "scenario_runner" / ".runtime"):
        run = root / "ve_gps_full_route" / name
        if run.is_dir():
            return run
    return package / ".runtime" / "ve_gps_full_route" / name

REAL_RUN = _historical_run("20261024_205129_023285")
MALFORMED_DEBUG_RUN = _historical_run("20261024_230026_748759")
RUN_ID = "run-test"
T0 = 5000.0


def make_run(root, *, crawl=True, timeout=True, seconds=100, hz=10):
    """Ego speeds up for 30 s then crawls (or keeps going); NPC 40 m ahead on +x."""
    driver = root / "driver"
    driver.mkdir(parents=True)
    rows, x, npc_obs = [], 0.0, []
    for i in range(seconds * hz):
        t = i / hz
        speed = min(3.0, t / 10) if t < 30 else (0.1 if crawl else 3.0)
        x += speed / hz
        ns = int((T0 + t) * 1e9)
        reason = "jamming_dynamic_yield" if i % 2 else "jamming_dynamic_clear"
        valid = i % 5 != 0
        obj = {"center_world": [x + 25, 0.0], "heading_deg": 0.0, "width_m": 2.0, "length_m": 4.5,
               "track_id": i // 50, "observed_speed_mps": 0.8, "planning_speed_mps": 0.0,
               "planning_speed_source": "unconfirmed", "motion_state": "uncertain", "speed_mps": None}
        base = {"schema": "aimdrive_pipeline_v1", "run_id": RUN_ID, "clock_domain": "host_monotonic_ns",
                "cycle_id": i, "logger_code_sha256": "x" * 64}
        rows.append(dict(base, stage="perception", sequence=3 * i, timestamp_ns=ns, ve_valid=valid,
                         ve_sequence=i, objects=[obj] if valid else None))
        rows.append(dict(base, stage="planning", sequence=3 * i + 1, timestamp_ns=ns + 1000000,
                         planning_mode="static_obstacle" if valid else "unavailable",
                         planning_reason=reason if valid else "unavailable",
                         target_velocity_mps=(0.0 if i % 2 else 4.0) if valid else None,
                         planned_trajectory=[[x + k, 0.0] for k in range(5)] if valid else None,
                         input_objects=[]))
        rows.append(dict(base, stage="control", sequence=3 * i + 2, timestamp_ns=ns + 2000000,
                         planning_cycle_id=i if valid else None, sent=True, sent_ns=ns + 3000000,
                         target_velocity_mps=(0.0 if i % 2 else 4.0) if valid else None,
                         final_wire_command={"accel": 0.0, "brake": 1.0 if i % 2 or not valid else 0.0,
                                             "steer_normalized": 0.0},
                         observed_ego={"x": x, "y": 0.0, "yaw_grid_rad": 0.0, "speed_mps": speed}))
        if i % hz == 0 and t >= 12:
            npc_obs.append({"time": T0 + t, "x": x + 40.0, "y": 0.0, "yaw_deg": 0.0, "speed_mps": 0.3,
                            "route_gap_m": 40.0, "actor_id": "1"})
    (driver / "pipeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if timeout:
        runs = root / "tunnel_vehicle_runs"
        runs.mkdir()
        end = seconds - 1
        (runs / (RUN_ID + ".json")).write_text(json.dumps({
            "run_seed": 1, "actor_id": "1", "observations": npc_obs, "encounter_verified": False,
            "completed": False, "failure_reason": "Tunnel vehicle timed out", "travelled_m": 12.0,
            "approach_speed_mps": 0.1, "plan": {"behavior": "vehicle_slow", "speed_mps": 1.2},
            "actual_spawn": {"time": T0 + 12.0}, "spawn_elapsed_sec": 10.0,
            "transitions": [{"from": "moving", "to": "failed", "elapsed_sec": 10.0 + end - 12.0 - 0.5}],
            "last_observation": {"route_gap_m": 40.0}}), encoding="utf-8")
        (root / "report.json").write_text(json.dumps({"status": "FAILED", "error": "Tunnel vehicle timed out"}),
                                          encoding="utf-8")
        (root / "scenario_context.json").write_text(json.dumps(
            {"scenario_config": {"tunnel_vehicle_dynamic": {"encounter_gap_m": 20.0, "slow_ego_speed_ratio": 0.6}}}),
            encoding="utf-8")


class FailureReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run"

    def by_id(self, result):
        return {f["id"]: f for f in result["findings"]}

    def test_acc_traceback_and_unknown_speed_are_shown_as_separate_evidence(self):
        make_run(self.root, seconds=2, hz=2, timeout=False)
        pipeline = self.root / "driver" / "pipeline.jsonl"
        rows = [json.loads(line) for line in pipeline.read_text(encoding="utf-8").splitlines()]
        last = next(row for row in reversed(rows) if row["stage"] == "perception")
        last["objects"][0]["planning_speed_mps"] = None
        pipeline.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        (self.root / "report.json").write_text(json.dumps(
            {"status": "FAILED", "error": "AIMDrive expert process stopped unexpectedly"}), encoding="utf-8")
        (self.root / "driver.console.log").write_text(
            'Traceback (most recent call last):\n'
            '  File "adaptive_cruise_control.py", line 209, in get_target_velocity\n'
            '    velocity_error = ego_vel - self.object_velocity\n'
            "TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'\n", encoding="utf-8")
        path, report = build_report(self.root)
        findings = self.by_id(report)
        self.assertEqual(findings["runtime.acc_type_error"]["status"], "OBSERVED")
        self.assertEqual(findings["hyp.ve_unknown_speed_acc"]["status"], "SUSPECTED")
        self.assertEqual(report["summary"]["suspected"][0]["id"], "hyp.ve_unknown_speed_acc")
        self.assertIn("hyp.ve_unknown_speed_acc", path.read_text(encoding="utf-8"))

    def test_generic_driver_exit_does_not_invent_acc_cause(self):
        make_run(self.root, seconds=2, hz=2, timeout=False)
        (self.root / "report.json").write_text(json.dumps(
            {"status": "FAILED", "error": "AIMDrive expert process stopped unexpectedly"}), encoding="utf-8")
        self.assertNotIn("hyp.ve_unknown_speed_acc", self.by_id(analyze(self.root)))

    def test_static_route_blockage_precedes_watchdog_in_first_screen(self):
        make_run(self.root, seconds=70, hz=2, timeout=False)
        pipeline = self.root / "driver" / "pipeline.jsonl"
        rows = [json.loads(line) for line in pipeline.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["cycle_id"] >= 60:
                if row["stage"] == "perception":
                    row["ve_valid"] = True
                elif row["stage"] == "planning":
                    row.update(planning_mode="static_obstacle", planning_reason="static_recorded_blocked",
                               target_velocity_mps=0.0)
                elif row["stage"] == "control":
                    row["final_wire_command"].update(accel=0.0, brake=1.0)
        pipeline.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        (self.root / "report.json").write_text(json.dumps({
            "status": "FAILED", "error": "EGO progress watchdog abort: reason=no_route_progress "
            "stalled=180.0s threshold=180.0s"}), encoding="utf-8")
        (self.root / "driver" / "expert.log").write_text(
            "[STATIC_FAILURE] reason=static_recorded_blocked error=tracking_error search={}\n",
            encoding="utf-8")
        path, result = build_report(self.root)
        findings = self.by_id(result)
        self.assertEqual(findings["plan.static_route_blocked"]["status"], "OBSERVED")
        self.assertEqual(findings["hyp.static_route_stall"]["status"], "SUSPECTED")
        self.assertEqual(result["summary"]["suspected"][0]["id"], "hyp.static_route_stall")
        self.assertIn("tracking_error", result["summary"]["headline"])
        self.assertIn("hyp.static_route_stall", path.read_text(encoding="utf-8"))

    def test_planning_debug_types_and_nested_fields(self):
        cases = [
            ({"static_avoidance": {"speed_source": "signal", "speed_limits": {"signal": 0.2}}}, "signal", 0.2, False),
            (json.dumps({"static_avoidance": {"speed_source": "signal", "speed_limits": {"signal": 0.2}}}), "signal", 0.2, False),
            ("unavailable", None, None, True),
            (None, None, None, False),
            ({"static_avoidance": "bad nested value"}, None, None, True),
            ({"static_avoidance": {"speed_limits": "bad limits"}}, None, None, True),
            (json.dumps({"static_avoidance": json.dumps({"speed_source": "static"})}), "static", None, False),
            ("[]", None, None, True),
            ({"static_avoidance": {"speed_source": {"bad": True}}}, None, None, True),
            ({"static_avoidance": {"speed_limits": {"signal": "not a speed"}}}, None, None, True),
        ]
        for index, (value, source, signal, malformed) in enumerate(cases):
            with self.subTest(index=index):
                root = Path(self.temporary.name) / ("case_%d" % index)
                make_run(root, seconds=2, hz=2, timeout=False)
                path = root / "driver" / "pipeline.jsonl"
                lines = path.read_text(encoding="utf-8").splitlines()
                row = json.loads(lines[1])
                row["debug_details"] = value
                lines[1] = json.dumps(row)
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                result = analyze(root)
                first = result["_ctx"]["table"][0]
                self.assertEqual((first["speed_source"], first["signal_limit"]), (source, signal))
                warnings = result["data_quality_warnings"]
                self.assertEqual(bool(warnings), malformed)
                if malformed:
                    self.assertEqual(warnings[0]["actual_type"], type(warnings[0]["raw_value"]).__name__)
                    self.assertIn("debug_details", warnings[0]["field"])
                    finding = self.by_id(result)["input.planning_debug_details"]
                    self.assertEqual(finding["status"], "INSUFFICIENT_EVIDENCE")
                    self.assertEqual(finding["evidence"][0]["line"], 2)
                else:
                    self.assertNotIn("input.planning_debug_details", self.by_id(result))

    @unittest.skipUnless((MALFORMED_DEBUG_RUN / "driver" / "pipeline.jsonl").is_file(), "real malformed debug log not present")
    def test_real_malformed_debug_run_generates_html(self):
        path, payload = build_report(MALFORMED_DEBUG_RUN, Path(self.temporary.name) / "real_malformed")
        self.assertTrue(path.is_file())
        self.assertIn("<html", path.read_text(encoding="utf-8"))
        warnings = payload["data_quality_warnings"]
        self.assertTrue(any(w["cycle_id"] == 2094 and w["line"] == 6284 and
                            w["raw_value"] == "unavailable" for w in warnings))
        self.assertEqual(self.by_id(payload)["input.planning_debug_details"]["status"], "INSUFFICIENT_EVIDENCE")

    def test_timeout_and_stall_are_both_reported(self):
        make_run(self.root)
        result = analyze(self.root)
        self.assertEqual(result["failure"]["type"], "timeout")
        self.assertEqual(result["failure_types"], ["stall", "timeout"])
        self.assertTrue(result["stall"]["detected"])
        self.assertEqual(result["counts"]["cycles"], 1000)

    def test_stages_are_joined_by_cycle_and_line(self):
        make_run(self.root)
        findings = self.by_id(analyze(self.root))
        link = findings["link.quality"]
        import re
        match = re.search(r"cycle (\d+)개, 세 단계 모두 존재 (\d+)", link["claim"])
        self.assertEqual(match.group(1), match.group(2))
        first = link["evidence"][0]
        self.assertEqual((first["file"], first["line"]), ("driver/pipeline.jsonl", 1))

    def test_no_verified_without_verification_record(self):
        make_run(self.root)
        result = analyze(self.root)
        self.assertNotIn("VERIFIED", {f["status"] for f in result["findings"]})
        self.assertIn("SUSPECTED", {f["status"] for f in result["findings"]})

    def test_verified_only_with_matching_counterfactual_record(self):
        make_run(self.root)
        bad = [{"finding_id": "hyp.yield_chatter", "kind": "counterfactual_rerun",
                "same_seed_and_setup": True, "outcome_changed": False, "evidence_file": "rerun.json"}]
        self.assertNotEqual(self.by_id(analyze(self.root, verifications=bad))["hyp.yield_chatter"]["status"], "VERIFIED")
        good = [dict(bad[0], outcome_changed=True)]
        finding = self.by_id(analyze(self.root, verifications=good))["hyp.yield_chatter"]
        self.assertEqual(finding["status"], "VERIFIED")
        self.assertEqual(finding["verification"]["evidence_file"], "rerun.json")

    def test_ve_track_is_never_equated_with_gt_npc(self):
        make_run(self.root)
        result = analyze(self.root)
        self.assertTrue(result["alignment"]["verified"])
        self.assertEqual(self.by_id(result)["assoc.ve_gt"]["status"], "INSUFFICIENT_EVIDENCE")

    def test_misaligned_clock_blocks_gt_comparison(self):
        make_run(self.root)
        path = self.root / "tunnel_vehicle_runs" / (RUN_ID + ".json")
        data = json.loads(path.read_text(encoding="utf-8"))
        for obs in data["observations"]:
            obs["time"] += 7.0                       # GT clock shifted relative to the pipeline clock
        path.write_text(json.dumps(data), encoding="utf-8")
        result = analyze(self.root)
        self.assertFalse(result["alignment"]["verified"])
        self.assertEqual(result["npc_series"], [])
        self.assertFalse(result["bev"]["npc_shown"])
        self.assertTrue(all(f["npc"] is None for f in result["bev"]["frames"]))

    def test_stall_only_run_without_explicit_failure(self):
        make_run(self.root, timeout=False)
        result = analyze(self.root)
        self.assertEqual(result["failure"]["type"], "stall")
        self.assertEqual(result["failure_types"], ["stall"])

    def test_moving_ego_is_not_a_stall(self):
        make_run(self.root, crawl=False, timeout=False)
        result = analyze(self.root)
        self.assertFalse(result["stall"]["detected"])
        self.assertIsNone(result["failure"])

    def test_completed_run_does_not_invent_a_failure_cause_from_ve_gaps(self):
        make_run(self.root, crawl=False, timeout=False)
        (self.root / "report.json").write_text(json.dumps({"status": "COMPLETE", "error": None}), encoding="utf-8")
        (self.root / "driver" / "report.json").write_text(
            json.dumps({"status": "STOPPED", "error": None}), encoding="utf-8")
        path, result = build_report(self.root)
        self.assertEqual(result["run_status"], "COMPLETE")
        self.assertIsNone(result["failure"])
        self.assertEqual(result["failure_types"], [])
        self.assertEqual(result["summary"]["termination"]["category"], "complete")
        self.assertEqual(result["summary"]["suspected"], [])
        self.assertFalse(any(f["group"] == "hypothesis" for f in result["findings"]))
        self.assertIn("주행 완료 리포트", path.read_text(encoding="utf-8"))

    def test_detect_stall_needs_sustained_low_progress(self):
        table = [{"ns": int(i * 1e8), "ego_x": i * 0.1 * 0.2, "ego_y": 0.0, "cycle": i} for i in range(1500)]
        self.assertTrue(detect_stall(table, DEFAULTS)["detected"])
        table = [{"ns": int(i * 1e8), "ego_x": i * 0.1 * 2.0, "ego_y": 0.0, "cycle": i} for i in range(1500)]
        self.assertFalse(detect_stall(table, DEFAULTS)["detected"])

    def test_report_is_light_and_references_resolve(self):
        make_run(self.root)
        source_size = (self.root / "driver" / "pipeline.jsonl").stat().st_size
        path, payload = build_report(self.root)
        html = path.read_text(encoding="utf-8")
        self.assertLess(len(html.encode("utf-8")), source_size)
        self.assertNotIn("logger_code_sha256", html)
        self.assertLessEqual(len(payload["excerpts"]), 400)
        cited = [(r["file"], r["line"]) for f in payload["findings"] for r in f["evidence"]
                 if r.get("line") and r.get("kind") != "source"]
        self.assertTrue(cited)
        lines = (self.root / "driver" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()
        for file, line in cited:
            row = json.loads(lines[line - 1])
            self.assertEqual(row["run_id"], RUN_ID)
            self.assertIn("%s:%d" % (file, line), payload["excerpts"])
        self.assertEqual(payload["verified_causes"], [])

    def test_does_not_modify_inputs(self):
        make_run(self.root)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        build_report(self.root, self.root.parent / "out")
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_nonfinite_nested_log_evidence_still_produces_strict_json_report(self):
        make_run(self.root, seconds=2, hz=2)
        evidence = {"planning": {"speed_limits": {"section": float('inf')},
                                  "values": [float('-inf'), float('nan'), 2.5]}}
        with patch('scenario_runner.validation.failure_report.collect_excerpts', return_value=evidence):
            path, payload = build_report(self.root)
        def reject_constant(value):
            raise ValueError(value)
        decoded = json.loads(path.with_name('report_data.json').read_text(encoding='utf-8'),
                             parse_constant=reject_constant)
        self.assertEqual(decoded['excerpts']['planning']['speed_limits']['section'], 'inf')
        self.assertEqual(decoded['excerpts']['planning']['values'], ['-inf', 'nan', 2.5])
        self.assertTrue(path.is_file())
        self.assertEqual(payload, decoded)

    def test_incident_collision_layout(self):
        make_run(self.root, timeout=False)
        folder = self.root.parent / "incident_run-test_000001"
        folder.mkdir()
        rows = [json.loads(l) for l in (self.root / "driver" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
        for stage in ("perception", "planning", "control"):
            (folder / (stage + "_trace.jsonl")).write_text(
                "".join(json.dumps(r) + "\n" for r in rows if r["stage"] == stage), encoding="utf-8")
        end = T0 + 90.0
        (folder / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "incident_id": "run-test_000001", "source_run_id": RUN_ID,
            "controller": "ve_gps_udp", "failure": {"event_type": "collision_started", "actor_ids": ["1"],
                                                    "captured_monotonic_sec": end}}), encoding="utf-8")
        result = analyze(folder)
        self.assertEqual(result["failure"]["type"], "collision")
        self.assertEqual(result["source_kind"], "incident")
        ids = {f["id"] for f in result["findings"]}
        self.assertIn("existing.perception.collision_object", ids)
        cited = [r for f in result["findings"] for r in f["evidence"] if r.get("line")]
        self.assertTrue(all(r["file"].endswith("_trace.jsonl") for r in cited if "trace" in r["file"]))

    @unittest.skipUnless((REAL_RUN / "driver" / "pipeline.jsonl").is_file(), "real timeout log not present")
    def test_real_timeout_run_generates_report(self):
        out = Path(self.temporary.name) / "real"
        path, payload = build_report(REAL_RUN, out)
        self.assertTrue(path.is_file())
        self.assertEqual(payload["run_id"], "1792842692687_0000007b")
        self.assertEqual(payload["failure"]["type"], "timeout")
        self.assertIn("stall", payload["failure_types"])
        self.assertEqual(payload["verified_causes"], [])
        self.assertLess(path.stat().st_size, 2 * 1024 * 1024)
        findings = {f["id"]: f for f in payload["findings"]}
        self.assertEqual(findings["assoc.ve_gt"]["status"], "INSUFFICIENT_EVIDENCE")
        for fid in ("fail.event", "ego.progress", "npc.gt_state"):
            self.assertEqual(findings[fid]["status"], "OBSERVED")


if __name__ == "__main__":
    unittest.main()
