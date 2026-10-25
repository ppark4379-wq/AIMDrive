"""Stage-by-stage diagnosis: offline, synthetic runs plus the real 20261024_215633_456370 log."""

import json
from pathlib import Path
import tempfile
import unittest

from ..adapters.ve_signal import FullRouteSignalPolicy
from ..validation.failure_analysis import analyze
from ..validation.failure_report import build_report
from ..validation.paths import repo_root

REAL_RUN = None
NEW_RUN = None
REPO = repo_root()
RUN_ID = "run-stage"
T0 = 10000.0
RUN_OFFSET = 3.0          # Runner elapsed 0 == pipeline t 3.0


def _historical_run(name):
    """Drive logs were not moved: look in the new .runtime first, then the old MoraiDrive one."""
    package = Path(__file__).resolve().parents[1]
    for root in (package / ".runtime", package.parents[2] / "MoraiDrive" / "scenario_runner" / ".runtime"):
        run = root / "ve_gps_full_route" / name
        if run.is_dir():
            return run
    return package / ".runtime" / "ve_gps_full_route" / name
REAL_RUN = _historical_run("20261024_215633_456370")
NEW_RUN = _historical_run("20261024_230026_748759")


class Gate(FullRouteSignalPolicy):
    """The real accept()/output() state machine with the route binding stubbed out."""

    def bind(self, pose, points):
        self.head_source, self.accepted_codes, self.permission_codes = "native_binary", (1, 16), (16,)
        return True


def default_gt(elapsed):
    return 16 if elapsed < 15 or elapsed >= 60 else 1


def build_signal_run(root, *, hz=20, seconds=120, late_latency=0.1, signal=True, gt=default_gt, audit=False,
                     ego_blocked=False, head_noise=False, gap=None):
    """Scripted run: the ego follows the real gate output (go only on code 16)."""
    driver = root / "driver"
    driver.mkdir(parents=True)
    gate = Gate([], [])
    packets, ctrl, pipeline, truth, csv_rows = [], [], [], [], []
    latency = lambda t: 0.1 if t < 40 else late_latency
    j, next_sensor = 0, 0.0
    queue, last_seq, last_recv, x, v, seq_row = [], None, None, 0.0, 5.0, 0
    track = []
    for k in range(seconds * hz):
        t = k / hz
        now = T0 + t
        while next_sensor <= t:                          # VE sensor frames at 5 Hz
            elapsed = next_sensor - RUN_OFFSET
            lost = gap is not None and gap[0] <= next_sensor < gap[1]          # VE frames never arrive
            code = gt(elapsed)
            if head_noise and next_sensor >= 40 and j % 3 == 0:
                code = None                                                    # VE head undecided
            if not lost:
                queue.append(dict(seq=j, tp=next_sensor, arrive=next_sensor + latency(next_sensor), code=code,
                                  conf=0.8 if code is None else 0.99999))
            j += 1
            next_sensor += 0.2
        while queue and queue[0]["arrive"] <= t:
            packet = queue.pop(0)
            recv = T0 + packet["tp"]
            if signal:
                gate.output(now)
                gate.accept({"ego_pose": {}, "sensor_received_ns": int(round(recv * 1e9)),
                             "native_signal": {"code": packet["code"], "confidence": packet["conf"]}}, [], now)
            packets.append(dict(sequence=packet["seq"], sensor_received_ns=int(round(recv * 1e9)),
                                sent_ns=int(round((T0 + packet["arrive"]) * 1e9)),
                                native_signal={"code": packet["code"], "confidence": packet["conf"]}, signal={}))
            last_seq, last_recv = packet["seq"], recv
        code = gate.output(now) if signal else 0
        age = None if last_recv is None else now - last_recv
        valid = age is not None and age <= 0.5
        target = 5.0 if code == 16 and not ego_blocked else 0.0
        v += max(-4.0 / hz, min(2.0 / hz, target - v))
        x += v / hz
        track.append((t, x, v))
        brake = 0.0 if target > 0 and valid else 1.0
        base = {"schema": "aimdrive_pipeline_v1", "run_id": RUN_ID, "clock_domain": "host_monotonic_ns", "cycle_id": k}
        ns = int(round(now * 1e9))
        pipeline.append(dict(base, stage="perception", sequence=3 * k, timestamp_ns=ns, ve_valid=valid,
                             ve_sequence=last_seq, objects=[] if valid else None))
        pipeline.append(dict(base, stage="planning", sequence=3 * k + 1, timestamp_ns=ns + 1000000,
                             planning_mode="static_obstacle" if valid else "unavailable",
                             planning_reason="signal_stop" if valid else "unavailable",
                             target_velocity_mps=target if valid else None, planned_trajectory=None,
                             input_objects=[],
                             debug_details={"static_avoidance": {"speed_source": "signal" if valid else None,
                                                                 "speed_limits": {"signal": 0.0 if code != 16 else 14.0}}}))
        pipeline.append(dict(base, stage="control", sequence=3 * k + 2, timestamp_ns=ns + 2000000,
                             planning_cycle_id=k if valid else None, sent=True, sent_ns=ns + 3000000,
                             target_velocity_mps=target if valid else None,
                             final_wire_command={"accel": 0.0 if brake else 0.3, "brake": brake, "steer_normalized": 0.0},
                             observed_ego={"x": x, "y": 0.0, "yaw_grid_rad": 0.0, "speed_mps": v}))
        row = dict(monotonic=now, VE_sequence=last_seq, signal_id="S1" if signal else "", signal_code=code,
                   signal_head="native_binary" if signal else "unsupported",
                   perception_age_ms=age * 1000 if valid else None,
                   planned_command=dict(accel=0.0 if brake else 0.3, brake=brake, front_steer_rad=0.0))
        if audit and signal:
            row["signal_gate"] = gate.audit(now)
        ctrl.append(row)
    for i in range(0, seconds * 5 + 1):                   # Runner side: elapsed clock, 2 s GT samples
        elapsed = i / 5.0
        index = min(len(track) - 1, max(0, int(round((elapsed + RUN_OFFSET) * hz))))
        csv_rows.append("%.3f,%.4f,0.0,0.0,%.4f" % (elapsed, track[index][1], track[index][2]))
        if i % 10 == 0 and signal:
            truth.append(dict(elapsed_s=elapsed, unix=0.0, colors={"S1": gt(elapsed)}, x=track[index][1], y=0.0))
    write_lines = lambda path, rows: path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_lines(driver / "pipeline.jsonl", pipeline)
    write_lines(driver / "control.jsonl", ctrl)
    write_lines(driver / "inputs.jsonl", packets)
    write_lines(root / "signal_truth.jsonl", truth)
    (root / "scenario_ego_trace.csv").write_text("elapsed_s,x,y,yaw_deg,speed\n" + "\n".join(csv_rows) + "\n", encoding="utf-8")
    (root / "report.json").write_text(json.dumps({"status": "FAILED", "error": "Synthetic termination"}), encoding="utf-8")
    return gate


class StageDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run"

    def stage(self, result):
        self.assertIsNone(result["stage_error"])
        return {f["rule_code"]: f for f in result["findings"] if f.get("group") == "stage"}

    def test_green_with_stale_input_is_traced_through_the_gate(self):
        build_signal_run(self.root, late_latency=0.46)
        stage = self.stage(analyze(self.root))
        for code in ("GREEN_BUT_STOPPED", "SIGNAL_PROPAGATION_LOSS", "SIGNAL_STALE_REJECTED", "SIGNAL_PERMISSION_RESET"):
            self.assertTrue(stage[code]["detected"], code)
        self.assertEqual(stage["SIGNAL_PERMISSION_RESET"]["status"], "SUSPECTED")        # replay, not an observation
        self.assertGreaterEqual(stage["SIGNAL_PERMISSION_RESET"]["counts"]["resets"], 3)
        chain = stage["SIGNAL_GATE_CAUSE_CHAIN"]
        self.assertEqual(chain["status"], "SUSPECTED")
        self.assertIn("미검증", chain["causal"]["statement"])

    def test_route_straight_gt_48_is_a_green_window(self):
        build_signal_run(self.root, late_latency=0.46)
        truth = self.root / "signal_truth.jsonl"
        rows = [json.loads(line) for line in truth.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["colors"]["S1"] == 16:
                row["colors"]["S1"] = 48
        truth.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (self.root / "driver" / "signal_targets.json").write_text(json.dumps({
            "targets": [{"signal_id": "S1", "signal_movement": "straight"}]}), encoding="utf-8")
        result = analyze(self.root)
        self.assertTrue(result["stage"]["meta"]["windows"])
        self.assertTrue(self.stage(result)["GREEN_BUT_STOPPED"]["detected"])

    def test_gt_48_requires_logged_route_movement(self):
        build_signal_run(self.root, late_latency=0.46)
        truth = self.root / "signal_truth.jsonl"
        rows = [json.loads(line) for line in truth.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["colors"]["S1"] == 16:
                row["colors"]["S1"] = 48
        truth.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        result = analyze(self.root)
        self.assertEqual(result["stage"]["meta"]["windows"], [])

    def test_code_zero_is_never_counted_as_red(self):
        build_signal_run(self.root, late_latency=0.46)
        loss = self.stage(analyze(self.root))["SIGNAL_PROPAGATION_LOSS"]
        self.assertTrue(any("적색(1)을 낸 cycle 0개" in fact for fact in loss["facts"]))
        self.assertTrue(any("미확정(0)" in fact for fact in loss["facts"]))

    def test_audit_makes_reset_an_observation(self):
        build_signal_run(self.root, late_latency=0.46, audit=True)
        stage = self.stage(analyze(self.root))
        reset = stage["SIGNAL_PERMISSION_RESET"]
        self.assertEqual(reset["status"], "OBSERVED")
        self.assertEqual(reset["counts"]["source"], "audit")
        self.assertTrue(stage["SIGNAL_STALE_REJECTED"]["detected"])
        self.assertTrue(any("Audit" in fact for fact in stage["SIGNAL_STALE_REJECTED"]["facts"]))

    def test_healthy_green_departure_is_not_a_failure(self):
        build_signal_run(self.root, late_latency=0.1)
        stage = self.stage(analyze(self.root))
        for code in ("GREEN_BUT_STOPPED", "SIGNAL_PROPAGATION_LOSS", "SIGNAL_STALE_REJECTED", "SIGNAL_PERMISSION_RESET",
                     "CONTROL_RESPONSE_MISMATCH", "SIGNAL_GATE_CAUSE_CHAIN"):
            self.assertFalse(stage[code]["detected"], code)

    def test_normal_red_stop_is_not_flagged(self):
        build_signal_run(self.root, late_latency=0.1, gt=lambda e: 16 if e < 15 else 1)
        result = analyze(self.root)
        stage = self.stage(result)
        self.assertFalse([c for c, f in stage.items() if f["detected"]])
        self.assertNotIn("SUSPECTED", {f["status"] for f in stage.values()})
        # the long stop is real behaviour (a stall) but it is not attributed to the signal chain
        self.assertEqual([b["id"] for b in result["summary"]["behavioral"] if b["id"].startswith("stage.")], [])
        self.assertEqual(result["summary"]["suspected"], [x for x in result["summary"]["suspected"] if not x["id"].startswith("stage.")])

    def test_segment_without_signal_is_not_applicable(self):
        build_signal_run(self.root, signal=False, ego_blocked=True)
        stage = self.stage(analyze(self.root))
        for code in ("GREEN_BUT_STOPPED", "SIGNAL_PROPAGATION_LOSS", "SIGNAL_STALE_REJECTED", "SIGNAL_PERMISSION_RESET"):
            self.assertFalse(stage[code]["detected"], code)
            self.assertEqual(stage[code]["status"], "NOT_APPLICABLE", code)

    def test_unaligned_ground_truth_is_never_used(self):
        build_signal_run(self.root, late_latency=0.46)
        # a Runner trace without Ego motion cannot fix the GT clock, so GT-based rules must refuse to judge
        (self.root / "scenario_ego_trace.csv").write_text("elapsed_s,x,y,yaw_deg,speed\n0,0,0,0,0\n1,0,0,0,0\n", encoding="utf-8")
        stage = self.stage(analyze(self.root))
        self.assertEqual(stage["GREEN_BUT_STOPPED"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(stage["GREEN_BUT_STOPPED"]["detected"])

    def test_verified_requires_a_counterfactual_record(self):
        build_signal_run(self.root, late_latency=0.46)
        self.assertNotIn("VERIFIED", {f["status"] for f in analyze(self.root)["findings"]})
        record = [{"finding_id": "stage.SIGNAL_GATE_CAUSE_CHAIN", "kind": "counterfactual_rerun",
                   "same_seed_and_setup": True, "outcome_changed": True, "evidence_file": "rerun.json"}]
        chain = {f["id"]: f for f in analyze(self.root, verifications=record)["findings"]}["stage.SIGNAL_GATE_CAUSE_CHAIN"]
        self.assertEqual(chain["status"], "VERIFIED")
        weak = [dict(record[0], outcome_changed=False)]
        chain = {f["id"]: f for f in analyze(self.root, verifications=weak)["findings"]}["stage.SIGNAL_GATE_CAUSE_CHAIN"]
        self.assertEqual(chain["status"], "SUSPECTED")


    def test_audit_run_classifies_why_green_was_blocked(self):
        build_signal_run(self.root, late_latency=0.46, audit=True)
        stage = self.stage(analyze(self.root))
        block = stage["SIGNAL_GATE_BLOCK_REASON"]
        self.assertEqual((block["status"], block["counts"]["source"]), ("OBSERVED", "audit"))
        self.assertTrue(block["detected"])
        self.assertTrue(set(block["counts"]["dominant"]) & {"debounce_samples", "no_packet", "stale_invalidated"})
        self.assertGreater(block["counts"]["blocked"], block["counts"]["passed"])
        reset = stage["SIGNAL_DEBOUNCE_INSUFFICIENT"]["counts"]
        self.assertEqual(sum(reset["reset_reasons"].values()), reset["resets"])      # events, not rows
        self.assertGreaterEqual(stage["SIGNAL_STALE_REJECTED"]["counts"]["stale_resets"], 1)

    def test_head_that_never_says_green_is_a_perception_finding_not_a_gate_finding(self):
        build_signal_run(self.root, late_latency=0.1, head_noise=True, audit=True)
        stage = self.stage(analyze(self.root))
        self.assertTrue(stage["SIGNAL_HEAD_MISMATCH"]["detected"])
        build_signal_run(Path(self.temporary.name) / "healthy", late_latency=0.1, audit=True)
        healthy = self.stage(analyze(Path(self.temporary.name) / "healthy"))
        self.assertFalse(healthy["SIGNAL_HEAD_MISMATCH"]["detected"])

    def test_data_gap_is_perception_unavailable_not_a_red_signal(self):
        build_signal_run(self.root, late_latency=0.1, gap=(70.0, 95.0))
        result = analyze(self.root)
        stage = self.stage(result)
        self.assertTrue(stage["PERCEPTION_UNAVAILABLE"]["detected"])
        block = stage["SIGNAL_GATE_BLOCK_REASON"]
        self.assertEqual(block["counts"].get("gate_red", 0), 0)               # code 0 during the gap is never counted as red
        self.assertGreater(block["counts"].get("no_packet", 0) + block["counts"].get("stale_invalidated", 0), 0)
        self.assertGreater(stage["PERCEPTION_UNAVAILABLE"]["counts"]["invalid"], 0)
        nodes = {n["id"]: n for n in result["stage"]["pipeline"]["nodes"]}
        self.assertEqual(nodes["perception"]["state"], "problem")

    def test_pipeline_marks_unverified_stage_instead_of_asserting_it(self):
        build_signal_run(self.root, late_latency=0.46, audit=True)
        nodes = {n["id"]: n for n in analyze(self.root)["stage"]["pipeline"]["nodes"]}
        self.assertEqual(nodes["gate"]["state"], "problem")
        self.assertEqual(nodes["ego"]["state"], "unverified")                        # no command acknowledgement exists
        self.assertIn("승인", nodes["ego"]["summary"])
        build_signal_run(Path(self.temporary.name) / "red", late_latency=0.1, gt=lambda e: 16 if e < 15 else 1)
        red = {n["id"]: n for n in analyze(Path(self.temporary.name) / "red")["stage"]["pipeline"]["nodes"]}
        self.assertNotIn("problem", {n["state"] for n in red.values()})

    def test_summary_is_korean_first_and_hides_other_scenario_hypotheses(self):
        build_signal_run(self.root, late_latency=0.46, audit=True)
        summary = analyze(self.root)["summary"]
        self.assertTrue(summary["headline"])
        self.assertIn("Synthetic termination", summary["termination"]["title_ko"])
        self.assertTrue(summary["suspected"])
        self.assertEqual(summary["suspected"][0]["code"], "SIGNAL_GATE_CAUSE_CHAIN")
        self.assertIn("미검증", summary["headline"])
        low = {item["id"] for item in summary["insufficient_low"]}
        self.assertTrue({"hyp.coupled_yield_deadlock", "hyp.perception_track_fragmentation"} <= low)
        shown = {item["id"] for item in summary["insufficient"]}
        self.assertFalse(shown & low)
        html = build_report(self.root)[0].read_text(encoding="utf-8")
        self.assertIn("EXECUTIVE SUMMARY", html)
        self.assertNotIn("drawFour", html)

    def test_ground_truth_never_reaches_driving_inputs(self):
        package = Path(__file__).resolve().parents[1]
        for path in (package / "adapters" / "ve_signal.py", REPO / "AIM2026" / "Expert" / "runtime.py"):
            self.assertNotIn("signal_truth", path.read_text(encoding="utf-8"), path.name)
        source = (Path(__file__).resolve().parents[1] / "validation" / "stage_diagnosis.py").read_text(encoding="utf-8")
        for forbidden in ("import rospy", "morai_msgs", "import run_ve_gps_expert", "GetTrafficLightStatus"):
            self.assertNotIn(forbidden, source)

    def test_report_has_four_groups_and_stage_chain(self):
        build_signal_run(self.root, late_latency=0.46)
        path, payload = build_report(self.root)
        html = path.read_text(encoding="utf-8")
        for label in ("TERMINATION CAUSE", "BEHAVIORAL FAILURE", "SUSPECTED CAUSE", "INSUFFICIENT EVIDENCE"):
            self.assertIn(label, html)
        self.assertEqual(payload["summary"]["termination"]["reason"], "Synthetic termination")
        self.assertTrue(payload["stage"]["chain"]["t"])
        self.assertEqual(payload["verified_causes"], [])

    @unittest.skipUnless((REAL_RUN / "driver" / "control.jsonl").is_file(), "real signal-stall log not present")
    def test_real_run_detects_signal_propagation_loss(self):
        result = analyze(REAL_RUN)
        stage = self.stage(result)
        self.assertTrue(stage["SIGNAL_PROPAGATION_LOSS"]["detected"])
        self.assertTrue(stage["SIGNAL_STALE_REJECTED"]["detected"])
        self.assertTrue(stage["GREEN_BUT_STOPPED"]["detected"])
        self.assertTrue(stage["PERCEPTION_UNAVAILABLE"]["detected"])
        self.assertFalse(stage["CONTROL_RESPONSE_MISMATCH"]["detected"])
        self.assertEqual(stage["SIGNAL_PERMISSION_RESET"]["status"], "SUSPECTED")
        self.assertEqual(stage["SIGNAL_GATE_CAUSE_CHAIN"]["status"], "SUSPECTED")
        self.assertGreaterEqual(result["stage"]["meta"]["replay_agreement"], 0.9)
        self.assertTrue(result["stage"]["meta"]["gt_clock"]["verified"])
        self.assertNotIn("VERIFIED", {f["status"] for f in result["findings"]})
        summary = result["summary"]
        self.assertEqual(summary["termination"]["category"], "static_obstacle_displacement")
        self.assertIn("stage.GREEN_BUT_STOPPED", [item["id"] for item in summary["behavioral"]])
        self.assertNotIn("장애물", " ".join(item["title_ko"] + item["detail_ko"] for item in summary["behavioral"]))

    @unittest.skipUnless((NEW_RUN / "driver" / "control.jsonl").is_file(), "20261024_230026_748759 log not present")
    def test_latest_real_run_uses_audit_and_matches_the_replay(self):
        result = analyze(NEW_RUN)
        stage = self.stage(result)
        meta = result["stage"]["meta"]
        self.assertEqual(meta["allow_source"], "audit")
        self.assertGreaterEqual(meta["replay_vs_audit"]["output_agreement"], 0.95)
        self.assertGreaterEqual(meta["replay_vs_audit"]["reason_agreement"], 0.9)
        block = stage["SIGNAL_GATE_BLOCK_REASON"]
        self.assertEqual((block["status"], block["counts"]["source"]), ("OBSERVED", "audit"))
        self.assertEqual(block["counts"]["dominant"][0], "debounce_samples")
        self.assertTrue(stage["SIGNAL_PROPAGATION_LOSS"]["detected"])
        self.assertTrue(stage["SIGNAL_HEAD_MISMATCH"]["detected"])          # differs from the earlier run: head output itself
        self.assertEqual(stage["SIGNAL_GATE_CAUSE_CHAIN"]["status"], "SUSPECTED")
        self.assertEqual(result["summary"]["termination"]["category"], "progress_watchdog")
        self.assertNotIn("VERIFIED", {f["status"] for f in result["findings"]})

    @unittest.skipUnless((NEW_RUN / "driver" / "control.jsonl").is_file() and (REAL_RUN / "driver" / "control.jsonl").is_file(),
                         "both real logs required")
    def test_earlier_run_is_reevaluated_with_the_same_rules(self):
        old = self.stage(analyze(REAL_RUN))
        self.assertFalse(old["SIGNAL_HEAD_MISMATCH"]["detected"])           # earlier run: head was right 99% of the time
        self.assertTrue(old["SIGNAL_STALE_REJECTED"]["detected"])
        self.assertEqual(old["SIGNAL_GATE_CAUSE_CHAIN"]["status"], "SUSPECTED")
        self.assertEqual(old["SIGNAL_GATE_BLOCK_REASON"]["counts"]["source"], "replay")
        self.assertEqual(old["SIGNAL_GATE_BLOCK_REASON"]["status"], "SUSPECTED")   # reconstructed, not observed


if __name__ == "__main__":
    unittest.main()
