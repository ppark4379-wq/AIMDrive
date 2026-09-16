"""Offline regressions for reconnects and readiness heartbeat timing."""
from collections import defaultdict
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

INTERNAL = Path(__file__).resolve().parents[1] / "internal_sensor_record"
sys.path.insert(0, str(INTERNAL))
sys.path.insert(0, str(INTERNAL / "tools"))

from bridge.bridge_health import IMPLEMENTATION, check_bridge
from probe_sensor_record_readiness import (
    clock_ingress_failure, snapshot_observations, source_status,
)
from render_sensor_record_readiness import next_actions, render, state_for


def health_at(now):
    return dict(implementation=IMPLEMENTATION, binary_json_patch=True,
                boot_id="test-boot", code_sha256={"test": "hash"},
                monotonic_ns=now, listening=True)


def check(health, now):
    return check_bridge(health, now_ns=now, expected_boot="test-boot",
                        expected_hashes={"test": "hash"}, require_streams=False)


class SensorRecordReadinessTest(unittest.TestCase):
    def test_callback_during_unsubscribe_cannot_poison_closed_window(self):
        now = 10_000_000_000
        rows = defaultdict(list)
        lock = threading.Lock()
        rows["bridge_health"].append((now-1, health_at(now-2)))
        with patch("probe_sensor_record_readiness.time.monotonic_ns", return_value=now):
            snapshot, end = snapshot_observations(rows, lock)
        # A ROS callback can finish after the observation deadline while
        # subscriber.unregister() is closing other subscriptions.
        with lock:
            rows["bridge_health"].append((now+100, health_at(now+99)))
        self.assertEqual(check(snapshot["bridge_health"][-1][1], end), [])
        self.assertTrue(check(rows["bridge_health"][-1][1], end)[0].startswith(
            "BRIDGE_HEALTH_CLOCK_INVALID:"))
        self.assertEqual(len(snapshot["bridge_health"]), 1)

    def test_missing_stale_and_future_health_still_block_readiness_without_identity_error(self):
        now = 10_000_000_000
        for health, code in [
            ({}, "BRIDGE_HEALTH_PENDING:"),
            (health_at(now-2_500_000_001), "BRIDGE_HEALTH_STALE:"),
            (health_at(now+1), "BRIDGE_HEALTH_CLOCK_INVALID:"),
        ]:
            with self.subTest(code=code):
                failures = check(health, now)
                self.assertTrue(any(f.startswith(code) for f in failures))
                self.assertFalse(any(f.startswith("BRIDGE_NOT_PATCHED:") for f in failures))
                self.assertEqual(clock_ingress_failure([], health, failures),
                                 "clock_ingress_unavailable")
        self.assertEqual(check(health_at(now), now), [])

    def test_wrong_implementation_is_still_fatal_even_with_old_heartbeat(self):
        health = health_at(0)
        health["code_sha256"] = {"different": "hash"}
        failures = check(health, 10_000_000_000)
        self.assertTrue(any(f.startswith("BRIDGE_NOT_PATCHED:") for f in failures))
        self.assertTrue(any(f.startswith("BRIDGE_HEALTH_STALE:") for f in failures))

    def test_absent_constant_reversing_and_new_epoch_are_distinct(self):
        now = 10_000_000_000
        for stamps, expected in [([], ["missing_or_stale:ego"]),
                                 ([100, 100], ["nonadvancing_source:ego"]),
                                 ([100, 0, 1], ["reversing_source:ego"]),
                                 ([0, 0, 1, 2], [])]:
            with self.subTest(stamps=stamps):
                row, failures = source_status("ego", [(now, s) for s in stamps], now)
                self.assertEqual(failures, expected)
                if stamps:
                    self.assertEqual(row["source_first_ns"], stamps[0])
                    self.assertEqual(row["source_last_ns"], stamps[-1])
        # A fresh observation window after reset has no inherited timestamp.
        self.assertEqual(source_status("ego", [(now, 0), (now, 1)], now)[1], [])

    def test_clock_path_is_classified_only_with_valid_bridge_evidence(self):
        health = health_at(10)
        self.assertEqual(clock_ingress_failure([], health, []),
                         "morai_simulator_network_clock_not_sent")
        health["streams"] = {"/clock": {"samples": 5, "receive_age_ms": 10}}
        self.assertEqual(clock_ingress_failure([], health, []),
                         "clock_ingress_not_delivered_to_ros")
        self.assertEqual(clock_ingress_failure([], health, ["BRIDGE_HEALTH_STALE: old"]),
                         "clock_ingress_unavailable")
        self.assertIsNone(clock_ingress_failure([(10, 1)], health, []))

    def test_timestamp_failure_does_not_request_network_reconnect(self):
        for stamps in ([100, 100], [100, 0, 1]):
            row, failures = source_status("ego", [(10, s) for s in stamps], 10)
            report = {"ready": False, "failures": failures, "streams": {"ego": row}}
            actions = " ".join(next_actions(report))
            self.assertNotIn("Network Settings", actions)
            self.assertNotIn("Disconnect", actions)
            self.assertIn("External Sync OFF", actions)
            state = state_for("ego", row, failures)
            self.assertIn("역행" if len(set(stamps))>1 else "정지", state)

    def test_stale_heartbeat_render_keeps_clock_path_unknown_and_explains_retry(self):
        report = dict(ready=False, failures=["missing_or_stale:clock",
            "clock_ingress_unavailable", "BRIDGE_HEALTH_STALE: stale health heartbeat"],
            streams={"bridge_health": {"heartbeat_age_s": 3.1}})
        output = render(report)
        self.assertIn("3.100s", output)
        self.assertIn("다음 검사", output)
        self.assertIn("송신 경로 판정 대기", output)
        self.assertNotIn("MORAI→bridge 패킷 없음", output)
        self.assertNotIn("Simulator Network", output)


if __name__ == "__main__":
    unittest.main()
