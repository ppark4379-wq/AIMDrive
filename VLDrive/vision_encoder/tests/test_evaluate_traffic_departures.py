"""CPU-only tests of bounded train event selection and evaluation accounting."""
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import evaluate_traffic_departures as evaluation


def measurement(frame, green_frame=30):
    green = frame >= green_frame
    state = 16 if green else 1
    return {
        "frame_id": frame, "native_sync_frame": frame * 5 + 1,
        "grpc_label_tick_raw": frame * 5 + 1, "simulator_labels_source": "grpc",
        "navigation": {"affected_light_id_raw": "signal_a", "affected_light_state_raw": state},
        "traffic_lights": [{"tl_id": "signal_a", "state": state}],
        "is_red_light_present": not green, "speed": 0.0002 if frame <= green_frame else 0.2,
    }


class DepartureEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.rows = [measurement(frame) for frame in range(50)]

    def events(self, rows=None, train=None):
        return evaluation.select_events(rows or self.rows, train or range(40), "route", [30],
                                        post_green_frames=5, pre_stop_samples=10)

    def test_bounded_known_red_stops_and_exact_first_green_plus_five(self):
        event = self.events()[0]
        self.assertEqual(event["evaluated_departure_frames"], [30, 31, 32, 33, 34, 35])
        self.assertEqual(event["contiguous_stopped_interval"], [0, 29])
        self.assertEqual(len(event["evaluated_stopped_frames"]), 10)
        self.assertEqual(event["evaluated_stopped_frames"][0], 0)
        self.assertEqual(event["evaluated_stopped_frames"][-1], 29)
        self.assertEqual(event["known_red_stopped_candidate_count"], 30)

    def test_any_departure_frame_outside_train_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "train split"):
            self.events(train=range(35))

    def test_unknown_or_ambiguous_green_is_not_a_verified_transition(self):
        for state in (None, 33):
            rows = copy.deepcopy(self.rows)
            rows[30]["navigation"]["affected_light_state_raw"] = state
            rows[30]["traffic_lights"][0]["state"] = state
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "red-to-known-green"):
                self.events(rows=rows)

    def test_changed_signal_or_gap_in_window_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[35]["native_sync_frame"] += 5
        with self.assertRaisesRegex(ValueError, "native ticks"):
            self.events(rows=rows)
        rows = copy.deepcopy(self.rows)
        rows[33]["navigation"]["affected_light_id_raw"] = "different"
        with self.assertRaisesRegex(ValueError, "unknown/different"):
            self.events(rows=rows)

    def test_offline_future_waypoint_values_cannot_change_selection(self):
        before = self.events()
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row["future_waypoints"] = [[999, -999]] * 5
            row["gps_x"] = 999
            row["gps_y"] = -999
        self.assertEqual(before, self.events(rows=changed))

    def test_decoder_hook_tokens_match_feature_model_pooling_exactly(self):
        torch.manual_seed(42)
        decoder = torch.randn(1, 2506, 1, 256)
        hidden = decoder[0].permute(1, 0, 2)
        spatial = hidden[:, :2500].reshape(1, 50, 50, -1).permute(0, 3, 1, 2)
        spatial = F.adaptive_avg_pool2d(spatial, (10, 10)).view(1, -1, 100).permute(0, 2, 1)
        expected = torch.cat([spatial, hidden[:, 2500].view(1, 1, -1), hidden[:, 2501:2506]], 1)
        actual = evaluation.tokens_from_decoder(decoder)
        self.assertEqual(tuple(actual.shape), (1, 106, 256))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_latency_censoring_and_stationary_future_gt_accounting(self):
        event = self.events()[0]
        frames = event["evaluated_stopped_frames"] + event["evaluated_departure_frames"]
        rows = {frame: {"frame_id": frame, "predicted_traffic_class": int(frame >= 33),
                        "ade_m": 1.0, "fde_m": 2.0, "gt_endpoint_distance_m": 0.0,
                        "predicted_endpoint_distance_m": 0.5} for frame in frames}
        rows[29]["gt_endpoint_distance_m"] = 4.0  # Stopped now, moves during future GT.
        report = evaluation.summarize_event(event, rows)
        self.assertEqual(report["first_predicted_not_red_frame"], 33)
        self.assertAlmostEqual(report["traffic_not_red_latency_sec"], 0.3)
        self.assertEqual(report["stationary_future_gt_samples"], 9)
        for row in rows.values():
            row["predicted_traffic_class"] = 0
        censored = evaluation.summarize_event(event, rows)
        self.assertIsNone(censored["traffic_not_red_latency_sec"])
        self.assertTrue(censored["latency_right_censored"])
        self.assertEqual(censored["observation_horizon_sec"], 0.5)

    def test_token_change_reports_traffic_and_waypoint_queries_separately(self):
        before = np.ones((106, 256), dtype=np.float32)
        after = before.copy()
        after[100] *= 2
        result = evaluation.token_comparison({30: before}, {30: after})["30"]
        self.assertEqual(result["traffic"]["relative_l2"], 1.0)
        self.assertEqual(result["spatial"]["relative_l2"], 0.0)
        self.assertEqual(result["waypoint"]["relative_l2"], 0.0)

    def test_cli_is_bounded_and_baseline_needs_no_new_checkpoint(self):
        base = ["dataset", "--split-file", "split.json", "--traffic-label-sidecar", "labels.json",
                "--old-checkpoint", "old.pth", "--old-sha256", "a" * 64, "--output", "new_output"]
        args = evaluation.parse_args(base + ["--dry-run"])
        self.assertIsNone(args.new_checkpoint)
        self.assertTrue(args.dry_run)
        for extra in (["--new-checkpoint", "new.pth"], ["--post-green-frames", "5000"],
                      ["--first-green-frames", "991", "991"]):
            with self.subTest(extra=extra), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                evaluation.parse_args(base + extra)


if __name__ == "__main__":
    unittest.main()
