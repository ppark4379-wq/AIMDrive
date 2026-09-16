import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/04_evaluate_expert_driving.py"
SPEC = importlib.util.spec_from_file_location("expert_event_evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
from expert_event_reporting import build_event_report, detect_off_route_events, load_off_route_config, match_event


class ExpertEventReportingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.episode = self.base / "run/episode_000"
        self.episode.mkdir(parents=True)
        self.config = {"enabled": True, "expert_max_cross_track_error_m": 10,
                       "off_route_timeout_sec": 3, "off_route_grace_sec": 0}
        self.sources = {
            "receipt": {i: {"receive_ns": (100 + i) * 1000000000, "source_ns": i * 100000000}
                        for i in range(11)},
            "position": {i: {"xy": [i, 0], "source_ns": i * 100000000} for i in range(11)},
            "warnings": [],
        }
        self.write_json("episode.json", {"scenario_config": self.config})

    def write_json(self, relative, value):
        path = self.episode / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def progress(self, t, cte=11):
        return {"event": "progress", "sequence": int(t), "elapsed_seconds": t,
                "receive_monotonic_ns": int((100 + t) * 1000000000),
                "cross_track_error_m": cte, "ego": {"x": t, "y": 0}}

    def detect(self, events):
        warnings = []
        records = list(enumerate(events, 1))
        return detect_off_route_events(records, dict(self.config), self.sources, warnings), warnings

    def test_off_route_duration_and_signed_distance_use_saved_threshold(self):
        events, warnings = self.detect([self.progress(t, -12) for t in range(4)] + [self.progress(4, 0)])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["duration_seconds"], 3)
        self.assertEqual(events[0]["duration_clock"], "receive_monotonic_ns")
        self.assertEqual(events[0]["max_cross_track_error_m"], 12)
        self.assertEqual(events[0]["start_sensor_bundle_index"], 0)
        self.assertEqual(events[0]["end_sensor_bundle_index"], 3)
        self.assertEqual(events[0]["progress_samples"], 4)
        self.assertFalse(events[0]["continuous_off_route_verified"])
        self.assertFalse(warnings)

    def test_short_excursions_and_threshold_equality_are_not_events(self):
        events = [self.progress(0), self.progress(2), self.progress(3, 10), self.progress(4), self.progress(6)]
        self.assertEqual(self.detect(events)[0], [])

    def test_corrupt_or_invalid_progress_breaks_intervals(self):
        for bad in (None, {"event": "progress"}, self.progress(2, float("nan"))):
            with self.subTest(bad=bad):
                self.assertEqual(self.detect([self.progress(0), self.progress(1), bad,
                                              self.progress(3), self.progress(4)])[0], [])

    def test_clock_reset_splits_intervals(self):
        result, warnings = self.detect([self.progress(0), self.progress(2), self.progress(1), self.progress(3)])
        self.assertEqual(result, [])
        self.assertTrue(warnings)

    def test_timeout_uses_host_duration_not_slow_scenario_clock(self):
        events = [self.progress(t) for t in range(4)]
        for event in events:
            event["elapsed_seconds"] *= .1
        self.assertEqual(len(self.detect(events)[0]), 1)

    def test_elapsed_fallback_is_explicit_when_all_receipts_are_missing(self):
        events = [self.progress(t) for t in range(4)]
        for event in events:
            del event["receive_monotonic_ns"]
        found, warnings = self.detect(events)
        self.assertEqual(found[0]["duration_clock"], "elapsed_seconds")
        self.assertIsNone(found[0]["start_sensor_bundle_index"])
        self.assertTrue(warnings)

    def test_saved_large_timeout_and_grace_are_respected(self):
        self.config["off_route_timeout_sec"] = 1e12
        self.assertEqual(self.detect([self.progress(t) for t in range(11)])[0], [])
        self.config.update(off_route_timeout_sec=3, off_route_grace_sec=2)
        self.assertEqual(self.detect([self.progress(t) for t in range(5)])[0], [])

    def test_config_defaults_and_invalid_values_are_disclosed(self):
        (self.episode / "episode.json").unlink()
        warnings = []
        config = load_off_route_config(self.episode, warnings)
        self.assertEqual(config["off_route_timeout_sec"], 3)
        self.assertEqual(config["source"], "runner_defaults")
        self.assertTrue(warnings)
        self.write_json("episode.json", {"scenario_config": {"off_route_timeout_sec": "broken"}})
        self.assertFalse(load_off_route_config(self.episode, [])["enabled"])

    def test_traffic_without_ego_has_time_only_candidate(self):
        result = match_event({"receive_monotonic_ns": 102010000000}, self.sources)
        self.assertEqual(result["sensor_bundle_index"], 2)
        self.assertEqual(result["match_confidence"], "TIME_ONLY")
        self.assertIsNone(result["position_candidate_index"])
        self.assertFalse(result["physical_collision_time_verified"])

    def test_time_only_never_selects_ties_or_out_of_range(self):
        for time in (102500000000, 99000000000):
            result = match_event({"receive_monotonic_ns": time}, self.sources)
            self.assertIsNone(result["sensor_bundle_index"])
            self.assertNotEqual(result["match_confidence"], "TIME_ONLY")

    def test_explicit_events_preserve_subtypes_and_do_not_duplicate_interventions(self):
        names = ["traffic_light_override", "traffic_light_override_verified",
                 "traffic_light_override_verification_timeout", "stuck", "expert_input_warning"]
        records = [(i, {"event": name, "receive_monotonic_ns": 102000000000,
                        "intervention": True, "light_id": "LIGHT"}) for i, name in enumerate(names, 1)]
        records.append((6, {"event": "custom_runner_change", "intervention": True}))
        report = build_event_report(self.episode, records, [], {"full_mission_completed": True}, self.sources)
        self.assertEqual([e["type"] for e in report["events"]], names + ["scenario_intervention"])
        self.assertEqual(report["events"][0]["details"]["light_id"], "LIGHT")
        self.assertEqual(report["events"][-1]["source_event"], "custom_runner_change")

    def test_incomplete_outcome_is_reported_without_rejection_or_invented_time(self):
        for outcome in ({"full_mission_completed": False, "reason": "interrupted"},
                        {"status": "failed", "reason": "KeyboardInterrupt"}):
            report = build_event_report(self.episode, [], [], outcome, self.sources)
            event = report["events"][0]
            self.assertEqual(event["type"], "lap_incomplete")
            self.assertEqual(event["reason"], outcome["reason"])
            self.assertIsNone(event["sensor_bundle_index"])
        for outcome in (None, {}, {"full_mission_completed": "false"}, {"full_mission_completed": True}):
            self.assertFalse(build_event_report(self.episode, [], [], outcome, self.sources)["events"])

    def test_end_to_end_combines_events_without_changing_source_or_training_policy(self):
        campaign = self.base / "campaigns/test"
        campaign.mkdir(parents=True)
        (campaign / "runs.tsv").write_text("lap\trun_name\tstatus\n1\trun\tcaptured\n", encoding="utf-8")
        self.write_json("outcome.json", {"full_mission_completed": True})
        records = [dict(self.progress(0), event="collision"),
                   {"event": "traffic_light_override", "receive_monotonic_ns": 101000000000,
                    "light_id": "L1", "training_visible": False, "intervention": True}]
        records += [self.progress(t) for t in range(2, 6)]
        records.append(self.progress(6, 0))
        (self.episode / "events.jsonl").write_text("\n".join(json.dumps(e) for e in records), encoding="utf-8")
        self.write_json("native_associations/associations.json", [
            {"index": i, "gps": {"receive_monotonic_ns": row["receive_ns"],
                                  "source_timestamp_ns": row["source_ns"]}}
            for i, row in self.sources["receipt"].items()])
        self.write_json("gt_timeline_v1/frames.json", [
            {"index": i, "ego_xy": row["xy"], "lidar_source_ns": row["source_ns"]}
            for i, row in self.sources["position"].items()])
        before = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        report = evaluation.evaluate_campaign(self.base, "test")
        lap = report["laps"][0]
        self.assertEqual([e["type"] for e in lap["events"]], ["collision", "traffic_light_override", "off_route"])
        self.assertEqual(lap["collisions"][0]["sensor_bundle_index"], lap["events"][0]["sensor_bundle_index"])
        self.assertEqual(lap["events"][1]["match_confidence"], "TIME_ONLY")
        for forbidden in ("usable_for_training", "bad_events", "EXCLUDE", "REJECT", "training_visible"):
            self.assertNotIn(forbidden, json.dumps(report))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
