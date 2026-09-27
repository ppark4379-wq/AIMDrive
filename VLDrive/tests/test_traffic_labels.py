import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders.split_manifest import build_chronological_split
from data.loaders.traffic_labels import (
    build_traffic_label_sidecar, canonical_sha256, checked_traffic_target,
    load_traffic_label_sidecar, sha256_file, sidecar_summary, traffic_record,
)


def measurement(frame=0, state=1, affecting="signal_a", lights=None):
    if lights is None:
        lights = [{"tl_id": affecting, "state": state}] if state is not None and affecting else []
    return {
        "frame_id": frame, "native_sync_frame": 100 + frame * 5,
        "grpc_label_tick_raw": 100 + frame * 5, "simulator_labels_source": "grpc",
        "navigation": {"route_valid": True, "route_index": frame,
                       "affected_light_id_raw": affecting, "affected_light_state_raw": state},
        "traffic_lights": lights,
        "is_red_light_present": [123] if state is not None and state & 1 else [],
        "theta": 0.0, "gps_x": frame * 0.2, "gps_y": 1.0,
        "x_command": frame * 0.2 + 15.0, "y_command": 1.0,
        "speed": 2.0, "gt_command": 4, "command": 4,
    }


class TrafficPolicyTests(unittest.TestCase):
    def test_known_red_and_green_preserve_label(self):
        for state, label in [(1, 0), (16, 1), (48, 1)]:
            row = traffic_record("route", "train", 0, measurement(state=state))
            self.assertEqual(row["traffic_light"], label)
            self.assertTrue(row["traffic_available"])

    def test_unknown_and_id_mismatch_are_masked_not_relabelled(self):
        for lights in [[], [{"tl_id": "other", "state": 1}]]:
            row = traffic_record("route", "train", 0, measurement(state=None, lights=lights))
            self.assertEqual(row["traffic_light"], 1)
            self.assertFalse(row["traffic_available"])
            self.assertEqual(row["reason"], "unknown_affected_signal")

    def test_unverified_nonnull_state_is_also_masked(self):
        row = traffic_record("route", "train", 0, measurement(lights=[{"tl_id": "other", "state": 1}]))
        self.assertFalse(row["traffic_available"])
        self.assertEqual(row["reason"], "unverified_affected_signal")

    def test_red_green_composites_are_not_forced_go(self):
        for state in [33, 49]:
            row = traffic_record("route", "train", 0, measurement(state=state))
            self.assertEqual(row["traffic_light"], 0)
            self.assertFalse(row["traffic_available"])
            self.assertEqual(row["reason"], "ambiguous_red_green_permission")

    def test_no_affected_signal_is_distinct_from_unknown(self):
        row = traffic_record("route", "train", 0, measurement(state=None, affecting=None))
        self.assertTrue(row["traffic_available"])
        self.assertEqual(row["reason"], "no_affected_signal_preserved")

    def test_distant_red_is_not_changed_to_green(self):
        source = measurement()
        source["navigation"]["distance_to_stopline"] = 101.45
        row = traffic_record("route", "train", 0, source)
        self.assertEqual(row["traffic_light"], 0)
        self.assertTrue(row["traffic_available"])

    def test_tick_mismatch_cannot_be_available(self):
        source = measurement()
        source["grpc_label_tick_raw"] -= 1
        row = traffic_record("route", "train", 0, source)
        self.assertFalse(row["traffic_available"])
        self.assertEqual(row["reason"], "unaligned_or_non_grpc")


class TrafficSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.route = self.root / "data" / "route_a"
        self.route.mkdir(parents=True)
        (self.root / "dataset_index.txt").write_text("data/route_a/ 80\n", encoding="utf-8")
        self.measurements = [measurement(i) for i in range(80)]
        self.measurements[0] = measurement(0, state=None)
        self.measurements[1] = measurement(1, state=33)
        self.aggregate = self.route / "measurements_all.json"
        self.write(self.aggregate, self.measurements)
        self.write(self.route / "route.json", {"name": "route_a"})
        self.split = self.root / "split.json"
        self.write(self.split, build_chronological_split(self.root))
        self.path = self.root / "traffic_labels_v2.json"
        self.artifact = build_traffic_label_sidecar(self.root, self.split)
        self.write(self.path, self.artifact)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def load(self):
        return load_traffic_label_sidecar(self.path, self.root, self.split)

    def test_reproducible_and_source_immutable(self):
        before = {p: sha256_file(p) for p in [self.aggregate, self.split, self.route / "route.json"]}
        self.assertEqual(self.artifact, build_traffic_label_sidecar(self.root, self.split))
        records, provenance = self.load()
        self.assertEqual(provenance["sidecar_sha256"], sha256_file(self.path))
        self.assertEqual(before, {p: sha256_file(p) for p in before})
        self.assertEqual(checked_traffic_target(records, "route_a", 0, self.measurements[0]), (1, False))
        summary = sidecar_summary(self.artifact)
        self.assertEqual(summary["newly_masked"], 2)
        self.assertEqual(summary["relabeled"], 0)

    def test_tampered_label_mask_tick_reason_or_hash_fails_even_if_digest_updated(self):
        changes = [("traffic_light", 0), ("traffic_available", True),
                   ("native_sync_frame", 105), ("reason", "known_affected_signal_preserved"),
                   ("measurement_sha256", "0" * 64), ("traffic_available", 0)]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                altered = copy.deepcopy(self.artifact)
                altered["records"][0][field] = value
                altered["records_sha256"] = canonical_sha256(altered["records"])
                self.write(self.path, altered)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.load()

    def test_missing_or_duplicate_frame_fails(self):
        for mode in ("missing", "duplicate"):
            altered = copy.deepcopy(self.artifact)
            if mode == "missing":
                altered["records"].pop()
            else:
                altered["records"].append(altered["records"][0])
            self.write(self.path, altered)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.load()

    def test_changed_aggregate_source_fails(self):
        self.measurements[0]["speed"] = 99.0
        self.write(self.aggregate, self.measurements)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.load()

    def test_changed_route_source_fails(self):
        self.write(self.route / "route.json", {"name": "changed"})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.load()

    def test_changed_split_bytes_fail(self):
        self.split.write_text(self.split.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.load()

    def test_per_frame_source_or_missing_frame_fails(self):
        records, _ = self.load()
        altered = copy.deepcopy(self.measurements[0])
        altered["grpc_label_tick_raw"] += 5
        with self.assertRaisesRegex(ValueError, "per-frame source mismatch"):
            checked_traffic_target(records, "route_a", 0, altered)
        with self.assertRaisesRegex(ValueError, "Missing"):
            checked_traffic_target(records, "route_a", 999, altered)

    def test_policy_or_schema_mismatch_fails(self):
        for field in ["schema_version", "policy"]:
            altered = copy.deepcopy(self.artifact)
            altered[field] = "wrong"
            self.write(self.path, altered)
            with self.assertRaisesRegex(ValueError, "schema/policy"):
                self.load()

    def test_real_loader_changes_only_traffic_mask_not_inputs_or_future_ego_gt(self):
        import numpy as np
        import torch
        from PIL import Image
        from data.loaders.morai_perception_dataset import MoraiPerceptionDataset

        for name in ("front", "left", "right", "rear"):
            folder = self.route / ("rgb_" + name)
            folder.mkdir()
            Image.new("RGB", (8, 8), (11, 22, 33)).save(folder / "0000.jpg")
        (self.route / "measurements_full").mkdir()
        full = self.route / "measurements_full" / "0000.json"
        self.write(full, self.measurements[0])
        (self.route / "lidar").mkdir()
        np.save(self.route / "lidar" / "0000.npy", np.zeros((4, 4), dtype=np.float32))
        plain = MoraiPerceptionDataset(root=self.root, split_file=self.split)
        corrected = MoraiPerceptionDataset(root=self.root, split_file=self.split,
                                          traffic_label_sidecar=self.path)
        for ds in [plain, corrected]:
            ds.rgb_transform = lambda image: torch.from_numpy(np.asarray(image).copy())
            ds.rgb_center_transform = ds.rgb_transform
            ds.multi_view_transform = ds.rgb_transform
        with mock.patch("data.loaders.morai_perception_dataset.lidar_to_raw_features",
                        side_effect=lambda array, **kwargs: (array, len(array))):
            old_input, old_target = plain[0]
            new_input, new_target = corrected[0]
            self.assertTrue(old_target["traffic_available"].item())
            self.assertFalse(new_target["traffic_available"].item())
            for key in old_target:
                if key != "traffic_available":
                    self.assertTrue(torch.equal(old_target[key], new_target[key]), key)
            for key in old_input:
                np.testing.assert_array_equal(old_input[key], new_input[key], err_msg=key)
            self.assertGreater(float(new_target["waypoints"].abs().max()), 0)
            self.assertIsNone(plain.traffic_label_provenance)
            self.assertEqual(corrected.traffic_label_provenance["sidecar_sha256"], sha256_file(self.path))
            altered = copy.deepcopy(self.measurements[0])
            altered["speed"] += 1
            self.write(full, altered)
            with self.assertRaisesRegex(ValueError, "per-frame source mismatch"):
                corrected[0]


if __name__ == "__main__":
    unittest.main()
