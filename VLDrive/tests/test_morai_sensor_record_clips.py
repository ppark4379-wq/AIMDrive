import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.morai_sensor_record_clip_dataset import (
    CACHE_SCHEMA, CLIP_META_SCHEMA, CLIP_SCHEMA,
    SensorRecordVLDriveDataset, sha256_file)
from tools.build_sensor_record_clips import build_clips


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows),
                          encoding="utf-8")


class SensorRecordClipDatasetTests(unittest.TestCase):
    def test_notice_builder_keeps_navigation_clip_and_records_notice_ranges(self):
        from tools.build_notice_instructions import snap_zones
        from utils.notice_policy import RouteNoticePolicy
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "route.txt").write_text("".join("{} 0 0\n".format(i) for i in range(6)))
            write_json(root / "points.json", dict(map_id="test", global_path="route.txt", zones=[
                dict(id="static", label="static", type="static", text="Watch obstacles.",
                     start=[2, 0], end=[3, 0], color=[1, 0, 0])]))
            snap_zones(root / "points.json", root / "zones.json")
            notices = RouteNoticePolicy(root / "zones.json")
            class Navigation:
                points = [(float(i), 0.) for i in range(6)]
                def instruction(self, index):
                    return 4, "Follow the road."
            manifest = root / "manifest.json"
            manifest.write_text("sealed")
            entries = [dict(frame_id=i, sensor_time_ns=1_000_000_000 + i * 100_000_000,
                            navigation=dict(route_index=i)) for i in range(6)]
            document = dict(source_clock="sensor_record_simulation_time",
                            episode="/run/lap/episode_000", entries=entries)
            clips, _ = build_clips(dict(train=[(manifest, document)], val=[]), Navigation(), notices)
            self.assertEqual(len(clips), 1)
            self.assertEqual([(n["start_frame"], n["stop_frame"], n["text"])
                              for n in clips[0]["notice_ranges"]],
                             [(0, 1, ""), (2, 3, "Watch obstacles."), (4, 5, "")])

    def test_clip_instructions_ignore_teacher_fields(self):
        class Policy:
            @staticmethod
            def instruction(route_index):
                return (1, "turn left") if route_index >= 2 else (4, "follow")

        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "lap.json"
            manifest.write_text("sealed source identity\n", encoding="utf-8")
            entries = [{
                "frame_id": frame,
                "sensor_time_ns": 1_000_000_000 + frame * 100_000_000,
                "navigation": {"route_index": frame},
                "waypoints": [[frame, 999]],
                "objects": [{"teacher": frame}],
                "traffic_light": frame % 2,
                "ego_pose": {"future_teacher": frame},
            } for frame in range(5)]
            document = {"source_clock": "sensor_record_simulation_time",
                        "episode": "/data/run/lap/episode_000", "entries": entries}
            first, _ = build_clips({"train": [(manifest, document)],
                                    "val": [(manifest, document)]}, Policy())
            for entry in entries:
                entry["waypoints"] = [[-999, -999]]
                entry["objects"] = [{"different": True}]
                entry["traffic_light"] = -1
                entry["ego_pose"] = {"different": True}
            second, _ = build_clips({"train": [(manifest, document)],
                                     "val": [(manifest, document)]}, Policy())
            self.assertEqual(first, second)
            self.assertEqual([row["instruction"] for row in first[:2]],
                             ["follow", "turn left"])

    def test_windows_do_not_cross_time_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, clips = root / "cache", root / "clips"
            cache.mkdir(); clips.mkdir()
            rows = []
            stamp = 1_000_000_000
            for frame in range(15):
                if frame == 7:
                    stamp += 100_000_000
                feature = cache / "features" / ("%02d.npy" % frame)
                target = cache / "targets" / ("%02d.npy" % frame)
                feature.parent.mkdir(exist_ok=True); target.parent.mkdir(exist_ok=True)
                np.save(feature, np.full((106, 256), frame, dtype=np.float16), allow_pickle=False)
                np.save(target, np.full((5, 2), frame, dtype=np.float32), allow_pickle=False)
                rows.append({
                    "schema_version": CACHE_SCHEMA, "episode_id": "episode-a",
                    "frame_id": frame, "sensor_time_ns": stamp, "split": "train",
                    "visual_feature": {"path": feature.relative_to(cache).as_posix(),
                                       "shape": [106, 256], "dtype": "float16",
                                       "sha256": sha256_file(feature)},
                    "future_trajectory": {"path": target.relative_to(cache).as_posix(),
                                          "shape": [5, 2], "dtype": "float32",
                                          "horizons_s": [0.5, 1.0, 1.5, 2.0, 2.5],
                                          "source": "sealed_future_ego_positions",
                                          "sha256": sha256_file(target)},
                })
                stamp += 100_000_000
            manifest = cache / "manifest.jsonl"
            write_jsonl(manifest, rows)
            fingerprint = hashlib.sha256(b"same-publications").hexdigest()
            write_json(cache / "cache_meta.json", {
                "schema_version": CACHE_SCHEMA, "status": "complete",
                "publication_set_sha256": fingerprint, "checkpoint_sha256": "a" * 64,
                "feature_shape": [106, 256], "trajectory_shape": [5, 2],
                "manifest_sha256": sha256_file(manifest),
            })
            clip_rows = [{
                "schema_version": CLIP_SCHEMA, "clip_id": "clip-a",
                "episode_id": "episode-a", "split": "train",
                "start_frame_id": 0, "stop_frame_id": 14,
                "start_sensor_time_ns": rows[0]["sensor_time_ns"],
                "stop_sensor_time_ns": rows[-1]["sensor_time_ns"],
                "expected_period_ns": 100_000_000, "command": 4,
                "instruction": "Proceed along this route.",
            }]
            clip_manifest = clips / "clips.jsonl"
            write_jsonl(clip_manifest, clip_rows)
            write_json(clips / "clip_meta.json", {
                "schema_version": CLIP_META_SCHEMA, "status": "complete",
                "publication_set_sha256": fingerprint,
                "clips_sha256": sha256_file(clip_manifest),
            })

            dataset = SensorRecordVLDriveDataset(
                cache_root=cache, clip_index=clips, split="train", sequence_length=3)
            self.assertEqual(len(dataset), 11)
            self.assertEqual(dataset.exclusions["discontinuous"], 1)
            for index in range(len(dataset)):
                times = dataset[index]["metadata"]["sensor_time_ns"].numpy()
                self.assertTrue(np.all(np.diff(times) == 100_000_000))
            sample = dataset[0]
            self.assertEqual(tuple(sample["model_inputs"]["visual_feature"].shape),
                             (3, 106, 256))
            self.assertEqual(tuple(sample["training_targets"]["future_trajectory"].shape),
                             (3, 5, 2))
            self.assertEqual(sample["model_inputs"]["instruction"],
                             "Proceed along this route.")

            # Notice transitions must keep the entry window and exclude its old-frame losses.
            clip_rows[0].update(notice_zones_sha256="notice-zones", notice_ranges=[
                dict(start_frame=0, stop_frame=3, text="", zone_ids=[]),
                dict(start_frame=4, stop_frame=5, text="Watch obstacles.", zone_ids=["static"]),
                dict(start_frame=6, stop_frame=14, text="", zone_ids=[])])
            write_jsonl(clip_manifest, clip_rows)
            meta_path = clips / "clip_meta.json"
            meta = json.loads(meta_path.read_text())
            meta.update(notice_zones_sha256="notice-zones", clips_sha256=sha256_file(clip_manifest))
            write_json(meta_path, meta)
            dataset = SensorRecordVLDriveDataset(
                cache_root=cache, clip_index=clips, split="train", sequence_length=3)
            entry = dataset[2]  # [2, 3, 4]
            self.assertEqual(entry["model_inputs"]["notice_text"], "Watch obstacles.")
            self.assertEqual(entry["training_targets"]["frame_loss_mask"].tolist(), [False, False, True])
            exit_sample = dataset[4]  # [4, 5, 6]
            self.assertEqual(exit_sample["model_inputs"]["notice_text"], "")
            self.assertEqual(exit_sample["training_targets"]["frame_loss_mask"].tolist(), [False, False, True])

    def test_cache_and_clip_publications_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, clips = root / "cache", root / "clips"
            cache.mkdir(); clips.mkdir()
            manifest, clip_manifest = cache / "manifest.jsonl", clips / "clips.jsonl"
            write_jsonl(manifest, [])
            write_jsonl(clip_manifest, [])
            write_json(cache / "cache_meta.json", {
                "schema_version": CACHE_SCHEMA, "status": "complete",
                "publication_set_sha256": "a", "checkpoint_sha256": "b",
                "feature_shape": [106, 256], "trajectory_shape": [5, 2],
                "manifest_sha256": sha256_file(manifest),
            })
            write_json(clips / "clip_meta.json", {
                "schema_version": CLIP_META_SCHEMA, "status": "complete",
                "publication_set_sha256": "different",
                "clips_sha256": sha256_file(clip_manifest),
            })
            with self.assertRaisesRegex(ValueError, "different publications"):
                SensorRecordVLDriveDataset(cache_root=cache, clip_index=clips,
                                           split="train", sequence_length=3)


if __name__ == "__main__":
    unittest.main()
