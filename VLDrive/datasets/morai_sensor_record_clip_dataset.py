"""VLDrive temporal windows over sealed MORAI Sensor Record token caches.

The visual encoder runs once per frame in ``cache_sensor_record_tokens.py``.
This dataset groups those immutable tokens using semantic clips built only from
the recorded global path and the current route progress.  Object, traffic-light
and future-ego labels never participate in instruction generation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.notice_policy import current_condition_suffix


CACHE_SCHEMA = "morai-sensor-record-vision-cache-v1"
CLIP_SCHEMA = "morai-sensor-record-semantic-clip-v1"
CLIP_META_SCHEMA = "morai-sensor-record-semantic-clips-v1"
FEATURE_SHAPE = (106, 256)
TRAJECTORY_SHAPE = (5, 2)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def publication_set_sha256(publications):
    # Paths legitimately differ across PCs/NAS mounts. Content digests are the
    # portable identity; duplicate references to the same publication count once.
    return canonical_sha256(sorted({str(row["sha256"]) for row in publications}))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSONL at {}:{}".format(path, line_number)) from error
    return rows


def safe_artifact(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or root not in path.parents:
        raise ValueError("Unsafe cache artifact path: {}".format(relative))
    return path


class SensorRecordVLDriveDataset(Dataset):
    """Return fixed-length causal windows from semantic Sensor Record clips."""

    trajectory_coordinate_convention = "lmdrive_left_negative_forward"

    def __init__(self, *, cache_root, clip_index, split, sequence_length=11,
                 verify_hashes=True):
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.clip_root = Path(clip_index).expanduser().resolve()
        self.split = "val" if split in ("val", "validation") else str(split)
        if self.split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.verify_hashes = bool(verify_hashes)
        self._verified = set()

        cache_meta_path = self.cache_root / "cache_meta.json"
        cache_manifest_path = self.cache_root / "manifest.jsonl"
        clip_meta_path = self.clip_root / "clip_meta.json"
        clips_path = self.clip_root / "clips.jsonl"
        for path in (cache_meta_path, cache_manifest_path, clip_meta_path, clips_path):
            if not path.is_file():
                raise ValueError("Missing completed Sensor Record clip artifact: {}".format(path))
        self.cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        self.clip_meta = json.loads(clip_meta_path.read_text(encoding="utf-8"))
        self.notice_zones_sha256 = self.clip_meta.get("notice_zones_sha256")
        if self.cache_meta.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("Wrong Sensor Record visual cache schema")
        if self.clip_meta.get("schema_version") != CLIP_META_SCHEMA:
            raise ValueError("Wrong Sensor Record semantic clip schema")
        if self.cache_meta.get("status") != "complete" or self.clip_meta.get("status") != "complete":
            raise ValueError("Sensor Record cache/clip index is incomplete")
        if self.cache_meta.get("feature_shape") != list(FEATURE_SHAPE):
            raise ValueError("Visual cache feature shape mismatch")
        if self.cache_meta.get("trajectory_shape") != list(TRAJECTORY_SHAPE):
            raise ValueError("Visual cache trajectory shape mismatch")
        if self.cache_meta.get("manifest_sha256") != sha256_file(cache_manifest_path):
            raise ValueError("Visual cache manifest hash mismatch")
        if self.clip_meta.get("clips_sha256") != sha256_file(clips_path):
            raise ValueError("Semantic clip manifest hash mismatch")
        if (self.cache_meta.get("publication_set_sha256")
                != self.clip_meta.get("publication_set_sha256")):
            raise ValueError("Token cache and semantic clips came from different publications")

        entries = {}
        for row in read_jsonl(cache_manifest_path):
            if row.get("schema_version") != CACHE_SCHEMA:
                raise ValueError("Mixed Sensor Record token cache schema")
            if row.get("split") != self.split:
                continue
            key = (str(row["episode_id"]), int(row["frame_id"]))
            if key in entries:
                raise ValueError("Duplicate cached Sensor Record frame: {}".format(key))
            if int(row.get("sensor_time_ns", -1)) < 0:
                raise ValueError("Invalid cached sensor time")
            feature_ref, trajectory_ref = row.get("visual_feature", {}), row.get("future_trajectory", {})
            if (feature_ref.get("shape") != list(FEATURE_SHAPE)
                    or feature_ref.get("dtype") != "float16"):
                raise ValueError("Cached feature row contract mismatch")
            if (trajectory_ref.get("shape") != list(TRAJECTORY_SHAPE)
                    or trajectory_ref.get("dtype") != "float32"
                    or trajectory_ref.get("horizons_s") != [0.5, 1.0, 1.5, 2.0, 2.5]
                    or trajectory_ref.get("source") != "sealed_future_ego_positions"):
                raise ValueError("Cached trajectory row contract mismatch")
            entries[key] = row
        if not entries:
            raise ValueError("Token cache has no frames for split {}".format(self.split))
        self.entries = entries

        clips = []
        clip_ids = set()
        for row in read_jsonl(clips_path):
            if row.get("schema_version") != CLIP_SCHEMA:
                raise ValueError("Mixed semantic clip schema")
            if row.get("split") != self.split:
                continue
            clip_id = str(row["clip_id"])
            if clip_id in clip_ids:
                raise ValueError("Duplicate semantic clip: {}".format(clip_id))
            clip_ids.add(clip_id)
            if int(row["start_frame_id"]) > int(row["stop_frame_id"]):
                raise ValueError("Invalid semantic clip frame range")
            if int(row["start_sensor_time_ns"]) > int(row["stop_sensor_time_ns"]):
                raise ValueError("Invalid semantic clip time range")
            if not str(row.get("instruction", "")).strip():
                raise ValueError("Semantic clip has no instruction")
            if bool("notice_ranges" in row) != bool(self.notice_zones_sha256):
                raise ValueError("Mixed notice-enabled and navigation-only clips")
            if self.notice_zones_sha256:
                if row.get("notice_zones_sha256") != self.notice_zones_sha256:
                    raise ValueError("Semantic clip notice zone hash mismatch")
                next_frame = int(row["start_frame_id"])
                for notice in row["notice_ranges"]:
                    if (notice["start_frame"] != next_frame
                            or notice["stop_frame"] < next_frame
                            or not isinstance(notice["text"], str)
                            or not isinstance(notice["zone_ids"], list)
                            or bool(notice["text"]) != bool(notice["zone_ids"])):
                        raise ValueError("Invalid or discontinuous clip notice ranges")
                    next_frame = notice["stop_frame"] + 1
                if next_frame != int(row["stop_frame_id"]) + 1:
                    raise ValueError("Incomplete clip notice coverage")
            clips.append(row)
        if not clips:
            raise ValueError("No semantic clips for split {}".format(self.split))

        self.windows = []
        self.exclusions = {"missing_cached_frame": 0, "discontinuous": 0,
                           "short_clip": 0}
        used_frames = set()
        for clip in clips:
            episode_id = str(clip["episode_id"])
            rows = [entries[(episode_id, frame)]
                    for frame in range(int(clip["start_frame_id"]),
                                       int(clip["stop_frame_id"]) + 1)
                    if (episode_id, frame) in entries]
            expected_count = int(clip["stop_frame_id"]) - int(clip["start_frame_id"]) + 1
            self.exclusions["missing_cached_frame"] += expected_count - len(rows)
            rows.sort(key=lambda row: int(row["frame_id"]))

            runs = []
            current = []
            period = int(clip["expected_period_ns"])
            tolerance = max(5_000_000, int(period * 0.1))
            for row in rows:
                if current:
                    before = current[-1]
                    delta = int(row["sensor_time_ns"]) - int(before["sensor_time_ns"])
                    if (int(row["frame_id"]) != int(before["frame_id"]) + 1
                            or abs(delta - period) > tolerance):
                        self.exclusions["discontinuous"] += 1
                        runs.append(current)
                        current = []
                current.append(row)
            if current:
                runs.append(current)
            made_window = False
            for run in runs:
                if len(run) < self.sequence_length:
                    continue
                for end in range(self.sequence_length, len(run) + 1):
                    window = run[end - self.sequence_length:end]
                    self.windows.append((clip, window))
                    made_window = True
                    used_frames.update((episode_id, int(row["frame_id"])) for row in window)
            if not made_window:
                self.exclusions["short_clip"] += 1
        if not self.windows:
            raise ValueError("No valid consecutive Sensor Record VLDrive windows")
        self.used_frame_count = len(used_frames)
        self.checkpoint_sha256 = str(self.cache_meta["checkpoint_sha256"])
        self.cache_meta_sha256 = sha256_file(cache_meta_path)
        self.clip_meta_sha256 = sha256_file(clip_meta_path)

    def __len__(self):
        return len(self.windows)

    def _array(self, ref, shape, label):
        path = safe_artifact(self.cache_root, ref["path"])
        if self.verify_hashes and path not in self._verified:
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("{} artifact hash mismatch: {}".format(label, path))
            self._verified.add(path)
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != shape or not np.isfinite(array).all():
            raise ValueError("Invalid {} array: {}".format(label, path))
        return torch.from_numpy(np.asarray(array, dtype=np.float32))

    def __getitem__(self, index):
        clip, rows = self.windows[int(index)]
        visual = torch.stack([self._array(row["visual_feature"], FEATURE_SHAPE,
                                                "visual feature") for row in rows])
        trajectory = torch.stack([self._array(row["future_trajectory"], TRAJECTORY_SHAPE,
                                                    "future trajectory") for row in rows])
        sample = {
            "model_inputs": {
                "visual_feature": visual,
                "instruction": str(clip["instruction"]),
            },
            "training_targets": {"future_trajectory": trajectory},
            "metadata": {
                "run_id": str(clip["episode_id"]),
                "clip_id": str(clip["clip_id"]),
                "sample_ids": ["{}/{}".format(row["episode_id"], row["frame_id"])
                               for row in rows],
                "frame_ids": torch.tensor([int(row["frame_id"]) for row in rows],
                                           dtype=torch.int64),
                "sensor_time_ns": torch.tensor([int(row["sensor_time_ns"]) for row in rows],
                                                dtype=torch.int64),
                "command": int(clip["command"]),
                "input_mode": "cached",
                "purpose": self.split,
            },
        }
        if self.notice_zones_sha256:
            def notice_for(frame):
                return next(n for n in clip["notice_ranges"]
                            if n["start_frame"] <= frame <= n["stop_frame"])
            notices = [notice_for(int(row["frame_id"])) for row in rows]
            sample["model_inputs"]["notice_text"] = notices[-1]["text"]
            begin = current_condition_suffix([n["text"] for n in notices])
            sample["training_targets"]["frame_loss_mask"] = torch.arange(len(rows)) >= begin
            sample["metadata"]["notice_zone_ids"] = "|".join(notices[-1]["zone_ids"])
        return sample

    def describe(self):
        description = {
            "schema_version": "morai-sensor-record-vldrive-dataset-v1",
            "purpose": self.split,
            "input_mode": "cached",
            "sequence_length": self.sequence_length,
            "cached_frames": len(self.entries),
            "used_frames": self.used_frame_count,
            "windows": len(self.windows),
            "excluded": dict(self.exclusions),
            "checkpoint_sha256": self.checkpoint_sha256,
            "cache_meta_sha256": self.cache_meta_sha256,
            "clip_meta_sha256": self.clip_meta_sha256,
            "publication_set_sha256": self.cache_meta["publication_set_sha256"],
            "feature_contract": list(FEATURE_SHAPE),
            "trajectory_source": "sealed future Ego positions at 0.5/1.0/1.5/2.0/2.5 s",
            "instruction_source": "sealed official global path plus current route_index",
            "continuity_source": "frame_id and sensor_record_simulation_time",
            "model_inputs": ["cached visual feature", "navigation instruction"],
            "teacher_input_exclusions": ["objects", "traffic_light", "future waypoints", "Ego_topic"],
        }
        if self.notice_zones_sha256:
            description["notice_zones_sha256"] = self.notice_zones_sha256
            description["model_inputs"].append("route notice instruction")
            description["waypoint_supervision_policy"] = "current_navigation_notice_contiguous_suffix"
        return description
