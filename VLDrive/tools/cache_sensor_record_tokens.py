#!/usr/bin/env python3
"""Cache final visual-encoder tokens for sealed Sensor Record publications."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import configure_runtime
configure_runtime()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.morai_sensor_record_dataset import (
    PathResolver, SensorRecordDataset, digest, load_publications)
from datasets.morai_sensor_record_clip_dataset import (
    CACHE_SCHEMA, FEATURE_SHAPE, TRAJECTORY_SHAPE,
    publication_set_sha256, sha256_file)
from models.vision_encoder import create_vision_encoder


FEATURE_MODEL = "memfuser_baseline_e1d3_return_feature"


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2,
                                  allow_nan=False, default=str) + "\n", encoding="utf-8")
    pending.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def write_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    pending.replace(path)


def relative(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def resolve_inputs(args):
    config_path = args.config or ROOT / "configs/morai_transfer.local.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    manifests = list(args.manifest_index or config.get("manifest_indices", []))
    if not manifests and config.get("manifest_index"):
        manifests.append(config["manifest_index"])
    campaigns = list(config.get("manifest_campaigns", [])) if args.manifest_index is None else []
    campaigns += list(args.manifest_campaign)
    for campaign in campaigns:
        campaign = Path(campaign).expanduser()
        candidates = sorted(campaign.glob("validation_*/training_dataset*/manifest_index.json"),
                            key=lambda path: (path.stat().st_mtime_ns, str(path)))
        if not candidates:
            raise ValueError("Campaign has no sealed training publication: {}".format(campaign))
        manifests.append(candidates[-1])
    checkpoint = args.checkpoint or config.get("checkpoint")
    checkpoint_sha = args.checkpoint_sha256 or config.get("checkpoint_sha256")
    if not manifests or not checkpoint or not checkpoint_sha:
        raise ValueError("Set publications and the final visual checkpoint with SHA256")
    resolver = PathResolver(config.get("path_mappings", []))
    return ([Path(path).expanduser().resolve() for path in manifests], resolver,
            Path(checkpoint).expanduser().resolve(), str(checkpoint_sha))


def load_feature_model(path, expected_sha256, device):
    if digest(path) != expected_sha256:
        raise ValueError("Visual checkpoint SHA256 mismatch")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if (checkpoint.get("feature_model_name") != FEATURE_MODEL
            or checkpoint.get("token_shape") != list(FEATURE_SHAPE)
            or not isinstance(checkpoint.get("state_dict"), dict)):
        raise ValueError("Checkpoint does not declare the trained [106,256] feature contract")
    model = create_vision_encoder({"name": FEATURE_MODEL}, pretrained_backbone=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def record_valid(path, output, expected=None):
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != CACHE_SCHEMA:
            return False
        if expected is not None:
            episode_id, frame_id, sensor_time_ns, split = expected
            if (str(row.get("episode_id")) != str(episode_id)
                    or int(row.get("frame_id", -1)) != int(frame_id)
                    or int(row.get("sensor_time_ns", -1)) != int(sensor_time_ns)
                    or row.get("split") != split):
                return False
        for key, shape in (("visual_feature", FEATURE_SHAPE),
                           ("future_trajectory", TRAJECTORY_SHAPE)):
            ref = row[key]
            artifact = (output / ref["path"]).resolve()
            if output.resolve() not in artifact.parents or not artifact.is_file():
                return False
            if sha256_file(artifact) != ref["sha256"]:
                return False
            array = np.load(artifact, allow_pickle=False, mmap_mode="r")
            if tuple(array.shape) != shape or not np.isfinite(array).all():
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def cache_split(args, split, manifests, resolver, model, device, completed):
    dataset = SensorRecordDataset(manifests, resolver, velocity_source="gps")
    identity_by_native = {}
    for manifest_path, document in manifests:
        native = str(resolver(document["native_root"], manifest_path.parent))
        episode = resolver(document["episode"], manifest_path.parent)
        episode_id = "{}_{}".format(Path(episode).parent.name,
                                     digest(manifest_path)[:12])
        if native in identity_by_native:
            raise ValueError("Duplicate native root in publication split")
        identity_by_native[native] = episode_id
    identities = []
    for native, entry, _, _ in dataset.rows:
        episode_id = identity_by_native.get(str(Path(native).resolve()))
        if episode_id is None:
            raise ValueError("Dataset row has no source episode identity")
        identities.append((episode_id, int(entry["frame_id"]), int(entry["sensor_time_ns"])))
    missing_indices = []
    for index, (episode_id, frame_id, _) in enumerate(identities):
        record = args.output / "records" / episode_id / ("{:06d}.json".format(frame_id))
        if record_valid(record, args.output,
                        (episode_id, frame_id, identities[index][2], split)):
            completed.add((episode_id, frame_id))
        else:
            missing_indices.append(index)
    if not missing_indices:
        return len(dataset)
    loader = DataLoader(Subset(dataset, missing_indices),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    cursor = 0
    for inputs, targets in loader:
        count = int(inputs["rgb_front"].shape[0])
        source_indices = missing_indices[cursor:cursor + count]
        batch_identities = [identities[index] for index in source_indices]
        cursor += count
        model_inputs = {key: value.to(device, non_blocking=True)
                        for key, value in inputs.items()}
        autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda" else nullcontext())
        with torch.inference_mode(), autocast:
            features = model(model_inputs)
        features = features.float().cpu().numpy()
        trajectories = targets["waypoints"].numpy()
        if tuple(features.shape[1:]) != FEATURE_SHAPE or not np.isfinite(features).all():
            raise RuntimeError("Invalid frozen visual token output")
        for result_index, (episode_id, frame_id, sensor_time_ns) in enumerate(batch_identities):
            feature = features[result_index].astype(np.float16)
            trajectory = np.asarray(trajectories[result_index], dtype=np.float32)
            if tuple(trajectory.shape) != TRAJECTORY_SHAPE or not np.isfinite(trajectory).all():
                raise RuntimeError("Invalid future trajectory cache target")
            feature_path = args.output / "features" / episode_id / ("{:06d}.npy".format(frame_id))
            target_path = args.output / "targets" / episode_id / ("{:06d}.npy".format(frame_id))
            write_npy(feature_path, feature)
            write_npy(target_path, trajectory)
            row = {
                "schema_version": CACHE_SCHEMA,
                "episode_id": episode_id,
                "frame_id": frame_id,
                "sensor_time_ns": sensor_time_ns,
                "split": split,
                "visual_feature": {
                    "path": relative(feature_path, args.output),
                    "shape": list(FEATURE_SHAPE), "dtype": "float16",
                    "sha256": sha256_file(feature_path),
                },
                "future_trajectory": {
                    "path": relative(target_path, args.output),
                    "shape": list(TRAJECTORY_SHAPE), "dtype": "float32",
                    "horizons_s": [0.5, 1.0, 1.5, 2.0, 2.5],
                    "source": "sealed_future_ego_positions",
                    "sha256": sha256_file(target_path),
                },
            }
            record = args.output / "records" / episode_id / ("{:06d}.json".format(frame_id))
            record.parent.mkdir(parents=True, exist_ok=True)
            write_json(record, row)
            completed.add((episode_id, frame_id))
            if len(completed) == 1 or len(completed) % args.log_interval == 0:
                print("cached frames={}".format(len(completed)), flush=True)
    if cursor != len(missing_indices):
        raise RuntimeError("Token cache loader skipped missing Sensor Record frames")
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest-index", action="append", type=Path)
    parser.add_argument("--manifest-campaign", action="append", type=Path, default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.log_interval) < 1 or args.workers < 0:
        raise ValueError("Invalid batch/worker/log settings")
    indexes, resolver, checkpoint, checkpoint_sha = resolve_inputs(args)
    publication_info, manifests = load_publications(indexes, resolver)
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and (args.output / "cache_meta.json").exists():
        raise FileExistsError("Completed token cache already exists: {}".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": CACHE_SCHEMA,
        "status": "partial",
        "publication_set_sha256": publication_set_sha256(publication_info["publications"]),
        "publications": publication_info["publications"],
        "checkpoint_sha256": checkpoint_sha,
        "feature_model": FEATURE_MODEL,
        "feature_shape": list(FEATURE_SHAPE),
        "trajectory_shape": list(TRAJECTORY_SHAPE),
        "velocity_source": "causal_gps_interval",
        "target_point_source": "sealed_official_route_25m_lookahead",
        "source_clock": "sensor_record_simulation_time",
        "implementation_sha256": {
            "tools/cache_sensor_record_tokens.py": digest(Path(__file__)),
            "datasets/morai_sensor_record_dataset.py": digest(
                ROOT / "datasets/morai_sensor_record_dataset.py"),
            "datasets/morai_sensor_record_clip_dataset.py": digest(
                ROOT / "datasets/morai_sensor_record_clip_dataset.py"),
            "models/vision_encoder.py": digest(ROOT / "models/vision_encoder.py"),
        },
    }
    partial = args.output / "cache_meta.partial.json"
    if partial.is_file():
        existing = json.loads(partial.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("Partial cache belongs to different inputs/checkpoint/code")
    else:
        write_json(partial, identity)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    model = load_feature_model(checkpoint, checkpoint_sha, device)
    completed, expected = set(), 0
    for split in ("train", "val"):
        expected += cache_split(args, split, manifests[split], resolver,
                                model, device, completed)
    rows = []
    for path in sorted((args.output / "records").glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (str(row["episode_id"]), int(row["frame_id"])) in completed:
            rows.append(row)
    rows.sort(key=lambda row: (row["split"], row["episode_id"], int(row["frame_id"])))
    if len(rows) != expected or len(completed) != expected:
        raise RuntimeError("Token cache completeness mismatch: rows={} completed={} expected={}".format(
            len(rows), len(completed), expected))
    manifest = args.output / "manifest.jsonl"
    write_jsonl(manifest, rows)
    final = dict(identity)
    final.update(status="complete", frames=expected,
                 manifest_sha256=sha256_file(manifest))
    write_json(args.output / "cache_meta.json", final)
    partial.unlink()
    print(json.dumps({"output": str(args.output), "frames": expected,
                      "checkpoint_sha256": checkpoint_sha}, indent=2))


if __name__ == "__main__":
    main()
