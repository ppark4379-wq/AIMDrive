#!/usr/bin/env python3
"""
Stage 6D-2A
-----------
MORAI recorded Camera + VLP16
    -> frozen public LMDrive vision encoder
    -> cached F_i [106,256]

This consumes the *fixed* Stage 6D-1 v2 trajectory-label manifest.

Important runtime-parity rule
-----------------------------
/Ego_topic is training-only teacher data.  It is NOT fed to the frozen visual
encoder.  Until the runtime speed source is resolved, LMDrive's visual
encoder velocity input remains an explicit 0.0 placeholder.

No performance claim is made by this cache step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from aimdrive.model.morai_lmdrive_vision_adapter import (
    MoraiLMDriveVisionAdapter,
    MoraiVisionAdapterConfig,
    load_frozen_lmdrive_vision_encoder,
)


REQUIRED_LABEL_SCHEMA = "morai-stage6d1-ego-teacher-v2"
REQUIRED_LABEL_TIMEBASE = "ego_source_stamp_ns"


def load_records(path, max_frames):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            labels = row.get("labels") or {}
            if labels.get("schema_version") != REQUIRED_LABEL_SCHEMA:
                raise RuntimeError(
                    "Stage 6D-2 requires fixed Stage 6D-1 v2 labels; "
                    "got schema={!r} frame={}".format(
                        labels.get("schema_version"),
                        row.get("frame_index"),
                    )
                )
            if labels.get("trajectory_timebase") != REQUIRED_LABEL_TIMEBASE:
                raise RuntimeError(
                    "Stage 6D-2 requires trajectory_timebase={!r}; got {!r}".format(
                        REQUIRED_LABEL_TIMEBASE,
                        labels.get("trajectory_timebase"),
                    )
                )
            rows.append(row)
            if max_frames > 0 and len(rows) >= max_frames:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6e", action="store_true", help="Require validated Stage6E run, all frames and original encoder contract")
    parser.add_argument("--allow-legacy-offline", action="store_true", help="Explicit offline regression only; no automatic training promotion")
    parser.add_argument("--lidar-axis-policy", choices=("morai_flu_to_lmdrive_fru", "legacy_passthrough"),
                        default="morai_flu_to_lmdrive_fru")
    parser.add_argument(
        "--dataset",
        default="data/stage6d1_teacher_smoke",
    )
    parser.add_argument(
        "--manifest",
        default="manifest_stage6d1_labels.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "~/workspace/lmdrive_assets/weights/"
            "LMDrive-vision-encoder-r50-v1.0/"
            "vision-encoder-r50.pth.tar"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="visual_features_stage6d2",
    )
    parser.add_argument(
        "--output-manifest",
        default="manifest_stage6d2_features.jsonl",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--preprocessing",
        choices=("lmdrive_compat", "morai_full_fov"),
        default="lmdrive_compat",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--velocity-mps",
        type=float,
        default=0.0,
        help=(
            "Runtime-parity placeholder only. Do NOT replace with /Ego_topic "
            "teacher speed just because it is available during training."
        ),
    )
    parser.add_argument(
        "--single-sweep",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = Path(args.dataset).expanduser().resolve()
    if ROOT not in dataset.parents:
        parser.error("Dataset output must remain inside this repository")
    if args.stage6e:
        if args.max_frames != 0 or args.velocity_mps != 0 or args.preprocessing != "lmdrive_compat":
            parser.error("Stage6E requires --max-frames 0, velocity 0 and lmdrive_compat")
        from aimdrive.data.stage6e_validation import validate_run
        quality = validate_run(dataset, allow_legacy=args.allow_legacy_offline)
        if quality["quality_status"] != "valid" and not (args.allow_legacy_offline and quality["quality_status"]=="warning"):
            raise RuntimeError("Rejected Stage6E input: " + str(quality["reasons"]))
    manifest = dataset / args.manifest
    rows = load_records(manifest, args.max_frames)
    if not rows:
        raise RuntimeError("No labeled rows found")

    out_dir = dataset / args.output_dir
    out_manifest = dataset / args.output_manifest
    stats_path = dataset / "feature_cache_stats.json"
    partial = out_manifest.with_suffix(out_manifest.suffix+".partial")
    for path in (out_dir, out_manifest, partial, stats_path):
        if dataset not in path.resolve().parents or path.exists():
            raise RuntimeError("Refusing existing or out-of-run feature output: " + str(path))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    checkpoint_sha = digest.hexdigest()

    adapter = MoraiLMDriveVisionAdapter(
        MoraiVisionAdapterConfig(
            preprocessing=args.preprocessing,
            lidar_axis_policy=args.lidar_axis_policy,
            use_previous_lidar_sweep=(not args.single_sweep),
        )
    )

    encoder, load_result = load_frozen_lmdrive_vision_encoder(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("Encoder is not frozen/eval")
    out_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 108)
    print("MORAI Stage 6D-2A — FROZEN LMDRIVE FEATURE CACHE")
    print("=" * 108)
    print("dataset              :", dataset)
    print("manifest             :", manifest)
    print("strict checkpoint    :", load_result)
    print("frames requested     :", len(rows))
    print("preprocessing        :", args.preprocessing)
    print("two-sweep LiDAR      :", not args.single_sweep)
    print("vision velocity      :", args.velocity_mps, "(runtime-parity placeholder)")
    print("cache dtype          :", args.cache_dtype)
    print("label timebase       :", REQUIRED_LABEL_TIMEBASE)
    print("=" * 108)

    previous_lidar = None
    previous_frame_index = None
    elapsed = []
    written = 0

    np_dtype = np.float16 if args.cache_dtype == "float16" else np.float32

    with partial.open("x", encoding="utf-8") as output:
        for row_index, row in enumerate(rows):
            frame_index = int(row["frame_index"])

            # Never concatenate a previous sweep across a discontinuity.
            if (
                previous_frame_index is None
                or frame_index != previous_frame_index + 1
            ):
                previous_lidar = None

            model_input, current_lidar, metadata = adapter.build_model_input(
                dataset_root=dataset,
                record=row,
                previous_lidar_xyzi=previous_lidar,
                velocity_mps=args.velocity_mps,
                device=args.device,
            )

            if not args.single_sweep:
                previous_lidar = current_lidar
            previous_frame_index = frame_index

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=args.device.startswith("cuda")
                ):
                    feature = encoder(model_input)

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - start) * 1000.0
            elapsed.append(forward_ms)

            feature = feature.detach().float().cpu()
            if tuple(feature.shape) != (1, 106, 256):
                raise RuntimeError(
                    "Expected frozen LMDrive feature [1,106,256], got {}".format(
                        tuple(feature.shape)
                    )
                )
            if not bool(torch.isfinite(feature).all()):
                raise RuntimeError("Non-finite frozen visual feature")

            feature_np = feature[0].numpy().astype(np_dtype, copy=False)
            if not np.isfinite(feature_np).all():
                raise RuntimeError("Feature overflow after cache dtype conversion")
            rel = Path(args.output_dir) / "{:08d}.npy".format(frame_index)
            np.save(dataset / rel, feature_np, allow_pickle=False)
            if not np.array_equal(np.load(dataset / rel, allow_pickle=False), feature_np):
                raise RuntimeError("Feature cache read/write roundtrip mismatch")

            enriched = dict(row)
            enriched["visual_feature"] = {
                "schema_version": "morai-stage6d2-frozen-lmdrive-v1",
                "path": str(rel),
                "shape": [106, 256],
                "dtype": args.cache_dtype,
                "source": "frozen_lmdrive_r50",
                "checkpoint_sha256": checkpoint_sha,
                "trainable_encoder": False,
                "preprocessing": args.preprocessing,
                "two_lidar_sweeps": bool(not args.single_sweep),
                "lidar_rotation_deg": 0.0,
                "lidar_axis_policy": args.lidar_axis_policy,
                "velocity_mps": float(args.velocity_mps),
                "velocity_source": "runtime-parity-placeholder-zero" if args.velocity_mps==0 else "explicit-constant",
                "num_points": int(metadata["num_points"]),
                "forward_ms": float(forward_ms),
            }
            output.write(json.dumps(enriched, separators=(",", ":")) + "\n")
            written += 1

            if row_index < 3 or (row_index + 1) % 10 == 0:
                print(
                    "frame {:08d}  feature={}  points={}  {:.2f} ms".format(
                        frame_index,
                        tuple(feature_np.shape),
                        metadata["num_points"],
                        forward_ms,
                    )
                )

    arr = np.asarray(elapsed, dtype=np.float64)
    with stats_path.open("x") as handle:
        json.dump(dict(schema_version="stage6e-feature-cache-v1", frames=written, checkpoint=str(checkpoint),
                       checkpoint_sha256=checkpoint_sha, strict_load=str(load_result), frozen=True, eval=True,
                       shape=[106,256], dtype=args.cache_dtype, preprocessing=args.preprocessing,
                       velocity_source="runtime-parity-placeholder-zero", lidar_rotation_deg=0,
                       lidar_axis_policy=args.lidar_axis_policy,
                       roundtrip_verified=written,
                       forward_ms=dict(median=float(np.median(arr)), p95=float(np.percentile(arr,95)))), handle, indent=2)
    partial.rename(out_manifest)
    print()
    print("written              :", written)
    print(
        "vision forward [ms]  : median={:.3f} p95={:.3f} max={:.3f}".format(
            float(np.median(arr)),
            float(np.percentile(arr, 95)),
            float(np.max(arr)),
        )
    )
    print("output manifest      :", out_manifest)
    print("FROZEN FEATURE CACHE : PASS")
    print("=" * 108)


if __name__ == "__main__":
    main()
