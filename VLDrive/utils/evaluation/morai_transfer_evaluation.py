"""Evaluate a trained MORAI visual encoder per sealed publication and lap.

This path never creates an optimizer or changes model weights.  It deliberately
keeps publication/lap metrics separate so a new campaign cannot hide a sensor
or label regression behind an aggregate validation score.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data.src.loaders.morai_sensor_record_dataset import (
    PathResolver, SensorRecordDataset, digest, load_publication)
from models.vision_encoder import create_vision_encoder
from utils.evaluation.morai_transfer_metrics import evaluate


ROOT = Path(__file__).resolve().parents[2]


def add_arguments(parser):
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="MORAI visual checkpoint-last.pth.tar to evaluate")
    parser.add_argument("--checkpoint-sha256",
                        help="Optional expected checkpoint SHA256")
    parser.add_argument("--config", type=Path,
                        help="Defaults to configs/morai_transfer.local.json")
    parser.add_argument("--manifest-index", type=Path, action="append",
                        help="Repeat to evaluate publications separately")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--velocity-source", choices=("gps", "zero"), default="gps")
    parser.add_argument("--diagnostic-limit", type=int, default=0,
                        help="Explicit per-lap frame limit; 0 evaluates every frame")
    return parser


def _write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    pending.replace(path)


def _resolve(args):
    config_path = args.config or ROOT / "configs/morai_transfer.local.json"
    if args.config and not config_path.is_file():
        raise FileNotFoundError("Missing configuration: " + str(config_path))
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    manifests = args.manifest_index or [Path(p) for p in config.get("manifest_indices", [])]
    if not manifests:
        raise ValueError("No publications selected; set --manifest-index or config manifest_indices")
    args.manifest_index = [Path(p).expanduser().resolve() for p in manifests]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Missing evaluation checkpoint: " + str(args.checkpoint))
    if args.threads < 1 or args.workers < 0 or args.diagnostic_limit < 0:
        raise ValueError("Invalid thread/worker/limit setting")
    if args.output is None:
        args.output = args.checkpoint.parent / (
            "evaluation_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns() % 1000000))
    args.output = args.output.expanduser().resolve()
    return PathResolver(config.get("path_mappings", []))


def _load_model(checkpoint, expected_sha256, device):
    actual_sha256 = digest(checkpoint)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError("Evaluation checkpoint SHA256 mismatch")
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Expected checkpoint containing state_dict")
    if payload.get("token_shape") not in (None, [106, 256]):
        raise ValueError("Checkpoint has an incompatible visual token contract")
    model = create_vision_encoder({"name": "memfuser_baseline_e1d3"}, pretrained_backbone=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device), payload, actual_sha256


def run(args):
    resolver = _resolve(args)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for a bounded diagnostic")
    args.output.mkdir(parents=True, exist_ok=False)
    model, checkpoint, checkpoint_sha256 = _load_model(
        args.checkpoint, args.checkpoint_sha256, args.device)
    run_config = {
        "schema": "morai_visual_evaluation_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_steps": checkpoint.get("steps"),
        "manifest_indices": [str(p) for p in args.manifest_index],
        "split": args.split,
        "velocity_source": args.velocity_source,
        "diagnostic_limit_per_lap": args.diagnostic_limit,
        "weights_updated": False,
    }
    _write_json(args.output / "config.json", run_config)

    results = []
    for publication_number, index_path in enumerate(args.manifest_index, 1):
        print("[EVAL] validating publication %d/%d: %s" % (
            publication_number, len(args.manifest_index), index_path), flush=True)
        publication, splits = load_publication(index_path, resolver)
        indexed_rows = publication[args.split]
        manifest_rows = splits[args.split]
        if len(indexed_rows) != len(manifest_rows):
            raise ValueError("Publication lap metadata mismatch")
        for lap_number, ((manifest_path, manifest), index_row) in enumerate(
                zip(manifest_rows, indexed_rows), 1):
            dataset = SensorRecordDataset(
                [(manifest_path, manifest)], resolver, args.velocity_source)
            selected = dataset
            if args.diagnostic_limit:
                selected = Subset(dataset, range(min(len(dataset), args.diagnostic_limit)))
            loader = DataLoader(selected, batch_size=1, shuffle=False, num_workers=args.workers)
            label = "publication_%02d_lap_%03d" % (
                publication_number, int(index_row.get("lap", lap_number)))
            print("[EVAL] %s: %d admitted frames" % (label, len(selected)), flush=True)
            metrics = evaluate(
                model, loader, args.device, progress_phase="evaluation:" + label,
                detection_loss_policy="lmdrive_lav_v1")
            result = {
                "publication": str(index_path),
                "publication_sha256": digest(index_path),
                "manifest": str(manifest_path),
                "manifest_sha256": digest(manifest_path),
                "lap": index_row.get("lap", lap_number),
                "split": args.split,
                "coverage": dict(dataset.coverage),
                "excluded": dict(dataset.excluded),
                "metrics": metrics,
            }
            results.append(result)
            _write_json(args.output / (label + ".json"), result)
            print(json.dumps({"completed": label, "metrics": metrics}, allow_nan=False), flush=True)

    report = dict(run_config, results=results, completed=True,
                  comparison_note="Compare identical checkpoint metrics across publications/laps; no aggregate score is used.")
    _write_json(args.output / "report.json", report)
    print("완료: " + str(args.output / "report.json"), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
