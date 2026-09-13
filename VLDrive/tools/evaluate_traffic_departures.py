#!/usr/bin/env python3
"""Bounded, train-only VE stop/departure checkpoint comparison (no training).

The optional zero-velocity pass is a counterfactual sensitivity test, not an
offline driving-performance score. Original images, LiDAR, measurements and
checkpoints are read only; every output directory must be new.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import vision_encoder_training as trainer
from datasets.traffic_labels import traffic_record

DEFAULT_GREEN_FRAMES = (991, 1591, 2491, 2791)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--traffic-label-sidecar", type=Path, required=True)
    parser.add_argument("--route", help="Required if the train split contains multiple routes")
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--old-sha256", required=True)
    parser.add_argument("--new-checkpoint", type=Path)
    parser.add_argument("--new-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-green-frames", type=int, nargs="+", default=DEFAULT_GREEN_FRAMES)
    parser.add_argument("--post-green-frames", type=int, default=5)
    parser.add_argument("--pre-stop-samples", type=int, default=10,
                        help="Evenly sample at most this many known-red, zero-speed pregreen frames")
    parser.add_argument("--stop-speed-mps", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--velocity-zero-intervention", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only metadata/checkpoint audit; do not load images or build a model")
    args = parser.parse_args(argv)
    if bool(args.new_checkpoint) != bool(args.new_sha256):
        parser.error("--new-checkpoint and --new-sha256 must be supplied together")
    for digest in (args.old_sha256, args.new_sha256):
        if digest is not None and (len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest)):
            parser.error("Checkpoint SHA-256 must contain exactly 64 hexadecimal characters")
    if not 0 <= args.post_green_frames <= 50 or not 1 <= args.pre_stop_samples <= 50:
        parser.error("Bounded evaluation requires post-green-frames 0..50 and pre-stop-samples 1..50")
    if not 0 <= args.stop_speed_mps <= 0.1 or not 1 <= len(args.first_green_frames) <= 8:
        parser.error("Use 1..8 departure events and stop-speed-mps in [0, 0.1]")
    if len(set(args.first_green_frames)) != len(args.first_green_frames):
        parser.error("Departure frames must be unique")
    return args


def known_signal(row, route, frame):
    decision = traffic_record(route, "train", frame, row)
    state = decision["affected_light_state_raw"]
    known = bool(decision["traffic_available"] and decision["affected_light_id_raw"]
                 and state is not None)
    green = bool(known and not (int(state) & 1) and (int(state) & (16 | 32)))
    red = bool(known and decision["traffic_light"] == 0)
    return decision, green, red


def select_events(measurements, train_frames, route, first_green_frames,
                  post_green_frames=5, pre_stop_samples=10, stop_speed_mps=0.01):
    """Selection uses current recorded signals/speed, not future waypoint errors."""
    train_frames = set(train_frames)
    events = []
    for green_frame in sorted(first_green_frames):
        departure = list(range(green_frame, green_frame + post_green_frames + 1))
        if green_frame <= 0 or any(frame not in train_frames for frame in departure):
            raise ValueError("Departure event leaves the train split: {}".format(green_frame))
        current, green, _ = known_signal(measurements[green_frame], route, green_frame)
        previous, _, previous_red = known_signal(measurements[green_frame - 1], route, green_frame - 1)
        if not green or not previous_red or previous["affected_light_id_raw"] != current["affected_light_id_raw"]:
            raise ValueError("Event is not a verified red-to-known-green transition: {}".format(green_frame))
        cursor = green_frame - 1
        eligible = []
        while cursor in train_frames and float(measurements[cursor]["speed"]) <= stop_speed_mps:
            state, _, red = known_signal(measurements[cursor], route, cursor)
            if red and state["affected_light_id_raw"] == current["affected_light_id_raw"]:
                eligible.append(cursor)
            cursor -= 1
        eligible.reverse()
        if not eligible:
            raise ValueError("No known-red stopped training frames before {}".format(green_frame))
        positions = np.linspace(0, len(eligible) - 1, min(pre_stop_samples, len(eligible)), dtype=int)
        stopped = [eligible[int(position)] for position in positions]
        native_green = int(measurements[green_frame]["native_sync_frame"])
        for frame in stopped + departure:
            if int(measurements[frame]["native_sync_frame"]) - native_green != (frame - green_frame) * 5:
                raise ValueError("Discontinuous 10 Hz native ticks in departure evaluation")
        for frame in departure:
            signal, is_green, _ = known_signal(measurements[frame], route, frame)
            if not is_green or signal["affected_light_id_raw"] != current["affected_light_id_raw"]:
                raise ValueError("Postgreen window contains an unknown/different signal")
        events.append({
            "first_green_frame": green_frame,
            "signal_id": current["affected_light_id_raw"],
            "first_green_raw_state": current["affected_light_state_raw"],
            "contiguous_stopped_interval": [cursor + 1, green_frame - 1],
            "known_red_stopped_candidate_count": len(eligible),
            "evaluated_stopped_frames": stopped,
            "evaluated_departure_frames": departure,
            "sample_period_sec": 0.1,
        })
    return events


def tokens_from_decoder(output):
    """Exactly memfuser.return_feature pooling, from the same head forward pass."""
    hidden = output[0].permute(1, 0, 2)
    if tuple(hidden.shape[1:]) != (2506, 256):
        raise ValueError("Unexpected decoder output shape: {}".format(tuple(hidden.shape)))
    batch = hidden.shape[0]
    spatial = hidden[:, :2500].reshape(batch, 50, 50, 256).permute(0, 3, 1, 2)
    spatial = F.adaptive_avg_pool2d(spatial, (10, 10)).reshape(batch, 256, 100).permute(0, 2, 1)
    tokens = torch.cat((spatial, hidden[:, 2500:2501], hidden[:, 2501:2506]), dim=1)
    if tuple(tokens.shape[1:]) != trainer.TOKEN_SHAPE or not bool(torch.isfinite(tokens).all()):
        raise ValueError("Invalid connector tokens")
    return tokens


def tensor_identity(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def summarize_event(event, rows):
    departure = [rows[frame] for frame in event["evaluated_departure_frames"]]
    stopped = [rows[frame] for frame in event["evaluated_stopped_frames"]]
    first_not_red = next((row["frame_id"] for row in departure if row["predicted_traffic_class"] == 1), None)
    first = event["first_green_frame"]
    stable_gt_stop = [row for row in stopped if row["gt_endpoint_distance_m"] <= 0.05]
    return {
        "first_green_frame": first,
        "first_predicted_not_red_frame": first_not_red,
        "traffic_not_red_latency_sec": None if first_not_red is None else (first_not_red - first) * 0.1,
        "latency_right_censored": first_not_red is None,
        "observation_horizon_sec": (departure[-1]["frame_id"] - first) * 0.1,
        "green_head_accuracy": float(np.mean([row["predicted_traffic_class"] == 1 for row in departure])),
        "red_stopped_false_not_red_rate": float(np.mean([row["predicted_traffic_class"] == 1 for row in stopped])),
        "departure_ade_m": float(np.mean([row["ade_m"] for row in departure])),
        "departure_fde_m": float(np.mean([row["fde_m"] for row in departure])),
        "stopped_ade_m": float(np.mean([row["ade_m"] for row in stopped])),
        "stopped_fde_m": float(np.mean([row["fde_m"] for row in stopped])),
        "stationary_future_gt_samples": len(stable_gt_stop),
        "stationary_future_predicted_endpoint_distance_mean_m": (
            float(np.mean([row["predicted_endpoint_distance_m"] for row in stable_gt_stop]))
            if stable_gt_stop else None),
    }


def checkpoint_metadata(path, expected_sha, split_sha):
    actual_sha = trainer.sha256_file(path)
    if actual_sha != expected_sha.lower():
        raise ValueError("Evaluation checkpoint SHA-256 mismatch: {}".format(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {"model_name": trainer.MODEL_NAME, "feature_model_name": trainer.FEATURE_MODEL_NAME,
                "token_shape": list(trainer.TOKEN_SHAPE), "split_sha256": split_sha}
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError("Evaluation checkpoint provenance mismatch: {}".format(name))
    if checkpoint.get("args", {}).get("lidar_axis_policy") != "morai_flu_to_lmdrive_fru":
        raise ValueError("Evaluation requires the recorded MORAI-to-LMDrive LiDAR policy")
    return {
        "path": str(path.resolve()), "sha256": actual_sha, "epoch": int(checkpoint["epoch"]),
        "traffic_label_provenance": checkpoint.get("traffic_label_provenance"),
        "training_implementation_sha256": checkpoint.get("implementation_sha256"),
    }


def load_fixed_inputs(dataset, route, frames, seed):
    lookup = {(relative, int(frame)): i for i, (relative, frame) in enumerate(dataset.route_frames)}
    samples, raw, identities = {}, {}, {}
    for frame in frames:
        # Train membership is retained; only LiDAR's point shuffle is seeded.
        state = np.random.get_state()
        np.random.seed((seed + frame) % (2 ** 32))
        try:
            inputs, targets = dataset[lookup[(route, frame)]]
        finally:
            np.random.set_state(state)
        batch_inputs, batch_targets = default_collate([(inputs, targets)])
        samples[frame] = (batch_inputs, batch_targets)
        identities[str(frame)] = tensor_identity(batch_inputs)
        raw[frame] = dataset._read_json(dataset._route_path(route) / "measurements_full" / ("{:04d}.json".format(frame)))
    return samples, raw, identities


@torch.inference_mode()
def evaluate_checkpoint(path, samples, raw, device, velocity_zero=False):
    model = trainer.create_model(trainer.MODEL_NAME)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.to(device).eval()
    capture = {}
    def hook(_module, _inputs, output):
        capture["tokens"] = tokens_from_decoder(output).float().cpu()
    handle = model.decoder.register_forward_hook(hook)
    results, tokens, interventions = {}, {}, {}
    try:
        for frame, (host_inputs, host_targets) in samples.items():
            inputs = trainer.move_to_device(host_inputs, device)
            output = model(inputs)
            waypoint = output[1][0].float().cpu()
            gt = host_targets["waypoints"][0].float()
            error = torch.linalg.vector_norm(waypoint - gt, dim=-1)
            probabilities = output[2][0].float().softmax(dim=-1).cpu()
            tokens[frame] = capture["tokens"][0].numpy().copy()
            nav = raw[frame].get("navigation", {})
            results[frame] = {
                "frame_id": frame, "native_sync_frame": int(host_targets["native_sync_frame"][0]),
                "current_speed_mps": float(host_inputs["velocity"][0, 0]),
                "affected_light_id_raw": nav.get("affected_light_id_raw"),
                "affected_light_state_raw": nav.get("affected_light_state_raw"),
                "traffic_available": bool(host_targets["traffic_available"][0]),
                "traffic_gt_class": int(host_targets["traffic_light"][0]),
                "predicted_traffic_class": int(probabilities.argmax()),
                "predicted_red_probability": float(probabilities[0]),
                "predicted_not_red_probability": float(probabilities[1]),
                "waypoints": waypoint.tolist(), "waypoint_gt": gt.tolist(),
                "ade_m": float(error.mean()), "fde_m": float(error[-1]),
                "predicted_endpoint_distance_m": float(torch.linalg.vector_norm(waypoint[-1])),
                "gt_endpoint_distance_m": float(torch.linalg.vector_norm(gt[-1])),
            }
            if velocity_zero:
                changed = dict(inputs)
                changed["velocity"] = torch.zeros_like(inputs["velocity"])
                counterfactual = model(changed)
                changed_prob = counterfactual[2][0].float().softmax(dim=-1).cpu()
                token_delta = capture["tokens"][0].numpy() - tokens[frame]
                interventions[frame] = {
                    "frame_id": frame,
                    "interpretation": "input sensitivity only; not ground-truth driving performance",
                    "actual_speed_mps": results[frame]["current_speed_mps"],
                    "intervened_speed_mps": 0.0,
                    "not_red_probability_actual": float(probabilities[1]),
                    "not_red_probability_zero_velocity": float(changed_prob[1]),
                    "waypoint_delta_l2": float(torch.linalg.vector_norm(counterfactual[1][0].float().cpu() - waypoint)),
                    "token_delta_relative_l2": float(np.linalg.norm(token_delta) / max(np.linalg.norm(tokens[frame]), 1e-12)),
                }
            print("evaluated {} frame={}".format(path.name, frame), flush=True)
    finally:
        handle.remove()
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results, tokens, interventions


def token_comparison(old, new):
    results = {}
    for frame in sorted(old):
        before, after = old[frame], new[frame]
        if before.shape != trainer.TOKEN_SHAPE or after.shape != trainer.TOKEN_SHAPE:
            raise ValueError("Token comparison shape mismatch")
        groups = {}
        for label, indices in (("all", slice(None)), ("spatial", slice(0, 100)),
                               ("traffic", slice(100, 101)), ("waypoint", slice(101, 106))):
            a, b = before[indices].ravel(), after[indices].ravel()
            groups[label] = {
                "relative_l2": float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12)),
                "cosine_similarity": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
                "rms_change": float(np.sqrt(np.mean((b - a) ** 2))),
            }
        results[str(frame)] = groups
    return results


def main(argv=None):
    args = parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError("Evaluation requires a new --output directory")
    # make_loader(..., training=False) configures deterministic evaluation RGB transforms.
    args.train_stride = args.validation_stride = 1
    args.batch_size, args.workers = 1, 0
    args.lidar_axis_policy = "morai_flu_to_lmdrive_fru"
    dataset, _ = trainer.make_loader(args, "train", False)
    routes = sorted({relative for relative, _ in dataset.route_frames})
    if args.route is None and len(routes) != 1:
        raise ValueError("Choose --route for a multi-route training split")
    route = args.route or routes[0]
    train_frames = [int(frame) for relative, frame in dataset.route_frames if relative == route]
    if not train_frames:
        raise ValueError("Requested route is absent from the training split")
    measurements = dataset._measurements_all(dataset._route_path(route))
    events = select_events(measurements, train_frames, route, args.first_green_frames,
                           args.post_green_frames, args.pre_stop_samples, args.stop_speed_mps)
    frames = sorted({frame for event in events for key in ("evaluated_stopped_frames", "evaluated_departure_frames")
                     for frame in event[key]})
    split_sha = trainer.sha256_file(args.split_file)
    checkpoints = {"old": checkpoint_metadata(args.old_checkpoint, args.old_sha256, split_sha)}
    if args.new_checkpoint:
        checkpoints["new"] = checkpoint_metadata(args.new_checkpoint, args.new_sha256, split_sha)
        if checkpoints["new"]["traffic_label_provenance"] != dataset.traffic_label_provenance:
            raise ValueError("New checkpoint was not trained with this corrected-label provenance")
    report = {
        "schema": "morai_traffic_departures_v1", "split": "train", "route": route,
        "events": events, "evaluated_frames": frames, "checkpoints": checkpoints,
        "evaluation_implementation_sha256": trainer.implementation_provenance(),
        "evaluation_script_sha256": trainer.sha256_file(__file__),
        "traffic_label_provenance": dataset.traffic_label_provenance,
        "precision": "float32_no_autocast", "current_velocity_source": "current_recorded_ego_speed",
        "input_policy": "evaluation_RGB_no_augmentation; one seeded_LiDAR_sample_reused_across_checkpoints",
        "liDAR_seed": args.seed, "dry_run": args.dry_run,
        "interpretation": "bounded train-set head/token comparison; not closed-loop driving performance",
        "automatic_training_or_promotion": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.dry_run:
        trainer.verify_rgb_backbone_cache()
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback is automatic")
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        samples, raw, identities = load_fixed_inputs(dataset, route, frames, args.seed)
        report["input_sha256"] = identities
        report["models"] = {}
        all_tokens = {}
        for label, metadata in checkpoints.items():
            rows, tokens, interventions = evaluate_checkpoint(
                Path(metadata["path"]), samples, raw, device, args.velocity_zero_intervention)
            report["models"][label] = {"events": [summarize_event(event, rows) for event in events],
                                      "frames": [rows[frame] for frame in frames]}
            if interventions:
                report["models"][label]["velocity_zero_sensitivity_only"] = interventions
            all_tokens[label] = tokens
            token_path = args.output / (label + "_tokens.npz")
            with token_path.open("xb") as stream:
                np.savez_compressed(stream, frame_ids=np.asarray(frames),
                                    tokens=np.stack([tokens[frame] for frame in frames]))
            report["models"][label]["token_artifact_sha256"] = trainer.sha256_file(token_path)
        if "new" in all_tokens:
            report["token_changes_old_to_new"] = token_comparison(all_tokens["old"], all_tokens["new"])
        for metadata in checkpoints.values():
            if trainer.sha256_file(metadata["path"]) != metadata["sha256"]:
                raise RuntimeError("Checkpoint changed during read-only evaluation")
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "frames": len(frames), "dry_run": args.dry_run}), flush=True)


if __name__ == "__main__":
    main()
