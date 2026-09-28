#!/usr/bin/env python3
"""Bounded, paired MORAI encoder fine-tuning: legacy hue .4 versus hue omitted.

DISABLED.  This tool supplies its own training parameters to the stage-1
trainer through ``trainer_args``, which is the one thing stage-1 no longer
permits: training values may come only from a config file loaded by
``utils.config.vision_encoder_config.load_config``.  The trainer rejects any other
object, so this tool stops before it can run.

To bring it back, move the fixed protocol values out of ``trainer_args`` into a
config file under ``configs/`` and load it here.  The protocol's own pinning
(SOURCE_SHA, protocol_sha256, implementation_sha256) still applies; hash the
config file alongside them so the paired arms stay reproducible.

Preparation is CPU/read-only with respect to inputs. GPU execution requires an
explicit --action run and the shared training lock. Original assets are never
overwritten; this diagnostic does not promote a checkpoint automatically.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.vision_encoder.recipes import pretrain as trainer
from data.src.loaders.traffic_labels import traffic_record

REPO = ROOT.parent
OUTPUT_PARENT = ROOT / "outputs/logs"
SOURCE = ROOT / "outputs/checkpoints/vision_encoder/morai-full-loop-epoch25.pth.tar"
SOURCE_SHA = "5bd84eecae64fff810a5db4d7cae9f55db1ef5844aad2edf0c1bc25e3fcbdbb4"
LOCK = ROOT / ".runtime/training_runs/last_encoder.lock"
GREEN_FRAMES = (991, 1591, 2491, 2791)
ARMS = ("legacy_hue", "preserve_hue")
RGB_KEYS = {"rgb_front", "rgb_center", "rgb_left", "rgb_right", "rgb_rear"}


def stable_seed(*parts):
    encoded = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


@contextmanager
def local_data_rng(seed):
    """Seed only CPU augmentation/LiDAR RNG, without touching CUDA dropout RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)


class PairedColorJitter:
    """Sample exactly legacy ColorJitter RNG, then optionally omit only hue."""
    def __init__(self, apply_hue, trace):
        self.legacy = transforms.ColorJitter(.4, .4, .4, .4)
        self.apply_hue = bool(apply_hue)
        self.trace = trace

    def __call__(self, image):
        order, brightness, contrast, saturation, hue = self.legacy.get_params(
            self.legacy.brightness, self.legacy.contrast,
            self.legacy.saturation, self.legacy.hue)
        self.trace.append({"kind": "color", "order": order.tolist(),
                           "brightness": brightness, "contrast": contrast,
                           "saturation": saturation, "sampled_hue": hue})
        for index in order.tolist():
            if index == 0 and brightness is not None:
                image = TF.adjust_brightness(image, brightness)
            elif index == 1 and contrast is not None:
                image = TF.adjust_contrast(image, contrast)
            elif index == 2 and saturation is not None:
                image = TF.adjust_saturation(image, saturation)
            elif index == 3 and hue is not None and self.apply_hue:
                image = TF.adjust_hue(image, hue)
        return image


class TraceGeometry:
    def __init__(self, transform, trace):
        self.transform, self.trace = transform, trace

    def __call__(self, image):
        before = image.size
        result = self.transform(image)
        self.trace.append({"kind": type(self.transform).__name__,
                           "input_size": before, "output_size": result.size})
        return result


def install_paired_transforms(dataset, apply_hue, trace):
    for name in ("rgb_transform", "rgb_center_transform", "multi_view_transform"):
        pipeline = getattr(dataset, name)
        if not isinstance(pipeline, transforms.Compose):
            raise TypeError("Expected the production Compose RGB transform")
        result, replacements = [], 0
        for transform in pipeline.transforms:
            if isinstance(transform, transforms.ColorJitter):
                result.append(PairedColorJitter(apply_hue, trace))
                replacements += 1
            elif type(transform).__name__ in ("Resize2FixedSize", "RandomResize", "CenterCrop"):
                result.append(TraceGeometry(transform, trace))
            else:
                result.append(transform)
        if replacements != 1:
            raise ValueError("Training RGB pipeline must contain exactly one ColorJitter")
        setattr(dataset, name, transforms.Compose(result))


def tensor_identity(values):
    digest = hashlib.sha256()
    for name, item in sorted(values.items()):
        value = torch.as_tensor(item).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class SeededSamples(Dataset):
    def __init__(self, dataset, indices, seed, epoch, training, trace=None):
        self.dataset, self.indices = dataset, list(indices)
        self.seed, self.epoch, self.training = seed, epoch, training
        self.trace = trace if trace is not None else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        route, frame = self.dataset.route_frames[index]
        sample_seed = stable_seed("sample", self.seed, self.epoch, route, int(frame)) if self.training else int(frame)
        self.trace.clear()
        with local_data_rng(sample_seed):
            inputs, targets = self.dataset[index]
        trace_sha = hashlib.sha256(json.dumps(self.trace, sort_keys=True).encode()).hexdigest()
        audit = {"dataset_index": index, "frame_id": int(frame), "sample_seed": sample_seed,
                 "augmentation_trace_sha256": trace_sha,
                 "non_rgb_sha256": tensor_identity({k: v for k, v in inputs.items() if k not in RGB_KEYS}),
                 "targets_sha256": tensor_identity(targets),
                 "rgb_sha256": tensor_identity({k: v for k, v in inputs.items() if k in RGB_KEYS})}
        return inputs, targets, audit


def epoch_order(size, seed, epoch):
    generator = torch.Generator().manual_seed(stable_seed("order", seed, epoch))
    return torch.randperm(size, generator=generator).tolist()


def optimizer_budget(samples, epochs, batch_size=1, accumulation_steps=32):
    batches = math.ceil(samples / batch_size)
    return {"samples_per_epoch": samples, "microbatches_per_epoch": batches,
            "updates_per_epoch": math.ceil(batches / accumulation_steps),
            "epochs": epochs, "presentations_per_arm": samples * epochs,
            "updates_per_arm": math.ceil(batches / accumulation_steps) * epochs}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=REPO / "MoraiDrive/dataset_local/competition_train")
    parser.add_argument("--split-file", type=Path, default=ROOT / "configs/morai_full_loop_80_20.json")
    parser.add_argument("--traffic-label-sidecar", type=Path, default=ROOT / ".runtime/traffic_labels_v2/full_loop_20260911.json")
    parser.add_argument("--seed", type=int, default=607)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-frames", type=int, default=0,
                        help="Only with --smoke; choose 8..64 original train examples")
    parser.add_argument("--lock-fd", type=int, help="Inherited descriptor of the existing GPU training lock")
    args = parser.parse_args(argv)
    if args.max_train_frames and not args.smoke:
        parser.error("--max-train-frames is allowed only in a separate --smoke run")
    if args.smoke and not 8 <= (args.max_train_frames or 8) <= 64:
        parser.error("Smoke sample bound must be 8..64")
    for key in ("output", "dataset_root", "split_file", "traffic_label_sidecar"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if (args.output.parent.parent != OUTPUT_PARENT.resolve()
            or not args.output.parent.name.startswith("diagnosis_upstream_hue_")
            or not re.fullmatch(r"(?:smoke_)?signal_[a-z0-9_]+", args.output.name)):
        parser.error("Output must be a new signal_* or smoke_signal_* child under outputs/logs/diagnosis_upstream_hue_*")
    if args.smoke != ("smoke" in args.output.name):
        parser.error("Smoke must use a separately named output containing 'smoke'; full run must not")
    return args


def trainer_args(args):
    return SimpleNamespace(
        dataset_root=args.dataset_root, split_file=args.split_file,
        traffic_label_sidecar=args.traffic_label_sidecar,
        train_stride=1, validation_stride=1, batch_size=1, workers=0,
        accumulation_steps=32, lidar_axis_policy="morai_flu_to_lmdrive_fru",
        rgb_hue_jitter=.4, reference_lr=8e-5, reference_rgb_lr=3.2e-5,
        weight_decay=.05, waypoint_weight=.5, traffic_light_weight=.1,
        clip_grad=5., freeze_rgb_epochs=0, warm_start=SOURCE,
        warm_start_sha256=SOURCE_SHA, resume=None, epochs=1 if args.smoke else 3,
        warmup_epochs=0, seed=args.seed, output=args.output,
    )


def source_hashes():
    result = trainer.implementation_provenance()
    result["training/experiments/paired_hue_finetune.py"] = trainer.sha256_file(__file__)
    return result


def inspect_inputs(args):
    if trainer.sha256_file(SOURCE) != SOURCE_SHA:
        raise ValueError("Original epoch25 checkpoint SHA mismatch")
    config = trainer_args(args)
    dataset, _ = trainer.make_loader(config, "train", False)
    if len(dataset) != 3352 or not dataset.traffic_label_provenance:
        raise ValueError("Expected the audited 3352-frame original train dataset")
    records = []
    for index, (route, frame) in enumerate(dataset.route_frames):
        row = dataset._measurements_all(dataset._route_path(route))[frame]
        decision = traffic_record(route, "train", frame, row)
        records.append({"dataset_index": index, "route": route, "frame_id": int(frame),
                        "speed": float(row["speed"]), "label": int(decision["traffic_light"]),
                        "available": bool(decision["traffic_available"])})
    groups = {}
    for label, name in ((0, "red"), (1, "notred")):
        for stop, suffix in ((True, "stop"), (False, "moving")):
            groups[name + "_" + suffix] = sum(r["available"] and r["label"] == label
                and (r["speed"] <= .001) == stop for r in records)
    if groups != {"red_stop": 2042, "red_moving": 336, "notred_stop": 8, "notred_moving": 784}:
        raise ValueError("Audited label/current-speed group counts changed")
    indices = list(range(len(records)))
    if args.smoke:
        mandatory = {g + offset for g in GREEN_FRAMES for offset in (-1, 0)}
        indices = [r["dataset_index"] for r in records if r["frame_id"] in mandatory]
        candidates = [i for i in range(len(records)) if i not in indices]
        indices += candidates[: (args.max_train_frames or 8) - len(indices)]
        indices.sort()
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True, mmap=True)
    expected = {"model_name": trainer.MODEL_NAME, "feature_model_name": trainer.FEATURE_MODEL_NAME,
                "token_shape": [106, 256], "split_sha256": trainer.sha256_file(args.split_file)}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError("Original checkpoint provenance mismatch: " + key)
    return dataset, records, indices, {
        "source_epoch_zero_based": int(checkpoint["epoch"]),
        "historical_augmentation_known": False,
        "historical_note": "Epoch25 lacks an augmentation/source snapshot; this is a prospective paired intervention, not replay of proven historical draws.",
        "traffic_label_provenance": dataset.traffic_label_provenance,
        "groups_valid_full_train": groups,
        "mask_count": sum(not r["available"] for r in records),
    }


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def build_protocol(args, records, indices, metadata):
    return {
        "schema": "paired_hue_encoder_finetune_v1", "smoke": args.smoke,
        "seed": args.seed, "arms": list(ARMS), "epochs": 1 if args.smoke else 3,
        "selected_indices": indices, "selected_frame_ids": [records[i]["frame_id"] for i in indices],
        "dataset_root": str(args.dataset_root), "split_file": str(args.split_file),
        "split_sha256": trainer.sha256_file(args.split_file),
        "traffic_label_sidecar": str(args.traffic_label_sidecar),
        "checkpoint_path": str(SOURCE), "checkpoint_sha256": SOURCE_SHA,
        "implementation_sha256": source_hashes(), "source_metadata": metadata,
        "budget": optimizer_budget(len(indices), 1 if args.smoke else 3),
        "training": {"optimizer": "AdamW", "fusion_lr": 5e-6, "rgb_lr": 2e-6,
                     "weight_decay": .05, "betas": [.9, .999], "eps": 1e-8,
                     "batch_size": 1, "accumulation_steps": 32, "schedule": "constant",
                     "clip_norm": 5., "waypoint_L1_weight": .5, "masked_traffic_CE_weight": .1,
                     "AMP": "float16", "new_scaler_initial_scale": 1024.,
                     "overflow_policy": "fail on nonfinite unscaled gradients; never silently allow unequal skipped optimizer updates",
                     "trainable": "original RGB/PointPillars/fusion/queries/heads; no backbone freeze",
                     "initializer": "weights-only original epoch25; fresh optimizer/scaler/epoch zero"},
        "pairing": "Per-frame/epoch local Python,NumPy,CPU torch RNG; both ColorJitters sample legacy hue factor; preserve arm omits only applying hue. Same shuffled indices and per-microbatch CUDA dropout seed. Compare nonRGB/GT/augmentation-trace hashes.",
        "geometry": "Unchanged production legacy resize/crop; no letterbox",
        "evaluation": "common initial plus each epoch of both arms on original train only, FP32/no augmentation/current velocity/seeded LiDAR; traffic3170 valid and waypoint3352 (smoke subset separately marked); fixed-final selection",
        "selection": "fixed final epoch only; no best-epoch/model promotion",
        "new_independent_events": 0,
        "numerical_limit": "Paired RNG does not assert bitwise determinism of every CUDA scatter/autograd kernel.",
    }


def prepare(args):
    _, records, indices, metadata = inspect_inputs(args)
    protocol = build_protocol(args, records, indices, metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_new(args.output / "hue_protocol.json", protocol)
    print(json.dumps({"prepared": str(args.output), "budget": protocol["budget"]}), flush=True)


@contextmanager
def training_lock(inherited_fd=None):
    import fcntl
    if inherited_fd is not None:
        expected, actual = LOCK.stat(), os.fstat(inherited_fd)
        if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("Inherited lock descriptor is not the expected training lock")
        fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Shared open-file description belongs to the parent: do not unlock it.
        yield
    else:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def model_state_identity(model):
    return tensor_identity(model.state_dict())


def parameter_changes(model, initial):
    summary = {group: {"parameters": 0, "changed_parameters": 0, "delta_squared": 0.}
               for group in ("rgb_backbone", "lidar_backbone", "fusion_queries_heads")}
    for name, parameter in model.named_parameters():
        group = "rgb_backbone" if name.startswith("rgb_backbone.") else (
            "lidar_backbone" if name.startswith("lidar_backbone.") else "fusion_queries_heads")
        delta = parameter.detach().cpu().float() - initial[name].float()
        summary[group]["parameters"] += parameter.numel()
        summary[group]["changed_parameters"] += int(torch.count_nonzero(delta))
        summary[group]["delta_squared"] += float(delta.square().sum())
    for result in summary.values():
        result["delta_l2"] = math.sqrt(result.pop("delta_squared"))
    return summary


def optimizer_steps(optimizer):
    return sorted({int(state["step"]) for state in optimizer.state.values() if "step" in state})


def train_one_epoch(model, dataset, indices, optimizer, scaler, config, epoch, arm_dir, trace):
    model.train()
    trainer.set_rgb_frozen(model, False)
    selected = [indices[i] for i in epoch_order(len(indices), config.seed, epoch)]
    samples = SeededSamples(dataset, selected, config.seed, epoch, True, trace)
    loader = DataLoader(samples, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    totals, hashes = trainer.empty_totals(), {key: hashlib.sha256() for key in
        ("order", "non_rgb_sha256", "targets_sha256", "augmentation_trace_sha256", "rgb_sha256")}
    initial_steps = optimizer_steps(optimizer)
    prior_updates = initial_steps[0] if initial_steps else 0
    update_count, norms, started = 0, [], time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (inputs, targets, audit) in enumerate(loader):
        for key, digest in hashes.items():
            value = int(audit["frame_id"][0]) if key == "order" else audit[key][0]
            digest.update(str(value).encode() + b"\n")
        model_seed = stable_seed("model_dropout", config.seed, epoch, batch_index)
        torch.random.default_generator.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
        group_start = batch_index // 32 * 32
        group_size = min(32, len(loader) - group_start)
        inputs = trainer.move_to_device(inputs, "cuda")
        targets = trainer.move_to_device(targets, "cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(inputs)
            loss, metrics = trainer.loss_and_metrics(output, targets, config)
        scaler.scale(loss / group_size).backward()
        if batch_index - group_start + 1 == group_size:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad,
                                                  error_if_nonfinite=True)
            norms.append(float(norm))
            scaler.step(optimizer)
            scaler.update()
            update_count += 1
            if optimizer_steps(optimizer) != [prior_updates + update_count]:
                raise RuntimeError("Skipped or mismatched optimizer updates invalidate the paired run")
            optimizer.zero_grad(set_to_none=True)
        trainer.add_batch(totals, loss, metrics)
        if (batch_index + 1) % 200 == 0 or batch_index + 1 == len(loader):
            print(json.dumps({"arm": arm_dir.name, "epoch": epoch + 1,
                "microbatch": batch_index + 1, "total": len(loader),
                "updates_this_epoch": update_count, "elapsed_sec": time.monotonic() - started}), flush=True)
    result = {"epoch_one_based": epoch + 1, "metrics": trainer.finalize_totals(totals),
              "audit_hashes": {k: v.hexdigest() for k, v in hashes.items()},
              "optimizer_updates": update_count, "optimizer_state_steps": optimizer_steps(optimizer),
              "actual_lrs": [g["lr"] for g in optimizer.param_groups],
              "gradient_norm_mean": float(np.mean(norms)), "gradient_norm_max": max(norms),
              "scaler": scaler.state_dict(), "runtime_seconds": time.monotonic() - started,
              "frame_order": [int(dataset.route_frames[i][1]) for i in selected]}
    return result


def metrics_for_predictions(probability, labels, available, speed, predicted, gt, frames):
    error = np.linalg.norm(predicted - gt, axis=-1)
    predicted_class = probability >= .5
    result = {"waypoint_all": {"count": len(frames), "ADE_m": float(error.mean()),
                               "FDE_m": float(error[:, -1].mean())}, "groups": {}}
    groups = {"traffic_valid": available, "red_stop": available & (labels == 0) & (speed <= .001),
              "red_moving": available & (labels == 0) & (speed > .001),
              "notred_stop": available & (labels == 1) & (speed <= .001),
              "notred_moving": available & (labels == 1) & (speed > .001),
              "firstgreen": np.isin(frames, GREEN_FRAMES)}
    for name, mask in groups.items():
        if not mask.any():
            result["groups"][name] = {"count": 0}
            continue
        p, y, pred = probability[mask], labels[mask], predicted_class[mask]
        result["groups"][name] = {"count": int(mask.sum()), "accuracy": float((pred == y).mean()),
            "red_count": int((y == 0).sum()), "notred_count": int((y == 1).sum()),
            "red_false_notred_count": int(((y == 0) & pred).sum()),
            "notred_correct_count": int(((y == 1) & pred).sum()),
            "NLL": float(-(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1))).mean()),
            "ADE_m": float(error[mask].mean()), "FDE_m": float(error[mask, -1].mean()),
            "mean_pred_endpoint_distance_m": float(np.linalg.norm(predicted[mask, -1], axis=-1).mean())}
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    result["events"] = [{"first_green": green, "frames": [
        {"frame_id": frame, "speed": float(speed[lookup[frame]]),
         "notred_probability": float(probability[lookup[frame]]),
         "label": int(labels[lookup[frame]]), "available": bool(available[lookup[frame]]),
         "FDE_m": float(error[lookup[frame], -1])}
        for frame in (green - 30, *range(green - 2, green + 6)) if frame in lookup]}
        for green in GREEN_FRAMES]
    return result


@torch.inference_mode()
def evaluate_model(model, args, indices, destination):
    destination.mkdir(parents=True, exist_ok=False)
    dataset, _ = trainer.make_loader(trainer_args(args), "train", False)
    samples = SeededSamples(dataset, indices, args.seed, 0, False)
    loader = DataLoader(samples, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    rows, audit, token_rows = [], hashlib.sha256(), {}
    special_frames = {g + offset for g in GREEN_FRAMES for offset in (-30, -2, -1, 0, 1, 2, 3, 4, 5)}
    capture = {"enabled": False}
    def hook(_module, _args, output):
        if not capture["enabled"]:
            return
        hidden = output[0].permute(1, 0, 2)
        if tuple(hidden.shape[1:]) != (2506, 256):
            raise ValueError("Unexpected decoder shape")
        spatial = hidden[:, :2500].reshape(1, 50, 50, 256).permute(0, 3, 1, 2)
        spatial = F.adaptive_avg_pool2d(spatial, (10, 10)).reshape(1, 256, 100).permute(0, 2, 1)
        tokens = torch.cat((spatial, hidden[:, 2500:2506]), dim=1)[0]
        if tuple(tokens.shape) != (106, 256) or not bool(torch.isfinite(tokens).all()):
            raise ValueError("Invalid connector tokens")
        capture["tokens"] = tokens.float().cpu().numpy()
    handle = model.decoder.register_forward_hook(hook)
    try:
        for inputs, targets, identity in loader:
            frame = int(targets["frame_id"][0])
            capture["enabled"] = frame in special_frames
            for key in ("non_rgb_sha256", "rgb_sha256", "targets_sha256"):
                audit.update(identity[key][0].encode())
            frame_identity = hashlib.sha256("".join(identity[key][0] for key in
                ("non_rgb_sha256", "rgb_sha256", "targets_sha256")).encode()).hexdigest()
            output = model(trainer.move_to_device(inputs, "cuda"))
            rows.append((frame, int(targets["traffic_light"][0]), bool(targets["traffic_available"][0]),
                         float(inputs["velocity"][0, 0]), output[2][0].float().cpu().numpy(),
                         output[1][0].float().cpu().numpy(), targets["waypoints"][0].numpy(), frame_identity))
            if capture["enabled"]:
                token_rows[frame] = capture.pop("tokens")
            if len(rows) % 500 == 0 or len(rows) == len(samples):
                print(json.dumps({"evaluation": destination.name, "completed": len(rows), "total": len(samples)}), flush=True)
    finally:
        handle.remove()
    frames = np.asarray([r[0] for r in rows])
    labels = np.asarray([r[1] for r in rows])
    available = np.asarray([r[2] for r in rows])
    speed = np.asarray([r[3] for r in rows])
    logits = np.stack([r[4] for r in rows])
    pred, gt = np.stack([r[5] for r in rows]), np.stack([r[6] for r in rows])
    probability = torch.from_numpy(logits).softmax(-1)[:, 1].numpy()
    if not all(np.isfinite(v).all() for v in (logits, pred, gt, speed)):
        raise ValueError("Nonfinite evaluation output")
    report = metrics_for_predictions(probability, labels, available, speed, pred, gt, frames)
    report.update({"split": "train", "smoke_subset": args.smoke,
                   "bounded_subset": len(indices) != 3352,
                   "input_target_sha256": audit.hexdigest(), "precision": "FP32",
                   "token_shape": [106, 256], "no_augmentation": True})
    np.savez_compressed(destination / "predictions.npz", frame_ids=frames, labels=labels,
                        available=available, current_speed=speed, traffic_logits=logits,
                        waypoints=pred, waypoint_GT=gt,
                        input_target_identity=np.asarray([r[7] for r in rows]))
    token_ids = sorted(token_rows)
    np.savez_compressed(destination / "event_tokens.npz", frame_ids=np.asarray(token_ids),
                        tokens=np.stack([token_rows[f] for f in token_ids]))
    write_json_new(destination / "report.json", report)
    return report


def assert_arm_pairing(before, after):
    if len(before) != len(after):
        raise ValueError("Arms have different numbers of completed epochs")
    for a, b in zip(before, after):
        for key in ("order", "non_rgb_sha256", "targets_sha256", "augmentation_trace_sha256"):
            if a["audit_hashes"][key] != b["audit_hashes"][key]:
                raise ValueError("Arms differ beyond hue: " + key)
        if a["optimizer_state_steps"] != b["optimizer_state_steps"] or a["actual_lrs"] != b["actual_lrs"]:
            raise ValueError("Arms have mismatched optimizer exposure")


def compare_initial_predictions(baseline_path, probe_path):
    with np.load(baseline_path) as baseline, np.load(probe_path) as probe:
        lookup = {int(frame): i for i, frame in enumerate(baseline["frame_ids"])}
        positions = np.asarray([lookup[int(frame)] for frame in probe["frame_ids"]])
        for name in ("labels", "available", "current_speed", "waypoint_GT", "input_target_identity"):
            if not np.array_equal(baseline[name][positions], probe[name]):
                raise ValueError("Initial forward probe input/target mismatch: " + name)
        report = {"count": len(positions), "input_target_equal": True, "atol": 1e-5, "rtol": 1e-5}
        for name in ("traffic_logits", "waypoints"):
            before, after = baseline[name][positions], probe[name]
            report[name] = {"bitwise_equal": bool(np.array_equal(before, after)),
                            "max_abs_difference": float(np.abs(before - after).max())}
            if not np.allclose(before, after, atol=1e-5, rtol=1e-5):
                raise ValueError("Initial models differ beyond numerical tolerance: " + name)
    return report


def run(args):
    protocol_path = args.output / "hue_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["smoke"] != args.smoke or protocol["seed"] != args.seed:
        raise ValueError("CLI disagrees with prepared protocol")
    if protocol["implementation_sha256"] != source_hashes():
        raise ValueError("Source changed after preparation; prepare a new diagnostic run")
    for key in ("dataset_root", "split_file", "traffic_label_sidecar"):
        if protocol[key] != str(getattr(args, key)):
            raise ValueError("Input path differs from prepared protocol")
    _, records, indices, metadata = inspect_inputs(args)
    if indices != protocol["selected_indices"] or metadata != protocol["source_metadata"]:
        raise ValueError("Audited input selection or source metadata changed")
    if protocol != build_protocol(args, records, indices, metadata):
        raise ValueError("Prepared protocol differs from the current fixed-budget configuration")
    if any((args.output / name).exists() for name in ("baseline", *ARMS, "hue_run_started.json")):
        raise FileExistsError("Run outputs already exist; no overwrite/resume is allowed")
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
    trainer.verify_rgb_backbone_cache()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for explicit run; no CPU training fallback")
    with training_lock(args.lock_fd):
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json_new(args.output / "hue_run_started.json", {"pid": os.getpid(), "time": time.time(),
                       "protocol_sha256": trainer.sha256_file(protocol_path), "lock_fd_inherited": args.lock_fd is not None,
                       "python": sys.version, "torch": str(torch.__version__), "torchvision": torchvision.__version__,
                       "Pillow": PIL.__version__, "CUDA": torch.version.cuda, "GPU": torch.cuda.get_device_name(0),
                       "cudnn_deterministic": True, "TF32": False})
        base_state_sha, baseline, arm_results = None, None, {}
        for arm in ARMS:
            trainer.seed_everything(args.seed)
            config = trainer_args(args)
            config.implementation_sha256 = protocol["implementation_sha256"]
            dataset, _ = trainer.make_loader(config, "train", True)
            config.traffic_label_provenance = dataset.traffic_label_provenance
            trace = []
            install_paired_transforms(dataset, arm == "legacy_hue", trace)
            model = trainer.create_model(trainer.MODEL_NAME).to("cuda")
            optimizer, base_lr, rgb_lr = trainer.optimizer_for(model, config)
            if not math.isclose(base_lr, 5e-6) or not math.isclose(rgb_lr, 2e-6):
                raise ValueError("Unexpected effective-batch-scaled learning rate")
            scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
            start, _, config.run_initialization = trainer.load_training_initialization(model, optimizer, scaler, config)
            if start != 0 or optimizer.state:
                raise ValueError("Weights-only restart must reset optimizer and epoch")
            initial_sha = model_state_identity(model)
            if base_state_sha is not None and initial_sha != base_state_sha:
                raise ValueError("Arm initial weights/buffers differ")
            base_state_sha = initial_sha
            initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
            arm_dir = args.output / arm
            arm_dir.mkdir(exist_ok=False)
            initial_forward = None
            if baseline is None:
                baseline = evaluate_model(model, args, indices, args.output / "baseline")
            else:
                probe_frames = {g + offset for g in GREEN_FRAMES for offset in (-1, 0)}
                probe_indices = [i for i in indices if int(dataset.route_frames[i][1]) in probe_frames]
                if len(probe_indices) != 8:
                    raise ValueError("Initial forward pairing check requires the fixed eight frames")
                evaluate_model(model, args, probe_indices, arm_dir / "initial_probe")
                initial_forward = compare_initial_predictions(args.output / "baseline/predictions.npz",
                                                                arm_dir / "initial_probe/predictions.npz")
                write_json_new(arm_dir / "initial_forward_pairing.json", initial_forward)
            records = []
            for epoch in range(protocol["epochs"]):
                record = train_one_epoch(model, dataset, indices, optimizer, scaler, config, epoch, arm_dir, trace)
                if arm == "preserve_hue":
                    assert_arm_pairing([arm_results["legacy_hue"]["training_epochs"][epoch]], [record])
                changes = parameter_changes(model, initial)
                if any(group["changed_parameters"] == 0 for group in changes.values()):
                    raise RuntimeError("Expected actual RGB, LiDAR and fusion/query/head updates")
                record["parameter_changes_since_initialization"] = changes
                records.append(record)
                config.output = arm_dir
                config.rgb_hue_jitter = .4 if arm == "legacy_hue" else 0.
                config.paired_hue_protocol_sha256 = trainer.sha256_file(protocol_path)
                config.checkpoint_selection = "fixed_final_epoch; best_fde_field_unused_infinity"
                trainer.save_checkpoint(arm_dir / ("checkpoint-epoch{:02d}.pth.tar".format(epoch + 1)),
                                        model, optimizer, scaler, epoch, float("inf"), config, record["metrics"])
                eval_name = "final_eval" if epoch + 1 == protocol["epochs"] else "eval_epoch{:02d}".format(epoch + 1)
                final = evaluate_model(model, args, indices, arm_dir / eval_name)
                if final["input_target_sha256"] != baseline["input_target_sha256"]:
                    raise ValueError("Initial and epoch evaluation inputs/GT differ")
                record["evaluation"] = final
                record["evaluation_directory"] = str(arm_dir / eval_name)
                write_json_new(arm_dir / ("training_epoch{:02d}.json".format(epoch + 1)), record)
            checkpoint_path = arm_dir / ("checkpoint-epoch{:02d}.pth.tar".format(protocol["epochs"]))
            arm_results[arm] = {"initial_state_sha256": initial_sha, "initialization": config.run_initialization,
                               "initial_forward_pairing": initial_forward,
                               "training_epochs": records, "final_evaluation": final,
                               "checkpoint": str(checkpoint_path), "checkpoint_sha256": trainer.sha256_file(checkpoint_path)}
            write_json_new(arm_dir / "arm_report.json", arm_results[arm])
            del initial, model, optimizer, scaler, dataset
            gc.collect()
            torch.cuda.empty_cache()
        assert_arm_pairing(arm_results[ARMS[0]]["training_epochs"], arm_results[ARMS[1]]["training_epochs"])
        if trainer.sha256_file(SOURCE) != SOURCE_SHA or source_hashes() != protocol["implementation_sha256"]:
            raise RuntimeError("Original checkpoint or implementation changed during run")
        write_json_new(args.output / "hue_results.json", {"schema": protocol["schema"], "smoke": args.smoke,
                       "protocol_sha256": trainer.sha256_file(protocol_path), "baseline": baseline,
                       "arms": arm_results, "paired_input_target_trace_checks_passed": True,
                       "original_checkpoint_unchanged": True, "source_unchanged": True,
                       "automatic_promotion": False})
        print(json.dumps({"completed": str(args.output), "smoke": args.smoke}), flush=True)


DISABLED_REASON = (
    "training/experiments/paired_hue_finetune.py is disabled: it builds stage-1 training "
    "parameters in code (trainer_args), and stage-1 now accepts them only from "
    "a config file. Move the protocol values into configs/ and load them with "
    "utils.config.vision_encoder_config.load_config to re-enable it."
)


def main(argv=None):
    raise SystemExit(DISABLED_REASON)


def _disabled_main(argv=None):
    args = parse_args(argv)
    if args.action == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
