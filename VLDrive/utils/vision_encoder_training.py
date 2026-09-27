"""Stage 1: train the LMDrive vision-encoder structure on synchronized MORAI data.

Run it through ``python train.py vision-encoder --config <file>``; this module
holds the loop and the checkpoint format so tools/ and tests/ can import the
same code.

Every training parameter comes from that config file and from nothing else.
The command line carries only the config path, the loaded config is frozen, and
the functions here reject anything that is not one, so no caller can substitute
or override a value.

The public full LMDrive encoder checkpoint is intentionally not used.  The
ResNet-50d ImageNet backbone follows the paper, while PointPillars, fusion,
queries, and prediction heads are learned from the MORAI expert trajectory.

Formerly ``vision_encoder/train_morai_pretrain.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.morai_perception_dataset import MoraiPerceptionDataset
from datasets.loader import create_perception_loader
from models.vision_encoder import create_vision_encoder
from utils.vision_encoder_config import ConfigError, load_config, require_config


MODEL_NAME = "memfuser_baseline_e1d3"
FEATURE_MODEL_NAME = "memfuser_baseline_e1d3_return_feature"
TOKEN_SHAPE = (106, 256)
RGB_BACKBONE_FILENAME = "resnet50d_ra2-464e36ba.pth"
RGB_BACKBONE_SHA256 = (
    "464e36baa940a80971e717454e61e8f6fc1a1b284b7a3dbb6f6fd4c3da78533d"
)


def add_arguments(parser):
    """Stage-1 command line: the config file, and nothing that could override it."""
    parser.add_argument("--config", type=Path, required=True,
                        help="stage-1 training config, e.g. configs/vldrive_train.json")
    return parser


def config_from_args(parser, config):
    """Load the config named on the command line; parameters come only from it."""
    try:
        return load_config(config.config)
    except ConfigError as error:
        parser.error(str(error))


def parse_args(argv=None):
    parser = add_arguments(argparse.ArgumentParser(
        description="MORAI expert-imitation pretraining for the VLDrive vision encoder"))
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(values, device):
    return {key: value.to(device, non_blocking=True) for key, value in values.items()}


def make_loader(config, split, training):
    require_config(config)
    stride = config.train_stride if training else config.validation_stride
    dataset = MoraiPerceptionDataset(
        root=str(config.dataset_root),
        split_file=str(config.split_file),
        split=split,
        with_lidar=True,
        multi_view=True,
        lidar_axis_policy=config.lidar_axis_policy,
        sample_stride=stride,
        traffic_label_sidecar=(str(config.traffic_label_sidecar)
                               if config.traffic_label_sidecar else None),
    )
    loader = create_perception_loader(
        dataset,
        input_size=(3, 224, 224),
        multi_view_input_size=(3, 128, 128),
        batch_size=config.batch_size,
        is_training=training,
        scale=[0.9, 1.1] if training else [1.0, 1.0],
        color_jitter=0.4 if training else 0.0,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
        hue_jitter=config.rgb_hue_jitter if training else 0.0,
    )
    return dataset, loader


def set_rgb_frozen(model, frozen):
    for parameter in model.rgb_backbone.parameters():
        parameter.requires_grad_(not frozen)
    if frozen:
        model.rgb_backbone.eval()


def optimizer_for(model, config):
    require_config(config)
    effective_batch = config.batch_size * config.accumulation_steps
    lr = config.reference_lr * effective_batch / 512.0
    rgb_lr = config.reference_rgb_lr * effective_batch / 512.0
    rgb_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("rgb_backbone."):
            rgb_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": lr, "name": "fusion_lidar_heads"},
            {"params": rgb_parameters, "lr": rgb_lr, "name": "rgb_backbone"},
        ],
        weight_decay=config.weight_decay,
    )
    return optimizer, lr, rgb_lr


def lr_factor(epoch, config):
    if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
        return float(epoch + 1) / float(config.warmup_epochs)
    cosine_epochs = max(config.epochs - config.warmup_epochs, 1)
    progress = float(epoch - config.warmup_epochs) / float(cosine_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_epoch_lr(optimizer, base_lr, rgb_lr, epoch, config):
    factor = lr_factor(epoch, config)
    optimizer.param_groups[0]["lr"] = base_lr * factor
    optimizer.param_groups[1]["lr"] = rgb_lr * factor


def loss_and_metrics(output, target, config):
    require_config(config)
    predicted_waypoints = output[1]
    waypoint_gt = target["waypoints"]
    waypoint_loss = F.l1_loss(predicted_waypoints, waypoint_gt)

    traffic_mask = target["traffic_available"].bool()
    if traffic_mask.any():
        traffic_loss = F.cross_entropy(
            output[2][traffic_mask], target["traffic_light"][traffic_mask]
        )
        traffic_correct = (
            output[2][traffic_mask].argmax(dim=1)
            == target["traffic_light"][traffic_mask]
        ).sum()
        traffic_count = traffic_mask.sum()
    else:
        traffic_loss = output[2].sum() * 0.0
        traffic_correct = torch.zeros((), device=output[2].device)
        traffic_count = torch.zeros((), device=output[2].device)

    distance = torch.linalg.vector_norm(predicted_waypoints - waypoint_gt, dim=-1)
    total = config.waypoint_weight * waypoint_loss + config.traffic_light_weight * traffic_loss
    return total, {
        "waypoint_l1": waypoint_loss.detach(),
        "traffic_loss": traffic_loss.detach(),
        "ade_sum": distance.sum().detach(),
        "fde_sum": distance[:, -1].sum().detach(),
        "waypoint_count": torch.tensor(distance.numel(), device=distance.device),
        "sample_count": torch.tensor(distance.shape[0], device=distance.device),
        "traffic_correct": traffic_correct.detach(),
        "traffic_count": traffic_count.detach(),
    }


def empty_totals():
    return {
        "loss": 0.0,
        "waypoint_l1": 0.0,
        "traffic_loss": 0.0,
        "ade_sum": 0.0,
        "fde_sum": 0.0,
        "waypoint_count": 0,
        "sample_count": 0,
        "traffic_correct": 0,
        "traffic_count": 0,
    }


def finalize_totals(total):
    batches = max(total.pop("batches"), 1)
    return {
        "loss": total["loss"] / batches,
        "waypoint_l1": total["waypoint_l1"] / batches,
        "traffic_loss": total["traffic_loss"] / batches,
        "ade_m": total["ade_sum"] / max(total["waypoint_count"], 1),
        "fde_m": total["fde_sum"] / max(total["sample_count"], 1),
        "traffic_accuracy": total["traffic_correct"] / max(total["traffic_count"], 1),
        "traffic_samples": total["traffic_count"],
    }


def add_batch(total, loss, metrics):
    total["loss"] += float(loss.detach())
    total["waypoint_l1"] += float(metrics["waypoint_l1"])
    total["traffic_loss"] += float(metrics["traffic_loss"])
    for key in ("ade_sum", "fde_sum", "waypoint_count", "sample_count",
                "traffic_correct", "traffic_count"):
        total[key] += float(metrics[key])
    total["batches"] = total.get("batches", 0) + 1


def train_epoch(model, loader, optimizer, scaler, device, config, epoch, amp_enabled):
    model.train()
    rgb_frozen = epoch < config.freeze_rgb_epochs
    set_rgb_frozen(model, rgb_frozen)
    optimizer.zero_grad(set_to_none=True)
    total = empty_totals()
    context = (
        (lambda: torch.amp.autocast(device_type="cuda"))
        if amp_enabled
        else nullcontext
    )

    available_batches = len(loader)
    planned_batches = (
        min(available_batches, config.max_train_batches)
        if config.max_train_batches
        else available_batches
    )
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= planned_batches:
            break
        group_start = (batch_index // config.accumulation_steps) * config.accumulation_steps
        group_size = min(config.accumulation_steps, planned_batches - group_start)
        inputs = move_to_device(inputs, device)
        targets = move_to_device(targets, device)
        with context():
            output = model(inputs)
            loss, metrics = loss_and_metrics(output, targets, config)
            # Scale the final short accumulation group by its actual size.
            scaled_loss = loss / group_size
        scaler.scale(scaled_loss).backward()
        should_step = (batch_index - group_start + 1) == group_size
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        add_batch(total, loss, metrics)
        completed = batch_index + 1
        if completed % config.log_interval == 0 or completed == planned_batches:
            print(
                "train epoch={} batch={}/{} loss={:.6f}".format(
                    epoch, completed, planned_batches, float(loss.detach())
                ),
                flush=True,
            )
    return finalize_totals(total)


@torch.no_grad()
def validate(model, loader, device, config, amp_enabled):
    model.eval()
    total = empty_totals()
    context = (
        (lambda: torch.amp.autocast(device_type="cuda"))
        if amp_enabled
        else nullcontext
    )
    available_batches = len(loader)
    planned_batches = (
        min(available_batches, config.max_validation_batches)
        if config.max_validation_batches
        else available_batches
    )
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= planned_batches:
            break
        inputs = move_to_device(inputs, device)
        targets = move_to_device(targets, device)
        with context():
            output = model(inputs)
            loss, metrics = loss_and_metrics(output, targets, config)
        add_batch(total, loss, metrics)
        completed = batch_index + 1
        if completed % config.log_interval == 0 or completed == planned_batches:
            print(
                "validation batch={}/{} loss={:.6f}".format(
                    completed, planned_batches, float(loss.detach())
                ),
                flush=True,
            )
    return finalize_totals(total)


@dataclass
class RunState:
    """Values a run derives; the config stays frozen and authoritative."""

    implementation_sha256: dict
    traffic_label_provenance: object = None
    run_initialization: object = None


def sha256_file(path):
    if '#' in str(path):
        from utils.config_files import document_bytes
        return hashlib.sha256(document_bytes(path)).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_provenance():
    """Freeze the trainer, target construction, and encoder implementation identity."""
    paths = (
        "train.py", "utils/vision_encoder_training.py",
        "utils/vision_encoder_config.py",
        "datasets/split_manifest.py", "datasets/traffic_labels.py",
        "datasets/morai_perception_dataset.py", "datasets/lidar.py",
        "datasets/loader.py", "datasets/transforms.py",
        "models/vision_encoder.py", "models/pointpillars.py", "models/resnet.py",
        "models/transformer.py",
    )
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def prepare_output_directory(config):
    """A weights-only run may never reuse a populated output directory."""
    require_config(config)
    output = Path(config.output)
    if output.exists() and config.warm_start and any(output.iterdir()):
        raise FileExistsError("Weights-only warm start requires a new or empty output.dir")
    if output.exists() and not config.resume:
        protected = ("checkpoint-last.pth.tar", "model-best.pth.tar", "metrics.jsonl",
                     "feature_contract.json", "run_provenance.json")
        if any((output / name).exists() for name in protected):
            raise FileExistsError(
                "Output already contains a training run; set a new output.dir "
                "or initialization.resume"
            )
    output.mkdir(parents=True, exist_ok=True)


def load_training_initialization(model, optimizer, scaler, config, state):
    """Keep weights-only transfer distinct from optimizer/epoch continuation."""
    require_config(config)
    if config.warm_start:
        source = Path(config.warm_start).resolve()
        actual_sha = sha256_file(source)
        if actual_sha != config.warm_start_sha256:
            raise ValueError("Warm-start checkpoint SHA-256 mismatch")
        checkpoint = torch.load(str(source), map_location="cpu", weights_only=True)
        target_split_sha = sha256_file(config.split_file)
        declared_source_split = config.warm_start_source_split_sha256
        expected = {
            "model_name": MODEL_NAME,
            "feature_model_name": FEATURE_MODEL_NAME,
            "token_shape": list(TOKEN_SHAPE),
            "split_sha256": declared_source_split or target_split_sha,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError("Warm-start checkpoint provenance mismatch: {}".format(key))
        initialization = checkpoint.get("initialization", {})
        if initialization.get("full_lmdrive_encoder") is not False:
            raise ValueError("Warm start must be a MORAI-trained encoder, not full LMDrive weights")
        source_args = checkpoint.get("args", {})
        if source_args.get("lidar_axis_policy") != config.lidar_axis_policy:
            raise ValueError("Warm-start LiDAR axis policy mismatch")
        if optimizer.state:
            raise ValueError("Weights-only warm start requires a freshly created optimizer")
        source_state = checkpoint["state_dict"]
        current_state = model.state_dict()
        if source_state.keys() != current_state.keys():
            raise ValueError("Warm-start model state keys mismatch")
        for name, value in source_state.items():
            current = current_state[name]
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError("Warm-start model tensor shape/dtype mismatch: {}".format(name))
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("Warm-start model tensor is not finite: {}".format(name))
        model.load_state_dict(source_state, strict=True)
        return 0, float("inf"), {
            "kind": "weights_only_warm_start",
            "source_checkpoint": str(source),
            "source_sha256": actual_sha,
            "source_epoch": int(checkpoint["epoch"]),
            "source_split_sha256": checkpoint["split_sha256"],
            "target_split_sha256": target_split_sha,
            "split_change_declared": bool(declared_source_split
                                          and declared_source_split != target_split_sha),
            "source_initialization": initialization,
            "source_implementation_sha256": checkpoint.get("implementation_sha256"),
            "source_traffic_label_provenance": checkpoint.get("traffic_label_provenance"),
            "optimizer_restored": False,
            "scaler_restored": False,
            "epoch_restored": False,
            "best_score_restored": False,
            "schedule": "new_run_from_epoch_zero",
            "exact_resume": False,
        }
    if config.resume:
        source = Path(config.resume).resolve()
        checkpoint = torch.load(str(source), map_location="cpu", weights_only=True)
        if checkpoint.get("split_sha256") != sha256_file(config.split_file):
            raise ValueError("Resume checkpoint was trained with a different split manifest")
        if checkpoint.get("traffic_label_provenance") != state.traffic_label_provenance:
            raise ValueError("Resume traffic labels differ; use initialization.warm_start for corrected labels")
        recorded_hue = checkpoint.get("args", {}).get("rgb_hue_jitter")
        requested_hue = config.rgb_hue_jitter
        if recorded_hue is not None and requested_hue is not None and recorded_hue != requested_hue:
            raise ValueError("Resume RGB hue augmentation differs; use initialization.warm_start for changed augmentation")
        if recorded_hue is None and requested_hue is not None:
            print(
                "WARNING: resume checkpoint has no recorded RGB hue augmentation; "
                "using augmentation.rgb_hue_jitter={}. Historical augmentation cannot be verified."
                .format(requested_hue), flush=True,
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_fde_m", float("inf"))), {
            "kind": "optimizer_resume",
            "source_checkpoint": str(source),
            "optimizer_restored": True,
            "scaler_restored": True,
            "epoch_restored": True,
            "exact_resume": False,
            "note": "Legacy checkpoints do not capture RNG or data-loader state",
        }
    return 0, float("inf"), {"kind": "imagenet_rgb_random_other", "exact_resume": False}


def verify_rgb_backbone_cache():
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / RGB_BACKBONE_FILENAME
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Required ImageNet RGB backbone is missing: {}. "
            "Download the ResNet-50d backbone only; do not use the full LMDrive encoder."
            .format(checkpoint)
        )
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != RGB_BACKBONE_SHA256:
        raise RuntimeError(
            "RGB backbone SHA-256 mismatch: expected {}, got {} ({})".format(
                RGB_BACKBONE_SHA256, actual_sha256, checkpoint
            )
        )
    return checkpoint


def save_checkpoint(path, model, optimizer, scaler, epoch, best_fde, config, state, metrics):
    require_config(config)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_fde_m": float(best_fde),
            "metrics": metrics,
            "model_name": MODEL_NAME,
            "feature_model_name": FEATURE_MODEL_NAME,
            "token_shape": list(TOKEN_SHAPE),
            "split_sha256": sha256_file(config.split_file),
            "initialization": {
                "full_lmdrive_encoder": False,
                "rgb_backbone": ("MORAI warm-start checkpoint" if config.warm_start
                                 else "resnet50d ImageNet pretrained"),
                "lidar_backbone": "MORAI warm-start checkpoint" if config.warm_start else "random",
                "fusion_decoder_queries_heads": (
                    "MORAI warm-start checkpoint" if config.warm_start else "random"),
            },
            "run_initialization": state.run_initialization,
            "traffic_label_provenance": state.traffic_label_provenance,
            "implementation_sha256": state.implementation_sha256,
            "args": config.flat_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


@torch.no_grad()
def verify_feature_contract(checkpoint_path, sample_inputs, device):
    # pretrained_backbone=False: the strict load below replaces every weight.
    feature_model = create_vision_encoder({"name": FEATURE_MODEL_NAME}, pretrained_backbone=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    result = feature_model.load_state_dict(checkpoint["state_dict"], strict=True)
    feature_model.to(device).eval()
    features = feature_model(sample_inputs)
    expected = (sample_inputs["rgb_front"].shape[0],) + TOKEN_SHAPE
    if tuple(features.shape) != expected or not bool(torch.isfinite(features).all()):
        raise RuntimeError("Invalid connector feature output: {}".format(tuple(features.shape)))
    return str(result), tuple(features.shape)


def run(config):
    """Train the encoder from a loaded config; no other source supplies a value."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. Verify a Blackwell-capable PyTorch/CUDA build before training."
        )
    require_config(config)
    state = RunState(implementation_sha256=implementation_provenance())
    prepare_output_directory(config)
    seed_everything(config.seed)

    backbone_path = verify_rgb_backbone_cache()
    print("verified RGB backbone: {}".format(backbone_path))

    device = torch.device("cuda:0")
    amp_enabled = config.amp
    train_dataset, train_loader = make_loader(config, "train", True)
    val_dataset, val_loader = make_loader(config, "validation", False)
    state.traffic_label_provenance = getattr(train_dataset, "traffic_label_provenance", None)
    if state.traffic_label_provenance != getattr(val_dataset, "traffic_label_provenance", None):
        raise ValueError("Train and validation traffic-label provenance differs")
    if config.traffic_label_sidecar and not state.traffic_label_provenance:
        raise ValueError("Dataset did not validate the requested traffic-label sidecar")
    print("train samples={} validation samples={}".format(len(train_dataset), len(val_dataset)))

    model = create_vision_encoder({"name": MODEL_NAME}).to(device)
    optimizer, base_lr, rgb_lr = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_fde, state.run_initialization = load_training_initialization(
        model, optimizer, scaler, config, state
    )
    if start_epoch >= config.epochs:
        raise ValueError("No epochs remain; schedule.epochs is the total for a resume. "
                         "Use initialization.warm_start with a new output directory instead.")
    provenance = {
        "schema": "morai_vision_encoder_run_v2",
        "initialization": state.run_initialization,
        "split_sha256": sha256_file(config.split_file),
        "traffic_label_provenance": state.traffic_label_provenance,
        "implementation_sha256": state.implementation_sha256,
        "actual_base_lr": base_lr,
        "actual_rgb_lr": rgb_lr,
        "config_path": str(config.config_path),
        "config_sha256": config.config_sha256,
        "config": config.source_document(),
        "args": config.flat_dict(),
    }
    # Resume appends a new audit entry without replacing the original run manifest.
    provenance_path = config.output / "run_provenance.json"
    if config.resume and provenance_path.exists():
        provenance_path = config.output / ("resume_provenance_{}.json".format(time.time_ns()))
    with provenance_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, sort_keys=True), flush=True)

    history_path = config.output / "metrics.jsonl"
    last_path = config.output / "checkpoint-last.pth.tar"
    best_path = config.output / "model-best.pth.tar"
    started = time.time()
    for epoch in range(start_epoch, config.epochs):
        apply_epoch_lr(optimizer, base_lr, rgb_lr, epoch, config)
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch, amp_enabled
        )
        val_metrics = validate(model, val_loader, device, config, amp_enabled)
        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "rgb_lr": optimizer.param_groups[1]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True))
        is_best = val_metrics["fde_m"] < best_fde
        if is_best:
            best_fde = val_metrics["fde_m"]
        save_checkpoint(last_path, model, optimizer, scaler, epoch, best_fde, config, state,
                        val_metrics)
        if is_best:
            shutil.copy2(last_path, best_path)

    selected_path = best_path if config.feature_checkpoint == "best" else last_path
    if not selected_path.is_file():
        raise RuntimeError("Training produced no selected feature checkpoint")
    sample_inputs, _ = next(iter(val_loader))
    sample_inputs = move_to_device(sample_inputs, device)
    load_result, shape = verify_feature_contract(selected_path, sample_inputs, device)
    report = {
        "best_checkpoint": str(best_path),
        "selected_checkpoint": str(selected_path),
        "checkpoint_selection": config.feature_checkpoint,
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "run_initialization": state.run_initialization,
        "traffic_label_provenance": state.traffic_label_provenance,
        "implementation_sha256": state.implementation_sha256,
        "strict_load": load_result,
        "connector_token_shape": list(shape),
        "lidar_axis_policy": config.lidar_axis_policy,
        "velocity_source": "current_ego_speed",
        "target_point_source": "global_path_x_command_y_command_training_head_only",
        "waypoint_gt_source": "future_synchronized_ego_positions",
    }
    (config.output / "feature_contract.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report
