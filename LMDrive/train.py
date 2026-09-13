#!/usr/bin/env python3
"""Train LMDrive from a YAML config.

  stage 1, vision encoder pre-training  python train.py --config configs/vision_encoder_train.yaml
  stage 2, instruction fine-tuning       python train.py --config configs/lmdrive_train.yaml
  multi-GPU                              torchrun --nproc_per_node=8 train.py --config ...
  override values                        python train.py --config ... --options data.batch_size=8 training.amp=true

Checkpoints are written to outputs/checkpoints/<run>/ and logs (TensorBoard events,
summary.csv or log.txt, config.yaml, train.log) to outputs/logs/<run>/.
"""
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from evaluate import VISION_ENCODER_LOSS_NAMES, cuda_autocast, evaluate_lmdrive, evaluate_vision_encoder, move_to_device
from utils.config import load_config, save_config
from utils.distributed import barrier, cleanup_distributed, init_distributed, reduce_mean, unwrap_model
from utils.logger import (
    MetricLogger,
    SmoothedValue,
    append_csv_row,
    append_json_line,
    attach_log_file,
    create_tensorboard_writer,
    setup_logging,
)
from utils.metrics import AverageMeter

logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="training config, e.g. configs/vision_encoder_train.yaml")
    parser.add_argument("--options", nargs="*", default=[], help="config overrides written as key.sub=value")
    return parser.parse_args()


def seed_everything(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def create_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def prepare_run(cfg, ctx):
    """Create outputs/checkpoints/<run> and outputs/logs/<run> on the main process."""
    run_name = cfg["training"].get("run_name") or "%s-%s" % (
        cfg.get("experiment", cfg["stage"]), datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(cfg["output_dir"], "checkpoints", run_name)
    log_dir = os.path.join(cfg["output_dir"], "logs", run_name)
    if ctx.is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        save_config(cfg, os.path.join(log_dir, "config.yaml"))
        attach_log_file(os.path.join(log_dir, "train.log"))
        logger.info("run %s | checkpoints %s | logs %s", run_name, checkpoint_dir, log_dir)
    return checkpoint_dir, log_dir


# --------------------------------------------------------------------------- stage 1

def build_vision_encoder_optimizer(model, opt_cfg, batch_size, world_size):
    """AdamW with learning rates scaled by the global batch size / 512, as in LMDrive."""
    if str(opt_cfg.get("name", "adamw")).lower() != "adamw":
        raise ValueError("Only AdamW is supported, as used by LMDrive")
    lr_scale = batch_size * world_size / 512.0
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    kwargs = dict(lr=float(opt_cfg["lr"]) * lr_scale, weight_decay=weight_decay, eps=float(opt_cfg.get("eps", 1e-8)))

    if opt_cfg.get("backbone_lr") is not None:
        # The camera CNN gets its own learning rate; weight decay applies to every tensor.
        backbone, others = [], []
        for name, param in model.named_parameters():
            (backbone if "backbone" in name and "lidar" not in name else others).append(param)
        logger.info("%d tensors in the CNN backbone, %d tensors in other modules", len(backbone), len(others))
        groups = [{"params": others}, {"params": backbone, "lr": float(opt_cfg["backbone_lr"]) * lr_scale}]
        return torch.optim.AdamW(groups, **kwargs)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if len(param.shape) == 1 or name.endswith(".bias") else decay).append(param)
    kwargs["weight_decay"] = 0.0
    groups = [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]
    return torch.optim.AdamW(groups, **kwargs)


def train_vision_encoder_one_epoch(epoch, model, loader, optimizer, criterion, ctx, scaler=None,
                                   clip_grad_norm=None, writer=None, log_interval=50):
    from utils.visualization import vision_encoder_images

    meters = OrderedDict((name, AverageMeter()) for name in VISION_ENCODER_LOSS_NAMES)
    batch_time, data_time = AverageMeter(), AverageMeter()
    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end)
        batch_size = inputs["rgb_front"].size(0)
        inputs = move_to_device(inputs, ctx.device)
        targets = move_to_device(targets, ctx.device)

        with cuda_autocast(scaler is not None):
            outputs = model(inputs)
            losses = criterion(outputs, targets)
        loss = losses["loss"]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        num_updates += 1
        for name in VISION_ENCODER_LOSS_NAMES:
            meters[name].update(losses[name].item(), batch_size)
        batch_time.update(time.time() - end)

        if batch_idx == last_idx or batch_idx % log_interval == 0:
            lrs = [group["lr"] for group in optimizer.param_groups]
            reduced = OrderedDict(
                (name, reduce_mean(losses[name].detach(), ctx.world_size).item()) for name in VISION_ENCODER_LOSS_NAMES)
            if ctx.is_main:
                if writer is not None:
                    for name, value in reduced.items():
                        writer.add_scalar("train/" + name, value, num_updates)
                    writer.add_scalar("train/lr", sum(lrs) / len(lrs), num_updates)
                    for tag, image in vision_encoder_images(inputs, outputs, targets).items():
                        writer.add_image("train/" + tag, image, num_updates)
                logger.info(
                    "Train: %d [%4d/%d (%3.0f%%)]  Loss(traffic): %.6f (%.4f)  Loss(waypoints): %.6f (%.4f)  "
                    "Loss(light): %.6f (%.4f)  Loss: %.6f (%.4f)  Time: %.3fs, %.2f/s  LR: %.3e  Data: %.3f",
                    epoch, batch_idx, len(loader), 100.0 * batch_idx / max(last_idx, 1),
                    meters["loss_traffic"].val, meters["loss_traffic"].avg,
                    meters["loss_waypoints"].val, meters["loss_waypoints"].avg,
                    meters["loss_traffic_light_state"].val, meters["loss_traffic_light_state"].avg,
                    meters["loss"].val, meters["loss"].avg,
                    batch_time.val, batch_size * ctx.world_size / max(batch_time.val, 1e-9),
                    sum(lrs) / len(lrs), data_time.val,
                )
        end = time.time()
    return OrderedDict(loss=meters["loss"].avg)


def train_vision_encoder(cfg, ctx):
    from datasets.carla_perception_dataset import build_perception_dataset, create_perception_loader
    from models.vision_encoder import create_vision_encoder
    from utils.checkpoint import TopKCheckpointSaver, load_vision_encoder_weights, resume_vision_encoder_training
    from utils.losses import VisionEncoderLoss
    from utils.schedulers import CosineLRScheduler

    data_cfg, model_cfg = cfg["data"], cfg["model"]
    opt_cfg, sched_cfg, train_cfg = cfg["optimizer"], cfg["scheduler"], cfg["training"]
    torch.backends.cudnn.benchmark = True
    seed_everything(int(cfg.get("seed", 42)), ctx.rank)
    checkpoint_dir, log_dir = prepare_run(cfg, ctx)

    model = create_vision_encoder(model_cfg)
    if train_cfg.get("initial_checkpoint"):
        load_vision_encoder_weights(model, train_cfg["initial_checkpoint"], strict=True)
    logger.info("Model %s created, param count: %d",
                model_cfg.get("name", "vision_encoder"), sum(p.numel() for p in model.parameters()))
    model.to(ctx.device)

    batch_size = int(data_cfg["batch_size"])
    optimizer = build_vision_encoder_optimizer(model, opt_cfg, batch_size, ctx.world_size)
    amp = bool(train_cfg.get("amp", False))
    scaler = create_grad_scaler() if amp else None

    start_epoch = 0
    if train_cfg.get("resume"):
        start_epoch = resume_vision_encoder_training(model, train_cfg["resume"], optimizer, scaler)
        logger.info("Resumed %s, continuing at epoch %d", train_cfg["resume"], start_epoch)
    if ctx.distributed:
        model = DistributedDataParallel(model, device_ids=[ctx.local_rank], find_unused_parameters=False)

    epochs = int(sched_cfg["epochs"])
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        lr_min=float(sched_cfg.get("min_lr", 0.0)),
        warmup_t=int(sched_cfg.get("warmup_epochs", 0)),
        warmup_lr_init=float(sched_cfg.get("warmup_lr", 0.0)),
    )
    num_epochs = epochs + int(sched_cfg.get("cooldown_epochs", 0))
    if start_epoch > 0:
        scheduler.step(start_epoch)
    logger.info("Scheduled epochs: %d", num_epochs)

    augmentation = data_cfg.get("augmentation") or {}
    loader_args = dict(
        input_size=data_cfg.get("input_size", (3, 224, 224)),
        multi_view_input_size=data_cfg.get("multi_view_input_size", (3, 128, 128)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        distributed=ctx.distributed,
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )
    train_set = build_perception_dataset(data_cfg, "train")
    val_set = build_perception_dataset(data_cfg, "val")
    train_loader = create_perception_loader(
        train_set, batch_size=batch_size, is_training=True,
        scale=augmentation.get("scale", (1.0, 1.0)), color_jitter=float(augmentation.get("color_jitter", 0.0)),
        **loader_args)
    val_loader = create_perception_loader(
        val_set, batch_size=int(data_cfg.get("val_batch_size", batch_size)), is_training=False, **loader_args)
    logger.info("train: %d frames, %d samples per epoch | val: %d frames, %d samples",
                len(train_set.route_frames), len(train_set), len(val_set.route_frames), len(val_set))

    criterion = VisionEncoderLoss()
    writer = create_tensorboard_writer(log_dir) if ctx.is_main else None
    saver = None
    if ctx.is_main:
        saver = TopKCheckpointSaver(checkpoint_dir, max_history=int(train_cfg.get("checkpoint_history", 5)),
                                    decreasing=bool(train_cfg.get("eval_metric_decreasing", True)))
    eval_metric = train_cfg.get("eval_metric", "loss")
    log_interval = int(train_cfg.get("log_interval", 50))
    clip_grad_norm = opt_cfg.get("clip_grad_norm")
    clip_grad_norm = float(clip_grad_norm) if clip_grad_norm is not None else None
    summary_path = os.path.join(log_dir, "summary.csv")

    best_metric = best_epoch = None
    try:
        for epoch in range(start_epoch, num_epochs):
            if ctx.distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_metrics = train_vision_encoder_one_epoch(
                epoch, model, train_loader, optimizer, criterion, ctx, scaler=scaler,
                clip_grad_norm=clip_grad_norm, writer=writer, log_interval=log_interval)
            eval_metrics = evaluate_vision_encoder(
                model, val_loader, criterion, ctx, amp=amp, writer=writer, epoch=epoch, log_interval=log_interval)
            scheduler.step(epoch + 1)

            if ctx.is_main:
                row = OrderedDict(epoch=epoch)
                row.update(("train_" + key, value) for key, value in train_metrics.items())
                row.update(("eval_" + key, value) for key, value in eval_metrics.items())
                append_csv_row(summary_path, row, write_header=best_metric is None)
                state = {
                    "epoch": epoch,
                    "arch": model_cfg.get("name", "vision_encoder"),
                    "state_dict": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "version": 2,
                    "config": cfg,
                    "metric": eval_metrics[eval_metric],
                }
                if scaler is not None:
                    state["amp_scaler"] = scaler.state_dict()
                best_metric, best_epoch = saver.save(epoch, state, metric=eval_metrics[eval_metric])
    except KeyboardInterrupt:
        logger.warning("Training interrupted")
    if best_metric is not None:
        logger.info("*** Best metric: %s (epoch %s)", best_metric, best_epoch)
    if writer is not None:
        writer.close()


# --------------------------------------------------------------------------- stage 2

def train_lmdrive_one_epoch(epoch, model, loader, optimizer, scheduler, ctx, scaler=None, writer=None,
                            log_freq=50, accum_grad_iters=1):
    model.train()
    use_amp = scaler is not None
    iters_per_epoch = len(loader)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    logger.info("Start training epoch %d, %d iters per inner epoch.", epoch, iters_per_epoch)

    for i, samples in enumerate(metric_logger.log_every(loader, log_freq, "Train: data epoch: [%d]" % epoch)):
        samples = move_to_device(samples, ctx.device)
        scheduler.step(cur_epoch=epoch, cur_step=i)
        with cuda_autocast(use_amp):
            output = model(samples)
            loss = output["loss"] / accum_grad_iters

        if writer is not None:
            for key, value in output.items():
                scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                writer.add_scalar("train/%s_iter" % key, scalar, epoch * iters_per_epoch + i)

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i + 1) % accum_grad_iters == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(**output)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    if writer is not None:
        for key, meter in metric_logger.meters.items():
            writer.add_scalar("train/%s_epoch" % key, meter.global_avg, epoch)
    logger.info("Averaged stats: %s", metric_logger.global_avg())
    return {key: "{:.3f}".format(meter.global_avg) for key, meter in metric_logger.meters.items()}


def train_lmdrive(cfg, ctx):
    from datasets.carla_instruction_dataset import build_instruction_dataset, create_instruction_loader
    from models.lmdrive import LMDriveModel
    from utils.checkpoint import load_lmdrive_checkpoint, save_lmdrive_checkpoint
    from utils.schedulers import LinearWarmupCosineLRScheduler

    data_cfg, opt_cfg, sched_cfg, train_cfg = cfg["data"], cfg["optimizer"], cfg["scheduler"], cfg["training"]
    seed_everything(int(cfg.get("seed", 42)), ctx.rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint_dir, log_dir = prepare_run(cfg, ctx)

    train_set = build_instruction_dataset(data_cfg, "train")
    val_set = build_instruction_dataset(data_cfg, "val") if data_cfg.get("val") else None
    logger.info("train clips: %d | val clips: %s", len(train_set), len(val_set) if val_set is not None else "none")

    model = LMDriveModel.from_config(cfg["model"])
    model.to(ctx.device)
    train_model = model
    if ctx.distributed:
        train_model = DistributedDataParallel(model, device_ids=[ctx.local_rank], find_unused_parameters=False)

    optim_params = model.get_optimizer_params(float(opt_cfg.get("weight_decay", 0.05)))
    logger.info("number of trainable parameters: %d",
                sum(p.numel() for group in optim_params for p in group["params"]))
    init_lr = float(opt_cfg["lr"])
    optimizer = torch.optim.AdamW(optim_params, lr=init_lr, betas=(0.9, float(opt_cfg.get("beta2", 0.999))))
    scaler = create_grad_scaler() if bool(train_cfg.get("amp", True)) else None

    start_epoch = 0
    if train_cfg.get("resume"):
        epoch, _ = load_lmdrive_checkpoint(model, train_cfg["resume"], optimizer=optimizer, scaler=scaler)
        start_epoch = 0 if epoch is None else epoch + 1
        logger.info("Resumed %s, continuing at epoch %d", train_cfg["resume"], start_epoch)

    max_epoch = int(sched_cfg["epochs"])
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer,
        max_epoch=max_epoch,
        min_lr=float(sched_cfg["min_lr"]),
        init_lr=init_lr,
        warmup_steps=int(sched_cfg.get("warmup_steps", 0)),
        warmup_start_lr=float(sched_cfg.get("warmup_lr", -1)),
    )

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = create_instruction_loader(train_set, batch_size, is_training=True,
                                             num_workers=num_workers, distributed=ctx.distributed)
    val_loader = None
    if val_set is not None:
        val_loader = create_instruction_loader(val_set, int(data_cfg.get("val_batch_size", batch_size)),
                                               is_training=False, num_workers=num_workers,
                                               distributed=ctx.distributed)

    writer = create_tensorboard_writer(log_dir) if ctx.is_main else None
    stats_path = os.path.join(log_dir, "log.txt")
    log_freq = int(train_cfg.get("log_interval", 50))
    accum_grad_iters = int(train_cfg.get("accum_grad_iters", 1))
    best_loss, best_epoch = float("inf"), 0

    def save(name, epoch):
        if ctx.is_main:
            path = os.path.join(checkpoint_dir, name)
            logger.info("Saving checkpoint at epoch %d to %s", epoch, path)
            save_lmdrive_checkpoint(path, model, optimizer, scaler, cfg, epoch)

    for epoch in range(start_epoch, max_epoch):
        if ctx.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train_lmdrive_one_epoch(
            epoch, train_model, train_loader, optimizer, scheduler, ctx, scaler=scaler, writer=writer,
            log_freq=log_freq, accum_grad_iters=accum_grad_iters)
        if ctx.is_main:
            append_json_line(stats_path, {"epoch": epoch, **{"train_" + k: v for k, v in train_stats.items()}})

        if val_loader is not None:
            val_stats = evaluate_lmdrive(train_model, val_loader, ctx, writer=writer, epoch=epoch)
            if val_stats["agg_metrics"] < best_loss:
                best_loss, best_epoch = val_stats["agg_metrics"], epoch
                save("checkpoint_best.pth", epoch)
            val_stats["best_epoch"] = best_epoch
            if ctx.is_main:
                append_json_line(stats_path, {"epoch": epoch, **{"val_" + k: v for k, v in val_stats.items()}})
        else:
            save("checkpoint_%d.pth" % epoch, epoch)
        if train_cfg.get("save_last_checkpoint", True):
            save("checkpoint_last.pth", epoch)
        barrier()

    if writer is not None:
        writer.close()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.options)
    ctx = init_distributed()
    setup_logging(ctx.is_main)
    try:
        if cfg["stage"] == "vision_encoder":
            train_vision_encoder(cfg, ctx)
        elif cfg["stage"] == "lmdrive":
            train_lmdrive(cfg, ctx)
        else:
            raise ValueError("Unknown stage %r (expected vision_encoder or lmdrive)" % cfg["stage"])
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
