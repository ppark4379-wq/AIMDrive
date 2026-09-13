#!/usr/bin/env python3
"""Evaluate an LMDrive checkpoint on a dataset split, without a simulator.

  stage 1  python evaluate.py --config configs/vision_encoder_train.yaml \
               --checkpoint outputs/checkpoints/<run>/model_best.pth.tar
  stage 2  python evaluate.py --config configs/lmdrive_train.yaml \
               --checkpoint outputs/checkpoints/<run>/checkpoint_best.pth

Stage 1 reports the pre-training losses and traffic-light / stop-sign accuracy.
Stage 2 reports the waypoint L1 loss, the end-flag loss and end-flag accuracy.
Metrics are printed and written to outputs/logs/eval-<stage>-<time>.json.
Closed-loop driving evaluation needs a simulator and is not part of this project.
"""
import argparse
import json
import logging
import os
import time
from collections import OrderedDict

import torch

from utils.config import load_config, resolve_user_path
from utils.distributed import barrier, cleanup_distributed, init_distributed, reduce_mean
from utils.logger import MetricLogger, setup_logging
from utils.metrics import AverageMeter, accuracy

logger = logging.getLogger("evaluate")

VISION_ENCODER_LOSS_NAMES = (
    "loss",
    "loss_traffic",
    "loss_velocity",
    "loss_waypoints",
    "loss_traffic_light_state",
    "loss_stop_sign",
)
VISION_ENCODER_METRIC_NAMES = VISION_ENCODER_LOSS_NAMES + ("acc_traffic_light_state", "acc_stop_sign")


def move_to_device(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    if isinstance(sample, dict):
        return {key: move_to_device(value, device) for key, value in sample.items()}
    if isinstance(sample, list):
        return [move_to_device(value, device) for value in sample]
    return sample


def cuda_autocast(enabled):
    return torch.autocast("cuda", enabled=bool(enabled) and torch.cuda.is_available())


@torch.no_grad()
def evaluate_vision_encoder(model, loader, criterion, ctx, amp=False, writer=None, epoch=0, log_interval=50):
    from utils.visualization import vision_encoder_images

    meters = OrderedDict((name, AverageMeter()) for name in VISION_ENCODER_METRIC_NAMES)
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch_size = inputs["rgb_front"].size(0)
        inputs = move_to_device(inputs, ctx.device)
        targets = move_to_device(targets, ctx.device)
        with cuda_autocast(amp):
            outputs = model(inputs)

        values = criterion(outputs, targets)
        values["acc_traffic_light_state"] = accuracy(outputs[2], targets[3])[0]
        values["acc_stop_sign"] = accuracy(outputs[3], targets[6])[0]
        if ctx.distributed:
            values = OrderedDict((name, reduce_mean(value.detach(), ctx.world_size)) for name, value in values.items())
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        for name in VISION_ENCODER_METRIC_NAMES:
            meters[name].update(values[name].item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if ctx.is_main and (batch_idx == last_idx or batch_idx % log_interval == 0):
            logger.info(
                "Test: [%4d/%d]  Time: %.3f (%.3f)  Loss: %.4f (%.4f)  Loss(traffic): %.4f (%.4f)  "
                "Loss(waypoints): %.4f (%.4f)  Loss(light): %.6f (%.4f)  Acc(light): %.2f (%.2f)",
                batch_idx, last_idx, batch_time.val, batch_time.avg,
                meters["loss"].val, meters["loss"].avg,
                meters["loss_traffic"].val, meters["loss_traffic"].avg,
                meters["loss_waypoints"].val, meters["loss_waypoints"].avg,
                meters["loss_traffic_light_state"].val, meters["loss_traffic_light_state"].avg,
                meters["acc_traffic_light_state"].val, meters["acc_traffic_light_state"].avg,
            )
            if writer is not None:
                for tag, image in vision_encoder_images(inputs, outputs, targets).items():
                    writer.add_image("val/%d_%s" % (batch_idx, tag), image, epoch)

    if writer is not None:
        for name in VISION_ENCODER_METRIC_NAMES:
            writer.add_scalar("val/" + name, meters[name].avg, epoch)
    metrics = OrderedDict((name, meters[name].avg) for name in VISION_ENCODER_METRIC_NAMES)
    metrics["l1_error"] = metrics["loss"]  # name LMDrive used for this same total loss
    return metrics


@torch.no_grad()
def evaluate_lmdrive(model, loader, ctx, writer=None, epoch=0):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    sums = OrderedDict()
    batches = 0
    for samples in metric_logger.log_every(loader, 10, "Evaluation"):
        samples = move_to_device(samples, ctx.device)
        with cuda_autocast(True):
            output = model(samples)
        batches += 1
        for key, value in output.items():
            sums[key] = sums.get(key, 0.0) + (value.item() if torch.is_tensor(value) else float(value))
    barrier()

    if ctx.distributed:
        packed = torch.tensor([batches] + list(sums.values()), dtype=torch.float64, device=ctx.device)
        torch.distributed.all_reduce(packed)
        batches = packed[0].item()
        sums = OrderedDict(zip(sums.keys(), packed[1:].tolist()))
    metrics = OrderedDict((key, value / max(batches, 1)) for key, value in sums.items())
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar("val/%s_epoch" % key, value, epoch)
    logger.info("Eval Epoch %s, %s", epoch, ", ".join("%s: %.3f" % item for item in metrics.items()))
    metrics["agg_metrics"] = metrics.get("loss", float("inf"))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the training config of the checkpoint's stage")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", help="data split defined in the config (default: val)")
    parser.add_argument("--output", default="", help="metrics JSON path")
    parser.add_argument("--options", nargs="*", default=[], help="config overrides written as key.sub=value")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.options)
    ctx = init_distributed()
    setup_logging(ctx.is_main)
    checkpoint = resolve_user_path(args.checkpoint)
    data_cfg = cfg["data"]
    batch_size = int(data_cfg.get("val_batch_size", data_cfg["batch_size"]))
    num_workers = int(data_cfg.get("num_workers", 4))
    stage = cfg["stage"]

    if stage == "vision_encoder":
        from datasets.carla_perception_dataset import build_perception_dataset, create_perception_loader
        from models.vision_encoder import create_vision_encoder
        from utils.checkpoint import load_vision_encoder_weights
        from utils.losses import VisionEncoderLoss

        model = create_vision_encoder(cfg["model"], pretrained_backbone=False)
        load_vision_encoder_weights(model, checkpoint, strict=True)
        model.to(ctx.device)
        dataset = build_perception_dataset(data_cfg, args.split)
        loader = create_perception_loader(
            dataset, batch_size=batch_size, is_training=False,
            input_size=data_cfg.get("input_size", (3, 224, 224)),
            multi_view_input_size=data_cfg.get("multi_view_input_size", (3, 128, 128)),
            num_workers=num_workers, distributed=ctx.distributed,
            pin_memory=bool(data_cfg.get("pin_memory", False)),
        )
        metrics = evaluate_vision_encoder(
            model, loader, VisionEncoderLoss(), ctx, amp=cfg["training"].get("amp", False),
            log_interval=int(cfg["training"].get("log_interval", 50)))
    elif stage == "lmdrive":
        from datasets.carla_instruction_dataset import build_instruction_dataset, create_instruction_loader
        from models.lmdrive import LMDriveModel
        from utils.checkpoint import load_lmdrive_checkpoint

        model = LMDriveModel.from_config(cfg["model"])
        load_lmdrive_checkpoint(model, checkpoint)
        model.to(ctx.device)
        dataset = build_instruction_dataset(data_cfg, args.split)
        loader = create_instruction_loader(dataset, batch_size, is_training=False,
                                           num_workers=num_workers, distributed=ctx.distributed)
        metrics = evaluate_lmdrive(model, loader, ctx)
    else:
        raise ValueError("Unknown stage %r" % stage)

    if ctx.is_main:
        output = args.output or os.path.join(
            cfg["output_dir"], "logs", "eval-%s-%s.json" % (stage, time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        record = OrderedDict(stage=stage, checkpoint=checkpoint, split=args.split, samples=len(dataset),
                             metrics=metrics)
        with open(output, "w") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record, indent=2))
        logger.info("metrics written to %s", output)
    cleanup_distributed()


if __name__ == "__main__":
    main()
