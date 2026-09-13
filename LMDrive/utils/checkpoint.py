"""Checkpoint saving, loading and resuming for both training stages.

Stage 1 files use the timm layout of LMDrive ({"state_dict", "optimizer", "epoch", ...}),
so the released vision encoder checkpoint and new ones are interchangeable.
Stage 2 files use the LAVIS layout ({"model", "optimizer", "scaler", "epoch", "config"})
and store only trainable weights; frozen LLM and vision encoder weights come from
their own files.
"""
import logging
import operator
import os
import shutil
import time
from collections import OrderedDict

import torch

from .distributed import unwrap_model

logger = logging.getLogger(__name__)


def load_checkpoint_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def strip_module_prefix(state_dict):
    return OrderedDict((k[len("module."):] if k.startswith("module.") else k, v) for k, v in state_dict.items())


def load_vision_encoder_weights(model, path, strict=True):
    checkpoint = load_checkpoint_file(path)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    result = model.load_state_dict(strip_module_prefix(state_dict), strict=strict)
    logger.info("Loaded vision encoder weights from %s", path)
    return result


def resume_vision_encoder_training(model, path, optimizer=None, scaler=None):
    """Restore model/optimizer/AMP state and return the epoch to continue from."""
    checkpoint = load_checkpoint_file(path)
    unwrap_model(model).load_state_dict(strip_module_prefix(checkpoint["state_dict"]))
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "amp_scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["amp_scaler"])
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return 0
    return epoch + 1 if checkpoint.get("version", 1) > 1 else epoch


def _retry(action, attempts=3, delay=0.25):
    """Run action, retrying with growing pauses; returns the last error or None.

    On Windows drive mounts another process can hold a freshly written checkpoint,
    which makes moving or deleting it fail; on Linux filesystems the first try wins.
    """
    for attempt in range(attempts):
        try:
            action()
            return None
        except OSError as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(delay * 2 ** attempt)
    return last_error


def _replace_file(source, destination):
    error = _retry(lambda: os.replace(source, destination))
    if error is None:
        return
    logger.warning("Could not move %s onto %s (%s); copying instead", source, destination, error)
    shutil.copyfile(source, destination)
    if _retry(lambda: os.remove(source)) is not None:
        logger.warning("Left the temporary checkpoint %s in place; the next save overwrites it", source)


def _link_or_copy(source, destination):
    if os.path.exists(destination):
        try:
            os.remove(destination)
        except OSError:
            pass
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


class TopKCheckpointSaver:
    """Keeps last.pth.tar, model_best.pth.tar and the best max_history checkpoint-<epoch>.pth.tar."""

    def __init__(self, checkpoint_dir, max_history=5, decreasing=True):
        assert max_history >= 1
        self.checkpoint_dir = checkpoint_dir
        self.max_history = max_history
        self.decreasing = decreasing  # lower metric is better
        self.cmp = operator.lt if decreasing else operator.gt
        self.checkpoint_files = []  # (path, metric), best first
        self.best_metric = None
        self.best_epoch = None

    def save(self, epoch, state, metric=None):
        tmp_path = os.path.join(self.checkpoint_dir, "tmp.pth.tar")
        last_path = os.path.join(self.checkpoint_dir, "last.pth.tar")
        torch.save(state, tmp_path)
        _replace_file(tmp_path, last_path)

        worst = self.checkpoint_files[-1] if self.checkpoint_files else None
        if len(self.checkpoint_files) < self.max_history or metric is None or self.cmp(metric, worst[1]):
            if len(self.checkpoint_files) >= self.max_history:
                for path, _ in self.checkpoint_files[self.max_history - 1:]:
                    try:
                        os.remove(path)
                    except OSError as error:
                        logger.error("Could not delete %s: %s", path, error)
                self.checkpoint_files = self.checkpoint_files[:self.max_history - 1]
            path = os.path.join(self.checkpoint_dir, "checkpoint-%d.pth.tar" % epoch)
            _link_or_copy(last_path, path)
            self.checkpoint_files.append((path, metric))
            self.checkpoint_files.sort(key=lambda item: item[1], reverse=not self.decreasing)
            logger.info("Current checkpoints:\n%s", "\n".join(" %s" % (item,) for item in self.checkpoint_files))

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                _link_or_copy(last_path, os.path.join(self.checkpoint_dir, "model_best.pth.tar"))
        return self.best_metric, self.best_epoch


def _frozen_parameter_names(model):
    return {name for name, param in model.named_parameters(remove_duplicate=False) if not param.requires_grad}


def save_lmdrive_checkpoint(path, model, optimizer=None, scaler=None, config=None, epoch=None):
    model = unwrap_model(model)
    frozen = _frozen_parameter_names(model)
    state_dict = OrderedDict((k, v) for k, v in model.state_dict().items() if k not in frozen)
    torch.save({
        "model": state_dict,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
    }, path)


def load_lmdrive_checkpoint(model, path, optimizer=None, scaler=None):
    """Load trainable weights (strict=False) and optionally optimizer/scaler; returns (epoch, result)."""
    model = unwrap_model(model)
    checkpoint = load_checkpoint_file(path)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    result = model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    frozen = _frozen_parameter_names(model)
    missing_trainable = [k for k in result.missing_keys if k not in frozen]
    if missing_trainable:
        logger.warning("Checkpoint %s lacks %d trainable tensors, e.g. %s",
                       path, len(missing_trainable), missing_trainable[:5])
    if result.unexpected_keys:
        logger.warning("Checkpoint %s has %d unexpected tensors, e.g. %s",
                       path, len(result.unexpected_keys), result.unexpected_keys[:5])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info("Loaded LMDrive checkpoint %s (epoch %s)", path, checkpoint.get("epoch"))
    return checkpoint.get("epoch"), result
