"""Vision training configuration, initialization, freeze policy and optimizer setup."""
import math
import torch
from torch import nn
from models.ve.vision_encoder import create_vision_encoder
from data.src.loaders.morai_sensor_record_dataset import digest
from utils.config.vision_encoder_config import (
    ConfigError, VisionEncoderTrainConfig, load_config, require_config,
)

GROUPS = ('rgb_backbone.', 'lidar_backbone.', 'encoder.', 'decoder.',
          'traffic_pred_head.', 'traffic_light_pred_head.', 'waypoints_generator.')

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


def make_model(checkpoint, expected_sha256, device, initialization='lmdrive_base'):
    if digest(checkpoint) != expected_sha256:
        raise ValueError('Initial checkpoint SHA256 mismatch')
    # Only trusted tensor data; no old timm pickle imports needed.
    doc = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    if not isinstance(doc,dict) or 'state_dict' not in doc:
        raise ValueError('Expected LMDrive state_dict checkpoint')
    if initialization == 'lmdrive_base' and ('pretraining' in doc or 'morai_initialization' in doc):
        raise ValueError('This path starts from the base LMDrive checkpoint, not an earlier MORAI run')
    model = create_vision_encoder({'name':'memfuser_baseline_e1d3'}, pretrained_backbone=False)
    model.load_state_dict(doc['state_dict'], strict=True)
    model.requires_grad_(True)
    model.stop_sign_head.requires_grad_(False)
    return model.to(device)


# Partial scopes for geometry-only fine-tuning; the RGB backbone, encoder,
# native traffic-light head and waypoint generator stay at the reference weights.
PARTIAL_SCOPES = {
    'lidar_head': ('lidar_backbone.', 'traffic_pred_head.'),
    'decoder_head': ('lidar_backbone.', 'decoder.', 'traffic_pred_head.'),
}


def configure_trainable_scope(model, scope):
    if scope == 'full':
        model.requires_grad_(True)
        model.stop_sign_head.requires_grad_(False)
        return tuple(GROUPS)
    if scope in PARTIAL_SCOPES:
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.startswith(PARTIAL_SCOPES[scope]):
                parameter.requires_grad_(True)
        return PARTIAL_SCOPES[scope]
    model.requires_grad_(False)
    model.traffic_pred_head.requires_grad_(True)
    return ('traffic_pred_head.',)


def train_mode(model, scope='full'):
    if scope in PARTIAL_SCOPES:
        model.eval()  # BN running statistics stay frozen in every scope; no dropout
        return
    if scope == 'object_head':
        model.eval()
        model.traffic_pred_head.train()
        return
    model.train()
    # v3_lav physical batch 1: keep pretrained BN running stats, train affine.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
