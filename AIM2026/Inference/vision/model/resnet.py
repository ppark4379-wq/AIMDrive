"""ResNet-D backbones for the camera branch of the LMDrive vision encoder.

Ported from the timm fork shipped with LMDrive (timm/models/resnet.py,
Copyright 2020 Ross Wightman, Apache-2.0).  Only the "D" variants LMDrive uses
are kept: a deep 3x3 stem, average-pool downsampling, and no attention,
anti-aliasing, DropBlock or DropPath.  Module names, construction order and
weight initialisation are unchanged, so ImageNet weights, released LMDrive
checkpoints and seeded initialisation reproduce the original exactly.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

IMAGENET_WEIGHT_URLS = {
    "resnet18d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth",
    "resnet26d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth",
    "resnet50d": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, outplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.act2 = nn.ReLU(inplace=True)
        # Registered last, as in timm, so named_modules() order (and therefore
        # the order of random initialisation) matches the original.
        self.downsample = downsample
        self.stride = stride

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        return self.act2(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.act3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        return self.act3(x)


def downsample_avg(in_channels, out_channels, stride):
    """ResNet-D shortcut: average pool, then a 1x1 projection."""
    if stride == 1:
        pool = nn.Identity()
    else:
        pool = nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False)
    return nn.Sequential(
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResNetD(nn.Module):
    """ResNet with a deep 32-32-64 stem and average-pool downsampling."""

    def __init__(self, block, layers, in_chans=3, stem_width=32, num_classes=1000):
        super().__init__()
        inplanes = stem_width * 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, inplanes, 3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for stage_index, (planes, num_blocks) in enumerate(zip((64, 128, 256, 512), layers)):
            stride = 1 if stage_index == 0 else 2
            downsample = None
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = downsample_avg(inplanes, planes * block.expansion, stride)
            blocks = []
            for block_index in range(num_blocks):
                blocks.append(block(
                    inplanes,
                    planes,
                    stride if block_index == 0 else 1,
                    downsample if block_index == 0 else None,
                ))
                inplanes = planes * block.expansion
            self.add_module("layer%d" % (stage_index + 1), nn.Sequential(*blocks))

        self.num_features = 512 * block.expansion
        # The ImageNet classifier is never used by LMDrive.  It is still created
        # because its random initialisation advances the RNG exactly as the
        # original timm model did, which keeps seeded runs reproducible.
        self.fc = nn.Linear(self.num_features, num_classes)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if hasattr(module, "zero_init_last_bn"):
                module.zero_init_last_bn()


class ResNetFeatureExtractor(nn.ModuleDict):
    """Stem and four residual stages; returns the last stage feature map.

    Equivalent to timm FeatureListNet(out_indices=[4]), so parameter names such
    as conv1.0.weight and layer4.2.bn3.weight are unchanged.
    """

    STAGES = ("conv1", "bn1", "act1", "maxpool", "layer1", "layer2", "layer3", "layer4")

    def __init__(self, resnet):
        super().__init__(OrderedDict((name, getattr(resnet, name)) for name in self.STAGES))
        self.out_channels = resnet.num_features

    def forward(self, x):
        for module in self.values():
            x = module(x)
        return x


RESNET_LAYOUTS = {
    "resnet18d": (BasicBlock, (2, 2, 2, 2)),
    "resnet26d": (Bottleneck, (2, 2, 2, 2)),
    "resnet50d": (Bottleneck, (3, 4, 6, 3)),
}


def load_imagenet_weights(resnet, name, weights_path=None):
    """Load timm ImageNet weights, downloaded once into the torch hub cache."""
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            IMAGENET_WEIGHT_URLS[name], progress=False, map_location="cpu")
    # timm drops the classifier when it builds a feature extractor.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    result = resnet.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("fc.")]
    if missing or result.unexpected_keys:
        raise RuntimeError("ImageNet weights for %s do not match: missing=%s unexpected=%s"
                           % (name, missing, list(result.unexpected_keys)))


def create_resnet_features(name, pretrained=True, weights_path=None, in_chans=3):
    if name not in RESNET_LAYOUTS:
        raise KeyError("Unknown backbone %r, choose from %s" % (name, sorted(RESNET_LAYOUTS)))
    block, layers = RESNET_LAYOUTS[name]
    resnet = ResNetD(block, layers, in_chans=in_chans)
    if pretrained:
        load_imagenet_weights(resnet, name, weights_path)
    return ResNetFeatureExtractor(resnet)
