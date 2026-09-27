"""Camera image transforms used by LMDrive.

CARLA cameras are 800x600.  The front camera is resized to a fixed 341x256 and
centre-cropped to 224; side and rear cameras go 195x146 -> crop 128.  The centre
("focus") view is a 128x128 crop of the un-resized front image.
"""
import random

import torch
from torchvision import transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Fixed resize applied before the centre crop, keyed by crop size.
RESIZE_BEFORE_CROP = {
    112: (170, 128),
    128: (195, 146),
    224: (341, 256),
    256: (340, 288),
}


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        return pil_img.resize(self.size)


class RandomResize:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, pil_img):
        width, height = pil_img.size
        r_scale = (self.scale[1] - self.scale[0]) * random.random() + self.scale[0]
        return pil_img.resize((int(width * r_scale), int(height * r_scale)))


def create_camera_transform(input_size, is_training=False, scale=None, color_jitter=0.0, need_scale=True,
                            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, hue_jitter=None):
    """input_size is an int or (C, H, W); training adds random resize and colour jitter.

    hue_jitter overrides the hue amplitude only.  MORAI traffic lights are
    told apart by colour, so hue_jitter=0 keeps that signal intact while
    brightness/contrast/saturation augmentation stays on.
    """
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
        crop = input_size[-1]
    else:
        img_size = input_size
        crop = input_size

    steps = []
    if need_scale:
        size = RESIZE_BEFORE_CROP.get(crop, (int((crop + 32) / 3.0 * 4.0), crop + 32))
        steps.append(Resize2FixedSize(size))
    if is_training:
        if scale:
            steps.append(RandomResize(scale))
        if color_jitter > 0:
            steps.append(transforms.ColorJitter(
                brightness=color_jitter, contrast=color_jitter, saturation=color_jitter,
                hue=color_jitter if hue_jitter is None else hue_jitter))
    steps.append(transforms.CenterCrop(img_size))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    return transforms.Compose(steps)
