"""Build the stage-1 DataLoader and attach the camera transforms it needs.

MoraiPerceptionDataset returns already-decoded PIL images; the transforms are
attached here (not in the dataset) so training and validation can differ in
augmentation while reading exactly the same frames.

Formerly ``vision_encoder/timm/data/carla_loader.py::create_carla_loader``.
"""
import torch

from .transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_camera_transform


def attach_camera_transforms(dataset, input_size=(3, 224, 224), multi_view_input_size=(3, 128, 128),
                             is_training=False, scale=(1.0, 1.0), color_jitter=0.4,
                             mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, hue_jitter=None):
    """Front 224, focus crop 128, side/rear 128 - the LMDrive encoder input contract."""
    dataset.rgb_transform = create_camera_transform(
        input_size, is_training=is_training, scale=scale, color_jitter=color_jitter,
        mean=mean, std=std, hue_jitter=hue_jitter)
    dataset.rgb_center_transform = create_camera_transform(
        128, is_training=is_training, scale=None, color_jitter=color_jitter,
        mean=mean, std=std, need_scale=False, hue_jitter=hue_jitter)
    dataset.multi_view_transform = create_camera_transform(
        multi_view_input_size, is_training=is_training, scale=scale, color_jitter=color_jitter,
        mean=mean, std=std, hue_jitter=hue_jitter)
    return dataset


def create_perception_loader(dataset, input_size=(3, 224, 224), batch_size=1,
                             multi_view_input_size=(3, 128, 128), is_training=False,
                             scale=(1.0, 1.0), color_jitter=0.4, mean=IMAGENET_DEFAULT_MEAN,
                             std=IMAGENET_DEFAULT_STD, num_workers=1, distributed=False,
                             collate_fn=None, pin_memory=False, persistent_workers=True,
                             hue_jitter=None):
    """Argument order matches the timm create_carla_loader this replaces."""
    attach_camera_transforms(dataset, input_size=input_size, multi_view_input_size=multi_view_input_size,
                             is_training=is_training, scale=scale, color_jitter=color_jitter,
                             mean=mean, std=std, hue_jitter=hue_jitter)

    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_training)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn or torch.utils.data.dataloader.default_collate,
        pin_memory=pin_memory,
        drop_last=is_training,
        persistent_workers=persistent_workers and num_workers > 0,
    )
