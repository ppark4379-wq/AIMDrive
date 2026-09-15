"""MORAI camera/LiDAR preprocessing preserved from the validated v3_lav path.

Input tensors: front 224x224; left/right/rear/front-focus 128x128;
40,000 XYZI points with an explicit valid count. Applies each manifest's
LiDAR-to-base transform, FLU -> FRU conversion and uint8 reflectance scaling.
Single sweep, no augmentation by default. It does not freeze any model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms


IMAGENET_DEFAULT_MEAN = (
    0.485,
    0.456,
    0.406,
)
IMAGENET_DEFAULT_STD = (
    0.229,
    0.224,
    0.225,
)


class Resize2FixedSize(object):
    def __init__(
        self,
        size,
    ):
        self.size = size

    def __call__(
        self,
        pil_img,
    ):
        return pil_img.resize(
            self.size
        )


class RandomResize(object):
    """Match LMDrive's CARLA training-time random resize exactly."""

    def __init__(self, scale):
        self.scale = tuple(float(value) for value in scale)
        if len(self.scale) != 2 or not 0 < self.scale[0] <= self.scale[1]:
            raise ValueError("scale must contain two positive ordered values")

    def __call__(self, pil_img):
        width, height = pil_img.size
        ratio = ((self.scale[1] - self.scale[0]) * random.random()
                 + self.scale[0])
        return pil_img.resize((int(width * ratio), int(height * ratio)))


class LetterboxToSquare(object):
    """
    Preserve the complete MORAI FOV while producing a square tensor.

    This is an experimental MORAI alternative, not the first smoke default.
    """
    def __init__(
        self,
        size,
    ):
        self.size = int(size)

    def __call__(
        self,
        pil_img,
    ):
        return ImageOps.pad(
            pil_img,
            (
                self.size,
                self.size,
            ),
            method=Image.BILINEAR,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )


def create_lmdrive_rgb_transform(
    input_size,
    need_scale=True,
    is_training=False,
    scale=None,
    color_jitter=0.0,
):
    """
    Reproduce the public LMDrive agent's RGB transform.
    """
    if isinstance(
        input_size,
        (tuple, list),
    ):
        img_size = (
            input_size[-2:]
        )
        input_size_num = (
            input_size[-1]
        )
    else:
        img_size = input_size
        input_size_num = input_size

    transform_list = []

    if need_scale:
        if input_size_num == 112:
            transform_list.append(
                Resize2FixedSize(
                    (170, 128)
                )
            )
        elif input_size_num == 128:
            transform_list.append(
                Resize2FixedSize(
                    (195, 146)
                )
            )
        elif input_size_num == 224:
            transform_list.append(
                Resize2FixedSize(
                    (341, 256)
                )
            )
        elif input_size_num == 256:
            transform_list.append(
                Resize2FixedSize(
                    (288, 288)
                )
            )
        else:
            raise ValueError(
                "Unsupported LMDrive crop size"
            )

    if is_training:
        if scale is not None:
            transform_list.append(RandomResize(scale))
        if color_jitter > 0:
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter,
            ))

    transform_list.append(
        transforms.CenterCrop(
            img_size
        )
    )
    transform_list.append(
        transforms.ToTensor()
    )
    transform_list.append(
        transforms.Normalize(
            mean=torch.tensor(
                IMAGENET_DEFAULT_MEAN
            ),
            std=torch.tensor(
                IMAGENET_DEFAULT_STD
            ),
        )
    )

    return transforms.Compose(
        transform_list
    )


def create_morai_full_fov_transform(
    output_size,
):
    """
    Experimental alternative preserving all decoded camera FOV.

    Do not use this to claim pretrained compatibility. It exists so MORAI
    training can later compare exact-LMDrive preprocessing against full-FOV
    preprocessing.
    """
    return transforms.Compose(
        [
            LetterboxToSquare(
                output_size
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(
                    IMAGENET_DEFAULT_MEAN
                ),
                std=torch.tensor(
                    IMAGENET_DEFAULT_STD
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class MoraiVisionAdapterConfig:
    preprocessing: str = (
        "lmdrive_compat"
    )
    max_lidar_points: int = 40000
    remove_ego_box_m: float = 1.2
    use_previous_lidar_sweep: bool = True
    lidar_axis_policy: str = "morai_flu_to_lmdrive_fru"
    lidar_translation_xyz: Tuple[float, float, float] = (0., 0., 0.)
    lidar_rotation_xyzw: Tuple[float, float, float, float] = (0., 0., 0., 1.)
    lidar_intensity_policy: str = 'legacy_raw'

    def validate(self):
        if self.lidar_intensity_policy not in ('legacy_raw', 'reflectance_uint8_unit'):
            raise ValueError('Unsupported LiDAR intensity policy')
        if (len(self.lidar_translation_xyz) != 3 or len(self.lidar_rotation_xyzw) != 4
                or not np.isfinite(self.lidar_translation_xyz).all()
                or not np.isfinite(self.lidar_rotation_xyzw).all()
                or abs(np.linalg.norm(self.lidar_rotation_xyzw) - 1.) > 1e-4):
            raise ValueError("Invalid static LiDAR-to-base transform")
        if self.lidar_axis_policy not in ("morai_flu_to_lmdrive_fru", "legacy_passthrough"):
            raise ValueError("Unsupported LiDAR axis policy")
        if self.preprocessing not in (
            "lmdrive_compat",
            "morai_full_fov",
        ):
            raise ValueError(
                "preprocessing must be "
                "lmdrive_compat or morai_full_fov"
            )

        if self.max_lidar_points <= 0:
            raise ValueError(
                "max_lidar_points must be positive"
            )

        if self.remove_ego_box_m < 0.0:
            raise ValueError(
                "remove_ego_box_m must be >= 0"
            )


class MoraiLMDriveVisionAdapter(object):
    def __init__(
        self,
        config=None,
        rgb_augmentation_policy="none",
    ):
        self.config = (
            config
            or MoraiVisionAdapterConfig()
        )
        self.config.validate()
        if rgb_augmentation_policy not in ("none", "lmdrive_scale_color_v1"):
            raise ValueError("Unsupported RGB augmentation policy")
        self.rgb_augmentation_policy = rgb_augmentation_policy
        is_training = rgb_augmentation_policy == "lmdrive_scale_color_v1"
        scale = (0.9, 1.1) if is_training else None
        color_jitter = 0.1 if is_training else 0.0

        if (
            self.config.preprocessing
            == "lmdrive_compat"
        ):
            self.front_transform = (
                create_lmdrive_rgb_transform(
                    224,
                    is_training=is_training,
                    scale=scale,
                    color_jitter=color_jitter,
                )
            )
            self.multi_view_transform = (
                create_lmdrive_rgb_transform(
                    128,
                    is_training=is_training,
                    scale=scale,
                    color_jitter=color_jitter,
                )
            )
        else:
            self.front_transform = (
                create_morai_full_fov_transform(
                    224
                )
            )
            self.multi_view_transform = (
                create_morai_full_fov_transform(
                    128
                )
            )

        # LMDrive center/focus view:
        # front RGB -> resize 800x600 -> center crop 128.
        # Keep this semantic in both preprocessing modes for Stage 6C-2.
        self.center_transform = (
            create_lmdrive_rgb_transform(
                128,
                need_scale=False,
                is_training=is_training,
                scale=None,
                color_jitter=color_jitter,
            )
        )

    @staticmethod
    def read_rgb(
        path,
    ):
        path = str(path)

        bgr = cv2.imread(
            path,
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            raise FileNotFoundError(
                path
            )

        return cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

    def camera_tensors(
        self,
        *,
        front_rgb,
        left_rgb,
        right_rgb,
        rear_rgb,
    ):
        front_pil = Image.fromarray(
            front_rgb
        )
        left_pil = Image.fromarray(
            left_rgb
        )
        right_pil = Image.fromarray(
            right_rgb
        )
        rear_pil = Image.fromarray(
            rear_rgb
        )

        front_tensor = (
            self.front_transform(
                front_pil
            )
        )

        left_tensor = (
            self.multi_view_transform(
                left_pil
            )
        )
        right_tensor = (
            self.multi_view_transform(
                right_pil
            )
        )
        rear_tensor = (
            self.multi_view_transform(
                rear_pil
            )
        )

        center_rgb = cv2.resize(
            front_rgb,
            (800, 600),
            interpolation=cv2.INTER_LINEAR,
        )

        center_tensor = (
            self.center_transform(
                Image.fromarray(
                    center_rgb
                )
            )
        )

        return {
            "rgb_front": (
                front_tensor
            ),
            "rgb_left": (
                left_tensor
            ),
            "rgb_right": (
                right_tensor
            ),
            "rgb_rear": (
                rear_tensor
            ),
            "rgb_center": (
                center_tensor
            ),
        }

    def lidar_to_raw_features(
        self,
        current_xyzi,
        *,
        previous_xyzi=None,
        rng_seed=0,
    ):
        current = np.asarray(
            current_xyzi,
            dtype=np.float32,
        )

        if (
            current.ndim != 2
            or current.shape[1] != 4
        ):
            raise ValueError(
                "current_xyzi must be Nx4"
            )

        sweeps = [
            current
        ]

        if (
            self.config.use_previous_lidar_sweep
            and previous_xyzi is not None
        ):
            previous = np.asarray(
                previous_xyzi,
                dtype=np.float32,
            )

            if (
                previous.ndim != 2
                or previous.shape[1] != 4
            ):
                raise ValueError(
                    "previous_xyzi must be Nx4"
                )

            sweeps.append(
                previous
            )

        cloud = np.concatenate(
            sweeps,
            axis=0,
        )

        # Static extrinsics only. No per-frame privileged Ego pose enters inputs.
        x, y, z, w = self.config.lidar_rotation_xyzw
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
        cloud[:, :3] = cloud[:, :3] @ rotation.T + self.config.lidar_translation_xyz

        # concatenate owns its memory; never mutate either original sweep.
        if self.config.lidar_axis_policy == "morai_flu_to_lmdrive_fru":
            cloud[:, 1] *= -1.0

        # Same central ego-box removal logic as public LMDrive.
        radius = float(
            self.config.remove_ego_box_m
        )

        inside_ego = (
            (cloud[:, 0] > -radius)
            & (cloud[:, 0] < radius)
            & (cloud[:, 1] > -radius)
            & (cloud[:, 1] < radius)
        )

        cloud = cloud[
            ~inside_ego
        ]

        finite = np.isfinite(
            cloud
        ).all(axis=1)

        cloud = cloud[
            finite
        ]

        # Keep the original public model's practical [-100,100] protection,
        # but reject whole points instead of mutating individual coordinates.
        bounded = (
            np.abs(
                cloud[:, :3]
            )
            <= 100.0
        ).all(axis=1)

        cloud = cloud[
            bounded
        ]

        if self.config.lidar_intensity_policy == 'reflectance_uint8_unit':
            # Opt-in for verified MORAI Intensity mode, never semantic/instance
            # IDs. Do not infer a policy from a particular frame's maximum.
            if np.any((cloud[:, 3] < 0) | (cloud[:, 3] > 255)):
                raise ValueError('Expected MORAI intensity in [0,255]')
            cloud[:, 3] /= 255.0

        rng = np.random.RandomState(
            int(rng_seed)
        )

        order = rng.permutation(
            len(cloud)
        )

        cloud = cloud[
            order
        ]

        max_points = int(
            self.config.max_lidar_points
        )

        num_points = min(
            max_points,
            len(cloud),
        )

        output = np.zeros(
            (
                max_points,
                4,
            ),
            dtype=np.float32,
        )

        if num_points:
            output[
                :num_points,
                :,
            ] = cloud[
                :num_points,
                :4,
            ]

        # IMPORTANT:
        # No CARLA rotate_lidar(..., -90) here.
        return (
            output,
            int(num_points),
        )

