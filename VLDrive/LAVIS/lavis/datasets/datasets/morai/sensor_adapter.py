"""
Stage 6C-2 — MORAI recorded frame -> official LMDrive frozen vision-encoder input.

This adapter intentionally targets the public LMDrive R50 visual encoder
`memfuser_baseline_e1d3_return_feature`.

Official LMDrive visual encoder input contract:
    rgb_front   [B,3,224,224]
    rgb_left    [B,3,128,128]
    rgb_right   [B,3,128,128]
    rgb_rear    [B,3,128,128]
    rgb_center  [B,3,128,128]
    lidar       [B,40000,4]
    num_points  [B,1]
    velocity    [B,1]

The return-feature model emits 106 x 256:
    100 pooled traffic tokens
    1 traffic-light-state token
    5 waypoint-query tokens

MORAI-specific LiDAR decision:
    DO NOT apply the CARLA agent's -90 degree LiDAR rotation.

The 2026-09-06 TF/image projection audit supports MORAI ROS +x forward,
+y left, +z up. LMDrive's CARLA input, after undoing its sensor yaw -90,
uses +x forward, +y right, +z up. Convert y at the encoder boundary only;
raw recordings and trajectory/route labels keep their original ROS axes.
Legacy caches used passthrough and must be explicitly replayed as legacy.

This module is only the frozen-encoder smoke adapter. Route tokens, ego yaw,
CCDP, MEFA and DDIA are intentionally outside this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    use_previous_lidar_sweep: bool = False
    lidar_axis_policy: str = "morai_flu_to_lmdrive_fru"

    def validate(self):
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
    ):
        self.config = (
            config
            or MoraiVisionAdapterConfig()
        )
        self.config.validate()

        if (
            self.config.preprocessing
            == "lmdrive_compat"
        ):
            self.front_transform = (
                create_lmdrive_rgb_transform(
                    224
                )
            )
            self.multi_view_transform = (
                create_lmdrive_rgb_transform(
                    128
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

    def build_model_input(
        self,
        *,
        dataset_root,
        record,
        previous_lidar_xyzi=None,
        velocity_mps=0.0,
        device="cuda",
    ):
        dataset_root = Path(
            dataset_root
        )

        cameras = record[
            "cameras"
        ]

        required = (
            "front",
            "left",
            "right",
            "rear",
        )

        missing = [
            role
            for role in required
            if role not in cameras
        ]

        if missing:
            raise ValueError(
                "Stage 6C-2 frozen LMDrive smoke "
                "requires current four-camera setup; "
                "missing {}".format(
                    missing
                )
            )

        rgb = {}

        for role in required:
            rgb[role] = (
                self.read_rgb(
                    dataset_root
                    / cameras[
                        role
                    ][
                        "path"
                    ]
                )
            )

        camera = (
            self.camera_tensors(
                front_rgb=rgb[
                    "front"
                ],
                left_rgb=rgb[
                    "left"
                ],
                right_rgb=rgb[
                    "right"
                ],
                rear_rgb=rgb[
                    "rear"
                ],
            )
        )

        current_lidar = np.load(
            dataset_root
            / record[
                "lidar"
            ][
                "path"
            ],
            allow_pickle=False,
        )

        lidar, num_points = (
            self.lidar_to_raw_features(
                current_lidar,
                previous_xyzi=(
                    previous_lidar_xyzi
                ),
                rng_seed=int(
                    record[
                        "frame_index"
                    ]
                ),
            )
        )

        torch_input = {}

        for key, tensor in (
            camera.items()
        ):
            torch_input[key] = (
                tensor
                .unsqueeze(0)
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

        torch_input[
            "lidar"
        ] = (
            torch.from_numpy(
                lidar
            )
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        torch_input[
            "num_points"
        ] = (
            torch.tensor(
                [
                    [
                        num_points
                    ]
                ],
                dtype=torch.int64,
                device=device,
            )
        )

        torch_input[
            "velocity"
        ] = (
            torch.tensor(
                [
                    [
                        float(
                            velocity_mps
                        )
                    ]
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        return (
            torch_input,
            current_lidar,
            {
                "num_points": (
                    num_points
                ),
                "preprocessing": (
                    self.config.preprocessing
                ),
                "velocity_mps": (
                    float(
                        velocity_mps
                    )
                ),
                "velocity_source": "current_ego_speed",
                "lidar_rotation_deg": (
                    0.0
                ),
                "lidar_axis_policy": self.config.lidar_axis_policy,
                "two_sweep": (
                    bool(
                        self.config.use_previous_lidar_sweep
                    )
                ),
            },
        )


def load_frozen_lmdrive_vision_encoder(
    *,
    checkpoint_path,
    device="cuda",
):
    """
    Load only the official public LMDrive perception encoder.
    No LLM / Q-Former is constructed in Stage 6C-2.
    """
    from timm.models import (
        create_model,
    )

    model = create_model(
        "memfuser_baseline_e1d3_return_feature"
    )

    checkpoint = torch.load(
        str(
            Path(
                checkpoint_path
            )
            .expanduser()
            .resolve()
        ),
        map_location="cpu",
    )

    if (
        not isinstance(
            checkpoint,
            dict,
        )
        or "state_dict"
        not in checkpoint
    ):
        raise ValueError(
            "Expected LMDrive vision checkpoint with 'state_dict'"
        )

    load_result = (
        model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True,
        )
    )

    model = model.to(
        device
    )
    model.eval()

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    return (
        model,
        load_result,
    )
