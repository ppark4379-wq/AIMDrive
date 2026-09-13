#!/usr/bin/env python3
"""Show what the encoder actually receives under each front-camera geometry.

Columns: the raw MORAI front image, then the 224x224 tensor the encoder sees
under the repository's current fixed resize, under an aspect-preserving crop,
and under a letterbox that keeps the full horizontal field of view.  A CARLA
frame is drawn on the last row so the reference geometry is visible too.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from inspect_vision_tokens import CropToAspect, PadToAspect  # noqa: E402


def fixed_resize(image, size=(341, 256)):
    return image.resize(size)


def centre_crop(image, side=224):
    width, height = image.size
    left, top = (width - side) // 2, (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def variants(image):
    return {
        "legacy\n(fixed 341x256)": centre_crop(fixed_resize(image)),
        "carla_ratio\n(crop to 4:3)": centre_crop(fixed_resize(CropToAspect()(image))),
        "carla_fov\n(letterbox to 4:3)": centre_crop(fixed_resize(PadToAspect()(image))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morai-root", type=Path, required=True)
    parser.add_argument("--route", default="full_loop_basic_drive_2026_09_09_01_44_15")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--carla-image", type=Path, help="Optional CARLA rgb_full frame")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    rows = []
    for frame in frames:
        path = args.morai_root / "data" / args.route / "rgb_front" / "{:04d}.jpg".format(frame)
        rows.append(("MORAI {}".format(frame), Image.open(path).convert("RGB")))
    if args.carla_image and args.carla_image.is_file():
        full = Image.open(args.carla_image).convert("RGB")
        rows.append(("CARLA reference", full.crop((0, 0, 800, 600))))

    columns = 4
    fig, axes = plt.subplots(len(rows), columns,
                             figsize=(3.1 * columns, 2.5 * len(rows)), squeeze=False)
    for row, (name, image) in enumerate(rows):
        axes[row][0].imshow(image)
        axes[row][0].set_ylabel("{}\n{}x{}".format(name, *image.size), fontsize=8)
        axes[row][0].set_xticks([]); axes[row][0].set_yticks([])
        if row == 0:
            axes[row][0].set_title("original front camera", fontsize=9)
        for column, (title, view) in enumerate(variants(image).items(), start=1):
            axes[row][column].imshow(view)
            axes[row][column].set_xticks([]); axes[row][column].set_yticks([])
            if row == 0:
                axes[row][column].set_title(title + "\n224x224 into encoder", fontsize=9)
    fig.suptitle("What the vision encoder actually receives from the front camera", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(str(args.output))


if __name__ == "__main__":
    main()
