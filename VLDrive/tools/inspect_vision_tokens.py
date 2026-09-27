#!/usr/bin/env python3
"""Run a vision encoder alone on chosen frames and dump its visual tokens.

No connector, Q-Former or language model is involved.  The same weights produce
both the frozen [106,256] connector tokens and the original pre-training head
predictions, so a CARLA-pretrained checkpoint can be compared against a
MORAI-trained one, and MORAI frames can be fed either with the repository's
current geometry or with geometry matched to what the CARLA encoder was
trained on.

Camera geometry, for a 100 degree horizontal field of view:

  CARLA 800x600   front resize 341x256 -> crop 224  = 76.0 deg H, 76.0 deg V
  MORAI 1280x720  front resize 341x256 -> crop 224  = 76.0 deg H, 60.4 deg V
                  (16:9 squeezed into 4:3, so objects are 1.335x narrower)

  carla_ratio  crop the MORAI front to 4:3 first    = 60.4 deg H, 60.4 deg V
               (no distortion; narrower view than CARLA)
  carla_fov    letterbox the MORAI front to 4:3     = 76.0 deg H, 76.0 deg V
               (no distortion and CARLA's angular scale, with black bars)

The 128x128 focus view is an un-resized centre crop, so it covers 21.6 deg on
CARLA but only 13.6 deg on MORAI; both corrected modes restore 21.6 deg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import configure_runtime

configure_runtime()

import numpy as np
import torch
from PIL import Image

from datasets.morai_perception_dataset import MoraiPerceptionDataset
from datasets.loader import create_perception_loader
from models.vision_encoder import create_vision_encoder

MODEL_NAME = "memfuser_baseline_e1d3"
TOKEN_SHAPE = (106, 256)
BEV_TOKENS = 100
TRAFFIC_TOKEN = 100
WAYPOINT_TOKENS = slice(101, 106)
CARLA_FOCUS_FRACTION = 128.0 / 800.0


def sha256(path):
    if '#' in str(path):
        from utils.config_files import document_bytes
        return hashlib.sha256(document_bytes(path)).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CropToAspect:
    """Centre crop to a target aspect so a later fixed resize cannot distort."""

    def __init__(self, ratio=4.0 / 3.0):
        self.ratio = float(ratio)

    def __call__(self, image):
        width, height = image.size
        if width / height <= self.ratio:
            return image
        target = int(round(height * self.ratio))
        left = (width - target) // 2
        return image.crop((left, 0, left + target, height))


class PadToAspect:
    """Letterbox to a target aspect, keeping the full horizontal field of view."""

    def __init__(self, ratio=4.0 / 3.0):
        self.ratio = float(ratio)

    def __call__(self, image):
        width, height = image.size
        target = int(round(width / self.ratio))
        if target <= height:
            return image
        canvas = Image.new(image.mode, (width, target))
        canvas.paste(image, (0, (target - height) // 2))
        return canvas


class FocusCrop:
    """Match CARLA's 21.6 degree focus view, then resize to the expected size."""

    def __init__(self, fraction=CARLA_FOCUS_FRACTION, size=128):
        self.fraction = float(fraction)
        self.size = int(size)

    def __call__(self, image):
        width, height = image.size
        side = int(round(width * self.fraction))
        side = max(1, min(side, height))
        left = (width - side) // 2
        top = (height - side) // 2
        patch = image.crop((left, top, left + side, top + side))
        return patch.resize((self.size, self.size), Image.BILINEAR)


class PreTransform:
    def __init__(self, pre, base):
        self.pre = pre
        self.base = base

    def __call__(self, image):
        return self.base(self.pre(image))


def apply_geometry(dataset, geometry):
    """Insert a geometry correction ahead of the repository's RGB transforms."""
    if geometry == "legacy":
        return dict(mode="legacy", front="fixed_resize_341x256", focus="raw_centre_crop_128")
    front = CropToAspect() if geometry == "carla_ratio" else PadToAspect()
    dataset.rgb_transform = PreTransform(front, dataset.rgb_transform)
    dataset.rgb_center_transform = PreTransform(FocusCrop(), dataset.rgb_center_transform)
    return dict(
        mode=geometry,
        front="centre_crop_to_4_3" if geometry == "carla_ratio" else "letterbox_to_4_3",
        focus="centre_crop_{:.1f}pct_then_resize_128".format(100 * CARLA_FOCUS_FRACTION),
    )


def load_encoder(checkpoint_path, device):
    """Accept both the official LMDrive release and our MORAI checkpoints."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    # pretrained_backbone=False: the strict load below replaces every weight.
    model = create_vision_encoder({"name": MODEL_NAME}, pretrained_backbone=False)
    result = model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, dict(
        checkpoint=str(checkpoint_path),
        sha256=sha256(checkpoint_path),
        declared_arch=checkpoint.get("arch") or checkpoint.get("model_name"),
        source_epoch=checkpoint.get("epoch"),
        strict_load=str(result),
        tensors=len(state),
    )


def build_morai(args, frames):
    dataset = MoraiPerceptionDataset(
        root=str(args.dataset_root), split_file=str(args.split_file),
        split="train", with_lidar=True, multi_view=True,
        lidar_axis_policy="morai_flu_to_lmdrive_fru", sample_stride=1,
    )
    dataset.split = "validation"
    index = {int(entry[1]): entry for entry in dataset.route_frames}
    missing = [f for f in frames if f not in index]
    if missing:
        raise ValueError("Frames are outside the split: {}".format(missing))
    dataset.route_frames = [index[f] for f in frames]
    return dataset, frames


def bev_grid(tokens):
    return np.linalg.norm(tokens[:BEV_TOKENS], axis=1).reshape(10, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--frames", required=True, help="Comma separated frame ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-geometry", choices=("legacy", "carla_ratio", "carla_fov"),
                        default="legacy", help="how to reshape the front view before the encoder")
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT.parent / "MoraiDrive/dataset_local/competition_train")
    parser.add_argument("--split-file", type=Path,
                        default=ROOT / "configs/vldrive_train.json#split_manifests/morai_full_loop_95_05")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("Use a new output directory: {}".format(args.output))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = list(dict.fromkeys(int(x) for x in args.frames.split(",") if x.strip()))

    model, provenance = load_encoder(args.checkpoint, device)
    dataset, ordered = build_morai(args, frames)
    loader = create_perception_loader(
        dataset, input_size=(3, 224, 224), multi_view_input_size=(3, 128, 128),
        batch_size=1, is_training=False, scale=[1.0, 1.0], color_jitter=0.0,
        num_workers=args.workers, pin_memory=False, persistent_workers=False,
    )
    geometry = apply_geometry(dataset, args.rgb_geometry)

    records = []
    tokens_all = {}
    stride = 1
    cursor = 0
    for inputs, targets in loader:
        count = int(inputs["rgb_front"].shape[0])
        identities = dataset.route_frames[cursor:cursor + count * stride:stride]
        cursor += count * stride
        model_inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            model.return_feature = True
            tokens = model(model_inputs).float().cpu().numpy()
            model.return_feature = False
            _, waypoints, traffic_light_state, _, _ = model(model_inputs)
        traffic_light_state = torch.softmax(traffic_light_state.float(), dim=-1).cpu().numpy()
        waypoints = waypoints.float().cpu().numpy()

        for offset, entry in enumerate(identities):
            frame_id = int(entry[1])
            token = tokens[offset]
            if tuple(token.shape) != TOKEN_SHAPE or not np.isfinite(token).all():
                raise RuntimeError("Invalid token output for frame {}".format(frame_id))
            tokens_all[frame_id] = token
            grid = bev_grid(token)
            records.append(dict(
                frame_id=frame_id,
                bev_norm_mean=float(grid.mean()), bev_norm_min=float(grid.min()),
                bev_norm_max=float(grid.max()), bev_norm_std=float(grid.std()),
                traffic_token_norm=float(np.linalg.norm(token[TRAFFIC_TOKEN])),
                waypoint_token_norm=[float(np.linalg.norm(v)) for v in token[WAYPOINT_TOKENS]],
                traffic_light_probabilities=[float(v) for v in traffic_light_state[offset]],
                predicted_waypoints=waypoints[offset].tolist(),
                velocity_mps=float(inputs["velocity"][offset]),
            ))

    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "tokens.npz",
                        **{str(k): v for k, v in tokens_all.items()})
    summary = dict(
        schema="morai-vision-token-inspection-v2",
        label=args.label, source=args.source, encoder=provenance,
        rgb_geometry=geometry,
        dataset_root=str(args.dataset_root if args.source == "morai" else args.carla_root),
        preprocessing=dict(lidar_axis_policy="morai_flu_to_lmdrive_fru" if args.source == "morai"
                           else "carla_native", lidar_sweeps=1),
        frames=ordered, records=records,
        note="Vision encoder only; no connector, Q-Former or language model was run.",
    )
    (args.output / "inspection.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), frames=len(records), label=args.label,
                          source=args.source, geometry=geometry["mode"],
                          strict_load=provenance["strict_load"]), indent=2))


if __name__ == "__main__":
    main()
