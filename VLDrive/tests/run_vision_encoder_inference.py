#!/usr/bin/env python3
"""Run the frozen stage-1 vision encoder and draw what its prediction heads say.

    python tests/run_vision_encoder_inference.py --config configs/vldrive_realtime.json#presets/vision_encoder_inference

Training uses the encoder's ``return_feature`` path, which stops at the 106
visual tokens and never runs the prediction heads.  This script takes the other
path: it loads a checkpoint, freezes it, runs the full forward, and renders each
head separately so the encoder can be judged by eye while driving a route.

Per selected frame it writes, into ``output.dir``:

    frame_<id>_bev_detection.{pdf,png}   oriented boxes over the objectness map
    frame_<id>_bev_features.{pdf,png}    BEV token activation VLDrive pools
    frame_<id>_traffic_light.{pdf,png}   red / not-red probability + camera
    frame_<id>_waypoints.{pdf,png}       the 0.5 ... 2.5 s predicted path

plus ``inference_manifest.json`` recording the checkpoint, its SHA-256, the
config and every frame drawn.

Every parameter comes from the config file; the command line carries only its
path.  The checkpoint is set in the config, so pointing this at a different
encoder means editing or copying the config, never passing a flag.

This file deliberately does not start with ``test_``: it is a runnable tool, not
a pytest case, so a plain ``pytest tests/`` will not collect it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common.runtime import configure_runtime

configure_runtime()

import numpy as np
import torch

from data.src.loaders.loader import attach_camera_transforms
from data.src.loaders.morai_native_capture import MoraiNativeCaptureDataset
from data.src.loaders.morai_perception_dataset import MoraiPerceptionDataset
from models.ve.vision_encoder import create_vision_encoder
from utils.visualization import vision_encoder_visualization as viz
from utils.config.vision_encoder_inference_config import ConfigError, load_config, require_config
from training.vision_encoder_training import MODEL_NAME, sha256_file

# The head-bearing variant; return_feature would stop at the 106 tokens.
HEAD_MODEL_NAME = MODEL_NAME


def build_dataset(config):
    """The evaluation view of the data: no augmentation, deterministic LiDAR."""
    require_config(config)
    if config.dataset_format == "morai_native_capture":
        dataset = MoraiNativeCaptureDataset(
            root=str(config.dataset_root),
            route_csv=str(config.route_csv) if config.route_csv else None,
            lidar_axis_policy=config.lidar_axis_policy,
        )
    else:
        dataset = MoraiPerceptionDataset(
            root=str(config.dataset_root),
            split_file=str(config.split_file),
            split=config.split,
            with_lidar=True,
            multi_view=True,
            lidar_axis_policy=config.lidar_axis_policy,
            sample_stride=1,
            traffic_label_sidecar=None,
        )
    attach_camera_transforms(dataset, input_size=(3, 224, 224),
                             multi_view_input_size=(3, 128, 128),
                             is_training=False, scale=(1.0, 1.0), color_jitter=0.0)
    return dataset


def select_indices(config, dataset):
    """Dataset indices to render, either the configured frame ids or a stride."""
    require_config(config)
    if config.frame_ids:
        by_frame = {int(frame): position
                    for position, (_, frame) in enumerate(dataset.route_frames)}
        missing = [frame for frame in config.frame_ids if frame not in by_frame]
        if missing:
            raise SystemExit("frames.ids not present in the data: {}".format(missing))
        return [by_frame[frame] for frame in config.frame_ids]
    return list(range(0, len(dataset), config.stride))[:config.limit]


def check_contract(config, contract):
    """Compare the recipe a checkpoint was trained with against this dataset.

    Some encoders ship a ``morai_encoder_contract`` describing the exact inputs
    they saw.  Feeding them anything else is out of distribution, so the parts
    this script can honour are applied and the rest is reported rather than
    quietly ignored.
    """
    notes = []
    # The sensor-side fields live under "adapter"; the model-side ones do not.
    adapter = contract.get("adapter") or {}
    if adapter.get("lidar_axis_policy") != config.lidar_axis_policy:
        notes.append("lidar_axis_policy: checkpoint {!r} vs config {!r}".format(
            adapter.get("lidar_axis_policy"), config.lidar_axis_policy))
    if adapter.get("max_lidar_points") not in (None, 40000):
        notes.append("max_lidar_points {}".format(adapter["max_lidar_points"]))
    if adapter.get("use_previous_lidar_sweep"):
        notes.append("checkpoint expects a two-sweep LiDAR input; this dataset gives one")
    translation = adapter.get("lidar_translation_xyz")
    if translation and any(abs(float(value)) > 1e-6 for value in translation):
        notes.append("lidar_translation_xyz {}; this dataset does not translate the "
                     "point cloud, so confirm whether the recording already is "
                     "ego-centred".format(list(translation)))
    policy = adapter.get("lidar_intensity_policy")
    if policy not in (None, "raw", "reflectance_uint8_unit"):
        notes.append("lidar_intensity_policy {!r} is not handled; this dataset passes "
                     "the recorded intensity through unchanged".format(policy))
    elif policy == "reflectance_uint8_unit":
        notes.append("lidar_intensity_policy 'reflectance_uint8_unit' is applied as /255, "
                     "but the dataset already zeroed intensity above 100 "
                     "(6.5% of points on the sampled frame)")
    shape = contract.get("feature_shape")
    if shape is not None and list(shape) != [106, 256]:
        notes.append("feature_shape {}".format(shape))
    if contract.get("token_order") not in (None, "bev100_light1_waypoint5"):
        notes.append("token_order {!r}".format(contract["token_order"]))
    zeroed = [name for name in ("velocity_input", "target_point_input")
              if contract.get(name) == "constant_zero"]
    if zeroed:
        notes.append("{} recorded as constant_zero, but the real values are passed "
                     "through: zeroing them collapses the 2.5 s path to 3-4 m at "
                     "9-22 m/s, while the real values give 21-35 m".format(
                         " and ".join(zeroed)))
    return notes


def apply_contract(batch, contract):
    """Put the batch into the form the checkpoint measurably responds to.

    LiDAR intensity is rescaled: our recordings carry raw reflectance 0..149,
    and feeding that unchanged holds the trained detection head down to a peak
    objectness of 0.026, against 0.229 once the channel is scaled towards unit
    range.  Dividing by 255, by 100, or zeroing the channel all land within
    0.002 of each other, so the encoder barely reads the value and mostly needs
    it to be small.  ``reflectance_uint8_unit`` is read here as "uint8
    reflectance mapped to [0, 1]".

    The contract's ``velocity_input``/``target_point_input`` = ``constant_zero``
    is deliberately NOT applied.  Measured over five frames, zeroing both makes
    the head predict a 3.2-4.4 m path where the ego is doing 9-22 m/s, and
    zeroing target_point alone makes it worse (1.8-2.6 m); passing both through
    gives 20.6-34.8 m against a 22-55 m ballistic estimate.  The real values are
    the ones that produce physically sensible trajectories, so whatever those
    contract fields describe, it is not what to feed at inference.  The mismatch
    is reported by check_contract rather than silently applied.
    """
    forced = []
    policy = (contract.get("adapter") or {}).get("lidar_intensity_policy")
    if policy == "reflectance_uint8_unit":
        lidar = batch["lidar"].clone()
        lidar[:, :, 3] = lidar[:, :, 3] / 255.0
        batch["lidar"] = lidar
        forced.append("lidar_intensity/=255")
    return forced


def load_encoder(config):
    """Load the checkpoint into the head-bearing encoder and freeze it."""
    require_config(config)
    if not config.checkpoint.is_file():
        raise SystemExit("Checkpoint not found: {}".format(config.checkpoint))
    # pretrained_backbone=False: the strict load below replaces every weight.
    model = create_vision_encoder({"name": HEAD_MODEL_NAME}, pretrained_backbone=False)
    checkpoint = torch.load(str(config.checkpoint), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    result = model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    contract = checkpoint.get("morai_encoder_contract") or {}
    provenance = {
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "strict_load": str(result),
        "source_epoch": checkpoint.get("epoch"),
        "source_steps": checkpoint.get("steps"),
        "model_name": checkpoint.get("model_name"),
        "split_sha256": checkpoint.get("split_sha256"),
        "encoder_contract": contract or None,
    }
    return model, provenance, contract


def detection_head_is_trained(state_or_model):
    """Warn when the detection head is still at its random initialisation.

    MORAI recordings carry no object annotations, so stage 1 leaves
    ``traffic_pred_head`` and ``stop_sign_head`` unsupervised: their weights
    never receive a gradient.  A BEV detection figure drawn from such a
    checkpoint shows noise, not objects, and saying so beats letting the figure
    be read as a result.
    """
    weights = dict(state_or_model.named_parameters())["traffic_pred_head.0.weight"]
    # An untrained Linear keeps the exactly symmetric Kaiming-uniform bound.
    bound = float(weights.abs().max())
    limit = 1.0 / (weights.shape[1] ** 0.5)
    return bool(abs(bound - limit) > 1e-3 * limit)


@torch.no_grad()
def infer(model, dataset, index, device, contract=None):
    """One frame through the encoder, on the configured device."""
    inputs, targets = dataset[index]
    batch = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            tensor = value.unsqueeze(0)
        elif isinstance(value, np.ndarray):
            # lidar arrives as a numpy [40000, 4] array, not a tensor.
            tensor = torch.from_numpy(value).unsqueeze(0)
        else:
            tensor = torch.tensor([value])
        batch[key] = tensor.to(device)
    forced = apply_contract(batch, contract or {})
    detection, waypoints, traffic_light, stop_sign, bev_features = model(batch)
    return {
        "detection": detection[0].float().cpu().numpy(),
        "waypoints": waypoints[0].float().cpu().numpy(),
        "traffic_light": traffic_light[0].float().cpu().numpy(),
        "stop_sign": stop_sign[0].float().cpu().numpy(),
        "bev_features": bev_features[0].float().cpu().numpy(),
        "camera": viz.denormalize_image(inputs["rgb_front"].numpy()),
        "speed": float(inputs["velocity"][0]),
        "target_point": inputs["target_point"].numpy(),
        "frame_id": int(targets["frame_id"]),
        "route": dataset.route_frames[index][0],
        "forced_inputs": forced,
    }


def render_frame(config, result, warning):
    """Draw and save the requested figures for one frame; returns the paths."""
    stem = Path(config.output) / "frame_{:04d}".format(result["frame_id"])
    subtitle = "{}  frame {:04d}".format(result["route"], result["frame_id"])
    written = []

    if config.bev_source in ("detection", "both"):
        title = "BEV map"
        note = subtitle if not warning else subtitle + "   [" + warning + "]"
        figure = viz.render_bev_detection(result["detection"],
                                          threshold=config.detection_threshold,
                                          title=title, subtitle=note)
        written += viz.save_figure(figure, stem.with_name(stem.name + "_bev_detection"),
                                   config.formats, config.dpi)

    if config.bev_source in ("features", "both"):
        figure = viz.render_bev_features(result["bev_features"], subtitle=subtitle)
        written += viz.save_figure(figure, stem.with_name(stem.name + "_bev_features"),
                                   config.formats, config.dpi)

    figure = viz.render_traffic_light(
        result["traffic_light"],
        camera_image=result["camera"] if config.show_camera else None,
        subtitle=subtitle)
    written += viz.save_figure(figure, stem.with_name(stem.name + "_traffic_light"),
                               config.formats, config.dpi)

    figure = viz.render_waypoints(result["waypoints"], speed=result["speed"],
                                  target_point=result["target_point"], subtitle=subtitle)
    written += viz.save_figure(figure, stem.with_name(stem.name + "_waypoints"),
                               config.formats, config.dpi)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="inference config, e.g. configs/vldrive_realtime.json#presets/vision_encoder_inference")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as error:
        parser.error(str(error))

    if config.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("runtime.device=cuda but no CUDA device is available")
    device = torch.device(config.device)

    dataset = build_dataset(config)
    indices = select_indices(config, dataset)
    if not indices:
        raise SystemExit("No frames selected; check frames.ids / frames.stride / frames.limit")

    model, provenance, contract = load_encoder(config)
    model.to(device)
    if contract:
        print("encoder contract: {} step(s), token order {}".format(
            provenance.get("source_steps"), contract.get("token_order")))
        for note in check_contract(config, contract):
            print("  MISMATCH: {}".format(note), flush=True)

    warning = ""
    if not detection_head_is_trained(model):
        warning = "detection head untrained"
        print("WARNING: traffic_pred_head is still at its random initialisation in this "
              "checkpoint, so the BEV detection figure shows noise rather than objects. "
              "MORAI recordings carry no object annotations, so stage 1 never supervises it. "
              "The BEV feature figure and the waypoint/traffic-light heads are unaffected.",
              flush=True)

    Path(config.output).mkdir(parents=True, exist_ok=True)
    print("checkpoint : {}".format(config.checkpoint))
    print("data       : {} format, {} frames available".format(
        config.dataset_format, len(dataset)))
    print("rendering  : {} frame(s) -> {}".format(len(indices), config.output), flush=True)

    started = time.time()
    frames = []
    for position, index in enumerate(indices, 1):
        result = infer(model, dataset, index, device, contract)
        written = render_frame(config, result, warning)
        frames.append({
            "dataset_index": int(index),
            "frame_id": result["frame_id"],
            "route": result["route"],
            "speed_mps": result["speed"],
            "forced_inputs": result["forced_inputs"],
            "traffic_light_logits": result["traffic_light"].tolist(),
            "waypoints": result["waypoints"].tolist(),
            "figures": [str(path.relative_to(ROOT)) for path in written],
        })
        print("  [{}/{}] frame {:04d}  ->  {} file(s)".format(
            position, len(indices), result["frame_id"], len(written)), flush=True)

    manifest = {
        "schema": "morai-vision-encoder-inference-report-v1",
        "config_path": str(config.config_path),
        "config_sha256": config.config_sha256,
        "config": config.flat_dict(),
        "encoder": provenance,
        "detection_head_trained": not warning,
        "encoder_contract_mismatches": check_contract(config, contract) if contract else [],
        "forced_inputs": frames[0]["forced_inputs"] if frames else [],
        "device": str(device),
        "elapsed_sec": time.time() - started,
        "frames": frames,
    }
    path = Path(config.output) / "inference_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote {}".format(path))


if __name__ == "__main__":
    main()
