"""Versioned MORAI encoder inputs/outputs; no checkpoint-name heuristics.

The SensorRecord model uses (right, -forward) for navigation and waypoints.
LiDAR tensors are (forward, right, up, unit reflectance) in the ego base frame.
Transport clock and source-frame calibration are separate from this contract.
"""
from dataclasses import asdict, fields, replace
import copy
import json
import math

import numpy as np
import torch

from .morai_transfer_adapter import MoraiLMDriveVisionAdapter, MoraiVisionAdapterConfig


SCHEMA = "morai_sensor_record_encoder_v2"
WAYPOINT_AXES = "right_negative_forward"
LEGACY_WAYPOINT_AXES = "left_negative_forward"
DETECTION_AXES = "right_forward_offsets_right_negative_forward_heading"


def world_target_to_local(target_xy, ego_xy, yaw, axes=WAYPOINT_AXES):
    delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(ego_xy, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.array([s * delta[0] - c * delta[1],
                    -c * delta[0] - s * delta[1]], dtype=np.float32)
    if axes == LEGACY_WAYPOINT_AXES:
        out[0] *= -1
    elif axes != WAYPOINT_AXES:
        raise ValueError("Unknown waypoint axes: " + str(axes))
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite local navigation target")
    return out


def sensor_record_encoder_contract(adapter_configs, velocity_source):
    """Record actual loader recipes, including each calibrated sensor origin."""
    configs = []
    for config in adapter_configs:
        config.validate()
        doc = asdict(config)
        if doc not in configs:
            configs.append(doc)
    if not configs:
        raise ValueError("Cannot build an encoder contract without a dataset adapter")
    if velocity_source not in ("gps", "zero"):
        raise ValueError("Unsupported SensorRecord velocity source")
    return dict(schema=SCHEMA, adapter=configs[0], native_adapter_configs=configs,
                camera_recipe="lmdrive_compat_front_focus_resize800x600_center128",
                waypoint_output_axes=WAYPOINT_AXES, target_point_axes=WAYPOINT_AXES,
                detection_output_axes=DETECTION_AXES,
                velocity_input="causal_gps_source_time" if velocity_source == "gps" else "constant_zero",
                target_point_input="official_route_25m", target_lookahead_m=25.0,
                waypoint_offsets_source_seconds=[0.5, 1., 1.5, 2., 2.5],
                feature_shape=[106, 256], token_order="bev100_light1_waypoint5",
                lidar_tensor_frame="base_fru", raw_lidar_frame="sensor_flu")


def validate_encoder_contract(contract):
    if contract.get("schema") != SCHEMA:
        raise ValueError("A versioned MORAI encoder contract is required; export a SHA-pinned sidecar")
    if contract.get("waypoint_output_axes") not in (WAYPOINT_AXES, LEGACY_WAYPOINT_AXES):
        raise ValueError("Missing/unsupported waypoint output axes")
    if contract.get("target_point_axes") != contract["waypoint_output_axes"]:
        raise ValueError("Target/waypoint axes disagree")
    if contract.get("detection_output_axes") != DETECTION_AXES:
        raise ValueError("Unsupported detection output axes")
    if contract.get("velocity_input") not in ("causal_gps_source_time", "constant_zero"):
        raise ValueError("Unsupported velocity input contract")
    if contract.get("target_point_input") not in ("official_route_25m", "constant_zero"):
        raise ValueError("Unsupported navigation input contract")
    cfg = adapter_config(contract["adapter"])
    if cfg.use_previous_lidar_sweep or cfg.preprocessing != "lmdrive_compat":
        raise ValueError("Live adapter currently supports one-sweep lmdrive_compat only")
    return copy.deepcopy(contract)


def checkpoint_encoder_contract(blob, checkpoint_sha256, sidecar=None):
    """Known legacy recipes are migrated explicitly; unknown weights fail closed."""
    embedded = blob.get("morai_encoder_contract") or {}
    if sidecar is not None:
        if sidecar.get("schema") != "morai_encoder_contract_sidecar_v1":
            raise ValueError("Unsupported contract sidecar schema")
        if sidecar.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("Contract sidecar checkpoint SHA256 mismatch")
        contract = validate_encoder_contract(sidecar["contract"])
        if (embedded.get("schema") == SCHEMA
                and json.dumps(contract, sort_keys=True) != json.dumps(embedded, sort_keys=True)):
            raise ValueError("Sidecar contradicts the checkpoint's embedded contract")
        return contract
    if embedded.get("schema") == SCHEMA:
        return validate_encoder_contract(embedded)
    sensor = blob.get("morai_sensor_contract") or {}
    if (embedded.get("schema") == "morai_encoder_recipe_v1"
            and embedded.get("velocity_input") == "constant_zero"
            and embedded.get("target_point_input") == "constant_zero"
            and sensor.get("waypoint_output_axes") == WAYPOINT_AXES):
        contract = sensor_record_encoder_contract([adapter_config(embedded["adapter"])], "zero")
        contract["target_point_input"] = "constant_zero"
        contract["migrated_from"] = "morai_encoder_recipe_v1_with_sensor_contract"
        return validate_encoder_contract(contract)
    raise ValueError("Checkpoint lacks a verified input/output contract. Run "
                     "tools/export_morai_encoder_contract.py and pass --contract <sidecar.json>")


def adapter_config(value):
    known = {f.name for f in fields(MoraiVisionAdapterConfig)}
    if set(value) - known:
        raise ValueError("Unknown adapter settings: " + str(sorted(set(value) - known)))
    config = MoraiVisionAdapterConfig(**value)
    config.validate()
    return config


class LiveInputAdapter:
    """The actual training camera/LiDAR implementation, with explicit raw frame.

    sensor_flu requires a measured lidar-to-base transform supplied by the
    caller. base_flu means that transform was already applied upstream; it is
    never applied twice. Neither option is inferred from detection scores.
    """
    def __init__(self, contract, *, lidar_frame, lidar_to_base=None):
        self.contract = validate_encoder_contract(contract)
        config = adapter_config(contract["adapter"])
        if lidar_frame == "sensor_flu":
            if not lidar_to_base:
                raise ValueError("sensor_flu requires an explicit lidar_to_base calibration")
            config = replace(config,
                             lidar_translation_xyz=tuple(lidar_to_base["translation_xyz"]),
                             lidar_rotation_xyzw=tuple(lidar_to_base["rotation_xyzw"]))
        elif lidar_frame == "base_flu":
            if lidar_to_base:
                raise ValueError("base_flu must not apply a second lidar_to_base transform")
            config = replace(config, lidar_translation_xyz=(0., 0., 0.),
                             lidar_rotation_xyzw=(0., 0., 0., 1.))
        else:
            raise ValueError("Set lidar.frame to sensor_flu or base_flu after checking the UDP origin")
        self.adapter = MoraiLMDriveVisionAdapter(config)

    def assemble(self, images, points, *, velocity, target_point, rng_seed=0):
        tensors = self.adapter.camera_tensors(**{
            name + "_rgb": images["rgb_" + name] for name in ("front", "left", "right", "rear")})
        lidar, count = self.adapter.lidar_to_raw_features(points, rng_seed=rng_seed)
        if not count:
            raise ValueError("Empty LiDAR after preprocessing")
        target = np.asarray(target_point, dtype=np.float32).reshape(2)
        if not np.isfinite(target).all() or not math.isfinite(velocity):
            raise ValueError("Nonfinite scalar inputs")
        if self.contract["velocity_input"] == "constant_zero":
            velocity = 0.
        if self.contract["target_point_input"] == "constant_zero":
            target = np.zeros(2, dtype=np.float32)
        tensors.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(count),
                       velocity=torch.tensor([velocity], dtype=torch.float32),
                       target_point=torch.from_numpy(target))
        return tensors
