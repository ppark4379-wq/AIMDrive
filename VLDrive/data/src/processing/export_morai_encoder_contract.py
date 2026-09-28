#!/usr/bin/env python3
"""Export a SHA-pinned contract without changing checkpoint bytes.

Older SensorRecord transfer checkpoints omitted their model-input contract.
Reconstruct it from the checkpoint's own publication hashes and sealed manifest
sensor contracts, not from its filename or the best-looking detection score.
This reads JSON evidence only; it does not run training or inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from data.src.loaders.morai_encoder_contract import (
    SCHEMA, checkpoint_encoder_contract, sensor_record_encoder_contract, validate_encoder_contract,
)
from data.src.loaders.morai_transfer_adapter import MoraiVisionAdapterConfig
from inference.navigation.morai_route import navigation_recipe


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified_json(path, expected):
    if digest(path) != expected:
        raise ValueError("Changed publication/manifest evidence: " + str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reconstruct_contract(blob):
    existing = blob.get("morai_encoder_contract") or {}
    if existing.get("schema") == SCHEMA:
        return validate_encoder_contract(existing), []
    if existing.get("schema") == "morai_encoder_recipe_v1":
        return checkpoint_encoder_contract(blob, "embedded-contract"), []
    provenance = blob.get("pretraining") or {}
    if provenance.get("schema") != "morai_visual_transfer_v1":
        raise ValueError("Unsupported legacy checkpoint: requires an explicit versioned contract; "
                         "do not guess output axes from checkpoint name")
    configs, evidence, route_hashes, navigation = [], [], set(), []
    for publication in provenance["publications"]:
        index_path = Path(publication["path"])
        index = verified_json(index_path, publication["sha256"])
        for split in ("train", "val"):
            for row in index[split]:
                path = index_path.parent / row["path"]
                doc = verified_json(path, row["sha256"])
                source = doc["contract"]
                # The sealed manifest carries the original sensor contract.
                extrinsic = source["lidar_to_base"]
                if (source.get("raw_lidar_axes") != "FLU" or source.get("lidar_sweeps") != 1
                        or source.get("preprocessing") != "lmdrive_compat"
                        or source.get("waypoint_output_axes") != "right_negative_forward"):
                    raise ValueError("Unsupported manifest input/output contract: " + str(path))
                config = MoraiVisionAdapterConfig(
                    use_previous_lidar_sweep=False, lidar_intensity_policy="reflectance_uint8_unit",
                    lidar_translation_xyz=tuple(extrinsic["translation_xyz"]),
                    lidar_rotation_xyzw=tuple(extrinsic["rotation_xyzw"]))
                configs.append(config)
                nav = doc["navigation_contract"]
                if nav["target_distance_m"] != 25.:
                    raise ValueError("Unexpected target lookahead")
                route_hashes.add(nav["global_path_sha256"])
                recipe = navigation_recipe(nav)
                if recipe not in navigation:
                    navigation.append(recipe)
        evidence.append(publication)
    contract = sensor_record_encoder_contract(configs, provenance["velocity_source"])
    contract["route_source_sha256s"] = sorted(route_hashes)
    contract["navigation_recipes"] = navigation
    return validate_encoder_contract(contract), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    blob = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    contract, evidence = reconstruct_contract(blob)
    result = dict(schema="morai_encoder_contract_sidecar_v1", checkpoint_sha256=checkpoint_hash,
                  contract=contract, source_publications=evidence,
                  note="Describes training inputs. UDP origin/time calibration is not proved by this export.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
