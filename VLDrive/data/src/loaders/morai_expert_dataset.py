"""VLDrive windows backed by cached tokens from the MORAI expert dataset.

The source dataset remains read-only.  Cache records contain only frozen visual
features, actual future ego trajectories, and provenance.  Route commands are
provided separately as deterministic navigation instructions and never used as
trajectory targets.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_SCHEMA = "morai-vldrive-vision-cache-v1"
INSTRUCTION_SCHEMA = "morai-global-path-instructions-v1"
LOOKAHEAD_INSTRUCTION_SCHEMA = "morai-global-path-instructions-v2"
SPLIT_SCHEMA = "morai-vision-split-v1"
FEATURE_SHAPE = (106, 256)
TRAJECTORY_SHAPE = (5, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def safe_artifact(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Cache artifact escapes its root: {}".format(relative))
    return path


def load_split(path: Path):
    from utils.config.config_files import document_bytes
    content = document_bytes(path)
    spec = json.loads(content)
    if spec.get("schema_version") != SPLIT_SCHEMA:
        raise ValueError("Expected {} split manifest".format(SPLIT_SCHEMA))
    if spec.get('future_offsets') != [5, 10, 15, 20, 25]:
        raise ValueError('MORAI cache requires future offsets [5,10,15,20,25]')
    if int(spec.get("purge_frames", 0)) < 25:
        raise ValueError("Split purge is shorter than the future horizon")
    routes = spec.get('routes', [])
    if not routes or len({route['path'] for route in routes}) != len(routes):
        raise ValueError('Split requires unique, nonempty route entries')
    for route in routes:
        train, purge, val = (route[key] for key in ('train', 'purge', 'validation'))
        for bounds in (train, purge, val):
            if not 0 <= bounds['start'] <= bounds['stop']:
                raise ValueError('Invalid split range for ' + route['path'])
        if (purge['start'] != train['stop'] + 1 or val['start'] != purge['stop'] + 1
                or purge['stop'] - purge['start'] + 1 < int(spec['purge_frames'])
                or train['stop'] + 25 >= val['start']):
            raise ValueError('Split ranges overlap or leak future ego targets across validation')
        if val['stop'] + 25 >= int(route['frame_count']):
            raise ValueError('Validation extends beyond the available future ego horizon')
    return spec, hashlib.sha256(content).hexdigest()


def split_members(spec, split):
    key = "validation" if split in ("val", "validation") else split
    if key not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    members = set()
    for route in spec["routes"]:
        bounds = route[key]
        for frame in range(int(bounds["start"]), int(bounds["stop"]) + 1):
            members.add((str(route["path"]), frame))
    return key, members


def load_instruction_ranges(path: Path):
    by_route = {}
    policies = set()
    for row in read_jsonl(path):
        required = {
            "schema_version", "route_id", "start_frame", "stop_frame",
            "command", "text", "source", "route_sha256",
        }
        if row.get("schema_version") == LOOKAHEAD_INSTRUCTION_SCHEMA:
            required |= {"instruction_policy", "lookahead_m", "progress_source",
                         "curve_lookahead_m", "curve_threshold_deg"}
            lookahead_m = float(row.get("lookahead_m", 0))
            curve_lookahead_m = float(row.get("curve_lookahead_m", 0))
            curve_threshold_deg = float(row.get("curve_threshold_deg", 0))
            if (row.get("instruction_policy") != "route_lookahead"
                    or row.get("progress_source") != "current_ego_route_projection"
                    or not math.isfinite(lookahead_m) or lookahead_m <= 0
                    or not math.isfinite(curve_lookahead_m) or curve_lookahead_m <= 0
                    or not math.isfinite(curve_threshold_deg) or not 0 < curve_threshold_deg < 180):
                raise ValueError("Invalid route-lookahead instruction policy")
            policies.add((LOOKAHEAD_INSTRUCTION_SCHEMA, lookahead_m, curve_lookahead_m, curve_threshold_deg))
        elif row.get("schema_version") == INSTRUCTION_SCHEMA:
            policies.add((INSTRUCTION_SCHEMA, None))
        else:
            raise ValueError("Invalid global-path instruction record")
        if set(row) != required:
            raise ValueError("Invalid global-path instruction record")
        if (not isinstance(row["text"], str) or not row["text"].strip()
                or row["source"] != "recorded_global_path"):
            raise ValueError("Instruction must be a nonempty recorded-global-path command")
        start, stop = int(row["start_frame"]), int(row["stop_frame"])
        if start < 0 or stop < start:
            raise ValueError("Invalid instruction frame range")
        by_route.setdefault(row["route_id"], []).append(row)
    if len(policies) != 1:
        raise ValueError("Instruction file must use one consistent policy")
    for route, rows in by_route.items():
        rows.sort(key=lambda row: row["start_frame"])
        for previous, current in zip(rows, rows[1:]):
            if int(previous["stop_frame"]) + 1 != int(current["start_frame"]):
                raise ValueError("Instruction ranges are not contiguous for {}".format(route))
    return by_route


class MoraiExpertVLDriveDataset(Dataset):
    """T-frame samples from continuous source spans for frozen-token training."""

    trajectory_coordinate_convention = "lmdrive_left_negative_forward"

    def __init__(self, dataset_root, *, cache_root, split_file, split,
                 sequence_length=11, instructions=None, verify_hashes=True,
                 instruction_boundary_policy="exclude", frame_sampling_interval=1):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        if type(frame_sampling_interval) is not int or frame_sampling_interval < 1:
            raise ValueError('Invalid frame_sampling_interval')
        self.frame_sampling_interval = frame_sampling_interval
        self.verify_hashes = bool(verify_hashes)
        self.instruction_boundary_policy = str(instruction_boundary_policy)
        if self.instruction_boundary_policy not in ("exclude", "current_with_history"):
            raise ValueError("instruction_boundary_policy must be exclude or current_with_history")
        self._verified = set()
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        split_spec, split_sha = load_split(Path(split_file).expanduser().resolve())
        self.split, allowed = split_members(split_spec, split)
        self.split_sha256 = split_sha

        meta_path = self.cache_root / "cache_meta.json"
        manifest_path = self.cache_root / "manifest.jsonl"
        self.cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.cache_meta_sha256 = sha256_file(meta_path)
        if self.cache_meta.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("Wrong MORAI visual cache schema")
        dataset_index = self.dataset_root / "dataset_index.txt"
        if (not dataset_index.is_file()
                or self.cache_meta.get("dataset_index_sha256") != sha256_file(dataset_index)):
            raise ValueError("Visual cache and MORAI dataset index differ")
        if not isinstance(self.cache_meta.get("implementation_sha256"), dict):
            raise ValueError("Visual cache has no implementation provenance")
        if self.cache_meta.get("split_sha256") != split_sha:
            raise ValueError("Visual cache and split manifest differ")
        if self.cache_meta.get("feature_shape") != list(FEATURE_SHAPE):
            raise ValueError("Visual cache feature shape mismatch")
        if self.cache_meta.get("velocity_source") != "current_ego_speed":
            raise ValueError("Visual cache was not produced with current ego speed")
        if self.cache_meta.get("target_point_source") != "recorded_global_path_x_command_y_command":
            raise ValueError("Visual cache has the wrong navigation target-point source")
        if self.cache_meta.get("lidar_sweeps") != 1:
            raise ValueError("Visual cache must use exactly the current LiDAR sweep")
        if self.cache_meta.get("lidar_axis_policy") != "morai_flu_to_lmdrive_fru":
            raise ValueError("Visual cache LiDAR axis policy mismatch")
        if sha256_file(manifest_path) != self.cache_meta.get("manifest_sha256"):
            raise ValueError("Visual cache manifest hash mismatch")

        instruction_path = Path(instructions).expanduser().resolve() if instructions else None
        if instruction_path is None or not instruction_path.is_file():
            raise ValueError("MORAI VLDrive requires generated global-path instructions")
        self.instructions_path = instruction_path
        self.instruction_sha256 = sha256_file(instruction_path)
        instruction_ranges = load_instruction_ranges(instruction_path)
        first_instruction = next(iter(instruction_ranges.values()))[0]
        self.instruction_schema = first_instruction["schema_version"]
        self.instruction_policy = first_instruction.get("instruction_policy", "current_link")
        self.instruction_lookahead_m = first_instruction.get("lookahead_m")
        self.instruction_curve_lookahead_m = first_instruction.get("curve_lookahead_m")
        self.instruction_curve_threshold_deg = first_instruction.get("curve_threshold_deg")
        for route in split_spec['routes']:
            route_id = route['path']
            route_file = safe_artifact(self.dataset_root / 'data', route_id + '/route.json')
            if not route_file.is_file():
                raise ValueError('Instruction route file is missing: ' + str(route_file))
            route_hash = sha256_file(route_file)
            if any(row['route_sha256'] != route_hash for row in instruction_ranges.get(route_id, [])):
                raise ValueError('Instruction route hash mismatch: ' + route_id)

        def instruction_for(route, frame):
            matches = [row for row in instruction_ranges.get(route, [])
                       if int(row["start_frame"]) <= frame <= int(row["stop_frame"])]
            if len(matches) != 1:
                raise ValueError("Frame has no unique instruction: {} {}".format(route, frame))
            return matches[0]["text"], int(matches[0]["command"])

        entries = []
        seen = set()
        for row in read_jsonl(manifest_path):
            if row.get("schema_version") != CACHE_SCHEMA:
                raise ValueError("Mixed visual cache schema")
            key = (str(row["route_id"]), int(row["frame_id"]))
            if key in seen:
                raise ValueError("Duplicate visual cache frame: {}".format(key))
            seen.add(key)
            if key not in allowed:
                continue
            if row.get("split") != self.split:
                raise ValueError("Cache row split disagrees with split manifest")
            if row["visual_feature"].get("shape") != list(FEATURE_SHAPE):
                raise ValueError("Cached feature row shape mismatch")
            if row["future_trajectory"].get("shape") != list(TRAJECTORY_SHAPE):
                raise ValueError("Cached trajectory row shape mismatch")
            if (row['future_trajectory'].get('source') != 'future_synchronized_ego_positions'
                    or row['future_trajectory'].get('horizons_s') != [0.5, 1.0, 1.5, 2.0, 2.5]):
                raise ValueError('Cached waypoint GT must be the five actual future ego positions')
            text, command = instruction_for(key[0], key[1])
            entry = dict(row)
            entry["instruction"] = text
            entry["command"] = command
            entries.append(entry)
        if len(entries) != len(allowed):
            missing = sorted(allowed - seen)[:5]
            raise ValueError("Visual cache is incomplete for {}: {}".format(self.split, missing))
        entries.sort(key=lambda row: (row["route_id"], int(row["frame_id"])))
        self.entries = entries
        self.checkpoint_sha256 = self.cache_meta["checkpoint_sha256"]
        self.lidar_axis_policy = self.cache_meta["lidar_axis_policy"]

        self.windows = []
        self.exclusions = {"discontinuous": 0, "instruction_change": 0}
        span = (self.sequence_length - 1) * self.frame_sampling_interval + 1
        for end in range(span - 1, len(entries)):
            window = entries[end + 1 - span:end + 1]
            continuous = all(
                current["route_id"] == following["route_id"]
                and int(following["frame_id"]) == int(current["frame_id"]) + 1
                and int(following["native_sync_frame"]) == int(current["native_sync_frame"]) + 5
                for current, following in zip(window, window[1:])
            )
            if not continuous:
                self.exclusions["discontinuous"] += 1
                continue
            if (self.instruction_boundary_policy == "exclude"
                    and len({row["instruction"] for row in window}) != 1):
                self.exclusions["instruction_change"] += 1
                continue
            self.windows.append((end + 1 - span, end + 1))
        if not self.windows:
            raise ValueError("No valid consecutive MORAI VLDrive windows")

    def __len__(self):
        return len(self.windows)

    def _array(self, ref, shape, label):
        path = safe_artifact(self.cache_root, ref["path"])
        if self.verify_hashes and path not in self._verified:
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("{} artifact hash mismatch: {}".format(label, path))
            self._verified.add(path)
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != shape or not np.isfinite(array).all():
            raise ValueError("Invalid {} array: {}".format(label, path))
        return torch.from_numpy(np.asarray(array, dtype=np.float32))

    def __getitem__(self, index):
        begin, end = self.windows[int(index)]
        continuous_rows = self.entries[begin:end]
        rows = continuous_rows[::self.frame_sampling_interval]
        visual = torch.stack([
            self._array(row["visual_feature"], FEATURE_SHAPE, "visual feature")
            for row in rows
        ])
        trajectory = torch.stack([
            self._array(row["future_trajectory"], TRAJECTORY_SHAPE, "future trajectory")
            for row in rows
        ])
        current_instruction = rows[-1]["instruction"]
        targets = {"future_trajectory": trajectory}
        if self.instruction_boundary_policy == "current_with_history":
            # The shared text prefix is the instruction available at the anchor.
            # Keep prior visual history, but never supervise old-frame outputs
            # using an instruction that had not yet been issued at those frames.
            suffix_begin = len(continuous_rows) - 1
            while suffix_begin > 0 and continuous_rows[suffix_begin - 1]["instruction"] == current_instruction:
                suffix_begin -= 1
            targets["frame_loss_mask"] = torch.arange(0, len(continuous_rows), self.frame_sampling_interval) >= suffix_begin
        sample = {
            "model_inputs": {
                "visual_feature": visual,
                "instruction": current_instruction,
            },
            "training_targets": targets,
            "metadata": {
                "run_id": rows[0]["route_id"],
                "sample_ids": ["{}/{:04d}".format(row["route_id"], int(row["frame_id"])) for row in rows],
                "frame_ids": torch.tensor([int(row["frame_id"]) for row in rows], dtype=torch.int64),
                "native_sync_frames": torch.tensor([int(row["native_sync_frame"]) for row in rows], dtype=torch.int64),
                "command": int(rows[-1 if self.instruction_boundary_policy == "current_with_history" else 0]["command"]),
                "input_mode": "cached",
                "purpose": self.split,
            },
        }
        if self.instruction_boundary_policy == "current_with_history":
            sample["metadata"]["frame_commands"] = torch.tensor(
                [int(row["command"]) for row in rows], dtype=torch.int64)
        return sample

    def describe(self):
        description = {
            "schema_version": "morai-expert-vldrive-dataset-v1",
            "purpose": self.split,
            "input_mode": "cached",
            "sequence_length": self.sequence_length,
            "frame_sampling_interval": self.frame_sampling_interval,
            "frames": len(self.entries),
            "windows": len(self.windows),
            "excluded_windows": dict(self.exclusions),
            "checkpoint_sha256": self.checkpoint_sha256,
            "cache_meta_sha256": self.cache_meta_sha256,
            "cache_manifest_sha256": self.cache_meta["manifest_sha256"],
            "cache_implementation_sha256": self.cache_meta["implementation_sha256"],
            "split_sha256": self.split_sha256,
            "instruction_sha256": self.instruction_sha256,
            "feature_contract": list(FEATURE_SHAPE),
            "trajectory_source": "future synchronized ego positions",
            "instruction_source": "recorded global path command",
            "vision_target_point_source": self.cache_meta["target_point_source"],
            "vision_target_point_usage": "pretraining_waypoint_head_only_ignored_by_token_extractor",
            "model_inputs": ["cached visual feature", "navigation instruction"],
        }
        if (self.instruction_boundary_policy != "exclude"
                or self.instruction_schema != INSTRUCTION_SCHEMA):
            description.update(
                schema_version="morai-expert-vldrive-dataset-v2",
                instruction_schema=self.instruction_schema,
                instruction_policy=self.instruction_policy,
                instruction_lookahead_m=self.instruction_lookahead_m,
                instruction_curve_lookahead_m=self.instruction_curve_lookahead_m,
                instruction_curve_threshold_deg=self.instruction_curve_threshold_deg,
                instruction_boundary_policy=self.instruction_boundary_policy,
                waypoint_supervision_policy=("current_instruction_contiguous_suffix"
                    if self.instruction_boundary_policy == "current_with_history" else "all_frames"),
            )
        return description
