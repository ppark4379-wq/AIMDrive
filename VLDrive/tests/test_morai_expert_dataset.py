import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from datasets.morai_expert_dataset import (
    CACHE_SCHEMA,
    INSTRUCTION_SCHEMA,
    LOOKAHEAD_INSTRUCTION_SCHEMA,
    SPLIT_SCHEMA,
    MoraiExpertVLDriveDataset,
    load_split,
    load_instruction_ranges,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _build_fixture(tmp_path):
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    dataset_root.mkdir()
    cache_root.mkdir()
    (dataset_root / "dataset_index.txt").write_text("data/route_a/ 66\n", encoding="utf-8")
    route_file = dataset_root / 'data' / 'route_a' / 'route.json'
    route_file.parent.mkdir(parents=True)
    route_file.write_text('{"nav_road_options": [4,3]}', encoding='utf-8')
    split = {
        "schema_version": SPLIT_SCHEMA,
        "purge_frames": 25,
        "future_offsets": [5, 10, 15, 20, 25],
        "routes": [{
            "path": "route_a",
            "frame_count": 66,
            "train": {"start": 0, "stop": 9},
            "purge": {"start": 10, "stop": 34},
            "validation": {"start": 35, "stop": 40},
        }],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    instructions = [
        dict(schema_version=INSTRUCTION_SCHEMA, route_id="route_a", start_frame=0, stop_frame=4,
             command=4, text="Proceed along this route.", source="recorded_global_path", route_sha256=_sha(route_file)),
        dict(schema_version=INSTRUCTION_SCHEMA, route_id="route_a", start_frame=5, stop_frame=65,
             command=3, text="Please proceed straight at the next intersection.",
             source="recorded_global_path", route_sha256=_sha(route_file)),
    ]
    instruction_path = tmp_path / "instructions.jsonl"
    _jsonl(instruction_path, instructions)

    rows = []
    for frame in list(range(10)) + list(range(35, 41)):
        feature_path = cache_root / "features" / "route_a" / ("%04d.npy" % frame)
        target_path = cache_root / "targets" / "route_a" / ("%04d.npy" % frame)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, np.full((106, 256), frame, dtype=np.float16), allow_pickle=False)
        np.save(target_path, np.full((5, 2), frame / 10.0, dtype=np.float32), allow_pickle=False)
        rows.append({
            "schema_version": CACHE_SCHEMA,
            "route_id": "route_a",
            "frame_id": frame,
            "native_sync_frame": 100 + frame * 5,
            "split": "train" if frame < 10 else "validation",
            "visual_feature": {"path": feature_path.relative_to(cache_root).as_posix(),
                               "shape": [106, 256], "dtype": "float16", "sha256": _sha(feature_path)},
            "future_trajectory": {"path": target_path.relative_to(cache_root).as_posix(),
                                  "shape": [5, 2], "dtype": "float32", "sha256": _sha(target_path),
                                  "source": "future_synchronized_ego_positions",
                                  "horizons_s": [0.5, 1.0, 1.5, 2.0, 2.5]},
        })
    manifest_path = cache_root / "manifest.jsonl"
    _jsonl(manifest_path, rows)
    meta = {
        "schema_version": CACHE_SCHEMA,
        "dataset_index_sha256": _sha(dataset_root / "dataset_index.txt"),
        "implementation_sha256": {"cache_tool.py": "implementation-hash"},
        "split_sha256": _sha(split_path),
        "feature_shape": [106, 256],
        "velocity_source": "current_ego_speed",
        "target_point_source": "recorded_global_path_x_command_y_command",
        "lidar_sweeps": 1,
        "lidar_axis_policy": "morai_flu_to_lmdrive_fru",
        "manifest_sha256": _sha(manifest_path),
        "checkpoint_sha256": "vision-checkpoint-hash",
    }
    (cache_root / "cache_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return dataset_root, cache_root, split_path, instruction_path


def test_actual_ego_targets_and_global_path_instructions_are_separate(tmp_path):
    dataset_root, cache_root, split, instructions = _build_fixture(tmp_path)
    dataset = MoraiExpertVLDriveDataset(
        dataset_root, cache_root=cache_root, split_file=split, split="train",
        sequence_length=3, instructions=instructions,
    )

    # Windows spanning frame 4 -> 5 are removed because their instruction changes.
    assert len(dataset) == 6
    assert dataset.exclusions["instruction_change"] == 2
    sample = dataset[0]
    assert tuple(sample["model_inputs"]["visual_feature"].shape) == (3, 106, 256)
    assert tuple(sample["training_targets"]["future_trajectory"].shape) == (3, 5, 2)
    assert sample["model_inputs"]["instruction"] == "Proceed along this route."
    assert np.isclose(sample["training_targets"]["future_trajectory"][-1, 0, 0].item(), 0.2)
    assert "objects" not in sample["model_inputs"]
    assert dataset.describe()["trajectory_source"] == "future synchronized ego positions"


def test_notice_entry_and_exit_keep_history_but_mask_old_condition(tmp_path):
    from utils.notice_policy import NOTICE_SCHEMA
    root, cache, split, instructions = _build_fixture(tmp_path)
    notices = tmp_path / "notices.jsonl"
    common = dict(schema_version=NOTICE_SCHEMA, route_id="route_a", source="current_ego_snapped_global_path",
                  route_sha256=_sha(root / "data/route_a/route.json"),
                  zones_sha256="zones", global_path_sha256="global-path")
    _jsonl(notices, [dict(common, start_frame=0, stop_frame=2, text="", zone_ids=[]),
                     dict(common, start_frame=3, stop_frame=5, text="Watch obstacles.", zone_ids=["static"]),
                     dict(common, start_frame=6, stop_frame=65, text="", zone_ids=[])])
    dataset = MoraiExpertVLDriveDataset(root, cache_root=cache, split_file=split, split="train",
        sequence_length=3, instructions=instructions, notices=notices)
    entry = dataset[1]  # [1, 2, 3]
    assert entry["model_inputs"]["notice_text"] == "Watch obstacles."
    assert entry["training_targets"]["frame_loss_mask"].tolist() == [False, False, True]
    assert dataset[2]["training_targets"]["frame_loss_mask"].tolist() == [False, True, True]
    exit_sample = dataset[4]  # [4, 5, 6]: both a nav transition and notice exit
    assert exit_sample["model_inputs"]["notice_text"] == ""
    assert exit_sample["training_targets"]["frame_loss_mask"].tolist() == [False, False, True]


def test_validation_never_uses_purged_boundary_frames(tmp_path):
    dataset_root, cache_root, split, instructions = _build_fixture(tmp_path)
    dataset = MoraiExpertVLDriveDataset(
        dataset_root, cache_root=cache_root, split_file=split, split="validation",
        sequence_length=3, instructions=instructions,
    )
    assert len(dataset) == 4
    first = dataset[0]
    assert first["metadata"]["frame_ids"].tolist() == [35, 36, 37]
    assert first["metadata"]["native_sync_frames"].tolist() == [275, 280, 285]


def test_declared_purge_does_not_hide_actual_target_overlap(tmp_path):
    _, _, split_path, _ = _build_fixture(tmp_path)
    split = json.loads(split_path.read_text())
    split['routes'][0]['purge']['stop'] = 14
    split['routes'][0]['validation']['start'] = 15
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match='overlap or leak'):
        load_split(split_path)


def test_changed_global_route_rejects_stale_instructions(tmp_path):
    root, cache, split, instructions = _build_fixture(tmp_path)
    (root / 'data/route_a/route.json').write_text('{"nav_road_options": [1,2]}')
    with pytest.raises(ValueError, match='route hash mismatch'):
        MoraiExpertVLDriveDataset(root, cache_root=cache, split_file=split,
                                 split='train', sequence_length=3, instructions=instructions)


def test_cache_hash_mismatch_is_rejected(tmp_path):
    dataset_root, cache_root, split, instructions = _build_fixture(tmp_path)
    feature = cache_root / "features" / "route_a" / "0000.npy"
    feature.write_bytes(feature.read_bytes() + b"corrupt")
    dataset = MoraiExpertVLDriveDataset(
        dataset_root, cache_root=cache_root, split_file=split, split="train",
        sequence_length=3, instructions=instructions,
    )
    try:
        dataset[0]
    except ValueError as error:
        assert "artifact hash mismatch" in str(error)
    else:
        raise AssertionError("Corrupted cached token was accepted")


def test_instruction_boundary_keeps_history_without_supervising_past_with_new_text(tmp_path):
    root, cache, split, instructions = _build_fixture(tmp_path)
    dataset = MoraiExpertVLDriveDataset(
        root, cache_root=cache, split_file=split, split="train", sequence_length=3,
        instructions=instructions, instruction_boundary_policy="current_with_history")
    assert len(dataset) == 8
    assert dataset.exclusions["instruction_change"] == 0
    boundary = dataset[3]
    assert boundary["metadata"]["frame_ids"].tolist() == [3, 4, 5]
    assert boundary["model_inputs"]["instruction"] == "Please proceed straight at the next intersection."
    assert boundary["training_targets"]["frame_loss_mask"].tolist() == [False, False, True]
    assert boundary["metadata"]["frame_commands"].tolist() == [4, 4, 3]
    assert dataset[4]["training_targets"]["frame_loss_mask"].tolist() == [False, True, True]
    assert dataset[5]["training_targets"]["frame_loss_mask"].tolist() == [True, True, True]
    assert dataset.describe()["waypoint_supervision_policy"] == "current_instruction_contiguous_suffix"


def test_repeated_instruction_only_supervises_latest_contiguous_suffix(tmp_path):
    root, cache, split, instructions = _build_fixture(tmp_path)
    original = [json.loads(line) for line in instructions.read_text().splitlines()]
    records = [dict(original[0], stop_frame=2),
               dict(original[1], start_frame=3, stop_frame=4),
               dict(original[0], start_frame=5, stop_frame=65)]
    _jsonl(instructions, records)
    dataset = MoraiExpertVLDriveDataset(
        root, cache_root=cache, split_file=split, split="train", sequence_length=5,
        instructions=instructions, instruction_boundary_policy="current_with_history")
    sample = dataset[1]
    assert sample["metadata"]["frame_ids"].tolist() == [1, 2, 3, 4, 5]
    assert sample["training_targets"]["frame_loss_mask"].tolist() == [False, False, False, False, True]


def test_boundary_policy_still_cannot_cross_train_purge_validation(tmp_path):
    root, cache, split, instructions = _build_fixture(tmp_path)
    dataset = MoraiExpertVLDriveDataset(
        root, cache_root=cache, split_file=split, split="validation", sequence_length=3,
        instructions=instructions, instruction_boundary_policy="current_with_history")
    assert len(dataset) == 4
    assert dataset[0]["metadata"]["frame_ids"].tolist() == [35, 36, 37]


def test_v2_instruction_policy_has_explicit_provenance(tmp_path):
    root, cache, split, instructions = _build_fixture(tmp_path)
    records = [json.loads(line) for line in instructions.read_text().splitlines()]
    for row in records:
        row.update(schema_version=LOOKAHEAD_INSTRUCTION_SCHEMA,
                   instruction_policy="route_lookahead", lookahead_m=30.0,
                   progress_source="current_ego_route_projection",
                   curve_lookahead_m=20.0, curve_threshold_deg=20.0)
    _jsonl(instructions, records)
    dataset = MoraiExpertVLDriveDataset(root, cache_root=cache, split_file=split,
                                      split="train", sequence_length=3, instructions=instructions)
    assert dataset.describe()["instruction_policy"] == "route_lookahead"
    assert dataset.describe()["instruction_lookahead_m"] == 30.0
    assert dataset.describe()["schema_version"] == "morai-expert-vldrive-dataset-v2"
    records[1]["lookahead_m"] = 10.0
    _jsonl(instructions, records)
    with pytest.raises(ValueError, match="consistent policy"):
        load_instruction_ranges(instructions)
