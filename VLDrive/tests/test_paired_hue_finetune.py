"""CPU-only tests; no full encoder construction, CUDA work or original writes."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import paired_hue_finetune as pair
from datasets.transforms import create_camera_transform


def picture():
    array = np.zeros((144, 256, 3), dtype=np.uint8)
    array[:, :85] = (230, 15, 20)
    array[:, 85:170] = (15, 230, 20)
    array[:, 170:] = (20, 15, 230)
    return Image.fromarray(array)


class FixtureDataset:
    def __init__(self):
        self.route_frames = [("fixture", index) for index in range(3)]
        self.rgb_transform = create_camera_transform(224, is_training=True,
            scale=[.9, 1.1], color_jitter=.4, hue_jitter=.4)
        self.rgb_center_transform = create_camera_transform(128, is_training=True,
            scale=None, color_jitter=.4, hue_jitter=.4, need_scale=False)
        self.multi_view_transform = create_camera_transform(128, is_training=True,
            scale=[.9, 1.1], color_jitter=.4, hue_jitter=.4)

    def __getitem__(self, index):
        image = picture()
        # Same LiDAR-before-RGB order as the real dataset adapter.
        lidar = np.random.permutation(16).astype(np.float32).reshape(4, 4)
        values = {"rgb_front": self.rgb_transform(image),
                  "rgb_center": self.rgb_center_transform(image),
                  "rgb_left": self.multi_view_transform(image),
                  "rgb_right": self.multi_view_transform(image),
                  "rgb_rear": self.multi_view_transform(image),
                  "lidar": lidar, "num_points": 4,
                  "velocity": torch.tensor([0.]), "target_point": torch.tensor([1., 2.])}
        target = {"waypoints": torch.arange(10).reshape(5, 2).float(),
                  "traffic_light": torch.tensor(1), "traffic_available": torch.tensor(True),
                  "frame_id": torch.tensor(index), "native_sync_frame": torch.tensor(index * 5)}
        return values, target


def test_legacy_jitter_exact_tensor_and_rng_consumption():
    for seed in range(32):
        with pair.local_data_rng(seed):
            expected = transforms.ToTensor()(transforms.ColorJitter(.4, .4, .4, .4)(picture()))
            expected_rng = torch.get_rng_state()
        trace = []
        with pair.local_data_rng(seed):
            actual = transforms.ToTensor()(pair.PairedColorJitter(True, trace)(picture()))
            actual_rng = torch.get_rng_state()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(actual_rng, expected_rng)
        assert len(trace) == 1 and trace[0]["sampled_hue"] is not None


def test_hue_omission_keeps_other_params_order_and_rng_identical():
    for seed in range(16):
        traces, states = [], []
        for hue in (True, False):
            trace = []
            with pair.local_data_rng(seed):
                pair.PairedColorJitter(hue, trace)(picture())
                states.append(torch.get_rng_state())
            traces.append(trace)
        assert traces[0] == traces[1]
        assert torch.equal(*states)


def test_seeded_samples_pair_all_noncolor_inputs_gt_geometry_and_order():
    legacy, safe, actual = FixtureDataset(), FixtureDataset(), FixtureDataset()
    trace_a, trace_b = [], []
    pair.install_paired_transforms(legacy, True, trace_a)
    pair.install_paired_transforms(safe, False, trace_b)
    samples_a = pair.SeededSamples(legacy, [2, 0, 1], 607, 2, True, trace_a)
    samples_b = pair.SeededSamples(safe, [2, 0, 1], 607, 2, True, trace_b)
    for position in range(3):
        inputs_a, targets_a, audit_a = samples_a[position]
        inputs_b, targets_b, audit_b = samples_b[position]
        with pair.local_data_rng(audit_a["sample_seed"]):
            actual_inputs, actual_targets = actual[audit_a["dataset_index"]]
        assert pair.tensor_identity(actual_inputs) == pair.tensor_identity(inputs_a)
        assert pair.tensor_identity(actual_targets) == pair.tensor_identity(targets_a)
        for key in ("sample_seed", "augmentation_trace_sha256", "non_rgb_sha256", "targets_sha256"):
            assert audit_a[key] == audit_b[key]
        assert audit_a["rgb_sha256"] != audit_b["rgb_sha256"]
        for key in inputs_a:
            assert np.asarray(inputs_a[key]).shape == np.asarray(inputs_b[key]).shape
        assert pair.tensor_identity(targets_a) == pair.tensor_identity(targets_b)


def test_data_rng_restores_parent_rng_and_never_seeds_cuda():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with mock.patch("torch.cuda.manual_seed_all", side_effect=AssertionError("Touched CUDA RNG")):
        with pair.local_data_rng(42):
            random.random(), np.random.random(), torch.rand(5)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_orders_cover_original_examples_once_and_change_by_epoch():
    first = pair.epoch_order(3352, 607, 0)
    assert sorted(first) == list(range(3352))
    assert first == pair.epoch_order(3352, 607, 0)
    assert first != pair.epoch_order(3352, 607, 1)
    assert pair.optimizer_budget(3352, 3)["updates_per_arm"] == 315
    assert pair.optimizer_budget(3352, 3)["presentations_per_arm"] == 10056
    assert pair.optimizer_budget(8, 1)["updates_per_arm"] == 1


def test_pairing_guard_rejects_changes_and_short_arms():
    row = {"audit_hashes": {k: k for k in
        ("order", "non_rgb_sha256", "targets_sha256", "augmentation_trace_sha256")},
        "optimizer_state_steps": [105], "actual_lrs": [5e-6, 2e-6]}
    pair.assert_arm_pairing([row], [copy.deepcopy(row)])
    for key in row["audit_hashes"]:
        changed = copy.deepcopy(row)
        changed["audit_hashes"][key] = "wrong"
        with pytest.raises(ValueError, match="beyond hue"):
            pair.assert_arm_pairing([row], [changed])
    with pytest.raises(ValueError, match="numbers of completed epochs"):
        pair.assert_arm_pairing([row], [])
    changed = copy.deepcopy(row)
    changed["optimizer_state_steps"] = [104]
    with pytest.raises(ValueError, match="optimizer exposure"):
        pair.assert_arm_pairing([row], [changed])


def test_cli_smoke_bound_requires_separate_output():
    base = pair.OUTPUT_PARENT / "diagnosis_upstream_hue_fixture" / "signal_v1"
    assert pair.parse_args(["--output", str(base)]).action == "prepare"
    assert pair.parse_args(["--output", str(base.with_name("smoke_signal_v1")), "--smoke"]).smoke
    for extra in (["--max-train-frames", "8"], ["--smoke"],
                  ["--smoke", "--max-train-frames", "65"]):
        with pytest.raises(SystemExit):
            pair.parse_args(["--output", str(base)] + list(extra))
    with pytest.raises(SystemExit):
        pair.parse_args(["--output", str(pair.ROOT)])


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = torch.nn.Linear(2, 2)
        self.lidar_backbone = torch.nn.Linear(2, 2)
        self.fusion = torch.nn.Linear(2, 2)


def test_actual_cpu_updates_include_backbones_and_preserve_source(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"fixture": True}))
    source_path = tmp_path / "source.pt"
    source_model = TinyEncoder()
    torch.save({"state_dict": source_model.state_dict(), "epoch": 24,
                "model_name": pair.trainer.MODEL_NAME, "feature_model_name": pair.trainer.FEATURE_MODEL_NAME,
                "token_shape": [106, 256], "split_sha256": pair.trainer.sha256_file(split),
                "initialization": {"full_lmdrive_encoder": False},
                "optimizer": {"must_not_restore": True}, "scaler": {"must_not_restore": True},
                "args": {"lidar_axis_policy": "morai_flu_to_lmdrive_fru"}}, source_path)
    original_sha = pair.trainer.sha256_file(source_path)
    args = SimpleNamespace(dataset_root=tmp_path, split_file=split, traffic_label_sidecar=None,
                           seed=607, smoke=True, output=tmp_path / "new")
    config = pair.trainer_args(args)
    config.warm_start, config.warm_start_sha256 = source_path, original_sha
    initial_ids = []
    for _ in range(2):
        model = TinyEncoder()
        optimizer, fusion_lr, rgb_lr = pair.trainer.optimizer_for(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        start, _, provenance = pair.trainer.load_training_initialization(model, optimizer, scaler, config)
        assert start == 0 and not optimizer.state and not provenance["optimizer_restored"]
        assert fusion_lr == pytest.approx(5e-6) and rgb_lr == pytest.approx(2e-6)
        initial_ids.append(pair.model_state_identity(model))
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        model.train()
        pair.trainer.set_rgb_frozen(model, False)
        sum(p.sum() for p in model.parameters()).backward()
        optimizer.step()
        assert pair.optimizer_steps(optimizer) == [1]
        assert all(g["changed_parameters"] > 0 for g in pair.parameter_changes(model, initial).values())
    assert initial_ids[0] == initial_ids[1]
    assert pair.trainer.sha256_file(source_path) == original_sha


def test_evaluation_mask_affects_traffic_only_not_waypoint_count():
    probability = np.array([.1, .1, .9])
    labels, available = np.array([0, 1, 1]), np.array([True, True, False])
    result = pair.metrics_for_predictions(probability, labels, available, np.zeros(3),
        np.zeros((3, 5, 2)), np.ones((3, 5, 2)), np.array([990, 991, 992]))
    assert result["waypoint_all"]["count"] == 3
    assert result["groups"]["traffic_valid"]["count"] == 2
    assert result["groups"]["notred_stop"]["count"] == 1
    assert result["groups"]["firstgreen"]["notred_correct_count"] == 0


def test_inherited_lock_is_validated_and_remains_parent_owned(tmp_path):
    import fcntl
    lock_path = tmp_path / "training.lock"
    with mock.patch.object(pair, "LOCK", lock_path), lock_path.open("a+") as parent:
        fcntl.flock(parent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pair.training_lock(parent.fileno()):
            pass
        with lock_path.open("a+") as outsider:
            with pytest.raises(BlockingIOError):
                fcntl.flock(outsider.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        wrong_path = tmp_path / "wrong.lock"
        with wrong_path.open("a+") as wrong:
            with pytest.raises(ValueError, match="not the expected"):
                with pair.training_lock(wrong.fileno()):
                    pass


def test_direct_lock_is_released_by_its_owner(tmp_path):
    import fcntl
    lock_path = tmp_path / "training.lock"
    with mock.patch.object(pair, "LOCK", lock_path):
        with pair.training_lock():
            pass
        with lock_path.open("a+") as next_owner:
            fcntl.flock(next_owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_diagnostic_outputs_cannot_overwrite_existing_file(tmp_path):
    path = tmp_path / "report.json"
    pair.write_json_new(path, {"original": 1})
    with pytest.raises(FileExistsError):
        pair.write_json_new(path, {"replacement": 2})
    assert json.loads(path.read_text()) == {"original": 1}


def test_initial_forward_pairing_uses_frame_ids_and_all_inputs_targets(tmp_path):
    baseline = tmp_path / "baseline.npz"
    probe = tmp_path / "probe.npz"
    values = {"frame_ids": np.array([990, 991, 992]), "labels": np.array([0, 1, 1]),
              "available": np.ones(3, dtype=bool), "current_speed": np.zeros(3),
              "waypoint_GT": np.ones((3, 5, 2), dtype=np.float32),
              "input_target_identity": np.array(["a", "b", "c"]),
              "traffic_logits": np.arange(6).reshape(3, 2).astype(np.float32),
              "waypoints": np.zeros((3, 5, 2), dtype=np.float32)}
    np.savez(baseline, **values)
    subset = {key: value[[1, 0]] for key, value in values.items()}
    np.savez(probe, **subset)
    report = pair.compare_initial_predictions(baseline, probe)
    assert report["count"] == 2
    assert report["traffic_logits"]["bitwise_equal"]
    subset["input_target_identity"] = np.array(["changed", "a"])
    np.savez(probe, **subset)
    with pytest.raises(ValueError, match="input/target mismatch"):
        pair.compare_initial_predictions(baseline, probe)


def test_initial_forward_pairing_fails_material_output_change(tmp_path):
    values = {"frame_ids": np.array([991]), "labels": np.array([1]),
              "available": np.ones(1, dtype=bool), "current_speed": np.zeros(1),
              "waypoint_GT": np.ones((1, 5, 2), dtype=np.float32),
              "input_target_identity": np.array(["same"]),
              "traffic_logits": np.ones((1, 2), dtype=np.float32),
              "waypoints": np.zeros((1, 5, 2), dtype=np.float32)}
    baseline, probe = tmp_path / "baseline.npz", tmp_path / "probe.npz"
    np.savez(baseline, **values)
    values["waypoints"][0, 0, 0] = .1
    np.savez(probe, **values)
    with pytest.raises(ValueError, match="numerical tolerance"):
        pair.compare_initial_predictions(baseline, probe)
