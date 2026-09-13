"""CPU-only guards for weights-only transfer; never construct the full encoder."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_morai_pretrain as trainer


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = torch.nn.Linear(2, 2)
        self.fusion = torch.nn.Linear(2, 2)


class FakeScaler:
    def __init__(self):
        self.state = {"scale": 65536.0}
        self.loads = 0

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.loads += 1
        self.state = copy.deepcopy(state)


class WarmStartTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.split = self.root / "split.json"
        self.split.write_text(json.dumps({"split": "fixture"}), encoding="utf-8")
        self.source = self.root / "original_epoch25.pth.tar"
        self.args = SimpleNamespace(
            warm_start=self.source, warm_start_sha256=None, resume=None,
            split_file=self.split, output=self.root / "new_run",
            traffic_label_sidecar=None, traffic_label_provenance=None,
            implementation_sha256={"fixture.py": "fixture-sha"},
            lidar_axis_policy="morai_flu_to_lmdrive_fru",
            batch_size=1, accumulation_steps=32,
            reference_lr=8e-5, reference_rgb_lr=3e-5, weight_decay=0.05,
            epochs=3, warmup_epochs=1,
        )
        source_model = TinyEncoder()
        source_optimizer, _, _ = trainer.optimizer_for(source_model, self.args)
        sum(parameter.sum() for parameter in source_model.parameters()).backward()
        source_optimizer.step()
        self.checkpoint = {
            "state_dict": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scaler": {"scale": 123.0},
            "epoch": 24, "best_fde_m": 0.3,
            "model_name": trainer.MODEL_NAME,
            "feature_model_name": trainer.FEATURE_MODEL_NAME,
            "token_shape": list(trainer.TOKEN_SHAPE),
            "split_sha256": trainer.sha256_file(self.split),
            "initialization": {"full_lmdrive_encoder": False},
            "args": {"lidar_axis_policy": self.args.lidar_axis_policy},
        }
        self.write_source()

    def write_source(self):
        torch.save(self.checkpoint, self.source)
        self.args.warm_start_sha256 = trainer.sha256_file(self.source)

    def new_training_state(self):
        model = TinyEncoder()
        optimizer, base_lr, rgb_lr = trainer.optimizer_for(model, self.args)
        return model, optimizer, FakeScaler(), base_lr, rgb_lr

    def test_weights_only_starts_at_zero_with_new_optimizer_scaler_and_schedule(self):
        original_sha = trainer.sha256_file(self.source)
        model, optimizer, scaler, base_lr, rgb_lr = self.new_training_state()
        start, best, provenance = trainer.load_training_initialization(
            model, optimizer, scaler, self.args)
        self.assertEqual(start, 0)
        self.assertEqual(best, float("inf"))
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(scaler.loads, 0)
        self.assertEqual(scaler.state, {"scale": 65536.0})
        self.assertFalse(provenance["exact_resume"])
        self.assertEqual(provenance["source_epoch"], 24)
        self.assertEqual(provenance["source_sha256"], original_sha)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, self.checkpoint["state_dict"][name], rtol=0, atol=0)
        rates = []
        for epoch in range(self.args.epochs):
            trainer.apply_epoch_lr(optimizer, base_lr, rgb_lr, epoch, self.args)
            rates.append(optimizer.param_groups[0]["lr"])
        for actual, expected in zip(rates, [5e-6, 5e-6, 2.5e-6]):
            self.assertAlmostEqual(actual, expected)
        # One local CPU update/checkpoint cannot mutate the parent checkpoint.
        sum(parameter.sum() for parameter in model.parameters()).backward()
        optimizer.step()
        trainer.prepare_output_directory(self.args)
        self.args.run_initialization = provenance
        trainer.save_checkpoint(self.args.output / "checkpoint-last.pth.tar", model,
                                optimizer, scaler, 0, 2.0, self.args, {})
        self.assertEqual(trainer.sha256_file(self.source), original_sha)
        saved = torch.load(self.args.output / "checkpoint-last.pth.tar", weights_only=True)
        self.assertEqual(saved["epoch"], 0)
        self.assertEqual(saved["run_initialization"], provenance)
        self.assertEqual(saved["implementation_sha256"], self.args.implementation_sha256)
        self.assertEqual(saved["initialization"]["lidar_backbone"], "MORAI warm-start checkpoint")

    def test_checkpoint_sha_checked_before_model_mutation(self):
        model, optimizer, scaler, _, _ = self.new_training_state()
        before = copy.deepcopy(model.state_dict())
        self.args.warm_start_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            trainer.load_training_initialization(model, optimizer, scaler, self.args)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_incompatible_provenance_and_strict_tensor_contract_are_rejected(self):
        original = copy.deepcopy(self.checkpoint)
        changes = [
            ("model_name", "different"),
            ("feature_model_name", "different"),
            ("token_shape", [104, 256]),
            ("split_sha256", "different"),
            ("initialization", {"full_lmdrive_encoder": True}),
            ("args", {"lidar_axis_policy": "native_passthrough"}),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                self.checkpoint = copy.deepcopy(original)
                self.checkpoint[key] = value
                self.write_source()
                model, optimizer, scaler, _, _ = self.new_training_state()
                with self.assertRaises(ValueError):
                    trainer.load_training_initialization(model, optimizer, scaler, self.args)
        for bad_tensor in (torch.zeros(3), torch.zeros(2, dtype=torch.float64),
                           torch.full((2,), float("nan"))):
            with self.subTest(tensor=bad_tensor):
                self.checkpoint = copy.deepcopy(original)
                self.checkpoint["state_dict"]["fusion.bias"] = bad_tensor
                self.write_source()
                model, optimizer, scaler, _, _ = self.new_training_state()
                with self.assertRaises(ValueError):
                    trainer.load_training_initialization(model, optimizer, scaler, self.args)

    def test_populated_output_and_source_directory_are_protected(self):
        self.args.output.mkdir()
        old = self.args.output / "anything.txt"
        old.write_text("must survive", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            trainer.prepare_output_directory(self.args)
        self.assertEqual(old.read_text(encoding="utf-8"), "must survive")
        self.args.output = self.source.parent
        with self.assertRaises(FileExistsError):
            trainer.prepare_output_directory(self.args)
        self.assertTrue(self.source.is_file())

    def test_resume_restores_optimizer_epoch_but_rejects_label_policy_change(self):
        self.args.warm_start = None
        self.args.resume = self.source
        model, optimizer, scaler, _, _ = self.new_training_state()
        start, best, provenance = trainer.load_training_initialization(
            model, optimizer, scaler, self.args)
        self.assertEqual(start, 25)
        self.assertEqual(best, 0.3)
        self.assertGreater(len(optimizer.state), 0)
        self.assertEqual(scaler.state, {"scale": 123.0})
        self.assertFalse(provenance["exact_resume"])
        self.args.traffic_label_provenance = {"sidecar_sha256": "corrected"}
        with self.assertRaisesRegex(ValueError, "traffic labels differ"):
            trainer.load_training_initialization(model, optimizer, scaler, self.args)

    def test_rgb_freeze_does_not_freeze_fusion(self):
        model = TinyEncoder()
        model.train()
        trainer.set_rgb_frozen(model, True)
        self.assertFalse(model.rgb_backbone.training)
        self.assertTrue(all(not p.requires_grad for p in model.rgb_backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.fusion.parameters()))
        model.train()
        trainer.set_rgb_frozen(model, False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_cli_mutual_exclusion_hash_and_legacy_defaults(self):
        base = [str(self.root), "--split-file", str(self.split)]
        defaults = trainer.parse_args(base)
        self.assertIsNone(defaults.warm_start)
        self.assertIsNone(defaults.traffic_label_sidecar)
        self.assertEqual(defaults.feature_checkpoint, "best")
        for extra in (["--warm-start", str(self.source)],
                      ["--resume", str(self.source), "--warm-start", str(self.source)],
                      ["--warm-start-sha256", "0" * 64]):
            with self.subTest(extra=extra), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    trainer.parse_args(base + extra)
        parsed = trainer.parse_args(base + ["--warm-start", str(self.source),
                                    "--warm-start-sha256", self.args.warm_start_sha256,
                                    "--feature-checkpoint", "last"])
        self.assertEqual(parsed.feature_checkpoint, "last")

    def test_sidecar_is_forwarded_to_dataset(self):
        args = trainer.parse_args([str(self.root), "--split-file", str(self.split),
                                   "--traffic-label-sidecar", str(self.root / "labels.json")])
        dataset = object()
        with mock.patch.object(trainer, "create_carla_dataset", return_value=dataset) as create:
            with mock.patch.object(trainer, "create_carla_loader", return_value="loader"):
                result, loader = trainer.make_loader(args, "train", True)
        self.assertIs(result, dataset)
        self.assertEqual(loader, "loader")
        self.assertEqual(create.call_args.kwargs["traffic_label_sidecar"], str(args.traffic_label_sidecar))

    def test_source_provenance_covers_runner_labels_and_model(self):
        provenance = trainer.implementation_provenance()
        for relative in ("train_morai_pretrain.py", "morai_data.py", "morai_traffic_labels.py",
                         "timm/data/carla_dataset.py", "timm/models/memfuser.py"):
            self.assertEqual(provenance[relative], trainer.sha256_file(ROOT / relative))


if __name__ == "__main__":
    unittest.main()
