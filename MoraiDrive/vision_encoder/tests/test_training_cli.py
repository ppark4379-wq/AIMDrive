"""Preflight checks before the CUDA training loop starts."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

READY = all(importlib.util.find_spec(name) for name in ("torch", "torch_scatter", "tensorboardX"))

@unittest.skipUnless(READY, "Install training dependencies")
class TrainingCliTest(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from train_pretrain import _parse_args
        self.parse = _parse_args
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "dataset_index.txt").write_text("data/train/ 3\n")
        (self.root / "dataset_index_test.txt").write_text("data/val/ 3\n")
        self.argv = ["train_pretrain.py", str(self.root), "--dataset", "morai",
                     "--model", "memfuser_baseline_e1d3", "--multi-view", "--with-lidar"]

    def test_index_selection_and_launcher_rank_alias(self):
        with patch.object(sys, "argv", self.argv + ["--local_rank", "1"]):
            args, _ = self.parse()
        self.assertEqual((args.train_routes, args.val_routes), (["train"], ["val"]))
        self.assertEqual(args.local_rank, 1)

    def test_overlapping_split_stops_before_model_initialization(self):
        (self.root / "dataset_index_test.txt").write_text("data/train/ 3\n")
        with patch.object(sys, "argv", self.argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.parse()
        self.assertEqual(error.exception.code, 2)

    def test_missing_required_sensor_stops_before_model_initialization(self):
        with patch.object(sys, "argv", self.argv[:-1]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.parse()
        self.assertEqual(error.exception.code, 2)
