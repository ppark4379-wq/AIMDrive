import argparse
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.vision_encoder.recipes import pretrain as trainer
from utils.config.vision_encoder_config import ConfigError
from data.src.loaders.loader import create_perception_loader
from data.src.loaders.transforms import create_camera_transform
from tests.stage1_config import make_config, write_config


def jitter(transform):
    return next(item for item in transform.transforms if isinstance(item, transforms.ColorJitter))


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return index


class ColorAugmentationTests(unittest.TestCase):
    def test_config_safe_default_and_explicit_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(make_config(directory).rgb_hue_jitter, 0.0)
            self.assertEqual(make_config(directory, rgb_hue_jitter=0.4).rgb_hue_jitter, 0.4)
            for value in (-0.1, 0.51, float("nan"), float("inf")):
                with self.subTest(value=value), self.assertRaises(ConfigError):
                    make_config(directory, rgb_hue_jitter=value)

    def test_stage1_command_line_cannot_override_the_config(self):
        """--config is the whole stage-1 command line."""
        parser = trainer.add_arguments(argparse.ArgumentParser())
        flags = {action.option_strings[0] for action in parser._actions
                 if action.option_strings}
        self.assertEqual(flags, {"-h", "--config"})

    def test_trainer_rejects_parameters_built_outside_a_config_file(self):
        namespace = SimpleNamespace(rgb_hue_jitter=0.4, batch_size=1, train_stride=1,
                                    validation_stride=1, workers=0)
        with self.assertRaisesRegex(TypeError, "only come from a config file"):
            trainer.make_loader(namespace, "train", True)

    def test_generic_legacy_transform_unchanged(self):
        common = dict(input_size=224, is_training=True, color_jitter=.4, scale=[.9, 1.1])
        legacy = create_camera_transform(**common)
        explicit = create_camera_transform(**common, hue_jitter=.4)
        self.assertEqual(jitter(legacy).hue, (-.4, .4))
        image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (144, 256, 3), dtype=np.uint8))
        for seed in (0, 607, 1234):
            random.seed(seed)
            torch.manual_seed(seed)
            expected = legacy(image)
            random.seed(seed)
            torch.manual_seed(seed)
            actual = explicit(image)
            self.assertTrue(torch.equal(expected, actual))

    def test_safe_transform_changes_only_hue_configuration(self):
        common = dict(input_size=224, is_training=True, color_jitter=.4, scale=[.9, 1.1])
        legacy = create_camera_transform(**common)
        safe = create_camera_transform(**common, hue_jitter=0.0)
        self.assertIsNone(jitter(safe).hue)
        for name in ("brightness", "contrast", "saturation"):
            self.assertEqual(getattr(jitter(legacy), name), getattr(jitter(safe), name))
        self.assertEqual([type(item) for item in legacy.transforms], [type(item) for item in safe.transforms])
        self.assertEqual(legacy.transforms[1].scale, safe.transforms[1].scale)

    def test_all_rgb_views_receive_safe_option(self):
        dataset = TinyDataset()
        create_perception_loader(dataset, (3, 224, 224), 1, multi_view_input_size=(3, 128, 128),
                            is_training=True, num_workers=0, persistent_workers=False, hue_jitter=0.0)
        for name in ("rgb_transform", "rgb_center_transform", "multi_view_transform"):
            self.assertIsNone(jitter(getattr(dataset, name)).hue)

    def test_generic_loader_still_defaults_to_legacy(self):
        dataset = TinyDataset()
        create_perception_loader(dataset, (3, 224, 224), 1, multi_view_input_size=(3, 128, 128),
                            is_training=True, num_workers=0, persistent_workers=False)
        for name in ("rgb_transform", "rgb_center_transform", "multi_view_transform"):
            self.assertEqual(jitter(getattr(dataset, name)).hue, (-.4, .4))

    def test_morai_loader_forwards_choice_without_changing_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_config(directory, rgb_hue_jitter=0.2)
        with mock.patch.object(trainer, "MoraiPerceptionDataset", return_value=TinyDataset()):
            with mock.patch.object(trainer, "create_perception_loader", return_value="loader") as create:
                trainer.make_loader(args, "train", True)
                self.assertEqual(create.call_args.kwargs["hue_jitter"], .2)
                trainer.make_loader(args, "validation", False)
                self.assertEqual(create.call_args.kwargs["color_jitter"], 0.0)
                self.assertEqual(create.call_args.kwargs["hue_jitter"], 0.0)

    def test_deterministic_inference_tensor_unchanged(self):
        image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (144, 256, 3), dtype=np.uint8))
        for size, scale in ((224, True), (128, False), (128, True)):
            common = dict(input_size=size, is_training=False, need_scale=scale, color_jitter=0.0)
            legacy = create_camera_transform(**common)(image)
            safe = create_camera_transform(**common, hue_jitter=0.0)(image)
            self.assertTrue(torch.equal(legacy, safe))

    def test_resume_rejects_recorded_augmentation_change_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_config(directory, resume="old.pt")
        state = trainer.RunState(implementation_sha256={}, traffic_label_provenance=None)
        checkpoint = {"split_sha256": "same", "traffic_label_provenance": None,
                      "args": {"rgb_hue_jitter": .4}}
        model = mock.Mock()
        with mock.patch.object(trainer, "sha256_file", return_value="same"):
            with mock.patch.object(torch, "load", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "hue augmentation differs"):
                    trainer.load_training_initialization(model, mock.Mock(), mock.Mock(),
                                                         args, state)
        model.load_state_dict.assert_not_called()

    def test_resume_same_recorded_augmentation_and_unknown_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_config(directory, resume="old.pt")
        state = trainer.RunState(implementation_sha256={}, traffic_label_provenance=None)
        for source_args, warning in (({"rgb_hue_jitter": 0.0}, False), ({}, True)):
            checkpoint = {"split_sha256": "same", "traffic_label_provenance": None,
                          "args": source_args, "state_dict": {}, "optimizer": {},
                          "scaler": {}, "epoch": 0}
            with self.subTest(source_args=source_args), mock.patch.object(trainer, "sha256_file", return_value="same"):
                with mock.patch.object(torch, "load", return_value=checkpoint), mock.patch("builtins.print") as log:
                    start, _, provenance = trainer.load_training_initialization(
                        mock.Mock(), mock.Mock(), mock.Mock(), args, state)
            self.assertEqual(start, 1)
            self.assertEqual(provenance["kind"], "optimizer_resume")
            self.assertEqual(log.called, warning)

    def test_safe_jitter_keeps_red_and_green_channel_order(self):
        transform = create_camera_transform(224, is_training=True, color_jitter=.4,
                                              scale=[.9, 1.1], hue_jitter=0.0)
        std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
        mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
        for color, channel in (((255, 0, 0), 0), ((0, 255, 0), 1)):
            image = Image.new("RGB", (256, 144), color)
            for seed in range(8):
                random.seed(seed)
                torch.manual_seed(seed)
                actual = transform(image) * std + mean
                self.assertEqual(int(actual.mean(dim=(1, 2)).argmax()), channel)


if __name__ == "__main__":
    unittest.main()
