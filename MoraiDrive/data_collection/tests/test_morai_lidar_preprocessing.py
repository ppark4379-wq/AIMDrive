"""LiDAR XYZ 범위 제한이 intensity를 손상하지 않는지 검증한다."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_dataset_module():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "MoraiDrive/vision_encoder/timm/data/morai_utils.py"
    spec = importlib.util.spec_from_file_location("morai_utils_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def test_xyz_clipping_preserves_intensity():
    module = _load_dataset_module()
    points = np.array([
        [2.0, 3.0, 4.0, 114.0],
        [101.0, 5.0, 6.0, 150.0],
    ], dtype=np.float32)
    processed, count = module.lidar_to_raw_features(points)
    rows = processed[:count]
    assert count == 2
    assert sorted(rows[:, 3].tolist()) == [114.0, 150.0]
    assert rows[np.argmax(rows[:, 3]), 0] == 0.0
