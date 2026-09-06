"""LiDAR XYZ 범위 제한이 intensity를 손상하지 않는지 검증한다."""
import importlib
import sys
import types
from pathlib import Path

import numpy as np


def _load_dataset_module():
    repo_root = Path(__file__).resolve().parents[3]
    timm_root = repo_root / "MoraiDrive" / "vision_encoder" / "timm"
    timm = types.ModuleType("timm")
    timm.__path__ = [str(timm_root)]
    sys.modules["timm"] = timm
    data_pkg = types.ModuleType("timm.data")
    data_pkg.__path__ = [str(timm_root / "data")]
    sys.modules["timm.data"] = data_pkg
    return importlib.import_module("timm.data.morai_dataset")


def test_xyz_clipping_preserves_intensity():
    module = _load_dataset_module()
    points = np.array([
        [2.0, 3.0, 4.0, 114.0],
        [101.0, 5.0, 6.0, 150.0],
    ], dtype=np.float32)
    processed, count = module.lidar_to_raw_features(points, rotation_deg=0)
    rows = processed[:count]
    assert count == 2
    assert sorted(rows[:, 3].tolist()) == [114.0, 150.0]
    assert rows[np.argmax(rows[:, 3]), 0] == 0.0
