"""Causal replay windows and read-only review conventions."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.dataset import ReplayDataset, child, local_trajectory
from inference.results import trajectory_metrics


def replay():
    frames = [dict(index=i, sensor_source_ns=(i + 1) * 100_000_000, zone_id=0) for i in range(30)]
    data = object.__new__(ReplayDataset)
    data.length, data.interval = 11, 2
    data.paths = SimpleNamespace(max_gap_ns=150_000_000, eligible=lambda identity, times: True)
    data.visual = {('drive', f['sensor_source_ns']): {} for f in frames}
    data.episode = lambda identity: {'document': {'frames': frames}}
    return data, frames


def test_causal_window_and_warmup():
    data, _ = replay()
    assert data.window('drive', 19) == (None, 'insufficient_history')
    window, reason = data.window('drive', 22)
    assert reason is None
    assert [f['index'] for f in window] == list(range(2, 23, 2))


def test_missing_intermediate_observation_is_not_hidden_by_subsampling():
    data, frames = replay()
    del data.visual[('drive', frames[1]['sensor_source_ns'])]
    assert data.window('drive', 20) == (None, 'missing_visual_cache')


def test_gap_in_source_frame_ids_is_not_bridged():
    data, frames = replay()
    frames[4]['index'] += 1
    assert data.window('drive', 20) == (None, 'source_frame_gap')


def test_no_future_targets_required_to_construct_input_window():
    data, _ = replay()
    data.targets = {}
    window, reason = data.window('drive', 20)
    assert reason is None and len(window) == 11


def test_path_invalid_history_blocks_joint_but_not_situation_window():
    data, _ = replay()
    data.paths.eligible = lambda identity, times: False
    assert data.window('drive', 20) == (None, 'invalid_path_history')
    window, reason = data.window('drive', 20, length=11, interval=1, require_path=False)
    assert reason is None and [f['index'] for f in window] == list(range(10, 21))


@pytest.mark.parametrize('name', ['../escape.npy', '/absolute.npy', 'C:/absolute.npy', r'..\escape.npy'])
def test_asset_path_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError):
        child(tmp_path, name)


def test_plot_coordinates_and_metrics_are_in_metres():
    assert np.allclose(local_trajectory([[10, 0], [0, 2]], dict(x=0, y=0, yaw_rad=0)),
                       [[0, -10], [2, 0]])
    assert trajectory_metrics(np.zeros((5, 2)), np.tile([3, 4], (5, 1))) == dict(ade_m=5., fde_m=5.)
    assert trajectory_metrics(np.zeros((5, 2)), None) == dict(ade_m=None, fde_m=None)
