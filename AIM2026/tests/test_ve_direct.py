"""Supplement visualization transport requires no ROS runtime."""
import json
from types import SimpleNamespace

from AIM2026.UDP.Detection import Box, EgoPose
from AIM2026.Visualization.ve_direct import DirectReader, write_frame
from AIM2026.Visualization.lidar_overlay import footprints


def test_direct_frame_preserves_geometry_and_unknown_speed(tmp_path):
    path = tmp_path / 'visualization.json'
    box = Box(10., 20., 45., 2., 4., None)
    pose = EgoPose(1., 2., 0., .5, 3.)
    write_frame(path, 7, SimpleNamespace(boxes=(box,), supplement_indices=(0,), inference_ms=12.), pose, None)
    reader = DirectReader(path)
    state, stale = reader.read()
    assert not stale
    assert state['supplement_boxes'] == [box.to_wire()]
    assert state['ego'] == pose.to_wire()
    write_frame(path, 8, None, None, 'missing:lidar')
    state, stale = reader.read()
    assert not state['ready'] and state['supplement_boxes'] == []
    assert state['reason'] == 'missing:lidar'


def test_old_frame_and_partial_write_remain_stale(tmp_path):
    path = tmp_path / 'visualization.json'
    path.write_text(json.dumps(dict(written_at=1., sequence=1, boxes=[])))
    reader = DirectReader(path)
    state, stale = reader.read()
    assert stale
    path.write_text('{')
    retained, stale = reader.read()
    assert retained == state and stale


def test_only_supplements_are_displayed_and_clear_on_missing_input(tmp_path):
    path = tmp_path / 'visualization.json'
    ve = Box(10., 20., 0., 2., 4., 1., (10., 20.), 100)
    lidar = Box(12., 20., 0., 2.5, 12., 2., (12., 20.), 200, lidar_track_id=7)
    pose = EgoPose(2., 20., 0., 0., 3.)
    write_frame(path, 1, SimpleNamespace(boxes=(ve, lidar), supplement_indices=(1,), inference_ms=12.), pose, None)
    state = json.loads(path.read_text())
    assert state['supplement_boxes'] == [lidar.to_wire()]
    assert 'lidar_boxes' not in state
    assert 'track_id' not in lidar.to_wire()  # Expert wire contract unchanged
    world = footprints(state, False)
    local = footprints(state, False, local=True)
    assert world[0]['center'] == (12., 20.)
    assert local[0]['center'] == (10., 0.)
    assert local[0]['corners'][0] == (16., 1.25)
    assert footprints(state, True) == []
    assert footprints(dict(ready=True, boxes=state['supplement_boxes']), False) == []
    write_frame(path, 2, None, None, 'missing:lidar')
    assert footprints(json.loads(path.read_text()), False) == []


def test_lidar_overlay_rotates_world_boxes_into_ego_frame():
    import math
    state = dict(ready=True, ego=dict(x=10., y=20., yaw_grid_rad=math.pi/2),
                 supplement_boxes=[dict(center_world=[10., 25.], heading_deg=90.,
                                   length_m=12., width_m=2., speed_mps=0., track_id=1)])
    row = footprints(state, False, local=True)[0]
    assert abs(row['center'][0]-5.) < 1e-8
    assert abs(row['center'][1]) < 1e-8
    assert abs(row['corners'][0][0]-11.) < 1e-8
    assert abs(row['corners'][0][1]-1.) < 1e-8


def test_snapshot_retries_temporary_windows_read_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from AIM2026.Visualization import ve_direct
    target = tmp_path / 'visualization.json'
    write_frame(target, 1, None, None, 'waiting')
    replace = Path.replace
    calls, sleeps = [], []

    def locked_once(source, destination):
        calls.append(source)
        if len(calls) == 1:
            raise PermissionError(13, 'sharing violation')
        return replace(source, destination)

    monkeypatch.setattr(Path, 'replace', locked_once)
    monkeypatch.setattr(ve_direct.time, 'sleep', sleeps.append)
    write_frame(target, 2, None, None, 'waiting')
    assert json.loads(target.read_text())['sequence'] == 2
    assert len(calls) == 2 and sleeps == [.002]
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_lock_preserves_last_snapshot_and_reports_error(tmp_path, monkeypatch):
    from pathlib import Path
    import pytest
    from AIM2026.Visualization import ve_direct
    target = tmp_path / 'visualization.json'
    write_frame(target, 1, None, None, 'waiting')
    sleeps = []

    def locked(source, destination):
        raise PermissionError(13, 'access denied')

    monkeypatch.setattr(Path, 'replace', locked)
    monkeypatch.setattr(ve_direct.time, 'sleep', sleeps.append)
    with pytest.raises(PermissionError):
        write_frame(target, 2, None, None, 'waiting')
    assert json.loads(target.read_text())['sequence'] == 1
    assert sleeps == [.002, .004, .006]
    assert not list(tmp_path.glob('*.tmp'))
