"""Failure injection for durable frame completion and collector recovery."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from writer.atomic_io import atomic_path, write_json
from writer.dataset_writer import DatasetWriter
from writer.async_frame_writer import AsyncFrameWriter
from writer.image_writer import ImageWriter
from writer.lidar_writer import LidarWriter
from writer.frame_commit import COMMIT_DIR, SESSION_FILE, read_commits
from sources.native_file_collector import _recover_resume_state, _checkpoint_path

VIEWS = ("front", "left", "right", "rear")


def setup_writer(tmp_path, on_commit=None):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"session_id": "session"}) + "\n")
    writer = DatasetWriter(str(tmp_path / "dataset"), "job", "route",
                           append_timestamp=False, transactional=True)
    root = Path(writer.output_dir)
    write_json(str(root / SESSION_FILE), {
        "manifest_path": str(manifest), "runtime_session_id": "session",
    })
    images = ImageWriter(str(root), VIEWS)
    lidar = LidarWriter(str(root))
    async_writer = AsyncFrameWriter(writer, images, lidar, on_commit=on_commit)
    return writer, async_writer, manifest


def submit(writer, index=0):
    writer.submit(index, {}, {"native_sync_frame": 100 + index * 5,
                             "runtime_session_id": "session"},
                  cameras={v: np.zeros((8, 8, 3), np.uint8) for v in VIEWS},
                  lidar_points=np.zeros((1, 4), np.float32))


def recover(tmp_path, manifest):
    return _recover_resume_state(str(tmp_path / "dataset"), "job", "route", str(manifest))[0]


def test_imwrite_false_is_failure_and_does_not_replace_existing_image(tmp_path):
    writer = ImageWriter(str(tmp_path), ["front"])
    target = tmp_path / "rgb_front" / "0000.jpg"
    target.write_bytes(b"previous-complete-file")
    with patch("writer.image_writer.cv2.imwrite", return_value=False):
        with pytest.raises(OSError, match="Failed to save camera image"):
            writer.write(0, {"front": np.zeros((8, 8, 3), np.uint8)})
    assert target.read_bytes() == b"previous-complete-file"
    assert list(target.parent.glob(".pending-*")) == []


def test_atomic_write_exception_preserves_destination(tmp_path):
    target = tmp_path / "frame.json"
    target.write_text("old")
    with pytest.raises(OSError):
        with atomic_path(str(target)) as temporary:
            Path(temporary).write_text("partial")
            raise OSError("disk full")
    assert target.read_text() == "old"
    assert list(tmp_path.glob(".pending-*")) == []


@pytest.mark.parametrize("failure", ["image", "lidar", "label", "receipt"])
def test_failed_frame_never_advances_completion(tmp_path, failure):
    committed = []
    writer, background, manifest = setup_writer(tmp_path, lambda *args: committed.append(args))
    root = Path(writer.output_dir)
    targets = {
        "image": "writer.image_writer.cv2.imwrite",
        "lidar": "writer.lidar_writer.save_numpy",
        "label": "writer.dataset_writer.save_numpy",
        "receipt": "writer.frame_commit.write_json",
    }
    kwargs = {"return_value": False} if failure == "image" else {"side_effect": OSError("disk full")}
    with patch(targets[failure], **kwargs):
        try:
            submit(background)
        except RuntimeError:
            pass  # Worker may fail before submit returns.
        with pytest.raises(RuntimeError, match="Asynchronous dataset write failed"):
            background.close()
    assert writer.frame_count == 0
    assert committed == []
    assert list((root / COMMIT_DIR).glob("[0-9]*.json")) == []
    if failure in ("image", "lidar"):
        assert list((root / "measurements_full").iterdir()) == []
    state = recover(tmp_path, manifest)
    assert state["next_sample_index"] == 0
    assert state["last_consumed_native_frame"] == -1
    assert list((root / "measurements_full").iterdir()) == []


def test_stale_checkpoint_and_torn_provenance_recover_only_committed_frames(tmp_path):
    writer, background, manifest = setup_writer(tmp_path)
    root = Path(writer.output_dir)
    submit(background)
    background.close()
    assert writer.frame_count == 1
    checkpoint = _checkpoint_path(str(tmp_path / "dataset"), "job", str(manifest))
    write_json(checkpoint, {"manifest_path": str(manifest), "output_dir": str(root),
                            "next_sample_index": 99, "last_consumed_native_frame": 900})
    (root / "measurements_full" / "0001.json").write_text("partial")
    (root / "rgb_front" / "0001.jpg").write_bytes(b"partial")
    (root / COMMIT_DIR / ".pending-crash.json").write_text("{")
    (root / "capture_provenance.jsonl").write_text('{"sample_index": "0099"')
    state = recover(tmp_path, manifest)
    assert state["next_sample_index"] == 1
    assert state["last_consumed_native_frame"] == 100
    assert not (root / "measurements_full" / "0001.json").exists()
    assert list((root / ".incomplete_frames").glob("*/measurements_full/0001.json"))
    rows = [json.loads(line) for line in (root / "capture_provenance.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["sample_index"] == "0000"
    resumed = DatasetWriter(str(tmp_path / "dataset"), "job", "route",
                            resume_route_dir=str(root), transactional=True)
    again = AsyncFrameWriter(resumed, ImageWriter(str(root), VIEWS), LidarWriter(str(root)))
    submit(again, 1)
    again.close()
    assert resumed.frame_count == 2
    assert [r["native_sync_frame"] for r in read_commits(root)] == [100, 105]


@pytest.mark.parametrize("change", ["missing", "same_size_corruption"])
def test_committed_sensor_damage_blocks_resume(tmp_path, change):
    writer, background, manifest = setup_writer(tmp_path)
    submit(background)
    background.close()
    path = Path(writer.output_dir) / "rgb_front" / "0000.jpg"
    if change == "missing":
        path.unlink()
    else:
        data = bytearray(path.read_bytes())
        data[len(data)//2] ^= 1
        path.write_bytes(data)
    with pytest.raises(RuntimeError, match="missing or changed"):
        recover(tmp_path, manifest)


def test_callback_failure_after_commit_is_recoverable_without_checkpoint(tmp_path):
    def callback(*args):
        raise OSError("provenance disk error")
    writer, background, manifest = setup_writer(tmp_path, callback)
    try:
        submit(background)
    except RuntimeError:
        pass
    with pytest.raises(RuntimeError):
        background.close()
    assert writer.frame_count == 1
    assert recover(tmp_path, manifest)["next_sample_index"] == 1


def test_missing_camera_cannot_commit(tmp_path):
    writer, background, manifest = setup_writer(tmp_path)
    try:
        background.submit(0, {}, {}, cameras={}, lidar_points=np.zeros((1, 4)))
    except RuntimeError:
        pass
    with pytest.raises(RuntimeError):
        background.close()
    assert writer.frame_count == 0


def test_torn_checkpoint_is_rebuilt_from_receipts(tmp_path):
    writer, background, manifest = setup_writer(tmp_path)
    submit(background)
    background.close()
    checkpoint = Path(_checkpoint_path(str(tmp_path / "dataset"), "job", str(manifest)))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text('{"next_sample_index":')
    assert recover(tmp_path, manifest)["next_sample_index"] == 1


def test_success_callback_observes_complete_payloads(tmp_path):
    observations = []
    def callback(frame_id, full):
        records = read_commits(writer.output_dir)
        observations.append((writer.frame_count, records[-1]["sample_index"]))
    writer, background, manifest = setup_writer(tmp_path, callback)
    submit(background)
    background.close()
    assert observations == [(1, 0)]


def test_existing_legacy_output_is_not_overwritten(tmp_path):
    root = tmp_path / "dataset" / "job" / "data" / "route"
    root.mkdir(parents=True)
    sentinel = root / "keep.txt"
    sentinel.write_text("existing data")
    with pytest.raises(RuntimeError, match="already exists"):
        DatasetWriter(str(tmp_path / "dataset"), "job", "route",
                      append_timestamp=False, transactional=True)
    assert sentinel.read_text() == "existing data"


def test_native_collector_failure_checkpoint_and_retry(tmp_path, capsys):
    from types import SimpleNamespace
    import cv2
    from test_sync_native_files import _record
    from sources.native_file_collector import run_sync_native_files

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(_record(i).to_dict()) + "\n" for i in (100, 105, 110)))
    args = SimpleNamespace(
        map_dir="", route_definition="", global_path="", route_spec="",
        route_definition_key="full_loop", route_start_index=None,
        route="route", job="job", map_id="map", no_route_timestamp=True,
        global_info="", morai_sync_root="", morai_sync_base="",
        morai_camera_sync_base="", morai_lidar_sync_base="",
        writer_queue_size=4, native_file_timeout=0.2,
        native_poll_interval=0.005, native_stable_checks=1,
        capture_manifest=str(manifest), manifest_poll_interval=0.005,
        duration=1, max_samples=3, require_route_valid=False,
        route_valid_max_error=3, morai_to_lmdrive_steer_sign=1,
        control_max_steer_deg=40,
    )
    config = SimpleNamespace(append_timestamp=False, image_ext="jpg", image_quality=90)
    payload = SimpleNamespace(
        cameras={v: np.zeros((8, 8, 3), np.uint8) for v in VIEWS},
        lidar_points=np.zeros((1, 4), np.float32),
    )
    original = cv2.imwrite
    calls = 0
    def fail_second_frame(*args, **kwargs):
        nonlocal calls
        calls += 1
        return False if calls == 5 else original(*args, **kwargs)
    dataset = str(tmp_path / "dataset")
    with patch("sources.native_file_collector.discover_sync_session_root", return_value=str(tmp_path)), \
         patch("sources.native_file_collector.MoraiSyncFileSource.load", return_value=payload):
        with patch("writer.image_writer.cv2.imwrite", side_effect=fail_second_frame):
            with pytest.raises(RuntimeError, match="Asynchronous dataset write failed"):
                run_sync_native_files(args, dataset, config, None)
        assert "done saved=1" in capsys.readouterr().out
        checkpoint = json.loads(Path(_checkpoint_path(dataset, "job", str(manifest))).read_text())
        assert checkpoint["next_sample_index"] == 1
        assert checkpoint["last_consumed_native_frame"] == 100
        root = Path(checkpoint["output_dir"])
        rows = (root / "capture_provenance.jsonl").read_text().splitlines()
        assert len(rows) == 1
        assert list((root / "measurements_full").glob("*.json")) == [root / "measurements_full" / "0000.json"]
        args.max_samples = 2
        output, count = run_sync_native_files(args, dataset, config, None)
        assert output == str(root) and count == 3
        assert [r["native_sync_frame"] for r in read_commits(root)] == [100, 105, 110]
        rows = (root / "capture_provenance.jsonl").read_text().splitlines()
        assert [json.loads(row)["sample_index"] for row in rows] == ["0000", "0001", "0002"]
