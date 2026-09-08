import os
import tempfile
import time
from types import SimpleNamespace

from writer.async_frame_writer import AsyncFrameWriter
from writer.image_writer import ImageWriter


class _RecordingDatasetWriter:
    def __init__(self):
        self.frames = []

    def write_frame(self, frame_id, measurement, full_measurement, **kwargs):
        time.sleep(0.005)
        self.frames.append(frame_id)


class _RecordingImageWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame_id, cameras):
        self.frames.append(frame_id)


class _RecordingLidarWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame_id, points):
        self.frames.append(frame_id)


def test_async_writer_flushes_frames_in_order():
    dataset = _RecordingDatasetWriter()
    images = _RecordingImageWriter()
    lidar = _RecordingLidarWriter()
    writer = AsyncFrameWriter(dataset, images, lidar, queue_size=8)

    for frame_id in range(4):
        writer.submit(
            frame_id, {}, {}, cameras={"front": object()},
            lidar_points=object(),
        )
    writer.close()

    assert dataset.frames == [0, 1, 2, 3]
    assert images.frames == [0, 1, 2, 3]
    assert lidar.frames == [0, 1, 2, 3]


def test_passive_writer_backpressure_waits_instead_of_failing():
    dataset = _RecordingDatasetWriter()
    writer = AsyncFrameWriter(
        dataset,
        _RecordingImageWriter(),
        _RecordingLidarWriter(),
        queue_size=1,
        block_when_full=True,
    )
    for frame_id in range(8):
        writer.submit(frame_id, {}, {})
    writer.close()
    assert dataset.frames == list(range(8))


def test_jpeg_payload_is_written_without_reencoding():
    jpeg_payload = b"\xff\xd8already-compressed\xff\xd9"
    camera = SimpleNamespace(
        encoded=jpeg_payload,
        encoding="jpeg",
        image=None,
    )
    with tempfile.TemporaryDirectory() as route_dir:
        writer = ImageWriter(route_dir, ["front"], ext="jpg", quality=10)
        writer.write(7, {"front": camera})
        with open(
            os.path.join(route_dir, "rgb_front", "0007.jpg"), "rb"
        ) as saved:
            assert saved.read() == jpeg_payload
