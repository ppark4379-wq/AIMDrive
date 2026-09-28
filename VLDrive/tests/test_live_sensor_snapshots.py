"""Raw sensor selection must be cheap, time-honest and stable while UDP runs."""
import io
from types import MethodType

import numpy as np
import pytest
import torch
from PIL import Image

from inference.sensors.live_frame import CAMERA_KEYS, LiveFrameSource


class Receiver:
    def __init__(self, payload, sequence=1, stamp=10., age=.01):
        self.payload, self.sequence, self.stamp, self.age = payload, sequence, stamp, age

    def snapshot(self):
        return self.payload, self.age, self.sequence, self.stamp


def jpeg(value):
    stream = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(stream, format='JPEG')
    return stream.getvalue()


@pytest.fixture
def source():
    instance = LiveFrameSource.__new__(LiveFrameSource)
    instance.adapter = object()
    instance.frames = 0
    instance.cameras = {key: Receiver(jpeg(40 + i * 20)) for i, key in enumerate(CAMERA_KEYS)}
    instance.lidar = Receiver(np.array([[5., 2., 0., 255.]], dtype=np.float32), sequence=7)
    instance.status = Receiver(dict(speed=5., yaw_degrees=3., sim_time=2.))
    instance.gps = Receiver(dict(longitude=126., latitude=37.))
    instance.prepared = []

    def prepare(self, images, sweep, *, device='cpu', rng_seed=0):
        self.prepared.append(dict(images=images, sweep=sweep, seed=rng_seed))
        return dict(num_points=torch.tensor([len(sweep)]), velocity=torch.zeros(1, 1))

    instance.prepare_encoder_frame = MethodType(prepare, instance)
    return instance


@pytest.mark.parametrize('failure', ['missing', 'stale', 'skew', 'no_stamp'])
def test_rejected_raw_snapshot_never_decodes_or_preprocesses(source, monkeypatch, failure):
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected sensor data reached JPEG decoding')

    monkeypatch.setattr(Image, 'open', forbidden)
    if failure == 'missing':
        source.lidar.payload = None
    elif failure == 'stale':
        source.status.stamp = 9.
    elif failure == 'skew':
        source.cameras['rgb_right'].stamp = 9.8
    else:
        source.cameras['rgb_right'].stamp = None
    raw, meta = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.01)
    assert raw is None and meta['missing']
    assert not source.prepared and source.frames == 0


def test_selected_snapshot_does_not_read_newer_receivers_when_prepared(source):
    raw, selected = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.01)
    assert not source.prepared
    assert raw['meta'] is selected
    original_front = raw['camera_bytes']['rgb_front']
    original_sweep = raw['sweep']
    source.cameras['rgb_front'] = Receiver(jpeg(220), sequence=2, stamp=10.1)
    source.lidar = Receiver(np.zeros((2, 4), np.float32), sequence=8, stamp=10.1)
    source.status.payload['speed'] = 25.
    source.gps.payload['longitude'] = 127.

    inputs, meta = source.prepare_snapshot(raw)
    assert inputs['velocity'].item() == 0
    assert meta['front_jpeg'] is original_front
    assert source.prepared[-1]['sweep'] is original_sweep
    assert source.prepared[-1]['seed'] == 6
    assert meta['sample_ids']['rgb_front'] == 1 and meta['sample_ids']['lidar'] == 7
    assert meta['received_at']['rgb_front'] == 10.
    assert meta['status_snapshot'][0]['speed'] == 5.
    assert meta['gps_snapshot'][0]['longitude'] == 126.
    assert 'lidar_points' not in selected  # Worker metadata does not modify queued raw input.
    assert meta is not selected and meta['sample_ids'] is not selected['sample_ids']


def test_reused_camera_keeps_old_timestamp_and_skips_only_identical_sample_decode(source, monkeypatch):
    original_open = Image.open
    decoded = []

    def counted(*args, **kwargs):
        decoded.append(True)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(Image, 'open', counted)
    first, _ = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.01)
    source.prepare_snapshot(first)
    assert len(decoded) == 4
    source.lidar.sequence += 1
    source.lidar.stamp = 10.1
    source.status.sequence += 1
    source.status.stamp = 10.1
    second, meta = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.11)
    source.prepare_snapshot(second)
    assert len(decoded) == 4
    assert meta['received_at']['rgb_front'] == 10.
    assert meta['received_at']['lidar'] == 10.1
    assert meta['ages']['rgb_front'] == pytest.approx(.11)
    assert meta['sample_ids']['rgb_front'] == first['meta']['sample_ids']['rgb_front']

    # Even identical bytes with a new sample identity must not reuse an old ID.
    source.cameras['rgb_front'].sequence += 1
    source.cameras['rgb_front'].stamp = 10.1
    third, _ = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.11)
    source.prepare_snapshot(third)
    assert len(decoded) == 5
    assert len(source._decoded_camera_cache) == 4


def test_old_raw_snapshot_age_is_not_refreshed_by_another_sensor(source):
    source.lidar.stamp = source.status.stamp = 10.4
    raw, meta = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.41)
    assert raw is None
    assert set(CAMERA_KEYS).issubset(meta['missing'])
    assert meta['received_at']['rgb_front'] == 10.
    assert meta['ages']['rgb_front'] == pytest.approx(.41)


def test_gps_has_separate_snapshot_and_navigation_owns_its_validation(source):
    source.gps = Receiver(None, sequence=0, stamp=None, age=float('inf'))
    raw, meta = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.01)
    assert raw is not None
    assert meta['gps_snapshot'] == (None, float('inf'), 0, None)
    assert set(meta['sample_ids']) == set(CAMERA_KEYS) | {'lidar', 'vehicle_status'}


def test_bad_jpeg_does_not_mutate_raw_metadata_or_run_lidar_preprocessing(source):
    source.cameras['rgb_right'].payload = b'not-a-jpeg'
    raw, selected = source.snapshot_raw(max_age=.3, max_skew=.15, now=10.01)
    inputs, meta = source.prepare_snapshot(raw)
    assert inputs is None and meta['missing'] == ['rgb_right(undecodable)']
    assert selected['missing'] == [] and not source.prepared
