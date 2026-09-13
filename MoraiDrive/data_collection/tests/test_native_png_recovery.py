import os
import struct
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from sources.morai_sync_file_source import (
    IncompleteNativeFrame, MANIFEST_TO_VIEW, MoraiSyncFileSource,
)
from writer.async_frame_writer import AsyncFrameWriter
from writer.image_writer import ImageWriter


def fixture_files(root):
    # Unequal red/blue values detect accidental BGR/RGB swaps.
    bgr = np.full((8, 8, 3), (20, 80, 180), dtype=np.uint8)
    ok, png = cv2.imencode('.png', bgr)
    assert ok
    encoded = png.tobytes()
    expected = {}
    for name in MANIFEST_TO_VIEW:
        expected[name] = name + '.png'
        with open(os.path.join(root, expected[name]), 'wb') as stream:
            stream.write(encoded)
    expected['lidar'] = 'lidar.bin'
    np.array([[1, 2, 3, 0.5]], dtype=np.float32).tofile(
        os.path.join(root, 'lidar.bin'))
    return {'native_sync_frame': 136, 'expected_sensor_files': expected}, encoded, bgr


def corrupt_idat_crc(encoded):
    result = bytearray(encoded)
    offset = 8
    while offset < len(result):
        length = struct.unpack_from('>I', result, offset)[0]
        if result[offset + 4:offset + 8] == b'IDAT':
            result[offset + 8 + length] ^= 1
            return bytes(result)
        offset += 12 + length
    raise AssertionError('PNG lacks IDAT')


def test_validated_native_image_is_not_decoded_again_by_async_writer():
    with tempfile.TemporaryDirectory() as root:
        record, _, bgr = fixture_files(root)
        source = MoraiSyncFileSource(root, stable_checks=1)
        payload = source.load(record)
        np.testing.assert_array_equal(payload.cameras['front'].image, bgr[:, :, ::-1])
        from unittest.mock import Mock
        writer = AsyncFrameWriter(Mock(), ImageWriter(root, MANIFEST_TO_VIEW.values(), ext='png'), Mock())
        with patch('cv2.imdecode', side_effect=AssertionError('second decode')):
            writer.submit(0, {}, {}, cameras=payload.cameras)
            writer.close()
        np.testing.assert_array_equal(cv2.imread(os.path.join(root, 'rgb_front', '0000.png')), bgr)


def test_crc_error_is_retried_even_when_file_size_does_not_change():
    with tempfile.TemporaryDirectory() as root:
        record, encoded, _ = fixture_files(root)
        front = os.path.join(root, record['expected_sensor_files']['camera_front'])
        with open(front, 'wb') as stream:
            stream.write(corrupt_idat_crc(encoded))

        def complete_file(_):
            with open(front, 'wb') as stream:
                stream.write(encoded)

        with patch('sources.morai_sync_file_source.time.sleep', side_effect=complete_file) as sleep:
            payload = MoraiSyncFileSource(root, timeout_sec=1, stable_checks=1).load(record)
        assert sleep.called
        assert payload.cameras['front'].image is not None


def test_persistent_crc_error_becomes_incomplete_and_next_frame_loads():
    with tempfile.TemporaryDirectory() as root:
        record, encoded, _ = fixture_files(root)
        front = os.path.join(root, record['expected_sensor_files']['camera_front'])
        with open(front, 'wb') as stream:
            stream.write(corrupt_idat_crc(encoded))
        source = MoraiSyncFileSource(root, timeout_sec=0, stable_checks=1)
        with pytest.raises(IncompleteNativeFrame, match='cannot decode') as caught:
            source.load(record)
        assert caught.value.native_sync_frame == 136
        with open(front, 'wb') as stream:
            stream.write(encoded)
        assert source.load(record).cameras['front'].image is not None


def test_opencv_exception_uses_incomplete_frame_path():
    with tempfile.TemporaryDirectory() as root:
        record, _, _ = fixture_files(root)
        with patch('cv2.imdecode', side_effect=cv2.error('invalid PNG')):
            with pytest.raises(IncompleteNativeFrame, match='invalid PNG'):
                MoraiSyncFileSource(root, timeout_sec=0, stable_checks=1).load(record)
