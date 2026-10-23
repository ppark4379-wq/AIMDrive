"""Competition LiDAR frames retain completeness checks across azimuth zero."""
from collections import Counter
import struct
import unittest
from unittest.mock import patch

import numpy as np
from inference.sensors.morai_udp import LidarReceiver, _Receiver


def receiver():
    with patch.object(_Receiver, '__init__', lambda *_a, **_k: None):
        source = LidarReceiver(0, frame_policy='morai_cyclic_frame_timestamp')
    source.audit_counts = Counter()
    return source


def packet(index, stamp=1_000_000):
    blocks = np.zeros(12, dtype=LidarReceiver.PACKET_DTYPE)
    blocks['flag'] = 0xEEFF
    indices = np.arange(index*12, index*12+12)
    blocks['azimuth'] = (750+indices*40-(indices % 17 == 0)) % 36000
    blocks['returns']['distance'], blocks['returns']['reflectivity'] = 5000, 35
    return blocks.tobytes()+struct.pack('<I', stamp)+b'\x37\x22'


class CyclicFrameTests(unittest.TestCase):
    def test_offset_start_quantization_and_internal_wrap_complete_without_extra_delay(self):
        source = receiver()
        for i in range(75):
            value = source._consume(packet(i))
            if i < 74:
                self.assertIsNone(value)
        self.assertEqual(value.shape, (900*32, 4))
        self.assertEqual(source._completed_metadata['frame_packets'], 75)
        self.assertTrue(source._completed_metadata['angular_coverage_complete'])
        self.assertIsNone(source._consume(packet(74)))

    def test_missing_duplicate_reordered_and_partial_frames_never_complete(self):
        for fault in ('missing', 'duplicate', 'reorder', 'partial', 'flag'):
            source = receiver()
            indices = list(range(75))
            if fault == 'missing': indices.remove(30)
            if fault == 'duplicate': indices.insert(30,29)
            if fault == 'reorder': indices[29:31] = [30,29]
            if fault == 'partial': indices = indices[1:]
            for i in indices:
                data = packet(i)
                if fault == 'flag' and i == 30: data = b'\0\0'+data[2:]
                self.assertIsNone(source._consume(data), fault)
            for i in range(75): value = source._consume(packet(i,1_100_000))
            self.assertIsNotNone(value, fault)

    def test_new_frame_never_includes_delayed_packets_from_old_timestamp(self):
        source = receiver()
        source._consume(packet(0))
        source._consume(packet(0,1_100_000))
        source._consume(packet(1))
        for i in range(1,75): value = source._consume(packet(i,1_100_000))
        self.assertIsNotNone(value)
        self.assertEqual(source.audit_counts['old_lidar_frame_packets'], 1)


if __name__ == '__main__':
    unittest.main()
