"""Regression coverage for the extra-frame LiDAR delay in live MORAI input."""
from collections import Counter
import struct
from unittest.mock import patch

import numpy as np
import unittest

from inference.sensors.morai_udp import LidarReceiver, _Receiver, resolve_lidar_hour_stamp


def receiver(policy='morai_frame_timestamp'):
    with patch.object(_Receiver, '__init__', lambda self,*a,**kw: None):
        r = LidarReceiver(0, frame_policy=policy)
    r.audit_counts = Counter()
    return r


def packet(index, stamp=1_000_000):
    blocks = np.zeros(12, dtype=LidarReceiver.PACKET_DTYPE)
    blocks['flag'] = 0xEEFF
    blocks['azimuth'] = np.arange(index*12, index*12+12)*40
    blocks['returns']['distance'] = 5000
    blocks['returns']['reflectivity'] = 35
    return blocks.tobytes() + struct.pack('<I', stamp) + b'\x37\x22'


def full_frame(r, stamp=1_000_000):
    output = None
    for index in range(75):
        sample = r._consume(packet(index, stamp))
        if index < 74:
            assert sample is None
        else:
            output = sample
    return output


def check_incomplete_frame(fault):
    r = receiver()
    indices = list(range(75))
    if fault == 'missing': indices.remove(30)
    if fault == 'duplicate': indices.insert(30, 29)
    if fault == 'reordered': indices[29:31] = [30,29]
    if fault == 'partial_start': indices=indices[1:]
    for i in indices:
        payload = packet(i)
        if fault == 'corrupt' and i == 30:
            payload = b'\0\0' + payload[2:]
        assert r._consume(payload) is None
    assert full_frame(r, 1_100_000) is not None


class LidarFrameTests(unittest.TestCase):
    def test_emits_at_last_packet_without_waiting_for_next_scan_and_preserves_cloud(self):
        new, old = receiver(), receiver('azimuth_wrap')
        fixed = full_frame(new)
        assert fixed.shape == (75*12*32, 4)
        assert full_frame(old) is None
        delayed = old._consume(packet(0, 1_100_000))
        np.testing.assert_array_equal(fixed, delayed)
        assert new._completed_metadata['source_stamp_us_within_hour'] == 1_000_000
        assert new._completed_metadata['frame_packets'] == 75
        assert new._consume(packet(0, 1_100_000)) is None


    def test_frame_clock_change_never_joins_clouds_and_late_old_packets_are_ignored(self):
        r=receiver()
        for i in range(20): assert r._consume(packet(i)) is None
        assert r._consume(packet(0,1_100_000)) is None
        assert r._consume(packet(20,1_000_000)) is None
        for i in range(1,75): result=r._consume(packet(i,1_100_000))
        assert result is not None
        assert r.audit_counts['incomplete_lidar_frames']==1
        assert r.audit_counts['old_lidar_frame_packets']==1


    def test_duplicate_finished_frame_cannot_be_emitted_twice(self):
        r=receiver(); assert full_frame(r) is not None
        for i in range(75): assert r._consume(packet(i)) is None
        assert full_frame(r,1_100_000) is not None


    def test_hour_rollover_and_source_clock_reference(self):
        r=receiver();assert full_frame(r,3_599_950_000) is not None
        assert full_frame(r,50_000) is not None
        hour=3_600_000_000_000
        assert resolve_lidar_hour_stamp(3_599_950_000,5*hour+10_000_000)==5*hour-50_000_000
        assert resolve_lidar_hour_stamp(50_000,5*hour-10_000_000)==5*hour+50_000_000
        with unittest.TestCase().assertRaises(ValueError):resolve_lidar_hour_stamp(10_000_000,5*hour)
        with unittest.TestCase().assertRaises(ValueError):resolve_lidar_hour_stamp(0,0)

    def test_incomplete_frames_recover(self):
        for fault in ("missing", "duplicate", "reordered", "corrupt", "partial_start"):
            with self.subTest(fault=fault):
                check_incomplete_frame(fault)


if __name__ == "__main__":
    unittest.main()
