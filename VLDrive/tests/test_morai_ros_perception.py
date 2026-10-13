from types import SimpleNamespace as NS
import unittest
import numpy as np

from inference.sensors.morai_ros_perception import StampedPerceptionBuffer, pointcloud_xyzi


def pair_buffer():
    buffer = StampedPerceptionBuffer()
    for role in buffer.rows:
        value = {'speed': 2., 'x': 0.} if role == 'ego' else role
        buffer.push(role, 1_000_000_000, value, received_ns=2_000_000_000)
    return buffer


class StampedPairTests(unittest.TestCase):
    def test_paired_samples_and_duplicate_are_distinct_from_missing(self):
        buffer = pair_buffer()
        selected, meta = buffer.paired(now_ns=2_100_000_000)
        self.assertTrue(meta['ready'])
        self.assertEqual(meta['matched_sensor_skew_ns'], 0)
        self.assertEqual(selected['ego']['value']['speed'], 2.)
        selected, meta = buffer.paired(now_ns=2_110_000_000)
        self.assertIsNone(selected)
        self.assertTrue(meta['waiting_for_new_frame'])
        self.assertTrue(meta['ready'])

    def test_sensor_age_cannot_be_hidden_by_new_ego(self):
        buffer = pair_buffer()
        buffer.push('ego', 1_010_000_000, {'speed': 3.}, received_ns=3_000_000_000)
        selected, meta = buffer.paired(now_ns=3_000_000_000)
        self.assertIsNone(selected)
        self.assertEqual(meta['invalid_reason'], 'stale_sensor_receipt')

    def test_interleaved_new_camera_frame_keeps_recent_complete_pair(self):
        buffer = pair_buffer()
        buffer.paired(now_ns=2_000_000_000)
        buffer.push('rgb_front', 1_100_000_000, 'new_front', received_ns=2_100_000_000)
        selected, meta = buffer.paired(now_ns=2_110_000_000)
        self.assertIsNone(selected)
        self.assertTrue(meta['ready'])
        self.assertTrue(meta['waiting_for_new_frame'])
        self.assertEqual(meta['stamp_ns'], 1_000_000_000)
        for role in tuple(buffer.rows)[1:]:
            value = {'speed': 3., 'x': 0.} if role == 'ego' else 'new_' + role
            buffer.push(role, 1_100_000_000, value, received_ns=2_110_000_000)
        selected, meta = buffer.paired(now_ns=2_120_000_000)
        self.assertTrue(meta['ready'])
        self.assertEqual(meta['stamp_ns'], 1_100_000_000)
        self.assertEqual(selected['rgb_front']['value'], 'new_front')

    def test_incomplete_next_frame_cannot_extend_old_pair_receipt_age(self):
        buffer = pair_buffer()
        buffer.paired(now_ns=2_000_000_000)
        buffer.push('rgb_front', 1_100_000_000, 'new_front', received_ns=2_360_000_000)
        selected, meta = buffer.paired(now_ns=2_360_000_000)
        self.assertIsNone(selected)
        self.assertFalse(meta['ready'])

    def test_future_ego_speed_is_never_selected(self):
        buffer = pair_buffer()
        buffer.push('ego', 1_001_000_000, {'speed': 100.}, received_ns=2_010_000_000)
        selected, meta = buffer.paired(now_ns=2_020_000_000)
        self.assertTrue(meta['ready'])
        self.assertEqual(selected['ego']['value']['speed'], 2.)

    def test_conflicting_duplicate_blocks_the_frame(self):
        buffer = pair_buffer()
        buffer.push('ego', 1_000_000_000, {'speed': 50.}, received_ns=2_010_000_000)
        selected, meta = buffer.paired(now_ns=2_020_000_000)
        self.assertIsNone(selected)
        self.assertEqual(meta['invalid_reason'], 'ego_conflicting_duplicate')

    def test_reset_clears_all_old_scene_samples(self):
        buffer = pair_buffer()
        buffer.push('ego', 1, {'speed': 0.}, received_ns=2_010_000_000)
        selected, meta = buffer.paired(now_ns=2_020_000_000)
        self.assertIsNone(selected)
        self.assertTrue(meta['invalid_reason'].startswith('missing:'))

    def test_missing_and_skewed_sensor_block_input(self):
        buffer = StampedPerceptionBuffer()
        self.assertFalse(buffer.paired(now_ns=2_000_000_000)[1]['ready'])
        buffer = pair_buffer()
        buffer.rows['lidar'].clear()
        buffer.push('lidar', 1_100_000_000, 'lidar', received_ns=2_000_000_000)
        self.assertEqual(buffer.paired(now_ns=2_100_000_000)[1]['invalid_reason'],
                         'unpaired_source_time:lidar')


class CloudTests(unittest.TestCase):
    def cloud(self, big=False):
        prefix = '>' if big else '<'
        dtype = np.dtype(dict(names=['x','y','z','intensity'],
            formats=[prefix+'f4']*4, offsets=[0,4,8,16], itemsize=24))
        raw = bytearray(64)
        points = np.ndarray((2,1), dtype=dtype, buffer=raw, strides=(32,24))
        for name, values in [('x',[1.,2.]),('y',[3.,4.]),('z',[5.,6.]),('intensity',[100.,200.])]:
            points[name][:,0] = values
        fields = [NS(name=name, datatype=7, offset=offset, count=1)
                  for name,offset in [('x',0),('y',4),('z',8),('intensity',16)]]
        return NS(width=1,height=2,point_step=24,row_step=32,fields=fields,
                  data=bytes(raw),is_bigendian=big)

    def test_cloud_padding_row_stride_and_endianness(self):
        for big in (False,True):
            np.testing.assert_allclose(pointcloud_xyzi(self.cloud(big)),
                [[1.,3.,5.,100.],[2.,4.,6.,200.]])

    def test_cloud_missing_intensity_and_truncated_payload_are_rejected(self):
        cloud = self.cloud(); cloud.fields = cloud.fields[:3]
        with self.assertRaises(ValueError): pointcloud_xyzi(cloud)
        cloud = self.cloud(); cloud.data = cloud.data[:-1]
        with self.assertRaises(ValueError): pointcloud_xyzi(cloud)


if __name__ == '__main__': unittest.main()
