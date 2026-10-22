"""Obstacle-only UDP input must not depend on waypoint navigation evidence."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from data.src.loaders.morai_encoder_contract import sensor_record_encoder_contract
from data.src.loaders.morai_transfer_adapter import MoraiVisionAdapterConfig
from inference.realtime_ve_objects import MAP_BASIS
from inference.sensors.live_frame import CAMERA_KEYS, LiveFrameSource
from inference.ve_objects_udp import RosEgoStatusReceiver, REQUIRED_STREAMS
from types import SimpleNamespace as NS
import threading
import time


class Receiver:
    def __init__(self, port, **kwargs):
        self.port, self.value, self.age = port, None, 0.
        self.samples = 1

    def latest(self):
        return self.value, self.age

    def latest_snapshot(self):
        return self.value, self.age, dict(sample_id=self.samples,
            received_ns=time.monotonic_ns()-int(self.age*1e9))

    def close(self):
        pass


class UdpPerceptionTests(unittest.TestCase):
    def test_camera_worker_closes_if_lidar_initialization_fails(self):
        with patch('inference.sensors.camera_process.CameraProcessGroup') as group, \
             patch('inference.sensors.live_frame.LidarReceiver', side_effect=OSError('occupied')):
            group.return_value.cameras = {}
            with self.assertRaisesRegex(OSError, 'occupied'):
                LiveFrameSource(self.ports, encoder_contract=self.contract, lidar_frame='base_flu',
                    velocity_source='vehicle_status', orientation_frame='true_enu',
                    perception_only=True, camera_receiver_mode='process')
            group.return_value.close.assert_called_once()

    def setUp(self):
        torch.set_num_threads(2)
        self.contract = sensor_record_encoder_contract(
            [MoraiVisionAdapterConfig(use_previous_lidar_sweep=False,
                                      lidar_intensity_policy='reflectance_uint8_unit')],
            'ego_topic', map_basis=MAP_BASIS)
        self.ports = dict(zip(CAMERA_KEYS, (10501, 10502, 10503, 10504)),
                          lidar=2368, vehicle_status=10909, gps=10506)

    def source(self, perception_only=True):
        with ExitStack() as stack:
            for name in ('CameraReceiver', 'LidarReceiver', 'VehicleStatusReceiver', 'GpsReceiver'):
                stack.enter_context(patch('inference.sensors.live_frame.' + name, Receiver))
            return LiveFrameSource(self.ports, route_csv='/unused/changed-route.txt',
                encoder_contract=self.contract, lidar_frame='base_flu',
                velocity_source='vehicle_status', orientation_frame='true_enu',
                perception_only=perception_only)

    def populate(self, source):
        ok, image = cv2.imencode('.png', np.zeros((16, 16, 3), np.uint8))
        self.assertTrue(ok)
        for receiver in source.cameras.values():
            receiver.value = image.tobytes()
        source.lidar.value = np.array([[5., 1., .5, 200.]], np.float32)
        source.status.value = dict(speed=3., yaw=0., yaw_degrees=0.)
        source.gps.value = dict(latitude=37., longitude=126.)
        source._ego_xy = lambda fix: (302535., 4123980.)

    def test_changed_route_is_unused_for_perception_but_full_mode_still_checks_it(self):
        with patch('inference.sensors.live_frame.official_route_tracker',
                   side_effect=ValueError('route hash mismatch')) as tracker:
            source = self.source()
            tracker.assert_not_called()
            self.assertIsNone(source.route)
            with self.assertRaisesRegex(ValueError, 'route hash mismatch'):
                self.source(perception_only=False)
            self.assertEqual(tracker.call_count, 1)

    def test_perception_preserves_sensor_preprocessing_and_removes_navigation_input(self):
        source = self.source()
        self.populate(source)
        source._target_point = lambda *args: self.fail('Navigation used for obstacle-only input')
        inputs, meta = source.build('cpu', utm_offsets=(302595., 4124145.))
        self.assertEqual(meta['ego_local'], (-60., -165.))
        self.assertTrue(meta['perception_only'])
        self.assertIsNone(meta['target_point'])
        self.assertNotIn('target_point', inputs)
        self.assertEqual(inputs['velocity'].tolist(), [[3.]])
        expected = source.input_adapter.assemble(
            {key: np.zeros((16,16,3),np.uint8) for key in CAMERA_KEYS},
            source.lidar.value, velocity=3., target_point=(0., 0.))
        for key, value in inputs.items():
            torch.testing.assert_close(value[0], expected[key], rtol=0, atol=0)

    def test_missing_or_stale_gps_still_blocks_world_coordinates(self):
        source = self.source()
        self.populate(source)
        for value, age in ((None, 0.), (source.gps.value, 2.)):
            source.gps.value, source.gps.age = value, age
            inputs, meta = source.build('cpu', max_age=.5)
            self.assertIsNone(inputs)
            self.assertIn('gps (world pose unavailable)', meta['missing'])

    def test_ros_ego_status_keeps_planar_speed_and_raw_heading(self):
        receiver = RosEgoStatusReceiver.__new__(RosEgoStatusReceiver)
        receiver.lock, receiver.value, receiver.received = threading.Lock(), None, 0.
        receiver.packets = receiver.samples = 0
        self.assertIsNone(receiver.latest()[0])
        receiver.callback(NS(velocity=NS(x=3., y=4.), heading=90., position=NS(x=-60., y=-165.),
                             header=NS(stamp=NS(to_nsec=lambda: 123456789))))
        self.assertEqual(receiver.latest_snapshot()[2]['source_stamp_ns'], 123456789)
        value, age = receiver.latest()
        self.assertEqual(value['speed'], 5.)
        self.assertEqual(value['yaw_degrees'], 90.)
        self.assertAlmostEqual(value['yaw'], np.pi / 2)
        self.assertLess(age, .1)
        self.assertEqual(value['ego_local'], (-60., -165.))
        receiver.callback(NS(velocity=NS(x=float('nan'), y=0.), heading=0., position=NS(x=0., y=0.)))
        receiver.callback(NS(velocity=NS(x=0., y=0.), heading=0., position=NS(x=float('nan'), y=0.)))
        self.assertEqual(receiver.samples, 1)

    def test_ego_pose_mode_uses_expert_position_even_without_gps(self):
        from data.src.loaders.morai_map_geometry import grid_yaw
        source = self.source()
        self.populate(source)
        source.pose_source = 'ego_topic'
        source.status.value['ego_local'] = (18., -495.)
        source._ego_xy = lambda _: self.fail('GPS conversion used for an Ego-topic pose')
        for fix in (None, dict(latitude=0., longitude=0.)):
            source.gps.value = fix
            source.gps.age = 3.
            inputs, meta = source.build('cpu', utm_offsets=(302595., 4124145.), max_age=.5)
            self.assertIsNotNone(inputs)
            self.assertEqual(meta['ego_local'], (18., -495.))
            self.assertEqual(meta['pose_source'], 'ego_topic')
            self.assertTrue(np.isfinite(grid_yaw(0., (*meta['ego_local'], 0.), MAP_BASIS, source_frame='true_enu')))
        self.assertNotIn('gps', REQUIRED_STREAMS)

    def test_ego_history_never_uses_future_pose_and_rejects_large_gap(self):
        receiver = RosEgoStatusReceiver.__new__(RosEgoStatusReceiver)
        receiver.lock = threading.Lock()
        receiver.packets = receiver.samples = 0
        for stamp, x in ((1_000_000_000, 10.), (1_020_000_000, 11.), (1_040_000_000, 12.)):
            receiver.callback(NS(velocity=NS(x=3.,y=0.),heading=0.,position=NS(x=x,y=0.),
                                 header=NS(stamp=NS(to_nsec=lambda: stamp))))
        value, _, identity = receiver.at_source_stamp(1_030_000_000)
        self.assertEqual(value['ego_local'], (11.,0.))
        self.assertEqual(identity['source_stamp_ns'], 1_020_000_000)
        self.assertIsNone(receiver.at_source_stamp(999_000_000)[0])
        self.assertIsNone(receiver.at_source_stamp(1_091_000_000)[0])
        self.assertEqual(receiver.at_source_stamp(1_090_000_000)[0]['ego_local'],(12.,0.))
        receiver.callback(NS(velocity=NS(x=0.,y=0.),heading=0.,position=NS(x=0.,y=0.),
                             header=NS(stamp=NS(to_nsec=lambda: 100_000_000))))
        self.assertIsNone(receiver.at_source_stamp(1_030_000_000)[0])

    def test_ego_pose_never_falls_back_to_gps_or_accepts_stale_status(self):
        source = self.source()
        self.populate(source)
        source.pose_source = 'ego_topic'
        for position in (None, (float('nan'), 0.)):
            source.status.value['ego_local'] = position
            inputs, meta = source.build('cpu', max_age=.5)
            self.assertIsNone(inputs)
            self.assertIn('ego_topic (world pose unavailable)', meta['missing'])
        source.status.value['ego_local'] = (1., 2.)
        source.status.age = 2.
        inputs, meta = source.build('cpu', max_age=.5)
        self.assertIsNone(inputs)
        self.assertIn('vehicle_status', meta['missing'])

    def test_ros_status_override_does_not_open_vehicle_udp_socket(self):
        receiver = Receiver('/Ego_topic')
        with ExitStack() as stack:
            for name in ('CameraReceiver','LidarReceiver','GpsReceiver'):
                stack.enter_context(patch('inference.sensors.live_frame.' + name, Receiver))
            vehicle_udp = stack.enter_context(patch('inference.sensors.live_frame.VehicleStatusReceiver'))
            source = LiveFrameSource(self.ports, encoder_contract=self.contract,
                lidar_frame='base_flu', velocity_source='ego_topic',
                orientation_frame='true_enu', perception_only=True, status_receiver=receiver)
            vehicle_udp.assert_not_called()
            self.assertIs(source.status, receiver)

    def test_lidar_gate_runs_before_camera_decode_and_stale_is_not_a_wait(self):
        source = self.source()
        self.populate(source)
        with patch('cv2.imdecode', side_effect=AssertionError('Duplicate sweep decoded cameras')):
            inputs, meta = source.build('cpu', after_lidar_sample_id=1)
            self.assertIsNone(inputs)
            self.assertTrue(meta['waiting_for_new_lidar'])
            source.lidar.age = 2.
            inputs, meta = source.build('cpu', max_age=.5, after_lidar_sample_id=1)
            self.assertIsNone(inputs)
            self.assertIn('lidar', meta['missing'])
            self.assertNotIn('waiting_for_new_lidar', meta)

    def test_new_sweep_during_preprocessing_cannot_change_selected_identity(self):
        source = self.source()
        self.populate(source)
        source.lidar.samples = 10
        original = source.input_adapter.assemble
        def complete_next_sweep(*args, **kwargs):
            source.lidar.samples = 11
            source.lidar.value = np.array([[10., 2., 1., 250.]], np.float32)
            return original(*args, **kwargs)
        with patch.object(source.input_adapter,'assemble',side_effect=complete_next_sweep):
            inputs, meta = source.build('cpu', utm_offsets=(302595.,4124145.),after_lidar_sample_id=9)
        self.assertEqual(meta['selected_samples']['lidar']['sample_id'],10)
        self.assertEqual(source.lidar.samples,11)
        following, next_meta = source.build('cpu',utm_offsets=(302595.,4124145.),after_lidar_sample_id=10)
        self.assertEqual(next_meta['selected_samples']['lidar']['sample_id'],11)
        self.assertFalse(torch.equal(inputs['lidar'], following['lidar']))

    def test_lidar_source_clock_is_resolved_against_sensor_header(self):
        source = self.source()
        self.populate(source)
        reference = 5 * 3_600_000_000_000 + 10_000_000
        lidar_snapshot = source.lidar.latest_snapshot
        status_snapshot = source.status.latest_snapshot
        def stamped_lidar():
            value, age, identity = lidar_snapshot()
            identity['source_stamp_us_within_hour'] = 3_599_990_000
            return value, age, identity
        def stamped_status():
            value, age, identity = status_snapshot()
            identity['source_stamp_ns'] = reference
            return value, age, identity
        source.lidar.latest_snapshot = stamped_lidar
        source.status.latest_snapshot = stamped_status
        inputs, meta = source.build('cpu')
        self.assertIsNotNone(inputs)
        self.assertEqual(meta['selected_samples']['lidar']['source_stamp_ns'], reference-20_000_000)
        self.assertEqual(meta['source_spread_ms'], 20.)
        source.status.latest_snapshot = status_snapshot
        inputs, meta = source.build('cpu')
        self.assertIsNone(inputs)
        self.assertIn('lidar_source_clock', meta['missing'])


if __name__ == '__main__':
    unittest.main()
