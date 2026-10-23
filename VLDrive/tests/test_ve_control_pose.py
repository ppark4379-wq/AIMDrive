"""Control must get live allowed ego sensors without waiting for inference."""
import copy
import json
import math
import socket
import time
import unittest
from types import SimpleNamespace

from inference.realtime_ve_objects import MAP_BASIS
from utils.common.ve_control_pose import ControlPoseStream, decode_control_pose
from data.src.loaders.morai_map_geometry import grid_yaw


def receivers():
    status = dict(yaw=.7, signed_speed_mps=10., accel=.2, brake=0., steer=-3., gear=4,
                  ego_local=(99999., 99999.), yaw_status=-90.)
    fix = dict(longitude=126.77, latitude=37.24)
    def status_snapshot():
        stamp = time.monotonic_ns()
        return status, .001, dict(received_ns=stamp-1_000_000, imu_received_ns=stamp,
                                 source_stamp_ns=123, sample_id=stamp)
    def gps_snapshot():
        stamp = time.monotonic_ns()
        return fix, .002, dict(received_ns=stamp-2_000_000, sample_id=stamp, source_stamp_ns=0)
    return SimpleNamespace(latest_snapshot=status_snapshot), SimpleNamespace(latest_snapshot=gps_snapshot)


class ControlPoseTests(unittest.TestCase):
    def test_latest_pose_projection_matches_existing_basis_without_status_gt(self):
        status, gps = receivers()
        stream = ControlPoseStream(status, gps, MAP_BASIS, 'generation', 'digest')
        try:
            packet = stream.sample()
            decode_control_pose(packet, 'generation', 'digest', time.monotonic_ns())
            pose = packet['ego_pose']
            self.assertAlmostEqual(pose['yaw_grid_rad'], grid_yaw(.7,
                (pose['x'],pose['y'],0.), MAP_BASIS, source_frame='true_enu'))
            self.assertEqual(pose['speed_mps'], 10.)
            self.assertNotEqual((pose['x'],pose['y']), (99999.,99999.))
            self.assertAlmostEqual(pose['yaw_raw_deg'], math.degrees(.7))
        finally:
            stream.close()

    def test_wrong_identity_source_future_and_stale_samples_are_rejected(self):
        status, gps = receivers()
        stream = ControlPoseStream(status, gps, MAP_BASIS, 'generation', 'digest')
        try:
            good = stream.sample()
            for field, value in (('generation','old'),('checkpoint_sha256','other'),
                    ('ego_position_source','/Ego_topic'),('ego_status_source','GT'),
                    ('sequence',0),('ready',False),('sent_ns',time.monotonic_ns()+1_000_000_000),
                    ('sensor_received_ns',time.monotonic_ns()-1_000_000_000)):
                packet = copy.deepcopy(good)
                packet[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    decode_control_pose(packet,'generation','digest',time.monotonic_ns())
            packet = copy.deepcopy(good)
            packet['selected_samples']['vehicle_status']['imu_received_ns'] -= 1_000_000_000
            with self.assertRaises(ValueError):
                decode_control_pose(packet,'generation','digest',time.monotonic_ns())
            packet = copy.deepcopy(good)
            packet['ego_pose']['yaw_grid_rad'] = float('nan')
            with self.assertRaises(ValueError):
                decode_control_pose(packet,'generation','digest',time.monotonic_ns())
            for stale in (status,gps):
                previous = stale.latest_snapshot
                stale.latest_snapshot = lambda: (previous()[0],.3,previous()[2])
                self.assertFalse(stream.sample()['ready'])
                stale.latest_snapshot = previous
        finally:
            stream.close()

    def test_pose_stream_continues_while_inference_caller_is_blocked(self):
        status, gps = receivers()
        receiver = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1',0))
        receiver.settimeout(.5)
        stream = ControlPoseStream(status,gps,MAP_BASIS,'generation','digest',
                                   target=receiver.getsockname()).start()
        try:
            first = json.loads(receiver.recv(8000))
            # Stand in for the main thread blocked in a slow model forward.
            time.sleep(.16)
            rows=[]
            receiver.setblocking(False)
            while True:
                try:rows.append(json.loads(receiver.recv(8000)))
                except BlockingIOError:break
            self.assertGreaterEqual(len(rows),3)
            self.assertGreater(rows[-1]['sensor_received_ns'],first['sensor_received_ns'])
            decode_control_pose(rows[-1],'generation','digest',time.monotonic_ns())
            self.assertIsNone(stream.error)
        finally:
            stream.close()
            receiver.close()


if __name__ == '__main__':
    unittest.main()
