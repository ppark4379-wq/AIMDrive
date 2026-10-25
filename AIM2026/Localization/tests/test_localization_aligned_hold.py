import math
import time
import unittest
from unittest.mock import patch

from AIM2026.Localization.gps_aligned_hold import AlignedGpsBlackoutHold, MotionTrack, aligned_forward, forward_vector
from AIM2026.Localization.gps_blackout_hold import SectionZone
from AIM2026.Localization.tests.support import BASIS, FakeGps, FakeStatus


def imu(yaw=0., pitch=0., rate=0.):
    return dict(yaw=yaw, quaternion=(-math.sin(yaw/2)*math.sin(pitch/2),
        math.cos(yaw/2)*math.sin(pitch/2), math.sin(yaw/2)*math.cos(pitch/2),
        math.cos(yaw/2)*math.cos(pitch/2)), angular_velocity=(0., 0., rate))


class AlignedMotionTests(unittest.TestCase):
    def test_forward_projection_includes_pitch_and_body_gyro(self):
        vector=forward_vector(imu(pitch=math.pi/3))
        self.assertAlmostEqual(math.hypot(*vector), .5)
        vector=forward_vector(imu(rate=1.), .02)
        self.assertAlmostEqual(math.atan2(vector[1],vector[0]), .02)

    def test_delayed_imu_is_matched_by_source_not_receipt(self):
        history=[(imu(0.),dict(received_ns=100,source_stamp_ns=1_000_000_000)),
                 (imu(.02),dict(received_ns=300,source_stamp_ns=1_020_000_000))]
        vector,metadata=aligned_forward(history,dict(received_ns=150,source_stamp_ns=1_010_000_000))
        self.assertAlmostEqual(math.atan2(vector[1],vector[0]), .01)
        self.assertEqual(metadata['method'],'source_interpolation')

    def test_source_clock_distance_is_independent_of_receipt_jitter(self):
        track=MotionTrack(lambda:0.)
        speed=[]; headings=[]
        for k,jitter in enumerate((0,5_000_000,-4_000_000,9_000_000,0)):
            identity=dict(received_ns=1_000_000_000+k*100_000_000+jitter,
                          source_stamp_ns=10_000_000_000+k*100_000_000)
            speed.append((dict(signed_speed=36.),identity))
            headings.append((imu(),identity))
        track.update(speed,headings,2_000_000_000)
        self.assertIsNone(track.failure)
        self.assertAlmostEqual(track.states[-1]['xy'][0],4.)

    def test_regressing_source_clock_refuses_instead_of_using_receipt_time(self):
        track=MotionTrack(lambda:0.)
        identities=[dict(received_ns=100_000_000,source_stamp_ns=2_000_000_000),
                    dict(received_ns=120_000_000,source_stamp_ns=1_000_000_000)]
        track.update([(dict(signed_speed=36.),i) for i in identities],
                     [(imu(),i) for i in identities],200_000_000)
        self.assertEqual(track.failure,'invalid_motion_interval')


class CausalAnchorTests(unittest.TestCase):
    def test_real_gps_loss_advances_from_allowed_sensors_and_stale_speed_refuses_pose(self):
        gps, status = FakeGps(), FakeStatus()
        hold = AlignedGpsBlackoutHold(gps, status, BASIS)
        hold.core.convergence_deg = lambda xy: 0.
        now = [1_000_000_000]
        with patch('time.monotonic_ns', side_effect=lambda: now[0]):
            status.add(now[0], 10., 0., source_ns=10_000_000_000)
            gps.set(hold._fix_from_grid((10., -500.)), now[0])
            hold.latest_snapshot()
            gps.fix = None
            for k in range(1, 51):
                now[0] = 1_000_000_000+k*20_000_000
                status.add(now[0], 10., 0., source_ns=10_000_000_000+k*20_000_000)
                value, _, meta = hold.latest_snapshot()
                self.assertEqual(meta['position_mode'], 'dead_reckoning')
            self.assertAlmostEqual(hold._grid(value)[0], 20., places=5)
            now[0] += 300_000_000
            value, _, meta = hold.latest_snapshot()
            self.assertIsNone(value)
            self.assertEqual(meta['position_mode'], 'unavailable')

    def test_motion_compensated_anchor_has_no_moving_average_position_lag_or_blackout_gps_feedback(self):
        gps,status=FakeGps(),FakeStatus()
        hold=AlignedGpsBlackoutHold(gps,status,BASIS,anchor_window_s=.4)
        hold.core.convergence_deg=lambda xy:0.
        hold.zone=SectionZone((5.,0.),(100.,0.))
        now=[1_000_000_000]
        with patch('time.monotonic_ns',side_effect=lambda:now[0]):
            for k in range(20):
                now[0]=1_000_000_000+k*20_000_000
                status.add(now[0],10.,0.,source_ns=10_000_000_000+k*20_000_000)
                gps.set(hold._fix_from_grid((k*.2+(.08 if k%2 else -.08),0.)),now[0])
                hold.latest_snapshot()
            self.assertAlmostEqual(hold.core.anchor_xy[0],3.8,places=5)
            self.assertAlmostEqual(hold.anchor_metadata['correction_xy'][0],-.08,places=5)
            # Discarded GPS is deliberately 20 m away. Only wheel/IMU motion advances position.
            now[0]+=20_000_000
            status.add(now[0],10.,0.,source_ns=10_400_000_000)
            gps.set(hold._fix_from_grid((25.,0.)),now[0])
            value,_,meta=hold.latest_snapshot()
            self.assertEqual(meta['position_mode'],'dead_reckoning')
            self.assertAlmostEqual(hold._grid(value)[0],4.,places=5)
            self.assertFalse(meta['anchor_filter']['GPS_inside_blackout_used'])
            self.assertAlmostEqual(meta['simulated_comparison']['estimated_xy'][0],4.,places=5)


if __name__=='__main__':
    unittest.main()
