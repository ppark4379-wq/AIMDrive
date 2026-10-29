"""A short sensor gap during the GPS blackout must not end dead reckoning for good.

The failure held until GPS returned, so the Ego stopped in the tunnel, twice:
10-29 (competition MORAI): one IMU stamp came one second back, the receiver took it for a simulator reset and emptied
the IMU history; the next status sample had no IMU sample before it (imu_missing_or_stale).
10-27: the simulator lagged and sent status and IMU 0.3-0.54 s apart (status_stale, invalid_motion_interval).
"""
import math
import unittest
from unittest.mock import patch

from AIM2026.Localization.gps_aligned_hold import AlignedGpsBlackoutHold, IMU_GAP_MAX_S, MotionTrack, TRACK_GAP_MAX_S
from AIM2026.Localization.tests.support import BASIS, FakeGps, FakeStatus

MS = 1_000_000
SOURCE0 = 10_000*1_000*MS
POLL_LAG_S = .003       # polls (and the GPS anchor) come 3 ms after the status sample: 0.03 m at 10 m/s


def imu(yaw=0.):
    return dict(yaw=yaw, quaternion=(0., 0., math.sin(yaw/2), math.cos(yaw/2)), angular_velocity=(0., 0., 0.))


class Stream:
    """Status every 20 ms at 10 m/s, IMU every 20 ms (2 ms later) except in ``gaps`` = [(start_s, end_s)];
    in ``lags`` neither comes (a lagging simulator: its clock skips the interval)."""

    def __init__(self, gaps=(), yaw=lambda t: 0., start_ns=1_000*MS, lags=()):
        self.speed, self.imu, self.gaps, self.yaw, self.start, self.lags = [], [], gaps, yaw, start_ns, lags
        self.k = 0

    def until(self, t):
        k = self.k
        while k*.02 <= t+1e-9:
            at = k*.02
            k += 1
            if any(a <= at < b for a, b in self.lags):
                continue
            self.speed.append((dict(signed_speed=36.), dict(received_ns=self.start+(k-1)*20*MS,
                                                            source_stamp_ns=SOURCE0+(k-1)*20*MS)))
            if not any(a <= at < b for a, b in self.gaps):
                self.imu.append((imu(self.yaw(at)), dict(received_ns=self.start+(k-1)*20*MS+2*MS,
                                                         source_stamp_ns=SOURCE0+(k-1)*20*MS)))
        self.k = k
        return self.start+round(t*1e9)+3*MS


class MotionTrackGapTests(unittest.TestCase):
    def run_track(self, stream, end_s):
        track, waiting = MotionTrack(lambda: 0.), []
        t = 0.
        while t <= end_s+1e-9:
            now = stream.until(t)
            track.update(stream.speed, stream.imu, now)
            waiting.append(track.waiting)
            if track.failure:
                break
            t += .02
        return track, waiting

    def test_a_gap_is_waited_for_then_bridged_and_the_distance_stays_exact(self):
        track, waiting = self.run_track(Stream(gaps=[(.3, .9)]), 1.5)
        self.assertIsNone(track.failure)
        self.assertIn('imu_missing_or_stale', waiting)               # no position while the gap is open
        self.assertIsNone(waiting[-1])
        methods = {state['alignment']['method'] for state in track.states}
        self.assertIn('gap_interpolation', methods)
        last = track.states[-1]
        self.assertAlmostEqual(last['xy'][0], 10.*1.5, places=6)      # straight at 10 m/s, source clock
        self.assertAlmostEqual(last['xy'][1], 0., places=6)

    def test_the_heading_over_a_gap_moves_from_the_last_to_the_next_imu_sample(self):
        track, _ = self.run_track(Stream(gaps=[(.3, .9)], yaw=lambda t: 0. if t < .6 else .4), 1.2)
        self.assertIsNone(track.failure)
        bridged = [s['yaw'] for s in track.states if s['alignment']['method'] == 'gap_interpolation']
        self.assertTrue(bridged and all(0. < y < .4 for y in bridged))
        self.assertEqual(bridged, sorted(bridged))

    def test_a_gap_longer_than_the_limit_still_ends_the_track(self):
        track, _ = self.run_track(Stream(gaps=[(.3, .3+IMU_GAP_MAX_S+.6)]), 2.5)
        self.assertEqual(track.failure, 'imu_missing_or_stale')

    def test_a_lagging_simulator_is_integrated_across_its_skipped_interval(self):
        track, _ = self.run_track(Stream(lags=[(.3, .82)]), 1.5)
        self.assertIsNone(track.failure)
        self.assertAlmostEqual(track.states[-1]['xy'][0], 10.*1.5, places=6)

    def test_a_simulator_pause_longer_than_the_limit_still_ends_the_track(self):
        track, _ = self.run_track(Stream(lags=[(.3, .3+TRACK_GAP_MAX_S+.4)]), 2.5)
        self.assertEqual(track.failure, 'invalid_motion_interval')


class BlackoutHoldGapTests(unittest.TestCase):
    def setUp(self):
        self.gps, self.status = FakeGps(), FakeStatus()
        self.hold = AlignedGpsBlackoutHold(self.gps, self.status, BASIS)
        self.hold.core.convergence_deg = lambda xy: 0.
        self.now = [0]

    def drive(self, stream, start_s, end_s):
        modes, t = [], start_s
        while t <= end_s+1e-9:
            self.now[0] = stream.until(t)
            self.status.speed.history[:] = stream.speed
            self.status.imu.history[:] = stream.imu
            value, _, meta = self.hold.latest_snapshot()
            modes.append((round(t, 2), meta['position_mode'], meta.get('reason'), value))
            t += .02
        return modes

    def anchor(self, stream):
        self.now[0] = stream.until(0.)
        self.status.speed.history[:] = stream.speed
        self.status.imu.history[:] = stream.imu
        self.gps.set(self.hold._fix_from_grid((10., -500.)), self.now[0])
        self.assertEqual(self.hold.latest_snapshot()[2]['position_mode'], 'gps')
        self.gps.fix = None

    def test_an_emptied_imu_history_does_not_end_the_hold(self):
        stream = Stream()
        with patch('time.monotonic_ns', side_effect=lambda: self.now[0]):
            self.anchor(stream)
            modes = self.drive(stream, .02, .6)
            # The bad IMU sample (stamp one second back) empties the receiver's IMU history; it is all that is left
            # when the next status sample is integrated, and it came after that status sample.
            k = len(stream.speed)
            stream.until(k*.02)
            value, identity = stream.imu[-1]
            stream.imu[:] = [(value, dict(identity, source_stamp_ns=identity['source_stamp_ns']-998*MS))]
            modes += self.drive(stream, k*.02, 1.4)
        self.assertTrue(all(mode == 'dead_reckoning' for _, mode, _, _ in modes), modes)
        self.assertIsNone(self.hold.core.failure)
        self.assertAlmostEqual(self.hold._grid(modes[-1][3])[0], 10.+10.*(1.4-POLL_LAG_S), places=4)

    def test_an_imu_gap_refuses_a_position_and_the_hold_resumes_after_it(self):
        stream = Stream(gaps=[(.5, 1.1)])
        with patch('time.monotonic_ns', side_effect=lambda: self.now[0]):
            self.anchor(stream)
            modes = self.drive(stream, .02, 2.)
        refused = [m for m in modes if m[1] == 'unavailable']
        self.assertTrue(refused)
        self.assertTrue(all(reason == 'imu_missing_or_stale' for _, _, reason, _ in refused))
        self.assertEqual(modes[-1][1], 'dead_reckoning')
        self.assertIsNone(self.hold.core.failure)
        self.assertAlmostEqual(self.hold._grid(modes[-1][3])[0], 10.+10.*(2.-POLL_LAG_S), places=4)

    def test_a_long_imu_gap_still_ends_the_hold_until_gps_returns(self):
        stream = Stream(gaps=[(.5, .5+IMU_GAP_MAX_S+.5)])
        with patch('time.monotonic_ns', side_effect=lambda: self.now[0]):
            self.anchor(stream)
            modes = self.drive(stream, .02, 2.5)
        self.assertEqual(modes[-1][1], 'unavailable')
        self.assertEqual(self.hold.core.failure, 'imu_missing_or_stale')

    def test_a_lagging_simulator_refuses_a_position_and_the_hold_resumes_after_it(self):
        stream = Stream(lags=[(.5, 1.)])
        with patch('time.monotonic_ns', side_effect=lambda: self.now[0]):
            self.anchor(stream)
            modes = self.drive(stream, .02, 2.)
        self.assertTrue(any(m[1] == 'unavailable' and m[2] == 'status_stale' for m in modes))
        self.assertEqual(modes[-1][1], 'dead_reckoning')
        self.assertIsNone(self.hold.core.failure)
        self.assertAlmostEqual(self.hold._grid(modes[-1][3])[0], 10.+10.*(2.-POLL_LAG_S), places=4)

    def test_gps_after_a_bridged_gap_is_not_averaged_with_gps_before_it(self):
        # Run start, 10-29: MORAI held the car still for 0.38 s while the status kept 4 m/s; the bridged track gained
        # 1.4 m, and the 1 s mean of GPS offsets put the fused GPS output 1.4 m off for a second.
        stream = Stream(lags=[(.5, .9)])
        truth = lambda t: 10.+10.*(min(t, .5)+max(0., t-.9))          # the car stands still in the gap
        outputs = []
        with patch('time.monotonic_ns', side_effect=lambda: self.now[0]):
            t = 0.
            while t <= 1.5+1e-9:
                self.now[0] = stream.until(t)
                self.status.speed.history[:] = stream.speed
                self.status.imu.history[:] = stream.imu
                if not .5 <= t < .9:
                    self.gps.set(self.hold._fix_from_grid((truth(t), -500.)), self.now[0])
                value, _, meta = self.hold.latest_snapshot()
                if t >= .9 and meta['position_mode'] == 'gps':
                    outputs.append(self.hold._grid(value)[0]-truth(t))
                t += .02
        self.assertTrue(outputs)
        self.assertLess(max(abs(e) for e in outputs), .05)

    def test_late_status_samples_are_refused_without_ending_the_hold(self):
        stream = Stream()
        with patch('time.monotonic_ns', side_effect=lambda: self.now[0]):
            self.anchor(stream)
            self.drive(stream, .02, .5)
            self.now[0] += 260*MS                                     # a poll with the status 0.26 s old
            meta = self.hold.latest_snapshot()[2]
            self.assertEqual((meta['position_mode'], meta['reason']), ('unavailable', 'status_stale'))
            self.assertIsNone(self.hold.core.failure)
            modes = self.drive(stream, .52, .8)
        self.assertEqual(modes[-1][1], 'dead_reckoning')


if __name__ == '__main__':
    unittest.main()
