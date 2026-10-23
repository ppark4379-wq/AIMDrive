import math
import time
import unittest
from unittest.mock import patch
from gps_jamming.localization.gps_blackout_hold import DeadReckoner, GpsBlackoutHold, SectionZone
from localization_support import BASIS, FakeGps, FakeStatus

class DeadReckonerTests(unittest.TestCase):

    def test_straight_line_and_convergence(self):
        core = DeadReckoner(lambda xy: 0.0)
        core.anchor((0.0, 0.0), 0)
        for k in range(1, 51):
            self.assertTrue(core.step(k * 20000000, 10.0, 0.0))
        self.assertAlmostEqual(core.xy[0], 10.0, places=6)
        self.assertAlmostEqual(core.xy[1], 0.0, places=6)
        core = DeadReckoner(lambda xy: -1.35)
        core.anchor((0.0, 0.0), 0)
        for k in range(1, 51):
            core.step(k * 20000000, 10.0, 0.0)
        self.assertAlmostEqual(math.degrees(math.atan2(core.xy[1], core.xy[0])), -1.35, places=6)

    def test_limits_and_gaps_stop_the_hold(self):
        core = DeadReckoner(lambda xy: 0.0, max_hold_s=10.0, max_distance_m=5.0)
        core.anchor((0.0, 0.0), 0)
        ok = [core.step(k * 20000000, 10.0, 0.0) for k in range(1, 40)]
        self.assertFalse(all(ok))
        self.assertEqual(core.failure, 'hold_distance_limit')
        core = DeadReckoner(lambda xy: 0.0)
        core.anchor((0.0, 0.0), 0)
        self.assertFalse(core.step(400000000, 1.0, 0.0))
        self.assertTrue(core.failure.startswith('status_gap'))
        core = DeadReckoner(lambda xy: 0.0, max_hold_s=0.1)
        core.anchor((0.0, 0.0), 0)
        ok = [core.step(k * 20000000, 0.0, 0.0) for k in range(1, 10)]
        self.assertEqual(core.failure, 'hold_time_limit')
        self.assertFalse(DeadReckoner(lambda xy: 0.0).step(1, 1.0, 0.0))

    def test_latency_moves_the_anchor_forward_and_lever_follows_heading(self):
        core = DeadReckoner(lambda xy: 0.0, latency_s=0.1)
        core.anchor((0.0, 0.0), 0, true_yaw_rad=0.0, speed_mps=10.0)
        self.assertAlmostEqual(core.xy[0], 1.0, places=9)
        core = DeadReckoner(lambda xy: 0.0, lever_xy=(-1.0, 0.0))
        core.anchor((0.0, 0.0), 0, true_yaw_rad=0.0)
        self.assertAlmostEqual(core.ref_xy[0], 1.0, places=9)
        for k in range(1, 51):
            core.step(k * 20000000, 0.0, math.pi / 2 * k / 50)
        self.assertAlmostEqual(core.xy[0], 1.0, places=6)
        self.assertAlmostEqual(core.xy[1], -1.0, places=6)
        core = DeadReckoner(lambda xy: 0.0, latency_s=0.1)
        core.anchor((0.0, 0.0), 0, speed_mps=10.0)
        self.assertEqual(core.xy, (0.0, 0.0))
        with self.assertRaises(ValueError):
            DeadReckoner(lambda xy: 0.0, latency_s=1.0)

    def test_zone(self):
        zone = SectionZone((0.0, 0.0), (100.0, 0.0), half_width_m=5.0)
        self.assertTrue(zone.contains((50.0, 4.0)))
        self.assertFalse(zone.contains((50.0, 6.0)))
        self.assertFalse(zone.contains((-1.0, 0.0)))
        portal = SectionZone((29.589356, -487.650273), (-74.860745, -544.578033))
        self.assertTrue(portal.contains(portal.start))
        self.assertTrue(portal.contains((-74.860745, -544.578033)))
        with self.assertRaises(ValueError):
            SectionZone((0.0, 0.0), (float('nan'), 0.0))

    def test_anchor_velocity_is_used_for_the_first_acceleration_interval(self):
        core = DeadReckoner(lambda xy: 0.0)
        core.anchor((0.0, 0.0), 0, true_yaw_rad=0.0, speed_mps=0.0)
        core.step(100000000, 10.0, 0.0)
        self.assertAlmostEqual(core.xy[0], 0.5)

    def test_bad_source_intervals_fail_closed(self):
        for dt in (0.0, -0.1, float('nan'), 0.3):
            core = DeadReckoner(lambda xy: 0.0)
            core.anchor((0.0, 0.0), 0)
            self.assertFalse(core.step(20000000, 10.0, 0.0, dt_s=dt))
            self.assertEqual(core.failure, 'invalid_status_interval')
            self.assertEqual(core.xy, (0.0, 0.0))

class WrapperTests(unittest.TestCase):

    def setUp(self):
        self.gps, self.status = (FakeGps(), FakeStatus())
        self.hold = GpsBlackoutHold(self.gps, self.status, BASIS)
        self.origin_fix = self.hold._fix_from_grid((10.0, -500.0))

    def test_fresh_gps_passes_through(self):
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now)
        fix, age, identity = self.hold.latest_snapshot()
        self.assertEqual(identity['position_mode'], 'gps')
        self.assertNotIn('dead_reckoning', fix)
        self.assertLess(age, 0.15)

    def test_stale_gps_is_held_with_status_and_imu(self):
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now - 1000000000)
        self.hold.core.anchor(self.hold._grid(self.origin_fix), now - 1000000000)
        for k in range(1, 50):
            self.status.add(now - 1000000000 + k * 20000000, 5.0, 0.0)
        fix, age, identity = self.hold.latest_snapshot()
        self.assertTrue(fix['dead_reckoning'])
        self.assertEqual(identity['position_mode'], 'dead_reckoning')
        x, y = self.hold._grid(fix)
        self.assertAlmostEqual(math.hypot(x - 10.0, y + 500.0), 5.0 * 0.98, delta=0.02)
        self.assertLess(age, 0.1)
        self.assertEqual(self.hold.diagnostics()['blackout_hold']['dead_reckoning'], 1)

    def test_no_anchor_or_lost_status_refuses(self):
        fix, age, identity = self.hold.latest_snapshot()
        self.assertIsNone(fix)
        self.assertEqual(identity['reason'], 'no_gps_anchor')
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now - 2000000000)
        self.hold.core.anchor(self.hold._grid(self.origin_fix), now - 2000000000)
        self.status.add(now - 1000000000, 5.0, 0.0)
        fix, age, identity = self.hold.latest_snapshot()
        self.assertIsNone(fix)
        self.assertTrue(identity['reason'].startswith('status_'))

    def test_simulated_zone_discards_fixes_inside(self):
        self.hold.zone = SectionZone((0.0, -510.0), (40.0, -490.0), half_width_m=30.0)
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now)
        self.hold.core.anchor((0.0, -520.0), now - 100000000)
        self.status.add(now - 50000000, 0.0, 0.0)
        fix, _, identity = self.hold.latest_snapshot()
        self.assertEqual(identity['position_mode'], 'dead_reckoning')
        self.assertEqual(self.hold.counts['simulated_drops'], 1)
        self.assertEqual(identity['simulated_real_gps_xy'], [10.0, -500.0])

class ReceiptAlignedHoldTests(unittest.TestCase):
    """Exercise asynchronous inputs and polling without sockets or a simulator."""

    def setUp(self):
        self.now = 10000000000
        self.gps, self.status = (FakeGps(), FakeStatus())
        self.hold = GpsBlackoutHold(self.gps, self.status, BASIS, simulate_zone=SectionZone((0.0, 0.0), (100.0, 0.0), half_width_m=5.0))
        self.hold.core.convergence_deg = lambda xy: 0.0
        self.clock = patch('gps_jamming.localization.gps_blackout_hold.time.monotonic_ns', lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def anchor(self, speed=10.0):
        self.status.add(self.now - 1000000, speed, 0.0)
        self.gps.set(self.hold._fix_from_grid((-1.0, 0.0)), self.now)
        self.assertEqual(self.hold.latest_snapshot()[2]['position_mode'], 'gps')

    def test_anchor_uses_causal_inputs_once_per_fix(self):
        self.hold.core.latency_s = 0.1
        self.anchor(speed=2.0)
        self.assertAlmostEqual(self.hold.core.xy[0], -0.8, places=6)
        self.status.add(self.now + 10000000, 20.0, math.pi / 2)
        self.now += 20000000
        self.hold.latest_snapshot()
        self.assertAlmostEqual(self.hold.core.xy[0], -0.8, places=6)
        self.assertAlmostEqual(self.hold.core.xy[1], 0.0, places=6)
        self.assertEqual(self.hold.core.last_sample, (2.0, 0.0))

    def test_source_clock_integration_does_not_depend_on_poll_batching(self):
        outcomes = []
        for poll_between in (False, True):
            status, gps = (FakeStatus(), FakeGps())
            hold = GpsBlackoutHold(gps, status, BASIS)
            hold.core = DeadReckoner(lambda xy: 0.0)
            hold.core.anchor((0.0, 0.0), self.now, 0.0, 10.0)
            status.add(self.now + 20000000, 10.0, 0.0, source_ns=1000000000)
            if poll_between:
                self.now += 20000000
                hold.latest_snapshot()
                self.now -= 20000000
            status.add(self.now + 40000000, 10.0, 0.0, source_ns=1100000000)
            self.now += 40000000
            self.assertEqual(hold.latest_snapshot()[2]['position_mode'], 'dead_reckoning')
            outcomes.append(hold.core.xy[0])
            self.now -= 40000000
        self.assertAlmostEqual(outcomes[0], 1.2)
        self.assertAlmostEqual(outcomes[0], outcomes[1])

    def test_comparison_matches_receipt_and_does_not_correct_toward_dropped_gps(self):
        self.anchor()
        start_ns = self.now
        self.gps.set(self.hold._fix_from_grid((25.0, 0.0)), start_ns + 100000000)
        self.status.add(start_ns + 50000000, 10.0, 0.0)
        self.status.add(start_ns + 150000000, 10.0, 0.0)
        self.now += 160000000
        _, _, identity = self.hold.latest_snapshot()
        comparison = identity['simulated_comparison']
        self.assertEqual(comparison['gps_received_ns'], comparison['dr_received_ns'])
        self.assertEqual(comparison['gps_received_ns'], start_ns + 100000000)
        self.assertAlmostEqual(comparison['estimated_xy'][0], 0.0, places=6)
        self.assertAlmostEqual(comparison['real_gps_xy'][0], 25.0, places=6)
        self.assertAlmostEqual(self.hold.core.xy[0], 0.5, places=6)
        self.assertEqual(comparison['time_basis'], 'receipt_monotonic')
        self.assertEqual(comparison['alignment_method'], 'interpolated_dr_history')
        repeated = self.hold.latest_snapshot()[2]
        self.assertEqual(repeated['simulated_comparisons'], identity['simulated_comparisons'])
        self.assertEqual(self.hold.counts['simulated_drops'], 1)

    def test_comparison_waits_for_a_dr_state_after_the_gps_receipt(self):
        self.anchor()
        start_ns = self.now
        self.gps.set(self.hold._fix_from_grid((25.0, 0.0)), start_ns + 100000000)
        self.status.add(start_ns + 50000000, 10.0, 0.0)
        self.now += 110000000
        self.assertNotIn('simulated_comparison', self.hold.latest_snapshot()[2])
        self.status.add(start_ns + 150000000, 10.0, 0.0)
        self.now = start_ns + 160000000
        self.assertIn('simulated_comparison', self.hold.latest_snapshot()[2])

    def test_last_comparison_survives_gps_recovery_and_independent_polling(self):
        self.anchor()
        start_ns = self.now
        self.gps.set(self.hold._fix_from_grid((25.0, 0.0)), start_ns + 100000000)
        self.status.add(start_ns + 50000000, 10.0, 0.0)
        self.now += 110000000
        self.assertNotIn('simulated_comparison', self.hold.latest_snapshot()[2])
        self.status.add(start_ns + 150000000, 10.0, 0.0)
        self.gps.set(self.hold._fix_from_grid((101.0, 0.0)), start_ns + 180000000)
        self.now = start_ns + 190000000
        identity = self.hold.latest_snapshot()[2]
        self.assertEqual(identity['position_mode'], 'gps')
        comparison = identity['simulated_comparison']
        self.assertAlmostEqual(comparison['estimated_xy'][0], 0.0, places=6)
        self.assertEqual(comparison['gps_received_ns'], start_ns + 100000000)
        self.assertEqual(self.hold.latest_snapshot()[2]['simulated_comparison'], comparison)

    def full_comparison_window(self, speed=10.0, yaw=0.0):
        self.anchor()
        start_ns = self.now
        for k in range(1, 13):
            stamp = start_ns + k * 20000000
            self.gps.set(self.hold._fix_from_grid((25.0, 0.0)), stamp - 5000000)
            self.status.add(stamp, speed, yaw)
            self.now = stamp + 1000000
            identity = self.hold.latest_snapshot()[2]
        return identity

    def test_comparison_window_is_bounded_for_existing_udp_contract(self):
        identity = self.full_comparison_window()
        self.assertEqual(len(identity['simulated_comparisons']), 8)
        self.assertEqual(len({row['gps_received_ns'] for row in identity['simulated_comparisons']}), 8)

    def test_stale_imu_cannot_be_reused_with_fresh_speed(self):
        self.anchor()
        start_ns = self.now
        self.gps.fix = None
        self.status.speed.history.append((dict(signed_speed=36.0), dict(received_ns=start_ns + 240000000, source_stamp_ns=0)))
        self.status.speed.history.append((dict(signed_speed=36.0), dict(received_ns=start_ns + 260000000, source_stamp_ns=0)))
        self.now += 270000000
        value, _, identity = self.hold.latest_snapshot()
        self.assertIsNone(value)
        self.assertEqual(identity['reason'], 'imu_missing_or_stale')

    def test_silence_expires_even_when_no_new_status_is_processed(self):
        self.anchor()
        self.gps.fix = None
        self.now += 260000000
        self.assertEqual(self.hold.latest_snapshot()[2]['reason'], 'status_stale')

    def test_wall_time_limit_applies_without_new_status(self):
        self.anchor()
        self.hold.core.max_hold_s = 0.1
        self.gps.fix = None
        self.now += 110000000
        self.assertEqual(self.hold.latest_snapshot()[2]['reason'], 'hold_time_limit')

    def test_new_gps_recovers_from_refusal_and_resets_source_clock(self):
        self.anchor()
        self.gps.fix = None
        self.now += 300000000
        self.assertEqual(self.hold.latest_snapshot()[2]['position_mode'], 'unavailable')
        self.status.add(self.now - 1000000, 5.0, 0.0, source_ns=10)
        self.gps.set(self.hold._fix_from_grid((101.0, 0.0)), self.now)
        self.assertEqual(self.hold.latest_snapshot()[2]['position_mode'], 'gps')
        self.assertIsNone(self.hold.core.failure)
        self.assertEqual(self.hold._last_source_stamp_ns, 0)
        self.assertAlmostEqual(self.hold.core.xy[0], 101.0, places=6)

class LiveFrameSourceInterfaceTests(unittest.TestCase):
    """The VE process calls LiveFrameSource.stream_report() every few seconds; a missing attribute killed it at startup."""

    def test_wrapper_exposes_everything_the_receiver_contract_uses(self):
        gps = FakeGps()
        gps.port, gps.packets, gps.samples = (1, 2, 3)
        hold = GpsBlackoutHold(gps, FakeStatus(), BASIS)
        for name in ('port', 'packets', 'samples', 'latest', 'latest_snapshot', 'diagnostics', 'close'):
            self.assertTrue(hasattr(hold, name), name)
