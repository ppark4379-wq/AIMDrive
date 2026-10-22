"""GPS blackout hold: pass-through, dead reckoning, limits, development blackout injection."""
import math
import threading
import time
import unittest

from inference.sensors.gps_blackout_hold import DeadReckoner, GpsBlackoutHold, SectionZone

BASIS = {'schema': 'morai_projected_grid_basis_v1', 'position_frame': 'projected_grid',
         'source_orientation_frame': 'true_enu', 'orientation_frame': 'projected_grid',
         'crs': '+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs', 'origin_xyz': [302595.0, 4124145.0, 0.0]}


class FakeGps:
    def __init__(self):
        self.fix, self.received_ns, self.packets, self.samples = None, 0, 0, 0

    def set(self, fix, received_ns):
        self.fix, self.received_ns = fix, received_ns

    def latest_snapshot(self):
        if self.fix is None:
            return None, float('inf'), dict(sample_id=0, received_ns=0, source_stamp_ns=0)
        age = (time.monotonic_ns() - self.received_ns) / 1e9
        return dict(self.fix), age, dict(sample_id=1, received_ns=self.received_ns, source_stamp_ns=0)

    def diagnostics(self):
        return {}

    def close(self):
        pass


class History:
    def __init__(self):
        self.history, self.history_lock = [], threading.Lock()


class FakeStatus:
    def __init__(self):
        self.speed, self.imu = History(), History()

    def add(self, received_ns, speed_mps, yaw, source_ns=0):
        self.speed.history.append((dict(signed_speed=speed_mps*3.6), dict(received_ns=received_ns, source_stamp_ns=source_ns)))
        self.imu.history.append((dict(yaw=yaw), dict(received_ns=received_ns-1000, source_stamp_ns=0)))


class DeadReckonerTests(unittest.TestCase):
    def test_straight_line_and_convergence(self):
        core = DeadReckoner(lambda xy: 0.)
        core.anchor((0., 0.), 0)
        for k in range(1, 51):                       # 1 s at 50 Hz, 10 m/s due east (true yaw 0)
            self.assertTrue(core.step(k*20_000_000, 10., 0.))
        self.assertAlmostEqual(core.xy[0], 10., places=6)
        self.assertAlmostEqual(core.xy[1], 0., places=6)
        core = DeadReckoner(lambda xy: -1.35)        # grid yaw = true yaw + convergence
        core.anchor((0., 0.), 0)
        for k in range(1, 51):
            core.step(k*20_000_000, 10., 0.)
        self.assertAlmostEqual(math.degrees(math.atan2(core.xy[1], core.xy[0])), -1.35, places=6)

    def test_limits_and_gaps_stop_the_hold(self):
        core = DeadReckoner(lambda xy: 0., max_hold_s=10., max_distance_m=5.)
        core.anchor((0., 0.), 0)
        ok = [core.step(k*20_000_000, 10., 0.) for k in range(1, 40)]
        self.assertFalse(all(ok))
        self.assertEqual(core.failure, 'hold_distance_limit')
        core = DeadReckoner(lambda xy: 0.)
        core.anchor((0., 0.), 0)
        self.assertFalse(core.step(400_000_000, 1., 0.))           # 0.4 s without status
        self.assertTrue(core.failure.startswith('status_gap'))
        core = DeadReckoner(lambda xy: 0., max_hold_s=.1)
        core.anchor((0., 0.), 0)
        ok = [core.step(k*20_000_000, 0., 0.) for k in range(1, 10)]  # stopped car still times out
        self.assertEqual(core.failure, 'hold_time_limit')
        self.assertFalse(DeadReckoner(lambda xy: 0.).step(1, 1., 0.))  # never anchored

    def test_latency_moves_the_anchor_forward_and_lever_follows_heading(self):
        core = DeadReckoner(lambda xy: 0., latency_s=.1)
        core.anchor((0., 0.), 0, true_yaw_rad=0., speed_mps=10.)
        self.assertAlmostEqual(core.xy[0], 1., places=9)          # fix is 0.1 s old at 10 m/s
        core = DeadReckoner(lambda xy: 0., lever_xy=(-1., 0.))     # antenna 1 m behind the reference point
        core.anchor((0., 0.), 0, true_yaw_rad=0.)
        self.assertAlmostEqual(core.ref_xy[0], 1., places=9)
        for k in range(1, 51):                                     # quarter turn in place (speed 0)
            core.step(k*20_000_000, 0., math.pi/2*k/50)
        self.assertAlmostEqual(core.xy[0], 1., places=6)          # antenna swings behind the new heading
        self.assertAlmostEqual(core.xy[1], -1., places=6)
        core = DeadReckoner(lambda xy: 0., latency_s=.1)           # unknown heading: no compensation
        core.anchor((0., 0.), 0, speed_mps=10.)
        self.assertEqual(core.xy, (0., 0.))
        with self.assertRaises(ValueError):
            DeadReckoner(lambda xy: 0., latency_s=1.)

    def test_zone(self):
        zone = SectionZone((0., 0.), (100., 0.), half_width_m=5.)
        self.assertTrue(zone.contains((50., 4.)))
        self.assertFalse(zone.contains((50., 6.)))
        self.assertFalse(zone.contains((-1., 0.)))


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.gps, self.status = FakeGps(), FakeStatus()
        self.hold = GpsBlackoutHold(self.gps, self.status, BASIS)
        self.origin_fix = self.hold._fix_from_grid((10., -500.))

    def test_fresh_gps_passes_through(self):
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now)
        fix, age, identity = self.hold.latest_snapshot()
        self.assertEqual(identity['position_mode'], 'gps')
        self.assertNotIn('dead_reckoning', fix)
        self.assertLess(age, .15)

    def test_stale_gps_is_held_with_status_and_imu(self):
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now - 1_000_000_000)          # anchor 1 s ago
        self.hold.core.anchor(self.hold._grid(self.origin_fix), now - 1_000_000_000)
        for k in range(1, 50):
            self.status.add(now - 1_000_000_000 + k*20_000_000, 5., 0.)
        fix, age, identity = self.hold.latest_snapshot()
        self.assertTrue(fix['dead_reckoning'])
        self.assertEqual(identity['position_mode'], 'dead_reckoning')
        x, y = self.hold._grid(fix)
        self.assertAlmostEqual(math.hypot(x-10., y+500.), 5.*0.98, delta=.02)
        self.assertLess(age, .1)
        self.assertEqual(self.hold.diagnostics()['blackout_hold']['dead_reckoning'], 1)

    def test_no_anchor_or_lost_status_refuses(self):
        fix, age, identity = self.hold.latest_snapshot()
        self.assertIsNone(fix)
        self.assertEqual(identity['reason'], 'no_gps_anchor')
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now - 2_000_000_000)
        self.hold.core.anchor(self.hold._grid(self.origin_fix), now - 2_000_000_000)
        self.status.add(now - 1_000_000_000, 5., 0.)                 # 1 s gap after the anchor
        fix, age, identity = self.hold.latest_snapshot()
        self.assertIsNone(fix)
        self.assertTrue(identity['reason'].startswith('status_'))

    def test_simulated_zone_discards_fixes_inside(self):
        self.hold.zone = SectionZone((0., -510.), (40., -490.), half_width_m=30.)
        now = time.monotonic_ns()
        self.gps.set(self.origin_fix, now)
        self.hold.core.anchor((0., -520.), now - 100_000_000)
        self.status.add(now - 50_000_000, 0., 0.)
        fix, _, identity = self.hold.latest_snapshot()
        self.assertEqual(identity['position_mode'], 'dead_reckoning')
        self.assertEqual(self.hold.counts['simulated_drops'], 1)
        self.assertEqual(identity['simulated_real_gps_xy'], [10., -500.])   # the discarded real fix, for evidence


class BlackoutShapesTests(unittest.TestCase):
    """What a GPS blackout can look like on the wire must never reach the planner as a position."""

    @staticmethod
    def receiver():
        from inference.sensors.gps_imu_status import ValidGpsReceiver
        rec = ValidGpsReceiver.__new__(ValidGpsReceiver)   # no socket: only the sentence filter is under test
        rec._fix = {"latitude": None, "longitude": None, "altitude": None}
        return rec

    def test_valid_fix_passes(self):
        for sentence in (b"$GPRMC,120000,A,3714.3293,N,12646.5115,E,0.0,0.0,010100,,,A\r\n",
                         b"$GPGGA,120000,3714.3293,N,12646.5115,E,1,08,0.9,28.5,M,0,M,,\r\n"):
            fix = self.receiver()._consume(sentence)
            self.assertAlmostEqual(fix["latitude"], 37.2388, places=3)
            self.assertAlmostEqual(fix["longitude"], 126.7752, places=3)

    def test_blackout_shapes_are_rejected(self):
        shapes = {
            "status V": b"$GPRMC,120000,V,3714.3293,N,12646.5115,E,0.0,0.0,010100,,,N\r\n",
            "quality 0": b"$GPGGA,120000,3714.3293,N,12646.5115,E,0,00,99.9,28.5,M,0,M,,\r\n",
            "zero position, fix flagged valid": b"$GPGGA,120000,0000.0000,N,00000.0000,E,1,08,0.9,0.0,M,0,M,,\r\n",
            "zero position RMC": b"$GPRMC,120000,A,0000.0000,N,00000.0000,E,0.0,0.0,010100,,,A\r\n",
            "empty fields": b"$GPGGA,,,,,,0,00,,,M,,M,,\r\n",
            "garbage": b"\x00\x01 not nmea",
            "empty datagram": b"",
        }
        for name, payload in shapes.items():
            self.assertIsNone(self.receiver()._consume(payload), name)

    def test_cut_or_zero_gps_switches_to_the_hold_after_the_freshness_window(self):
        gps, status = FakeGps(), FakeStatus()
        hold = GpsBlackoutHold(gps, status, BASIS)
        now = time.monotonic_ns()
        gps.set(hold._fix_from_grid((10., -500.)), now - 400_000_000)       # last fix 0.4 s ago, nothing newer arrives
        hold.core.anchor(hold._grid(gps.fix), now - 400_000_000)
        for k in range(1, 20):
            status.add(now - 400_000_000 + k*20_000_000, 8., 0.)
        fix, age, identity = hold.latest_snapshot()
        self.assertEqual(identity['position_mode'], 'dead_reckoning')
        self.assertLess(age, .15)


class LiveFrameSourceInterfaceTests(unittest.TestCase):
    """The VE process calls LiveFrameSource.stream_report() every few seconds; a missing attribute killed it at startup."""

    def test_stream_report_accepts_the_wrapper(self):
        from inference.sensors.live_frame import LiveFrameSource
        gps = FakeGps()
        gps.port, gps.packets, gps.samples = 10506, 7, 3
        hold = GpsBlackoutHold(gps, FakeStatus(), BASIS)
        source = LiveFrameSource.__new__(LiveFrameSource)
        source._receivers = lambda: iter([('gps', hold)])
        rows = LiveFrameSource.stream_report(source)
        self.assertEqual(rows[0][:4], ('gps', 10506, 7, 3))

    def test_wrapper_exposes_everything_the_receiver_contract_uses(self):
        gps = FakeGps(); gps.port, gps.packets, gps.samples = 1, 2, 3
        hold = GpsBlackoutHold(gps, FakeStatus(), BASIS)
        for name in ('port', 'packets', 'samples', 'latest', 'latest_snapshot', 'diagnostics', 'close'):
            self.assertTrue(hasattr(hold, name), name)


class ControlPoseSourceTests(unittest.TestCase):
    def test_dead_reckoning_source_requires_opt_in(self):
        from utils.common.ve_control_pose import decode_control_pose, DEAD_RECKONING_SOURCE, SCHEMA
        now = time.monotonic_ns()
        packet = dict(schema=SCHEMA, generation='g', checkpoint_sha256='c', output_frame='projected_grid',
                      ego_position_source=DEAD_RECKONING_SOURCE, ego_status_source='UDP_status_speed_IMU_heading',
                      sequence=1, sent_ns=now, sensor_received_ns=now-1000, ready=True,
                      ego_pose=dict(x=1., y=2., yaw_raw_deg=0., yaw_grid_rad=0., speed_mps=3.),
                      selected_samples=dict(gps=dict(received_ns=now-1000),
                                            vehicle_status=dict(received_ns=now-500, imu_received_ns=now-500)))
        with self.assertRaises(ValueError):
            decode_control_pose(packet, 'g', 'c', now)
        self.assertIs(decode_control_pose(packet, 'g', 'c', now, ('GPS', DEAD_RECKONING_SOURCE)), packet)


if __name__ == '__main__':
    unittest.main()
