"""GPS localization must never reuse the full status packet's GT pose/yaw."""
from collections import deque
import ctypes
import math
import threading
import time
import unittest
from types import SimpleNamespace

from inference.sensors.gps_imu_status import (
    ValidGpsReceiver, GpsImuStatusReceiver, SpeedReceiver, HeadingReceiver)
from inference.sensors.morai_udp import CompetitionVehicleStatusPacket, ImuPacket


def receiver(cls):
    value = cls.__new__(cls)
    value.history, value.history_lock = deque(maxlen=200), threading.Lock()
    value.samples = 0
    return value


def status_packet(sec=1, nsec=20_000_000, speed=36.):
    packet = CompetitionVehicleStatusPacket()
    packet.header, packet.tail = b'#MoraiInfo$', b'\r\n'
    packet.sec, packet.nsec = sec, nsec
    packet.signed_vel = speed
    # Deliberately contradictory GT fields must not affect speed or heading.
    packet.vel_x, packet.vel_y, packet.yaw = 999., 999., -170.
    packet.pos_x, packet.pos_y = 12345., 56789.
    return bytes(packet)


class GpsImuStatusTests(unittest.TestCase):
    def test_invalid_fix_quality_status_checksum_and_zero_position(self):
        gps = ValidGpsReceiver.__new__(ValidGpsReceiver)
        gps._fix = dict(latitude=None, longitude=None, altitude=None)
        for data in (
            b'$GPRMC,123000,V,3723.0,N,12638.0,E,0,0,010126,,,A',
            b'$GPGGA,123000,3723.0,N,12638.0,E,0,08,1.0,28,M,0,M,,',
            b'$GPRMC,123000,A,3723.0,N,12638.0,E,0,0,010126,,,A*00',
            b'$GPRMC,123000,A,0000.0,N,00000.0,E,0,0,010126,,,A'):
            self.assertIsNone(gps._consume(data))
        value = gps._consume(b'$GPRMC,123000,A,3723.0,N,12638.0,E,0,0,010126,,,A')
        self.assertAlmostEqual(value['latitude'], 37+23/60)
        self.assertAlmostEqual(value['longitude'], 126+38/60)
        self.assertIsNone(gps._consume(b'$GPRMC,123001,V,,,,,,,,,,,'))

    def test_signed_status_speed_uses_kmh_and_heading_uses_imu(self):
        speed = receiver(SpeedReceiver)
        speed._consume(status_packet())
        imu = receiver(HeadingReceiver)
        packet = ImuPacket()
        packet.header, packet.tail = b'#IMUData$', b'\r\n'
        packet.sec, packet.nsec = 1, 10_000_000
        packet.ori_w, packet.ori_z = math.cos(math.pi/4), math.sin(math.pi/4)
        imu._consume(bytes(packet))
        source = GpsImuStatusReceiver.__new__(GpsImuStatusReceiver)
        source.speed, source.imu, source.max_age = speed, imu, .5
        value, age, identity = source.at_source_stamp(1_030_000_000)
        self.assertEqual(value['speed'], 10.)
        self.assertAlmostEqual(value['yaw_degrees'], 90.)
        self.assertNotIn('ego_local', value)
        self.assertNotIn('position', value)
        self.assertEqual(identity['imu_alignment'], 'causal_50ms')
        self.assertLess(age, .5)
        self.assertIsNone(source.at_source_stamp(1_000_000_000)[0])
        self.assertIsNone(source.at_source_stamp(1_100_000_000)[0])

    def test_duplicate_simulation_time_does_not_renew_history_freshness(self):
        speed = receiver(SpeedReceiver)
        self.assertIsNotNone(speed._consume(status_packet()))
        previous = speed.history[-1][1]['received_ns']
        self.assertIsNone(speed._consume(status_packet()))
        self.assertEqual(speed.history[-1][1]['received_ns'], previous)
        self.assertEqual(len(speed.history), 1)
        speed._consume(status_packet(sec=0))
        self.assertEqual(len(speed.history), 1)
        self.assertEqual(speed.history[-1][1]['source_stamp_ns'], 20_000_000)

    def test_stale_heading_blocks_status_even_with_fresh_speed(self):
        source = GpsImuStatusReceiver.__new__(GpsImuStatusReceiver)
        source.max_age = .5
        now = time.monotonic_ns()
        source.speed = SimpleNamespace(sample=lambda _: (
            dict(signed_speed=36.), dict(received_ns=now)))
        source.imu = SimpleNamespace(sample=lambda *_: (
            dict(yaw=0.), dict(received_ns=now-1_000_000_000, source_stamp_ns=0)))
        self.assertIsNone(source.latest()[0])


if __name__ == '__main__':
    unittest.main()
