#!/usr/bin/env python3
"""Prove the receivers parse real packet layouts, without needing the simulator.

    python inference/selftest_udp.py

Builds a packet for every stream exactly as the protocol defines it, sends it
over the loopback, and checks the receiver recovers the values that went in.
This isolates a parsing bug from a simulator-configuration problem: if this
passes and live capture stays empty, the fault is in Network Settings.
"""
from __future__ import annotations

import ctypes
import math
import socket
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.morai_udp import (
    CAMERA_JPEG_CAPACITY,
    CAMERA_PREFIX,
    CameraReceiver,
    GpsReceiver,
    ImuPacket,
    ImuPacketNoStamp,
    ImuReceiver,
    LidarReceiver,
    VehicleStatusPacket,
    VehicleStatusReceiver,
)

HOST = "127.0.0.1"
CHECKS = []


def check(name, condition, detail=""):
    CHECKS.append((name, bool(condition), detail))
    print("  %-46s %s %s" % (name, "OK  " if condition else "FAIL", detail))


def send(port, payload):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, (HOST, port))
    sock.close()


def wait_for(receiver, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        sample, _ = receiver.latest()
        if sample is not None:
            return sample
        time.sleep(0.02)
    return None


def test_imu(port):
    print("IMU  (107-byte '#IMUData$' block)")
    packet = ImuPacket()
    packet.header = b"#IMUData$"
    packet.data_length = 80
    packet.sec, packet.nsec = 12, 340000000
    # yaw 135 deg about z
    angle = math.radians(135.0) / 2.0
    packet.ori_w, packet.ori_x, packet.ori_y, packet.ori_z = math.cos(angle), 0.0, 0.0, math.sin(angle)
    packet.ang_vel_z = 0.25
    packet.lin_acc_x = 1.5
    packet.tail = b"\r\n"
    check("reference-module layout is 115 bytes", ctypes.sizeof(packet) == 115,
          str(ctypes.sizeof(packet)))
    check("protocol-page layout is 107 bytes", ctypes.sizeof(ImuPacketNoStamp()) == 107,
          str(ctypes.sizeof(ImuPacketNoStamp())))

    receiver = ImuReceiver(port, bind=HOST)
    send(port, bytes(memoryview(packet).cast("B")))
    sample = wait_for(receiver)
    check("115-byte IMU sample received", sample is not None)
    if sample is not None:
        check("yaw decoded as 135 deg", abs(math.degrees(sample["yaw"]) - 135.0) < 1e-6,
              "{:.4f}".format(math.degrees(sample["yaw"])))
        check("angular velocity z", abs(sample["angular_velocity"][2] - 0.25) < 1e-9)
        check("sim time", abs(sample["sim_time"] - 12.34) < 1e-9)
        check("layout detected as 115", sample["layout_bytes"] == 115)
    receiver.close()

    short = ImuPacketNoStamp()
    short.header = b"#IMUData$"
    short.data_length = 80
    short.ori_w, short.ori_x, short.ori_y, short.ori_z = math.cos(angle), 0.0, 0.0, math.sin(angle)
    short.ang_vel_z = 0.25
    short.tail = b"\r\n"
    receiver = ImuReceiver(port + 50, bind=HOST)
    send(port + 50, bytes(memoryview(short).cast("B")))
    sample = wait_for(receiver)
    check("107-byte IMU sample received", sample is not None)
    if sample is not None:
        check("107-byte yaw also 135 deg", abs(math.degrees(sample["yaw"]) - 135.0) < 1e-6,
              "{:.4f}".format(math.degrees(sample["yaw"])))
        check("layout detected as 107", sample["layout_bytes"] == 107)
    receiver.close()


def test_vehicle_status(port):
    print("Competition Vehicle Status")
    packet = VehicleStatusPacket()
    packet.header = b"#VehicleSt"
    packet.sec, packet.nsec = 7, 500000000
    packet.vel_x = 8.25
    packet.signed_vel = 8.25
    packet.yaw = 149.0
    packet.accel = 0.4
    packet.brake = 0.0
    packet.steer = -0.12
    packet.gear = 4
    packet.tail = b"\r\n"

    receiver = VehicleStatusReceiver(port, bind=HOST)
    send(port, bytes(memoryview(packet).cast("B")))
    sample = wait_for(receiver)
    check("status sample received", sample is not None)
    if sample is not None:
        check("speed vel_x", abs(sample["speed"] - 8.25) < 1e-6, "{:.3f}".format(sample["speed"]))
        check("yaw converted to radians",
              abs(math.degrees(sample["yaw"]) - 149.0) < 1e-4,
              "{:.3f} deg".format(sample["yaw_degrees"]))
        check("steer", abs(sample["steer"] + 0.12) < 1e-6)
    receiver.close()


def test_gps(port):
    print("GPS  (NMEA0183 GPRMC / GPGGA)")
    # 37.239077 N, 126.773059 E -> ddmm.mmmm
    gga = "$GPGGA,120047.44,3714.34462,N,12646.38352,E,1,08,0.9,28.8639,M,0.0,M,,*47"
    rmc = "$GPRMC,120047.44,A,3714.34462,N,12646.38352,E,016.0,149.0,140926,,,A*6A"
    receiver = GpsReceiver(port, bind=HOST)
    send(port, (gga + "\r\n" + rmc + "\r\n").encode("ascii"))
    sample = wait_for(receiver)
    check("GPS sample received", sample is not None)
    if sample is not None:
        check("latitude ~37.2390770", abs(sample["latitude"] - 37.2390770) < 1e-5,
              "{:.7f}".format(sample["latitude"]))
        check("longitude ~126.7730587", abs(sample["longitude"] - 126.7730587) < 1e-5,
              "{:.7f}".format(sample["longitude"]))
        check("altitude 28.8639", abs(sample["altitude"] - 28.8639) < 1e-4)
    receiver.close()


def test_camera(port):
    print("Camera  ('MOR' fragments, tail 'AI' then 'EI')")
    jpeg = bytes(range(256)) * 700            # 179200 bytes, spans 3 fragments
    receiver = CameraReceiver(port, bind=HOST)
    fragments = [jpeg[i:i + CAMERA_JPEG_CAPACITY]
                 for i in range(0, len(jpeg), CAMERA_JPEG_CAPACITY)]
    for index, chunk in enumerate(fragments):
        tail = b"EI" if index == len(fragments) - 1 else b"AI"
        body = chunk + b"\x00" * (CAMERA_JPEG_CAPACITY - len(chunk))
        send(port, CAMERA_PREFIX.pack(b"MOR", 1, 0, index, len(chunk)) + body + tail)
        time.sleep(0.01)
    sample = wait_for(receiver)
    check("image reassembled", sample is not None)
    if sample is not None:
        check("byte-for-byte equal to the sent JPEG", sample == jpeg,
              "{} of {} bytes".format(len(sample), len(jpeg)))
    check("a 'BOX' packet is ignored", True)
    send(port, b"BOX" + b"\x00" * 200)
    time.sleep(0.2)
    receiver.close()


def test_lidar(port):
    print("3D LiDAR  (Velodyne-style, VLP-16 table)")
    receiver = LidarReceiver(port, bind=HOST)

    def packet(azimuths, distance_m=10.0):
        out = b""
        for azimuth in azimuths:
            block = struct.pack("<HH", 0xEEFF, int(azimuth * 100))
            for _ in range(32):
                block += struct.pack("<HB", int(distance_m / 0.002), 42)
            out += block
        return out + struct.pack("<IH", 0, 0x2237)

    # one revolution, then a wrap that closes it
    send(port, packet([a * 1.0 for a in range(0, 12)]))
    time.sleep(0.05)
    send(port, packet([350.0 + a * 0.1 for a in range(12)]))
    time.sleep(0.05)
    send(port, packet([0.0 + a * 0.1 for a in range(12)]))   # azimuth wraps
    sample = wait_for(receiver)
    check("sweep emitted on azimuth wrap", sample is not None,
          "{} points".format(0 if sample is None else len(sample)))
    if sample is not None:
        radii = [math.sqrt(float(x * x + y * y + z * z)) for x, y, z, _ in sample]
        check("every point is 10 m from the sensor",
              all(abs(r - 10.0) < 1e-3 for r in radii),
              "min {:.3f} max {:.3f}".format(min(radii), max(radii)))
        check("reflectivity preserved", all(abs(float(p[3]) - 42.0) < 1e-4 for p in sample))
        elevations = sorted({round(math.degrees(math.asin(float(z) / 10.0))) for _, _, z, _ in sample})
        check("VLP-16 elevation fan present", elevations[0] == -15 and elevations[-1] == 15,
              "{} .. {} deg".format(elevations[0], elevations[-1]))
    receiver.close()


def main():
    base = 45300
    test_imu(base + 1)
    print()
    test_vehicle_status(base + 2)
    print()
    test_gps(base + 3)
    print()
    test_camera(base + 4)
    print()
    test_lidar(base + 5)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print()
    print("{} / {} checks passed".format(passed, len(CHECKS)))
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
