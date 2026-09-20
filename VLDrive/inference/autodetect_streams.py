#!/usr/bin/env python3
"""Work out where the simulator is publishing, and emit the relay's arguments.

    python inference/autodetect_streams.py --seconds 8

inference/udp_relay.py ships with the port layout measured on our own
simulator: Vehicle Status to 909, LiDAR to 2368, and the four cameras plus GPS
and IMU sharing 9090, told apart only by the port they were sent from.  Another
machine's Network Settings may not match, and on a competition PC there is no
time to read six sensor dialogs and translate them into ``--demux`` by hand.

So listen on every plausible destination port, classify what arrives by its own
bytes rather than by which port it came in on, and print the ``--ports`` and
``--demux`` strings that carry it to the encoder.  The encoder-side targets are
read from the realtime config, so the output always lands where the encoder is
actually listening.

Standard library only, so the bare Windows interpreter can run it.  Read-only:
every socket is receive-only and nothing is ever sent.

The one thing the bytes cannot settle is which side each camera is on.  The
front camera is found by its resolution (1280x720 against 640x480 for the
sides), and the remaining three are assigned left, right, rear in ascending
order of source port, which is how MORAI's own dialogs are laid out.  That
assumption is reported in ``warnings`` so it can be checked rather than
trusted.
"""
from __future__ import annotations

import argparse
import collections
import json
import selectors
import socket
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every destination port worth listening on: the ones our own simulator uses,
# the ranges MORAI's dialogs default to, and the encoder-side ports in case
# someone has already pointed the sensors straight at them.
CANDIDATE_PORTS = (
    list(range(900, 921)) + list(range(2368, 2373)) + [1111] +
    list(range(5000, 5011)) + list(range(6000, 6011)) +
    list(range(7500, 7521)) + list(range(8000, 8011)) +
    list(range(9090, 9101)) + list(range(10500, 10521)) + [10909]
)
# Stream name -> the key the realtime config uses for the encoder-side port.
CAMERA_ORDER = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")
CAMERA_HEADER = b"MOR"
STATUS_HEADER = b"#MoraiInfo$"
IMU_HEADER = b"#IMUData$"
LIDAR_SIZES = (1206, 1248)
# morai_udp.COMPETITION_STATUS_BYTES and VEHICLE_STATUS_BYTES.  A stock
# simulator publishes both, and they open with the same '#MoraiInfo$', so only
# the length tells them apart; rule 2-8 allows just the 181-byte one.  Measured
# here: 909 carries 181 bytes and 911 carries 229.
COMPETITION_STATUS_BYTES = 181
EGO_STATUS_BYTES = 229


def jpeg_dimensions(data):
    """(width, height) from the first fragment of a MORAI camera datagram.

    The 'MOR' prefix is 15 bytes and the JPEG that follows carries its SOF
    marker well inside the first 65000-byte fragment, so the front camera can
    be told from the side cameras without reassembling the whole image.
    """
    start = data.find(b"\xff\xd8")
    if start < 0:
        return None
    offset = start + 2
    while offset + 9 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            height, width = struct.unpack_from(">HH", data, offset + 5)
            return width, height
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        length = struct.unpack_from(">H", data, offset + 2)[0]
        offset += 2 + length
    return None


def classify(payload):
    """Which stream a datagram belongs to, from its own bytes."""
    if payload.startswith(CAMERA_HEADER):
        return "camera"
    if payload.startswith(STATUS_HEADER):
        if len(payload) == EGO_STATUS_BYTES:
            return "vehicle_status_ego"   # not ours: rule 2-8 withholds it
        return "vehicle_status"
    if payload.startswith(IMU_HEADER):
        return "imu"
    if payload.startswith(b"$GP") or payload.startswith(b"$GN"):
        return "gps"
    if len(payload) in LIDAR_SIZES:
        return "lidar"
    return None


def listen(ports, seconds, bind):
    """Collect one sample per (destination port, source port) pair."""
    selector = selectors.DefaultSelector()
    opened = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
        except OSError:
            pass
        try:
            sock.bind((bind, port))
        except OSError:
            sock.close()
            continue
        sock.setblocking(False)
        selector.register(sock, selectors.EVENT_READ, port)
        opened.append(port)

    counts = collections.Counter()
    sample = {}
    import time
    deadline = time.time() + seconds
    while time.time() < deadline:
        for key, _ in selector.select(timeout=0.25):
            port = key.data
            try:
                payload, address = key.fileobj.recvfrom(70000)
            except OSError:
                continue
            pair = (port, address[1])
            counts[pair] += 1
            # Keep an index-0 camera fragment: only that one carries the JPEG header.
            if pair not in sample or (payload.startswith(CAMERA_HEADER)
                                      and not sample[pair].startswith(CAMERA_HEADER)):
                sample[pair] = payload
            elif payload.startswith(CAMERA_HEADER) and jpeg_dimensions(sample[pair]) is None:
                sample[pair] = payload

    for key in list(selector.get_map().values()):
        key.fileobj.close()
    selector.close()
    return opened, counts, sample


def resolve(counts, sample, targets):
    """(streams, relay ports, relay demux, warnings) from what was captured."""
    warnings = []
    found = {}          # stream name -> list of (dest, source, packets)
    cameras = []        # (dest, source, packets, width)
    for (dest, source), packets in sorted(counts.items()):
        kind = classify(sample[(dest, source)])
        if kind is None:
            continue
        if kind == "camera":
            size = jpeg_dimensions(sample[(dest, source)])
            cameras.append((dest, source, packets, size[0] if size else 0))
        else:
            found.setdefault(kind, []).append((dest, source, packets))

    for dest, source, _ in found.pop("vehicle_status_ego", []):
        warnings.append("{}번 포트의 {}바이트 상태는 Ego Vehicle Status라 규정 2-8 대상이 "
                        "아닙니다. 중계하지 않습니다.".format(dest, EGO_STATUS_BYTES))
    if "vehicle_status" not in found:
        warnings.append("{}바이트 Competition Vehicle Status가 없습니다. 시뮬레이터에서 "
                        "해당 publisher를 켜야 합니다.".format(COMPETITION_STATUS_BYTES))

    streams = {}
    for kind, hits in found.items():
        if len(hits) > 1:
            where = ", ".join("{}<-{}".format(dest, source) for dest, source, _ in hits)
            warnings.append("{}: 여러 곳에서 왔습니다 ({}). 패킷이 가장 많은 쪽을 씁니다.".format(
                kind, where))
            hits.sort(key=lambda hit: -hit[2])
        streams[kind] = {"dest": hits[0][0], "source": hits[0][1], "packets": hits[0][2]}

    if cameras:
        widest = max(camera[3] for camera in cameras)
        front = [camera for camera in cameras if camera[3] == widest and widest]
        if len(cameras) != 4:
            warnings.append("카메라가 4대가 아니라 {}대 잡혔습니다.".format(len(cameras)))
        if not front:
            warnings.append("카메라 해상도를 읽지 못해 송신 포트 오름차순으로 배정합니다.")
            ordered = sorted(cameras, key=lambda camera: camera[1])
        else:
            rest = sorted((camera for camera in cameras if camera is not front[0]),
                          key=lambda camera: camera[1])
            ordered = [front[0]] + rest
            warnings.append("front는 해상도 {}px로 확정했고, left/right/rear는 "
                            "송신 포트 오름차순으로 배정했습니다. 화면에서 확인하세요.".format(widest))
        for name, camera in zip(CAMERA_ORDER, ordered):
            streams[name] = {"dest": camera[0], "source": camera[1], "packets": camera[2]}

    # A destination port carrying more than one stream has to be split by sender.
    by_dest = collections.defaultdict(list)
    for name, info in streams.items():
        by_dest[info["dest"]].append(name)
    shared = sorted(dest for dest, names in by_dest.items() if len(names) > 1)
    if len(shared) > 1:
        warnings.append("공유 포트가 {}개입니다. udp_relay.py는 하나만 나눌 수 있어 "
                        "{}만 처리합니다.".format(len(shared), shared[0]))

    ports, demux = [], []
    for dest in sorted(by_dest):
        names = by_dest[dest]
        if dest in shared[:1]:
            for name in sorted(names, key=lambda n: streams[n]["source"]):
                demux.append("{}={}".format(streams[name]["source"], targets[name]))
        else:
            for name in names:
                target = targets[name]
                ports.append(str(dest) if dest == target else "{}:{}".format(dest, target))
    return (streams,
            ",".join(ports),
            "{}:{}".format(shared[0], ",".join(demux)) if shared else "",
            warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/vision_encoder_realtime.json"),
                        help="encoder-side ports are read from this config")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--json", action="store_true", help="machine-readable output only")
    args = parser.parse_args()

    targets = json.loads(Path(args.config).read_text(encoding="utf-8"))["ports"]
    opened, counts, sample = listen(CANDIDATE_PORTS, args.seconds, args.bind)
    if not counts:
        result = {"ok": False, "error": "no packets", "listened": len(opened)}
        print(json.dumps(result) if args.json else
              "{}개 포트를 {:.0f}초 들었지만 아무것도 오지 않았습니다.".format(len(opened), args.seconds))
        return 1

    streams, ports, demux, warnings = resolve(counts, sample, targets)
    missing = [name for name in targets if name not in streams]
    result = {
        "ok": not missing,
        "streams": streams,
        "relay_ports": ports,
        "relay_demux": demux,
        "missing": missing,
        "warnings": warnings,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return 0 if not missing else 1

    print("%-16s %-8s %-8s %s" % ("stream", "dest", "source", "packets"))
    print("-" * 48)
    for name in sorted(streams, key=lambda n: streams[n]["dest"]):
        info = streams[name]
        print("%-16s %-8s %-8s %s" % (name, info["dest"], info["source"], info["packets"]))
    print()
    print("--ports {}".format(ports or "(없음)"))
    print("--demux {}".format(demux or "(없음)"))
    for warning in warnings:
        print("주의: {}".format(warning))
    if missing:
        print("찾지 못한 스트림: {}".format(", ".join(missing)))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
