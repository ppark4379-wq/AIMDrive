#!/usr/bin/env python3
"""Listen on candidate UDP ports and report what the simulator is actually sending.

    python inference/discover_udp.py                     # scan the usual ports
    python inference/discover_udp.py --ports 7505-7520   # scan a range
    python inference/discover_udp.py --seconds 20 --bind 0.0.0.0

Standard library only, so it runs under the bare Windows interpreter as well as
the WSL environment.  That matters: MORAI usually publishes to 127.0.0.1, which
only a process on the same Windows host can see, while the encoder runs in WSL
and needs the simulator pointed at the WSL address instead.  Run this on both
sides to find out which one the simulator is talking to.

For each port that receives anything it prints the packet count, the sizes seen,
the sender, and the first bytes, which is enough to tell the streams apart:

    'MOR'      camera image fragment   (3-byte header + 64997 payload)
    'BOX'      camera bounding boxes
    '$GPRMC'   GPS NMEA sentence
    52 bytes   IMU
    ~1206 B    3D LiDAR, Velodyne-style
"""
from __future__ import annotations

import argparse
import collections
import selectors
import socket
import sys
import time

# Ports MORAI's Network Settings tend to be configured with, plus the ones the
# official example scripts hard-code.
DEFAULT_PORTS = [
    1111,                               # the MORAI-NetworkModule examples
    2368, 2369,                         # Velodyne-style LiDAR
    7505, 7506, 7507, 7508, 7509, 7510,
    9090, 9091, 9092, 9093, 9094, 9095, 9096, 9097, 9098, 9099,
    5000, 5001, 5002, 5003,
    6000, 6001, 6002, 6003,
    8000, 8001, 8002, 8003,
]


def parse_ports(text):
    ports = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            low, high = piece.split("-", 1)
            ports.extend(range(int(low), int(high) + 1))
        else:
            ports.append(int(piece))
    return ports


def describe(payload):
    """A short guess at which stream a packet belongs to."""
    if payload[:3] == b"MOR":
        return "camera image fragment"
    if payload[:3] == b"BOX":
        return "camera bounding boxes"
    if payload[:1] == b"$":
        return "GPS NMEA"
    if len(payload) in (1206, 1248):
        return "LiDAR (Velodyne-style)"
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ports", default=None,
                        help="comma list and/or ranges, e.g. 7505-7510,9091")
    parser.add_argument("--bind", default="0.0.0.0", help="interface to listen on")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--buffer", type=int, default=70000,
                        help="recv size; MORAI camera datagrams are 65000 bytes")
    args = parser.parse_args()

    ports = parse_ports(args.ports) if args.ports else DEFAULT_PORTS
    selector = selectors.DefaultSelector()
    opened = []
    refused = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 20)
        except OSError:
            pass
        try:
            sock.bind((args.bind, port))
        except OSError as error:
            refused.append((port, error.strerror or str(error)))
            sock.close()
            continue
        sock.setblocking(False)
        selector.register(sock, selectors.EVENT_READ, port)
        opened.append(port)

    print("listening on {}  ports: {}".format(args.bind, len(opened)), flush=True)
    if refused:
        print("could not bind: {}".format(
            ", ".join("{} ({})".format(p, why) for p, why in refused[:8])), flush=True)

    counts = collections.Counter()
    sizes = collections.defaultdict(set)
    senders = {}
    first = {}
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        for key, _ in selector.select(timeout=0.25):
            port = key.data
            try:
                payload, address = key.fileobj.recvfrom(args.buffer)
            except OSError:
                continue
            counts[port] += 1
            sizes[port].add(len(payload))
            senders.setdefault(port, address)
            first.setdefault(port, payload[:16])

    for key in list(selector.get_map().values()):
        key.fileobj.close()
    selector.close()

    print()
    if not counts:
        print("no packets on any scanned port in {:.0f} s".format(args.seconds))
        print("If the simulator is running, its Network Settings are pointing somewhere")
        print("else; check the destination IP and port for each sensor and publisher.")
        return 1

    print("%7s %8s %-26s %-22s %s" % ("port", "packets", "sizes", "from", "first bytes / guess"))
    print("-" * 100)
    for port in sorted(counts):
        seen = sorted(sizes[port])
        shown = ", ".join(str(s) for s in seen[:4]) + ("..." if len(seen) > 4 else "")
        guess = describe(first[port])
        head = repr(first[port])[:26]
        print("%7d %8d %-26s %-22s %s %s" % (
            port, counts[port], shown, "{}:{}".format(*senders[port]), head,
            "<- " + guess if guess else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
