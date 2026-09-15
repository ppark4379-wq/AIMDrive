#!/usr/bin/env python3
"""Forward MORAI's localhost UDP streams to the machine running the encoder.

    python inference/udp_relay.py --to 172.18.254.59 --ports 909,2368

The simulator publishes to 127.0.0.1, which only a process on the same Windows
host can read, while the encoder runs under WSL with its own network namespace.
Rather than asking the simulator to target an address that changes whenever WSL
restarts, this sits on the Windows side, reads each configured port and repeats
every datagram to the same port on the encoder host.

Read-only in the direction that matters: it never sends anything to the
simulator, so no control command can escape through it.

Standard library only, so the bare Windows interpreter can run it with no
packages installed.  Run it in its own terminal and leave it going.
"""
from __future__ import annotations

import argparse
import selectors
import socket
import sys
import time

# The ports a stock competition setup publishes on, discovered from a running
# simulator: LiDAR 2368 (1206-byte Velodyne), Competition Vehicle Status 909
# (181 bytes), plus the camera/GPS/IMU ports once those sensors are enabled.
DEFAULT_PORTS = "909:10909,2368"
# MORAI can be configured so several sensors share one destination port and are
# told apart only by the port they were sent from.  A plain relay would lose
# that, because forwarding replaces the source port, so those streams are split
# onto a port each on the way out.  Measured on a running simulator:
DEFAULT_DEMUX = "9090:7501=10501,7502=10502,7503=10503,7504=10504,7506=10506,7507=10507"


def parse_ports(text):
    """"909:10909,2368" -> [(909, 10909), (2368, 2368)].

    A destination below 1024 cannot be bound without root on Linux, so the
    simulator's privileged ports have to land somewhere else on the encoder
    host; the mapping syntax makes that explicit rather than silent.
    """
    pairs = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        source, _, target = piece.partition(":")
        if "-" in source:
            low, high = source.split("-", 1)
            pairs.extend((port, port) for port in range(int(low), int(high) + 1))
        else:
            pairs.append((int(source), int(target) if target else int(source)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--to", required=True,
                        help="encoder host, e.g. the WSL address from `ip addr show eth0`")
    parser.add_argument("--ports", default=DEFAULT_PORTS)
    parser.add_argument("--demux", default=DEFAULT_DEMUX,
                        help="LISTEN:SRC=DST[,SRC=DST...]; split one shared port by sender")
    parser.add_argument("--listen", default="127.0.0.1",
                        help="where the simulator publishes (default: localhost)")
    parser.add_argument("--report", type=float, default=5.0,
                        help="seconds between throughput lines; 0 to stay quiet")
    args = parser.parse_args()

    mapping = parse_ports(args.ports)
    demux = {}
    if args.demux:
        listen_port, _, routes = args.demux.partition(":")
        table = {}
        for route in routes.split(","):
            source, _, target = route.partition("=")
            if source and target:
                table[int(source)] = int(target)
        if table:
            demux[int(listen_port)] = table
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        out.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
    except OSError:
        pass

    selector = selectors.DefaultSelector()
    bound, refused = [], []
    targets = {}
    for port, target in list(mapping) + [(p, None) for p in demux if p not in dict(mapping)]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        except OSError:
            pass
        try:
            sock.bind((args.listen, port))
        except OSError as error:
            refused.append((port, error.errno))
            sock.close()
            continue
        sock.setblocking(False)
        selector.register(sock, selectors.EVENT_READ, port)
        targets[port] = target
        bound.append(port)

    if not bound:
        print("could not bind any port; is another relay already running?")
        return 1
    print("relaying {} -> {}".format(args.listen, args.to))
    print("  ports: {}".format(", ".join(
        "{} (demux)".format(p) if p in demux
        else str(p) if targets[p] == p else "{}->{}".format(p, targets[p])
        for p in bound)))
    for listen_port, table in demux.items():
        print("  {} splits by sender: {}".format(listen_port, ", ".join(
            "{}->{}".format(s, d) for s, d in sorted(table.items()))))
    if refused:
        print("  in use by another process (skipped): {}".format(
            ", ".join(str(p) for p, _ in refused)))
    print("  Ctrl+C to stop", flush=True)

    counts = {port: 0 for port in bound}
    bytes_seen = {port: 0 for port in bound}
    unrouted = {port: {} for port in bound}
    last_report = time.time()
    previous = dict(counts)
    try:
        while True:
            for key, _ in selector.select(timeout=0.5):
                port = key.data
                try:
                    payload, sender = key.fileobj.recvfrom(70000)
                except OSError:
                    continue
                counts[port] += 1
                bytes_seen[port] += len(payload)
                if port in demux:
                    target = demux[port].get(sender[1])
                    if target is None:
                        unrouted[port][sender[1]] = unrouted[port].get(sender[1], 0) + 1
                        continue
                else:
                    target = targets[port]
                try:
                    out.sendto(payload, (args.to, target))
                except OSError:
                    pass
            now = time.time()
            if args.report and now - last_report >= args.report:
                elapsed = now - last_report
                active = ["{}:{:.0f}/s".format(port, (counts[port] - previous[port]) / elapsed)
                          for port in bound if counts[port] != previous[port]]
                total = sum(bytes_seen.values()) / 1e6
                print("  {}   [{:.1f} MB forwarded]".format(
                    "  ".join(active) if active else "idle", total), flush=True)
                previous = dict(counts)
                last_report = now
    except KeyboardInterrupt:
        print()
        print("stopped. forwarded per port:")
        for port in bound:
            print("  {:>5}  {:>9} packets  {:>8.1f} MB".format(
                port, counts[port], bytes_seen[port] / 1e6))
            for sender, dropped in sorted(unrouted[port].items()):
                print("        no route for sender {}: {} packets dropped".format(sender, dropped))
    finally:
        for key in list(selector.get_map().values()):
            key.fileobj.close()
        selector.close()
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
