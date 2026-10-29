#!/usr/bin/env python3
"""Forward MORAI's UDP streams to the machine running the encoder.

Copy of VLDrive/inference/sensors/udp_relay.py (excluded from git on the PC that has it), so the competition
launcher AIM2026/competition_relay.ps1 runs from this repository alone.

    python AIM2026/UDP/udp_relay.py --to 172.18.254.59 --ports 909,2368

The simulator publishes to 127.0.0.1, which only a process on the same Windows
host can read, while the encoder runs under WSL with its own network namespace.
Rather than asking the simulator to target an address that changes whenever WSL
restarts, this sits on the Windows side and forwards each configured stream.
Camera JPEG bytes are split into MOR packets of at most 1200 bytes on the
Windows-to-WSL hop. Other streams are forwarded byte-for-byte. This avoids
forwarding the camera's padded 65000-byte datagrams across that hop without
changing the JPEG, its timestamp, or the existing receiver's loss checks.

Read-only in the direction that matters: it never sends anything to the
simulator, so no control command can escape through it.

Standard library only, so the bare Windows interpreter can run it with no
packages installed.  Run it in its own terminal and leave it going.
"""
from __future__ import annotations

import argparse
import selectors
import socket
import struct
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

CAMERA_PREFIX = struct.Struct('<3s4i')
CAMERA_CAPACITY = 64979
DEFAULT_CAMERA_DATAGRAM_BYTES = 1200


def camera_datagrams(payload, maximum_bytes=DEFAULT_CAMERA_DATAGRAM_BYTES):
    """Refragment MORAI JPEG bytes without changing their frame or content.

    Native camera datagrams are padded to 65000 bytes, including final parts.
    The receiver accepts variable-sized MOR/AI/EI fragments. Reserve a fixed
    index range for each native part so a missing part still creates a gap;
    never renumber away loss or join different timestamps. Non-camera packets
    and BOX annotations are forwarded byte-for-byte. Zero opts into raw relay.
    """
    if maximum_bytes == 0 or not payload.startswith(b'MOR'):
        yield payload
        return
    if not 256 <= maximum_bytes <= 1400:
        raise ValueError('Camera datagram limit must be 0 or between 256 and 1400 bytes')
    if len(payload) < CAMERA_PREFIX.size + 2:
        raise ValueError('Truncated camera header')
    tag, sec, nsec, index, size = CAMERA_PREFIX.unpack_from(payload)
    tail = payload[-2:]
    body = payload[CAMERA_PREFIX.size:-2]
    if (index < 0 or size <= 0 or size > min(len(body), CAMERA_CAPACITY)
            or tail not in (b'AI', b'EI') or (tail == b'AI' and size != CAMERA_CAPACITY)):
        raise ValueError('Malformed native camera fragment')
    capacity = maximum_bytes - CAMERA_PREFIX.size - 2
    slots = (CAMERA_CAPACITY + capacity - 1) // capacity
    parts = (size + capacity - 1) // capacity
    if index * slots + parts - 1 > 0x7fffffff:
        raise ValueError('Camera fragment index overflow')
    for part in range(parts):
        data = body[part * capacity:min(size, (part + 1) * capacity)]
        end = tail if part == parts - 1 else b'AI'
        yield CAMERA_PREFIX.pack(tag, sec, nsec, index * slots + part, len(data)) + data + end


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
    parser.add_argument('--log', help='append timestamped reports and stall warnings to this file '
                                      '(written before the console, so a blocked console still leaves a trace)')
    parser.add_argument('--camera-datagram-bytes', type=int, default=DEFAULT_CAMERA_DATAGRAM_BYTES,
                        help='Camera UDP bytes, default 1200; 0 forwards native 65000-byte datagrams')
    args = parser.parse_args()
    if args.camera_datagram_bytes != 0 and not 256 <= args.camera_datagram_bytes <= 1400:
        parser.error('--camera-datagram-bytes must be 0 or 256..1400')

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
    print('  camera UDP maximum={} bytes (0=native)'.format(args.camera_datagram_bytes), flush=True)

    counts = {port: 0 for port in bound}
    bytes_seen = {port: 0 for port in bound}
    unrouted = {port: {} for port in bound}
    last_report = time.time()
    previous = dict(counts)
    sent_packets = 0
    send_errors = 0
    malformed_camera_packets = 0
    log_file = open(args.log, 'a', buffering=1) if args.log else None

    def note(text):
        if log_file is not None:
            log_file.write('{} {}\n'.format(time.strftime('%H:%M:%S'), text))
    note('relay start -> {} ports={}'.format(args.to, sorted(bound)))
    loop_end = time.monotonic()
    try:
        while True:
            events = selector.select(timeout=0.5)
            # select returns within 0.5 s, so a larger gap means this process
            # was suspended or blocked (console output, power throttling, ...).
            stalled = time.monotonic() - loop_end
            if stalled > 1.0:
                note('STALL: loop not scheduled for {:.1f} s'.format(stalled))
                print('  STALL: relay was not running for {:.1f} s'.format(stalled), file=sys.stderr, flush=True)
            for key, _ in events:
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
                    for datagram in camera_datagrams(payload, args.camera_datagram_bytes):
                        out.sendto(datagram, (args.to, target))
                        sent_packets += 1
                except ValueError as error:
                    malformed_camera_packets += 1
                    if malformed_camera_packets <= 3:
                        print('  invalid camera fragment: '+str(error), file=sys.stderr, flush=True)
                except OSError as error:
                    send_errors += 1
                    if send_errors <= 3:
                        print('  UDP send failed: '+str(error), file=sys.stderr, flush=True)
            now = time.time()
            if args.report and now - last_report >= args.report:
                elapsed = now - last_report
                active = ["{}:{:.0f}/s".format(port, (counts[port] - previous[port]) / elapsed)
                          for port in bound if counts[port] != previous[port]]
                total = sum(bytes_seen.values()) / 1e6
                note('{} sent={} errors={}'.format(
                    ' '.join(active) if active else 'idle', sent_packets, send_errors))
                print("  {}   [{:.1f} MB received; sent={} errors={} invalid_camera={}]".format(
                    "  ".join(active) if active else "idle", total, sent_packets,
                    send_errors, malformed_camera_packets), flush=True)
                previous = dict(counts)
                last_report = now
            loop_end = time.monotonic()
    except KeyboardInterrupt:
        print()
        print("stopped. received per input port:")
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
        if log_file is not None:
            note('relay stopped')
            log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
