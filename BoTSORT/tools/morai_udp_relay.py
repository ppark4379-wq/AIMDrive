#!/usr/bin/env python3
"""Receive MORAI UDP on Windows and relay unchanged JPEG/time in smaller UDP packets.

Uses only Python's standard library. Run on Windows, not inside WSL, when
large fragmented Windows-to-WSL UDP datagrams are being lost.
"""
import argparse
import json
from pathlib import Path
import selectors
import socket
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.morai_udp import JpegAssembler


def packetize_jpeg(jpeg, stamp, datagram_bytes=1200):
    """Preserve JPEG bytes and timestamp, using MORAI's zero-based wire format."""
    if not 64 <= datagram_bytes <= 1472:
        raise ValueError('datagram_bytes must be between 64 and 1472')
    if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
        raise ValueError('Invalid JPEG markers')
    sec, nsec = stamp
    if not 0 <= sec <= 0xffffffff or not 0 <= nsec < 1_000_000_000:
        raise ValueError('Invalid timestamp')
    payload_bytes = datagram_bytes - 21
    if (len(jpeg) + payload_bytes - 1) // payload_bytes > 4096:
        raise ValueError('JPEG needs too many fragments for the receiver')
    for index, start in enumerate(range(0, len(jpeg), payload_bytes)):
        part = jpeg[start:start + payload_bytes]
        final = start + len(part) == len(jpeg)
        yield (b'MOR' + struct.pack('<IIII', sec, nsec, index, len(part))
               + part + (b'EI' if final else b'AI'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind-ip', default='127.0.0.1')
    parser.add_argument('--destination-ip', required=True, help='Current WSL IPv4 address')
    parser.add_argument('--input-ports', nargs=3, type=int, default=[19291, 19293, 19295])
    parser.add_argument('--output-ports', nargs=3, type=int, default=[9291, 9293, 9295])
    parser.add_argument('--datagram-bytes', type=int, default=1200)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--stop-file', help='Stop cleanly when this local control file appears')
    parser.add_argument('--output', required=True, help='New directory for relay summary')
    args = parser.parse_args()
    if (args.duration <= 0 or not 64 <= args.datagram_bytes <= 1472
            or any(not 1 <= p <= 65535 for p in args.input_ports + args.output_ports)
            or len(set(args.input_ports)) != 3 or len(set(args.output_ports)) != 3):
        parser.error('Invalid duration, datagram size or port mapping')
    if set(args.input_ports) & set(args.output_ports):
        parser.error('Use distinct input/output ports to avoid local relay loops')
    for address in (args.bind_ip, args.destination_ip):
        try:
            socket.inet_pton(socket.AF_INET, address)
        except OSError:
            parser.error('IPv4 address required: ' + address)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    selector = selectors.DefaultSelector()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receivers = []
    names = ('front', 'left', 'right')
    assemblers = {key: JpegAssembler() for key in names}
    counts = {key: dict(input_datagrams=0, relayed_frames=0, output_datagrams=0,
                       output_bytes=0, last_timestamp=None) for key in names}
    start = time.monotonic()
    next_status = start
    stop_reason = 'duration'
    status = None
    failure = None
    try:
        status = (output / 'status.jsonl').open('x', buffering=1)
        for key, port, destination in zip(names, args.input_ports, args.output_ports):
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            receivers.append(receiver)
            receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            receiver.bind((args.bind_ip, port))
            receiver.setblocking(False)
            selector.register(receiver, selectors.EVENT_READ, (key, destination))
        print('READY: MORAI UDP -> relay -> WSL UDP', flush=True)
        while time.monotonic() - start < args.duration:
            for event, _ in selector.select(timeout=0.05):
                try:
                    packet, _ = event.fileobj.recvfrom(65535)
                except BlockingIOError:
                    continue
                key, port = event.data
                counts[key]['input_datagrams'] += 1
                frame = assemblers[key].feed(packet)
                if frame is None:
                    continue
                for smaller in packetize_jpeg(frame.jpeg, frame.stamp, args.datagram_bytes):
                    sender.sendto(smaller, (args.destination_ip, port))
                    counts[key]['output_datagrams'] += 1
                    counts[key]['output_bytes'] += len(smaller)
                counts[key]['relayed_frames'] += 1
                counts[key]['last_timestamp'] = frame.stamp
            for assembler in assemblers.values():
                assembler.expire(time.monotonic())
            now = time.monotonic()
            if now >= next_status:
                status.write(json.dumps({'elapsed_s': now - start, 'cameras': {
                    key: dict(**counts[key], invalid_packets=a.invalid_packets,
                              incomplete_frames=a.incomplete_frames,
                              pending_assemblies=len(a.pending))
                    for key, a in assemblers.items()}}) + '\n')
                next_status = now + 1
                if args.stop_file and Path(args.stop_file).exists():
                    stop_reason = 'stop_file'
                    break
    except KeyboardInterrupt:
        failure = 'interrupted'
        stop_reason = 'interrupted'
    except Exception as exc:
        failure = '%s: %s' % (type(exc).__name__, exc)
        raise
    finally:
        if status is not None:
            status.close()
        selector.close()
        sender.close()
        for receiver in receivers:
            receiver.close()
        report = dict(duration_s=time.monotonic() - start, failure=failure, stop_reason=stop_reason,
                      input_scope='MORAI RGB JPEG and original camera timestamp only', cameras={})
        for key, assembler in assemblers.items():
            report['cameras'][key] = dict(**counts[key], invalid_packets=assembler.invalid_packets,
                incomplete_frames=assembler.incomplete_frames, pending_assemblies=len(assembler.pending))
        (output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    return 0 if not failure and all(v['relayed_frames'] for v in counts.values()) else 2


if __name__ == '__main__':
    sys.exit(main())
