"""Capture original camera/LiDAR timestamps without inference or clock conversion.

Receipt-nearest comparisons are diagnostic evidence, NOT synchronized samples.
All senders are kept separate. No sensor clock settings are changed.
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import selectors
import socket
import statistics
import struct
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def timestamp(sensor, payload):
    if sensor == 'camera':
        if len(payload) < 21 or payload[:3] != b'MOR' or payload[-2:] not in (b'AI', b'EI'):
            return None
        sec, nsec, index, size = struct.unpack_from('<IIII', payload, 3)
        if nsec >= 10**9 or size == 0 or size > len(payload) - 21 or index >= 4096:
            return None
        return sec * 10**9 + nsec
    if (len(payload) != 1206 or payload[:2] != b'\xff\xee'
            or payload[1204] not in (0x37, 0x38) or payload[1205] != 0x22):
        return None
    return struct.unpack_from('<I', payload, 1200)[0]


def distribution(values):
    return {'count': len(values), 'min': min(values), 'median': statistics.median(values),
            'max': max(values)} if values else {'count': 0}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['sensor'], tuple(row['peer']))].append(row)
    report = {'status': 'diagnostic_only', 'synchronized': False, 'streams': [], 'comparisons': [],
              'warning': 'Receipt time includes transport delay. Units/epoch candidates require validation; no anchors inferred.'}
    for (sensor, peer), samples in groups.items():
        differences = [b['raw_timestamp'] - a['raw_timestamp'] for a, b in zip(samples, samples[1:])]
        report['streams'].append({
            'sensor': sensor, 'peer': peer, 'samples': len(samples),
            'first_raw': samples[0]['raw_timestamp'], 'last_raw': samples[-1]['raw_timestamp'],
            'positive_raw_increment': distribution([d for d in differences if d > 0]),
            'backward_transitions': sum(d < 0 for d in differences),
            'receive_span_s': (samples[-1]['receive_monotonic_ns'] - samples[0]['receive_monotonic_ns']) / 1e9})
    for (sensor, camera_peer), cameras in groups.items():
        if sensor != 'camera':
            continue
        receipts = [r['receive_monotonic_ns'] for r in cameras]
        for (kind, lidar_peer), lidars in groups.items():
            if kind != 'lidar':
                continue
            pairs = []
            for lidar in lidars:
                index = bisect_left(receipts, lidar['receive_monotonic_ns'])
                indices = [i for i in (index-1, index) if 0 <= i < len(cameras)]
                cam = min((cameras[i] for i in indices), key=lambda c: abs(c['receive_monotonic_ns']-lidar['receive_monotonic_ns']))
                if abs(cam['receive_monotonic_ns'] - lidar['receive_monotonic_ns']) <= 200_000_000:
                    pairs.append((cam, lidar))
            entry = {'camera_peer': camera_peer, 'lidar_peer': lidar_peer, 'receipt_nearest_pairs': len(pairs), 'candidates': []}
            for unit, scale in [('seconds', 10**9), ('milliseconds', 10**6), ('microseconds', 1000), ('nanoseconds', 1)]:
                # Test numeric relationships, without guessing wall-clock anchors.
                for epoch, period in [('same_origin', None), ('within_hour', 3600 * 10**9),
                                      ('uint32_ticks', (2**32)*scale)]:
                    residuals = []
                    for camera, lidar in pairs:
                        delta = lidar['raw_timestamp'] * scale - camera['raw_timestamp']
                        if period:
                            delta = (delta + period//2) % period - period//2
                        residuals.append(delta / 1e6)
                    entry['candidates'].append({'lidar_unit': unit, 'relation': epoch,
                                                'lidar_minus_camera_ms': distribution(residuals)})
            report['comparisons'].append(entry)
    if not any(k[0] == 'camera' for k in groups) or not any(k[0] == 'lidar' for k in groups):
        report['status'] = 'insufficient_sensor_data'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--camera-port', type=int, default=5050)
    parser.add_argument('--lidar-port', type=int, default=2369)
    parser.add_argument('--input', type=Path, help='Analyze a previously captured timestamps.jsonl')
    args = parser.parse_args()
    if args.input:
        rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
        print(json.dumps(summarize(rows), indent=2))
        return
    if args.duration <= 0:
        parser.error('--duration must be positive')
    output = ROOT / 'artifacts' / 'timestamp_audit' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True)
    rows, last, counts = [], {}, defaultdict(int)
    sockets = []
    try:
        with selectors.DefaultSelector() as selector:
            for sensor, port in [('camera', args.camera_port), ('lidar', args.lidar_port)]:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4*1024*1024)
                sock.bind((args.bind, port))
                sock.setblocking(False)
                selector.register(sock, selectors.EVENT_READ, sensor)
            print(f'Capturing camera :{args.camera_port}, LiDAR :{args.lidar_port} for {args.duration}s; output={output}', flush=True)
            start = time.monotonic()
            next_report = start + 5
            with (output / 'timestamps.jsonl').open('x', encoding='utf-8') as handle:
                while time.monotonic() - start < args.duration:
                    for key, _ in selector.select(timeout=.1):
                        payload, peer = key.fileobj.recvfrom(65535)
                        received = time.monotonic_ns()
                        sensor = key.data
                        raw = timestamp(sensor, payload)
                        counts[sensor + '_packets'] += 1
                        if raw is None:
                            # Camera port can also carry non-JPEG message types.
                            counts[sensor + '_ignored_or_invalid'] += 1
                            continue
                        stream_key = (sensor, peer)
                        if last.get(stream_key) == raw:
                            continue
                        last[stream_key] = raw
                        row = {'sensor': sensor, 'peer': peer, 'raw_timestamp': raw,
                               'receive_monotonic_ns': received,
                               'host_unix_ns': time.time_ns()}
                        rows.append(row)
                        handle.write(json.dumps(row) + '\n')
                    if time.monotonic() >= next_report:
                        print(dict(counts), flush=True)
                        next_report = time.monotonic() + 5
    except KeyboardInterrupt:
        print('Stopped; writing partial report.', flush=True)
    finally:
        for sock in sockets:
            sock.close()
    report = summarize(rows)
    report['packet_counts'] = dict(counts)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
