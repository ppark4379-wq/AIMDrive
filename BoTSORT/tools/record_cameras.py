#!/usr/bin/env python3
"""Bounded camera capture. UDP by default; optional ROS development input."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_mot.config import load_config
from camera_mot.sources import create_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/morai_three_cameras.yaml'))
    parser.add_argument('--cameras', nargs='+')
    parser.add_argument('--transport', choices=('udp', 'ros'))
    parser.add_argument('--duration', type=float, default=15)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('duration must be positive')
    config = load_config(args.config, args.cameras)
    if args.transport:
        config['transport'] = args.transport
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    for camera in config['cameras']:
        (output / camera).mkdir()
    source = create_source(config)
    start = time.monotonic()
    counts = {k: 0 for k in config['cameras']}
    try:
        with (output / 'frames.jsonl').open('x') as stream:
            while time.monotonic() - start < args.duration and not source.closed:
                frame = source.get()
                if frame is None:
                    continue
                name = '%s/e%d_%08d.jpg' % (frame.camera_id, frame.stream_epoch, frame.received_frame_id)
                (output / name).write_bytes(frame.jpeg)
                record = asdict(frame)
                record.pop('jpeg')
                record['image_path'] = name
                stream.write(json.dumps(record) + '\n')
                counts[frame.camera_id] += 1
    finally:
        source.close()
        report = {'duration_s': time.monotonic() - start, 'saved_frames': counts,
            'receiver': source.stats(), 'ignored_frames': source.ignored_frames}
        (output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    if not all(counts.values()):
        raise SystemExit('At least one requested camera produced no saved frame')


if __name__ == '__main__':
    main()
