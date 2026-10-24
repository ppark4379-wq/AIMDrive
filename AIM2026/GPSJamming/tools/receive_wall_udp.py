"""Inspect the Expert-facing UDP stream without ROS or model dependencies."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.UDP import WallUDPReceiver, PORT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--full', action='store_true', help='Print all XYZ coordinates')
    args = parser.parse_args()
    receiver = WallUDPReceiver(args.bind, args.port)
    try:
        while True:
            packet = receiver.poll()
            if packet is None:
                print(json.dumps(dict(status='waiting_or_stale', rejected=receiver.rejected)), flush=True)
            elif args.full:
                print(json.dumps(packet), flush=True)
            else:
                b = packet['boundaries']
                print(json.dumps(dict(sequence=packet['sequence'], frame_id=b['frame_id'],
                    frame_sequence=b.get('frame_sequence'), valid=b.get('valid'),
                    timestamp_status=b.get('timestamp_status'),
                    segments={s: len(b[s]) for s in ('left', 'right')},
                    inferred={s: any(v.get('inferred', False) for v in b[s]) for s in ('left', 'right')})), flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()


if __name__ == '__main__':
    main()
