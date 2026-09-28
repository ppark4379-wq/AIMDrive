"""Brief standalone Ego UDP diagnosis. Close the viewer before using this port."""
from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.interfaces.ego_udp import EgoUDPReceiver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9094)
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    with EgoUDPReceiver(bind=args.bind, port=args.port) as receiver:
        print(f'Listening {receiver.address}', flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            time.sleep(min(1, max(0, deadline-time.monotonic())))
            print(receiver.snapshot(), flush=True)


if __name__ == '__main__':
    main()
