#!/usr/bin/env python3
"""Small, deterministic Python-only probe; does not connect to MORAI or write data.

A failure indicates runtime misexecution, not its underlying hardware/OS cause.
Run the identical command before and after collection on each affected machine.
"""
import argparse
import faulthandler
import json
import math
import platform
import sys
import time
from types import SimpleNamespace


class LaneMatch:
    def __init__(self, distance, x, y):
        self.distance = distance
        self.x = x
        self.y = y


class LaneMap:
    def project(self, position, points):
        best = None
        for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            dx = end.x - start.x
            dy = end.y - start.y
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-9:
                continue
            ratio = ((position.x - start.x) * dx + (position.y - start.y) * dy) / length_sq
            ratio = max(0.0, min(1.0, ratio))
            x = start.x + ratio * dx
            y = start.y + ratio * dy
            distance = math.hypot(position.x - x, position.y - y)
            if best is None or distance < best.distance:
                best = LaneMatch(distance, x, y)
        return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=20000)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    faulthandler.enable()
    print(json.dumps({'python': sys.version, 'executable': sys.executable,
                      'platform': platform.platform(), 'iterations': args.iterations}), flush=True)
    points = [SimpleNamespace(x=float(i), y=0.0) for i in range(65)]
    lane_map = LaneMap()
    started = time.monotonic()
    for iteration in range(args.iterations):
        position = SimpleNamespace(x=12.5 + iteration % 20, y=3.0)
        match = lane_map.project(position, points)
        if (match.x, match.y, match.distance) != (position.x, 0.0, 3.0):
            raise AssertionError('iteration {}: actual={!r}, expected={!r}'.format(iteration, (match.x, match.y, match.distance), (position.x, 0.0, 3.0)))
    print('PASS {} projections in {:.2f}s'.format(args.iterations, time.monotonic() - started), flush=True)


if __name__ == '__main__':
    main()