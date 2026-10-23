"""Replay audit_wall_frame evidence without ROS or sensor access."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gps_jamming.lidar.walls import detect_walls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frame', type=Path)
    args = parser.parse_args()
    saved = json.loads(args.frame.with_suffix('.json').read_text(encoding='utf-8'))
    with np.load(args.frame) as frame:
        result, clouds = detect_walls(frame['points'], frame['plane'], **saved['config'])
    report = {}
    for side, sign in (('left', 1), ('right', -1)):
        report[side] = dict(before=saved['live'][side], after=[])
        for segment in result[side]:
            lo, hi = segment['observed_range']
            points = clouds[side]
            points = points[(points[:,0]>=lo)&(points[:,0]<=hi)]
            distance = sign*(points[:,1]-segment['line_slope']*points[:,0]-segment['line_intercept'])/np.hypot(1,segment['line_slope'])
            assert distance.min() >= saved['config']['clearance_m']-1e-9
            assert segment['points'][0][0] >= points[:,0].min()-1e-9
            assert segment['points'][-1][0] <= points[:,0].max()+1e-9
            report[side]['after'].append(dict(range=segment['observed_range'],
                count=len(points), crossing_points=int(np.sum(sign*points[:,1]<0)),
                minimum_clearance_m=float(distance.min())))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
