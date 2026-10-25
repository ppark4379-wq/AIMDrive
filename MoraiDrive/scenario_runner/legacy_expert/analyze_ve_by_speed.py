#!/usr/bin/env python3
"""VE vs GT box error by Ego speed, from a recording that has bbox_reference_frames.jsonl and gt_ego.jsonl.

Ego speed comes from position differences of the GT Ego stream (not from VE). Read-only.
Usage: analyze_ve_by_speed.py RUN_DIR [RUN_DIR ...] [--bins 0,20,40,60,70,80,90,100,130]
"""
import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np


def ego_speed_table(run):
    rows = [json.loads(l) for l in (run/'gt_ego.jsonl').read_text().splitlines() if l.strip()]
    t = np.array([r['source_stamp_ns']/1e9 for r in rows]); p = np.array([r['position'][:2] for r in rows])
    # speed over ~0.5 s windows so single-sample jitter does not create fake peaks
    out_t, out_v = [], []
    j = 0
    for i in range(len(t)):
        while t[i]-t[j] > .5:
            j += 1
        if t[i]-t[j] >= .25:
            out_t.append(t[i]); out_v.append(math.hypot(*(p[i]-p[j]))/(t[i]-t[j])*3.6)
    return np.array(out_t), np.array(out_v)


def collect(run):
    times, speeds = ego_speed_table(run)
    items, missed = [], []
    for line in (run/'bbox_reference_frames.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r['gt_source_stamp_ns']/1e9
        k = bisect.bisect_left(times, t)
        if k == len(times) or (k > 0 and abs(times[k-1]-t) < abs(times[k]-t)):
            k -= 1
        if k < 0 or abs(times[k]-t) > .5:
            continue
        speed = speeds[k]
        ego = r['ego']
        for m in r['score']['matched']:
            gt = r['gt'][m['gt_index']]
            rng = math.dist(gt['center_world'], [ego['x'], ego['y']])
            items.append((speed, rng, m))
        for gi in r['score'].get('missed_gt', []):
            gt = r['gt'][gi] if isinstance(gi, int) else gi
            rng = math.dist(gt['center_world'], [ego['x'], ego['y']]) if isinstance(gt, dict) and 'center_world' in gt else float('nan')
            missed.append((speed, rng))
    return items, missed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--bins', default='0,20,40,60,70,80,90,100,130')
    args = parser.parse_args()
    edges = [float(x) for x in args.bins.split(',')]
    items, missed = [], []
    for run in args.runs:
        a, b = collect(run); items += a; missed += b
    print('매칭 %d건, 놓침 %d건, 자차 속도 범위 %.0f~%.0f km/h' % (len(items), len(missed), min(i[0] for i in items), max(i[0] for i in items)))
    print('%-10s %6s %7s | %7s %7s %7s %7s %7s %7s %7s' % ('속도km/h', '매칭', '놓침', 'recall', '중심m', '앞뒤편향', '길이m', '폭m', '방향°', 'IoU'))
    for lo, hi in zip(edges, edges[1:]):
        s = [i for i in items if lo <= i[0] < hi]; n_miss = sum(lo <= m[0] < hi for m in missed)
        if not s and not n_miss:
            continue
        if len(s) < 10:
            print('%-10s %6d %7d | (표본 부족)' % ('%g-%g' % (lo, hi), len(s), n_miss)); continue
        g = lambda k: np.mean([abs(i[2][k]) for i in s])
        print('%-10s %6d %7d | %7.3f %7.2f %+7.2f %7.2f %7.2f %7.2f %7.3f' % (
            '%g-%g' % (lo, hi), len(s), n_miss, len(s)/(len(s)+n_miss), g('center_error_m'),
            np.mean([i[2]['longitudinal_bias_m'] for i in s]), g('length_error_m'), g('width_error_m'),
            g('heading_error_deg_mod180'), np.mean([i[2]['iou'] for i in s])))
    sp = np.array([i[0] for i in items]); cc = np.array([i[2]['center_error_m'] for i in items])
    print('속도-중심오차 상관 %.2f | 앞뒤 편향(음수=VE가 GT보다 뒤) 속도 상관 %.2f' % (
        np.corrcoef(sp, cc)[0, 1], np.corrcoef(sp, [i[2]['longitudinal_bias_m'] for i in items])[0, 1]))


if __name__ == '__main__':
    main()
