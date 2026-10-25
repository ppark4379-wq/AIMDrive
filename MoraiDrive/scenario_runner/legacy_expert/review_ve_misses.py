#!/usr/bin/env python3
"""Render the frames where VE missed a GT object inside its range, as ego-centred bird's-eye pictures.

GT boxes (green), VE boxes (red), missed GT (thick orange), ego (blue). Reads bbox_reference_frames.jsonl and
gt_ego.jsonl; writes one self-contained HTML (inline SVG). Read-only.
Usage: review_ve_misses.py OUT.html RUN_DIR [RUN_DIR ...] [--min-speed 50] [--max-frames 60]
"""
import argparse
import bisect
import html
import json
import math
from pathlib import Path

import numpy as np

sys_path_fix = None
try:
    from analyze_ve_by_speed import ego_speed_table
except ImportError:  # run from another directory
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_ve_by_speed import ego_speed_table

RANGE = dict(front=28., back=15., side=15.)
SCALE = 9.  # pixels per metre


def corners(center, heading_deg, length, width):
    a = math.radians(heading_deg); c, s = math.cos(a), math.sin(a)
    return [(center[0] + dx*c - dy*s, center[1] + dx*s + dy*c)
            for dx, dy in ((length/2, width/2), (length/2, -width/2), (-length/2, -width/2), (-length/2, width/2))]


def to_ego(point, ego, yaw):
    dx, dy = point[0]-ego['x'], point[1]-ego['y']
    return dx*math.cos(yaw)+dy*math.sin(yaw), -dx*math.sin(yaw)+dy*math.cos(yaw)   # forward, left


def svg_frame(frame, ego, yaw, missed):
    w, h = 2*RANGE['side']*SCALE, (RANGE['front']+RANGE['back'])*SCALE
    def px(f, l):    # forward up, left to the left
        return (RANGE['side']-l)*SCALE, (RANGE['front']-f)*SCALE
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" style="background:#f6f6f6;border:1px solid #bbb">' % (w, h)]
    for f in range(-10, 30, 10):
        x0, y0 = px(f, -RANGE['side']); x1, _ = px(f, RANGE['side'])
        parts.append('<line x1="%.0f" y1="%.0f" x2="%.0f" y2="%.0f" stroke="#ddd"/><text x="2" y="%.0f" font-size="9" fill="#999">%d m</text>' % (x0, y0, x1, y0, y0-2, f))
    def poly(box, color, width, fill='none', dash=''):
        pts = [px(*to_ego(p, ego, yaw)) for p in corners(box['center_world'], box.get('heading_deg', box.get('heading_raw_deg', 0.)), box['length_m'], box['width_m'])]
        parts.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%s" %s/>' % (' '.join('%.1f,%.1f' % p for p in pts), fill, color, width, dash))
    for i, g in enumerate(frame['gt']):
        poly(g, '#e07b00' if i in missed else '#2a9d3a', 3 if i in missed else 1.5, 'rgba(224,123,0,.25)' if i in missed else 'none')
    for v in frame['ve']:
        poly(v, '#d62828', 1.5, dash='stroke-dasharray="4 2"')
    ex, ey = px(0, 0)
    parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#1d4ed8" opacity=".7"/>' % (ex-1.0*SCALE, ey-0.9*SCALE, 2.0*SCALE, 4.6*SCALE) if False else
                 '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#1d4ed8" opacity=".7"/>' % (ex-1.0*SCALE, ey-3.3*SCALE, 2.0*SCALE, 4.6*SCALE))
    parts.append('</svg>')
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path); parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--min-speed', type=float, default=50.); parser.add_argument('--max-frames', type=int, default=60)
    args = parser.parse_args()
    items = []
    for run in args.runs:
        times, speeds = ego_speed_table(run)
        for line in (run/'bbox_reference_frames.jsonl').read_text().splitlines():
            r = json.loads(line); t = r['gt_source_stamp_ns']/1e9
            k = min(max(bisect.bisect_left(times, t), 0), len(times)-1)
            if abs(times[k]-t) > .5 or not args.min_speed <= speeds[k] < 200: continue
            ego = r['ego']; yaw = math.radians(ego['yaw_deg'])
            matched = {m['gt_index'] for m in r['score']['matched']}
            missed = set()
            for gi, g in enumerate(r['gt']):
                f, l = to_ego(g['center_world'], ego, yaw)
                if -RANGE['back'] <= f <= RANGE['front'] and abs(l) < RANGE['side'] and gi not in matched:
                    missed.add(gi)
            if missed:
                items.append((run.name, r, ego, yaw, missed, speeds[k]))
    # one frame per ~0.5 s so the page stays readable
    picked, last = [], {}
    for it in items:
        key = it[0]; t = it[1]['gt_source_stamp_ns']/1e9
        if key in last and t-last[key] < 0.5: continue
        last[key] = t; picked.append(it)
    step = max(1, len(picked)//args.max_frames)
    picked = picked[::step][:args.max_frames]
    cards = []
    for name, r, ego, yaw, missed, speed in picked:
        info = []
        for gi in sorted(missed):
            g = r['gt'][gi]; f, l = to_ego(g['center_world'], ego, yaw)
            near = min((math.dist(v['center_world'], g['center_world']) for v in r['ve']), default=None)
            info.append('%s: 앞%+.0f m 좌%+.0f m, NPC %.0f km/h, 가장 가까운 VE 박스 %s' % (
                html.escape(str(g.get('name', ''))), f, l, (g.get('speed_mps') or 0)*3.6, '없음' if near is None else '%.1f m' % near))
        cards.append('<div class="c"><div class="t">%s seq %s | 자차 %.0f km/h | VE 박스 %d개 / GT %d개</div>%s<div class="i">%s</div></div>' % (
            name[-6:], r['sequence'], speed, len(r['ve']), len(r['gt']), svg_frame(r, ego, yaw, missed), '<br>'.join(info)))
    args.output.write_text('<!doctype html><meta charset="utf-8"><title>VE 놓침 프레임</title><style>body{font-family:sans-serif;margin:14px}'
        '.c{display:inline-block;margin:6px;vertical-align:top}.t{font-size:12px;font-weight:600}.i{font-size:11px;color:#444;max-width:270px}</style>'
        '<h3>VE가 범위 안에서 놓친 GT (자차 %.0f km/h 이상) — %d장면 중 %d장 표시</h3>'
        '<p>초록=GT, <span style="color:#d62828">빨강 점선=VE</span>, <b style="color:#e07b00">주황 굵게=놓친 GT</b>, 파랑=자차(위쪽이 전방, 왼쪽이 좌측)</p>%s' % (
            args.min_speed, len(items), len(picked), ''.join(cards)))
    print('놓친 장면 %d개 중 %d장 표시 -> %s' % (len(items), len(picked), args.output))


if __name__ == '__main__':
    main()
