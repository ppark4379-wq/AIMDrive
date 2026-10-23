#!/usr/bin/env python3
"""Morning summary of a tunnel campaign (ve_full_route_campaign --include-tunnel --gps-blackout-hold --simulate-gps-blackout).
Per trial: result, how long and how far the position was held, and the hold error against the discarded real GPS fix
(logged as simulated_real_gps_xy), split along/lateral with the vehicle heading. Read-only.
Usage: analyze_tunnel_hold.py CAMPAIGN_DIR"""
import json, math, re, sys
from pathlib import Path
import numpy as np

DRIVER = re.compile(r'\[GPS\+VE 주행 로그\]\s*(\S+)/driver\.console\.log')


def main():
    root = Path(sys.argv[1])
    report = json.loads((root/'report.json').read_text()) if (root/'report.json').exists() else {}
    print('캠페인 %s | 옵션: hold=%s latency=%s simulate=%s | 요청 %s회, 완료 %s회' % (
        root.name, report.get('gps_blackout_hold'), report.get('gps_latency_s'), report.get('simulate_gps_blackout'),
        report.get('requested'), len(report.get('trials', []))))
    print('%-3s %-16s %7s | %6s %7s %7s | %-24s %-24s | %s' % ('시험', '결과', '소요s', '유지', '유지s', '거리m', '앞뒤 오차 m (최대/중앙)', '옆 오차 m (최대/중앙)', '비고'))
    all_along, all_lat, tunnel_runs = [], [], 0
    for trial in sorted(root.glob('trial_*')):
        result = json.loads((trial/'result.json').read_text()) if (trial/'result.json').exists() else {}
        text = (trial/'console.log').read_text(errors='replace') if (trial/'console.log').exists() else ''
        found = DRIVER.search(text)
        control = Path(found.group(1))/'driver'/'control.jsonl' if found else None
        hold = []
        if control and control.exists():
            for line in control.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                gps = ((row.get('control_pose_selected_samples') or {}).get('gps')) or {}
                if gps.get('position_mode') == 'dead_reckoning' and row.get('position'):
                    hold.append((row, gps))
        note = ''
        if hold:
            tunnel_runs += 1
            along, lat = [], []
            for row, gps in hold:
                real = gps.get('simulated_real_gps_xy')
                if not real:
                    continue
                p, yaw = row['position'], row['position']['yaw_grid_rad']
                # The discarded real fix was received `age` seconds ago: move it forward so both describe the same instant.
                age = gps.get('simulated_real_gps_age_s') or 0.
                real = (real[0] + p['speed_mps']*age*math.cos(yaw), real[1] + p['speed_mps']*age*math.sin(yaw))
                d = np.array([p['x']-real[0], p['y']-real[1]])
                along.append(d[0]*math.cos(yaw)+d[1]*math.sin(yaw)); lat.append(-d[0]*math.sin(yaw)+d[1]*math.cos(yaw))
            hold_s = max(g.get('hold_s') or 0. for _, g in hold)
            dist = max((g.get('dead_reckoning') or {}).get('distance_m', 0.) for _, g in hold)
            brake = sum(1 for r, _ in hold if (r.get('planned_command') or {}).get('brake', 0) >= .99)
            if along:
                all_along += along; all_lat += lat
                a, l = np.abs(along), np.abs(lat)
                cell = ('%.3f / %.3f' % (a.max(), np.median(a)), '%.3f / %.3f' % (l.max(), np.median(l)))
            else:
                cell = ('(실제 GPS 기록 없음)', '-')
            note = '정지 명령 %d프레임' % brake if brake else ''
            print('%-3s %-16s %7s | %6d %7.1f %7.1f | %-24s %-24s | %s' % (trial.name[-2:], result.get('result', '?'), result.get('elapsed_s', '?'),
                  len(hold), hold_s, dist, cell[0], cell[1], note))
        else:
            reached = 'tunnel' in text.lower()
            print('%-3s %-16s %7s | %6s %7s %7s | %-24s %-24s | %s' % (trial.name[-2:], result.get('result', '?'), result.get('elapsed_s', '?'),
                  '-', '-', '-', '-', '-', '터널 음영 구간 도달 기록 없음'))
    print()
    if all_along:
        a, l = np.abs(all_along), np.abs(all_lat)
        print('터널 유지 구간이 있던 시험 %d회 | 앞뒤 오차 최대 %.3f m (중앙값 %.3f) | 옆 오차 최대 %.3f m (중앙값 %.3f)' % (
            tunnel_runs, a.max(), np.median(a), l.max(), np.median(l)))
        v = np.array(all_along)
        print('  앞뒤 오차 부호 평균 %+.3f m (양수 = 유지 위치가 실제 GPS보다 앞; 실제 GPS 수신 경과시간 보정 후)' % v.mean())
    else:
        print('터널 유지 구간에서 비교할 실제 GPS 기록이 없습니다.')


if __name__ == '__main__':
    main()
