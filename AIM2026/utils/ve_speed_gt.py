#!/usr/bin/env python3
"""VE object speed vs simulator ground truth for a driven run (evaluation only; nothing feeds back into driving).

record   Poll the simulator's gRPC GetAllActorsState read-only (default 10 Hz) and write gt_actors.jsonl with the
         Windows monotonic receipt clock that Expert/VE also use. Run it beside a drive; stop with Ctrl+C.
analyze  Match the run's VE boxes (inputs.jsonl, after tracking) to GT actors at each frame's sensor time and report
         speed errors of the three VE speeds: speed_mps (VE head), observed_speed_mps (tracker, from positions) and
         planning_speed_mps (what the Expert plans with).

GT speed: vehicles use global_velocity (km/h); pedestrians and obstacles use the change of their position (their
velocity field is empty or a command), as in MoraiDrive/data_collection ros/actor_ground_truth.py.

usage:
  python AIM2026/utils/ve_speed_gt.py record --out AIM2026/outputs/gt/NAME [--target 127.0.0.1:7789] [--hz 10]
  python AIM2026/utils/ve_speed_gt.py analyze --run AIM2026/outputs/expert/RUN --gt AIM2026/outputs/gt/NAME
"""
import argparse
import bisect
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT/'MoraiDrive/data_collection/third_party/grpc_inha_univ/src/proto'
MATCH_M = 2.0             # VE box centre within this of the GT box centre (plus half the GT box's short side)
MOVING_MPS = .8           # GT moving at least this fast ...
STILL_MPS = .25           # ... or standing (at most this fast); in between is not scored as either
DIFF_WINDOW_S = .25       # half-window of the position difference for non-vehicle GT speed


def record(out, target, hz):
    sys.path.insert(0, str(PROTO))
    import grpc
    from morai.actor.actor_pb2_grpc import ActorStub
    from morai.actor.actor_get_pb2 import GetAllActorsFilter
    out.mkdir(parents=True, exist_ok=True)
    request = GetAllActorsFilter(client_key='', vehicle=True, pedestrian=True, obstacle=True)
    period = 1./hz
    rows = errors = 0
    with grpc.insecure_channel(target) as channel, (out/'gt_actors.jsonl').open('a', buffering=1) as stream:
        stub = ActorStub(channel)
        print('recording %s -> %s (Ctrl+C to stop)' % (target, out/'gt_actors.jsonl'), flush=True)
        try:
            while True:
                begin = time.monotonic_ns()
                try:
                    response = stub.GetAllActorsState(request, timeout=.5)
                    end = time.monotonic_ns()
                    actors = []
                    for s in response.states:
                        b, gv = s.bounding_box, s.global_velocity
                        actors.append(dict(id=s.actor_info.id.value, type=int(s.actor_info.object_type),
                                           xy=[b.location.x, b.location.y], yaw_deg=b.rotation.z,
                                           extent=[b.extent.x, b.extent.y], global_velocity_kmh=[gv.x, gv.y]))
                    stream.write(json.dumps(dict(request_ns=begin, received_ns=end, sim_timestamp=int(response.timestamp),
                                                 actors=actors)) + '\n')
                    rows += 1
                except Exception as exc:             # the simulator restarting must not end the recording
                    errors += 1
                    stream.write(json.dumps(dict(request_ns=begin, received_ns=time.monotonic_ns(), error=str(exc)[:200])) + '\n')
                time.sleep(max(0., period-(time.monotonic_ns()-begin)/1e9))
        except KeyboardInterrupt:
            pass
    print('stopped: %d snapshots, %d errors' % (rows, errors))


def _load_gt(path):
    """Per actor: sorted times [s] and rows (x, y, short side, vehicle speed or None, type)."""
    tracks = {}
    for line in path.open(encoding='utf-8'):
        try:
            row = json.loads(line)
        except ValueError:
            break
        if 'actors' not in row:
            continue
        t = .5*(row['request_ns']+row['received_ns'])/1e9
        for a in row['actors']:
            if a['id'] in ('Ego', '-1', '0'):
                key = 'Ego'
            else:
                key = a['id']
            vehicle = a['type'] == 1
            speed = math.hypot(*a['global_velocity_kmh'])/3.6 if vehicle else None
            tracks.setdefault(key, ([], []))
            tracks[key][0].append(t)
            tracks[key][1].append((a['xy'][0], a['xy'][1], 2*min(a['extent']), speed, a['type']))
    return tracks


def _gt_at(times, rows, t):
    """Interpolated position and GT speed of one actor at time t (None outside its record)."""
    i = bisect.bisect_left(times, t)
    if i == 0 or i >= len(times) or times[i]-times[i-1] > .5:
        return None
    w = (t-times[i-1])/(times[i]-times[i-1])
    a, b = rows[i-1], rows[i]
    x, y = a[0]+w*(b[0]-a[0]), a[1]+w*(b[1]-a[1])
    if a[3] is not None and b[3] is not None:
        speed, direction = a[3]+w*(b[3]-a[3]), None
    else:                                             # position difference over +-DIFF_WINDOW_S
        j0 = bisect.bisect_left(times, t-DIFF_WINDOW_S)
        j1 = min(len(times)-1, bisect.bisect_right(times, t+DIFF_WINDOW_S)-1)
        if j1 <= j0 or times[j1]-times[j0] < .1:
            return None
        dx, dy = rows[j1][0]-rows[j0][0], rows[j1][1]-rows[j0][1]
        speed = math.hypot(dx, dy)/(times[j1]-times[j0])
        direction = math.atan2(dy, dx) if speed >= MOVING_MPS else None
    return x, y, a[2], speed, direction, a[4]


def _stats(values):
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: v[min(len(v)-1, int(q*len(v)))]
    return dict(n=len(v), mean=round(sum(v)/len(v), 3), p50=round(pick(.5), 3), p90=round(pick(.9), 3),
                p95=round(pick(.95), 3), max=round(v[-1], 3))


def analyze(run, gt_dir, offset_s):
    tracks = _load_gt(gt_dir/'gt_actors.jsonl')
    kinds = {1: 'vehicle', 2: 'pedestrian', 3: 'obstacle'}
    ego_err, pairs = [], []
    for line in (run/'inputs.jsonl').open(encoding='utf-8'):
        try:
            p = json.loads(line)
        except ValueError:
            break
        if not p.get('ready') or not p.get('boxes'):
            continue
        t = p['sensor_received_ns']/1e9+offset_s
        if 'Ego' in tracks and p.get('ego_pose'):
            ego = _gt_at(*tracks['Ego'], t)
            if ego:
                ego_err.append(math.hypot(ego[0]-p['ego_pose']['x'], ego[1]-p['ego_pose']['y']))
        gts = [(k, g) for k, g in ((k, _gt_at(*v, t)) for k, v in tracks.items() if k != 'Ego') if g]
        used = set()
        for box in sorted(p['boxes'], key=lambda b: b.get('track_id') or 0):
            if box.get('planning_speed_source') in ('static_memory', 'static_map'):
                continue
            bx, by = box['center_world']
            best = min(((math.hypot(g[0]-bx, g[1]-by), k, g) for k, g in gts if k not in used), default=None)
            if best is None or best[0] > MATCH_M+.5*best[2][2]:
                continue
            distance, key, g = best
            used.add(key)
            pairs.append(dict(t=round(t, 3), gt_id=key, kind=kinds.get(g[5], str(g[5])), gt_speed=g[3],
                              gt_direction=g[4], track_id=box.get('track_id'), match_m=round(distance, 2),
                              range_m=round(math.hypot(bx-p['ego_pose']['x'], by-p['ego_pose']['y']), 1) if p.get('ego_pose') else None,
                              motion_state=box.get('motion_state'), speed_mps=box.get('speed_mps'),
                              observed_speed_mps=box.get('observed_speed_mps'),
                              planning_speed_mps=box.get('planning_speed_mps'),
                              planning_source=box.get('planning_speed_source'),
                              planning_heading_deg=box.get('planning_heading_deg')))
    report = dict(run=run.name, gt=str(gt_dir), clock_offset_s=offset_s, matched=len(pairs),
                  ego_position_difference_m=_stats(ego_err),
                  note='ego difference includes the GPS antenna vs GT box-centre offset and the GPS delay', groups={})
    for kind in sorted({q['kind'] for q in pairs}):
        for motion, test in (('moving', lambda s: s >= MOVING_MPS), ('standing', lambda s: s <= STILL_MPS)):
            group = [q for q in pairs if q['kind'] == kind and test(q['gt_speed'])]
            if not group:
                continue
            entry = dict(observations=len(group), gt_speed=_stats([q['gt_speed'] for q in group]),
                         motion_state={s: sum(1 for q in group if q['motion_state'] == s)
                                       for s in sorted({q['motion_state'] for q in group})})
            for field in ('speed_mps', 'observed_speed_mps', 'planning_speed_mps'):
                valid = [q for q in group if q[field] is not None and math.isfinite(q[field])]
                entry[field] = dict(coverage=round(len(valid)/len(group), 3),
                                    abs_error=_stats([abs(q[field]-q['gt_speed']) for q in valid]),
                                    bias=round(sum(q[field]-q['gt_speed'] for q in valid)/len(valid), 3) if valid else None)
            directions = [abs((math.radians(q['planning_heading_deg'])-q['gt_direction']+math.pi) % (2*math.pi)-math.pi)
                          for q in group if q['gt_direction'] is not None and q['planning_heading_deg'] is not None
                          and (q['planning_speed_mps'] or 0.) >= MOVING_MPS]
            if directions:
                entry['direction_error_deg'] = _stats([math.degrees(d) for d in directions])
            report['groups'][kind+'/'+motion] = entry
    (gt_dir/'speed_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    with (gt_dir/'speed_pairs.jsonl').open('w', encoding='utf-8') as stream:
        for q in pairs:
            stream.write(json.dumps(q) + '\n')
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('record')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--target', default='127.0.0.1:7789')
    r.add_argument('--hz', type=float, default=10.)
    a = sub.add_parser('analyze')
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--gt', type=Path, required=True)
    a.add_argument('--clock-offset-s', type=float, default=0., help='added to VE sensor times before matching')
    args = parser.parse_args()
    if args.command == 'record':
        record(args.out, args.target, args.hz)
    else:
        analyze(args.run, args.gt, args.clock_offset_s)


if __name__ == '__main__':
    main()
