#!/usr/bin/env python3
"""Passive Object_topic recording and offline VE footprint/speed diagnostics.

No publishers, actor control, GT feedback, or model inference. LiDAR physical
capture time is unavailable; receipt-time comparisons are approximate.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import signal
import sys
import time
import threading

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / 'VLDrive'), str(REPO / 'MoraiDrive')]


def serialize_gt(msg, received_ns):
    boxes, errors, seen = [], [], set()
    for kind, field, count in (('npc', 'npc_list', 'num_of_npcs'),
                               ('obstacle', 'obstacle_list', 'num_of_obstacle'),
                               ('pedestrian', 'pedestrian_list', 'num_of_pedestrian')):
        objects = getattr(msg, field)
        if len(objects) != getattr(msg, count):
            errors.append('count_mismatch:' + kind)
        for o in objects:
            key = kind + ':' + str(o.unique_id)
            position = [float(getattr(o.position, a)) for a in 'xyz']
            velocity = [float(getattr(o.velocity, a)) for a in 'xyz']
            size = [float(getattr(o.size, a)) for a in 'xyz']
            heading = float(o.heading)
            if (key in seen or not all(math.isfinite(x) for x in position + velocity + size + [heading])
                    or min(size[:2]) <= 0.):
                errors.append('invalid_or_duplicate:' + key)
                continue
            seen.add(key)
            # Live Object_topic vehicles: x=length,y=width (e.g. Sorento 4.80x2.10).
            # Do not inherit size_wlh_m naming from the older dataset converter.
            boxes.append(dict(track_id=key, kind=kind, name=str(o.name), type=int(o.type),
                center_world=position[:2], z_m=position[2], heading_raw_deg=heading,
                length_m=size[0], width_m=size[1], height_m=size[2], raw_size_xyz_m=size,
                velocity_raw_kmh=velocity, speed_mps=math.hypot(*velocity[:2]) / 3.6))
    return dict(schema='ve_gt_observation_v2_native_lwh',
                received_ns=received_ns, source_stamp_ns=msg.header.stamp.to_nsec(),
                source_frame=msg.header.frame_id, boxes=boxes, errors=errors,
                possibly_truncated=len(boxes) >= 20)


def serialize_geometry(response, start_ns, end_ns):
    def xyz(value):
        return [float(getattr(value, k)) for k in 'xyz']
    return dict(start_ns=start_ns, received_ns=end_ns, source_timestamp_raw=int(response.timestamp),
        states=[dict(actor_id=str(s.actor_info.id.value), kind=int(s.actor_info.object_type),
                     position=xyz(s.transform.location), rotation=xyz(s.transform.rotation),
                     bbox_position=xyz(s.bounding_box.location), bbox_rotation=xyz(s.bounding_box.rotation),
                     bbox_extent=xyz(s.bounding_box.extent)) for s in response.states])


def record_geometry(output, target, done):
    # Passive snapshots only. Never create SimulationWorld or set actor state.
    sys.path.insert(0, str(REPO/'MoraiDrive/data_collection/third_party/grpc_inha_univ/src/proto'))
    import grpc
    from morai.actor.actor_pb2_grpc import ActorStub
    from morai.actor.actor_get_pb2 import GetAllActorsFilter
    with grpc.insecure_channel(target) as channel, (output/'gt_geometry.jsonl').open('x', buffering=1) as stream:
        stub = ActorStub(channel)
        request = GetAllActorsFilter(client_key='', vehicle=True, pedestrian=True, obstacle=True)
        while not done[0]:
            begin = time.monotonic_ns()
            try:
                response = stub.GetAllActorsState(request, timeout=.25)
                row = serialize_geometry(response, begin, time.monotonic_ns())
            except Exception as exc:
                row = dict(start_ns=begin, received_ns=time.monotonic_ns(), error=str(exc))
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            time.sleep(max(0., .1-(time.monotonic_ns()-begin)/1e9))


def record(output, geometry_target=None):
    import rospy
    from morai_msgs.msg import ObjectStatusList, EgoVehicleStatus
    output.mkdir(parents=True, exist_ok=True)
    done = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.__setitem__(0, True))
    rospy.init_node('ve_gt_observer', anonymous=True, disable_signals=True)
    geometry_thread = None
    if geometry_target:
        geometry_thread = threading.Thread(target=record_geometry, args=(output, geometry_target, done), daemon=True)
        geometry_thread.start()
    with (output / 'gt_objects.jsonl').open('x', buffering=1) as stream, (output/'gt_ego.jsonl').open('x', buffering=1) as ego_stream:
        def callback(msg):
            received_ns = time.monotonic_ns()
            try:
                row = serialize_gt(msg, received_ns)
            except Exception as exc:
                row = dict(received_ns=received_ns, source_stamp_ns=0,
                           boxes=[], errors=[repr(exc)], possibly_truncated=False)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        sub = rospy.Subscriber('/Object_topic', ObjectStatusList, callback,
                               queue_size=100, tcp_nodelay=True)
        def ego_callback(msg):
            ego_stream.write(json.dumps(dict(received_ns=time.monotonic_ns(),
                source_stamp_ns=msg.header.stamp.to_nsec(),
                position=[float(getattr(msg.position,k)) for k in 'xyz'], heading_deg=float(msg.heading),
                velocity=[float(getattr(msg.velocity,k)) for k in 'xyz']))+'\n')
        ego_sub = rospy.Subscriber('/Ego_topic', EgoVehicleStatus, ego_callback, queue_size=100, tcp_nodelay=True)
        (output / 'gt_observer_ready.json').write_text(json.dumps(dict(
            pid=__import__('os').getpid(), topic='/Object_topic', geometry_target=geometry_target, vehicle_control=False)))
        try:
            while not done[0] and not rospy.is_shutdown():
                time.sleep(.05)
        finally:
            sub.unregister()
            ego_sub.unregister()
            done[0] = True
            if geometry_thread:
                geometry_thread.join(timeout=1.)


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    import numpy as np
    a = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(a):
        return dict(count=0, mean=None, median=None, p95=None, max=None)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def timeline(rows):
    """Keep actual message receipt, reject stale repeated header snapshots."""
    previous, first_seen = None, None
    for row in rows:
        stamp = row['source_stamp_ns']
        if stamp != previous or first_seen is None:
            first_seen = row['received_ns']
        row['source_held_ms'] = (row['received_ns'] - first_seen) / 1e6
        previous = stamp
    return [row['received_ns'] for row in rows]


def nearest(rows, times, target_ns, tolerance_ns=75_000_000):
    i = bisect_left(times, target_ns)
    candidates = [j for j in (i-1, i) if 0 <= j < len(rows)]
    if not candidates:
        return None
    j = min(candidates, key=lambda j: abs(times[j] - target_ns))
    row = rows[j]
    if (abs(times[j] - target_ns) > tolerance_ns or row['errors']
            or row['possibly_truncated'] or row['source_held_ms'] > 100.):
        return None
    return row


def rect(box):
    from utils.evaluation.morai_object_metrics import _rectangle
    angle = math.radians(box['heading_deg'])
    return _rectangle(box['center_world'], [math.cos(angle), math.sin(angle)],
                      box['length_m'], box['width_m'])


def roi(ego):
    from utils.evaluation.morai_object_metrics import _rectangle
    yaw = math.radians(ego['yaw_deg'])
    c, s = math.cos(yaw), math.sin(yaw)
    return _rectangle([ego['x'] + 5*c, ego['y'] + 5*s], [c, s], 50., 50.)


def in_roi(box, area):
    from utils.evaluation.morai_object_metrics import _intersection
    return _intersection(rect(box), area) > 0.


def match_centers(gt, ve, gate=2.):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not gt or not ve:
        return []
    distances = np.array([[math.dist(g['center_world'], v['center_world']) for v in ve] for g in gt])
    cost = np.full((len(gt), len(ve)+len(gt)), 1e6)
    cost[:, :len(ve)] = np.where(distances <= gate, distances, 1e6)
    for i in range(len(gt)):
        cost[i, len(ve)+i] = (len(gt)+1)*gate+1.
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distances[i,j])) for i,j in zip(rows,cols)
            if j < len(ve) and distances[i,j] <= gate]


def score_frame(gt, ve, ego):
    from utils.evaluation.morai_object_metrics import oriented_iou
    pairs = match_centers(gt, ve, gate=2.)
    matched = []
    yaw = math.radians(ego['yaw_deg'])
    for i, j, distance in pairs:
        g, v = gt[i], ve[j]
        dx, dy = [v['center_world'][k] - g['center_world'][k] for k in (0, 1)]
        speed, planning = v.get('speed_mps'), v.get('planning_speed_mps')
        heading = abs((v['heading_deg'] - g['heading_deg'] + 90.) % 180. - 90.)
        matched.append(dict(gt_index=i, ve_index=j, gt_id=g['track_id'], gt_kind=g['kind'],
            gt_name=g['name'], ve_track_id=v['track_id'], motion_state=v['motion_state'],
            gt_speed_mps=g['speed_mps'], raw_speed_mps=speed, planning_speed_mps=planning,
            center_error_m=distance, longitudinal_bias_m=math.cos(yaw)*dx+math.sin(yaw)*dy,
            lateral_bias_m=-math.sin(yaw)*dx+math.cos(yaw)*dy,
            length_error_m=abs(v['length_m']-g['length_m']),
            width_error_m=abs(v['width_m']-g['width_m']), heading_error_deg_mod180=heading,
            iou=oriented_iou(rect(g), rect(v)),
            raw_speed_error_mps=abs(speed-g['speed_mps']) if speed is not None else None,
            raw_speed_bias_mps=speed-g['speed_mps'] if speed is not None else None,
            planning_speed_error_mps=abs(planning-g['speed_mps']) if planning is not None else None))
    return dict(gt_count=len(gt), ve_count=len(ve), matched=matched,
                missed_gt=[i for i in range(len(gt)) if i not in {p[0] for p in pairs}],
                unmatched_ve=[i for i in range(len(ve)) if i not in {p[1] for p in pairs}])


def summarize(frames):
    pairs = [p for row in frames for p in row['score']['matched']]
    gt = sum(r['score']['gt_count'] for r in frames)
    ve = sum(r['score']['ve_count'] for r in frames)
    fields = ('center_error_m', 'longitudinal_bias_m', 'lateral_bias_m', 'length_error_m',
              'width_error_m', 'heading_error_deg_mod180', 'iou', 'raw_speed_error_mps',
              'raw_speed_bias_mps', 'planning_speed_error_mps')
    metrics = {k: stats([p[k] for p in pairs]) for k in fields}
    return dict(frames=len(frames), gt_observations=gt, ve_observations=ve, matched=len(pairs),
        center_recall_2m=len(pairs)/gt if gt else None,
        center_precision_2m=len(pairs)/ve if ve else None,
        missed_gt=gt-len(pairs), unmatched_ve=ve-len(pairs),
        matched_iou_ge_0_3=sum(p['iou'] >= .3 for p in pairs),
        matched_iou_ge_0_5=sum(p['iou'] >= .5 for p in pairs),
        metrics=metrics, moving_gt_speed_error_mps=stats([
            p['raw_speed_error_mps'] for p in pairs if p['gt_speed_mps'] >= .8]),
        stationary_gt_speed_error_mps=stats([
            p['raw_speed_error_mps'] for p in pairs if p['gt_speed_mps'] <= .25]))


def breakdown(frames, category):
    result = {}
    for row in frames:
        paired = {p['gt_index']: p for p in row['score']['matched']}
        for i, b in enumerate(row['gt']):
            distance = math.dist(b['center_world'], [row['ego']['x'], row['ego']['y']])
            key = (b['name'] if category == 'model' else
                   ('0_10m' if distance < 10 else '10_20m' if distance < 20 else '20_30m' if distance < 30 else '30m_plus'))
            group = result.setdefault(key, dict(gt_observations=0, pairs=[]))
            group['gt_observations'] += 1
            if i in paired:
                group['pairs'].append(paired[i])
    for group in result.values():
        pairs = group.pop('pairs')
        group.update(matched=len(pairs), center_recall_2m=len(pairs)/group['gt_observations'],
            center_error_m=stats([p['center_error_m'] for p in pairs]),
            raw_speed_error_mps=stats([p['raw_speed_error_mps'] for p in pairs]),
            iou=stats([p['iou'] for p in pairs]))
    return result


def residual_jitter(frames):
    previous, changes, switches = {}, [], 0
    for row in frames:
        for p in row['score']['matched']:
            g, v = row['gt'][p['gt_index']], row['ve'][p['ve_index']]
            residual = [v['center_world'][i]-g['center_world'][i] for i in (0,1)]
            last = previous.get(g['track_id'])
            if last and 0 < row['target_ns']-last[0] <= 250_000_000:
                changes.append(math.dist(residual,last[1]))
                switches += last[2] != p['ve_track_id']
            previous[g['track_id']] = row['target_ns'], residual, p['ve_track_id']
    return dict(center_residual_step_m=stats(changes), ve_id_switches=switches,
                policy='Matched same GT actor <=250ms apart; subtract GT motion; includes receipt timing noise')


def gt_motion_check(rows):
    """Independent unit check: displacement / GT header delta vs reported km/h."""
    histories, checks = defaultdict(list), defaultdict(list)
    for row in rows:
        if row['errors']:
            continue
        for b in row['boxes']:
            history = histories[b['track_id']]
            if not history or row['source_stamp_ns'] > history[-1][0]:
                history.append((row['source_stamp_ns'],row['received_ns'],b))
    for key, history in histories.items():
        times = [r[0] for r in history]
        for i, (stamp, receipt, b) in enumerate(history):
            j = bisect_left(times, stamp+300_000_000)
            if j >= len(history):
                continue
            after, received, bb = history[j]
            dt = (after-stamp)/1e9
            if not .3 <= dt <= .6 or bb['name'] != b['name']:
                continue
            derived = math.dist(b['center_world'],bb['center_world'])/dt
            raw = math.hypot(*bb['velocity_raw_kmh'][:2])
            if derived > .5 and raw > .5:
                checks[key].append((derived,raw,derived/raw,abs(derived-raw/3.6)))
    return {k:dict(samples=len(v), position_speed_mps=stats([a[0] for a in v]),
        raw_planar_velocity=stats([a[1] for a in v]),
        displacement_to_raw_ratio=stats([a[2] for a in v]),
        kmh_conversion_error_mps=stats([a[3] for a in v])) for k,v in checks.items()}


def render_review(frames, output):
    # All comparable driving frames; standalone HTML, local coordinates only.
    compact = []
    for row in frames:
        ego = row['ego']
        c, s = math.cos(math.radians(ego['yaw_deg'])), math.sin(math.radians(ego['yaw_deg']))
        def points(box):
            return [[250 + 8*(s*(x-ego['x'])-c*(y-ego['y'])),
                     300 - 8*(c*(x-ego['x'])+s*(y-ego['y']))] for x, y in rect(box)['polygon']]
        shapes = ''.join('<polygon points="{}" fill="none" stroke="{}" stroke-width="2"/>'.format(
            ' '.join('{:.2f},{:.2f}'.format(*p) for p in points(b)), color)
            for boxes, color in ((row['gt'], '#16803c'), (row['ve'], '#db3c32')) for b in boxes)
        summary = dict(sequence=row['sequence'], gt_delta_ms=row['gt_delta_ms'], score=row['score'])
        compact.append('<section><svg width="500" height="480" viewBox="0 0 500 480">'
                       '<rect x="50" y="60" width="400" height="400" fill="#f6f7f8" stroke="#999"/>'
                       '<path d="M250 285 L244 304 L256 304 Z" fill="#333"/>' + shapes +
                       '</svg><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></section>')
    (output/'gt_comparison_review.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>VE / Object_topic 비교</title><style>section{display:none;gap:20px}pre{font-size:12px;max-height:80vh;overflow:auto}input{width:70%}</style>'
        '<h2>초록: MORAI Object_topic / 빨강: VE</h2><p>LiDAR 수신 시점에 가까운 GT. 원본 객체 기준점·비동기 센서의 오차 포함.</p>'
        '<button onclick="show(index-1)">이전</button><input type="range" id="slider" min="0" max="'+str(max(0,len(compact)-1))+'" value="0" oninput="show(+this.value)">'
        '<button onclick="show(index+1)">다음</button><span id="label"></span>' + ''.join(compact) +
        '<script>let index=0,rows=document.querySelectorAll("section");function show(i){if(!rows.length)return;rows[index].style.display="none";index=Math.max(0,Math.min(rows.length-1,i));rows[index].style.display="flex";slider.value=index;label.textContent=(index+1)+" / "+rows.length;}show(0);document.onkeydown=e=>{if(e.key==="ArrowRight")show(index+1);if(e.key==="ArrowLeft")show(index-1);};</script>', encoding='utf-8')


def analyze(output):
    from data.src.loaders.morai_map_geometry import validate_map_basis
    from inference.realtime_ve_objects import MAP_BASIS
    geo = validate_map_basis(MAP_BASIS)
    rows = load_rows(output/'gt_objects.jsonl')
    legacy_axis_rows = 0
    for row in rows:
        if not row.get('schema'):
            # First recorder version used the dataset's W/L naming. Preserve
            # raw files, reverse only that known serialization in this analysis.
            legacy_axis_rows += 1
            for b in row['boxes']:
                b['raw_size_xyz_m'] = [b['width_m'], b['length_m'], b['height_m']]
                b['length_m'], b['width_m'] = b['width_m'], b['length_m']
        elif row['schema'] != 've_gt_observation_v2_native_lwh':
            raise ValueError('Unknown GT recording axes schema')
    times = timeline(rows)
    callbacks = {r['ros_stamp_ns']: r for r in load_rows(output/'expert_input.jsonl')}
    packets = [r for r in load_rows(output/'adapter.jsonl')
               if r['delivery_audit']['ros_stamp_ns'] in callbacks]
    for row in rows:
        for b in row['boxes']:
            b['heading_deg'] = b['heading_raw_deg'] + geo.convergence_degrees([*b['center_world'], b['z_m']])
    scored, skipped = {'input_receipt': [], 'expert_receipt': []}, Counter()
    for packet in packets:
        p = packet['ego_pose']
        ego = dict(x=p['x'], y=p['y'], yaw_deg=p['yaw_raw_deg'] + geo.convergence_degrees([p['x'],p['y'],0.]))
        area = roi(ego)
        ve = [b for b in packet['boxes'] if in_roi(b, area)]
        for mode, target in (
            ('input_receipt', packet['audit']['selected_samples']['lidar']['received_ns']),
            ('expert_receipt', callbacks[packet['delivery_audit']['ros_stamp_ns']]['received_ns'])):
            reference = nearest(rows, times, target)
            if reference is None:
                skipped[mode] += 1
                continue
            gt = [b for b in reference['boxes'] if in_roi(b, area)]
            scored[mode].append(dict(sequence=packet['sequence'], target_ns=target,
                gt_delta_ms=(reference['received_ns']-target)/1e6,
                gt_source_stamp_ns=reference['source_stamp_ns'], ego=ego, gt=gt, ve=ve,
                score=score_frame(gt, ve, ego)))
    report = dict(schema='ve_object_topic_comparison_v1', status='COMPLETE' if scored['input_receipt'] else 'NO_COMPARABLE_FRAMES',
        run=str(output), checkpoint_sha256=packets[0]['checkpoint_sha256'] if packets else None,
        gt_messages=len(rows), expert_packets=len(packets), skipped_unpaired=dict(skipped),
        gt_invalid_messages=sum(bool(r['errors']) for r in rows),
        gt_possibly_truncated_messages=sum(r['possibly_truncated'] for r in rows),
        coordinate_policy='native live GT and VE x=length,y=width; native headings true_ENU -> projected_grid; positions unchanged',
        legacy_recorder_axis_rows_corrected_in_analysis=legacy_axis_rows,
        speed_policy='GT hypot(velocity.x,velocity.y)/3.6; VE raw scalar m/s and Expert planning m/s scored separately',
        pairing_policy='nearest GT receipt <=75ms; source header held >100ms excluded; no temporal extrapolation',
        matching_policy='one-to-one 2m center gate before assignment; IoU and regression on these same matches; no class matching',
        roi_policy='positive-area intersection in ego BEV forward [-20,30], lateral [-25,25] m; no visibility/occlusion test',
        limitations=['Receipt time is not physical capture time; GT and sensor delays can differ.',
            'Object_topic reports actor reference positions, not independently certified geometric centers.',
            'Unmatched predictions are relative to this topic; complete world inventory is not certified.',
            'Matched errors exclude misses; always interpret together with recall.',
            'Single scenario observations are correlated, not independent validation frames.'],
        summaries={k:summarize(v) for k,v in scored.items()},
        by_gt_model=breakdown(scored['input_receipt'], 'model'),
        by_gt_range=breakdown(scored['input_receipt'], 'range'),
        temporal_residual=residual_jitter(scored['input_receipt']),
        gt_speed_unit_crosscheck=gt_motion_check(rows),
        timing_alignment_ms={k:stats([abs(r['gt_delta_ms']) for r in v]) for k,v in scored.items()},
        training_updates=0, gt_used_for_driving=False)
    groups = defaultdict(list)
    for row in scored['input_receipt']:
        for b in row['gt']:
            groups[(b['kind'], b['name'])].append(b)
    report['observed_gt_models'] = [dict(kind=k[0], name=k[1], observations=len(v),
        width_m=stats([b['width_m'] for b in v]), length_m=stats([b['length_m'] for b in v]),
        speed_mps=stats([b['speed_mps'] for b in v])) for k,v in sorted(groups.items())]
    (output/'gt_comparison_frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in scored['input_receipt']))
    (output/'gt_comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    render_review(scored['input_receipt'], output)
    print(json.dumps(dict(status=report['status'], report=str(output/'gt_comparison_report.json'))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--record', type=Path)
    group.add_argument('--analyze', type=Path)
    parser.add_argument('--geometry-target')
    args = parser.parse_args()
    record(args.record, args.geometry_target) if args.record else analyze(args.analyze)


if __name__ == '__main__':
    main()
