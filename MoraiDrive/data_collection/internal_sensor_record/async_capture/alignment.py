"""Local raw timing inspection. No NAS, model imports, labels, or time rewriting.

The index refers to original serialized bytes. A receive-causal association is
an observation policy, not a claim about simultaneous sensor exposure.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from async_capture.journal import (MAGIC, PREFIX, MAX_PAYLOAD, digest, json_bytes,
                                   verify_run, write_json)
from async_capture.live import header_metadata


REQUIRED = ('camera:Camera-1', 'camera:Camera-2', 'camera:Camera-3',
            'camera:Camera-4', 'lidar', 'ego', 'gnss', 'imu')


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def stream_key(meta):
    return meta['role'] + (':' + str(meta['frame_id']) if meta['role'] == 'camera' else '')


def decode(meta, payload, classes):
    import roslib.message
    typ = meta['ros_type']
    if typ not in classes:
        classes[typ] = roslib.message.get_message_class(typ)
    cls = classes[typ]
    if cls is None or cls._md5sum != meta['md5sum']:
        raise ValueError('ROS definition mismatch: ' + typ)
    if header_metadata(payload, cls._has_header) != {
            k: meta.get(k) for k in ('seq', 'source_timestamp_ns', 'frame_id')}:
        raise ValueError('Raw header differs from journal')
    return cls().deserialize(payload)


def recorded_classes(manifest):
    # Interpret the exact schema captured in the run, independent of whichever
    # morai_msgs version happens to be installed in the current shell.
    from genpy.dynamic import generate_dynamic
    classes = {}
    definitions=manifest.get('extra',{}).get('topic_state',{}).get('definitions',{})
    for typ,definition in definitions.items():
        cls=generate_dynamic(typ,definition['definition'])[typ]
        if cls._md5sum!=definition['md5sum']:
            raise ValueError('Captured ROS definition hash differs: '+typ)
        classes[typ]=cls
    return classes


def lowdimensional(meta, payload, classes):
    role = meta['role']
    if role not in ('ego', 'gnss', 'imu', 'sync_mode', 'traffic_light', 'collision', 'clock'):
        return None
    msg = decode(meta, payload, classes)
    if role == 'clock':
        return dict(clock_ns=msg.clock.to_nsec())
    if role == 'ego':
        return dict(position=[msg.position.x, msg.position.y, msg.position.z],
                    velocity=[msg.velocity.x, msg.velocity.y, msg.velocity.z],
                    speed=math.hypot(msg.velocity.x, msg.velocity.y), yaw=math.radians(msg.heading))
    if role == 'gnss':
        return dict(latitude=msg.latitude, longitude=msg.longitude, altitude=msg.altitude)
    if role == 'imu':
        q = msg.orientation
        return dict(yaw=math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)),
                    angular_velocity=[msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                    acceleration=[msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    if role == 'sync_mode':
        return dict(status=bool(msg.status), frame=msg.frame)
    if role == 'traffic_light':
        return dict(signal_id=str(msg.trafficLightIndex), status=int(msg.trafficLightStatus))
    return dict(objects=len(msg.collision_object), objects_a=len(msg.collision_objecta))


def build_index(raw_run, output, *, max_active_records=1000000, diagnostic_failed_run=False):
    """Hash original chunks during the SAME sequential pass used to index them.

    Output must be a newly created diagnostic directory. A failed run has no
    INDEX_COMPLETE seal; existing raw files are always opened read-only.
    Explicit diagnostic_failed_run accepts sealed failed captures for analysis
    only; the index retains raw failure status and never grants training access.
    """
    started = time.monotonic()
    raw_run, output = Path(raw_run), Path(output)
    manifest, session = verify_run(raw_run, payloads=False, require_success=not diagnostic_failed_run)
    identity = digest(raw_run / 'raw_manifest.json')
    extra = manifest.get('extra', {})
    start, end = extra.get('recording_start_monotonic_ns'), extra.get('recording_end_monotonic_ns')
    if start is None or end is None or end <= start or not session.get('boot_id'):
        raise ValueError('A sealed active window and receive clock domain are required')
    output.mkdir(parents=True, exist_ok=False)
    active, classes, counts = [], recorded_classes(manifest), Counter()
    expected, total_bytes = 0, 0
    index_hash = hashlib.sha256()
    with (output / 'record_index.jsonl').open('xb') as out:
        for chunk in manifest['chunks']:
            path = raw_run / chunk['path']
            before = path.stat()
            h, count = hashlib.sha256(), 0
            with path.open('rb') as f:
                def read(n):
                    b = f.read(n)
                    if len(b) != n:
                        raise ValueError('Truncated raw record')
                    h.update(b)
                    return b
                if read(len(MAGIC)) != MAGIC:
                    raise ValueError('Invalid raw magic')
                while f.tell() < chunk['bytes']:
                    nm, nb = PREFIX.unpack(read(PREFIX.size))
                    if not 0 < nm <= 1024**2 or nb > MAX_PAYLOAD:
                        raise ValueError('Invalid record length')
                    meta = json.loads(read(nm))
                    offset = f.tell()
                    payload = read(nb)
                    if meta['record_id'] != expected:
                        raise ValueError('Raw record sequence mismatch')
                    row = dict(meta, chunk_path=chunk['path'], payload_offset=offset,
                               payload_bytes=nb, stream=stream_key(meta))
                    encoded = json_bytes(row) + b'\n'
                    out.write(encoded)
                    index_hash.update(encoded)
                    counts[row['stream']] += 1
                    # Keep a 250 ms preceding context for time-varying streams.
                    # Actor model identities are immutable only within a tracked
                    # incarnation and may have been queried during warmup, so
                    # retain their complete pre-end provenance as sparse context.
                    active_context = (start-250000000 <= meta['receive_monotonic_ns'] <= end
                        or (meta.get('role') == 'actor_identity'
                            and meta['receive_monotonic_ns'] <= end))
                    if active_context:
                        if len(active) >= max_active_records:
                            raise ValueError('Active index memory limit exceeded; use shorter runs')
                        row['payload_sha256'] = hashlib.sha256(payload).hexdigest()
                        row['signal'] = lowdimensional(meta, payload, classes)
                        active.append(row)
                    expected += 1
                    count += 1
                if f.read(1):
                    raise ValueError('Unsealed trailing bytes')
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Raw changed during indexing')
            if h.hexdigest() != chunk['sha256'] or count != chunk['records']:
                raise ValueError('Raw chunk hash/count mismatch')
            total_bytes += chunk['bytes']
        out.flush()
        import os
        os.fsync(out.fileno())
    # Recheck small sealed identity; a second full payload pass is unnecessary.
    verify_run(raw_run, payloads=False, require_success=not diagnostic_failed_run)
    if digest(raw_run / 'raw_manifest.json') != identity or expected != manifest['statistics']['records']:
        raise ValueError('Raw identity/count changed')
    info = dict(schema='morai_raw_alignment_index_v1', raw_run_id=manifest['run_id'],
                raw_status=manifest['status'], diagnostic_failed_run=diagnostic_failed_run,
                raw_manifest_sha256=identity, boot_id=session['boot_id'],
                raw_path=str(raw_run.resolve()), active_start_ns=start, active_end_ns=end,
                records=expected, bytes_verified=total_bytes, streams=dict(counts),
                index_sha256=index_hash.hexdigest(), training_visible=False,
                physical_simultaneity_verified=False, seconds=time.monotonic()-started)
    write_json(output / 'active_signals.json', active)
    info['active_signals_sha256'] = digest(output / 'active_signals.json')
    write_json(output / 'INDEX_COMPLETE.json', info)
    return info, active, session


class PayloadReader:
    """Random access to indexed originals, with per-record content verification."""
    def __init__(self, raw_run):
        self.root, self.files = Path(raw_run), {}

    def read(self, row):
        name = row['chunk_path']
        if Path(name).parent != Path('chunks') or Path(name).suffix != '.raw':
            raise ValueError('Unexpected chunk path')
        if name not in self.files:
            self.files[name] = (self.root / name).open('rb')
        f = self.files[name]
        f.seek(row['payload_offset'])
        b = f.read(row['payload_bytes'])
        if hashlib.sha256(b).hexdigest() != row['payload_sha256']:
            raise ValueError('Raw record mutated after indexing')
        return b

    def close(self):
        for f in self.files.values():
            f.close()


def group_streams(rows):
    result = defaultdict(list)
    for row in rows:
        result[row['stream']].append(row)
    for stream in result.values():
        stream.sort(key=lambda r: (r['receive_monotonic_ns'], r['record_id']))
    return result


def latest_causal(rows, anchor, *, admitted=False, times=None):
    # Rows are receive-sorted. gRPC results must also be available at the anchor.
    t = anchor['receive_monotonic_ns']
    times = times if times is not None else [r['receive_monotonic_ns'] for r in rows]
    for i in range(bisect_right(times,t)-1,-1,-1):
        row=rows[i]
        if max(row['receive_monotonic_ns'], row.get('available_monotonic_ns', 0)) > t:
            continue
        if admitted and row['record_id'] >= anchor['record_id']:
            continue
        return row
    return None


def pose_holds(ego):
    holds, first = [], 0
    for j in range(1, len(ego)+1):
        if j < len(ego) and ego[j]['signal']['position'] == ego[first]['signal']['position']:
            continue
        a, b = ego[first], ego[j-1]
        duration = (b['receive_monotonic_ns']-a['receive_monotonic_ns'])/1e9
        if duration >= .1 and min(r['signal']['speed'] for r in ego[first:j]) > 1:
            holds.append(dict(start_ns=a['receive_monotonic_ns'], end_ns=b['receive_monotonic_ns'],
                              duration_s=duration, first_record=a['record_id'], last_record=b['record_id'],
                              source_elapsed_s=(b['source_timestamp_ns']-a['source_timestamp_ns'])/1e9,
                              reported_speed_mps=a['signal']['speed']))
        first = j
    return holds


def timing_report(rows, info, required=REQUIRED):
    streams = group_streams(rows)
    receive_times={k:[r['receive_monotonic_ns'] for r in rs] for k,rs in streams.items()}
    start, end = info['active_start_ns'], info['active_end_ns']
    report = dict(schema='morai_receive_alignment_quality_v1', raw_run_id=info['raw_run_id'],
                  raw_manifest_sha256=info['raw_manifest_sha256'], active_duration_s=(end-start)/1e9,
                  training_visible=False, physical_simultaneity_verified=False,
                  source_timestamps_modified=False, streams={}, associations={},
                  clock_fit_scope='Per-stream rate/jitter diagnostic; NOT absolute latency or a shared exposure clock')
    for name, all_rows in streams.items():
        rs = [r for r in all_rows if start <= r['receive_monotonic_ns'] <= end]
        if not rs:
            continue
        rx = np.array([r['receive_monotonic_ns'] for r in rs], dtype=np.int64)
        result = dict(count=len(rs), receive_period_ms=distribution(np.diff(rx)/1e6),
                      receive_rate_hz=(len(rs)-1)/((int(rx[-1])-int(rx[0]))/1e9) if len(rs)>1 and rx[-1]>rx[0] else None,
                      enqueue_delay_ms=distribution([(r['enqueue_monotonic_ns']-r['receive_monotonic_ns'])/1e6 for r in rs]),
                      writer_queue_ms=distribution([r['writer_queue_wait_ns']/1e6 for r in rs]))
        original = sorted(rs, key=lambda r: r['record_id'])
        result['journal_receive_inversions'] = sum(b['receive_monotonic_ns']<a['receive_monotonic_ns'] for a,b in zip(original,original[1:]))
        wall=np.array([r['receive_wall_ns'] for r in rs],dtype=np.int64) if all('receive_wall_ns' in r for r in rs) else None
        if wall is not None:
            changes=np.diff(wall)-np.diff(rx)
            result['wall_backwards']=int((np.diff(wall)<0).sum())
            result['wall_clock_steps']=[dict(before_record_id=rs[i]['record_id'],after_record_id=rs[i+1]['record_id'],
                before_receive_ns=int(rx[i]),after_receive_ns=int(rx[i+1]),wall_offset_change_ms=float(changes[i]/1e6))
                for i in np.flatnonzero(np.abs(changes)>50000000)]
        if all(r.get('source_timestamp_ns') is not None for r in rs) and len(rs)>2:
            src = np.array([r['source_timestamp_ns'] for r in rs], dtype=np.int64)
            result.update(source_period_ms=distribution(np.diff(src)/1e6),
                          source_backwards=int((np.diff(src)<0).sum()), source_repeats=int((np.diff(src)==0).sum()))
            result['source_repeats_with_changed_decoded_state']=sum(
                a['source_timestamp_ns']==b['source_timestamp_ns'] and a.get('signal')!=b.get('signal')
                for a,b in zip(rs,rs[1:]))
            x, y = (src-src[0])/1e9, (rx-rx[0])/1e9
            if np.ptp(x)>0:
                slope, intercept = np.polyfit(x, y, 1)
                result.update(receive_seconds_per_source_second=float(slope),
                              affine_residual_ms=distribution((y-(slope*x+intercept))*1000))
        report['streams'][name] = result
    report['clock_findings']=dict(
        wall_clock_steps=report['streams'].get('ego',{}).get('wall_clock_steps',[]),
        source_receive_rate_difference_streams=[k for k,v in report['streams'].items()
            if abs(v.get('receive_seconds_per_source_second',1)-1)>.005],
        action='Keep native source and receive monotonic clocks separate. Do not use wall subtraction for alignment or latency.')
    holds = pose_holds([r for r in streams.get('ego', []) if start<=r['receive_monotonic_ns']<=end])
    report['moving_pose_holds'] = holds
    anchors, ages, deltas, skew = [], defaultdict(list), Counter(), []
    candidate_count, compatible = 0, 0
    for anchor in streams.get('lidar', []):
        t = anchor['receive_monotonic_ns']
        if not start <= t <= end:
            continue
        selection, reasons, tight = {}, [], True
        for name in required:
            row = anchor if name == 'lidar' else latest_causal(streams.get(name, []), anchor, times=receive_times.get(name,[]))
            legacy = anchor if name == 'lidar' else latest_causal(streams.get(name, []), anchor, admitted=True,times=receive_times.get(name,[]))
            if row is None:
                reasons.append('missing:'+name)
                tight = False
                continue
            age = (t-row['receive_monotonic_ns'])/1e6
            selection[name] = dict(record_id=row['record_id'], receive_age_ms=age,
                                   source_timestamp_ns=row.get('source_timestamp_ns'),
                                   receive_monotonic_ns=row['receive_monotonic_ns'],
                                   admitted_before_lidar=row['record_id']<=anchor['record_id'])
            ages[name].append(age)
            deltas[name] += int(legacy is None or legacy['record_id']!=row['record_id'])
            if age > 250:
                reasons.append('receive_age_over_250ms:'+name)
            limit = 25 if name=='ego' else 75 if name=='gnss' else 50
            tight &= age<=limit
        in_hold = any(h['start_ns']<=t<=h['end_ns'] for h in holds)
        if in_hold:
            reasons.append('moving_pose_hold')
        elif any(h['start_ns']<=v['receive_monotonic_ns']<=h['end_ns'] for h in holds for v in selection.values()):
            reasons.append('selected_sensor_touches_moving_pose_hold')
        cam = [v['receive_monotonic_ns'] for k,v in selection.items() if k.startswith('camera:')]
        cam_skew = (max(cam)-min(cam))/1e6 if len(cam)==4 else None
        if cam_skew is not None:
            skew.append(cam_skew)
        compatible += int(not reasons)
        candidate_count += int(tight and not reasons)
        anchors.append(dict(lidar_record_id=anchor['record_id'], receive_monotonic_ns=t,
                            selection=selection, reasons=reasons, camera_receive_spread_ms=cam_skew,
                            strict_receive_candidate=bool(tight and not reasons),
                            physical_simultaneity_verified=False, training_visible=False))
    report['associations'] = dict(lidar_anchors=len(anchors), complete_receive_age_250ms_without_hold=compatible,
        strict_receive_candidate_count=candidate_count, strict_thresholds_ms=dict(camera=50, ego=25, imu=50, gnss=75),
        threshold_scope='Provisional receive-age screening only; not exposure tolerance',
        receive_age_ms={k:distribution(v) for k,v in ages.items()}, camera_receive_spread_ms=distribution(skew),
        selections_changed_vs_journal_order=dict(deltas))
    return report, anchors
