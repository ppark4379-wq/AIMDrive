"""Auditable bounded receipt groups, not a sensor-exposure synchronizer.

All LiDAR anchors remain in the result, including rejects. Limits are an
explicit diagnostic receipt policy, not a validated physical timing tolerance.
"""
from bisect import bisect_right
from collections import Counter

from async_capture.realtime_pairing import CameraTimeline
from async_capture.temporal_quality import inspect_stream

MS = 1_000_000
CAMERAS = tuple('camera:Camera-%d' % i for i in range(1, 5))
LIMITS = {**{k: 150*MS for k in CAMERAS}, 'lidar': 250*MS,
          'ego': 100*MS, 'gnss': 150*MS, 'imu': 100*MS}


def build_receipt_groups(rows, start_ns, end_ns):
    streams = {k: sorted((r for r in rows if r['stream'] == k and
                          start_ns-250*MS <= r['receive_monotonic_ns'] <= end_ns),
                         key=lambda r: (r['receive_monotonic_ns'], r['record_id'])) for k in LIMITS}
    quality = {k: inspect_stream(v, gap_ns=LIMITS[k], start_ns=start_ns, end_ns=end_ns)
               for k, v in streams.items()}
    events = sorted((dict(e, stream=k) for k, q in quality.items() for e in q['events']),
                    key=lambda e: (e['start_ns'], e['end_ns']))
    # Merge suspect intervals so every decision has bounded lookup cost.
    intervals = []
    for event in events:
        if not event.get('blocks_association', True):
            continue
        a, b = event['start_ns'], event['end_ns']
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], b)
        else:
            intervals.append([a, b])
    ends = [b for a, b in intervals]
    cameras = {k: CameraTimeline(streams[k]) for k in CAMERAS}
    receipts = {k: [r['receive_monotonic_ns'] for r in v] for k, v in streams.items()}
    by_id = {r['record_id']: r for v in streams.values() for r in v}
    used, groups = set(), []
    for anchor in streams['lidar']:
        rx = anchor['receive_monotonic_ns']
        if rx < start_ns:
            continue
        deadline = rx+100*MS
        reasons, selected = [], {'lidar': anchor['record_id']}
        if deadline > end_ns:
            reasons.append('decision_window_truncated')
        if quality['lidar']['invalid'].get(anchor['record_id']):
            reasons.append('lidar_timestamp_quarantined')
        if anchor.get('enqueue_monotonic_ns', end_ns+1) > deadline:
            reasons.append('lidar_not_available_by_deadline')
        # Include teacher history and camera look-ahead. Never join across a
        # known gap or source discontinuity, even when nearest receipt is close.
        segment = bisect_right(ends, rx-100*MS)
        if segment < len(intervals) and intervals[segment][0] <= deadline:
            reasons.append('decision_window_intersects_temporal_discontinuity')
        for key in CAMERAS:
            result = cameras[key].select(anchor, max_receive_error_ns=50*MS,
                                         observed_until_ns=end_ns)
            rid = result['record_id']
            if rid is None:
                reasons.append(key+':'+result['reason'])
            elif quality[key]['invalid'].get(rid):
                reasons.append(key+':timestamp_quarantined')
            elif rid in used:
                reasons.append(key+':camera_reuse')
            else:
                selected[key] = rid
        for key, age in (('ego', 50*MS), ('gnss', 100*MS), ('imu', 50*MS)):
            index = bisect_right(receipts[key], rx)-1
            if index < 0:
                reasons.append(key+':no_previous_state')
                continue
            row = streams[key][index]
            first_stamp_rx = quality[key]['stamp_first_receive_ns'][row['record_id']]
            if rx-row['receive_monotonic_ns'] > age:
                reasons.append(key+':stale_state')
            elif rx-first_stamp_rx > age:
                reasons.append(key+':source_stamp_held_too_long')
            elif row.get('enqueue_monotonic_ns', end_ns+1) > deadline:
                reasons.append(key+':not_available_by_deadline')
            elif quality[key]['invalid'].get(row['record_id']):
                reasons.append(key+':timestamp_quarantined')
            else:
                selected[key] = row['record_id']
        if not reasons:
            used.update(selected[k] for k in CAMERAS)
        groups.append(dict(anchor_record_id=anchor['record_id'], receive_monotonic_ns=rx,
            decision_monotonic_ns=deadline, segment_id=segment,
            status='rejected' if reasons else 'receipt_candidate', reasons=reasons,
            selected={k: {name: by_id[rid].get(name) for name in
                ('record_id', 'seq', 'source_timestamp_ns', 'receive_monotonic_ns',
                 'receive_monotonic_raw_ns', 'receive_wall_ns', 'enqueue_monotonic_ns',
                 'payload_sha256')} for k, rid in selected.items()},
            physical_simultaneity_verified=False, training_visible=False))
    accepted = [g for g in groups if not g['reasons']]
    summary = dict(anchors=len(groups), receipt_candidates=len(accepted),
        rejected=len(groups)-len(accepted),
        rejection_reasons=dict(Counter(r for g in groups for r in g['reasons'])),
        temporal_event_counts=dict(Counter(e['kind'] for e in events)),
        quarantined_records={k: len(q['invalid']) for k, q in quality.items()},
        policy=dict(clock='receive_monotonic_ns', wait_ns=100*MS, camera_max_gap_ns=50*MS,
                    stream_gap_limits_ns=LIMITS, unique_camera_records=True,
                    source_timestamp_rewritten=False, physical_tolerance_validated=False),
        training_visible=False, physical_simultaneity_verified=False)
    return summary, groups, quality, events
