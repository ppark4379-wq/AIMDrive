"""Bounded receipt groups for explicitly observed internal sensor scheduling.

Grouping uses receipt monotonic time, never equality of unrelated headers.
Completion is later than the LiDAR anchor. This is a delivery contract, not a
proof of simultaneous exposure. Selection never consults scene/teacher labels.
"""
from collections import Counter

REQUIRED = tuple('camera:Camera-'+str(i) for i in range(1,5))+('lidar','gnss','imu')


def packet_groups(rows, *, end_ns, quiet_ns=40_000_000, max_span_ns=80_000_000,
                  wait_for_incomplete=True):
    if not 1_000_000 <= quiet_ns <= 100_000_000 or not quiet_ns <= max_span_ns <= 200_000_000:
        raise ValueError('Invalid bounded grouping window')
    packets=sorted((r for r in rows if r['stream'] in REQUIRED),
                   key=lambda r:(r['receive_monotonic_ns'],r['record_id']))
    if len(packets)>1_000_000:raise ValueError('Packet metadata memory limit exceeded')
    ids=[r['record_id'] for r in packets]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate raw record identity')
    groups=[];current=[];previous={};clock_failed=False
    def finish():
        if not current:return
        count=Counter(r['stream'] for r in current);reasons=[]
        first=current[0]['receive_monotonic_ns'];last=current[-1]['receive_monotonic_ns']
        for name in REQUIRED:
            if count[name]!=1:reasons.append(('missing:' if not count[name] else 'duplicate:')+name)
        if last-first>max_span_ns:reasons.append('delivery_span_exceeded')
        available=max(max(r['receive_monotonic_ns'],r.get('available_monotonic_ns',0),
                          r.get('enqueue_monotonic_ns',0)) for r in current)
        decision=max(last+quiet_ns,available)
        if wait_for_incomplete and any(count[name]!=1 for name in REQUIRED):
            decision=max(decision,first+max_span_ns)
        if available>last+quiet_ns:reasons.append('callback_admission_late')
        if decision>end_ns:reasons.append('unclosed_tail')
        if clock_failed:reasons.append('source_clock_missing_or_nonincreasing')
        selection={r['stream']:r for r in current} if not reasons else {}
        lidar=[r for r in current if r['stream']=='lidar']
        groups.append(dict(group_index=len(groups),status='rejected' if reasons else 'receipt_group_candidate',
            reasons=reasons,record_ids=[r['record_id'] for r in current],counts=dict(count),selection=selection,
            lidar_record_ids=[r['record_id'] for r in lidar],
            anchor_receive_monotonic_ns=lidar[0]['receive_monotonic_ns'] if len(lidar)==1 else None,
            first_receive_monotonic_ns=first,last_receive_monotonic_ns=last,
            available_monotonic_ns=available,decision_monotonic_ns=decision,
            delivery_span_ns=last-first,physical_simultaneity_verified=False,
            teacher_time_alignment_verified=False,training_visible=False))
    for row in packets:
        t=row['receive_monotonic_ns']
        if current and t-current[-1]['receive_monotonic_ns']>=quiet_ns:
            count=Counter(r['stream'] for r in current)
            complete=all(count[name]==1 for name in REQUIRED)
            # Quiet is sufficient to close a complete bundle, not to declare
            # a still-missing sensor before the existing span budget expires.
            if not wait_for_incomplete or complete or t-current[0]['receive_monotonic_ns']>max_span_ns:
                finish();current=[]
        source=row.get('source_timestamp_ns');name=row['stream']
        if source is None or (name in previous and source<=previous[name]):clock_failed=True
        if source is not None:previous[name]=source
        current.append(row)
    finish()
    return groups


def annotate_internal_cycles(groups, rows, *, expected_step_ns=100_000_000):
    """Check the observed internal10 state-clock contract without dating images.

    Entire missing cycles have no received group. Keep those gaps explicit so
    a consumer cannot mistake adjacent array entries for consecutive cycles.
    Equality of LiDAR/GPS/IMU stamps is a check, not an exposure proof.
    """
    by_id = {r['record_id']: r for r in rows}
    lidar = sorted((r for r in rows if r['stream']=='lidar'),
                   key=lambda r: (r['receive_monotonic_ns'], r['record_id']))
    gaps, segments, segment = [], {}, 0
    previous = None
    for row in lidar:
        if previous is not None:
            a, b = previous.get('source_timestamp_ns'), row.get('source_timestamp_ns')
            delta = b-a if a is not None and b is not None else None
            if delta != expected_step_ns:
                segment += 1
                gaps.append(dict(previous_record_id=previous['record_id'], record_id=row['record_id'],
                    source_delta_ns=delta, expected_step_ns=expected_step_ns,
                    missing_cycles_if_nominal_cadence=(delta//expected_step_ns-1
                        if delta is not None and delta>expected_step_ns and delta%expected_step_ns==0 else None)))
        segments[row['record_id']] = segment
        previous = row
    for group in groups:
        ids = group['lidar_record_ids']
        group['internal_cycle_segment_id'] = segments[ids[0]] if len(ids)==1 else None
        group['observed_lidar_simulation_timestamp_ns'] = (by_id[ids[0]].get('source_timestamp_ns')
                                                         if len(ids)==1 else None)
        group['same_numeric_lidar_gps_imu_stamp'] = None
        if group['status']=='receipt_group_candidate':
            stamps=[group['selection'][k]['source_timestamp_ns'] for k in ('lidar','gnss','imu')]
            group['same_numeric_lidar_gps_imu_stamp'] = len(set(stamps))==1
            if len(set(stamps))!=1:
                group['reasons'].append('lidar_gps_imu_source_disagreement')
                group['status']='rejected'
                group['selection']={}
        group['camera_exposure_simulation_timestamp_ns'] = None
    return gaps
