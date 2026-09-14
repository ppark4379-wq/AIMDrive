"""Conservative offline exclusion, never an exposure certificate."""
import math

POLICY = dict(version='bundle_screen_v1', minimum_points=20, maximum_zero_error_px=6.,
              shifted_improvement_px=1., lifecycle_padding_ns=1_000_000_000,
              shifts=list(range(-3, 4)), physical_simultaneity_verified=False)
VIEWS = ('front', 'left', 'right', 'rear')


def screen(groups, edges, events):
    evidence = {}
    for row in edges:
        key = (row['group_index'], row['view'])
        if key in evidence:
            raise ValueError('Duplicate content evidence')
        evidence[key] = row
    mutations = []
    opened = {}
    for event in events:
        if event['event'] == 'actor_lifecycle_started':
            if event['action'] in opened:
                raise ValueError('Overlapping lifecycle action')
            opened[event['action']] = event['receive_monotonic_ns']
        elif event['event'] == 'actor_lifecycle_finished':
            start = opened.pop(event['action'], event['started_monotonic_ns'])
            mutations.append((start, event['receive_monotonic_ns']))
    mutations.extend((start, float('inf')) for start in opened.values())
    result = []
    previous_source = None
    segment = -1
    previous_selected = False
    for group in groups:
        reasons = list(group['reasons'])
        if group['status'] != 'receipt_group_candidate' and not reasons:
            reasons.append('receipt_group_not_candidate')
        a, b = group['first_receive_monotonic_ns'], group['decision_monotonic_ns']
        if any(a <= end+POLICY['lifecycle_padding_ns'] and b >= start-POLICY['lifecycle_padding_ns']
               for start, end in mutations):
            reasons.append('actor_lifecycle_transition')
        for view in VIEWS:
            e = evidence.get((group['group_index'], view))
            scores = e.get('scores_px', []) if e else []
            if (not e or not e.get('valid') or e['points'] < POLICY['minimum_points'] or
                    len(scores) != 7 or any(v is None or not math.isfinite(v) for v in scores)):
                reasons.append('content_unobserved:'+view)
            elif e['category'] not in ('moving', 'turning'):
                reasons.append('timing_unobservable_low_motion:'+view)
            elif scores[3] > POLICY['maximum_zero_error_px']:
                reasons.append('projection_inconclusive:'+view)
            elif scores[3]-min(scores) > POLICY['shifted_improvement_px']:
                reasons.append('shifted_content_suspect:'+view)
        selected = not reasons
        source = group.get('observed_lidar_simulation_timestamp_ns')
        if selected and (not previous_selected or source is None or previous_source is None or source-previous_source != 100_000_000):
            segment += 1
        result.append(dict(group_index=group['group_index'], reasons=reasons,
            status='diagnostic_candidate' if selected else 'excluded',
            selected_record_ids=group['record_ids'] if selected else [],
            source_timestamp_ns=source, contiguous_segment_id=segment if selected else None,
            physical_simultaneity_verified=False, training_visible=False))
        previous_source, previous_selected = source, selected
    return result
