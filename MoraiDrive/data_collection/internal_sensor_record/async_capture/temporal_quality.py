"""Per-stream timestamp evidence. Never reinterpret receipt as exposure time."""
from collections import defaultdict


def inspect_stream(rows, *, gap_ns, start_ns=None, end_ns=None, step_ns=250_000_000):
    rows = sorted(rows, key=lambda r: (r['receive_monotonic_ns'], r['record_id']))
    invalid = defaultdict(set)
    events = []
    stamp_first_receive = {}
    if rows:
        stamp_first_receive[rows[0]['record_id']] = rows[0]['receive_monotonic_ns']
    for previous, current in zip(rows, rows[1:]):
        a, b = previous['receive_monotonic_ns'], current['receive_monotonic_ns']
        if b-a > gap_ns:
            events.append(dict(kind='receive_gap', start_ns=a, end_ns=b,
                               previous_record_id=previous['record_id'], record_id=current['record_id'],
                               receive_delta_ns=b-a))
        sa, sb = previous.get('source_timestamp_ns'), current.get('source_timestamp_ns')
        if sa is not None and sb is not None:
            if sb == sa:
                stamp_first_receive[current['record_id']] = stamp_first_receive[previous['record_id']]
                events.append(dict(kind='source_stamp_repeated', start_ns=a, end_ns=b,
                    previous_record_id=previous['record_id'], record_id=current['record_id'],
                    blocks_association=False, receive_delta_ns=b-a))
            reason = ('source_timestamp_backwards' if sb < sa else
                      'source_step_relative_receive' if abs((sb-sa)-(b-a)) > step_ns else None)
            if reason:
                # Both endpoints are suspect: a backwards transition can mean
                # either the earlier stamp was too large or the latter too small.
                for row in (previous, current):
                    invalid[row['record_id']].add(reason)
                events.append(dict(kind=reason, start_ns=a, end_ns=b,
                    previous_record_id=previous['record_id'], record_id=current['record_id'],
                    source_delta_ns=sb-sa, receive_delta_ns=b-a))
        stamp_first_receive.setdefault(current['record_id'], b)
        if previous.get('caller_id') != current.get('caller_id'):
            for row in (previous, current):
                invalid[row['record_id']].add('publisher_changed')
            events.append(dict(kind='publisher_changed', start_ns=a, end_ns=b))
    for row in rows:
        if row.get('source_timestamp_ns') is None or row['source_timestamp_ns'] <= 0:
            invalid[row['record_id']].add('source_stamp_missing_or_zero')
        if row.get('enqueue_monotonic_ns', -1) < row['receive_monotonic_ns']:
            invalid[row['record_id']].add('invalid_availability_clock')
    if rows:
        if start_ns is not None and rows[0]['receive_monotonic_ns']-start_ns > gap_ns:
            events.append(dict(kind='initial_receive_gap', start_ns=start_ns,
                               end_ns=rows[0]['receive_monotonic_ns']))
        if end_ns is not None and end_ns-rows[-1]['receive_monotonic_ns'] > gap_ns:
            events.append(dict(kind='terminal_receive_gap', start_ns=rows[-1]['receive_monotonic_ns'],
                               end_ns=end_ns))
    elif start_ns is not None and end_ns is not None:
        events.append(dict(kind='stream_absent', start_ns=start_ns, end_ns=end_ns))
    return dict(invalid={k: sorted(v) for k, v in invalid.items()}, events=events,
                stamp_first_receive_ns=stamp_first_receive)


def update_receipt_bounds(info, mono, source):
    """Callbacks can finish out of arrival order; freshness follows arrival."""
    if mono < info['first_receive_ns']:
        info.update(first_receive_ns=mono, first_source_ns=source)
    if mono >= info.get('last_receive_ns', -1):
        info.update(last_receive_ns=mono, last_source_ns=source)
