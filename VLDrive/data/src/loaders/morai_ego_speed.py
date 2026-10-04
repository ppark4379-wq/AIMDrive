"""Causal /Ego_topic planar speed in m/s, from sealed recorded signals."""
from bisect import bisect_right
import math

def ego_speed_timeline(signals):
    values = {}
    for row in signals:
        if row.get('role') != 'ego' or row.get('source_timestamp_ns') is None:
            continue
        stamp = int(row['source_timestamp_ns'])
        speed = float(row['signal']['speed'])
        if not math.isfinite(speed) or not 0 <= speed <= 70:
            raise ValueError('Invalid recorded Ego speed')
        if stamp in values and values[stamp] != speed:
            raise ValueError('Conflicting Ego speed at the same source time')
        values[stamp] = speed
    if not values:
        raise ValueError('No recorded Ego speed')
    times = sorted(values)
    return times, [values[t] for t in times]

def causal_ego_speed(times, speeds, stamp_ns, max_age_ns=50_000_000):
    index = bisect_right(times, int(stamp_ns)) - 1
    if index < 0 or int(stamp_ns) - times[index] > max_age_ns:
        raise ValueError('No causal Ego speed within 50ms')
    return speeds[index]
