"""Position provenance and bounded selection at the perception frame's time."""


def position_identity(record):
    meta = record.meta or {}
    identity = dict(meta.get('identity', meta))
    identity.setdefault('sample_id', record.sequence)
    identity.setdefault('received_ns', getattr(record, 'received_ns', None)
                        or round(record.received_at * 1e9))
    identity.setdefault('source_stamp_ns', getattr(record, 'source_stamp_ns', None)
                        or (round(record.source_time * 1e9) if record.source_time else 0))
    identity.setdefault('position_mode', 'dead_reckoning' if isinstance(record.value, dict)
                        and record.value.get('dead_reckoning') else 'gps')
    return identity


def position_at_frame(records, frame, source_time=None, max_skew_s=.15):
    """Bounded position at the frame's source clock, or a receipt approximation.

    Normal NMEA fixes have no shared acquisition clock: their receipt alignment
    remains an explicit approximation. Source-stamped DR uses acquisition time;
    a matching observation may have arrived after the frame, before processing.
    """
    records = tuple(records)
    # Availability is a receipt-clock event, not a synthetic old GPS fix. An
    # expired/failed estimator must not revive an earlier DR state merely because
    # that state's acquisition clock is close to the perception frame's clock.
    barrier = max((record.received_at for record in records
                   if (record.value is None or position_identity(record)['position_mode']=='unavailable')
                   and record.received_at <= frame.received_at), default=float('-inf'))
    candidates = []
    for record in records:
        if record.received_at < barrier:
            continue
        stamped = source_time is not None and record.source_time not in (None, 0)
        delta = source_time - record.source_time if stamped else frame.received_at - record.received_at
        # Source LiDAR timestamps are truncated to whole microseconds.
        epsilon = 1e-6 if stamped else 1e-9
        if -epsilon <= delta <= max_skew_s:
            candidates.append((max(0., delta), record))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None
