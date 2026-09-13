"""Bounded delayed association candidates; no scene labels or clock rewriting.

These policies test whether RealTime input association can improve. A clock
offset/bound is an explicit hypothesis, not proof of an exposure timestamp.
"""
from bisect import bisect_left,bisect_right
from async_capture.temporal_quality import inspect_stream


class CameraTimeline:
    def __init__(self, rows):
        self.rows=sorted(rows,key=lambda r:(r['receive_monotonic_ns'],r['record_id']))
        self.receipts=[r['receive_monotonic_ns'] for r in self.rows]
        if len({r['record_id'] for r in self.rows})!=len(self.rows):raise ValueError('Repeated record identity')
        self.quality = inspect_stream(self.rows, gap_ns=250_000_000)

    def select(self, anchor, *, field='receive_monotonic_ns', offset_ns=0,
               wait_ns=100_000_000, latest=False, phase_interval_ns=None, max_error_ns=None,
               require_phase_stable_selection=True, max_receive_error_ns=250_000_000,
               observed_until_ns=None):
        if not 0<=wait_ns<=250_000_000:raise ValueError('Unbounded decision delay')
        if not 0<=max_receive_error_ns<=250_000_000:raise ValueError('Unbounded receipt distance')
        deadline=anchor['receive_monotonic_ns']+wait_ns
        result=dict(status='rejected',decision_monotonic_ns=deadline,record_id=None,reason=None,
                    physical_simultaneity_verified=False,training_visible=False)
        if observed_until_ns is not None and deadline>observed_until_ns:
            return dict(result,reason='decision_window_truncated')
        if anchor.get(field) is None:return dict(result,reason='anchor_clock_missing')
        lo=bisect_left(self.receipts,anchor['receive_monotonic_ns']-250_000_000)
        hi=bisect_right(self.receipts,deadline)
        available=[r for r in self.rows[lo:hi] if 'enqueue_monotonic_ns' in r
            and max(r['receive_monotonic_ns'],r['enqueue_monotonic_ns'])<=deadline and r.get(field) is not None]
        if not available:return dict(result,reason='no_available_camera')
        # Receipt-only historical studies need not have source stamps, but a
        # present, demonstrably broken source stamp must not pass silently.
        available=[r for r in available if not [reason for reason in self.quality['invalid'].get(r['record_id'],[])
            if reason!='source_stamp_missing_or_zero' or field=='source_timestamp_ns']]
        if not available:return dict(result,reason='camera_timestamp_quarantined')
        target=anchor[field]+offset_ns
        def nearest(t):return min(available,key=lambda r:(abs(r[field]-t),r[field],r['record_id']))
        if latest:
            earlier=[r for r in available if r[field]<=target]
            if not earlier:return dict(result,reason='no_previous_camera')
            chosen=max(earlier,key=lambda r:(r[field],r['record_id']))
        else:chosen=nearest(target)
        if abs(chosen['receive_monotonic_ns']-anchor['receive_monotonic_ns'])>max_receive_error_ns:
            return dict(result,reason='receive_time_gap_exceeded')
        if phase_interval_ns is not None:
            lower,upper=phase_interval_ns
            if lower>upper:return dict(result,reason='incompatible_phase_interval')
            if not lower<=offset_ns<=upper:raise ValueError('Representative outside model interval')
            endpoints=[anchor[field]+x for x in (lower,upper)]
            if require_phase_stable_selection and any(nearest(t)['record_id']!=chosen['record_id'] for t in endpoints):
                return dict(result,reason='selection_depends_on_unknown_phase')
            error=max(abs(chosen[field]-t) for t in endpoints)
        else:error=abs(chosen[field]-target)
        if max_error_ns is not None and error>max_error_ns:
            return dict(result,reason='model_time_gap_exceeded',model_error_bound_ns=error)
        return dict(result,status='association_candidate',record_id=chosen['record_id'],reason=None,
            available_monotonic_ns=max(chosen['receive_monotonic_ns'],chosen['enqueue_monotonic_ns']),
            camera_receive_minus_lidar_ns=chosen['receive_monotonic_ns']-anchor['receive_monotonic_ns'],
            model_error_bound_ns=error,clock_field=field,offset_hypothesis_ns=offset_ns)
