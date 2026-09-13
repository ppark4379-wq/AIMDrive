"""Optional admission guard for an instrumented, unmodified single clock source."""


class ClockSourceGuard:
    def __init__(self):
        self.identity = None
        self.backward_count = None

    def observe(self, health, receive_monotonic_ns):
        clock = health.get('clock_ingress')
        if not clock:
            return ['clock_source_instrumentation_missing']
        failures = []
        if clock.get('trace_dropped') or clock.get('trace_errors'):
            failures.append('clock_source_trace_incomplete')
        if clock.get('timestamps_modified') is not False or clock.get('policy') != 'observe':
            failures.append('clock_source_not_unmodified_observation')
        # Health's monotonic_ns is the start of its collection, before the
        # source snapshot. Use this callback's later receipt time for freshness.
        now = receive_monotonic_ns
        connected = {r['client_id'] for r in health['clients']}
        sources = {key: r for key,r in clock.get('clients',{}).items()
                   if key in connected and r.get('samples',0)>=2
                   and r.get('last_receive_ns') is not None
                   and 0<=now-r['last_receive_ns']<=3_000_000_000}
        if len(sources)!=1:
            failures.append('clock_source_count:'+str(len(sources)))
            return failures
        key,row = next(iter(sources.items()))
        identity = (health['instance_id'],key,clock['generation'])
        if self.identity is not None and identity != self.identity:
            failures.append('clock_source_changed_during_capture')
        if self.backward_count is not None and row['backward'] != self.backward_count:
            failures.append('clock_source_regressed_during_capture')
        if self.identity is None and not failures:
            self.identity = identity
            self.backward_count = row['backward']
        return failures
