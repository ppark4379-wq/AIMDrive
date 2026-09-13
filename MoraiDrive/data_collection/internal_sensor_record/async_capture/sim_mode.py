"""Read-only admission for explicitly selected, externally unticked modes."""


def require_free_running_mode(observation,expected='realtime'):
    types={'realtime':'SYNC_MODE_TYPE_UNSPECIFIED',
           'simulation_free_run':'SYNC_MODE_TYPE_TIME_MANAGEMENT'}
    if expected not in types:raise ValueError('Unknown capture time mode')
    response=observation.get('response',{})
    if (not observation.get('available') or response.get('type')!=types[expected]
            or response.get('tick_period')!=0):
        raise ValueError('Expected confirmed free-running mode '+expected+'; no automatic mode change: '+str(observation))
    return dict(requested_time_mode=expected,observed_type=response['type'],external_tick_period=0,
        native_tick_owner=False,physical_simultaneity_verified=False)


class FreeRunningModeContinuity:
    """Separate a transient read failure from an observed mode violation.

    The live ROS SyncModeInfo subscription remains the per-message continuity
    guard.  This slower gRPC witness confirms the richer mode enum and external
    tick period, but a single overloaded management RPC is not evidence that
    the simulator changed modes.
    """
    def __init__(self, expected, max_consecutive_unavailable=3):
        if not 2 <= max_consecutive_unavailable <= 10:
            raise ValueError('Require 2..10 consecutive unavailable observations')
        self.expected=expected
        self.max_consecutive_unavailable=int(max_consecutive_unavailable)
        self.consecutive_unavailable=0
        self.total_unavailable=0

    def observe(self, observation):
        if observation.get('available'):
            contract=require_free_running_mode(observation,self.expected)
            previous=self.consecutive_unavailable
            self.consecutive_unavailable=0
            return dict(status='confirmed',recovered_after_unavailable=previous,
                        contract=contract)
        self.consecutive_unavailable += 1
        self.total_unavailable += 1
        status=dict(status='transient_unavailable',
                    consecutive_unavailable=self.consecutive_unavailable,
                    total_unavailable=self.total_unavailable,
                    max_consecutive_unavailable=self.max_consecutive_unavailable)
        if self.consecutive_unavailable >= self.max_consecutive_unavailable:
            raise ValueError('Free-running mode gRPC witness unavailable '
                +str(self.consecutive_unavailable)+' consecutive times: '+str(observation))
        return status
