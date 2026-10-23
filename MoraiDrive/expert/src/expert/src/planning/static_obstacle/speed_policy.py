"""Recorded-path speed handoff; the original rise-only ramp state is retained."""
from ...config.parameters import lattice as lattice_parameters


def _smooth_static_target_speed(self, target_velocity, vehicle_state, lattice_result):
    """Carry a rise-only speed ramp through the recorded-to-global handoff."""
    target_velocity = max(0., target_velocity)
    details = getattr(lattice_result, 'debug_details', {}).get('static_avoidance', {})
    recorded = details.get('planner') == 'recorded_route_graph'
    if recorded and details.get('path_validated', False):
        self._static_speed_ramp = True
    if not getattr(self, '_static_speed_ramp', False):
        return target_velocity
    if getattr(lattice_result, 'mode', '') in ('highway', 'highway_lane', 'roundabout_entry'):
        self._static_speed_ramp = False
        return target_velocity
    previous = getattr(self, 'previous_target_velocity', None)
    if previous is None:
        previous = max(0., vehicle_state.velocity)
    previous = max(0., previous)
    # Brake requests are never delayed or lifted by acceleration smoothing.
    if target_velocity <= previous:
        return target_velocity
    dt = min(.2, max(.001, getattr(self, 'control_dt', 1./30.)))
    ceiling = previous + lattice_parameters.STATIC_TARGET_ACCEL*dt
    target = min(target_velocity, ceiling)
    if not recorded and target >= target_velocity:
        self._static_speed_ramp = False
    return target

