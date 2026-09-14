"""Fail-closed policy for brief, coherent MORAI Simulation Time stalls.

Simulation Time can stop advancing for a few wall-clock seconds while MORAI
renders or flushes Sensor Record data.  During that interval every simulated
sensor stops together, so no simulation scene is skipped.  A partial stream
loss while /clock advances is different: that can remove a training input and
must fail immediately.
"""


_CLOCK_STALL = frozenset((
    "ros_clock_not_advancing",
    "missing_or_stale_ros_clock",
))


def is_coherent_simulation_pause(failures):
    """Return true only for liveness failures explained by a stopped /clock."""
    failures = tuple(dict.fromkeys(str(value) for value in failures))
    if not any(value in _CLOCK_STALL for value in failures):
        return False
    for value in failures:
        if value in _CLOCK_STALL:
            continue
        if value.startswith("bridge_missing_stream:"):
            continue
        if value.startswith("missing_or_stale_simulation_stream:"):
            continue
        # Same contract when an explicitly partitioned bridge topology is used.
        if ":missing_stream:" in value:
            continue
        return False
    return True


class SimulationPauseGrace:
    """Track a bounded coherent pause without hiding partial sensor failures."""

    def __init__(self, grace_seconds=8.0):
        if not 3.0 <= float(grace_seconds) <= 30.0:
            raise ValueError("Simulation pause grace must be 3..30 seconds")
        self.grace_ns = int(float(grace_seconds) * 1e9)
        self.active = None
        self.incidents = []

    def observe(self, failures, now_ns):
        failures = tuple(dict.fromkeys(str(value) for value in failures))
        now_ns = int(now_ns)
        if not failures:
            if self.active is None:
                return {"action": "healthy"}
            incident = dict(self.active)
            incident.update(
                status="recovered",
                end_monotonic_ns=now_ns,
                duration_ns=now_ns - incident["start_monotonic_ns"],
            )
            self.incidents.append(incident)
            self.active = None
            return {"action": "recovered", "incident": incident}

        if not is_coherent_simulation_pause(failures):
            return {
                "action": "fail",
                "reason": "noncoherent_or_integrity_health_failure",
                "failures": list(failures),
            }

        if self.active is None:
            self.active = {
                "status": "active",
                "start_monotonic_ns": now_ns,
                "first_failures": list(failures),
                "max_observed_duration_ns": 0,
            }
        elapsed_ns = now_ns - self.active["start_monotonic_ns"]
        self.active["last_monotonic_ns"] = now_ns
        self.active["last_failures"] = list(failures)
        self.active["max_observed_duration_ns"] = elapsed_ns
        if elapsed_ns > self.grace_ns:
            return {
                "action": "fail",
                "reason": "coherent_simulation_pause_exceeded_grace",
                "duration_ns": elapsed_ns,
                "grace_ns": self.grace_ns,
                "failures": list(failures),
            }
        return {
            "action": "hold",
            "duration_ns": elapsed_ns,
            "grace_ns": self.grace_ns,
            "failures": list(failures),
        }

    def snapshot(self, now_ns):
        incidents = list(self.incidents)
        active = None
        if self.active is not None:
            active = dict(self.active)
            active["observed_until_monotonic_ns"] = int(now_ns)
            active["duration_ns"] = int(now_ns) - active["start_monotonic_ns"]
        return {
            "schema": "morai_simulation_pause_continuity_v1",
            "grace_ns": self.grace_ns,
            "recovered_incidents": incidents,
            "active_incident": active,
        }
