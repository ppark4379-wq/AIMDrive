"""Release a deliberately blocking static layout after a prolonged Ego stop.

Only a preset that declares ``blockage_release`` arms this; every other run is
unchanged.  When every recorded bypass is blocked the Expert stops behind the
layout, and without a release the lap could never finish.  After the Ego has
stood still close behind the layout for the declared time, every run-owned
static obstacle is destroyed and its absence is read back.  The release is
disclosed as an intervention event that is not training visible.

The stop must last both the wall time and the scenario (simulation) time.
Under Sensor Record synchronization the simulation runs about three times slower
than the wall clock, and only the simulated stop becomes training frames: a
6 s wall wait there was a 1.9 s simulated stop, so every stopped frame's 2.5 s
future already contained the restart after the release.
"""

import math
import time

from ...utils import safe_print as print
from . import actor_lifecycle
from . import traffic_light_override


DEFAULTS = {"wait_wall_sec": 6.0, "wait_scenario_sec": 6.0, "stopped_speed_mps": 0.3,
            "max_gap_m": 12.0}
LIMITS = {"wait_wall_sec": (0.5, 60.0), "wait_scenario_sec": (0.5, 60.0),
          "stopped_speed_mps": (0.05, 2.0), "max_gap_m": (1.0, 40.0)}
# A red stop line this close ahead explains the stop; do not count it.
SIGNAL_STOP_LOOKAHEAD_M = 8.0
# Obstacles slightly behind the Ego centre still bound the stopped position.
BEHIND_TOLERANCE_M = 2.0
# Under Sensor Record synchronization MORAI applies a destroy on a later
# simulation step, so absence is polled: two consecutive empty readbacks within
# ABSENCE_POLLS * ABSENCE_POLL_SEC (5 s) prove the removal.
ABSENCE_POLL_SEC = 0.1
ABSENCE_POLLS = 50
ABSENCE_CONFIRMATIONS = 2


def validate_policy(raw):
    """Preset ``blockage_release`` value -> complete policy, or None when absent."""
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("blockage_release must be a mapping or true")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError("Unknown blockage_release keys: {}".format(unknown))
    policy = dict(DEFAULTS)
    for key, value in raw.items():
        value = float(value)
        low, high = LIMITS[key]
        if not (math.isfinite(value) and low <= value <= high):
            raise ValueError("blockage_release.{} must be in [{}, {}]".format(key, low, high))
        policy[key] = value
    return policy


def initialize(context):
    context.static_blockage_release_policy = None
    context.static_blockage_armed_reported = False
    context.static_blockage_wait = None
    context.static_blockage_released = False


def _emit(context, event_name, **fields):
    # Scenario-only Expert verification has no collection event writer; keep the
    # evidence in its persistent batch log as the traffic override does.
    print("[STATIC BLOCKAGE] {} {}".format(event_name, fields))
    writer = getattr(context, "event", None)
    if callable(writer):
        writer(event_name, training_visible=False, **fields)


def gap_to_layout(context, current_s):
    """Route distance from the Ego to the nearest layout obstacle not yet passed."""
    gaps = [float(item["route_s"]) - float(current_s)
            for item in getattr(context, "static_obstacle_expectations", [])]
    ahead = [gap for gap in gaps if gap >= -BEHIND_TOLERANCE_M]
    return min(ahead) if ahead else None


def _red_stop_line_ahead(context, current_s):
    stop = traffic_light_override.upcoming_stop(context, current_s, SIGNAL_STOP_LOOKAHEAD_M)
    if stop is None:
        return False
    state = traffic_light_override._read_signal(context, stop["light_id"])
    # An unreadable light is treated as red: never release on ambiguous evidence.
    return state is None or traffic_light_override.straight_movement_is_blocked(state["color"])


def update(context, elapsed, ego_state, current_s):
    policy = getattr(context, "static_blockage_release_policy", None)
    if not policy or context.static_blockage_released:
        return
    if not context.static_blockage_armed_reported:
        context.static_blockage_armed_reported = True
        _emit(context, "static_blockage_release_armed", policy=dict(policy),
              actor_ids=[str(item["actor_id"]) for item in context.static_obstacle_expectations])
    now = time.monotonic()
    speed = float(ego_state.get("speed", float("inf")))
    gap = gap_to_layout(context, current_s)
    stopped = (speed <= policy["stopped_speed_mps"] and gap is not None
               and gap <= policy["max_gap_m"] and not _red_stop_line_ahead(context, current_s))
    waiting = context.static_blockage_wait
    if not stopped:
        if waiting is not None:
            _emit(context, "static_blockage_wait_reset", route_s=float(current_s),
                  waited_wall_sec=float(now - waiting["started_wall"]), speed_mps=speed, gap_m=gap)
        context.static_blockage_wait = None
        return
    if waiting is None:
        context.static_blockage_wait = {"started_wall": now, "started_elapsed": float(elapsed)}
        _emit(context, "static_blockage_wait_started", route_s=float(current_s), gap_m=gap,
              speed_mps=speed, wait_clock="host_monotonic")
        return
    waited = now - waiting["started_wall"]
    if (waited < policy["wait_wall_sec"]
            or float(elapsed) - waiting["started_elapsed"] < policy["wait_scenario_sec"]):
        return
    release(context, elapsed, current_s, gap, waited, waiting["started_elapsed"])


def wait_until_absent(context, actor_ids):
    """Poll the obstacle inventory until none of actor_ids is listed (twice in a row)."""
    wanted = set(actor_ids)
    present, polls, confirmations = set(wanted), [], 0
    for poll in range(1, ABSENCE_POLLS + 1):
        try:
            states = list(context.grpc.get_all_obstacle_states())
            present = wanted & {str(state.actor_info.id.value).strip() for state in states}
            polls.append({"poll": poll, "inventory_size": len(states), "present": sorted(present)})
            confirmations = 0 if present else confirmations + 1
        except Exception as error:  # a failed query proves nothing
            polls.append({"poll": poll, "error": str(error)})
            confirmations = 0
        if confirmations >= ABSENCE_CONFIRMATIONS:
            break
        time.sleep(ABSENCE_POLL_SEC)
    # Keep the evidence compact: the first readbacks and the final one.
    return confirmations >= ABSENCE_CONFIRMATIONS, present, polls[:3] + polls[-1:], len(polls)


def rearm_signal_override(context, current_s):
    """Let the traffic-light override act again at the stop line the Ego waits at.

    The Expert in these layouts stops at the stop line in front of the blockage.
    The override changes an exact light only once per stop line, with an impulse
    green; the 6 s simulated wait can outlast it, and the Ego then waited for the
    natural cycle after the release.
    """
    stop = traffic_light_override.upcoming_stop(context, current_s, SIGNAL_STOP_LOOKAHEAD_M)
    completed = getattr(context, "traffic_override_completed", None)
    if stop is None or completed is None:
        return
    key = (str(stop["light_id"]), round(float(stop["route_s"]), 3))
    if key in completed:
        completed.discard(key)
        traffic_light_override.reset_wait(context)
        _emit(context, "static_blockage_signal_override_rearmed", light_id=key[0],
              stopline_route_s=key[1], route_s=float(current_s))


def release(context, elapsed, current_s, gap, waited_wall_sec, started_elapsed):
    """Destroy every run-owned static obstacle and prove each one is gone."""
    expectations = list(context.static_obstacle_expectations)
    actor_ids = [str(item["actor_id"]).strip() for item in expectations]
    models = [item.get("model") for item in expectations]
    removed = []

    def operation():
        accepted = {actor_id: bool(context.grpc.destroy_obstacle_by_id(actor_id)) for actor_id in actor_ids}
        absent, present, observations, polls = wait_until_absent(context, actor_ids)
        rows = [{"actor_id": actor_id, "model": model, "destroy_accepted": accepted[actor_id],
                 "absent_after_readback": absent and actor_id not in present,
                 "inventory_polls": polls, "inventory_observations": observations}
                for actor_id, model in zip(actor_ids, models)]
        failures = [row for row in rows if not (row["destroy_accepted"] and row["absent_after_readback"])]
        if failures:
            raise RuntimeError("Static blockage release left obstacles in place: {}".format(failures))
        removed.extend(rows)

    actor_lifecycle.mutate(context, "static_blockage_release", float(current_s), operation,
                           actor_ids=actor_ids, models=models)
    context.static_blockage_released = True
    context.static_blockage_wait = None
    rearm_signal_override(context, current_s)
    # The scenario clock is the simulation timestamp under Sensor Record capture; it
    # lets dataset building drop frames whose 2.5 s future contains the release.
    scenario_now = getattr(context, "_scenario_now", None)
    clock_domain = getattr(context, "_clock_domain", None)
    _emit(context, "static_blockage_released", route_s=float(current_s), gap_m=gap,
          scenario_elapsed_sec=float(elapsed),
          scenario_clock_sec=float(scenario_now()) if callable(scenario_now) else None,
          scenario_clock_domain=clock_domain() if callable(clock_domain) else None,
          wait_wall_sec=float(waited_wall_sec),
          wait_scenario_sec=float(elapsed) - float(started_elapsed),
          actor_ids=actor_ids, models=models, removed=removed,
          policy=dict(context.static_blockage_release_policy),
          intervention=True, mission_distribution_changed=True)
