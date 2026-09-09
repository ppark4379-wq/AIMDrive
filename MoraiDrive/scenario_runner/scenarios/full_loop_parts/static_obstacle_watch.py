"""Read-only post-spawn monitoring; never respawn, move, or replace an actor."""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

from ...utils.transform_utils import nearest_point_index, polyline_length


@dataclass(frozen=True)
class WatchPolicy:
    sample_period_sec: float = 0.2
    rpc_timeout_sec: float = 0.3
    consecutive_failures: int = 3
    xy_tolerance_m: float = 3.0
    z_tolerance_m: float = 1.0
    clearance_m: float = 15.0


@dataclass
class StaticWatch:
    actor_id: str
    model: str
    expected_xyz: Tuple[float, float, float]
    end_s: float
    log_path: Path
    started_at: float
    policy: WatchPolicy
    active: bool = True
    next_sample_at: float = 0.0
    next_print_at: float = 0.0
    failures: int = 0
    samples: int = 0


def record(watch, event, **details):
    entry = {
        "event": event, "time_unix": time.time(),
        "elapsed_sec": round(time.monotonic() - watch.started_at, 3),
        "actor_id": watch.actor_id, "model": watch.model,
        "expected_xyz": watch.expected_xyz,
        **details,
    }
    text = json.dumps(entry, ensure_ascii=False, allow_nan=False)
    try:
        with watch.log_path.open("a", encoding="utf-8") as output:
            output.write(text + "\n")
    except OSError as error:
        # A diagnostic write failure must not suppress braking or actor cleanup.
        print(f"[StaticWatch] log write failed: {error}; record={text}", flush=True)


def arm(context):
    """Append to the existing successful spawn flow without changing its inputs."""
    context.static_obstacle_watch = None
    position = context.route_definition["zones"]["static_obstacle"]["position"]
    expected = tuple(float(position[key]) for key in ("x", "y", "z"))
    index = nearest_point_index(context.route_points, *expected[:2])
    nearest = context.route_points[index]
    policy = WatchPolicy()
    if math.hypot(nearest[0] - expected[0], nearest[1] - expected[1]) > policy.xy_tolerance_m:
        # Legacy after-static/highway routes may create a static actor behind
        # START. They must not acquire a new monitor for a section not driven.
        print("[StaticWatch] skipped: obstacle outside active route")
        return
    end_s = polyline_length(context.route_points[:index + 1]) + policy.clearance_m
    directory = Path(context.global_cfg["paths"]["runtime_root"]) / "static_obstacle_watch"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{context.scenario_name}_{context.run_instance_id}.jsonl"
    watch = StaticWatch(
        actor_id=str(context.static_obstacle.get_id()).strip(),
        model=context.static_obstacle_model, expected_xyz=expected, end_s=end_s,
        log_path=log_path, started_at=time.monotonic(), policy=policy,
    )
    context.static_obstacle_watch = watch
    record(watch, "armed", policy=asdict(policy), end_s=end_s,
           run_seed=context.run_seed, requested_scale=context.static_obstacle_scale)
    print(f"[StaticWatch] armed id={watch.actor_id} model={watch.model} log={log_path}")
    update(context, 0.0)


def read_states(context, policy):
    """The same run-owned obstacle query, with a bounded RPC deadline.

    The SDK convenience wrapper omits the timeout, so use its read-only stub
    directly. No SDK or shared adapter function is modified.
    """
    from proto.morai.actor.actor_get_pb2 import GetAllActorsFilter

    request = GetAllActorsFilter()
    request.client_key = context.grpc.client_key
    request.vehicle = False
    request.pedestrian = False
    request.obstacle = True
    stub = context.grpc.world._sim_adapter._actor_stub
    response = stub.GetAllActorsState(request, timeout=policy.rpc_timeout_sec)
    if response is None:
        raise RuntimeError("GetAllActorsState returned no response")
    return list(response.states)


def inspect_states(watch, states):
    matches = [state for state in states
               if str(state.actor_info.id.value).strip() == watch.actor_id]
    if not matches:
        return {"status": "missing", "observed_ids": [
            str(state.actor_info.id.value) for state in states
        ]}
    if len(matches) != 1:
        return {"status": "duplicate_id", "matches": len(matches)}
    location = matches[0].transform.location
    xyz = tuple(float(getattr(location, key)) for key in ("x", "y", "z"))
    if not all(math.isfinite(value) for value in xyz):
        return {"status": "invalid_position", "actual_xyz_text": repr(xyz)}
    dx, dy, dz = (value - expected for value, expected in zip(xyz, watch.expected_xyz))
    xy_error = math.hypot(dx, dy)
    status = "ok"
    if abs(dz) > watch.policy.z_tolerance_m:
        status = "z_deviation"
    elif xy_error > watch.policy.xy_tolerance_m:
        status = "xy_deviation"
    return {"status": status, "actual_xyz": xyz,
            "xy_error_m": xy_error, "z_error_m": dz}


def update(context, current_s):
    watch = getattr(context, "static_obstacle_watch", None)
    if watch is None or not watch.active:
        return
    now = time.monotonic()
    passed = float(current_s) >= watch.end_s
    if now < watch.next_sample_at:
        return
    try:
        observation = inspect_states(watch, read_states(context, watch.policy))
    except Exception as error:
        observation = {"status": "query_error", "error": str(error)}
    watch.next_sample_at = time.monotonic() + watch.policy.sample_period_sec
    watch.samples += 1
    previous_failures = watch.failures
    watch.failures = 0 if observation["status"] == "ok" else watch.failures + 1
    record(watch, "sample", route_s=float(current_s),
           consecutive_failures=watch.failures, **observation)
    if watch.failures or previous_failures or now >= watch.next_print_at:
        print(f"[StaticWatch] id={watch.actor_id} model={watch.model} "
              f"status={observation['status']} "
              f"xyz={observation.get('actual_xyz')} "
              f"failures={watch.failures}/{watch.policy.consecutive_failures}",
              flush=True)
        watch.next_print_at = now + 1.0
    if watch.failures >= watch.policy.consecutive_failures:
        watch.active = False
        record(watch, "fatal", **observation)
        reason = (f"Static obstacle post-spawn verification failed: "
                  f"id={watch.actor_id}, model={watch.model}, "
                  f"status={observation['status']}, log={watch.log_path}")
        print(f"[StaticWatch] FATAL {reason}", flush=True)
        context.stop_drive_control()
        raise RuntimeError(reason)
    if passed and observation["status"] == "ok":
        finish(context, "ego_cleared_static_section")


def finish(context, reason):
    watch = getattr(context, "static_obstacle_watch", None)
    if watch is None or not watch.active:
        return
    watch.active = False
    record(watch, "finished", reason=reason, samples=watch.samples)
    print(f"[StaticWatch] finished reason={reason} samples={watch.samples}")
