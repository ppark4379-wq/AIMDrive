"""Observed empty-road baseline for the tunnel, without changing Expert.

Map link bounds and widths are nominal: passing this test does not survey the
physical portals or walls. Foreign actors and SpawnPoints are never mutated.
"""

import copy
import json
import math
import os
import time
from pathlib import Path

from ...utils.transform_utils import polyline_length


MODE_NAME = "full_loop_tunnel_normal"
TUNNEL_LINK = "A2256W000126"
OPPOSING_LINK = "A2256W000131"
CHECK_PERIOD_SEC = 0.5
RPC_TIMEOUT_SEC = 1.0
ACTOR_MARGIN_M = 0.5


def _enabled(context):
    return getattr(getattr(context, "mission_mode", None), "name", None) == MODE_NAME


def _xyz(value, label):
    try:
        result = tuple(float(getattr(value, axis)) for axis in ("x", "y", "z"))
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid tunnel baseline " + label) from error
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError("Nonfinite tunnel baseline " + label)
    return result


def _points(values, label):
    try:
        points = [tuple(float(value) for value in row[:3]) for row in values]
    except (TypeError, ValueError) as error:
        raise RuntimeError("Invalid tunnel baseline " + label) from error
    if (
        len(points) < 2
        or any(len(row) != 3 or not all(math.isfinite(v) for v in row) for row in points)
        or polyline_length(points) <= 0.0
    ):
        raise RuntimeError("Invalid tunnel baseline " + label)
    return points


def _projection(points, x, y):
    """Return distance and progress along a finite polyline."""
    best = (float("inf"), 0.0)
    progress = 0.0
    for first, last in zip(points, points[1:]):
        dx, dy = last[0] - first[0], last[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        ratio = max(0.0, min(1.0, ((x - first[0]) * dx + (y - first[1]) * dy) / length ** 2))
        distance = math.hypot(x - first[0] - ratio * dx, y - first[1] - ratio * dy)
        if distance < best[0]:
            best = (distance, progress + ratio * length)
        progress += length
    return best


def _persist(context):
    report = context.tunnel_baseline_report
    runtime_root = context.global_cfg.get("paths", {}).get("runtime_root")
    if not runtime_root:
        runtime_root = Path(__file__).resolve().parents[2] / ".runtime"
    identity = str(getattr(context, "run_instance_id", "unstarted"))
    if Path(identity).name != identity or identity in ("", ".", ".."):
        raise RuntimeError("Invalid tunnel baseline run identity")
    path = Path(runtime_root) / "tunnel_baseline" / identity / "baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare(context):
    """Validate the mapped tunnel and sliced test route before world startup."""
    if not _enabled(context):
        return
    context.tunnel_baseline_report = {
        "schema": "aimdrive_tunnel_baseline_v1",
        "scenario": MODE_NAME,
        "status": "prepared",
        "failure_reason": "",
        "physical_portals_verified": False,
        "physical_walls_verified": False,
        "usable_physical_width_verified": False,
        "baseline_drive_passed": False,
        "external_spawn_points_disabled": False,
        "spawn_point_policy": "Only runner-owned actors are reset; foreign SpawnPoints are not enumerated or modified.",
        "occupancy_policy": "Fail on observed occupancy in either mapped lane and the test approach/exit, or on inventory failure.",
        "inventory_scope": "all clients, vehicle/pedestrian/obstacle",
        "inventory_rpc_timeout_sec": RPC_TIMEOUT_SEC,
        "inventory_check_period_sec": CHECK_PERIOD_SEC,
        "inventory_checks": 0,
        "last_inventory_elapsed_sec": None,
        "last_blocking_actors": [],
        "started_before_entry": False,
        "mapped_entry_observed": False,
        "mapped_exit_observed": False,
        "inside_observations": 0,
        "last_ego_route_s_m": None,
        "last_ego_xyz": None,
    }
    links = []
    for identity in (TUNNEL_LINK, OPPOSING_LINK):
        link = context.map_loader.get_link(identity)
        if str(link.get("road_type")) != "2":
            raise RuntimeError("Tunnel baseline requires mapped tunnel road_type=2: " + identity)
        points = _points(link.get("points", []), identity + " points")
        widths = [float(link.get(field, 0.0)) for field in ("width_start", "width_end")]
        if any(not math.isfinite(width) or width <= 0.0 for width in widths):
            raise RuntimeError("Invalid nominal tunnel lane width: " + identity)
        links.append({"id": identity, "points": points, "widths": widths})
    own, opposing = links
    if any(math.dist(a[:2], b[:2]) > max(own["widths"] + opposing["widths"]) * 2 for a, b in (
        (own["points"][0], opposing["points"][-1]),
        (own["points"][-1], opposing["points"][0]),
    )):
        raise RuntimeError("Mapped tunnel opposing lane does not match the Ego tunnel")

    route = _points(context.route_points, "test route")
    entry_distance, entry_s = _projection(route, *own["points"][0][:2])
    exit_distance, exit_s = _projection(route, *own["points"][-1][:2])
    length = polyline_length(route)
    if entry_distance > 1.0 or exit_distance > 1.0 or not (0.0 < entry_s < exit_s < length):
        raise RuntimeError("Tunnel baseline test route must include approach, mapped tunnel, and exit")
    context.tunnel_baseline_geometry = {
        "route": route,
        "lanes": links,
        "entry_s": entry_s,
        "exit_s": exit_s,
        # Extend the test route's tube to cover both directions in the approach
        # and exit. This is a conservative observation area, not a wall model.
        "test_corridor_half_width": 1.5 * max(own["widths"] + opposing["widths"]),
    }
    context.tunnel_baseline_last_check_elapsed = float("-inf")
    report = context.tunnel_baseline_report
    report["mapped_tunnel"] = {
        "ego_link": TUNNEL_LINK,
        "opposing_link": OPPOSING_LINK,
        "entry_xyz": own["points"][0],
        "exit_xyz": own["points"][-1],
        "length_m": polyline_length(own["points"]),
        "nominal_lane_widths_m": {lane["id"]: lane["widths"] for lane in links},
        "nominal_combined_width_m": sum(min(lane["widths"]) for lane in links),
        "bounds_source": "MGeo road_type=2 link endpoints; physical structure comparison pending",
    }
    report["test_route"] = {
        "start_xyz": route[0], "end_xyz": route[-1], "length_m": length,
        "mapped_entry_route_s_m": entry_s, "mapped_exit_route_s_m": exit_s,
        "approach_length_m": entry_s, "after_exit_length_m": length - exit_s,
        "conservative_guard_half_width_m": context.tunnel_baseline_geometry["test_corridor_half_width"],
    }
    _persist(context)


def _inventory(context):
    from proto.morai.actor.actor_get_pb2 import GetAllActorsFilter

    if context.grpc.world is None:
        raise RuntimeError("Tunnel baseline actor inventory: MORAI world is unavailable")
    query = GetAllActorsFilter(client_key="", vehicle=True, pedestrian=True, obstacle=True)
    try:
        response = context.grpc.world._sim_adapter._actor_stub.GetAllActorsState(
            query, timeout=RPC_TIMEOUT_SEC,
        )
    except Exception as error:
        raise RuntimeError("Tunnel baseline actor inventory RPC failed: " + str(error)) from error
    if response is None or not hasattr(response, "states"):
        raise RuntimeError("Tunnel baseline actor inventory response is unavailable")
    states = list(response.states)
    identifiers = [str(getattr(getattr(getattr(state, "actor_info", None), "id", None), "value", "")) for state in states]
    if any(not identity for identity in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Tunnel baseline actor inventory has missing or duplicate actor IDs")
    ego_id = str(context.grpc.get_ego().get_id())
    if not ego_id or identifiers.count(ego_id) != 1:
        raise RuntimeError("Tunnel baseline actor inventory is missing the known Ego")
    return states, ego_id


def _check_occupancy(context, elapsed):
    states, ego_id = _inventory(context)
    geometry = context.tunnel_baseline_geometry
    blockers = []
    ego_position = None
    for state in states:
        identity = str(state.actor_info.id.value)
        position = _xyz(state.transform.location, "actor position " + identity)
        if identity == ego_id:
            ego_position = position
            continue
        box = getattr(state, "bounding_box", None)
        if box is None:
            raise RuntimeError("Tunnel baseline actor has no bounding box: " + identity)
        extent = _xyz(box.extent, "actor extent " + identity)
        if min(extent) <= 0.0:
            raise RuntimeError("Tunnel baseline actor has invalid bounding box extent: " + identity)
        # MORAI bounding_box.location is the world-space box center, while
        # transform.location is the actor pivot. Prefer the actual box center.
        center = _xyz(box.location, "bounding box center " + identity) if hasattr(box, "location") else position
        radius = math.hypot(extent[0], extent[1]) + ACTOR_MARGIN_M
        route_distance, _ = _projection(geometry["route"], *center[:2])
        blocked = route_distance <= geometry["test_corridor_half_width"] + radius
        for lane in geometry["lanes"]:
            lane_distance, _ = _projection(lane["points"], *center[:2])
            blocked |= lane_distance <= max(lane["widths"]) / 2.0 + radius
        if blocked:
            blockers.append({
                "actor_id": identity,
                "client_key": str(getattr(state.actor_info, "client_key", "")),
                "world_bbox_center_xyz": center,
                "guard_radius_m": radius,
                "distance_to_test_route_m": route_distance,
            })
    report = context.tunnel_baseline_report
    report["inventory_checks"] += 1
    report["last_inventory_elapsed_sec"] = float(elapsed)
    report["last_blocking_actors"] = blockers
    context.tunnel_baseline_last_check_elapsed = float(elapsed)
    if blockers:
        raise RuntimeError("Tunnel baseline road is occupied: " + ", ".join(actor["actor_id"] for actor in blockers))
    return ego_position


def _fail(context, error):
    try:
        finish(context, "failed", str(error) or type(error).__name__)
    finally:
        stop = getattr(context, "stop_drive_control", None)
        if callable(stop):
            stop()


def before_drive(context):
    """Require a complete global inventory and empty test corridor."""
    if not _enabled(context):
        return
    try:
        ego_position = _check_occupancy(context, 0.0)
        distance, progress = _projection(context.tunnel_baseline_geometry["route"], *ego_position[:2])
        if distance > 1.0 or progress >= context.tunnel_baseline_geometry["entry_s"]:
            raise RuntimeError("Tunnel baseline Ego must start on the approach before the mapped entry")
        report = context.tunnel_baseline_report
        report["started_before_entry"] = True
        report["last_ego_route_s_m"] = progress
        report["last_ego_xyz"] = ego_position
        report["status"] = "running"
        _persist(context)
    except Exception as error:
        _fail(context, error)
        raise


def observe(context, elapsed, ego_state, current_s):
    """Record actual Ego traversal and guard against newly arriving traffic."""
    if not _enabled(context):
        return
    try:
        x, y = float(ego_state["x"]), float(ego_state["y"])
        z = float(ego_state.get("z", context.tunnel_baseline_geometry["route"][0][2]))
        progress = float(current_s)
        if not all(math.isfinite(value) for value in (x, y, z, progress, float(elapsed))):
            raise RuntimeError("Nonfinite tunnel baseline Ego observation")
        geometry = context.tunnel_baseline_geometry
        _, observed_progress = _projection(geometry["route"], x, y)
        # Use observed geometry instead of a potentially stale supplied progress.
        report = context.tunnel_baseline_report
        previous = report["last_ego_route_s_m"]
        report["last_ego_route_s_m"] = observed_progress
        report["last_ego_xyz"] = [x, y, z]
        if geometry["entry_s"] <= observed_progress <= geometry["exit_s"]:
            report["inside_observations"] += 1
            if report["started_before_entry"]:
                report["mapped_entry_observed"] = True
        if (
            previous is not None and previous < geometry["exit_s"] <= observed_progress
            and report["mapped_entry_observed"]
        ):
            report["mapped_exit_observed"] = True
        if float(elapsed) - context.tunnel_baseline_last_check_elapsed >= CHECK_PERIOD_SEC:
            _check_occupancy(context, elapsed)
            _persist(context)
    except Exception as error:
        _fail(context, error)
        raise


def finish(context, status, error=""):
    """Save drive outcome; never convert incomplete traversal into a pass."""
    if not _enabled(context) or not hasattr(context, "tunnel_baseline_report"):
        return
    report = context.tunnel_baseline_report
    if status == "passed" and (
        report["status"] == "failed" or report["failure_reason"]
        or not report["started_before_entry"]
        or not report["mapped_entry_observed"] or not report["mapped_exit_observed"]
        or not report["inside_observations"] or not report["inventory_checks"]
    ):
        reason = report["failure_reason"] or "Tunnel baseline ended without observing mapped tunnel entry and exit"
        report.update(status="failed", failure_reason=reason, baseline_drive_passed=False)
        _persist(context)
        raise RuntimeError(reason)
    report["status"] = str(status)
    if error:
        report["failure_reason"] = str(error)
    report["baseline_drive_passed"] = status == "passed"
    _persist(context)


class _ParkingPending(RuntimeError):
    def __init__(self, reason, readback):
        super().__init__(reason)
        self.readback = readback


def _wait_parking_stage(record, phase, read, *, now, sleep, timeout_sec,
                        minimum_seconds=0., minimum_stable_seconds=.2):
    started = now()
    deadline = started + timeout_sec
    since, count = None, 0
    record["phase"] = phase
    while now() <= deadline:
        try:
            observed = read()
        except RuntimeError as error:
            observed = getattr(error, "readback", None)
            if observed is None:
                raise
            record["checks"].append(dict(phase=phase, status="pending", reason=str(error),
                                         readback=copy.deepcopy(observed)))
            record["last_readback"] = copy.deepcopy(observed)
            since, count = None, 0
        else:
            record["checks"].append(dict(phase=phase, status="observed", readback=copy.deepcopy(observed)))
            record["last_readback"] = copy.deepcopy(observed)
            since = now() if since is None else since
            count += 1
            # Require both repeated fresh reads and a continuous observation
            # window, including the longer window after releasing our publisher.
            if (now() <= deadline and now()-started >= minimum_seconds
                    and count >= 3 and now()-since >= minimum_stable_seconds-1e-8):
                return observed
        if now() >= deadline:
            break
        sleep(min(.1, max(0., deadline-now())))
    raise RuntimeError("Tunnel baseline " + phase + " did not settle within "
                       + str(timeout_sec) + " wall seconds; last_readback="
                       + repr(record.get("last_readback")))


def park(context, *, now=None, sleep=None):
    """Verify the shared AUTO/D brake stop after the native recording closes.

    P is a separate actuator setting and is not required for a completed native
    demonstration. Observe physical stopping with the owned Expert stopped and
    an exclusive ROS brake owner, then release that owner and verify the same
    AUTO/D, brake, speed and pose for a continuous half-second window. No motion
    reset or KEYBOARD/P request may conceal an unsuccessful physical stop.
    """
    if not _enabled(context) or getattr(context.grpc, "world", None) is None:
        return
    from . import tunnel_static_startup as startup

    now, sleep = now or time.monotonic, sleep or time.sleep
    started = now()
    record = dict(status="stopping", phase="stop_owned_expert", checks=[],
                  stop_policy="common_auto_d_ros_brake", stop_timeout_wall_seconds=20.,
                  post_release_timeout_wall_seconds=3., physical_stop_verified=False,
                  owned_expert_stopped=False, controller_publishers_absent=False,
                  publisher_released_after_verified=False, post_release_verified=False,
                  external_input_disabled=False, final_verified=False,
                  gear_p_verified=False, velocity_reset_used=False)
    report = getattr(context, "tunnel_baseline_report", None)
    if report is not None:
        report["ego_stopped_and_parked"] = False
        report["final_stop_verified"] = False
        report["stop_policy"] = record["stop_policy"]
        report["final_parking"] = record
    try:
        controller = getattr(context, "expert_controller", None)
        process = getattr(controller, "process", None)
        context.stop_drive_control()
        if process is not None and process.poll() is None:
            raise RuntimeError("Owned Expert did not stop before final tunnel braking")
        record["owned_expert_stopped"] = True
        startup.hold_common_brake(context.grpc)

        def stopped_readback(reference=None):
            try:
                return startup.common_stopped_readback(context.grpc, reference)[1]
            except RuntimeError as error:
                observed = getattr(error, "readback", None)
                if observed is not None and (observed.get("brake") is None or observed.get("throttle") is None):
                    raise RuntimeError("Nonfinite tunnel final actuator observation") from error
                raise

        stopped = _wait_parking_stage(
            record, "physical_stop", stopped_readback,
            now=now, sleep=sleep, timeout_sec=20., minimum_seconds=1.5,
        )
        record.update(physical_stop_verified=True, actual_stop=copy.deepcopy(stopped))
        startup.release_common_brake(context.grpc)
        record["publisher_released_after_verified"] = True
        startup._require_no_controller()
        stopped_final = _wait_parking_stage(
            record, "post_release", lambda: stopped_readback(stopped),
            now=now, sleep=sleep, timeout_sec=3., minimum_stable_seconds=.5,
        )
        record.update(status="verified", final_verified=True, post_release_verified=True,
                      controller_publishers_absent=True, actual_final=copy.deepcopy(stopped_final))
        if report is not None:
            report["final_stop_verified"] = True
    except BaseException as error:
        record.update(status="failed", failure_reason=str(error) or type(error).__name__)
        raise
    finally:
        try:
            startup.release_common_brake(context.grpc)
        except BaseException as error:
            record.update(status="failed", final_verified=False,
                          failure_reason="Brake publisher cleanup failed: " + (str(error) or type(error).__name__))
            if report is not None:
                report["final_stop_verified"] = False
            raise
        finally:
            record["elapsed_wall_seconds"] = now()-started
            if report is not None:
                _persist(context)
