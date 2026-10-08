"""Observed empty-road baseline for the tunnel, without changing Expert.

Map link bounds and widths are nominal: passing this test does not survey the
physical portals or walls. Foreign actors and SpawnPoints are never mutated.
"""

import json
import math
import os
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


def park(context):
    """Stop the baseline Ego after the Expert exits, including failure paths."""
    if not _enabled(context) or getattr(context.grpc, "world", None) is None:
        return
    from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_P

    context.stop_drive_control()
    ego = context.grpc.get_ego()
    velocity_ok = bool(ego.set_velocity(0.0))
    gear_ok = bool(ego.set_vehicle_gear(GEAR_MODE_P))
    state = ego.get_actor_state()
    velocity = _xyz(state.global_velocity, "parked Ego velocity")
    stopped = math.sqrt(sum(value * value for value in velocity)) <= 0.1
    parked = int(state.vehicle_state.gear_mode) == int(GEAR_MODE_P)
    if not (velocity_ok and gear_ok and stopped and parked):
        raise RuntimeError("Tunnel baseline Ego stop/parking verification failed")
    if hasattr(context, "tunnel_baseline_report"):
        context.tunnel_baseline_report["ego_stopped_and_parked"] = True
        _persist(context)
