"""Parked-Ego actor preview, with owned cleanup and fresh global observations.

This module never starts Expert or commands Ego to drive. Zero-speed braking
is limited to owned NPCs. A valid preview is only evidence of placement and
stationary actors on the nominal MGeo road.
"""

import copy
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from .layout import (bbox_world, boxes_overlap, conservative_box_union, footprint_corners,
                     project_polyline, route_pose, validate_candidate)
from ..static_obstacle import lock_static_obstacle_transform


RPC_TIMEOUT_SEC = 1.0
SPAWN_READBACK_ATTEMPTS = 10
SPAWN_READBACK_TIMEOUT_SEC = 1.5
SPAWN_READBACK_RETRY_SEC = 0.1
CHECK_PERIOD_SEC = 0.2
CLEANUP_SETTLE_SEC = 1.5
CLEANUP_READBACK_TIMEOUT_SEC = 3.0
CLEANUP_READBACK_ATTEMPTS = 32
CLEANUP_READBACK_RETRY_SEC = 0.1
INITIAL_BODY_READBACK_TIMEOUT_SEC = 1.5
INITIAL_BODY_READBACK_ATTEMPTS = 17
MAX_SPEED_RAW = 0.05
POSE_TOLERANCE_M = 0.10
ROTATION_TOLERANCE_DEG = 0.5
SPAWN_TOLERANCE_M = 0.25
VEHICLE_GROUND_SNAP_Z_TOLERANCE_M = 1.0


class _SpawnReadbackPending(RuntimeError):
    """Only a newly created actor's initial registration may be retried."""


class _OwnedBaselineOverlap(RuntimeError):
    def __init__(self, actor_id):
        self.actor_id = str(actor_id)
        super().__init__("Owned request resolved to a pre-existing actor ID")


class _InitialBodySettlingMismatch(RuntimeError):
    """Only transient position/rotation disagreement may be re-observed."""


def _rpc_timeout(deadline=None):
    if deadline is None:
        return RPC_TIMEOUT_SEC
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("Initial spawn readback deadline expired")
    return min(RPC_TIMEOUT_SEC, remaining)


def _xyz(value, label):
    try:
        result = [float(getattr(value, axis)) for axis in ("x", "y", "z")]
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid " + label) from error
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError("Nonfinite " + label)
    return result


def _id(state):
    return str(getattr(getattr(getattr(state, "actor_info", None), "id", None), "value", ""))


def _inventory(grpc, report, deadline=None):
    from proto.morai.actor.actor_get_pb2 import GetAllActorsFilter

    query = GetAllActorsFilter(client_key="", vehicle=True, pedestrian=True, obstacle=True)
    response = grpc.world._sim_adapter._actor_stub.GetAllActorsState(query, timeout=_rpc_timeout(deadline))
    report["inventory_checks"] += 1
    if response is None or not hasattr(response, "states"):
        raise RuntimeError("Global actor inventory is unavailable")
    states = list(response.states)
    identifiers = [_id(state) for state in states]
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Global actor inventory has missing or duplicate actor IDs")
    return {identity: state for identity, state in zip(identifiers, states)}


def _read_one(grpc, handle, deadline=None):
    try:
        state = grpc.world._sim_adapter._actor_stub.GetActorState(handle.get_object_info(), timeout=_rpc_timeout(deadline))
    except Exception as error:
        # NOT_FOUND can precede initial registration. Transport failures and
        # every readback outside this initial handshake retain fail-fast behavior.
        if deadline is not None:
            import grpc as grpc_api
            if isinstance(error, grpc_api.RpcError) and error.code() == grpc_api.StatusCode.NOT_FOUND:
                raise _SpawnReadbackPending("Owned actor registration is not visible yet") from error
        raise
    if deadline is not None and (state is None or (not _id(state)
            and callable(getattr(state, "ListFields", None)) and not state.ListFields())):
        raise _SpawnReadbackPending("Owned actor readback is not visible yet")
    if state is None or not _id(state):
        raise RuntimeError("Owned actor readback is unavailable")
    return state


def _model(grpc, state, deadline=None):
    response = grpc.world._sim_adapter._actor_stub.GetOptionName(state.actor_info, timeout=_rpc_timeout(deadline))
    value = str(getattr(response, "value", ""))
    if not value:
        raise RuntimeError("Actor model readback is unavailable: " + _id(state))
    return value


def _unsupported_actor_lookup(error, *, known_global=False):
    import grpc as grpc_api
    if not isinstance(error, grpc_api.RpcError):
        return False
    if known_global and error.code() == grpc_api.StatusCode.NOT_FOUND:
        return True
    return (error.code() == grpc_api.StatusCode.UNKNOWN
            and "Exception was thrown by handler" in str(error.details()))


def _object_address(info):
    return {"id": str(info.id.value), "object_type": getattr(info, "object_type", None),
            "client_key": getattr(info, "client_key", "")}


def _same_actor_body(a, b):
    a, b = _observation(a), _observation(b)
    return (math.dist(a["position_xyz"], b["position_xyz"]) < .01
        and math.dist(a["world_bbox"]["center_xyz"], b["world_bbox"]["center_xyz"]) < .01
        and math.dist(a["world_bbox"]["extent_xyz"], b["world_bbox"]["extent_xyz"]) < .01
        and all(_angle_distance(x, y) < .01 for x,y in zip(
            a["world_bbox"]["rotation_xyz_deg"], b["world_bbox"]["rotation_xyz_deg"])))


def _body_differences(a, b):
    a, b = _observation(a), _observation(b)
    return {"position_distance_m":math.dist(a["position_xyz"],b["position_xyz"]),
        "bbox_center_distance_m":math.dist(a["world_bbox"]["center_xyz"],b["world_bbox"]["center_xyz"]),
        "bbox_extent_distance_m":math.dist(a["world_bbox"]["extent_xyz"],b["world_bbox"]["extent_xyz"]),
        "bbox_rotation_difference_deg":[_angle_distance(x,y) for x,y in zip(
            a["world_bbox"]["rotation_xyz_deg"],b["world_bbox"]["rotation_xyz_deg"])],
        "position_limit_m":.01,"bbox_center_limit_m":.01,"bbox_extent_limit_m":.01,"rotation_limit_deg":.01}


def _actor_type(kind):
    from proto.morai.common.enum_pb2 import OBJECT_TYPE_VEHICLE, OBJECT_TYPE_OBSTACLE
    if kind not in ("vehicle", "obstacle"):
        raise RuntimeError("Recorded spawn has an unsupported actor kind")
    return OBJECT_TYPE_VEHICLE if kind == "vehicle" else OBJECT_TYPE_OBSTACLE


def _recorded_spawn_global_binding(grpc, handle, inventory, before_ids, placement, entry, diagnostics, deadline):
    """Prove one recorded spawn from the fresh global delta, never respawn."""
    if placement is None or entry is None:
        raise RuntimeError("Unsupported alias lookup requires an exact recorded spawn request")
    _fixture_namespace(grpc, [entry])
    expected_type = _actor_type(placement["actor_kind"])
    requested_xyz = [float(placement["position"][axis]) for axis in ("x", "y", "z")]
    expected_yaw = float(placement["yaw_deg"]) + float(placement.get("planned_bbox", {}).get("heading_offset_deg", 0.))
    matches = []
    for identity in sorted(set(inventory)-before_ids):
        current = inventory[identity]
        if getattr(current.actor_info, "object_type", None) != expected_type:
            continue
        if _model(grpc, current, deadline=deadline) != placement["model"]:
            continue
        observed = _observation(current)
        z_tolerance = VEHICLE_GROUND_SNAP_Z_TOLERANCE_M if placement["actor_kind"] == "vehicle" else SPAWN_TOLERANCE_M
        if (math.dist(observed["position_xyz"][:2], requested_xyz[:2]) > SPAWN_TOLERANCE_M
                or abs(observed["position_xyz"][2]-requested_xyz[2]) > z_tolerance
                or _angle_distance(observed["raw_bbox_normalized"]["heading_deg"], expected_yaw) > 3.):
            continue
        source = getattr(current.actor_info, "client_key", "")
        if source and source != entry["source_client_key"]:
            raise RuntimeError("Matching spawn global readback names a foreign client namespace")
        matches.append(identity)
    if not matches and not set(inventory).difference(before_ids):
        raise _SpawnReadbackPending("Recorded spawn is not visible in fresh global inventory yet")
    if len(matches) != 1:
        raise RuntimeError("Recorded spawn cannot be bound to one unique new global actor")
    if diagnostics is not None:
        diagnostics.update(lookup_scope="recorded spawn plus fresh global inventory delta",
                           matching_global_actor_ids=matches)
    entry["spawn_global_binding_verified"] = True
    return matches[0]


def _observation(state):
    position = _xyz(state.transform.location, "actor position")
    rotation = _xyz(state.transform.rotation, "actor rotation")
    box = state.bounding_box
    extent = _xyz(box.extent, "world bounding box extent")
    if min(extent) <= 0:
        raise RuntimeError("Invalid world bounding box extent: " + _id(state))
    center = _xyz(box.location, "world bounding box center")
    box_rotation = _xyz(box.rotation, "world bounding box rotation")
    velocity = _xyz(state.global_velocity, "global actor velocity")
    angular_velocity = _xyz(state.angular_velocity, "angular actor velocity")
    return {"raw_actor_id": _id(state), "position_xyz": position,
            "rotation_xyz_deg": rotation, "global_velocity_raw": velocity,
            "speed_norm_raw": math.sqrt(sum(item * item for item in velocity)),
            "angular_velocity_raw": angular_velocity,
            "world_bbox": {"center_xyz": center, "extent_xyz": extent,
                           "rotation_xyz_deg": box_rotation},
            "raw_bbox_normalized": {"center_xyz": center,
                                    "width_m": 2 * extent[0], "length_m": 2 * extent[1],
                                    "heading_deg": box_rotation[2]}}


def _angle_distance(first, second):
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _stationary(observed, reference, label):
    if observed["speed_norm_raw"] > MAX_SPEED_RAW:
        raise RuntimeError(label + " is moving; global_velocity_raw=" + repr(observed["global_velocity_raw"]))
    if math.sqrt(sum(item * item for item in observed["angular_velocity_raw"])) > 0.05:
        raise RuntimeError(label + " is rotating; angular_velocity_raw=" + repr(observed["angular_velocity_raw"]))
    if reference is None:
        return
    for key in ("position_xyz",):
        if math.dist(observed[key], reference[key]) > POSE_TOLERANCE_M:
            raise RuntimeError(label + " pose moved; observed=" + repr(observed[key])
                               + "; reference=" + repr(reference[key]))
    for key in ("center_xyz", "extent_xyz"):
        if math.dist(observed["world_bbox"][key], reference["world_bbox"][key]) > POSE_TOLERANCE_M:
            raise RuntimeError(label + " world bounding box changed")
    for observed_rotation, reference_rotation in (
        (observed["rotation_xyz_deg"], reference["rotation_xyz_deg"]),
        (observed["world_bbox"]["rotation_xyz_deg"], reference["world_bbox"]["rotation_xyz_deg"]),
    ):
        if any(_angle_distance(a, b) > ROTATION_TOLERANCE_DEG for a, b in zip(observed_rotation, reference_rotation)):
            raise RuntimeError(label + " orientation moved")


def _park(grpc, ego, report, reference=None):
    from ...segment_modes import tunnel_static_startup as startup
    if startup.common_brake_hold_enabled(grpc):
        # An ongoing preview must reject drift before issuing any command that
        # could conceal it. Initial/final holds retain the existing bounded wait.
        if reference is not None:
            startup.common_stopped_readback(grpc, reference)
        startup.hold_common_brake(grpc)
        report["ego_parking_commands"] += 1
        report["ego_parking_policy"] = "verified_common_brake_hold"
        attempts = 20 if reference is None else 1
        last_error = None
        for attempt in range(attempts):
            try:
                return startup.common_stopped_readback(grpc, reference)
            except RuntimeError as error:
                last_error = error
                if attempt + 1 < attempts:
                    time.sleep(.1)
        raise RuntimeError("Common Ego brake hold was not observed: " + str(last_error))
    from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_P

    if reference is not None:
        previous = _read_one(grpc, ego)
        _stationary(_observation(previous), reference, "Ego")
        if int(previous.vehicle_state.gear_mode) != int(GEAR_MODE_P):
            raise RuntimeError("Ego left P gear during the preview")
    velocity_ok = bool(ego.set_velocity(0.0))
    gear_ok = bool(ego.set_vehicle_gear(GEAR_MODE_P))
    report["ego_parking_commands"] += 1
    if not (velocity_ok and gear_ok):
        raise RuntimeError("Ego zero velocity / P gear command was rejected")
    last_error = None
    # StartSimulation places a physical car on the road before its suspension
    # settles. Only the initial/final parking handshake may wait for that;
    # ongoing previews must reject motion immediately, before resetting speed.
    attempts = 20 if reference is None else 1
    for attempt in range(attempts):
        state = _read_one(grpc, ego)
        observed = _observation(state)
        try:
            _stationary(observed, None, "Ego")
            if int(state.vehicle_state.gear_mode) != int(GEAR_MODE_P):
                raise RuntimeError("Ego P gear was not observed")
            return state, observed
        except RuntimeError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(.1)
    raise RuntimeError(str(last_error) + "; global_velocity_raw=" + repr(observed["global_velocity_raw"])
                       + "; angular_velocity_raw=" + repr(observed["angular_velocity_raw"])
                       + "; position_xyz=" + repr(observed["position_xyz"]))


def _resolve_raw(grpc, handle, inventory, before_ids, requested_model, deadline=None,
                 *, placement=None, owned_entry=None, diagnostics=None):
    """Resolve client request IDs to authoritative global server IDs."""
    try:
        owned_state = _read_one(grpc, handle, deadline=deadline)
    except Exception as error:
        if not _unsupported_actor_lookup(error):
            raise
        if diagnostics is not None:
            diagnostics["alias_lookup_error"] = str(error)
        return _recorded_spawn_global_binding(grpc, handle, inventory, before_ids,
            placement, owned_entry, diagnostics, deadline)
    if owned_entry is not None:
        _fixture_namespace(grpc, [owned_entry])
        returned_key = getattr(owned_state.actor_info, "client_key", "")
        if returned_key and returned_key != owned_entry["source_client_key"]:
            raise RuntimeError("Owned alias readback names a foreign client namespace")
    identity = _id(owned_state)
    if identity in before_ids:
        raise _OwnedBaselineOverlap(identity)
    if identity in inventory and identity not in before_ids:
        state = inventory[identity]
        if _model(grpc, state, deadline=deadline) != requested_model:
            raise RuntimeError("Spawned actor model differs from the requested model")
        return identity
    observation = _observation(owned_state)
    matches = []
    for identity, state in inventory.items():
        if identity in before_ids:
            continue
        item = _observation(state)
        if (math.dist(item["position_xyz"], observation["position_xyz"]) < .01
                and math.dist(item["world_bbox"]["center_xyz"], observation["world_bbox"]["center_xyz"]) < .01
                and math.dist(item["world_bbox"]["extent_xyz"], observation["world_bbox"]["extent_xyz"]) < .01
                and _angle_distance(item["world_bbox"]["rotation_xyz_deg"][2], observation["world_bbox"]["rotation_xyz_deg"][2]) < .01
                and _model(grpc, state, deadline=deadline) == requested_model):
            matches.append(identity)
    if not matches and not set(inventory).difference(before_ids) and deadline is not None:
        raise _SpawnReadbackPending("Owned actor is not visible in global inventory yet")
    if len(matches) != 1:
        raise RuntimeError("Owned request ID cannot be resolved uniquely to global actor inventory")
    return matches[0]


def _wait_spawn_readback(grpc, handle, before_ids, requested_model, report,
                         diagnostics, ego_raw_id, ego_reference, *, placement=None, owned_entry=None):
    """Re-query one recorded spawn without issuing another creation request."""
    from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_P
    from ...segment_modes import tunnel_static_startup as startup

    started = time.monotonic()
    deadline = started + SPAWN_READBACK_TIMEOUT_SEC
    diagnostics.update(status="waiting", attempts=0, raw_actor_id=None,
                       max_attempts=SPAWN_READBACK_ATTEMPTS,
                       timeout_sec=SPAWN_READBACK_TIMEOUT_SEC, retry_errors=[])
    try:
        for attempt in range(SPAWN_READBACK_ATTEMPTS):
            diagnostics["attempts"] = attempt + 1
            fresh = _inventory(grpc, report, deadline=deadline)
            if ego_raw_id not in fresh:
                raise RuntimeError("Global inventory is missing the known Ego")
            _stationary(_observation(fresh[ego_raw_id]), ego_reference, "Ego")
            if startup.common_brake_hold_enabled(grpc):
                from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_D
                startup.common_stopped_readback(grpc, ego_reference)
                vehicle = fresh[ego_raw_id].vehicle_state
                if (int(vehicle.gear_mode) != int(GEAR_MODE_D)
                        or not math.isfinite(float(vehicle.brake)) or float(vehicle.brake) < .99
                        or not math.isfinite(float(vehicle.throttle)) or abs(float(vehicle.throttle)) > .01):
                    raise RuntimeError("Ego left the verified brake hold during initial spawn registration")
            elif int(fresh[ego_raw_id].vehicle_state.gear_mode) != int(GEAR_MODE_P):
                raise RuntimeError("Ego left P gear during initial spawn registration")
            try:
                raw_id = _resolve_raw(grpc, handle, fresh, before_ids, requested_model, deadline=deadline,
                    placement=placement, owned_entry=owned_entry, diagnostics=diagnostics)
            except Exception as error:
                if not (isinstance(error,_SpawnReadbackPending)
                        or placement is not None and owned_entry is not None and _unsupported_actor_lookup(error)):
                    raise
                diagnostics["retry_errors"].append(str(error))
                remaining = deadline - time.monotonic()
                if attempt + 1 == SPAWN_READBACK_ATTEMPTS or remaining <= 0:
                    raise RuntimeError("Initial spawn registration did not become visible within its retry budget") from error
                time.sleep(min(SPAWN_READBACK_RETRY_SEC, remaining))
                continue
            diagnostics.update(status="verified", raw_actor_id=raw_id)
            return fresh, raw_id
    except BaseException as error:
        diagnostics.update(status="failed", failure_reason=str(error) or type(error).__name__)
        raise
    finally:
        diagnostics["elapsed_sec"] = time.monotonic() - started


def _handle_for_request(grpc, kind, request_id):
    if kind == "vehicle":
        from api.vehicle import Vehicle
        return Vehicle(grpc.world._sim_adapter, grpc.client_key, request_id)
    from api.obstacle import Obstacle
    return Obstacle(grpc.world._sim_adapter, grpc.client_key, request_id)


def _conservative_box(planned, raw):
    """The readback may enlarge, but never shrink, a reviewed surface envelope."""
    return conservative_box_union(planned, raw)


def _opposing_lateral(geometry, station):
    point, _ = route_pose(geometry["own_points"], station)
    opposing_station, _, _ = project_polyline(geometry["opposing_points"], *point[:2])
    opposing, _ = route_pose(geometry["opposing_points"], opposing_station)
    return project_polyline(geometry["own_points"], *opposing[:2])[1]


def _occupancy(inventory, ego_id, owned_ids, geometry):
    blockers = []
    for identity, state in inventory.items():
        if identity == ego_id or identity in owned_ids:
            continue
        observed = _observation(state)
        box = observed["raw_bbox_normalized"]
        radius = .5 * math.hypot(box["width_m"], box["length_m"]) + .5
        # Guard both lanes plus the baseline's 40 m approach and 30 m exit.
        for key, width in (("own_points", geometry["own_lane_width_m"]),
                           ("opposing_points", geometry["opposing_lane_width_m"])):
            points = geometry[key]
            first, second = points[:2]
            penultimate, last = points[-2:]
            start_length = math.dist(first[:2], second[:2])
            end_length = math.dist(penultimate[:2], last[:2])
            start = tuple(first[axis] - 40. * (second[axis] - first[axis]) / start_length for axis in range(3))
            end = tuple(last[axis] + 30. * (last[axis] - penultimate[axis]) / end_length for axis in range(3))
            _, _, distance = project_polyline([start, *points, end], *box["center_xyz"][:2])
            if distance <= width * .5 + radius + .5:
                blockers.append(identity)
                break
    if blockers:
        raise RuntimeError("Foreign actors occupy the tunnel preview corridor: " + ", ".join(blockers))


def _geometry_checks(boxes, ego_box, geometry):
    expanded = geometry.get("placement_domain") == "two_lane_physical_road"
    if expanded:
        from shapely.geometry import Polygon
        boundary = geometry.get("road_boundary")
        if boundary is None:
            raise RuntimeError("Expanded placement requires the shared mapped road boundary")
        for box in boxes:
            corners = footprint_corners(box["center_xyz"], box["width_m"]+.30,
                                        box["length_m"]+.30, box["heading_deg"])
            if not boundary.covers_geometry(Polygon(corners)):
                raise RuntimeError("Observed expanded footprint lacks the shared road margin")
            if any(not 0 < project_polyline(geometry["own_points"], *corner)[0] < geometry["length_m"]
                   for corner in corners):
                raise RuntimeError("Observed expanded footprint crosses a mapped tunnel portal")
            if boxes_overlap(box, ego_box, margin_m=.25):
                raise RuntimeError("Observed actor overlaps the parked Ego")
        for index, first in enumerate(boxes):
            for second in boxes[index+1:]:
                if boxes_overlap(first, second, margin_m=.25):
                    raise RuntimeError("Observed candidate actor footprints overlap")
        return
    ego_width = max(ego_box["width_m"], 2.2)
    if ego_width + .5 > geometry["opposing_lane_width_m"]:
        raise RuntimeError("Nominal opposing lane does not provide full Ego-width clearance")
    for box in boxes:
        corners = footprint_corners(box["center_xyz"], box["width_m"], box["length_m"], box["heading_deg"])
        for corner in corners:
            station, lateral, _ = project_polyline(geometry["own_points"], *corner)
            if not .15 < station < geometry["length_m"] - .15:
                raise RuntimeError("Observed footprint crosses a nominal tunnel portal")
            if not any(project_polyline(geometry[key], *corner)[2] <= width * .5
                       for key, width in (("own_points", geometry["own_lane_width_m"]),
                                          ("opposing_points", geometry["opposing_lane_width_m"]))):
                raise RuntimeError("Observed footprint exceeds the nominal road boundary")
            opposing_lateral = _opposing_lateral(geometry, station)
            low = min(-geometry["own_lane_width_m"] * .5, opposing_lateral - geometry["opposing_lane_width_m"] * .5)
            high = max(geometry["own_lane_width_m"] * .5, opposing_lateral + geometry["opposing_lane_width_m"] * .5)
            if not low + .15 <= lateral <= high - .15:
                raise RuntimeError("Observed footprint lacks the nominal outer-road margin")
        station, _, _ = project_polyline(geometry["own_points"], *box["center_xyz"][:2])
        opposing_lateral = _opposing_lateral(geometry, station)
        laterals = [project_polyline(geometry["own_points"], *corner)[1] for corner in corners]
        outer = opposing_lateral + math.copysign(geometry["opposing_lane_width_m"] * .5, opposing_lateral)
        remaining = outer - max(laterals) if opposing_lateral > 0 else min(laterals) - outer
        if remaining < ego_width + .5:
            raise RuntimeError("Observed footprint removes the full Ego-width opposing-lane corridor")
        if boxes_overlap(box, ego_box, margin_m=.25):
            raise RuntimeError("Observed actor overlaps the parked Ego")
    for index, first in enumerate(boxes):
        for second in boxes[index+1:]:
            if boxes_overlap(first, second, margin_m=.25):
                raise RuntimeError("Observed candidate actor footprints overlap")


def _native_npc_route(map_loader, placement, geometry):
    """Keep opposing stationary NPCs on the actual reverse-flow lane."""
    lane = placement.get("placement_lane_id", "A2256W000126")
    following = {"A2256W000126": "A2256W000128", "A2256W000131": "A2256W000129"}
    if lane not in following:
        raise RuntimeError("Stationary NPC requires an exact mapped tunnel lane")
    position = [placement["position"][axis] for axis in ("x", "y")]
    if lane == "A2256W000126" and "placement_lane_id" not in placement:
        points = geometry["own_points"]
    else:
        current, departure = map_loader.get_link(lane), map_loader.get_link(following[lane])
        if (current.get("to_node_idx") != departure.get("from_node_idx")
                or math.dist(current["points"][-1][:2], departure["points"][0][:2]) > .05):
            raise RuntimeError("Stationary NPC lane and departure are not connected in native flow")
        points = current["points"]
        center = bbox_world(placement)["center_xyz"]
        station = project_polyline(points, *center[:2])[0]
        _, heading = route_pose(points, station)
        if _angle_distance(bbox_world(placement)["heading_deg"], math.degrees(heading)) > 3.:
            raise RuntimeError("Stationary NPC heading opposes its mapped native lane flow")
    waypoint = min(range(len(points)), key=lambda index: math.dist(points[index][:2], position))
    return [lane, following[lane]], waypoint


def _check_vehicle_profile(placement, observed):
    profile = placement.get("expected_api_raw")
    if not profile:
        return {"checked": False}
    size = profile["size_lwh_m"]
    extent = observed["world_bbox"]["extent_xyz"]
    actual_size = [2 * extent[1], 2 * extent[0], 2 * extent[2]]
    if max(abs(float(a) - float(b)) for a, b in zip(size, actual_size)) > .10:
        raise RuntimeError("Observed NPC raw collider dimensions differ from the reviewed vehicle profile")
    angle = math.radians(observed["rotation_xyz_deg"][2])
    dx, dy, dz = (a - b for a, b in zip(observed["world_bbox"]["center_xyz"], observed["position_xyz"]))
    # Reviewed profiles use actor FLU. Horizontal yaw-only comparison is valid
    # for the tunnel's level road; refuse tilted poses instead of assuming it.
    if any(_angle_distance(value, 0.) > 3. for value in observed["rotation_xyz_deg"][:2]):
        raise RuntimeError("NPC roll/pitch exceeds the reviewed level-road profile")
    offset = [dx * math.cos(angle) + dy * math.sin(angle), -dx * math.sin(angle) + dy * math.cos(angle), dz]
    expected_offset = profile["center_offset_actor_flu_m"]
    if max(abs(float(a) - float(b)) for a, b in zip(offset, expected_offset)) > .15:
        raise RuntimeError("Observed NPC collider offset differs from the reviewed vehicle profile")
    return {"checked": True, "actual_size_lwh_m": actual_size, "actual_center_offset_actor_flu_m": offset}


def _check_initial_vehicle_dimensions(placement, observed):
    profile = placement.get("expected_api_raw")
    if not profile:
        return {"checked": False}
    extent = observed["world_bbox"]["extent_xyz"]
    size = [2 * extent[1], 2 * extent[0], 2 * extent[2]]
    if max(abs(float(a) - float(b)) for a, b in zip(profile["size_lwh_m"], size)) > .10:
        raise RuntimeError("Observed NPC raw collider dimensions differ from the reviewed vehicle profile")
    return {"checked": False, "dimensions_checked": True, "actual_size_lwh_m": size,
            "offset_check_pending_native_route": True}


def _check_npc_api_variant(placement, observed, exact_observed):
    # The known pivot-centred API variant does not permit a tilted vehicle.
    # Keep this gate outside the offset fallback's exception handler.
    for state in (observed, exact_observed):
        if any(_angle_distance(value, 0.) > 3. for value in state["rotation_xyz_deg"][:2]):
            raise RuntimeError("NPC roll/pitch exceeds the reviewed level-road profile")
    for field in ("center_xyz", "extent_xyz"):
        if math.dist(observed["world_bbox"][field], exact_observed["world_bbox"][field]) > .01:
            raise RuntimeError("Owned and global NPC bounding box readbacks disagree")
    if any(_angle_distance(a, b) > .05 for a, b in zip(
            observed["world_bbox"]["rotation_xyz_deg"], exact_observed["world_bbox"]["rotation_xyz_deg"])):
        raise RuntimeError("Owned and global NPC bounding box rotations disagree")
    try:
        result = _check_vehicle_profile(placement, observed)
        _check_vehicle_profile(placement, exact_observed)
        return dict(result, catalog_offset_verified=result.get("checked") is True,
                    current_api_center_variant="reviewed_profile")
    except RuntimeError:
        source = placement.get("geometry_source", {})
        if (placement.get("scale") != [1., 1., 1.]
                or source.get("type") != "reviewed_vehicle_surface_and_api_collider_union"
                or source.get("model_key") != "npc|" + placement["model"] + "|1,1,1"
                or len(str(source.get("registry_sha256", ""))) != 64):
            raise
        if (math.dist(observed["world_bbox"]["center_xyz"], observed["position_xyz"]) > .01
                or math.dist(exact_observed["world_bbox"]["center_xyz"], exact_observed["position_xyz"]) > .01):
            raise
        dimensions = _check_initial_vehicle_dimensions(placement, observed)
        if dimensions.get("dimensions_checked") is not True:
            raise
        _check_initial_vehicle_dimensions(placement, exact_observed)
        return {"checked": False, "catalog_offset_verified": False,
                "current_api_center_variant": "actor_pivot", "dimensions_checked": True,
                "direct_and_global_query_agreement": True,
                "actual_size_lwh_m": dimensions["actual_size_lwh_m"],
                "physical_surface_source": "reviewed native-scale body and collider envelopes transformed with observed actor pose",
                "physical_mesh_verified": False, "collision_behavior_verified": False}


def _reviewed_collider_at_pose(placement, observed):
    profile = placement.get("expected_api_raw")
    if not profile:
        return None
    # Expand horizontal dimensions by the modest roll/pitch projection rather
    # than losing the vertical collider contribution on a sloping road.
    roll, pitch, yaw = (math.radians(value) for value in observed["rotation_xyz_deg"])
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))
    offset = profile["center_offset_actor_flu_m"]
    center = [observed["position_xyz"][axis] + sum(rotation[axis][column] * offset[column] for column in range(3)) for axis in range(3)]
    length, width, height = profile["size_lwh_m"]
    return {"center_xyz": center, "width_m": abs(cr)*width+abs(sr)*height,
            "length_m": abs(cp)*length+abs(sp)*(abs(sr)*width+abs(cr)*height),
            "heading_deg": math.degrees(yaw)}


def _persist(report, directory):
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "preview.json"
    report["report_path"] = str(path)
    temporary = directory / "preview.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ego_footprint(observed):
    """Measured body relative to the actual Ego transform, including overhang."""
    box = observed["raw_bbox_normalized"]
    yaw = math.radians(float(observed["rotation_xyz_deg"][2]))
    dx, dy = (float(a)-float(b) for a, b in zip(box["center_xyz"][:2], observed["position_xyz"][:2]))
    roll, pitch = (math.radians(float(value)) for value in observed["rotation_xyz_deg"][:2])
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))
    delta = [float(a)-float(b) for a, b in zip(box["center_xyz"], observed["position_xyz"])]
    intrinsic = [sum(rotation[row][column]*delta[row] for row in range(3)) for column in range(3)]
    return {"length_m": float(box["length_m"]), "width_m": float(box["width_m"]),
            "height_m": 2.*float(observed["world_bbox"]["extent_xyz"][2]),
            "center_actor_flu_m": intrinsic, "max_abs_roll_pitch_deg": 3.,
            "center_forward_m": dx*math.cos(yaw)+dy*math.sin(yaw),
            "center_left_m": -dx*math.sin(yaw)+dy*math.cos(yaw)}


def _fixture_namespace(grpc, owned):
    key = getattr(grpc, "client_key", None)
    if (not isinstance(key, str) or not key.strip() or key != key.strip()
            or key == "tunnel_static_trial_runner" or hasattr(grpc, "fixture_client_key")):
        raise RuntimeError("Cleanup requires the original nonempty fixture-owner namespace")
    for entry in owned:
        request = entry["handle"].get_object_info()
        if (entry.get("source_client_key") != key or getattr(request, "client_key", None) != key
                or str(request.id.value) != entry.get("source_handle_id")):
            raise RuntimeError("Owned request no longer matches its pinned fixture namespace")
    return key


def _validated_global_actor_state_once(grpc, entry, fixture_key, *, report=None, deadline=None):
    """Keep native addressing separate from the creating-handle ownership pin."""
    _fixture_namespace(grpc, [entry])
    if entry["source_client_key"] != fixture_key:
        raise RuntimeError("Owned actor handle differs from its pinned fixture namespace")
    pinned = copy.deepcopy(entry["raw_object_info"])
    address = _object_address(pinned)
    if address["id"] != entry["raw_id"]:
        raise RuntimeError("Pinned native cleanup identity differs from the verified raw actor ID")
    if address["client_key"] and address["client_key"] != fixture_key:
        raise RuntimeError("Pinned global address names a foreign client namespace")
    requested_model = entry.get("requested_model")
    if not requested_model:
        raise RuntimeError("Owned actor binding lacks its requested model pin")
    expected_type = _actor_type(entry["actor_kind"])
    evidence = {"lookup_scope": "original pinned global ObjectInfo; creating namespace is ownership only",
                "native_request": address, "creating_client_key": fixture_key,
                "creating_alias": entry["source_handle_id"], "snapshots":{},"timings":{}}
    entry["lookup_evidence"] = evidence
    read_started = time.monotonic()
    fresh = _inventory(grpc, report if report is not None else {"inventory_checks":0},deadline=deadline)
    evidence["timings"]["global_inventory"] = {"started":read_started,"finished":time.monotonic()}
    if entry["raw_id"] not in fresh:
        raise RuntimeError("Exact pinned raw actor is absent from fresh global inventory")
    current = fresh[entry["raw_id"]]
    evidence["snapshots"]["global_inventory"] = _observation(current)
    if _object_address(current.actor_info) != address:
        raise RuntimeError("Pinned global actor address or type changed")
    read_started = time.monotonic()
    current_model = _model(grpc,current,deadline=deadline)
    evidence["timings"]["model"] = {"started":read_started,"finished":time.monotonic()}
    if address["object_type"] != expected_type or current_model != requested_model:
        raise RuntimeError("Pinned global actor model or type changed")
    if entry.get("binding_phase") != "initial_before_parking":
        _stationary(_observation(current), entry.get("raw_observation"), "Pinned fixture actor")
    allowed_ids = {entry["raw_id"],entry["source_handle_id"],entry["request_id"]}
    observations = []
    for scope, target in (("creating_alias", entry["handle"]),
                          ("pinned_global_address", SimpleNamespace(get_object_info=lambda: pinned))):
        read_started = time.monotonic()
        try:
            observed = _read_one(grpc,target,deadline=deadline)
        except Exception as error:
            evidence["timings"][scope] = {"started":read_started,"finished":time.monotonic(),"error":str(error)}
            lookup_error = (error.__cause__ if scope=="pinned_global_address"
                and isinstance(error,_SpawnReadbackPending) and error.__cause__ is not None else error)
            if not _unsupported_actor_lookup(lookup_error,known_global=scope=="pinned_global_address"):
                raise
            evidence[scope] = {"unsupported_lookup":str(error),"supplement":"fresh exact global inventory"}
            continue
        evidence["timings"][scope] = {"started":read_started,"finished":time.monotonic()}
        evidence["snapshots"][scope] = _observation(observed)
        returned = _object_address(observed.actor_info)
        evidence[scope] = {"returned":returned,
            "returned_id_basis":"pinned raw ID" if returned["id"]==entry["raw_id"] else "exact creating alias"}
        if (returned["id"] not in allowed_ids or returned["object_type"] != expected_type
                or returned["client_key"] and returned["client_key"] != fixture_key):
            raise RuntimeError("Owned readback identity/type/client key differs from its pins")
        matches = [identity for identity,actor in fresh.items()
            if getattr(actor.actor_info,"object_type",None)==expected_type and _same_actor_body(actor,observed)]
        if any(identity!=entry["raw_id"] for identity in matches):
            raise RuntimeError("Owned alias readback is ambiguous or matches another global actor")
        differences = _body_differences(observed,current)
        evidence["body_deltas"] = differences
        evidence[scope]["body_deltas"] = differences
        if differences["bbox_extent_distance_m"] >= .01:
            raise RuntimeError("Owned alias/global bbox extent differs from its pinned actor")
        if not _same_actor_body(observed,current):
            raise _InitialBodySettlingMismatch("Owned alias/global body differs from its exact pinned global actor")
        observations.append(observed)
    # No new identity is selected here: the raw ID was uniquely bound at spawn.
    # Every supported individual response must agree with the current global ID.
    comparison = observations[0] if observations else current
    same = [identity for identity,actor in fresh.items()
        if getattr(actor.actor_info,"object_type",None)==expected_type
        and _same_actor_body(actor,comparison)]
    if same != [entry["raw_id"]]:
        raise RuntimeError("Owned alias readback is ambiguous in fresh global inventory")
    if deadline is not None and time.monotonic() >= deadline:
        raise RuntimeError("Actor body proof exceeded its readback deadline")
    evidence["verified_global_actor_id"] = entry["raw_id"]
    entry["cleanup_identity_basis"] = "creating request ownership pin plus fresh exact global ID/model/type/body proof"
    return current


def validated_global_actor_state(grpc, entry, fixture_key, *, report=None, deadline=None):
    """Re-observe only initial settling; frozen identity/body proof fails fast."""
    initial = entry.get("binding_phase") == "initial_before_parking"
    started = time.monotonic()
    if initial:
        initial_deadline = started + INITIAL_BODY_READBACK_TIMEOUT_SEC
        deadline = initial_deadline if deadline is None else min(deadline,initial_deadline)
    attempts, history = [], entry.setdefault("lookup_history",[])
    maximum = INITIAL_BODY_READBACK_ATTEMPTS if initial else 1
    for index in range(maximum):
        entry["lookup_evidence"] = {"status":"checking","binding_phase":entry.get("binding_phase","frozen")}
        try:
            result = _validated_global_actor_state_once(grpc,entry,fixture_key,report=report,deadline=deadline)
        except Exception as error:
            evidence = entry.setdefault("lookup_evidence",{})
            evidence.update(status="failed",failure_reason=str(error),elapsed_sec=time.monotonic()-started)
            attempts.append(copy.deepcopy(evidence))
            retry = initial and isinstance(error,_InitialBodySettlingMismatch)
            if not retry or index+1 == maximum or time.monotonic() >= deadline:
                evidence["attempts"] = attempts
                history.append(copy.deepcopy(evidence))
                raise
            time.sleep(min(SPAWN_READBACK_RETRY_SEC,max(0.,deadline-time.monotonic())))
            continue
        evidence = entry["lookup_evidence"]
        evidence.update(status="verified",elapsed_sec=time.monotonic()-started)
        attempts.append(copy.deepcopy(evidence))
        evidence["attempts"] = attempts
        history.append(copy.deepcopy(evidence))
        return result


def _owned_raw_info(grpc, entry, fixture_key, *, report=None, deadline=None):
    validated_global_actor_state(grpc,entry,fixture_key,report=report,deadline=deadline)
    info = copy.deepcopy(entry["raw_object_info"])
    return info, getattr(info,"client_key","")


def cleanup_owned_actors(grpc, owned, baseline_inventory, report):
    """Use bounded native ACKs, then prove exact absence and foreign retention."""
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    from proto.morai.common.type_pb2 import StringValue
    cleanup = report["cleanup"]
    baseline = set(baseline_inventory or {})
    baseline_foreign = baseline - {"Ego"}
    cleanup.update(baseline_foreign_actor_ids=sorted(baseline_foreign),
                   ownership_scope_basis="exact creating-handle client key; global inventory is not ownership proof")
    errors, individual_errors = [], []
    cleanup["individual_delete_errors"] = individual_errors
    fixture_key = _fixture_namespace(grpc, owned) if owned else None
    if owned:
        cleanup["fixture_client_key"] = fixture_key
        if any(entry.get("forbidden_baseline_actor_id") for entry in owned):
            raise RuntimeError("Owned request overlaps a preserved baseline actor; namespace cleanup refused")
        if any(entry["raw_id"] in baseline for entry in owned if entry["raw_id"] is not None):
            raise RuntimeError("Owned cleanup raw ID overlaps the pre-spawn baseline")
        for identity, actor in (baseline_inventory or {}).items():
            if identity != "Ego" and getattr(actor.actor_info, "client_key", "") == fixture_key:
                raise RuntimeError("Fixture namespace contains a preserved baseline actor; bulk cleanup refused")
        # Validate every known binding before issuing the first destructive RPC.
        pinned = {}
        bindings = []
        cleanup["owned_request_bindings"] = bindings
        for entry in owned:
            binding = {"request_id": entry["request_id"], "raw_actor_id": entry["raw_id"],
                       "source_client_key": entry["source_client_key"], "verified": False}
            bindings.append(binding)
            if entry["raw_id"] is None:
                binding["basis"] = "unresolved request; only original namespace may be cleaned"
                continue
            try:
                info, returned_key = _owned_raw_info(grpc, entry, fixture_key, report=report)
            except Exception as error:
                binding["error"] = str(error)
                binding["lookup_evidence"] = copy.deepcopy(entry.get("lookup_evidence",{}))
                binding["lookup_history"] = copy.deepcopy(entry.get("lookup_history",[]))
                raise
            if getattr(info, "object_type", None) != getattr(entry.get("raw_object_info", info), "object_type", None):
                raise RuntimeError("Owned actor type changed before cleanup")
            pinned[entry["request_id"]] = (info, returned_key)
            binding.update(verified=True, basis=entry["cleanup_identity_basis"], returned_client_key=returned_key,
                           lookup_evidence=copy.deepcopy(entry["lookup_evidence"]),
                           lookup_history=copy.deepcopy(entry.get("lookup_history",[])))
        stub = grpc.world._sim_adapter._actor_stub
        for entry in reversed(owned):
            row = {"request_id": entry["request_id"], "raw_actor_id": entry["raw_id"],
                   "source_client_key": entry["source_client_key"], "destroy_called": False}
            cleanup["actors"].append(row)
            if entry["raw_id"] is None:
                row["individual_delete"] = "unresolved_request; original namespace cleanup required"
                continue
            info, returned_key = pinned[entry["request_id"]]
            row.update(raw_object_info={"id": str(info.id.value), "client_key": info.client_key,
                "object_type": getattr(info, "object_type", None)}, returned_client_key=returned_key,
                identity_basis=entry["cleanup_identity_basis"],
                client_key_basis="original global lookup address; creating-handle ownership namespace is separate",
                lookup_evidence=copy.deepcopy(entry["lookup_evidence"]))
            try:
                row["destroy_called"] = True
                response = stub.DestroyActor(info, timeout=RPC_TIMEOUT_SEC)
                row["native_status"] = int(response.status) if response is not None else None
                row["native_description"] = str(getattr(response, "description", ""))
                row["destroy_accepted"] = response is not None and response.status == STATUS_CODE_SUCCESS
                if not row["destroy_accepted"]:
                    raise RuntimeError("Native actor delete rejected: " + row["native_description"])
            except Exception as error:
                row["destroy_error"] = str(error)
                individual_errors.append(str(error))
        namespace = {"client_key": fixture_key, "rpc_timeout_sec": RPC_TIMEOUT_SEC, "accepted": False}
        cleanup["namespace_cleanup"] = namespace
        try:
            response = stub.DestroyAllActors(StringValue(value=fixture_key), timeout=RPC_TIMEOUT_SEC)
            namespace.update(native_status=int(response.status) if response is not None else None,
                native_description=str(getattr(response, "description", "")),
                accepted=response is not None and response.status == STATUS_CODE_SUCCESS)
            if not namespace["accepted"]:
                raise RuntimeError("Native fixture namespace cleanup rejected: " + namespace["native_description"])
            namespace["settle_sec"] = CLEANUP_SETTLE_SEC
            time.sleep(CLEANUP_SETTLE_SEC)
        except Exception as error:
            namespace["error"] = str(error)
            errors.append(str(error))
    else:
        cleanup["namespace_cleanup"] = {"status": "not_needed", "reason": "no owned requests"}
    started = time.monotonic()
    deadline = started + CLEANUP_READBACK_TIMEOUT_SEC
    checks = {"attempts": 0, "timeout_sec": CLEANUP_READBACK_TIMEOUT_SEC,
              "max_attempts": CLEANUP_READBACK_ATTEMPTS, "verified": False}
    cleanup["global_readback"] = checks
    after = None
    for attempt in range(CLEANUP_READBACK_ATTEMPTS):
        checks["attempts"] = attempt + 1
        after = _inventory(grpc, report, deadline=deadline)
        missing = sorted(baseline - set(after))
        remaining = sorted(entry["raw_id"] for entry in owned if entry["raw_id"] is not None and entry["raw_id"] in after)
        checks.update(missing_baseline_actor_ids=missing, remaining_owned_actor_ids=remaining,
                      newly_observed_actor_ids=sorted(set(after)-baseline), elapsed_sec=time.monotonic()-started)
        if missing:
            raise RuntimeError("Preserved baseline actors disappeared during cleanup: " + ", ".join(missing))
        ambiguous = [entry for entry in owned if entry["raw_id"] is None]
        if not remaining and ambiguous:
            for entry in ambiguous:
                try:
                    actor = _read_one(grpc, entry["handle"], deadline=deadline)
                except Exception:
                    if set(after)-baseline:
                        raise RuntimeError("Ambiguous creation cleanup cannot be proven from global inventory")
                else:
                    if _id(actor) in after:
                        raise RuntimeError("Ambiguous owned request still exists after cleanup")
        if not remaining:
            checks["verified"] = True
            break
        if attempt + 1 == CLEANUP_READBACK_ATTEMPTS or time.monotonic() >= deadline:
            raise RuntimeError("Owned actors remain after cleanup: " + ", ".join(remaining))
        time.sleep(min(CLEANUP_READBACK_RETRY_SEC, max(0., deadline-time.monotonic())))
    cleanup["individual_delete_errors"] = individual_errors
    cleanup["individual_errors_recovered_by_verified_namespace"] = bool(individual_errors) and not errors and checks["verified"]
    if errors:
        raise RuntimeError("; ".join(errors))
    cleanup["verified"] = checks["verified"]


def preview_candidate(grpc, map_loader, candidate, hold_seconds=10.0, output_dir=None,
                      screen_fn=None, drive_fn=None):
    """Validate stationary actors, optionally run a verified trial, and clean up.

The default parked preview never starts Expert. A trial callback may run only
after measured geometry has produced a validated path. Unexpected failures,
including failed cleanup/readback, leave status=failed. Foreign actors are never
deleted or moved. The caller must exclude external control publishers first.
"""
    duration = float(hold_seconds)
    if not math.isfinite(duration) or not 0 <= duration <= 3600:
        raise ValueError("hold_seconds must be finite and in 0..3600")
    geometry = validate_candidate(candidate, map_loader)
    if geometry.get("placement_domain") == "two_lane_physical_road" and screen_fn is None:
        raise ValueError("Expanded placements require measured full-body production Jamming screening")
    directory = Path(output_dir) if output_dir is not None else None
    report = {"schema": "aimdrive_tunnel_static_preview_v1", "status": "started",
              "failure_reason": "", "expert_verified": False, "live_avoidance_verified": False,
              "physical_walls_verified": False, "physical_mesh_verified": False,
              "collision_behavior_verified": False, "planned": copy.deepcopy(candidate), "observed": [],
              "inventory_scope": "all clients, all actor kinds", "inventory_rpc_timeout_sec": RPC_TIMEOUT_SEC,
              "inventory_checks": 0, "ego_parking_commands": 0, "hold_seconds": duration,
              "stationary_samples": 0, "route_screened": False, "spawn_readbacks": [],
              "cleanup": {"verified": False, "actors": []}}
    owned, references = [], {}
    ego, ego_raw_id, ego_reference = None, None, None
    baseline_inventory = None
    interrupted = None
    _persist(report, directory)
    try:
        ego = grpc.get_ego()
        ego_state, ego_reference = _park(grpc, ego, report)
        report["ego_observed"] = copy.deepcopy(ego_reference)
        report["actual_ego_footprint"] = _ego_footprint(ego_reference)
        inventory = _inventory(grpc, report)
        ego_raw_id = _id(ego_state)
        if ego_raw_id not in inventory:
            # Ego has a fixed simulator identity; do not guess from model/pose.
            handle_id = str(ego.get_id())
            if handle_id not in inventory:
                raise RuntimeError("Global inventory is missing the known Ego")
            ego_raw_id = handle_id
        _occupancy(inventory, ego_raw_id, set(), geometry)
        baseline_ids = set(inventory)
        baseline_inventory = copy.deepcopy(inventory)
        for index, placement in enumerate(candidate["obstacles"]):
            _park(grpc, ego, report, reference=ego_reference)
            before = _inventory(grpc, report)
            _occupancy(before, ego_raw_id, {item["raw_id"] for item in owned if item["raw_id"]}, geometry)
            request_id = "tunnel_preview_" + uuid.uuid4().hex + "_" + str(index)
            kind = placement["actor_kind"]
            fallback = _handle_for_request(grpc, kind, request_id)
            entry = {"handle": fallback, "request_id": request_id, "raw_id": None,
                     "actor_kind":kind,"requested_model":placement["model"],"placement":copy.deepcopy(placement),
                     "source_handle_id": str(fallback.get_object_info().id.value),
                     "source_client_key": getattr(fallback.get_object_info(), "client_key", None)}
            _fixture_namespace(grpc, [entry])
            owned.append(entry)  # Track the exact owned request before an ambiguous RPC.
            transform = grpc.make_transform(*(placement["position"][axis] for axis in ("x", "y", "z")), placement["yaw_deg"])
            spawn_error = None
            try:
                if kind == "vehicle":
                    handle = grpc.world.spawn_vehicle(transform=transform, model_name=placement["model"],
                        label=request_id, request_id=request_id, velocity=0.0, pause=False, multi_ego=False)
                else:
                    from proto.morai.common.type_pb2 import Vector3
                    scale = Vector3(x=1., y=1., z=1.)
                    handle = grpc.world.spawn_obstacle(transform=transform, model_name=placement["model"],
                        label=request_id, request_id=request_id, scale=scale)
                if handle is not None:
                    if getattr(handle.get_object_info(), "client_key", None) != entry["source_client_key"]:
                        raise RuntimeError("Returned spawn handle differs from the pinned fixture namespace")
                    entry["handle"] = handle
                    entry["source_handle_id"] = str(handle.get_object_info().id.value)
            except Exception as error:
                spawn_error = str(error)
            handle = entry["handle"]
            readback = {"request_id": request_id, "spawn_rpc_error": spawn_error}
            report["spawn_readbacks"].append(readback)
            try:
                fresh, raw_id = _wait_spawn_readback(grpc, handle, set(before), placement["model"],
                    report, readback, ego_raw_id, ego_reference,placement=placement,owned_entry=entry)
            except Exception as error:
                if isinstance(error, _OwnedBaselineOverlap):
                    entry["forbidden_baseline_actor_id"] = error.actor_id
                raise RuntimeError("Candidate spawn could not be verified: " + str(error)
                    + ("; spawn RPC: " + spawn_error if spawn_error else "")) from error
            entry["raw_id"] = raw_id
            entry["raw_object_info"] = copy.deepcopy(fresh[raw_id].actor_info)
            entry["binding_phase"] = "initial_before_parking"
            try:
                entry["raw_object_info"], entry["raw_returned_client_key"] = _owned_raw_info(grpc, entry, entry["source_client_key"],report=report)
            finally:
                readback["lookup_evidence"] = copy.deepcopy(entry.get("lookup_evidence",{}))
            controls = {}
            original_physical_state = None
            original_physical_profile = None
            if kind == "vehicle":
                original_physical_state = _observation(fresh[raw_id])
                try:
                    original_physical_profile = _check_initial_vehicle_dimensions(placement, original_physical_state)
                except RuntimeError:
                    rejected = copy.deepcopy(original_physical_state)
                    rejected.update(actor_kind=kind, request_id=request_id, handle_id=str(handle.get_id()),
                                    stationary_controls_accepted=controls, spawn_rpc_error=spawn_error)
                    report["observed"].append(rejected)
                    raise
            if kind == "obstacle":
                # Reuse the existing freeze-and-reapply helper, feeding it bounded
                # global observations of the already proven run-owned raw ID.
                shim = SimpleNamespace(grpc=SimpleNamespace(get_all_obstacle_states=lambda: list(_inventory(grpc, report).values())))
                state, diagnostics = lock_static_obstacle_transform(shim, handle, raw_id, transform,
                    tuple(placement["position"][axis] for axis in ("x", "y", "z")), attempts=8, stable_samples=2, retry_seconds=.1)
                controls["static_transform_lock"] = diagnostics
                if state is None:
                    raise RuntimeError("Static obstacle could not be frozen at the requested pose")
            else:
                controls["speed_limit_zero"] = bool(handle.set_vehicle_dynamics_speed_limit(True, 0.0, reset=False))
                controls["velocity_zero"] = bool(handle.set_velocity(0.0))
                route, waypoint = _native_npc_route(map_loader, placement, geometry)
                controls["native_route"] = bool(grpc.set_vehicle_route(handle, route, decision_range=30.,
                    route_waypoint_indices={route[0]: waypoint}, label="tunnel_preview_parked_npc"))
                controls["default_ai_retained"] = True
                controls["default_physics_retained"] = True
                controls["native_route_links"] = route
                controls["native_route_start_waypoint"] = waypoint
                controls["brake_zero_speed"] = None
                if hasattr(handle, "control"):
                    from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_SPEED
                    controls["brake_zero_speed"] = bool(handle.control(long_cmd_type=LONG_CMD_TYPE_SPEED,
                        throttle=0., brake=1., steer=0., velocity=0., acceleration=0., frame=0))
                if not (controls["velocity_zero"] and controls["speed_limit_zero"] and controls["native_route"]):
                    raise RuntimeError("Parked NPC stationary commands were rejected")
            fresh = _inventory(grpc, report)
            if raw_id not in fresh:
                raise RuntimeError("Spawned candidate actor is missing after stationary commands")
            state = fresh[raw_id]
            observed = _observation(state)
            observed.update(actor_kind=kind, request_id=request_id, handle_id=str(handle.get_id()),
                            stationary_controls_accepted=controls, spawn_rpc_error=spawn_error)
            if kind == "vehicle":
                observed["spawn_physical_state_before_parking"] = original_physical_state
                observed["reviewed_profile_readback_before_parking"] = original_physical_profile
            report["observed"].append(observed)
            if kind == "vehicle":
                settled_reference = None
                stable_samples = 0
                for settle_attempt in range(40):
                    try:
                        _stationary(observed, settled_reference, "Candidate actor " + raw_id)
                        if settled_reference is None:
                            settled_reference = copy.deepcopy(observed)
                        stable_samples += 1
                        if stable_samples >= 8:
                            observed["initial_stationary_samples"] = stable_samples
                            break
                    except RuntimeError as error:
                        settled_reference = None
                        stable_samples = 0
                        if settle_attempt == 39:
                            raise
                    if settle_attempt == 39:
                        raise RuntimeError("Spawned NPC did not maintain a stable initial pose across eight fresh samples")
                    time.sleep(.1)
                    fresh = _inventory(grpc, report)
                    _occupancy(fresh, ego_raw_id, {item["raw_id"] for item in owned if item["raw_id"]}, geometry)
                    if raw_id not in fresh:
                        raise RuntimeError("Spawned NPC disappeared during its initial stationary handshake")
                    state = fresh[raw_id]
                    observed.update(_observation(state))
            _stationary(observed, None, "Candidate actor " + raw_id)
            requested_xyz = [float(placement["position"][axis]) for axis in ("x", "y", "z")]
            xy_error = math.dist(requested_xyz[:2], observed["position_xyz"][:2])
            z_error = abs(requested_xyz[2] - observed["position_xyz"][2])
            z_tolerance = VEHICLE_GROUND_SNAP_Z_TOLERANCE_M if kind == "vehicle" else SPAWN_TOLERANCE_M
            observed["spawn_pose_readback"] = {"requested_position_xyz": requested_xyz,
                "xy_error_m": xy_error, "z_error_m": z_error,
                "xy_tolerance_m": SPAWN_TOLERANCE_M, "z_tolerance_m": z_tolerance,
                "vehicle_initial_ground_snap_allowed": kind == "vehicle"}
            # Real NPC prefabs align their pivot to road/suspension height;
            # subsequent stability still compares the full observed 3D pose.
            if xy_error > SPAWN_TOLERANCE_M or z_error > z_tolerance:
                raise RuntimeError("Candidate actor pivot differs from its requested position")
            model = _model(grpc, state)
            if model != placement["model"]:
                raise RuntimeError("Candidate model readback changed")
            planned_box = bbox_world(placement)
            if _angle_distance(planned_box["heading_deg"], observed["raw_bbox_normalized"]["heading_deg"]) > 3.0:
                raise RuntimeError("Observed candidate heading differs from the reviewed model convention")
            effective_placement = copy.deepcopy(placement)
            effective_placement["position"] = dict(zip(("x", "y", "z"), observed["position_xyz"]))
            effective_placement["yaw_deg"] = observed["rotation_xyz_deg"][2] - float(placement["planned_bbox"]["heading_offset_deg"])
            reviewed_at_actual_pose = bbox_world(effective_placement)
            observed["reviewed_envelope_at_observed_pose"] = reviewed_at_actual_pose
            box = _conservative_box(_conservative_box(planned_box, reviewed_at_actual_pose), observed["raw_bbox_normalized"])
            reviewed_collider = _reviewed_collider_at_pose(placement, observed) if kind == "vehicle" else None
            if reviewed_collider is not None:
                observed["reviewed_collider_at_observed_pose"] = reviewed_collider
                box = _conservative_box(box, reviewed_collider)
            observed.update(box)
            observed.update(model=model, actor_kind=kind, request_id=request_id, handle_id=str(handle.get_id()),
                stationary_controls_accepted=controls, spawn_rpc_error=spawn_error,
                commanded_yaw_deg=float(placement["yaw_deg"]),
                observed_transform_yaw_offset_deg=(observed["rotation_xyz_deg"][2]-float(placement["yaw_deg"])+180.)%360.-180.,
                observed_bbox_yaw_offset_deg=(observed["world_bbox"]["rotation_xyz_deg"][2]-float(placement["yaw_deg"])+180.)%360.-180.,
                screening_geometry_source="union of original planned bounds, reviewed body and collider at observed pose, and raw world bbox")
            for metadata in ("placement_lane_id", "placement_region", "native_forward_yaw_deg", "static_yaw_offset_deg"):
                if metadata in placement:
                    observed[metadata] = copy.deepcopy(placement[metadata])
            if kind == "vehicle":
                observed["observed_gear_mode"] = int(state.vehicle_state.gear_mode)
                entry["raw_observation"] = copy.deepcopy(_observation(fresh[raw_id]))
                entry["binding_phase"] = "frozen"
                try:
                    exact_state = validated_global_actor_state(grpc,entry,entry["source_client_key"],report=report)
                finally:
                    observed["owned_actor_lookup_evidence"] = copy.deepcopy(entry.get("lookup_evidence",{}))
                exact_observed = _observation(exact_state)
                observed["owned_actor_state_at_validation"] = exact_observed
                observed["owned_actor_readback_basis"] = "validated fresh exact global state; individual getter support recorded separately"
                if math.dist(exact_observed["position_xyz"], observed["position_xyz"]) > .05:
                    raise RuntimeError("Owned and global NPC readbacks disagree on actor position")
                observed["reviewed_profile_readback"] = _check_npc_api_variant(placement, observed, exact_observed)
                supported = [scope for scope in ("creating_alias","pinned_global_address")
                    if "returned" in entry["lookup_evidence"].get(scope,{})]
                profile_readback = observed["reviewed_profile_readback"]
                if "direct_and_global_query_agreement" in profile_readback:
                    profile_readback["direct_and_global_query_agreement"] = bool(supported)
                profile_readback["query_agreement_evidence"] = {
                    "supported_direct_scopes":supported,
                    "fresh_global_inventory_actor_id":entry["lookup_evidence"]["verified_global_actor_id"],
                    "individual_getter_support_disclosed":True}
                observed.update(_conservative_box(box, exact_observed["raw_bbox_normalized"]))
            references[raw_id] = copy.deepcopy(observed)
            entry["raw_observation"] = copy.deepcopy(_observation(fresh[raw_id]))
            entry["binding_phase"] = "frozen"
            _occupancy(fresh, ego_raw_id, {item["raw_id"] for item in owned if item["raw_id"]}, geometry)
            _persist(report, directory)
        _geometry_checks(report["observed"], ego_reference["raw_bbox_normalized"], geometry)
        if screen_fn is not None:
            screening = screen_fn(candidate, map_loader, measured_geometry=report["observed"],
                                  actual_ego_state=ego_reference,
                                  actual_ego_footprint=report["actual_ego_footprint"])
            report["measured_route_screen"] = screening
            report["route_screened"] = True
            if (not isinstance(screening, dict) or screening.get("status") != "nominal_path_found"
                    or screening.get("path_found") is not True
                    or screening.get("geometry_checks", {}).get("passed") is not True):
                raise RuntimeError("Observed geometry did not produce an offline avoidance path")
        started = time.monotonic()
        while True:
            _, parked = _park(grpc, ego, report, reference=ego_reference)
            _stationary(parked, ego_reference, "Ego")
            fresh = _inventory(grpc, report)
            if ego_raw_id not in fresh:
                raise RuntimeError("Global inventory is missing the known Ego")
            _stationary(_observation(fresh[ego_raw_id]), ego_reference, "Ego")
            _occupancy(fresh, ego_raw_id, set(references), geometry)
            for identity, reference in references.items():
                if identity not in fresh:
                    raise RuntimeError("Candidate actor disappeared: " + identity)
                _stationary(_observation(fresh[identity]), reference, "Candidate actor " + identity)
                if _model(grpc, fresh[identity]) != reference["model"]:
                    raise RuntimeError("Candidate actor model changed: " + identity)
            report["stationary_samples"] += 1
            _persist(report, directory)
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                break
            time.sleep(min(CHECK_PERIOD_SEC, remaining))
        report["status"] = "valid_preview"
        report["stationary_observed_seconds"] = time.monotonic() - started
        if drive_fn is not None:
            if not report["route_screened"]:
                raise RuntimeError("Driving trial requires measured Expert path validation")
            report["owned_actor_geometry"] = copy.deepcopy(report["observed"])
            report["status"] = "driving_trial"
            _persist(report, directory)
            proof = drive_fn(report, owned)
            report["drive_result"] = proof
            required = ("completed", "collision_free", "returned_to_original_lane", "fixture_stationary")
            if (not isinstance(proof, dict) or proof.get("status") != "passed"
                    or any(proof.get(key) is not True for key in required)):
                raise RuntimeError("Driving trial did not provide complete passage/return evidence")
            report.update(status="valid_driving_trial", expert_verified=True, live_avoidance_verified=True)
    except BaseException as error:
        report["status"] = "failed"
        report["failure_reason"] = str(error) or type(error).__name__
        if not isinstance(error, Exception):
            interrupted = error
    finally:
        cleanup_errors = []
        if ego is not None:
            try:
                _park(grpc, ego, report)
            except Exception as error:
                cleanup_errors.append("Ego final parking: " + str(error))
        try:
            cleanup_owned_actors(grpc, owned, baseline_inventory, report)
            report["cleanup"]["verified"] = not cleanup_errors
        except Exception as error:
            cleanup_errors.append("Cleanup readback: " + str(error))
        if cleanup_errors:
            report["status"] = "failed"
            report["expert_verified"] = report["live_avoidance_verified"] = False
            report["cleanup"]["verified"] = False
            report["cleanup"]["errors"] = cleanup_errors
            report["failure_reason"] = "; ".join(filter(None, [report["failure_reason"], *cleanup_errors]))
        _persist(report, directory)
    if interrupted is not None:
        raise interrupted
    return report
