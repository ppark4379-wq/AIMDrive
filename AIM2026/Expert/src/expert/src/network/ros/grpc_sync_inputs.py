#!/usr/bin/env python3
"""Read one coherent planning snapshot from MORAI's read-only gRPC API."""

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import struct
import sys
import time

from ...obstacle.object_info import ObjectInfo
from ...vehicle_state import VehicleState


_OBJECT_TYPE_VEHICLE = 1
_OBJECT_TYPE_PEDESTRIAN = 2
_OBJECT_TYPE_OBSTACLE = 3
_ROS_OBJECT_TYPES = {
    _OBJECT_TYPE_VEHICLE: 1,
    _OBJECT_TYPE_PEDESTRIAN: 0,
    _OBJECT_TYPE_OBSTACLE: 2,
}
_NOT_DETECTED_IDS = {"", "not detected", "none", "null"}


@dataclass(frozen=True)
class GrpcPlanningSnapshot:
    tick: int
    source_time: float
    simulator_timestamp: int
    vehicle_state: VehicleState
    objects: list
    traffic_light: list


def _default_sdk_root():
    # .../AIM2026/Expert/src/expert/src/network/ros/<this file>
    morai_drive_root = Path(__file__).resolve().parents[7] / "MoraiDrive"
    return (
        morai_drive_root
        / "data_collection"
        / "third_party"
        / "grpc_inha_univ"
        / "src"
    )


def _wsl_default_gateway():
    """Return WSL's Windows-host gateway without invoking a shell command."""
    try:
        with open("/proc/net/route", "r", encoding="ascii") as route_file:
            next(route_file)
            for line in route_file:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == "00000000":
                    return socket.inet_ntoa(
                        struct.pack("<L", int(fields[2], 16))
                    )
    except (OSError, ValueError, StopIteration, struct.error):
        return None
    return None


def _host_candidates(configured_host):
    configured_host = str(configured_host or "auto").strip()
    candidates = []
    if configured_host.lower() != "auto":
        candidates.append(configured_host)
    else:
        gateway = _wsl_default_gateway()
        if gateway:
            candidates.append(gateway)
        candidates.append("127.0.0.1")
    return list(dict.fromkeys(candidates))


def _actor_id(actor_state):
    return str(actor_state.actor_info.id.value)


def _find_ego(states, ego_id):
    wanted = str(ego_id).casefold()
    for state in states:
        if _actor_id(state).casefold() == wanted:
            return state
    return None


def _vehicle_state(actor_state):
    transform = actor_state.transform
    # MORAI gRPC actor velocity uses the same km/h convention as Object_topic.
    return VehicleState(
        transform.location.x,
        transform.location.y,
        float(transform.rotation.z) * 3.141592653589793 / 180.0,
        float(actor_state.velocity.x) / 3.6,
    )


def _planning_object(actor_state):
    grpc_type = int(actor_state.actor_info.object_type)
    if grpc_type not in _ROS_OBJECT_TYPES:
        return None

    transform = actor_state.transform
    extent = actor_state.bounding_box.extent
    # Match morai_msgs/ObjectStatus exactly: position/heading are the actor
    # transform, and size is (width, length, height). gRPC extent is half-size.
    width = 2.0 * abs(float(extent.x))
    length = 2.0 * abs(float(extent.y))
    actor_id = _actor_id(actor_state)
    try:
        unique_id = int(actor_id)
    except ValueError:
        unique_id = actor_id
    return ObjectInfo(
        transform.location.x,
        transform.location.y,
        float(actor_state.velocity.x) / 3.6,
        _ROS_OBJECT_TYPES[grpc_type],
        name=actor_id,
        is_static=grpc_type == _OBJECT_TYPE_OBSTACLE,
        size_x=width,
        size_y=length,
        heading_deg=float(transform.rotation.z),
        unique_id=unique_id,
    )


def planning_snapshot_from_actor_states(actor_states, ego_id, tick_period_sec):
    states = list(actor_states.states)
    ego = _find_ego(states, ego_id)
    if ego is None:
        raise ValueError("gRPC actor snapshot has no Ego id={!r}".format(ego_id))

    # Preserve ObjectStatusList callback ordering: NPC, obstacle, pedestrian.
    object_groups = {1: [], 2: [], 0: []}
    for state in states:
        if state is ego:
            continue
        converted = _planning_object(state)
        if converted is not None:
            object_groups[converted.type].append(converted)
    objects = object_groups[1] + object_groups[2] + object_groups[0]

    tick = int(actor_states.sync_mode_tick_count)
    source_time = tick * float(tick_period_sec)
    traffic_light = []
    traffic_id = str(ego.vehicle_state.tl_id.value).strip()
    traffic_status = int(ego.vehicle_state.tl_color)
    if (
        traffic_id.casefold() not in _NOT_DETECTED_IDS
        and traffic_status >= 0
    ):
        now = time.monotonic()
        # gRPC returns the route-relevant light together with the Ego actor in
        # this ActorStates snapshot, so source and reference times are equal.
        traffic_light = [
            traffic_id,
            traffic_status,
            0,
            now,
            source_time,
            source_time,
        ]

    return GrpcPlanningSnapshot(
        tick=tick,
        source_time=source_time,
        simulator_timestamp=int(actor_states.timestamp),
        vehicle_state=_vehicle_state(ego),
        objects=objects,
        traffic_light=traffic_light,
    )


class MoraiGrpcSyncInputs:
    """Lazy, read-only Actor.GetAllActorsState client.

    This intentionally does not construct SimulationWorld: that SDK class
    mutates the scene during initialization. Only the generated Actor stub is
    used here.
    """

    def __init__(self, config):
        config = dict(config or {})
        self.host = os.environ.get(
            "MORAI_GRPC_HOST", str(config.get("host", "auto"))
        )
        self.port = int(os.environ.get("MORAI_GRPC_PORT", config.get("port", 7789)))
        self.ego_id = str(config.get("ego_id", "Ego"))
        self.rpc_timeout_sec = max(
            0.01, float(config.get("rpc_timeout_sec", 0.25))
        )
        self.retry_interval_sec = max(
            0.1, float(config.get("retry_interval_sec", 1.0))
        )
        self.tick_period_sec = max(
            0.001, float(config.get("tick_period_sec", 0.02))
        )
        self.sdk_root = Path(
            os.environ.get(
                "MORAI_GRPC_SDK_PATH",
                str(config.get("sdk_path") or _default_sdk_root()),
            )
        ).resolve()

        self._grpc = None
        self._actor_grpc = None
        self._filter_type = None
        self._channel = None
        self._stub = None
        self._connected_host = None
        self._next_retry_at = 0.0
        self._last_error = None
        self._load_sdk()

    @property
    def connected_host(self):
        return self._connected_host

    @property
    def last_error(self):
        return self._last_error

    def _load_sdk(self):
        proto_path = self.sdk_root / "proto"
        if not proto_path.is_dir():
            raise RuntimeError("MORAI gRPC proto directory not found: {}".format(proto_path))
        proto_path_text = str(proto_path)
        if proto_path_text not in sys.path:
            sys.path.insert(0, proto_path_text)

        try:
            import grpc
            import morai.actor.actor_pb2_grpc as actor_grpc
            from morai.actor.actor_get_pb2 import GetAllActorsFilter
        except ImportError as error:
            raise RuntimeError("MORAI gRPC Python dependency unavailable: {}".format(error))

        self._grpc = grpc
        self._actor_grpc = actor_grpc
        self._filter_type = GetAllActorsFilter

    def _disconnect(self):
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None
        self._connected_host = None

    def close(self):
        self._disconnect()

    def _request(self):
        return self._filter_type(
            client_key="",
            vehicle=True,
            pedestrian=True,
            obstacle=True,
        )

    def _connect_and_fetch(self):
        errors = []
        for host in _host_candidates(self.host):
            channel = self._grpc.insecure_channel(
                "{}:{}".format(host, self.port),
                options=[("grpc.max_receive_message_length", 33554432)],
            )
            stub = self._actor_grpc.ActorStub(channel)
            try:
                actor_states = stub.GetAllActorsState(
                    self._request(), timeout=self.rpc_timeout_sec
                )
            except self._grpc.RpcError as error:
                errors.append("{}: {}".format(host, error.code().name))
                channel.close()
                continue
            self._channel = channel
            self._stub = stub
            self._connected_host = host
            return actor_states
        raise RuntimeError("; ".join(errors) or "no gRPC host candidates")

    def fetch(self):
        now = time.monotonic()
        if self._stub is None and now < self._next_retry_at:
            return None
        try:
            if self._stub is None:
                actor_states = self._connect_and_fetch()
            else:
                actor_states = self._stub.GetAllActorsState(
                    self._request(), timeout=self.rpc_timeout_sec
                )
            snapshot = planning_snapshot_from_actor_states(
                actor_states,
                ego_id=self.ego_id,
                tick_period_sec=self.tick_period_sec,
            )
            self._last_error = None
            return snapshot
        except (self._grpc.RpcError, RuntimeError, ValueError) as error:
            self._last_error = str(error)
            self._next_retry_at = now + self.retry_interval_sec
            self._disconnect()
            return None
