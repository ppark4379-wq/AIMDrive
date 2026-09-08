"""Read frame-coherent object and traffic-light labels from MORAI gRPC.

This module is deliberately read-only.  It uses only Actor.GetAllActorsState;
it never creates SimulationWorld, changes SyncMode, or advances a tick.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time
import zlib

from ros.sensor_types import GTObject, TrafficLightState


_GRPC_PROTO = (
    Path(__file__).resolve().parent.parent
    / "third_party"
    / "grpc_inha_univ"
    / "src"
    / "proto"
)
_OBJECT_TYPES = {1: "vehicle", 2: "pedestrian", 3: "static"}
_NO_TRAFFIC_LIGHT = {"", "not detected", "none", "null"}


@dataclass(frozen=True)
class GrpcLabelSnapshot:
    tick: int
    simulator_timestamp_raw: int
    ego_x: float
    ego_y: float
    ego_heading_deg: float
    objects: list
    traffic_lights: list


def _wsl_default_gateway():
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


def host_candidates(configured_host):
    host = os.environ.get(
        "MORAI_GRPC_HOST", str(configured_host or "auto")
    ).strip()
    candidates = []
    if host.casefold() != "auto":
        candidates.append(host)
    else:
        gateway = _wsl_default_gateway()
        if gateway:
            candidates.append(gateway)
        candidates.append("127.0.0.1")
    return list(dict.fromkeys(candidates))


def _actor_id(state):
    return str(state.actor_info.id.value)


def _dataset_actor_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        # DatasetWriter requires integer IDs. Keep non-numeric gRPC actor IDs
        # stable across processes instead of using Python's randomized hash().
        return zlib.crc32(str(value).encode("utf-8")) & 0x7FFFFFFF


def _find_ego(states, ego_id):
    wanted = str(ego_id).casefold()
    return next(
        (state for state in states if _actor_id(state).casefold() == wanted),
        None,
    )


def _world_velocity_mps(state):
    """Return the actor's world ENU XY velocity in m/s.

    Current MORAI ActorState exposes ``global_velocity`` separately from the
    local ``velocity`` field.  The former already uses simulator/world axes and
    is consumed without an extra heading rotation.  The fallback keeps older
    proto packages usable by projecting their local longitudinal km/h value.
    """
    global_velocity = getattr(state, "global_velocity", None)
    if global_velocity is not None:
        return (
            float(global_velocity.x),
            float(global_velocity.y),
        )
    yaw = math.radians(float(state.transform.rotation.z))
    longitudinal = float(state.velocity.x) / 3.6
    return (
        math.cos(yaw) * longitudinal,
        math.sin(yaw) * longitudinal,
    )


def label_snapshot_from_actor_states(actor_states, ego_id="Ego"):
    states = list(actor_states.states)
    ego = _find_ego(states, ego_id)
    if ego is None:
        raise ValueError(
            "gRPC ActorStates has no Ego actor id={!r}".format(ego_id)
        )

    # Match ObjectStatusList ordering and field conventions: NPC, pedestrian,
    # obstacle; actor transform position/heading; size=(width,length,height).
    groups = {"vehicle": [], "pedestrian": [], "static": []}
    for state in states:
        if state is ego:
            continue
        object_type = _OBJECT_TYPES.get(int(state.actor_info.object_type))
        if object_type is None:
            continue
        transform = state.transform
        extent = state.bounding_box.extent
        vel_x, vel_y = _world_velocity_mps(state)
        if object_type == "static":
            vel_x = vel_y = 0.0
        groups[object_type].append(GTObject(
            obj_id=_dataset_actor_id(_actor_id(state)),
            obj_type=object_type,
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
            vel_x=vel_x,
            vel_y=vel_y,
            heading_deg=float(transform.rotation.z),
            width=2.0 * abs(float(extent.x)),
            length=2.0 * abs(float(extent.y)),
            height=2.0 * abs(float(extent.z)),
        ))
    objects = groups["vehicle"] + groups["pedestrian"] + groups["static"]

    traffic_lights = []
    traffic_id = str(ego.vehicle_state.tl_id.value).strip()
    traffic_state = int(ego.vehicle_state.tl_color)
    if (
        traffic_id.casefold() not in _NO_TRAFFIC_LIGHT
        and traffic_state >= 0
    ):
        traffic_lights.append(TrafficLightState(
            tl_id=traffic_id,
            state=traffic_state,
        ))

    return GrpcLabelSnapshot(
        tick=int(actor_states.sync_mode_tick_count),
        simulator_timestamp_raw=int(actor_states.timestamp),
        ego_x=float(ego.transform.location.x),
        ego_y=float(ego.transform.location.y),
        ego_heading_deg=float(ego.transform.rotation.z),
        objects=objects,
        traffic_lights=traffic_lights,
    )


def snapshot_matches_ego(snapshot, x, y, heading_deg,
                         max_distance_m=0.5, max_heading_error_deg=3.0):
    distance = math.hypot(
        float(snapshot.ego_x) - float(x),
        float(snapshot.ego_y) - float(y),
    )
    heading_error = abs(
        (float(snapshot.ego_heading_deg) - float(heading_deg) + 180.0)
        % 360.0 - 180.0
    )
    return (
        distance <= float(max_distance_m)
        and heading_error <= float(max_heading_error_deg)
    )


class MoraiGrpcStateReader:
    def __init__(self, config):
        config = dict(config or {})
        self.host = config.get("state_host", config.get("host", "auto"))
        self.port = int(os.environ.get(
            "MORAI_GRPC_PORT",
            config.get("state_port", config.get("port", 7789)),
        ))
        self.ego_id = str(config.get("ego_id", "Ego"))
        self.rpc_timeout_sec = max(
            0.01, float(config.get("state_rpc_timeout_sec", 0.25))
        )
        self.retry_interval_sec = max(
            0.01, float(config.get("state_retry_interval_sec", 0.05))
        )
        proto_path = str(_GRPC_PROTO)
        if proto_path not in sys.path:
            sys.path.insert(0, proto_path)
        import grpc
        import morai.actor.actor_pb2_grpc as actor_grpc
        from morai.actor.actor_get_pb2 import GetAllActorsFilter

        self._grpc = grpc
        self._actor_grpc = actor_grpc
        self._filter_type = GetAllActorsFilter
        self._channel = None
        self._stub = None
        self._prefetched = None
        self.connected_host = None
        self.last_error = None

    def _request(self):
        return self._filter_type(
            client_key="", vehicle=True, pedestrian=True, obstacle=True
        )

    def connect(self):
        errors = []
        for host in host_candidates(self.host):
            channel = self._grpc.insecure_channel(
                "{}:{}".format(host, self.port),
                options=[("grpc.max_receive_message_length", 33554432)],
            )
            stub = self._actor_grpc.ActorStub(channel)
            try:
                response = stub.GetAllActorsState(
                    self._request(), timeout=self.rpc_timeout_sec
                )
                snapshot = label_snapshot_from_actor_states(
                    response, self.ego_id
                )
            except (self._grpc.RpcError, ValueError) as error:
                errors.append("{}: {}".format(host, error))
                channel.close()
                continue
            self._channel = channel
            self._stub = stub
            self._prefetched = snapshot
            self.connected_host = host
            self.last_error = None
            return
        raise RuntimeError(
            "MORAI read-only gRPC state connection failed: {}".format(
                "; ".join(errors) or "no host candidates"
            )
        )

    def fetch(self):
        try:
            if self._stub is None:
                self.connect()
            if self._prefetched is not None:
                snapshot = self._prefetched
                self._prefetched = None
                return snapshot
            response = self._stub.GetAllActorsState(
                self._request(), timeout=self.rpc_timeout_sec
            )
            snapshot = label_snapshot_from_actor_states(response, self.ego_id)
            self.last_error = None
            return snapshot
        except (self._grpc.RpcError, RuntimeError, ValueError) as error:
            self.last_error = str(error)
            self.close()
            return None

    def fetch_required(self, timeout_sec, expected_tick=None):
        deadline = time.monotonic() + max(0.01, float(timeout_sec))
        last_tick = None
        while time.monotonic() < deadline:
            snapshot = self.fetch()
            if snapshot is not None:
                last_tick = int(snapshot.tick)
                if expected_tick is None or last_tick == int(expected_tick):
                    return snapshot
                if last_tick > int(expected_tick):
                    raise RuntimeError(
                        "MORAI gRPC ActorStates passed required tick: "
                        "expected={}, received={}".format(expected_tick, last_tick)
                    )
            time.sleep(self.retry_interval_sec)
        raise RuntimeError(
            "MORAI gRPC object/traffic snapshot unavailable: expected_tick={}, "
            "last_tick={}, error={}".format(
                expected_tick,
                last_tick,
                self.last_error or "timeout",
            )
        )

    def close(self):
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None
        self._prefetched = None
        self.connected_host = None


class GrpcStateHistory:
    """Continuously retain per-tick states for passive native-file capture."""

    def __init__(self, reader, poll_interval_sec=0.005, max_frames=2000,
                 ego_match_distance_m=0.5):
        self.reader = reader
        self.poll_interval_sec = max(0.001, float(poll_interval_sec))
        self.max_frames = max(10, int(max_frames))
        self.ego_match_distance_m = max(0.01, float(ego_match_distance_m))
        self._snapshots = OrderedDict()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._aligned = False

    @property
    def is_aligned(self):
        with self._condition:
            return self._aligned

    def start(self):
        # Fail before dataset writing when the required state API is unavailable.
        self.reader.connect()
        first = self.reader.fetch()
        if first is not None:
            self._store(first)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _store(self, snapshot):
        with self._condition:
            self._snapshots[snapshot.tick] = snapshot
            self._snapshots.move_to_end(snapshot.tick)
            while len(self._snapshots) > self.max_frames:
                self._snapshots.popitem(last=False)
            self._condition.notify_all()

    def _run(self):
        while not self._stop.is_set():
            snapshot = self.reader.fetch()
            if snapshot is not None:
                self._store(snapshot)
            self._stop.wait(self.poll_interval_sec)

    def _matches_ego(self, snapshot, ego):
        position = ego["position"]
        return snapshot_matches_ego(
            snapshot,
            position["x"],
            position["y"],
            ego["heading"],
            max_distance_m=self.ego_match_distance_m,
        )

    def wait_for_record(self, record, timeout_sec):
        native_frame = int(record["native_sync_frame"])
        ego = record["ego_state_for_frame"]
        deadline = time.monotonic() + max(0.01, float(timeout_sec))
        with self._condition:
            while time.monotonic() < deadline:
                # native_sync_frame and ActorStates.tick are the same simulator
                # counter.  Never infer an offset from pose alone: a stationary
                # Ego has the same pose across many ticks and could silently bind
                # objects/signals from an older or newer frame.
                snapshot = self._snapshots.get(native_frame)
                if snapshot is not None:
                    if not self._matches_ego(snapshot, ego):
                        return None
                    self._aligned = True
                    return snapshot

                if self._snapshots:
                    newest = next(reversed(self._snapshots))
                    if native_frame < newest:
                        # The polling history has already passed this tick without
                        # observing it.  Saving an approximate label is forbidden.
                        return None
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    self._condition.wait(timeout=min(0.05, remaining))
        return None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.reader.close()
