"""Read-only, deadline-bounded gRPC observations while the ROS tick owner holds.

No gRPC connect handshake, Start, Tick, or SetSynchronousMode is used here.
Raw actor timestamps and tick counters are preserved in their own domain.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def default_grpc_target():
    """Resolve the Windows MORAI gRPC endpoint without pinning one WSL lease.

    An explicit ``MORAI_GRPC_TARGET`` wins.  Otherwise the separate host/port
    variables are honored and ``auto`` resolves the WSL default gateway.  The
    If gateway discovery fails, an explicit host is required; a lease from a
    different workstation is never reused silently.
    """
    target = os.environ.get("MORAI_GRPC_TARGET", "").strip()
    if target:
        if ":" not in target:
            raise ValueError("MORAI_GRPC_TARGET must be HOST:PORT")
        return target
    host = os.environ.get("MORAI_GRPC_HOST", "auto").strip()
    port = os.environ.get("MORAI_GRPC_PORT", "7789").strip()
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("MORAI_GRPC_PORT must be 1..65535")
    if host.casefold() == "auto":
        host = None
        try:
            with open("/proc/net/route", "r", encoding="ascii") as stream:
                next(stream)
                for line in stream:
                    fields = line.split()
                    if len(fields) >= 3 and fields[1] == "00000000":
                        host = socket.inet_ntoa(
                            struct.pack("<L", int(fields[2], 16))
                        )
                        break
        except (OSError, ValueError, StopIteration, struct.error):
            host = None
        host = host or os.environ.get("MORAI_GRPC_FALLBACK_HOST", "").strip()
        if not host:
            raise ValueError("No MORAI gRPC host could be resolved")
    return "%s:%s" % (host, port)


class GrpcGroundTruth:
    def __init__(self, target, timeout=1.):
        project_root = Path(__file__).resolve().parents[3]
        proto = project_root/'data_collection/third_party/grpc_inha_univ/src/proto'
        sys.path.insert(0, str(proto))
        import grpc
        from google.protobuf.json_format import MessageToDict
        from morai.actor.actor_pb2_grpc import ActorStub
        from morai.actor.actor_get_pb2 import GetAllActorsFilter
        from morai.sensor.sensor_pb2_grpc import SensorStub
        from morai.simulation.simulation_pb2_grpc import SimulationStub
        from morai.infrastructure.infrastructure_pb2_grpc import InfrastructureStub
        from morai.infrastructure.traffic_light_pb2 import GetTrafficLightInfoParam
        from morai.common.type_pb2 import Empty
        from morai.common.object_info_pb2 import ObjectInfo
        from morai.common.enum_pb2 import OBJECT_TYPE_VEHICLE
        self.timeout = float(timeout)
        if not 0 < self.timeout <= 10: raise ValueError('RPC deadline must be 0..10s')
        self.channel = grpc.insecure_channel(target)
        self.actor = ActorStub(self.channel)
        self.sensor = SensorStub(self.channel)
        self.simulation = SimulationStub(self.channel)
        self.infrastructure = InfrastructureStub(self.channel)
        self.actor_query = GetAllActorsFilter(vehicle=True, pedestrian=True, obstacle=True)
        self.ego_query = ObjectInfo(object_type=OBJECT_TYPE_VEHICLE)
        self.ego_query.id.value = 'Ego'
        self.Empty, self.LightQuery = Empty, GetTrafficLightInfoParam
        self.ObjectInfo=ObjectInfo
        def to_dict(message):
            options = dict(preserving_proto_field_name=True)
            try:
                # protobuf <= 4.x
                return MessageToDict(
                    message,
                    including_default_value_fields=True,
                    **options,
                )
            except TypeError as error:
                # protobuf >= 5 renamed the option. Only retry this known API
                # mismatch; a real conversion failure must remain visible.
                if "including_default_value_fields" not in str(error):
                    raise
                return MessageToDict(
                    message,
                    always_print_fields_with_no_presence=True,
                    **options,
                )

        self.to_dict = to_dict

    def call(self, rpc, query):
        row = dict(request_monotonic_ns=time.monotonic_ns(), request_wall_ns=time.time_ns())
        try:
            value = rpc(query, timeout=self.timeout)
            row.update(receive_monotonic_ns=time.monotonic_ns(), receive_wall_ns=time.time_ns(),
                       response=self.to_dict(value), available=True)
        except Exception as error:
            row.update(receive_monotonic_ns=time.monotonic_ns(), receive_wall_ns=time.time_ns(),
                       available=False, error=str(error))
        return row

    def actors(self): return self.call(self.actor.GetAllActorsState, self.actor_query)

    def inspect(self):
        return dict(actors=self.actors(),
            mode=self.call(self.simulation.GetSynchronousMode, self.Empty()),
            sensors=self.call(self.sensor.GetSensorInfos, self.ego_query))

    def held_snapshot(self, light_ids, attempts=3):
        if not 1<=attempts<=3:raise ValueError('Bounded read-only snapshot attempts required')
        required=tuple(sorted(set(str(key) for key in light_ids)))
        history=[]
        for _ in range(attempts):
            value=self._held_snapshot_once(required)
            # MORAI can return a transient handler error for a light even
            # while both actor snapshots are stable. Retry the whole bracket
            # without advancing native time; never reuse an earlier color.
            lights=value.get('traffic_lights',{})
            missing=[key for key in required if not lights.get(key,{}).get('available')
                     or lights[key].get('response',{}).get('id',{}).get('value')!=key]
            value['required_light_read_failures']=missing
            if value['stable'] and not missing:
                value['prior_failed_attempts']=history
                return value
            history.append(value)
        value=history.pop()
        value['prior_failed_attempts']=history
        return value

    def _held_snapshot_once(self, light_ids):
        # Both actor inventories bracket all signal queries. A changing scene
        # invalidates association instead of silently using the newest state.
        first = self.actors()
        # Resolve current model identity inside the held-state bracket. Actor
        # numeric IDs can be reused after scenario resets, so do not cache a
        # model forever by ID or guess it from bounding-box dimensions.
        models={}
        owned_identities=getattr(self,'actor_identity_provider',lambda:{})()
        from google.protobuf.json_format import ParseDict
        for actor in first.get('response',{}).get('states',[]):
            info=actor.get('actor_info',{});key=info.get('id',{}).get('value')
            if not key or key=='Ego':continue
            query=ParseDict(info,self.ObjectInfo())
            # Client-scoped GetAll returns our alias as id but leaves the
            # actor_info.client_key empty. Restore only acknowledged ownership
            # for individual lookups; unrelated numeric actors stay unscoped.
            if key in owned_identities:query.client_key=owned_identities[key]
            models[key]=self.call(self.actor.GetOptionName,query)
            models[key]['query']=self.to_dict(query)
        light_ids=set(light_ids)
        # Preserve states referenced by all vehicles, including Ego's RPC link
        # association. Training still requires the independent lane binding.
        for actor in first.get('response',{}).get('states',[]):
            key=actor.get('vehicle_state',{}).get('tl_id',{}).get('value')
            if key and key not in ('Not Detected','Undefined'):light_ids.add(key)
        lights = {str(key): self.call(self.infrastructure.GetTrafficLightInfo,
                  self.LightQuery(type=2, value=str(key))) for key in sorted(set(light_ids))}
        last = self.actors()
        stable = False
        if first['available'] and last['available']:
            def scene(row):
                response = row['response']
                states = sorted(response.get('states', []),
                    key=lambda a: a.get('actor_info', {}).get('id', {}).get('value', ''))
                return canonical(dict(states=states, tick=response.get('sync_mode_tick_count')))
            stable = scene(first) == scene(last)
        spawn_metadata={}
        provider=getattr(self,'actor_metadata_provider',None)
        if provider is not None and first.get('available'):
            spawn_metadata=provider(int(first['response']['sync_mode_tick_count']))
        return dict(schema='morai_held_grpc_snapshot_v1', before=first, traffic_lights=lights,actor_models=models,
                    actor_spawn_metadata=spawn_metadata,
                    after=last, stable=stable,
                    clock_note='RPC receipt times are not native or camera exposure times')

    def close(self): self.channel.close()
