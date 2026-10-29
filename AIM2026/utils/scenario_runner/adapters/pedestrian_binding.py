"""Bind a creating alias once to a verified native MORAI pedestrian address."""
import copy
import math
import time


class PedestrianSpawnNotObserved(RuntimeError):
    """Creation was rejected and repeated inventories contain no new pedestrian."""


def address(state):
    info = state.actor_info
    return str(info.id.value).strip(), int(info.object_type), str(info.client_key)


class BoundPedestrian:
    _native_binding = True

    def __init__(self, client, request_id, state, model, speed_mps):
        self.client = client
        self.request_id = request_id
        self.info = copy.deepcopy(state.actor_info)
        self.model = model
        self.speed_mps = speed_mps
        self.last_position = state.transform.location
        self.last_read = time.monotonic()

    def get_id(self):
        return str(self.info.id.value)

    def get_object_info(self):
        return copy.deepcopy(self.info)

    def validate_state(self, state):
        expected = (self.get_id(), int(self.info.object_type), str(self.info.client_key))
        if address(state) != expected:
            raise RuntimeError('Pinned pedestrian native address changed')
        name = self.client.world._sim_adapter._actor_stub.GetOptionName(self.info, timeout=1.0)
        if getattr(name, 'value', None) != self.model:
            raise RuntimeError('Pinned pedestrian model changed')
        position = state.transform.location
        elapsed = max(0.0, time.monotonic() - self.last_read)
        distance = math.hypot(position.x - self.last_position.x, position.y - self.last_position.y)
        if distance > 2.0 + self.speed_mps * elapsed * 1.5:
            raise RuntimeError('Pinned pedestrian position discontinuity; refusing alias rebinding')
        return state

    def get_actor_state(self):
        matches = [s for s in self.client.get_global_pedestrian_states()
                   if str(s.actor_info.id.value).strip() == self.get_id()]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one native pedestrian {self.get_id()}, found {len(matches)}')
        state = self.validate_state(matches[0])
        self.last_position = copy.deepcopy(state.transform.location)
        self.last_read = time.monotonic()
        return state

    def control(self, direction, speed):
        self.get_actor_state()
        from proto.morai.actor.actor_control_pb2 import PedestrianCtrlCmd
        command = PedestrianCtrlCmd()
        command.actor_info.CopyFrom(self.info)
        command.direction.CopyFrom(direction)
        command.speed = float(speed)
        result = self.client.world._sim_adapter.control_pedestrian(command)
        return result is not None and result.status == 1

    def destroy(self):
        # Use the native address, including its original client_key. Never
        # put a raw numeric ID into the creating-alias namespace.
        self.get_actor_state()
        result = self.client.world._sim_adapter._actor_stub.DestroyActor(self.info, timeout=1.0)
        return result is not None and result.status == 1


def bind_spawn(client, before, request_id, model, transform, speed_mps, started,
               *, creation_rejected=False):
    baseline = {address(s)[0] for s in before}
    saw_new_pedestrian = False
    for attempt in range(10):
        if attempt:
            time.sleep(0.1)
        fresh = client.get_global_pedestrian_states()
        candidates = []
        for state in fresh:
            raw_id, kind, owner = address(state)
            if raw_id not in baseline and kind == 2:
                saw_new_pedestrian = True
            if raw_id in baseline or kind != 2 or (owner and owner != client.client_key):
                continue
            point = state.transform.location
            radius = 1.5 + speed_mps * max(0.0, time.monotonic() - started)
            if math.hypot(point.x - transform.location.x, point.y - transform.location.y) > radius:
                continue
            name = client.world._sim_adapter._actor_stub.GetOptionName(state.actor_info, timeout=1.0)
            if getattr(name, 'value', None) == model:
                candidates.append(state)
        if len(candidates) > 1:
            raise RuntimeError('Ambiguous new native pedestrians; refusing nearest-actor binding')
        if len(candidates) == 1:
            return BoundPedestrian(client, request_id, candidates[0], model, speed_mps)
    message = f'Could not bind spawned pedestrian {request_id} to a new native ID'
    if creation_rejected and not saw_new_pedestrian:
        raise PedestrianSpawnNotObserved(message)
    raise RuntimeError(message)
