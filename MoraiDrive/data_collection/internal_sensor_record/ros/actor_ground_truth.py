"""Passive gRPC actor inventory and timestamp-bracketed traffic queries.

RPC request/response receipt bounds are recorded, never equated to ROS clocks.
Normalized boxes are candidates until the capture contract verifies the API axes.
Traffic responses have no timestamp, so the surrounding actor timestamps remain
interval evidence and are never rewritten as exact light observation times.
"""
import math
import json
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from queue import Empty, Full, Queue

try:
    from ros.sensor_types import GTObject
    from ros.monotonic_wait import wait_until
except ModuleNotFoundError:
    from .sensor_types import GTObject
    from .monotonic_wait import wait_until


def traffic_query_due(previous_source, current_source, units_per_second, period_sec):
    """Pace untimestamped light queries in simulator source time."""
    if previous_source is None or current_source < previous_source:
        return True
    return current_source-previous_source >= int(float(units_per_second)*float(period_sec))


def query_traffic_light(call, request, light_id, encode, timeout, max_attempts=2,
                        monotonic_ns=time.monotonic_ns, wall_ns=time.time_ns):
    """Run a bounded query while preserving every failed/invalid attempt.

    GetTrafficLightInfo has no source timestamp. Retrying can recover a
    transient transport failure, but it cannot make the result instantaneous;
    the caller must still enclose the complete attempt sequence in actor-source
    observations.
    """
    if not 1 <= int(max_attempts) <= 3:
        raise ValueError('Traffic query attempts must be bounded to 1..3')
    attempts = []
    selected = None
    for attempt_index in range(int(max_attempts)):
        observation = dict(attempt_index=attempt_index,
                           request_monotonic_ns=monotonic_ns(),
                           request_wall_ns=wall_ns())
        try:
            response = call(request, timeout=timeout)
            observation['receive_monotonic_ns'] = monotonic_ns()
            observation['receive_wall_ns'] = wall_ns()
            observation['response'] = encode(response)
            observation['valid_response_identity'] = (
                observation['response'].get('id', {}).get('value') == light_id)
        except Exception as error:
            observation.update(receive_monotonic_ns=monotonic_ns(),
                               receive_wall_ns=wall_ns(), error=str(error),
                               valid_response_identity=False)
        observation['rpc_duration_ns'] = (
            observation['receive_monotonic_ns']-observation['request_monotonic_ns'])
        attempts.append(observation)
        if observation['valid_response_identity']:
            selected = observation
            break
    result = dict(selected or attempts[-1])
    result['attempts'] = attempts
    result['attempt_count'] = len(attempts)
    result['recovered_after_retry'] = bool(selected is not None and len(attempts) > 1)
    result['final_valid_response_identity'] = selected is not None
    # Keep the legacy field as the selected/final outcome for old readers.
    result['valid_response_identity'] = selected is not None
    return result


class ActorIdentityTracker:
    """Assign a local incarnation to each continuously present simulator actor.

    Actor numeric IDs can be reused. A model-name response is therefore cached
    only for one source-clock generation and one uninterrupted presence. This
    is identity provenance, not sensor-frame atomicity or geometry approval.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.capture_generation = None
        self.active = {}
        self.incarnation_counts = Counter()
        self.verified = {}
        self.pending = deque()
        self.pending_set = set()

    @staticmethod
    def _actor_key(actor):
        info = actor.get('actor_info', {})
        return (str(info.get('id', {}).get('value', '')),
                str(info.get('object_type', '')))

    @staticmethod
    def encode_instance_key(generation, actor_id, object_type, incarnation):
        return '%d|%s|%s|%d' % (int(generation), object_type, actor_id, int(incarnation))

    def observe(self, response, generation):
        with self.lock:
            if self.capture_generation != generation:
                self.capture_generation = generation
                self.active.clear()
                self.pending.clear()
                self.pending_set.clear()
            current = {}
            for actor in response.get('states', []):
                actor_id, object_type = self._actor_key(actor)
                if not actor_id or actor_id == 'Ego' or not object_type:
                    continue
                prior = self.active.get(actor_id)
                if prior is None or prior[0] != object_type:
                    base = (generation, actor_id, object_type)
                    incarnation = self.incarnation_counts[base]
                    self.incarnation_counts[base] += 1
                    instance_key = self.encode_instance_key(
                        generation, actor_id, object_type, incarnation)
                    if instance_key not in self.pending_set and instance_key not in self.verified:
                        self.pending.append(instance_key)
                        self.pending_set.add(instance_key)
                else:
                    incarnation, instance_key = prior[1], prior[2]
                current[actor_id] = (object_type, incarnation, instance_key)
            self.active = current
            return {actor_id: dict(capture_generation=int(generation),
                                   incarnation=int(value[1]),
                                   instance_key=value[2],
                                   object_type=value[0],
                                   model_name=self.verified.get(value[2]),
                                   model_identity_valid=value[2] in self.verified)
                    for actor_id, value in sorted(current.items())}

    def pending_actors(self, protobuf_states, limit):
        by_id = {str(actor.actor_info.id.value): actor for actor in protobuf_states
                 if str(actor.actor_info.id.value) != 'Ego'}
        with self.lock:
            selected = []
            scans = len(self.pending)
            for _ in range(scans):
                key = self.pending.popleft()
                self.pending_set.discard(key)
                active = next(((actor_id, value) for actor_id, value in self.active.items()
                               if value[2] == key), None)
                if active is None or key in self.verified:
                    continue
                actor = by_id.get(active[0])
                if actor is None:
                    continue
                if len(selected) < int(limit):
                    selected.append((key, actor, active[1][0], active[1][1]))
                else:
                    self.pending.append(key)
                    self.pending_set.add(key)
            return selected

    def complete(self, instance_key, model_name):
        model_name = str(model_name).strip()
        if not model_name:
            return False
        with self.lock:
            if not any(value[2] == instance_key for value in self.active.values()):
                return False
            previous = self.verified.get(instance_key)
            if previous is not None and previous != model_name:
                raise ValueError('Actor model identity changed within one incarnation')
            self.verified[instance_key] = model_name
            return True

    def retry(self, instance_key):
        with self.lock:
            if (instance_key not in self.verified and instance_key not in self.pending_set
                    and any(value[2] == instance_key for value in self.active.values())):
                self.pending.append(instance_key)
                self.pending_set.add(instance_key)


class ActorMotionHistory:
    """Causal source-clock differences; positions also handle paused pedestrians.

    The scale is explicit and must be checked against Ego SI motion. It does not
    map this source clock to any other sensor or to receive time.
    """
    def __init__(self, units_per_second=1e6):
        self.units = float(units_per_second)
        if not math.isfinite(self.units) or self.units <= 0:
            raise ValueError('Actor source clock units must be finite and positive')
        self.history = {}
        self.previous_source = None

    def update(self, response):
        source = int(response['timestamp'])
        if self.previous_source is not None and source < self.previous_source:
            self.history.clear()
        self.previous_source = source
        result, present = {}, set()
        for actor in response.get('states', []):
            ident = str(actor['actor_info']['id']['value'])
            present.add(ident)
            position = _xyz(actor['transform']['location'])
            kind = actor['actor_info']['object_type']
            history = self.history.setdefault(ident, deque(maxlen=20))
            if history and (history[-1][2] != kind or source < history[-1][0]):
                history.clear()
            if not history or source > history[-1][0]:
                history.append((source, position, kind))
            candidates = [h for h in history if .06 <= (source-h[0])/self.units <= .3]
            if candidates:
                old = min(candidates, key=lambda h: abs((source-h[0])/self.units - .12))
                dt = (source-old[0])/self.units
                velocity = [(b-a)/dt for a,b in zip(old[1], position)]
                if all(math.isfinite(v) for v in velocity) and math.sqrt(sum(v*v for v in velocity)) < 80:
                    result[ident] = {'velocity_xyz_mps': velocity, 'source_delta_sec': dt,
                                     'source_begin_raw': old[0], 'source_end_raw': source,
                                     'method': 'causal_actor_pivot_difference'}
        self.history = {k: v for k,v in self.history.items() if k in present}
        return result


class ActorStatePoller:
    def __init__(self, config):
        self.config = dict(config)
        self.period = float(config.get('period_sec', .05))
        self.timeout = float(config.get('rpc_timeout_sec', .15))
        self.traffic_source_period = float(config.get('traffic_source_period_sec', .05))
        self.traffic_query_attempts = int(config.get('traffic_query_attempts', 2))
        self.identity_queries_per_cycle = int(config.get('identity_queries_per_cycle', 8))
        if (not (.02 <= self.period <= 1 and .01 <= self.timeout <= 1)
                or not .02 <= self.traffic_source_period <= .5
                or not 1 <= self.traffic_query_attempts <= 3
                or not 0 <= self.identity_queries_per_cycle <= 20):
            raise ValueError('Actor polling interval/deadline outside bounded range')
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._history = deque(maxlen=100)
        self._stats = Counter()
        self._thread = None
        self.trace_path = None
        self.on_record = None  # Raw capture archives EVERY immutable RPC result.
        self.on_actor_identity_record = None
        self.on_traffic_record = None

    def start(self):
        proto = Path(__file__).resolve().parents[1] / 'third_party/grpc_inha_univ/src/proto'
        sys.path.insert(0, str(proto))
        import grpc
        from google.protobuf.json_format import MessageToDict
        from morai.actor.actor_pb2_grpc import ActorStub
        from morai.actor.actor_get_pb2 import GetAllActorsFilter
        from morai.infrastructure.infrastructure_pb2_grpc import InfrastructureStub
        from morai.infrastructure.traffic_light_pb2 import GetTrafficLightInfoParam
        self._channel = grpc.insecure_channel(self.config['target'])
        stub = ActorStub(self._channel)
        infrastructure = InfrastructureStub(self._channel)
        query = GetAllActorsFilter(vehicle=True, pedestrian=True, obstacle=True)
        # Log EVERY RPC timing, including slow/failed calls that no LiDAR anchor
        # happens to select. Sensor/GT payloads stay in the existing raw frames.
        trace = Path(self.trace_path).open('x') if self.trace_path else None
        identities = ActorIdentityTracker()
        identity_jobs = Queue(maxsize=256)
        self._identity_jobs = identity_jobs
        self._identity_producer_done = threading.Event()

        def identity_worker():
            # The state producer stops first. Drain identities it already
            # discovered before raw capture is allowed to seal.
            while not self._identity_producer_done.is_set() or not identity_jobs.empty():
                try:
                    job = identity_jobs.get(timeout=.05)
                except Empty:
                    continue
                instance_key, actor, object_type, incarnation, state_seq, source, tick = job
                identity = dict(schema='morai_grpc_actor_identity_v1', seq=state_seq,
                    target=self.config['target'], instance_key=instance_key,
                    capture_generation=int(instance_key.split('|', 1)[0]),
                    incarnation=int(incarnation), actor_id=str(actor.actor_info.id.value),
                    object_type=object_type, client_key=str(actor.actor_info.client_key),
                    before_actor_source_raw=source, before_actor_tick_count=tick,
                    request_monotonic_ns=time.monotonic_ns(), request_wall_ns=time.time_ns())
                try:
                    option = stub.GetOptionName(actor.actor_info, timeout=self.timeout)
                    identity['receive_monotonic_ns'] = time.monotonic_ns()
                    identity['receive_wall_ns'] = time.time_ns()
                    identity['response'] = MessageToDict(option,
                        preserving_proto_field_name=True,
                        including_default_value_fields=True)
                    model_name = identity['response'].get('value', '')
                    identity['valid_nonempty_model_name'] = identities.complete(
                        instance_key, model_name)
                    identity['model_name'] = str(model_name)
                except Exception as error:
                    identity.update(receive_monotonic_ns=time.monotonic_ns(),
                        receive_wall_ns=time.time_ns(), error=str(error),
                        valid_nonempty_model_name=False, model_name='')
                    identities.retry(instance_key)
                identity['available_monotonic_ns'] = time.monotonic_ns()
                identity['rpc_duration_ns'] = (identity['receive_monotonic_ns']-
                                               identity['request_monotonic_ns'])
                with self._lock:
                    ok = identity['valid_nonempty_model_name']
                    self._stats['actor_identity_success' if ok else
                                'actor_identity_failed'] += 1
                try:
                    self.on_actor_identity_record(identity)
                except Exception:
                    with self._lock:self._stats['identity_raw_record_errors'] += 1
                    self._stop.set()
                finally:
                    identity_jobs.task_done()

        def run():
            seq = 0
            previous_source = None
            previous_traffic_source = None
            generation = 0
            source_units=self.config.get('source_timestamp_units_per_second', 1e6)
            motion = ActorMotionHistory(source_units)
            previous_begin = None
            previous_wall = None
            while not self._stop.is_set():
                begin = time.monotonic_ns()
                row = {'seq': seq, 'request_monotonic_ns': begin,
                       'request_wall_ns': time.time_ns(), 'target': self.config['target']}
                row['request_interval_ns'] = begin - previous_begin if previous_begin is not None else None
                row['wall_clock_step_ns'] = (row['request_wall_ns'] - previous_wall - row['request_interval_ns']
                                             if previous_wall is not None else None)
                previous_begin = begin
                previous_wall = row['request_wall_ns']
                try:
                    response = stub.GetAllActorsState(query, timeout=self.timeout)
                    row['receive_monotonic_ns'] = time.monotonic_ns()
                    row['receive_wall_ns'] = time.time_ns()
                    row['response'] = MessageToDict(response, preserving_proto_field_name=True,
                                                   including_default_value_fields=True)
                    row['motion_estimates'] = motion.update(row['response'])
                    row['available_monotonic_ns'] = time.monotonic_ns()
                    source = int(response.timestamp)
                    if previous_source is not None and source < previous_source:
                        generation += 1
                        previous_traffic_source = None
                    previous_source = source
                    row['generation'] = generation
                    row['actor_instances'] = identities.observe(row['response'], generation)
                    row['rpc_duration_ns'] = row['receive_monotonic_ns'] - begin
                    row['filter'] = {'vehicle': True, 'pedestrian': True, 'obstacle': True,
                                     'client_key': ''}
                    key = 'success'
                except Exception as error:
                    row.update(receive_monotonic_ns=time.monotonic_ns(), error=str(error),
                               receive_wall_ns=time.time_ns())
                    row['rpc_duration_ns'] = row['receive_monotonic_ns'] - begin
                    row['available_monotonic_ns'] = time.monotonic_ns()
                    key = 'rpc_error'
                with self._lock:
                    self._history.append(row)
                    self._stats[key] += 1
                    if key == 'success':
                        self._stats['last_actor_success_monotonic_ns'] = row['available_monotonic_ns']
                        self._stats['last_actor_source_raw'] = source
                        self._stats['last_actor_generation'] = generation
                    self._stats['max_rpc_duration_ns'] = max(self._stats['max_rpc_duration_ns'], row['rpc_duration_ns'])
                    self._stats['max_request_interval_ns'] = max(self._stats['max_request_interval_ns'], row['request_interval_ns'] or 0)
                if self.on_record is not None:
                    try:
                        self.on_record(row)
                    except Exception:
                        with self._lock:
                            self._stats['raw_record_errors'] += 1
                        break
                # GetAllActorsState omits model names. Queue each new actor
                # incarnation for the independent identity RPC worker. The
                # poller's state/traffic cadence never waits for those calls.
                if key == 'success' and self.on_actor_identity_record is not None:
                    pending = identities.pending_actors(
                        response.states, self.identity_queries_per_cycle)
                    for instance_key, actor, object_type, incarnation in pending:
                        try:
                            identity_jobs.put_nowait((instance_key, actor, object_type,
                                incarnation, seq, source, int(response.sync_mode_tick_count)))
                        except Full:
                            identities.retry(instance_key)
                            with self._lock:self._stats['actor_identity_queue_full'] += 1
                # /GetTrafficLightStatus is a query-response stream and was
                # observed to have multi-second gaps. Query all route signals
                # plus the signal ID reported by Ego, and bracket them with two
                # timestamped GetAllActorsState responses.  The light RPC has
                # no timestamp of its own, so neither endpoint is rewritten as
                # an exact observation time.
                traffic = None
                query_traffic=(key=='success' and self.on_traffic_record is not None and
                    traffic_query_due(previous_traffic_source,source,source_units,
                                      self.traffic_source_period))
                if query_traffic:
                    previous_traffic_source=source
                    ego = [state for state in row['response'].get('states', [])
                           if state.get('actor_info', {}).get('id', {}).get('value') == 'Ego']
                    ego_light_id = (ego[0].get('vehicle_state', {}).get('tl_id', {}).get('value')
                                    if len(ego) == 1 else None)
                    configured_ids=[str(value) for value in self.config.get('traffic_light_ids',())]
                    if len(configured_ids)!=len(set(configured_ids)):
                        raise ValueError('Configured traffic light IDs must be unique')
                    light_ids=set(configured_ids)
                    if ego_light_id and ego_light_id not in ('Not Detected', 'Undefined'):
                        light_ids.add(str(ego_light_id))
                    if light_ids:
                        traffic = dict(schema='morai_grpc_traffic_light_bracket_v2',
                            seq=seq, target=self.config['target'],
                            source_query_period_sec=self.traffic_source_period,
                            configured_light_ids=configured_ids,
                            ego_reported_light_id=ego_light_id,
                            before_actor_source_raw=int(row['response']['timestamp']),
                            before_actor_tick_count=int(row['response']['sync_mode_tick_count']),
                            request_monotonic_ns=time.monotonic_ns(), request_wall_ns=time.time_ns(),
                            queries={})
                        try:
                            traffic['schema'] = 'morai_grpc_traffic_light_bracket_v3'
                            traffic['max_attempts_per_light'] = self.traffic_query_attempts
                            encode_light = lambda value: MessageToDict(value,
                                preserving_proto_field_name=True,
                                including_default_value_fields=True,
                                use_integers_for_enums=True)
                            for light_id in sorted(light_ids):
                                traffic['queries'][light_id] = query_traffic_light(
                                    infrastructure.GetTrafficLightInfo,
                                    GetTrafficLightInfoParam(type=2, value=light_id),
                                    light_id, encode_light, self.timeout,
                                    self.traffic_query_attempts)
                            after = stub.GetAllActorsState(query, timeout=self.timeout)
                            traffic['after_receive_monotonic_ns'] = time.monotonic_ns()
                            traffic['after_receive_wall_ns'] = time.time_ns()
                            traffic['after_actor_source_raw'] = int(after.timestamp)
                            traffic['after_actor_tick_count'] = int(after.sync_mode_tick_count)
                            traffic['available_monotonic_ns'] = time.monotonic_ns()
                            traffic['valid_response_identities'] = all(
                                value['valid_response_identity'] for value in traffic['queries'].values())
                            traffic_key = ('traffic_batch_success' if traffic['valid_response_identities']
                                           else 'traffic_batch_partial')
                        except Exception as error:
                            traffic.update(after_receive_monotonic_ns=time.monotonic_ns(),
                                after_receive_wall_ns=time.time_ns(),
                                available_monotonic_ns=time.monotonic_ns(), error=str(error),
                                valid_response_identities=False)
                            traffic_key = 'traffic_batch_error'
                        traffic['rpc_duration_ns'] = (
                            traffic['after_receive_monotonic_ns']-traffic['request_monotonic_ns'])
                        with self._lock:
                            self._stats[traffic_key] += 1
                            self._stats['last_traffic_available_monotonic_ns'] = traffic['available_monotonic_ns']
                            self._stats['last_traffic_source_raw'] = traffic['before_actor_source_raw']
                            self._stats['last_traffic_generation'] = generation
                            for light_id, value in traffic['queries'].items():
                                ok = value.get('valid_response_identity') is True
                                self._stats['traffic_query_success' if ok else 'traffic_query_failed'] += 1
                                key_name = ('traffic_query_success:' if ok else 'traffic_query_failed:') + light_id
                                self._stats[key_name] += 1
                                streak = 'traffic_query_failure_streak:' + light_id
                                self._stats[streak] = 0 if ok else self._stats[streak] + 1
                                attempts = value.get('attempts', [value])
                                self._stats['traffic_query_attempts'] += len(attempts)
                                self._stats['traffic_query_attempt_failures'] += sum(
                                    attempt.get('valid_response_identity') is not True
                                    for attempt in attempts)
                                if value.get('recovered_after_retry'):
                                    self._stats['traffic_query_recovered_after_retry'] += 1
                            self._stats['max_traffic_rpc_duration_ns'] = max(
                                self._stats['max_traffic_rpc_duration_ns'],traffic['rpc_duration_ns'])
                        try:
                            self.on_traffic_record(traffic)
                        except Exception:
                            with self._lock:self._stats['traffic_raw_record_errors'] += 1
                            break
                if trace:
                    timing = {k: v for k,v in row.items() if k not in ('response', 'motion_estimates')}
                    timing['source_timestamp_raw'] = row.get('response', {}).get('timestamp')
                    timing['actor_count'] = len(row.get('response', {}).get('states', []))
                    try:
                        trace.write(json.dumps(timing) + '\n')
                        trace.flush()
                    except OSError as error:
                        with self._lock:
                            self._stats['trace_errors'] += 1
                        # A full disk must not silently leave stale successful
                        # inventory as apparent current data.
                        with self._lock:
                            self._history.append(dict(seq=seq, receive_monotonic_ns=time.monotonic_ns(),
                                                      error='actor_trace_write: ' + str(error)))
                        break
                seq += 1
                wait_until(self._stop, begin + int(self.period * 1e9))
            if trace:
                trace.close()

        self._identity_thread = threading.Thread(
            target=identity_worker, name='passive-actor-identity', daemon=True)
        self._identity_thread.start()
        self._thread = threading.Thread(target=run, name='passive-actor-gt', daemon=True)
        self._thread.start()

    def at(self, anchor_ns):
        with self._lock:
            # Select the latest response, including errors. A newer error must
            # not be concealed by falling back to an older successful inventory.
            return max((r for r in self._history if r['receive_monotonic_ns'] <= anchor_ns),
                       key=lambda r: r['receive_monotonic_ns'], default=None)

    def status(self):
        with self._lock:
            return dict(self._stats)

    def stop(self):
        self._stop.set()
        if self._thread:
            # One traffic batch can contain all configured route lights and a
            # bounded retry for each, so a single RPC timeout is not enough.
            traffic_calls=max(1,2*len(self.config.get('traffic_light_ids',()))+2)
            self._thread.join(timeout=max(5.,traffic_calls*self.timeout+1.))
            if self._thread.is_alive():
                raise TimeoutError('Actor/traffic producer did not stop; raw evidence must remain unsealed')
            self._identity_producer_done.set()
            if getattr(self, '_identity_thread', None):
                jobs=getattr(self,'_identity_jobs',None)
                pending=jobs.qsize() if jobs is not None else 0
                drain_timeout=min(45.,max(5.,(pending+1)*(self.timeout+.05)))
                self._identity_thread.join(timeout=drain_timeout)
                if self._identity_thread.is_alive():
                    raise TimeoutError('Actor identity queue did not drain; raw evidence must remain unsealed')
            self._channel.close()


def _xyz(value):
    return [float(value[k]) for k in ('x', 'y', 'z')]


def normalize_inventory(snapshot, ros_objects, ego, anchor_ns, config):
    """Preserve every actor; invalid or incomplete observations cannot label background."""
    meta = {'schema': 'grpc_actor_bounds_v1', 'valid': False, 'reasons': [],
            'normalization_revision': 'actor_si_velocity_v2',
            'source_timestamp_units_per_second': config.get('source_timestamp_units_per_second', 1e6),
            'normalization_verified': config.get('geometry_verified') is True,
            'extent_axes': 'width_length_height',
            'speed_source': 'vehicle_global_velocity_kmh_div3.6; other_actor_source_position_difference',
            'snapshot': snapshot, 'ros_matches': [], 'ros_unmatched_ids': []}
    if snapshot is None or 'error' in snapshot:
        meta['reasons'].append('missing_or_failed_actor_rpc')
        return [], meta
    age = anchor_ns - snapshot['receive_monotonic_ns']
    duration = snapshot['receive_monotonic_ns'] - snapshot['request_monotonic_ns']
    meta.update(receive_age_ns=age, acquisition_age_upper_ns=age + duration)
    if not 0 <= age <= int(config.get('max_age_sec', .15) * 1e9):
        meta['reasons'].append('stale_actor_inventory')
    if not 0 <= duration <= int(config.get('max_rpc_duration_sec', .1) * 1e9):
        meta['reasons'].append('actor_rpc_latency')
    states = snapshot['response'].get('states', [])
    ros_by_id = {str(r['id']): r for r in ros_objects}
    seen = set()
    objects = []
    ego_states = []
    kinds = {'OBJECT_TYPE_VEHICLE': 'vehicle', 'OBJECT_TYPE_PEDESTRIAN': 'pedestrian',
             'OBJECT_TYPE_OBSTACLE': 'static'}
    for actor in states:
        info = actor.get('actor_info', {})
        ident = str(info.get('id', {}).get('value', ''))
        if ident == 'Ego':
            ego_states.append(actor)
            continue
        if not ident or ident in seen:
            meta['reasons'].append('missing_or_duplicate_actor_id')
            continue
        seen.add(ident)
        kind = kinds.get(info.get('object_type'))
        raw = ros_by_id.get(ident, {})
        box = actor.get('bounding_box', {})
        valid = True
        try:
            ext = _xyz(box['extent'])
            pos = _xyz(box['location'])
            rot = _xyz(box['rotation'])
            vel = _xyz(actor['global_velocity'])
            if kind == 'vehicle':
                vel = [v/3.6 for v in vel]
            else:
                estimate = snapshot.get('motion_estimates', {}).get(ident)
                if estimate is not None:
                    vel = list(estimate['velocity_xyz_mps'])
                else:
                    # A new or paused actor's local velocity can be a command,
                    # not actual motion; never turn an unsupported field into 0 GT.
                    valid = False
                    meta['reasons'].append('actor_motion_not_yet_measured:' + ident)
            pivot = _xyz(actor['transform']['location'])
            if not all(math.isfinite(x) for x in ext + pos + rot + vel + pivot):
                raise ValueError('nonfinite')
            if min(ext) <= 0 or kind is None:
                raise ValueError('missing_extent_or_type')
            # Native targets are upright 2D OBBs. Strong tilt requires a full
            # projected-box formulation, not silently treating this as upright.
            if any(abs((a + 180) % 360 - 180) > 10 for a in rot[:2]):
                valid = False
            if raw:
                error = math.dist(raw['position'], pivot)
                dt_bound = max(0, age + duration) / 1e9 + .25
                tolerance = .5 + math.hypot(*vel[:2]) * dt_bound
                dims_error = max(abs(a-b) for a, b in zip(
                    sorted(raw['size_xyz_raw_m']), sorted(2*x for x in ext)))
                meta['ros_matches'].append({'id': ident, 'pivot_error_m': error,
                                            'dimension_error_m': dims_error})
                if error > tolerance or raw['type'] != kind:
                    valid = False
                    meta['reasons'].append('ros_actor_identity_or_pose_mismatch')
        except (KeyError, ValueError, TypeError):
            ext, pos, rot, vel = [0.]*3, [0.]*3, [0.]*3, [0.]*3
            valid = False
        objects.append(GTObject(obj_id=ident, obj_type=kind or 'unknown',
            x=pos[0], y=pos[1], z=pos[2], vel_x=vel[0], vel_y=vel[1],
            heading_deg=rot[2], width=ext[0]*2, length=ext[1]*2, height=ext[2]*2,
            model_name=raw.get('model_name', ''), geometry_valid=valid))
        if not valid:
            meta['reasons'].append('invalid_actor_geometry:' + ident)
    meta['ros_unmatched_ids'] = sorted(set(ros_by_id) - seen - {'-1', '0', 'Ego'})
    if meta['ros_unmatched_ids']:
        # Birth/death disagreement is recorded and masked for this observation.
        meta['reasons'].append('ros_actor_absent_from_grpc_inventory')
    if len(ego_states) != 1:
        meta['reasons'].append('missing_or_duplicate_grpc_ego')
    else:
        a = ego_states[0]
        try:
            pos = _xyz(a['transform']['location'])
            vel = [v/3.6 for v in _xyz(a['global_velocity'])]
            error = math.hypot(pos[0]-ego.x, pos[1]-ego.y)
            meta['ego_position_error_m'] = error
            meta['ego_collision_objects'] = a.get('vehicle_state', {}).get('collision_objects', [])
            meta['ego_grpc_speed_mps'] = math.hypot(*vel[:2])
            bound = max(0, age + duration + abs(anchor_ns-ego.timestamp_ns)) / 1e9
            if not math.isfinite(error) or error > .75 + max(ego.speed, math.hypot(*vel[:2])) * bound:
                meta['reasons'].append('ros_grpc_ego_motion_mismatch')
        except (KeyError, TypeError, ValueError):
            meta['reasons'].append('invalid_grpc_ego')
    meta['actor_count'] = len(objects)
    meta['valid'] = not meta['reasons']
    return objects, meta
