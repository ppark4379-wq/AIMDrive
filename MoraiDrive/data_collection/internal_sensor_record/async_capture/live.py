"""Raw subscriptions with no image/point-cloud conversion or vehicle commands."""
from collections import Counter, deque
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time

from async_capture.journal import json_bytes
from async_capture.clock_health import snapshot as clock_snapshot
from async_capture.temporal_quality import update_receipt_bounds


def header_metadata(payload, has_header):
    if not has_header:
        return dict(seq=None, source_timestamp_ns=None, frame_id=None)
    if len(payload) < 16:
        raise ValueError('Truncated ROS header')
    seq, sec, nsec, length = struct.unpack_from('<IIII', payload)
    if nsec >= 10**9 or length > 65536 or 16 + length > len(payload):
        raise ValueError('Malformed ROS header')
    return dict(seq=seq, source_timestamp_ns=sec*10**9+nsec,
                frame_id=payload[16:16+length].decode('utf-8'))


def gt_poller_failures(status, required_light_ids, now_ns, max_age_ns=1_000_000_000,
                       source_units_per_second=1e6, traffic_source_period_sec=.05):
    """Fail readiness on missing/stale GT without treating a paused scene as stale.

    Actor RPC liveness is a receive-monotonic property. Traffic polling is
    deliberately paced in simulator source time, so its freshness must use the
    same source domain once both source witnesses exist.
    """
    failures = []
    actor_last = int(status.get('last_actor_success_monotonic_ns', 0))
    if not actor_last or now_ns - actor_last > max_age_ns:
        failures.append('actor_ground_truth_missing_or_stale')
    if required_light_ids:
        traffic_last = int(status.get('last_traffic_available_monotonic_ns', 0))
        actor_source=status.get('last_actor_source_raw')
        traffic_source=status.get('last_traffic_source_raw')
        actor_generation=status.get('last_actor_generation')
        traffic_generation=status.get('last_traffic_generation')
        source_lag_ok=False
        if actor_source is not None and traffic_source is not None:
            lag=int(actor_source)-int(traffic_source)
            max_lag=int(float(source_units_per_second)*max(
                .3,4*float(traffic_source_period_sec)))
            source_lag_ok=(actor_generation==traffic_generation and 0<=lag<=max_lag)
        # Backward compatibility for old status producers without source
        # witnesses. New live captures never substitute wall age when source
        # witnesses exist.
        if (not traffic_last or
                (actor_source is not None and traffic_source is not None and not source_lag_ok) or
                ((actor_source is None or traffic_source is None) and now_ns-traffic_last>max_age_ns)):
            failures.append('traffic_ground_truth_missing_or_stale')
        for light_id in required_light_ids:
            if not status.get('traffic_query_success:' + str(light_id), 0):
                failures.append('traffic_light_never_observed:' + str(light_id))
            if status.get('traffic_query_failure_streak:' + str(light_id), 0) >= 5:
                failures.append('traffic_light_repeated_rpc_failure:' + str(light_id))
    return failures


class RawSubscriptions:
    def __init__(self, writer, contract, time_mode='realtime'):
        if time_mode not in ('realtime','simulation_free_run'):raise ValueError('Unknown time mode')
        self.time_mode=time_mode
        self.writer, self.contract = writer, contract
        self.lock = threading.Lock()
        self.streams, self.definitions = {}, {}
        self.health = None
        self.bridge_instance = None
        self.topology=contract.get('capture',{}).get('bridge_topology')
        if self.topology is not None:
            from async_capture.bridge_topology import validate_config
            validate_config(self.topology)
        self.healths={};self.bridge_instances={}
        self.subscribers = []
        self.errors = []
        # Keep enough bounded context to diagnose rare ROS callback failures.
        # This is diagnostic-only: the existing errors list still makes the
        # capture fail closed.
        self.callback_errors = deque(maxlen=32)
        self.poller = None
        self.classes = {}
        self.callback_cv = threading.Condition()
        self.active_callbacks = 0
        self.stopping = False
        self.motion_origin = {}
        self.motion_current = {}
        self.clock_history = deque(maxlen=2000)
        self.clock_backward_events = deque(maxlen=128)
        self.require_ros_clock = bool(contract.get('capture',{}).get('require_ros_clock',False))
        self.record_ros_cameras = bool(
            contract.get('capture', {}).get('record_ros_cameras', True)
        )
        self.ros_camera_transport_policy = contract.get('capture', {}).get(
            'ros_camera_transport_policy', 'required'
        )
        if self.ros_camera_transport_policy not in ('required','unrecorded','bridge_dropped','absent'):
            raise ValueError('Unknown ROS camera transport policy')
        if self.record_ros_cameras != (self.ros_camera_transport_policy == 'required'):
            raise ValueError('ROS camera capture and transport policy disagree')
        self.clock_source_guard = None
        if contract.get('capture',{}).get('require_single_clock_source',False):
            if not self.require_ros_clock or time_mode!='simulation_free_run' or self.topology:
                raise ValueError('Single clock source guard requires actual clock and one Simulation Time bridge')
            from async_capture.clock_source import ClockSourceGuard
            self.clock_source_guard = ClockSourceGuard()

    def start(self):
        import rospy
        from data_collection.config import ROS
        from ros.actor_ground_truth import ActorStatePoller
        self.rospy = rospy
        rospy.init_node('morai_async_raw', anonymous=True, disable_signals=True)
        capture = self.contract.get('capture', {})
        specs = {capture.get('camera_topic', '/image_jpeg/compressed'): 'camera',
            ROS.topics['lidar']: 'lidar', ROS.topics['ego_state']: 'ego',
            ROS.topics['gnss']: 'gnss', ROS.topics['imu']: 'imu',
            ROS.topics['gt_objects']: 'objects', ROS.topics['traffic_light']: 'traffic_light',
            '/CollisionData': 'collision', '/tf': 'tf', '/clock': 'clock',
            '/SyncModeInfo': 'sync_mode', '/stage6e/bridge_health': 'bridge_health',
            '/expert_target_velocity': 'expert_target_velocity'}
        if not self.record_ros_cameras:
            specs.pop(capture.get('camera_topic', '/image_jpeg/compressed'))
        if capture.get('expert_control_topic'):
            specs[capture['expert_control_topic']] = 'expert_control'
        if capture.get('record_expert_paths', False):
            specs.update({'/local_path': 'expert_local_path', '/global_path': 'expert_global_path'})
        if self.topology is not None:
            specs.pop('/stage6e/bridge_health')
            for entry in self.topology:specs[entry['health_topic']]='bridge_health'
        for topic, role in specs.items():
            self.subscribers.append(rospy.Subscriber(topic, rospy.AnyMsg, self.receive,
                callback_args=(topic, role), queue_size=100, buff_size=64*1024**2,
                tcp_nodelay=True))
        cfg = capture.get('actor_ground_truth')
        if cfg:
            cfg=dict(cfg)
            cfg['traffic_light_ids']=capture.get('traffic_light_ground_truth',{}).get('light_ids',())
            self.poller = ActorStatePoller(cfg)
            self.poller.on_record = self.actor_record
            self.poller.on_actor_identity_record = self.actor_identity_record
            self.poller.on_traffic_record = self.traffic_record
            self.poller.start()

    def actor_record(self, row):
        self.writer.submit(dict(role='actor_inventory', encoding='json',
            receive_monotonic_ns=row['receive_monotonic_ns'],
            receive_wall_ns=row['receive_wall_ns'],
            available_monotonic_ns=row['available_monotonic_ns'],
            source_timestamp_raw=row.get('response', {}).get('timestamp'), seq=row['seq'],
            journal_observation_clocks=clock_snapshot()), json_bytes(row))

    def actor_identity_record(self, row):
        self.writer.submit(dict(role='actor_identity', encoding='json',
            receive_monotonic_ns=row['receive_monotonic_ns'],
            receive_wall_ns=row['receive_wall_ns'],
            available_monotonic_ns=row['available_monotonic_ns'],
            source_timestamp_raw=row.get('before_actor_source_raw'), seq=row['seq'],
            journal_observation_clocks=clock_snapshot()), json_bytes(row))

    def traffic_record(self, row):
        self.writer.submit(dict(role='traffic_light_inventory', encoding='json',
            receive_monotonic_ns=row['after_receive_monotonic_ns'],
            receive_wall_ns=row['after_receive_wall_ns'],
            available_monotonic_ns=row['available_monotonic_ns'],
            source_timestamp_raw=row.get('before_actor_source_raw'), seq=row['seq'],
            journal_observation_clocks=clock_snapshot()), json_bytes(row))

    def receive(self, msg, spec):
        clocks=clock_snapshot()
        mono, wall = clocks['monotonic_ns'],clocks['wall_ns']
        with self.callback_cv:
            if self.stopping:
                return
            self.active_callbacks += 1
        try:
            self._receive(msg, spec, mono, wall,clocks)
        finally:
            with self.callback_cv:
                self.active_callbacks -= 1
                self.callback_cv.notify_all()

    def _callback_error_diagnostic(self, msg, spec, mono, wall, clocks, exc,
                                   connection, typ, md5, cls, payload,
                                   serialization):
        topic, role = spec
        if payload is None and hasattr(msg, '_buff'):
            try:
                payload = bytes(msg._buff)
            except Exception:
                payload = None

        header = None
        if payload is not None and len(payload) >= 16:
            seq, sec, nsec, frame_length = struct.unpack_from('<IIII', payload)
            header = dict(
                seq=seq,
                sec=sec,
                nsec=nsec,
                frame_id_length=frame_length,
                payload_bytes_after_header=len(payload) - 16,
                nsec_in_range=nsec < 10**9,
                frame_id_length_in_range=frame_length <= 65536,
                frame_id_within_payload=16 + frame_length <= len(payload),
            )

        deserialization = None
        if payload is not None and cls is not None:
            try:
                cls().deserialize(payload)
                deserialization = dict(status='succeeded')
            except Exception as decode_error:
                deserialization = dict(
                    status='failed',
                    error=(type(decode_error).__name__ + ': ' +
                           str(decode_error)[:500]),
                )

        connection = connection if isinstance(connection, dict) else {}

        def connection_text(name):
            value = connection.get(name)
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            return str(value) if value is not None else None

        return dict(
            topic=topic,
            role=role,
            ros_type=typ,
            md5sum=md5,
            caller_id=connection_text('callerid'),
            connection_topic=connection_text('topic'),
            latching=connection_text('latching'),
            message_object_class=(type(msg).__module__ + '.' +
                                  type(msg).__name__),
            ros_class=(
                getattr(cls, '__module__', type(cls).__module__) + '.' +
                getattr(cls, '__name__', type(cls).__name__)
                if cls is not None else None
            ),
            ros_class_has_header=(
                bool(getattr(cls, '_has_header', False))
                if cls is not None else None
            ),
            has_raw_buffer=hasattr(msg, '_buff'),
            serialization=serialization,
            receive_monotonic_ns=mono,
            receive_wall_ns=wall,
            receive_monotonic_raw_ns=(
                clocks.get('monotonic_raw_ns') if clocks else None
            ),
            receive_clock_bracket_ns=(
                clocks.get('raw_bracket_ns') if clocks else None
            ),
            stopping_observed=bool(self.stopping),
            payload_bytes=(len(payload) if payload is not None else None),
            payload_sha256=(
                hashlib.sha256(payload).hexdigest()
                if payload is not None else None
            ),
            payload_prefix_hex=(
                payload[:64].hex() if payload is not None else None
            ),
            header_preview=header,
            ros_deserialization=deserialization,
            error=type(exc).__name__ + ': ' + str(exc)[:500],
        )

    def _receive(self, msg, spec, mono, wall, clocks=None):
        topic, role = spec
        connection = {}
        typ = md5 = None
        cls = payload = serialization = None
        try:
            import roslib.message
            connection = msg._connection_header
            typ, md5 = connection['type'], connection['md5sum']
            cls = self.classes.get(typ)
            if cls is None:
                cls = roslib.message.get_message_class(typ)
                if cls is None:
                    raise ValueError('Unavailable ROS message class: '+typ)
                self.classes[typ] = cls
            if cls._md5sum != md5:
                raise ValueError('ROS message definition differs: '+typ)
            # rospy's internal simulated-time subscriber shares /clock's
            # transport, so an AnyMsg callback can receive a typed Clock.
            # Clock contains exactly two uint32s: reserialization preserves
            # both transmitted values; never substitute rospy.Time.now().
            serialization = 'received_ros1_bytes'
            if hasattr(msg, '_buff'):
                payload = bytes(msg._buff)
            elif role == 'clock' and typ == 'rosgraph_msgs/Clock':
                import io
                buffer = io.BytesIO()
                msg.serialize(buffer)
                payload = buffer.getvalue()
                if len(payload) != 8:
                    raise ValueError('Invalid Clock payload length')
                serialization = 'typed_clock_lossless_reserialization'
            else:
                raise ValueError('Expected original serialized ROS bytes: '+typ)
            meta = dict(role=role, topic=topic, encoding='ros1', ros_type=typ, md5sum=md5,
                caller_id=connection.get('callerid'), serialization=serialization, receive_monotonic_ns=mono,
                receive_wall_ns=wall, **header_metadata(payload, cls._has_header))
            if clocks is not None:
                meta.update(receive_monotonic_raw_ns=clocks['monotonic_raw_ns'],
                            receive_clock_bracket_ns=clocks['raw_bracket_ns'])
            with self.lock:
                if role == 'clock':
                    sec,nsec=struct.unpack('<II',payload)
                    if nsec>=10**9: raise ValueError('Invalid /clock nanoseconds')
                    stamp=sec*10**9+nsec
                    if self.clock_history and stamp<self.clock_history[-1][1]:
                        # This is an observed regression, not proof that MORAI
                        # reset. Delayed duplicate /clock producers can also
                        # yield 860 -> 870 -> 860 -> 870 ms (observed in raw).
                        self.clock_backward_events.append(dict(
                            previous_receive_monotonic_ns=self.clock_history[-1][0],
                            previous_clock_ns=self.clock_history[-1][1],
                            receive_monotonic_ns=mono, clock_ns=stamp,
                            delta_ns=stamp-self.clock_history[-1][1],
                            caller_id=meta['caller_id'], cause='unresolved'))
                        self.errors.append('ros_clock_reset_during_capture')
                    self.clock_history.append((mono,stamp))
                witness=next(((t,v) for t,v in reversed(self.clock_history) if t<=mono),None)
                meta['preceding_received_clock_ns']=witness[1] if witness else None
                meta['preceding_clock_receive_monotonic_ns']=witness[0] if witness else None
            ok = self.writer.submit(meta, payload)
            stream = role + (':'+str(meta['frame_id']) if role == 'camera' else '')
            with self.lock:
                self.definitions[typ] = dict(md5sum=md5, definition=cls._full_text)
                info = self.streams.setdefault(stream, dict(count=0, bytes=0, first_receive_ns=mono,
                    first_source_ns=meta['source_timestamp_ns'], publishers=[]))
                if mono>=info.get('last_receive_ns',0):
                    info['last_preceding_received_clock_ns']=meta['preceding_received_clock_ns']
                update_receipt_bounds(info, mono, meta['source_timestamp_ns'])
                info['count'] += 1
                info['bytes'] += len(payload)
                if meta['caller_id'] not in info['publishers']:
                    info['publishers'].append(meta['caller_id'])
                if role == 'bridge_health':
                    health_msg = cls().deserialize(payload)
                    self.health = (json.loads(health_msg.data), mono, meta['caller_id'])
                    instance=self.health[0].get('instance_id')
                    previous=self.bridge_instances.get(topic)
                    if previous is not None and instance!=previous:
                        self.errors.append('bridge_instance_changed_during_run')
                    self.healths[topic]=self.health;self.bridge_instances[topic]=instance
                    self.bridge_instance=instance
                    if getattr(self,'clock_source_guard',None) is not None:
                        for error in self.clock_source_guard.observe(self.health[0],mono):
                            if error not in self.errors:self.errors.append(error)
                if role == 'sync_mode':
                    state=cls().deserialize(payload)
                    error=None
                    if self.time_mode=='realtime' and state.status:
                        error='sync_mode_activated_during_async_capture'
                    elif self.time_mode=='simulation_free_run' and (not state.status or state.can_send_tick):
                        error='simulation_free_run_state_changed'
                    if error and error not in self.errors:self.errors.append(error)
                if role in ('ego','gnss'):
                    small=cls().deserialize(payload)
                    value=([float(small.position.x),float(small.position.y),math.hypot(small.velocity.x,small.velocity.y)]
                        if role=='ego' else [float(small.latitude),float(small.longitude)])
                    if all(math.isfinite(x) for x in value) and (role!='gnss' or abs(value[0])+abs(value[1])>1e-6):
                        self.motion_current[role]=value
                        self.motion_origin.setdefault(role,value)
                if not ok and not self.errors:
                    self.errors.append('raw_writer_refused_record')
        except Exception as exc:
            try:
                diagnostic = self._callback_error_diagnostic(
                    msg, spec, mono, wall, clocks, exc, connection, typ, md5,
                    cls, payload, serialization,
                )
            except Exception as diagnostic_error:
                diagnostic = dict(
                    topic=topic,
                    role=role,
                    error=type(exc).__name__ + ': ' + str(exc)[:500],
                    diagnostic_error=(type(diagnostic_error).__name__ + ': ' +
                                      str(diagnostic_error)[:500]),
                )
            with self.lock:
                self.errors.append(type(exc).__name__+': '+str(exc))
                self.callback_errors.append(diagnostic)

    def readiness(self):
        from ros.bridge_preflight import validate_bridge
        now = time.monotonic_ns()
        with self.lock:
            streams = {k: dict(v) for k,v in self.streams.items()}
            failures = list(self.errors)
            health = self.health
            healths=dict(self.healths)
            clock_history=list(self.clock_history)
            clock_backward_events=list(self.clock_backward_events)
            callback_errors=list(self.callback_errors)
        expected = (
            ['camera:'+k for k in self.contract['capture']['camera_frame_ids']]
            if self.record_ros_cameras else []
        )
        expected += ['lidar', 'ego', 'gnss', 'imu']
        clock_policy=self.time_mode=='simulation_free_run' and self.require_ros_clock
        if clock_policy:
            failures += ros_clock_liveness(clock_history,streams,expected,now)
        else:
            for role in expected:
                s = streams.get(role, {})
                if s.get('count', 0) < 2 or now-s.get('last_receive_ns', 0) > 10**9:
                    failures.append('missing_or_stale:'+role)
        if self.topology is not None:
            from async_capture.bridge_topology import failures as topology_failures
            failures += topology_failures(self.topology,healths,streams,now)
        elif health is None:
            failures.append('missing_bridge_health')
        else:
            required_bridge_streams={'/lidar3D','/gps','/imu','/Ego_topic'}
            if self.ros_camera_transport_policy != 'absent':
                required_bridge_streams.update(
                    self.contract['capture']['camera_frame_ids']
                )
            failures += validate_bridge(
                health[0], now, health[2],
                max_stream_age_ms=3000 if clock_policy else 500,
                required_streams=required_bridge_streams,
                expected_camera_ingress_policy=(
                    'drop_before_ros'
                    if self.ros_camera_transport_policy == 'bridge_dropped'
                    else None
                ),
            )
            for role in expected:
                if streams.get(role, {}).get('publishers') != [health[0].get('node')]:
                    failures.append('unexpected_publisher:'+role)
        if self.poller:
            poller_status = self.poller.status()
            if (poller_status.get('raw_record_errors')
                    or poller_status.get('identity_raw_record_errors')
                    or poller_status.get('traffic_raw_record_errors')):
                failures.append('actor_raw_record_error')
            required_lights=self.contract.get('capture',{}).get(
                'traffic_light_ground_truth',{}).get('light_ids',())
            actor_cfg=self.contract.get('capture',{}).get('actor_ground_truth',{})
            failures += gt_poller_failures(poller_status, required_lights, now,
                source_units_per_second=actor_cfg.get('source_timestamp_units_per_second',1e6),
                traffic_source_period_sec=actor_cfg.get('traffic_source_period_sec',.05))
        motion=False
        with self.lock:
            if all(k in self.motion_current and k in self.motion_origin for k in ('ego','gnss')):
                e0,e=self.motion_origin['ego'],self.motion_current['ego']
                g0,g=self.motion_origin['gnss'],self.motion_current['gnss']
                de=math.hypot(e[0]-e0[0],e[1]-e0[1])
                dg=6371000*math.hypot(math.radians(g[0]-g0[0]),math.radians(g[1]-g0[1])*math.cos(math.radians(g[0])))
                motion=e[2]>.5 and min(de,dg)>1 and abs(de-dg)<max(2.,.4*max(de,dg))
        return dict(ready=not failures, failures=failures, streams=streams,motion_consistent=motion,
            clock_backward_events=clock_backward_events,
            callback_errors=callback_errors,
            definitions=dict(self.definitions), actor_poller=self.poller.status() if self.poller else None)

    def stop(self):
        with self.callback_cv:
            self.stopping = True
        for sub in self.subscribers:
            sub.unregister()
        if self.poller:
            self.poller.stop()
        with self.callback_cv:
            if not self.callback_cv.wait_for(lambda: self.active_callbacks == 0, timeout=5):
                raise TimeoutError('Callbacks did not drain; leave raw run unsealed')
        if hasattr(self, 'rospy'):
            self.rospy.signal_shutdown('raw recording completed')


def ros_clock_liveness(history,streams,expected,now):
    """Liveness only: 300ms observed SIM progress + 3s wall watchdog.

    Keeps transport queue/error guards and downstream 80ms pairing untouched.
    A timely stream can still contain a missing frame; offline cadence auditing
    remains mandatory. Receipt-clock witnesses never become exposure labels.
    """
    if not history or now-history[-1][0]>3_000_000_000:
        return ['missing_or_stale_ros_clock']
    current=history[-1][1]
    previous=next((t for t,v in reversed(history) if v!=current),None)
    if previous is None or now-previous>3_000_000_000:
        return ['ros_clock_not_advancing']
    errors=[]
    for role in expected:
        s=streams.get(role,{})
        stamp=s.get('last_preceding_received_clock_ns')
        if (s.get('count',0)<2 or now-s.get('last_receive_ns',0)>3_000_000_000
                or stamp is None or not 0<=current-stamp<=300_000_000):
            errors.append('missing_or_stale_simulation_stream:'+role)
    return errors
