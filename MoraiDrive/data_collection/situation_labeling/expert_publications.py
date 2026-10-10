"""Read captured Expert plans without equating receive time with exposure time.

The native LiDAR association identifies the exact *receipt* used as an anchor.
Only earlier publications within explicit receive/simulation-clock bounds are
admitted. Unstamped CtrlCmd/Float32/Path messages never acquire a fabricated
source timestamp. Sensor payloads are not read. Every decoded Expert payload
is checked against the sealed quality index and captured ROS definition.
"""
from bisect import bisect_right
from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct

SCHEMA = 'recorded_expert_publications_v1'
ROLES = {'expert_control': 'control', 'expert_target_velocity': 'target_velocity',
         'expert_local_path': 'local_path'}
CTRL_FIELDS = ['int32 longlCmdType', 'float64 accel', 'float64 brake',
               'float64 front_steer', 'float64 rear_steer', 'float64 velocity',
               'float64 acceleration']
MD5 = {'morai_msgs/CtrlCmd': 'b36ceb480d187645a32b95c2be82bf56',
       'std_msgs/Float32': '73fcbf46b49191e672908e50842a83d4',
       'nav_msgs/Path': '6227e2b7e9cce15051f669a5e197bbf7'}


class UnsupportedPublication(ValueError):
    """An intact recorded message cannot justify a supported current plan."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, expected=None):
    data = Path(path).read_bytes()
    digest = _digest(data)
    if expected is not None and digest != expected:
        raise ValueError('Recorded Expert input changed: ' + str(path))
    return json.loads(data), digest


def _header(data, pos):
    seq, sec, ns, length = struct.unpack_from('<IIII', data, pos)
    pos += 16
    if ns >= 1000000000 or length > 10000 or pos + length > len(data):
        raise ValueError('Invalid captured ROS header')
    frame = data[pos:pos + length].decode('utf-8')
    return pos + length, {'seq': seq, 'stamp_ns': sec * 1000000000 + ns, 'frame': frame}


def decode_publication(row, data, definitions):
    """Decode only a known captured schema; no ROS/genpy dependency."""
    typ, role = row.get('ros_type'), row.get('role')
    definition = definitions.get(typ) or {}
    if (typ not in MD5 or row.get('md5sum') != MD5[typ]
            or definition.get('md5sum') != row['md5sum']):
        raise ValueError('Unrecognized captured Expert ROS definition')
    if role == 'expert_control':
        fields = [x.split('#')[0].strip() for x in definition.get('definition', '').splitlines()
                  if x.split('#')[0].strip()]
        if typ != 'morai_msgs/CtrlCmd' or fields != CTRL_FIELDS or len(data) != 52:
            raise ValueError('Unexpected CtrlCmd layout')
        vals = struct.unpack('<i6d', data)
        result = dict(zip(('longlCmdType', 'accel', 'brake', 'front_steer_rad',
                           'rear_steer_rad', 'velocity_kmh', 'acceleration_mps2'), vals))
        if not all(math.isfinite(x) for x in vals[1:]):
            raise ValueError('Nonfinite CtrlCmd')
        if not 0 <= result['accel'] <= 1 or not 0 <= result['brake'] <= 1:
            raise ValueError('Invalid throttle/brake command')
        return result
    if role == 'expert_target_velocity':
        if typ != 'std_msgs/Float32' or len(data) != 4:
            raise ValueError('Unexpected target velocity layout')
        value = struct.unpack('<f', data)[0]
        if not math.isfinite(value):
            raise ValueError('Invalid Expert target velocity')
        if value < 0:
            # The publisher archives the controller's previous target verbatim.
            # Finite negative values exist in intact captures. They are neither
            # corrupt bytes nor a verified nonnegative target for this labeler.
            # Preserve the signed value; do not silently clamp it to a stop.
            raise UnsupportedPublication('Negative captured Expert target velocity',
                                         {'raw_value_mps': value})
        return {'value_mps': value, 'unit_basis': 'MORAI Expert /expert_target_velocity publisher'}
    if role != 'expert_local_path' or typ != 'nav_msgs/Path':
        raise ValueError('Unsupported Expert publication')
    pos, header = _header(data, 0)
    count = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    if count > 20000:
        raise ValueError('Unexpected Expert path point count')
    unsupported = count < 2 or header['frame'] != 'map'
    points = []
    for _ in range(count):
        pos, pose_header = _header(data, pos)
        unsupported |= pose_header['frame'] not in ('', 'map')
        values = struct.unpack_from('<7d', data, pos)
        pos += 56
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite Expert path pose')
        points.append(list(values[:3]))
    if pos != len(data):
        raise ValueError('Unexpected trailing Expert path bytes')
    if unsupported:
        raise UnsupportedPublication('Expert path lacks a supported map-frame polyline')
    return {'header': header, 'world_xyz': points}


class PublicationReader:
    def __init__(self, episode, max_age_ms=100.0):
        self.episode = Path(episode).resolve()
        self.bound_ns = int(max_age_ms * 1000000)
        if not 0 < self.bound_ns <= 250000000:
            raise ValueError('Expert publication bound must be in (0,250] ms')
        self.pins, self.handles, self.cache = {}, OrderedDict(), {}
        quality = self.episode / 'quality'
        seal, digest = _read(quality / 'QUALITY_COMPLETE.json')
        self.pins[str(quality / 'QUALITY_COMPLETE.json')] = digest
        expected = (seal.get('files') or {}).get('active_signals.json')
        if not expected:
            raise ValueError('Quality seal omits active signal index')
        signals, digest = _read(quality / 'active_signals.json', expected)
        self.pins[str(quality / 'active_signals.json')] = digest
        raw = self.episode / 'raw'
        completed, digest = _read(raw / 'RAW_COMPLETE.json')
        self.pins[str(raw / 'RAW_COMPLETE.json')] = digest
        if completed.get('status') != 'completed' or not completed.get('manifest_sha256'):
            raise ValueError('Captured Expert raw journal is incomplete')
        manifest, digest = _read(raw / 'raw_manifest.json', completed['manifest_sha256'])
        self.pins[str(raw / 'raw_manifest.json')] = digest
        if seal.get('raw_run_id') != manifest.get('run_id'):
            raise ValueError('Quality index belongs to another raw run')
        self.manifest_sha = digest
        self.chunks = {x['path']: x for x in manifest['chunks']}
        self.definitions = manifest.get('extra', {}).get('topic_state', {}).get('definitions', {})
        assoc_dir = self.episode / 'native_associations'
        assoc_seal, digest = _read(assoc_dir / 'COMPLETE.json')
        self.pins[str(assoc_dir / 'COMPLETE.json')] = digest
        if assoc_seal.get('raw_manifest_sha256') != self.manifest_sha:
            raise ValueError('Native associations belong to another raw run')
        expected = (assoc_seal.get('files') or {}).get('associations.json')
        if not expected:
            raise ValueError('Native associations lack a sealed frame mapping')
        associations, digest = _read(assoc_dir / 'associations.json', expected)
        self.pins[str(assoc_dir / 'associations.json')] = digest
        self.associations = {}
        for row in associations:
            if row['index'] in self.associations:
                raise ValueError('Duplicate native association frame')
            self.associations[row['index']] = row
        self.rows = {role: [] for role in (*ROLES, 'ego')}
        self.lidars, identities = {}, set()
        for row in signals:
            if row.get('role') not in (*ROLES, 'ego', 'lidar'):
                continue
            identity = row.get('record_id')
            if identity in identities:
                raise ValueError('Duplicate captured publication record ID')
            identities.add(identity)
            if type(row.get('receive_monotonic_ns')) is not int:
                continue
            if row['role'] == 'lidar':
                self.lidars[identity] = row
            else:
                self.rows[row['role']].append(row)
        for role, rows in self.rows.items():
            rows.sort(key=lambda r: (r['receive_monotonic_ns'], r['record_id']))
        self.times = {role: [r['receive_monotonic_ns'] for r in rows]
                      for role, rows in self.rows.items()}
        self.ego_by_source = {}
        for row in self.rows['ego']:
            self.ego_by_source.setdefault(row.get('source_timestamp_ns'), []).append(row)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def _before(self, role, sensor_ns, anchor, bound_ns=None, receive_bound_ns=None):
        bound = self.bound_ns if bound_ns is None else bound_ns
        receive_bound = bound if receive_bound_ns is None else receive_bound_ns
        rows, times = self.rows[role], self.times[role]
        end = bisect_right(times, anchor)
        for position in range(end - 1, -1, -1):
            row = rows[position]
            if anchor - row['receive_monotonic_ns'] > receive_bound:
                break
            clock = row.get('preceding_received_clock_ns')
            if type(clock) is int and 0 <= sensor_ns - clock <= bound:
                yield row

    def _payload(self, row):
        identity = row['record_id']
        if identity in self.cache:
            return self.cache[identity]
        relative = PurePosixPath(row['chunk_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parent != PurePosixPath('chunks'):
            raise ValueError('Unsafe captured raw chunk path')
        if str(relative) not in self.chunks:
            raise ValueError('Captured raw chunk absent from manifest')
        path = self.episode / 'raw' / str(relative)
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Captured chunk may not be a symlink')
        offset, count = row.get('payload_offset'), row.get('payload_bytes')
        if (type(offset) is not int or offset < 8 or type(count) is not int or count < 0
                or offset + count > self.chunks[str(relative)]['bytes']):
            raise ValueError('Captured payload exceeds sealed chunk bounds')
        handle = self.handles.pop(str(path), None)
        if handle is None:
            handle = path.open('rb')
        self.handles[str(path)] = handle
        if len(self.handles) > 8:
            _, old = self.handles.popitem(last=False)
            old.close()
        handle.seek(offset)
        data = handle.read(count)
        if len(data) != count or _digest(data) != row.get('payload_sha256'):
            raise ValueError('Captured Expert payload differs from sealed quality record')
        value = decode_publication(row, data, self.definitions)
        self.cache[identity] = value
        return value

    def _receipt(self, row, anchor, value):
        result = {key: row.get(key) for key in ('record_id', 'source_timestamp_ns',
                  'preceding_received_clock_ns', 'receive_monotonic_ns', 'payload_sha256',
                  'chunk_path', 'payload_offset', 'payload_bytes', 'caller_id', 'ros_type', 'md5sum')}
        result.update(value)
        result['receive_age_ms'] = (anchor - row['receive_monotonic_ns']) / 1000000.0
        result['raw_chunk'] = row['chunk_path']
        result['source_timestamp_fabricated'] = False
        return result

    def for_frame(self, frame):
        index, sensor_ns = frame['index'], frame['lidar_source_ns']
        native = self.associations.get(index)
        if native is None or native.get('scene_time_hypothesis_ns') != sensor_ns:
            return {'status': 'NO_VERIFIED_NATIVE_FRAME_ANCHOR', 'sensor_ns': sensor_ns}
        lidar = self.lidars.get(native.get('lidar_ros_record_id'))
        if lidar is None or lidar.get('source_timestamp_ns') != sensor_ns:
            return {'status': 'NO_VERIFIED_LIDAR_RECEIPT', 'sensor_ns': sensor_ns}
        anchor = lidar['receive_monotonic_ns']
        result = {'schema': SCHEMA, 'status': 'RECEIVE_BOUND_CURRENT_CONTEXT',
                  'sensor_ns': sensor_ns, 'lidar_record_id': lidar['record_id'],
                  'lidar_receive_monotonic_ns': anchor, 'max_age_ms': self.bound_ns / 1000000.0,
                  'source_sha256': self.pins, 'physical_sensor_synchrony_certified': False,
                  'timing_basis': 'last recorded publication received before associated LiDAR; separate receive/sim bounds',
                  'control': None, 'target_velocity': None, 'local_path': None, 'ego': None,
                  'reasons': [], 'unsupported_publications': [],
                  'raw_chunks_fully_rehashed': False}
        for role, name in ROLES.items():
            row = next(self._before(role, sensor_ns, anchor), None)
            if row is None:
                result['reasons'].append('no_fresh_preceding_' + role)
                continue
            # Invalid/mutated payloads are an integrity failure, not a missing label.
            try:
                value = self._payload(row)
            except UnsupportedPublication as error:
                # Empty/other-frame paths and signed negative targets can be
                # intact ROS publications without supporting this label. Never
                # replace the latest unsupported message with an older plan.
                result['reasons'].append('unsupported_current_' + role)
                result['unsupported_publications'].append(dict(
                    self._receipt(row, anchor, error.evidence),
                    role=role, purpose='unsupported_current_command_or_plan',
                    reason=str(error)))
                continue
            result[name] = self._receipt(row, anchor, value)
            if role == 'expert_target_velocity' and result[name]['value_mps'] > .01:
                # This is historical stop context, never the current target.
                # Simulation may run slower than wall time. Keep the 1s sim
                # history and a separate bounded 5s receipt history explicit.
                for prior in self._before(role, sensor_ns, row['receive_monotonic_ns'] - 1,
                                          1000000000, receive_bound_ns=5000000000):
                    try:
                        prior_value = self._payload(prior)
                    except UnsupportedPublication as error:
                        # Do not infer a departure across an unsupported target
                        # or reinterpret a negative target as a prior zero.
                        result['reasons'].append('unsupported_historical_' + role)
                        result['unsupported_publications'].append(dict(
                            self._receipt(prior, anchor, error.evidence),
                            role=role, purpose='prior_stop_context_search_barrier',
                            reason=str(error)))
                        break
                    if prior_value['value_mps'] == 0:
                        result[name]['prior_zero_target'] = self._receipt(prior, anchor, prior_value)
                        result[name]['prior_zero_target']['purpose'] = 'prior_stop_context_not_current_command'
                        result[name]['prior_zero_target']['history_sim_bound_ms'] = 1000
                        result[name]['prior_zero_target']['history_receive_bound_ms'] = 5000
                        break
        # Ego is GT teacher data, not an unstamped Expert decision. Its sealed
        # source-time association is the same hypothesis as gt_timeline_v1;
        # it may arrive >100ms *before* the LiDAR callback. Do not fabricate an
        # interpolation or change its clock to satisfy a command-receipt bound.
        exact = [r for r in self.ego_by_source.get(sensor_ns, [])
                 if r['receive_monotonic_ns'] <= anchor]
        ambiguous_exact = bool(exact and any(r.get('signal') != exact[0].get('signal') for r in exact))
        if ambiguous_exact:
            result['reasons'].append('ambiguous_same_source_ego_teacher')
            exact = []
        candidates = list(reversed(exact)) or self._before('ego', sensor_ns, anchor)
        for row in candidates:
            stamp, signal = row.get('source_timestamp_ns'), row.get('signal') or {}
            if ambiguous_exact and stamp == sensor_ns:
                continue
            if (type(stamp) is not int or not 0 <= sensor_ns - stamp <= self.bound_ns
                    or not isinstance(signal.get('position'), list)
                    or len(signal['position']) < 3):
                continue
            vals = [*signal['position'][:3], signal.get('speed'), signal.get('yaw')]
            if not all(type(x) in (float, int) and math.isfinite(x) for x in vals):
                continue
            result['ego'] = self._receipt(row, anchor, {'world_xyz': signal['position'][:3],
                'speed_mps': signal['speed'], 'yaw_rad': signal['yaw'],
                'ego_pose_source_age_ms': (sensor_ns - stamp) / 1000000.0,
                'source_association_basis': ('exact_sealed_gt_scene_source_hypothesis_and_prior_receipt'
                                             if stamp == sensor_ns else 'prior_receive_bound_ego_teacher'),
                'value_source': 'sealed normalized original ROS Ego GT teacher; yaw radians'})
            break
        return result


def attach_expert_publications(frames, episode, max_age_ms=100.0):
    """Attach internal evidence; unavailable archives never create fake plans."""
    episode = Path(episode)
    needed = [f for f in frames if f.get('zone_id') in
              ('highway', 'roundabout', 'static_obstacle', 'dynamic_obstacle')]
    if not needed:
        return {'schema': SCHEMA, 'frames': 0, 'sensor_payloads_read': 0}
    if not (episode / 'quality/QUALITY_COMPLETE.json').exists():
        for frame in needed:
            frame['expert_publications'] = {'status': 'NO_CAPTURED_EXPERT_PUBLICATIONS',
                                             'sensor_ns': frame['lidar_source_ns']}
        return {'schema': SCHEMA, 'frames': len(needed), 'unavailable': len(needed), 'sensor_payloads_read': 0}
    reader = PublicationReader(episode, max_age_ms)
    counts, roles = Counter(), Counter()
    try:
        for frame in needed:
            frame['expert_publications'] = context = reader.for_frame(frame)
            counts[context['status']] += 1
            for name in ROLES.values():
                roles[name] += context.get(name) is not None
        return {'schema': SCHEMA, 'frames': len(needed), 'status_counts': dict(counts),
                'available_roles': dict(roles), 'max_age_ms': max_age_ms,
                'source_sha256': reader.pins, 'decoded_expert_publications': len(reader.cache),
                'sensor_payloads_read': 0, 'raw_chunks_fully_rehashed': False,
                'physical_sensor_synchrony_certified': False}
    finally:
        reader.close()
