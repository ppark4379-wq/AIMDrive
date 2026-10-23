#!/usr/bin/env python3
"""Frozen VE UDP sensors -> loopback UDP footprints, without GT or control.

The receiver and sender run in the same WSL instance (shared monotonic clock).
The encoder emits class-agnostic boxes. The receiver adds local motion tracks.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'morai_ve_udp_objects_v2'
PORT = 12050
MAX_AGE_S = .5
MAX_PACKET_BYTES = 60000
REQUIRED_STREAMS = ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear',
                    'lidar', 'vehicle_status')


class RosEgoStatusReceiver:
    """Reuse the Expert's own Ego telemetry; never read simulator objects."""
    def __init__(self):
        import rospy
        from morai_msgs.msg import EgoVehicleStatus
        self.port, self.packets, self.samples = '/Ego_topic', 0, 0
        self.lock = threading.Lock()
        self.value, self.received = None, 0.
        self.subscriber = rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.callback,
                                           queue_size=1, tcp_nodelay=True)

    def callback(self, message):
        vx, vy, heading = float(message.velocity.x), float(message.velocity.y), float(message.heading)
        x, y = float(message.position.x), float(message.position.y)
        if not all(math.isfinite(value) for value in (vx, vy, heading, x, y)):
            return
        value = dict(speed=math.hypot(vx, vy), yaw=math.radians(heading),
                     yaw_degrees=heading, ego_local=(x, y))
        with self.lock:
            self.value, self.received = value, time.monotonic()
            self.source_stamp_ns = message.header.stamp.to_nsec()
            self.packets += 1
            self.samples += 1
            if not hasattr(self, 'history'):
                self.history = deque(maxlen=100)
            if self.history and self.source_stamp_ns < self.history[-1][2]['source_stamp_ns']:
                self.history.clear()
            self.history.append((value, self.received, dict(sample_id=self.samples,
                received_ns=int(self.received*1e9), source_stamp_ns=self.source_stamp_ns)))

    def latest(self):
        with self.lock:
            return self.value, time.monotonic() - self.received if self.value is not None else float('inf')

    def close(self):
        self.subscriber.unregister()

    def latest_snapshot(self):
        with self.lock:
            return (self.value, time.monotonic() - self.received if self.value is not None else float('inf'),
                    dict(sample_id=self.samples, received_ns=int(self.received * 1e9),
                         source_stamp_ns=getattr(self, 'source_stamp_ns', 0)))

    def diagnostics(self):
        with self.lock:
            return dict(packets=self.packets, samples=self.samples, counters={})

    def at_source_stamp(self, source_ns, max_delta_ns=50_000_000):
        """Causal Ego sample within the trained 50ms contract, never a future pose."""
        with self.lock:
            candidates = [item for item in getattr(self, 'history', ())
                          if 0 < item[2]['source_stamp_ns'] <= source_ns]
            if not candidates:
                return None, float('inf'), {}
            value, receipt, identity = max(candidates, key=lambda item: item[2]['source_stamp_ns'])
            if source_ns - identity['source_stamp_ns'] > max_delta_ns:
                return None, float('inf'), {}
            return value, time.monotonic()-receipt, dict(identity)


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Non-finite or non-numeric UDP value')
    return float(value)


def encode_packet(packet):
    data = json.dumps(packet, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError('VE UDP packet exceeds the bounded datagram size')
    return data


def decode_packet(data, generation, checkpoint_sha256, now_ns, max_age=MAX_AGE_S,
                  expected_frame='projected_grid', coordinates=None):
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError('Oversize VE datagram')
    packet = json.loads(data)
    if (packet['schema'] != SCHEMA or packet['generation'] != generation
            or packet['checkpoint_sha256'] != checkpoint_sha256
            or packet['output_frame'] != expected_frame
            or packet['scene_assumption'] != 'unknown'):
        raise ValueError('VE datagram identity/frame mismatch')
    if coordinates is not None:
        coordinates.validate_frame(packet)
    if type(packet['sequence']) is not int or packet['sequence'] < 1:
        raise ValueError('Invalid sequence')
    for key in ('sent_ns', 'sensor_received_ns'):
        if type(packet[key]) is not int:
            raise ValueError('Invalid monotonic timestamp')
    if not 0 <= now_ns - packet['sent_ns'] <= max_age * 1e9:
        raise ValueError('Stale or future datagram')
    if packet['ready'] is not True:
        raise ValueError('VE not ready: ' + str(packet.get('reason')))
    if not 0 < packet['sensor_received_ns'] <= packet['sent_ns']:
        raise ValueError('Invalid sensor receipt timestamp')
    if now_ns - packet['sensor_received_ns'] > max_age * 1e9:
        raise ValueError('Sensor snapshot expired')
    pose = packet['ego_pose']
    for key in ('x', 'y', 'yaw_raw_deg'):
        finite(pose[key])
    if expected_frame == 'enu':
        if (packet.get('ego_position_source') != 'GPS'
                or packet.get('ego_status_source') != 'UDP_status_speed_IMU_heading'):
            raise ValueError('ENU perception must use GPS/IMU/competition status')
        for key in ('z', 'yaw_enu_rad', 'speed_mps'):
            finite(pose[key])
        if abs(pose['speed_mps']) > 100:
            raise ValueError('Invalid ego speed')
    if not isinstance(packet['boxes'], list) or len(packet['boxes']) > 256:
        raise ValueError('Too many VE footprints')
    for box in packet['boxes']:
        if len(box['center_world']) != 2:
            raise ValueError('Invalid footprint center')
        for number in box['center_world']:
            finite(number)
        finite(box['heading_deg'])
        if not 0 < finite(box['width_m']) <= 30 or not 0 < finite(box['length_m']) <= 30:
            raise ValueError('Invalid physical footprint dimensions')
        if box['speed_mps'] is not None and not 0 <= finite(box['speed_mps']) <= 100:
            raise ValueError('Invalid scalar speed')
    return packet


def make_object_message(packet, message_type, object_type, stamp):
    """Tracked unknowns; npc_list carries nonstationary objects, not a car label.

    type remains -1. Positive IDs are local VE tracks, never simulator IDs.
    velocity.x remains scalar km/h; heading sign can be resolved by motion.
    """
    message = message_type()
    message.header.stamp, message.header.frame_id = stamp, 'map'
    for box in packet['boxes']:
        obj = object_type()
        state = box['motion_state']
        if state not in ('moving', 'stationary', 'uncertain') or box['track_id'] <= 0:
            raise ValueError('Valid VE motion track required')
        obj.unique_id, obj.type = box['track_id'], -1
        obj.name = 've_tracked_unknown:' + state
        obj.position.x, obj.position.y = box['center_world']
        obj.heading = box['planning_heading_deg']
        obj.size.x, obj.size.y, obj.size.z = box['length_m'], box['width_m'], 0.
        obj.velocity.x = box['planning_speed_mps'] * 3.6
        (message.obstacle_list if state == 'stationary' else message.npc_list).append(obj)
    message.num_of_obstacle = len(message.obstacle_list)
    message.num_of_npcs = len(message.npc_list)
    message.num_of_pedestrian = 0
    return message


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--audit-dir', type=Path, help='Record sensor and pipeline timings without changing predictions')
    parser.add_argument('--seconds', type=float, default=300.,
                        help='실행 시간(초), 0이면 종료 신호를 받을 때까지 실행')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('seconds must be finite and nonnegative; 0 means run until interrupted')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    gps_mode = config.get('runtime', {}).get('ego_source', 'ros_ego') == 'gps_imu_udp'
    coordinate_config = config.get('runtime', {}).get('coordinate_config')
    coordinates = localization = None
    if coordinate_config:
        if not gps_mode or config['runtime'].get('gps_blackout_hold', {}).get('enabled'):
            raise ValueError('Fixed ENU requires GPS/IMU UDP without blackout hold')
        sys.path.insert(0, str(ROOT.parent/'AIM2026/Expert/src'))
        from expert.src.localization.coordinate_transform import CoordinateTransform
        from expert.src.localization.gps_imu_localization import GpsImuLocalization
        from expert.src.network.udp.udp_manager import UdpManager, load_network_config
        coordinates = CoordinateTransform(coordinate_config)
        network_config = load_network_config(config['runtime']['competition_network'])
        if network_config['bind_ip'] != config['runtime']['bind']:
            raise ValueError('VE and Expert bind addresses differ')
        expected_ports = dict(gps=network_config['gps_port'], imu=network_config['imu_port'],
                              vehicle_status=network_config['status_port'])
        if any(config['ports'][name] != port for name,port in expected_ports.items()):
            raise ValueError('VE and Expert sensor ports differ')
    control_pose_hz = config.get('runtime', {}).get('control_pose_hz')
    if control_pose_hz is not None and (not gps_mode or not 1 <= float(control_pose_hz) <= 100):
        raise ValueError('Control pose stream requires GPS mode and 1..100Hz')
    if config.get('runtime', {}).get('ego_source', 'ros_ego') not in ('ros_ego', 'gps_imu_udp'):
        raise ValueError('Unknown explicit ego source')
    # The older viewer uses reusable sockets. Refuse an existing UDP consumer
    # instead of silently stealing its sensor datagrams.
    probes = []
    try:
        for name, port in config['ports'].items():
            if name == 'vehicle_status' and not gps_mode:
                continue  # Shared ROS Ego input, not a second vehicle UDP stream.
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probes.append(probe)
            try:
                probe.bind((config['runtime']['bind'], port))
            except OSError as exc:
                raise RuntimeError('UDP 센서 포트 사용 중: {}:{}; 기존 UDP 추론을 종료하세요'.format(name, port)) from exc
    finally:
        for probe in probes:
            probe.close()
    from inference.realtime_ve_objects import CHECKPOINTS, MAP_BASIS, validate_contract
    reviewed_location = any(config['checkpoint'] == v['path'] and config['checkpoint_sha256'] == v['sha256']
                            for v in CHECKPOINTS.values())
    # A deployed copy may live at a different Windows/Linux path. The loader
    # verifies the file bytes against the reviewed SHA256 before loading.
    reviewed_digest = any(config['checkpoint_sha256'] == v['sha256'] for v in CHECKPOINTS.values())
    if not (reviewed_location or reviewed_digest):
        raise ValueError('Use a reviewed E3/E4/P1 checkpoint identity')
    from utils.common.runtime import configure_runtime
    configure_runtime()
    import torch
    from data.src.loaders.morai_encoder_contract import checkpoint_encoder_contract
    from data.src.loaders.morai_map_geometry import grid_yaw
    from inference.sensors.live_frame import LiveFrameSource
    from inference.sensors.morai_ros_perception import forward_perception
    from utils.evaluation.morai_transfer_evaluation import _load_model
    from utils.evaluation.morai_ve_expert_adapter import decode_world_predictions
    from models.ve.signal_state import decode_native_signal
    torch.set_num_threads(4)
    device = config['runtime']['device']
    model, payload, digest = _load_model(Path(config['checkpoint']), config['checkpoint_sha256'], device)
    contract = validate_contract(checkpoint_encoder_contract(payload, digest), {'map_basis': MAP_BASIS})
    signal_head = None
    if config.get('signal_head'):
        from models.ve.signal_state import decode_checkpoint, extract_signal_feature, decode_signal, decode_native_signal
        signal_head = decode_checkpoint(torch.load(config['signal_head'], map_location='cpu'), digest).to(device)
        model.eval()
    route, lidar = config['route'], config['lidar']
    gps_receiver = None
    source_ports = dict(config['ports'])
    if coordinates is not None:
        network = UdpManager(network_config)
        localization = GpsImuLocalization(network, coordinates)
        status_receiver, gps_receiver = localization, network.gps
        source_ports.pop('imu')
    elif gps_mode:
        from inference.sensors.gps_imu_status import GpsImuStatusReceiver, ValidGpsReceiver
        status_receiver = GpsImuStatusReceiver(source_ports, bind=config['runtime']['bind'])
        try:
            gps_receiver = ValidGpsReceiver(source_ports['gps'], bind=config['runtime']['bind'], name='gps')
        except BaseException:
            status_receiver.close()
            raise
        # The status adapter already owns IMU; no second socket may consume it.
        source_ports.pop('imu')
        hold = config['runtime'].get('gps_blackout_hold') or {}
        if hold.get('enabled'):
            from inference.sensors.gps_blackout_hold import GpsBlackoutHold, SectionZone
            zone = hold.get('simulate_zone')
            try:
                gps_receiver = GpsBlackoutHold(gps_receiver, status_receiver, MAP_BASIS,
                    max_hold_s=float(hold.get('max_hold_s', 120.)), max_distance_m=float(hold.get('max_distance_m', 250.)),
                    latency_s=float(hold.get('latency_s', 0.)), lever_xy=tuple(hold.get('lever_xy', (0., 0.))),
                    simulate_zone=SectionZone(zone['start_xy'], zone['end_xy'], zone.get('half_width_m', 15.))
                    if zone else None)
            except BaseException:
                gps_receiver.close()
                status_receiver.close()
                raise
            print('[VE UDP] GPS 음영 유지 켜짐: IMU 방향 + 상태 속도 적분{}'.format(
                ' · 개발용 음영 주입 구간 사용' if zone else ''), flush=True)
    else:
        import rospy
        rospy.init_node('ve_udp_sensor_encoder', anonymous=True, disable_signals=True)
        status_receiver = RosEgoStatusReceiver()
    try:
        source = LiveFrameSource(source_ports, route_csv=route['csv'],
            lidar_elevations=lidar.get('elevations'), lidar_axis_policy=lidar['axis_policy'],
            bind=config['runtime']['bind'], encoder_contract=contract,
            lidar_frame=lidar['frame'], lidar_to_base=lidar['lidar_to_base'],
            velocity_source='vehicle_status' if gps_mode else 'ego_topic', route_map_root=route['map_root'],
            route_definition=route['definition'], orientation_frame='true_enu',
            perception_only=True, status_receiver=status_receiver, pose_source='gps' if gps_mode else 'ego_topic',
            diagnostics=args.audit_dir is not None,
            camera_receiver_mode=config['runtime'].get('camera_receiver_mode', 'process'), gps_receiver=gps_receiver,
            lidar_frame_policy=config['runtime'].get('lidar_frame_policy', 'morai_frame_timestamp'), localization=localization)
    except BaseException:
        status_receiver.close()
        if gps_receiver is not None:
            gps_receiver.close()
        raise
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    start, last_log, count, sequence = time.monotonic(), 0., 0, 0
    last_lidar = -1
    last_wait_logged = None
    datagram_sequence = 0
    audit = None
    if args.audit_dir is not None:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
        audit = (args.audit_dir / 'performance.jsonl').open('w', buffering=1)
    def audit_write(row):
        if audit is not None:
            audit.write(json.dumps(row, allow_nan=False) + '\n')
    print('[VE UDP] 센서 입력 대기. ROS 센서 브리지 9091은 사용하지 않습니다.', flush=True)
    control_pose = None
    try:
        if control_pose_hz is not None:
            from utils.common.ve_control_pose import ControlPoseStream
            control_pose = ControlPoseStream(status_receiver, gps_receiver, MAP_BASIS,
                args.generation, digest, hz=float(control_pose_hz), localization=localization).start()
        while not stopped[0] and (args.seconds == 0 or time.monotonic() - start < args.seconds):
            if control_pose is not None and control_pose.error is not None:
                raise RuntimeError('Control pose stream stopped: '+control_pose.error)
            begin = time.monotonic_ns()
            inputs, meta = source.build(device, utm_offsets=(route['utm_east_offset'], route['utm_north_offset']),
                                         max_age=MAX_AGE_S, after_lidar_sample_id=last_lidar)
            built_ns = time.monotonic_ns()
            if meta.get('waiting_for_new_lidar'):
                if last_wait_logged != last_lidar:
                    audit_write(dict(event='duplicate_lidar_gate', begin_ns=begin,
                        build_ms=(built_ns-begin)/1e6, selected_samples=meta['selected_samples'],
                        gate_lidar_id=last_lidar, skipped_before_preprocessing=True))
                    last_wait_logged = last_lidar
                time.sleep(.01)
                continue
            sequence += 1
            selected_pose = meta.get('localization_pose')
            packet = dict(schema=SCHEMA, generation=args.generation, sequence=sequence,
                checkpoint_sha256=digest, output_frame='enu' if coordinates else 'projected_grid', scene_assumption='unknown',
                sensor_received_ns=0, ready=False, reason='missing:' + ','.join(meta['missing']),
                boxes=[], ego_pose=None, training_updates=0,
                timing_policy=('lidar_source; causal_status_speed_50ms; GPS_receipt_freshness; IMU_alignment_in_audit; source_synchrony_unverified'
                               if gps_mode else 'morai_lidar_source_timestamp; causal_ego_50ms; latest_cameras; source_synchrony_unverified'))
            if coordinates is not None:
                packet.update(frame_id=coordinates.frame_id, reference_id=coordinates.reference_id)
            if inputs is not None:
                lidar_sequence = meta['selected_samples']['lidar']['sample_id']
                last_lidar = lidar_sequence
                required = (*REQUIRED_STREAMS, 'gps') if gps_mode else REQUIRED_STREAMS
                packet['sensor_received_ns'] = min(meta['selected_samples'][key]['received_ns']
                                                   for key in required)
                if selected_pose is not None:
                    selected = selected_pose['ego_pose']
                    x,y,yaw = selected['x'],selected['y'],selected['yaw_enu_rad']
                else:
                    x, y = meta['ego_local']
                    yaw = grid_yaw(math.radians(meta['yaw_degrees']), (x, y, 0.), MAP_BASIS, source_frame='true_enu')
                forward_ns = time.monotonic_ns()
                cuda_events = None
                if audit is not None and str(device).startswith('cuda'):
                    cuda_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    cuda_events[0].record()
                with torch.no_grad():
                    if signal_head is None:
                        detection, _light = forward_perception(model, inputs)
                        # Reuse the existing binary head/decoder. No left-arrow
                        # phase is inferred; the original signal policy handles
                        # unsupported movements as unavailable evidence.
                        packet['native_signal'] = decode_native_signal(_light)[0]
                    else:
                        token, (detection, _light) = extract_signal_feature(model, inputs, forward_perception)
                        signal_logits = signal_head(token)
                        packet['signal'] = decode_signal(signal_logits)[0]
                        packet['native_signal'] = decode_native_signal(_light)[0]
                        # Diagnostic only: preserve both heads on the same input,
                        # including the raw class before confidence abstention.
                        packet['signal_diagnostics'] = dict(
                            native_class_order=['red', 'green'],
                            native_probabilities=_light[0].softmax(0).cpu().tolist(),
                            three_code_order=[1, 16, 33],
                            three_probabilities=signal_logits[0].softmax(0).cpu().tolist())

                if cuda_events is not None:
                    cuda_events[1].record()
                decode_ns = time.monotonic_ns()
                boxes = decode_world_predictions(detection[0], dict(x=x, y=y, yaw=yaw), coordinates)['boxes']
                decoded_ns = time.monotonic_ns()
                count += 1
                packet.update(ready=True, reason=None, boxes=[{k: box[k] for k in
                    ('center_world', 'heading_deg', 'width_m', 'length_m', 'speed_mps')} for box in boxes],
                    ego_pose=dict(x=x, y=y, yaw_raw_deg=meta['yaw_degrees'], yaw_grid_rad=yaw,
                                  speed_mps=meta['speed']),
                    ego_status_source='UDP_status_speed_IMU_heading' if gps_mode else '/Ego_topic',
                    ego_position_source=('/Ego_topic' if not gps_mode else
                        'GPS_dead_reckoning' if meta['selected_samples'].get('gps', {}).get('position_mode') == 'dead_reckoning'
                        else 'GPS'),
                    ego_control=meta.get('ego_control') if gps_mode else None,
                    model_inference_ms=(time.monotonic_ns() - begin) / 1e6)
                if selected_pose is not None:
                    packet.update(ego_pose=selected_pose['ego_pose'], selected_samples=selected_pose['selected_samples'],
                                  sensor_received_ns=min(packet['sensor_received_ns'],selected_pose['sensor_received_ns']))
                if audit is not None:
                    selected = meta['selected_samples']
                    receipts = [x['received_ns'] for x in selected.values() if x['received_ns'] > 0]
                    stages = dict(meta['stage_ms'], pose_ms=(forward_ns-built_ns)/1e6,
                        forward_host_call_ms=(decode_ns-forward_ns)/1e6,
                        decode_including_gpu_wait_ms=(decoded_ns-decode_ns)/1e6,
                        forward_and_decode_ms=(decoded_ns-forward_ns)/1e6)
                    if cuda_events is not None and cuda_events[1].query():
                        stages['cuda_forward_elapsed_ms'] = cuda_events[0].elapsed_time(cuda_events[1])
                    packet['audit'] = dict(selected_samples=selected, gate_lidar_id=lidar_sequence,
                        stage_ms=stages, receipt_skew_ms=(max(receipts)-min(receipts))/1e6 if receipts else None,
                        oldest_selected_received_ns=min(receipts) if receipts else None,
                        source_spread_ms=meta.get('source_spread_ms'),
                        ego_alignment=meta.get('ego_alignment'),
                        model_velocity_mps=meta['model_velocity'],
                        lidar_points=meta['lidar_points'])
                if time.monotonic_ns() - packet['sensor_received_ns'] > MAX_AGE_S * 1e9:
                    packet.update(ready=False, reason='sensors_expired_during_inference')
            datagram_sequence += 1
            packet['datagram_sequence'] = datagram_sequence
            packet['sent_ns'] = time.monotonic_ns()
            sock.sendto(encode_packet(packet), ('127.0.0.1', PORT))
            audit_write(dict(event='sent', sent_ns=packet['sent_ns'], begin_ns=begin,
                datagram_sequence=datagram_sequence, sequence=sequence, ready=packet['ready'],
                reason=packet['reason'], model_forwards=count, audit=packet.get('audit')))
            if time.monotonic() - last_log >= 2.:
                audit_write(dict(event='streams', monotonic_ns=time.monotonic_ns(), streams=source.diagnostics()))
                print(json.dumps(dict(ready=packet['ready'], reason=packet['reason'],
                    model_forwards=count, objects=len(packet['boxes']),
                    streams=[dict(name=n, port=p, packets=k, samples=s, age_s=round(a, 3) if math.isfinite(a) else None)
                             for n, p, k, s, a in source.stream_report()]), ensure_ascii=False), flush=True)
                last_log = time.monotonic()
            time.sleep(.02 if inputs is not None else .1)
    finally:
        try:
            sequence += 1
            sock.sendto(encode_packet(dict(schema=SCHEMA, generation=args.generation, sequence=sequence,
                checkpoint_sha256=digest, output_frame='enu' if coordinates else 'projected_grid', scene_assumption='unknown',
                ready=False, reason='sender_stopped', sent_ns=time.monotonic_ns(), sensor_received_ns=0)), ('127.0.0.1', PORT))
            audit_write(dict(event='streams', monotonic_ns=time.monotonic_ns(), streams=source.diagnostics()))
            audit_write(dict(event='stopped', monotonic_ns=time.monotonic_ns(), model_forwards=count,
                             datagrams_journaled=datagram_sequence))
        finally:
            if control_pose is not None:
                control_pose.close()
            if audit is not None:
                audit.close()
            try:
                source.close()
            finally:
                sock.close()
        print(json.dumps(dict(status='STOPPED', model_forwards=count, training_updates=0)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
