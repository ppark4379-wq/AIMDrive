#!/usr/bin/env python3
"""UDP footprints replace only the original Expert's Object_topic input."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import xmlrpc.client

REPO = Path(__file__).resolve().parents[3]
INTERNAL = REPO / 'MoraiDrive/data_collection/internal_sensor_record'
sys.path[:0] = [str(INTERNAL), str(REPO / 'VLDrive')]
from python_environment import build_analysis_environment, build_capture_environment
from runtime_paths import ANALYSIS_PYTHON
from inference.ve_objects_udp import (PORT, MAX_AGE_S, MAX_PACKET_BYTES, decode_packet,
                                      make_object_message)
from inference.realtime_ve_objects import CHECKPOINTS
from inference.ve_object_tracking import VeObjectTracker

RUNS = REPO / 'MoraiDrive/.runtime/data_collection/ve_udp_original_expert'
CAPTURE_PARAMS = ('/morai_drive/sync_runtime_active', '/morai_drive/data_collector_ready')
EXPERT_NODE = '/ve_udp_original_expert'


class MasterCheckTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 2.
        return connection


def require_ros_master():
    uri = os.environ.get('ROS_MASTER_URI', 'http://127.0.0.1:11311')
    try:
        with xmlrpc.client.ServerProxy(uri, transport=MasterCheckTransport()) as master:
            code, message, _pid = master.getPid('/ve_udp_startup_check')
        if code != 1:
            raise RuntimeError(message)
    except (OSError, ValueError, RuntimeError, xmlrpc.client.Error) as exc:
        raise RuntimeError(
            'ROS master에 연결할 수 없습니다: ' + uri + '\n'
            '별도 터미널에서 다음 명령을 실행하고 열린 상태로 두세요:\n'
            '  cd ~/AIMDrive\n'
            '  bash MoraiDrive/data_collection/tools/01_internal_sensor_record_preflight.sh --scenario-only\n'
            'MORAI 차량 ROS 연결 후 PASS가 나오면 이 명령을 다시 실행하세요.'
        ) from exc


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare_expert(folder):
    """Use the standalone Expert's config and path; replace only perception input."""
    doc = json.loads((REPO / 'MoraiDrive/expert/src/expert/src/config/config.json').read_text())
    doc['common']['perception_source'] = 'ros'
    doc['common']['sync_grpc_inputs']['enabled'] = False
    config_path = folder / 'expert_config.json'
    route_path = REPO / 'MoraiDrive/expert/.runtime/path.csv'
    if not route_path.is_file():
        route_path = REPO / 'MoraiDrive/expert/.runtime/scenario_runner/path.csv'
    if not route_path.is_file():
        raise FileNotFoundError('기존 Expert의 path.csv가 없습니다: ' + str(route_path))
    write(config_path, doc)
    return config_path, route_path


def expert_command(drive=False):
    command = ['/usr/bin/python3', '-u', str(REPO / 'MoraiDrive/expert/main.py'),
               '/Object_topic:=/ve/Object_topic', '__name:=' + EXPERT_NODE.lstrip('/')]
    if not drive:
        command.append('/ctrl_cmd:=/ve/shadow_ctrl_cmd')
    return command


def input_status(packet, ego, now_ns):
    """Check source liveness, without a trial region or a control-command relay."""
    if packet is None or not 0 <= now_ns - packet['sensor_received_ns'] <= MAX_AGE_S * 1e9:
        return False, 'VE_input_stale'
    if ego is None or not 0 <= now_ns - ego['received_ns'] <= MAX_AGE_S * 1e9:
        return False, 'ROS_ego_stale'
    return True, 'ready'


def make_brake_message(message_type):
    return message_type(longlCmdType=1, accel=0., brake=1.,
                        front_steer=0., rear_steer=0.)


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 3.), (signal.SIGTERM, 2.), (signal.SIGKILL, 2.)):
        try:
            os.killpg(child.pid, sig)
            child.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


def capture_guard():
    tokens = ('collect_scenario_laps.sh', 'run_sensor_record_scenario.py', 'scenario_runner.main_runner')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if any(token in command for token in tokens):
            raise RuntimeError('수집/시나리오 러너가 실행 중입니다. 종료한 뒤 실행하세요.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive', action='store_true', help='기존 Expert가 /ctrl_cmd로 직접 제어')
    parser.add_argument('--check', action='store_true', help='메타데이터만 확인; 센서/ROS/모델 실행 없음')
    parser.add_argument('--seconds', type=float, default=180.,
                        help='Expert 실행 시간(초), 0이면 Ctrl+C까지 계속 실행 (기본: 180)')
    parser.add_argument('--checkpoint-label', choices=('E3', 'E4'), default='E3')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 <= args.seconds <= 1800:
        parser.error('seconds must be in [0,1800]; 0 means run until interrupted')
    selected = CHECKPOINTS[args.checkpoint_label]
    if not Path(selected['path']).is_file():
        raise FileNotFoundError(selected['path'])
    plan = dict(checkpoint=selected, sensor_input='MORAI_UDP_camera_lidar',
        ego_input='/Ego_topic_shared_with_original_Expert',
        ego_position_source='/Ego_topic',
        VE_output_udp='127.0.0.1:' + str(PORT), object_remap='/Object_topic:=/ve/Object_topic',
        planner='original LatticePlanningProcess', signal_input='/GetTrafficLightStatus',
        scene_assumption='unknown', predicted_class=False, predicted_scalar_speed=True,
        tracking_policy='world_position_receipt_time_v1', motion_state_estimated=True,
        predicted_velocity_vector=False, source_synchrony_verified=False,
        VE_waypoints_used=False, GT_obstacles_used=False, expert_core_modified=True,
        expert_change='accept tagged VE unknowns in existing object policies; original path generation',
        seconds=args.seconds, run_until_interrupted=args.seconds == 0,
        speed_profile='original_expert_config', trial_region=None,
        control_relay=False, control_output='/ctrl_cmd' if args.drive else '/ve/shadow_ctrl_cmd',
        mode='drive' if args.drive else 'shadow')
    if args.check:
        print(json.dumps(dict(status='METADATA_CHECKED', **plan), ensure_ascii=False, indent=2))
        return 0
    # Bootstrap only the original system-Python ROS dependencies.
    if os.environ.get('VE_UDP_ENV_READY') != '1':
        env = build_capture_environment(VE_UDP_ENV_READY='1')
        os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', str(Path(__file__)), *sys.argv[1:]], env)
    import rospy
    from morai_msgs.msg import CtrlCmd, EgoVehicleStatus, ObjectStatusList, ObjectStatus
    brake = make_brake_message(CtrlCmd)
    capture_guard()
    try:
        require_ros_master()
    except RuntimeError as exc:
        print('[FAIL] ' + str(exc), file=sys.stderr, flush=True)
        return 2
    rospy.init_node('ve_udp_object_adapter', anonymous=True, disable_signals=True)
    code, error, topology = rospy.get_master().getSystemState()
    if code != 1:
        raise RuntimeError(error)
    pubs, subs, _ = map(dict, topology)
    for topic in ('/ctrl_cmd', '/ve/Object_topic', '/ve/expert_ctrl_cmd', '/ve/shadow_ctrl_cmd'):
        if pubs.get(topic):
            raise RuntimeError('기존 실행 노드가 있습니다: ' + topic + ' ' + str(pubs[topic]))
    if any(rospy.get_param(key, False) for key in CAPTURE_PARAMS):
        raise RuntimeError('수집/Sync Runtime을 종료한 뒤 실행하세요')
    if args.drive and not subs.get('/ctrl_cmd'):
        raise RuntimeError('기존 MORAI 차량 제어 ROS 연결(/ctrl_cmd)이 없습니다')
    # Binding is exclusive. No additional ROS sensor bridge is opened.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', PORT))
    sock.setblocking(False)
    RUNS.mkdir(parents=True, exist_ok=True)
    lockfile = (RUNS / 'launcher.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(os.getpid())
    folder = RUNS / name
    folder.mkdir()
    write(folder / 'plan.json', plan)
    config = json.loads((REPO / 'VLDrive/configs/vision_encoder_realtime_best_20261003.json').read_text())
    config.update(checkpoint=selected['path'], checkpoint_sha256=selected['sha256'])
    write(folder / 'publisher_config.json', config)
    expert_config, path_csv = prepare_expert(folder)
    output_topic = '/ctrl_cmd' if args.drive else '/ve/shadow_ctrl_cmd'
    objects_pub = rospy.Publisher('/ve/Object_topic', ObjectStatusList, queue_size=1)
    # During driving only the original Expert publishes control. A brake
    # publisher is created after its process has stopped, during cleanup only.
    state_lock, state = threading.Lock(), {'ego': None, 'command': None, 'command_ns': None, 'commands': 0}

    def on_ego(message):
        with state_lock:
            state['ego'] = dict(x=float(message.position.x), y=float(message.position.y),
                heading=float(message.heading), speed_mps=float(message.velocity.x),
                accel=float(message.accel), brake=float(message.brake), received_ns=time.monotonic_ns())

    def on_command(message):
        with state_lock:
            state.update(command=message, command_ns=time.monotonic_ns(), commands=state['commands'] + 1)

    subscribers = [rospy.Subscriber('/Ego_topic', EgoVehicleStatus, on_ego, queue_size=1),
                   rospy.Subscriber(output_topic, CtrlCmd, on_command, queue_size=1)]
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    sender = expert = None
    packet, received, rejected = None, 0, 0
    invalid_since = None
    sequence, valid_since, expert_started, last_report, last_topology = 0, None, None, 0., 0.
    status, error, reason = 'STOPPED', None, 'waiting_for_UDP'
    generation = str(uuid.uuid4())
    tracker = VeObjectTracker()
    started = time.monotonic()
    print('[연결] UDP 카메라·LiDAR → VE → UDP 장애물 → 기존 Expert. 자차·신호등은 기존 ROS 입력 사용.', flush=True)
    print('[추적] 객체별 ID·이동 상태 추정. 분류 미상 객체도 기존 Expert의 감속·충돌 검사에 포함합니다.', flush=True)
    print('[제어] 기존 Expert → ' + output_topic + ' 직접 전송. 위치 제한·추가 속도 제한 없음.', flush=True)
    print('[실행] ' + ('시간 제한 없이 실행합니다. 종료: Ctrl+C.' if args.seconds == 0
                       else 'Expert 시작 후 {:g}초 실행합니다. 조기 종료: Ctrl+C.'.format(args.seconds)), flush=True)
    print('[로그] ' + str(folder), flush=True)
    sender_log = (folder / 'publisher.log').open('w', buffering=1)
    expert_log = (folder / 'expert.log').open('w', buffering=1)
    trace = (folder / 'adapter.jsonl').open('w', buffering=1)
    try:
        sender = subprocess.Popen([str(ANALYSIS_PYTHON), '-u', str(REPO / 'VLDrive/inference/ve_objects_udp.py'),
            '--config', str(folder / 'publisher_config.json'), '--generation', generation,
            '--seconds', str(0 if args.seconds == 0 else args.seconds + 150)],
            env=build_analysis_environment(ANALYSIS_PYTHON), stdout=sender_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        while not stopped[0] and not rospy.is_shutdown():
            now = time.monotonic()
            if args.seconds > 0 and expert_started is not None and now - expert_started >= args.seconds:
                status = 'COMPLETE'
                break
            if sender.poll() is not None:
                raise RuntimeError('VE 프로세스가 종료되었습니다: publisher.log 확인')
            if expert is not None and expert.poll() is not None:
                raise RuntimeError('Expert 프로세스가 종료되었습니다: expert.log 확인')
            for _ in range(128):
                try:
                    data, _address = sock.recvfrom(MAX_PACKET_BYTES + 1)
                except BlockingIOError:
                    break
                try:
                    candidate = decode_packet(data, generation, selected['sha256'], time.monotonic_ns())
                    if candidate['sequence'] <= sequence:
                        raise ValueError('out_of_order_packet')
                    sequence = candidate['sequence']
                    packet = tracker.update(candidate)
                    received += 1
                    valid_since = now if valid_since is None else valid_since
                    trace.write(json.dumps(packet, allow_nan=False) + '\n')
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    packet, valid_since = None, None
                    reason = str(exc)
                    rejected += 1
            with state_lock:
                current = dict(state)
            snapshot_ok, snapshot_reason = input_status(packet, current['ego'], time.monotonic_ns())
            if not snapshot_ok:
                valid_since = None
                if packet is not None:
                    reason = snapshot_reason
            if snapshot_ok:
                invalid_since = None
                # Re-publish the current still-fresh snapshot for the asynchronous
                # original Expert; publication time is not a camera timestamp.
                objects_pub.publish(make_object_message(packet, ObjectStatusList, ObjectStatus, rospy.Time.now()))
            elif expert is not None:
                invalid_since = now if invalid_since is None else invalid_since
                if now - invalid_since > 2.:
                    raise RuntimeError('VE/자차 입력이 2초 이상 끊겨 실행을 종료합니다: ' + snapshot_reason)
            if expert is None and snapshot_ok and valid_since is not None and now - valid_since >= .25:
                env = build_capture_environment(GT_BEV_DIRECT_USE_ENV_CONFIG='1',
                    GT_BEV_CONFIG_PATH=expert_config, GT_BEV_PATH_CSV=path_csv,
                    GT_BEV_DEBUG_OUTPUT_DIR=folder / 'expert_debug')
                env.pop('GT_BEV_MAP_DIR', None)
                expert = subprocess.Popen(expert_command(args.drive), cwd=REPO / 'MoraiDrive/expert', env=env,
                    stdout=expert_log, stderr=subprocess.STDOUT, start_new_session=True)
                expert_started = now
                print('[PASS] VE UDP·자차 수신 확인. 기존 Expert 시작 (' + plan['mode'] + ').', flush=True)
            if expert is None and now - started > 120.:
                raise RuntimeError('시작 입력 미준비: ' + reason + '. publisher.log의 streams를 확인하세요')
            if now - last_topology > 1.:
                code, why, current_topology = rospy.get_master().getSystemState()
                if code != 1:
                    raise RuntimeError(why)
                current_pubs, current_subs = dict(current_topology[0]), dict(current_topology[1])
                if any(rospy.get_param(key, False) for key in CAPTURE_PARAMS):
                    raise RuntimeError('Sync Runtime/수집이 시작되었습니다')
                for topic, allowed in ((output_topic, [EXPERT_NODE] if expert is not None else []),
                                       ('/ve/Object_topic', [rospy.get_name()])):
                    if any(node not in allowed for node in current_pubs.get(topic, [])):
                        raise RuntimeError('중복 출력 노드: ' + topic)
                if not args.drive and current_pubs.get('/ctrl_cmd'):
                    raise RuntimeError('다른 실제 차량 제어 노드가 시작되었습니다')
                # Original Expert must be remapped, never subscribe to GT.
                expert_nodes = current_subs.get('/ve/Object_topic', [])
                if any(node in current_subs.get('/Object_topic', []) for node in expert_nodes):
                    raise RuntimeError('Expert에 GT 장애물 구독이 남아 있습니다')
                last_topology = now
            if now - last_report >= 2.:
                command = current['command']
                row = dict(elapsed_s=round(now - started, 1), received=received,
                    objects=len(packet['boxes']) if packet else None, rejected=rejected,
                    motion_counts={state_name: sum(b['motion_state'] == state_name for b in packet['boxes'])
                                   for state_name in ('moving', 'stationary', 'uncertain')} if packet else None,
                    expert_commands=current['commands'], control_output=output_topic,
                    expert_accel=float(command.accel) if command else None,
                    expert_brake=float(command.brake) if command else None,
                    ego_speed_mps=current['ego']['speed_mps'] if current['ego'] else None,
                    ego_accel=current['ego']['accel'] if current['ego'] else None,
                    ego_brake=current['ego']['brake'] if current['ego'] else None,
                    reason=snapshot_reason if packet else reason, mode=plan['mode'])
                print(json.dumps(row, ensure_ascii=False), flush=True)
                trace.write(json.dumps(row) + '\n')
                last_report = now
            time.sleep(.025)
    except Exception as exc:
        status, error = 'FAILED', str(exc)
        print('[FAIL] ' + error, flush=True)
    finally:
        stop_child(expert)
        if expert is not None:
            control_pub = rospy.Publisher(output_topic, CtrlCmd, queue_size=1)
            for _ in range(10):
                control_pub.publish(brake)
                time.sleep(.05)
            control_pub.unregister()
        stop_child(sender)
        for subscriber in subscribers:
            subscriber.unregister()
        sock.close()
        sender_log.close(); expert_log.close(); trace.close(); lockfile.close()
        write(folder / 'report.json', dict(status=status, error=error, plan=plan,
            received_UDP_frames=received, rejected_packets=rejected, expert_commands=state['commands'],
            actual_drive_enabled=args.drive, training_updates=0,
            elapsed_s=time.monotonic() - started, performance_measured=False))
        print(json.dumps(dict(status=status, report=str(folder / 'report.json'))), flush=True)
    return 1 if status == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
