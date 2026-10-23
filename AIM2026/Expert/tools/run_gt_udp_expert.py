#!/usr/bin/env python3
"""Use the existing Expert planner with MORAI GT inputs and control over UDP."""
import argparse
import collections
import datetime
import fcntl
import functools
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / 'AIM2026/Expert/src')]
from gt_udp_protocol import parse_ego, parse_objects, parse_signal, pack_control


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def running_controllers():
    tokens = ('scenario_runner.main_runner', 'run_ve_udp_expert.py', 've_objects_udp.py',
              'run_gt_udp_expert.py', 'run_gt_udp_full_route.py', 'run_ve_gps_expert.py', 'run_ve_gps_full_route.py', 'adapters/expert_process.py',
              '/expert/main.py', 've_full_route_campaign.py')
    result = []
    parent = os.getppid()
    # A campaign waits for its scenario child, which now owns a GPS driver
    # grandchild. It is not a competing controller; unrelated campaigns are.
    ancestors = set()
    pid = parent
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            fields = (Path('/proc')/str(pid)/'status').read_text().splitlines()
            pid = int(next(line.split()[1] for line in fields if line.startswith('PPid:')))
        except (OSError, ValueError, StopIteration):
            break
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) in (os.getpid(), parent):
            continue
        try:
            argv = path.read_bytes().split(b'\0')
            command = ' '.join(a.decode(errors='replace') for a in argv)
        except OSError:
            continue
        # Inspect executable/script arguments, not shell -c contents or tests.
        if not argv or b'python' not in argv[0] or b'-c' in argv or b'pytest' in argv[0]:
            continue
        if int(path.parent.name) in ancestors and any(
                Path(a.decode(errors='replace')).name == 've_full_route_campaign.py' for a in argv):
            continue
        if any(token in command for token in tokens):
            result.append(dict(pid=int(path.parent.name), command=command))
    return result


class Inputs:
    """One exclusive socket per stream, drained separately from planning."""
    def __init__(self, bind_ip, ports):
        self.lock = threading.RLock()
        self.latest = {}
        self.receipts = {}
        self.counts = collections.Counter()
        self.errors = {}
        self.selector = selectors.DefaultSelector()
        self.sockets = []
        self.stopped = threading.Event()
        try:
            for name, parser in (('ego', parse_ego), ('objects', parse_objects), ('signal', parse_signal)):
                if name not in ports:
                    continue
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
                sock.bind((bind_ip, ports[name]))
                sock.setblocking(False)
                self.selector.register(sock, selectors.EVENT_READ, (name, parser))
        except BaseException:
            self.close()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped.is_set():
            for key, _ in self.selector.select(.05):
                name, parser = key.data
                for _ in range(256):
                    try:
                        data = key.fileobj.recv(65535)
                    except BlockingIOError:
                        break
                    now = time.monotonic()
                    try:
                        value = parser(data)
                    except (ValueError, UnicodeError) as error:
                        with self.lock:
                            self.counts[name + '_rejected'] += 1
                            self.errors[name] = str(error)
                        continue
                    with self.lock:
                        previous = self.latest.get(name)
                        # Repeated simulation timestamps cannot renew liveness.
                        if 'stamp_ns' in value and previous and value['stamp_ns'] <= previous['stamp_ns']:
                            self.counts[name + '_old_or_duplicate'] += 1
                            continue
                        self.latest[name] = value
                        self.receipts[name] = now
                        self.counts[name] += 1

    def accept_signal(self, value, error=None):
        with self.lock:
            self.latest['signal'] = value
            self.receipts['signal'] = time.monotonic()
            self.counts['signal'] += 1
            if error is not None:
                self.counts['signal_rejected'] += 1
                self.errors['signal'] = error
            else:
                self.errors.pop('signal', None)

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            return dict(self.latest), {k: now-v for k, v in self.receipts.items()}, dict(self.counts), dict(self.errors)

    def close(self):
        self.stopped.set()
        thread = getattr(self, 'thread', None)
        if thread is not None:
            thread.join(timeout=1.)
        self.selector.close()
        for sock in self.sockets:
            sock.close()


def expert_inputs(data, receipts, vehicle_type, object_type):
    ego = data['ego']
    state = vehicle_type(ego['position'][0], ego['position'][1],
                         math.radians(ego['heading_deg']), ego['speed_mps'])
    objects = [object_type(
        obj['position'][0], obj['position'][1], obj['velocity_kmh'][0] / 3.6, obj['type'],
        name='gt_udp_{}'.format(obj['unique_id']), is_static=obj['type'] == 2,
        size_x=obj['size'][0], size_y=obj['size'][1],
        heading_deg=obj['heading_deg'], unique_id=obj['unique_id'])
        for obj in data['objects']['objects']]
    light = data.get('signal')
    signal_value = ([light['signal_id'], light['status'], light['type'], receipts['signal']]
                    if light else [])
    return state, objects, signal_value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--path', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bind-ip', default='0.0.0.0')
    parser.add_argument('--sim-ip', default='127.0.0.1')
    parser.add_argument('--ego-port', type=int, default=8000)
    parser.add_argument('--object-port', type=int, default=8100)
    parser.add_argument('--signal-port', type=int, default=7700)
    parser.add_argument('--signal-source', choices=('udp', 'grpc'), default='udp')
    parser.add_argument('--grpc-target', default='auto')
    parser.add_argument('--control-port', type=int, default=7601)
    parser.add_argument('--control-source-port', type=int, default=7600)
    parser.add_argument('--control-format', choices=('single_steer55', 'front_rear59'), default='single_steer55')
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--startup-timeout', type=float, default=60.)
    parser.add_argument('--max-age', type=float, default=.5)
    parser.add_argument('--probe', action='store_true', help='UDP 수신만 확인하며 제어는 송신하지 않음')
    parser.add_argument('--check', action='store_true', help='소켓·MORAI 실행 없이 설정·Expert 초기화 검사')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    ports = dict(ego=args.ego_port, objects=args.object_port)
    if args.signal_source == 'udp':
        ports['signal'] = args.signal_port
    all_ports = [*ports.values(), args.control_source_port, args.control_port]
    if len(set(all_ports)) != len(all_ports) or not all(1 <= p <= 65535 for p in all_ports):
        parser.error('UDP 포트는 서로 다른 1~65535 값이어야 합니다')
    if not all(math.isfinite(v) for v in (args.seconds, args.startup_timeout, args.max_age)) or args.seconds < 0 or args.startup_timeout <= 0 or args.max_age <= 0:
        parser.error('시간 설정이 유효하지 않습니다')
    driving = None
    cursor = None
    if not args.probe or args.signal_source == 'grpc':
        if args.config is None or args.path is None or not args.config.is_file() or not args.path.is_file():
            parser.error('--config와 --path에 이번 실행의 Expert 설정·경로가 필요합니다')
        doc = json.loads(args.config.read_text())
        if doc['common'].get('perception_source', 'ros') != 'ros' or doc['common'].get('sync_grpc_inputs', {}).get('enabled', True):
            parser.error('GT UDP 전용 설정은 perception_source=ros, sync_grpc_inputs.enabled=false여야 합니다')
        if args.signal_source == 'grpc':
            from gt_grpc_signal import make_cursor
            cursor = make_cursor(args.config, args.path)
            if not args.probe and 0 not in doc['planning']['traffic_light_stop'].get('stop_statuses', []):
                parser.error('gRPC 신호 조회 실패 처리를 위해 이번 실행 설정 stop_statuses에 0이 필요합니다')
    if not args.probe:
        os.environ.update(GT_BEV_CONFIG_PATH=str(args.config.resolve()), GT_BEV_PATH_CSV=str(args.path.resolve()),
                          GT_BEV_MAP_DIR=doc['map']['data_dir'])
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.vehicle_state import VehicleState
        from expert.src.obstacle.object_info import ObjectInfo
        driving = AutonomousDriving()
    if args.check:
        print(json.dumps(dict(status='CHECKED', transport='UDP', signal_source=args.signal_source,
                              planner='existing_AutonomousDriving',
                              ports=ports, control_target=[args.sim_ip, args.control_port],
                              control_format=args.control_format, VE_used=False), ensure_ascii=False))
        return 0
    conflicts = running_controllers()
    if conflicts:
        raise RuntimeError('기존 주행 실행을 종료하세요: ' + json.dumps(conflicts, ensure_ascii=False))
    output = args.output or REPO / 'MoraiDrive/.runtime/gt_udp_expert' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    lock_path = REPO / 'MoraiDrive/.runtime/gt_udp_expert/launcher.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lockfile = lock_path.open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = Inputs(args.bind_ip, ports)
    signal_source = None
    encode_control = functools.partial(pack_control, layout=args.control_format)
    sender = None
    stopped = threading.Event()
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, lambda *_: stopped.set())
    commands = 0
    status, error = 'STOPPED', None
    started = last_log = last_control = time.monotonic()
    ready_since = None
    missing_since = None
    latest_control = None
    period = 1. / (driving.realtime_sampling_rate if driving else 30.)
    log = (output / 'inputs.jsonl').open('a', buffering=1)
    try:
        if args.signal_source == 'grpc':
            from gt_grpc_signal import SignalReader, SignalSource
            signal_source = SignalSource(inputs, cursor, SignalReader(args.grpc_target), preview=args.probe)
        if not args.probe:
            control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                control_socket.bind((args.bind_ip, args.control_source_port))
            except BaseException:
                control_socket.close()
                raise
            sender = control_socket
        print('[GT] 자차 UDP:{} · 장애물 UDP:{} · 신호={} → 기존 Expert → UDP {}:{}'.format(
            args.ego_port, args.object_port, args.signal_source, args.sim_ip, args.control_port), flush=True)
        print('[로그] ' + str(output), flush=True)
        while not stopped.is_set():
            now = time.monotonic()
            data, ages, counts, errors = inputs.snapshot()
            ready = all(k in data and ages[k] <= (2. if k == 'signal' else args.max_age)
                        for k in ('ego', 'objects', 'signal'))
            ready = ready and data.get('signal', {}).get('valid', True)
            # Once initialized, the existing signal controller owns per-signal
            # expiry. A UDP signal is not necessarily broadcast outside junctions.
            live = all(k in data and ages[k] <= args.max_age for k in ('ego', 'objects'))
            if ready_since is not None and not live:
                missing_since = now if missing_since is None else missing_since
                if now-missing_since > 2.:
                    raise RuntimeError('UDP 자차 또는 장애물 입력이 2초 이상 끊겼습니다')
            else:
                missing_since = None
            if ready_since is None and ready:
                ready_since = now
                print('[PASS] 자차·장애물 UDP + 신호 {} GT 확인'.format(args.signal_source)
                      + ('; 수신 검사 완료' if args.probe else '; 기존 Expert 주행 시작'), flush=True)
                if args.probe:
                    status = 'COMPLETE'
                    break
            if ready_since is None and now - started > args.startup_timeout:
                raise RuntimeError('UDP 입력 준비 시간 초과: ' + json.dumps(dict(ages=ages, errors=errors), ensure_ascii=False))
            if ready_since is not None and args.seconds > 0 and now-ready_since >= args.seconds:
                status = 'COMPLETE'
                break
            if sender is not None:
                if live and ready_since is not None:
                    receipts = {k: now-v for k, v in ages.items()}
                    vehicle, objects, traffic = expert_inputs(data, receipts, VehicleState, ObjectInfo)
                    command, _ = driving.execute(vehicle, objects, traffic, sync_mode=False,
                        control_dt=period if commands == 0 else max(.001, now-last_control),
                        ego_header_seq=counts['ego'], ego_source_ns=data['ego']['stamp_ns'])
                    payload = encode_control(command.accel, command.brake, command.front_steer, command.rear_steer)
                    latest_control = dict(accel=float(command.accel), brake=float(command.brake),
                                          steering=float(command.front_steer))
                else:
                    payload = encode_control(0., 1.)
                    latest_control = dict(accel=0., brake=1., steering=0.)
                sender.sendto(payload, (args.sim_ip, args.control_port))
                commands += 1
                last_control = now
            if now-last_log >= 2.:
                row = dict(status='DRIVING' if ready_since is not None and live else 'WAITING_FOR_UDP',
                    elapsed_s=round(now-started, 1), counts=counts, ages_s=ages, errors=errors,
                    objects=len(data['objects']['objects']) if 'objects' in data else None,
                    speed_kmh=data['ego']['speed_mps']*3.6 if 'ego' in data else None,
                    ego_control={k:data['ego'][k] for k in ('accel','brake','steering','gear','ctrl_mode')} if 'ego' in data else None,
                    expert_control=latest_control,
                    signal=data.get('signal'), signal_source=args.signal_source,
                    control_format=args.control_format, commands=commands, VE_used=False)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                write(output / 'status.json', row)
                log.write(json.dumps(row, ensure_ascii=False)+'\n')
                last_log = now
            stopped.wait(max(.001, period-(time.monotonic()-now)))
    except Exception as exc:
        status, error = 'FAILED', str(exc)
        print('[FAIL] ' + error, flush=True)
    finally:
        if sender is not None:
            try:
                for _ in range(10):
                    sender.sendto(encode_control(0., 1.), (args.sim_ip, args.control_port))
                    time.sleep(.03)
            except OSError as stop_error:
                status, error = 'FAILED', '종료 제동 송신 실패: ' + str(stop_error)
            finally:
                sender.close()
        data, ages, counts, errors = inputs.snapshot()
        if signal_source is not None:
            signal_source.close()
        inputs.close()
        log.close()
        lockfile.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        result = dict(status=status, error=error, counts=counts, rejected=errors, control_packets=commands,
                      transport='UDP', signal_source=args.signal_source,
                      control_format=args.control_format,
                      all_external_transport='UDP' if args.signal_source == 'udp' else 'UDP_and_gRPC_signal_GT',
                      VE_used=False, Expert_core_modified=False,
                      elapsed_s=time.monotonic()-started)
        write(output / 'report.json', result)
        write(output / 'status.json', result)
        print(json.dumps(dict(status=status, report=str(output / 'report.json')), ensure_ascii=False), flush=True)
    if args.probe and status == 'STOPPED':
        return 130  # Cancelling preflight must never continue into driving.
    return int(status == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
