"""AIM2026 owns inference, UDP and the adapter to the unchanged Expert core."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'MoraiDrive', ROOT/'VLDrive'):
    sys.path.insert(0, str(path))

from AIM2026.UDP.Detection import decode, NotReady
from AIM2026.UDP.Pose import decode_pose, SCHEMA as POSE_SCHEMA
from AIM2026.UDP.Cmd import CmdSender, Command
from AIM2026.Expert.adapter import prepare, planner_inputs
from AIM2026.Inference.timer_resolution import high_resolution_timer

# VE not-ready reasons that must stop using the last detection at once.
HARD_NOT_READY = ('publisher_stopped', 'position_unavailable')


def wire_command(command, max_steering, hold_steer=0.):
    """Expert command -> MORAI wire values. ``None`` brakes fully.

    While braking for missing inputs the wheel stays at ``hold_steer`` (the last
    sent value) instead of snapping to centre, which jerked the vehicle sideways.
    """
    if command is None:
        hold = float(hold_steer) if math.isfinite(hold_steer) else 0.
        return Command(brake=1., steer=max(-1., min(1., hold)))
    if command.rear_steer != 0 or not math.isfinite(max_steering) or max_steering <= 0:
        raise ValueError('Invalid steering contract')
    steer = command.front_steer / max_steering
    if not math.isfinite(steer):
        raise ValueError('Non-finite steering')
    return Command(accel=command.accel, brake=command.brake, steer=max(-1., min(1., steer)))


class Inputs:
    """Pin one publisher and reject stale/reordered frames before updating state."""
    def __init__(self, generation, digest, tracker):
        self.generation, self.digest, self.tracker = generation, digest, tracker
        self.packet = self.pose = None
        self.sequences = {False: 0, True: 0}
        self.frames = self.pose_frames = self.held_frames = 0
        self.reason = 'waiting_for_inference'

    def accept(self, data):
        row = json.loads(data)
        if not isinstance(row, dict):
            raise ValueError('Expected an object packet')
        is_pose = row.get('schema') == POSE_SCHEMA
        # An unrelated publisher must not overwrite or invalidate our state.
        if row.get('generation') != self.generation or row.get('checkpoint_sha256') != self.digest:
            return None
        seq = row.get('sequence')
        if type(seq) is not int or seq <= self.sequences[is_pose]:
            return None
        try:
            (decode_pose if is_pose else decode)(data, generation=self.generation,
                                                 checkpoint_sha256=self.digest)
        except NotReady as exc:
            self.sequences[is_pose] = seq
            self.reason = str(exc)
            if is_pose or exc.reason in HARD_NOT_READY:
                if is_pose:
                    self.pose = None
                else:
                    self.packet = None
                return 'invalid_pose' if is_pose else 'invalid_detection'
            # One skipped VE frame (e.g. no IMU sample in its window) keeps the
            # previous detection; ready() still drops it once its sensors are 0.5 s old.
            self.held_frames += 1
            return 'held_detection'
        except (ValueError, KeyError, TypeError) as exc:
            if is_pose:
                self.pose = None
            else:
                self.packet = None
            self.reason = str(exc)
            return 'invalid_pose' if is_pose else 'invalid_detection'
        self.sequences[is_pose] = seq
        if is_pose:
            self.pose = row
            self.pose_frames += 1
            return 'pose'
        self.packet = self.tracker.update(row)
        self.frames += 1
        return 'detection'

    def ready(self, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        return all(p is not None and 0 <= now-p['sensor_received_ns'] <= age
                   for p, age in ((self.packet, 500_000_000), (self.pose, 150_000_000)))


def publisher_config(config, output, generation):
    """Isolate inference ownership and validate the unchanged shared DR provider."""
    child_cfg = json.loads(json.dumps(config))
    child_cfg['Ego']['cmd_port'] = None  # Only the parent can own vehicle control.
    child_cfg['Ego']['cmd_source_port'] = None
    if (child_cfg.get('Localization') or {}).get('enabled', False):
        child_cfg['Localization']['record_dir'] = str(output/'localization')
        from AIM2026.GPSJamming.gps_jamming.localization.service import create_localization
        from types import SimpleNamespace
        validation_cfg = json.loads(json.dumps(child_cfg))
        validation_cfg['Localization']['record_dir'] = None
        validation = create_localization(SimpleNamespace(receivers={name: object() for name in ('gps', 'imu', 'vehicle_status')}), validation_cfg)
        validation.close()
    child_cfg['Detection']['generation'] = generation
    child_cfg['Inference']['status_file'] = str(output/'inference_status.json')
    return child_cfg


def main(argv=None, control_tick=None):
    import yaml
    from AIM2026.Inference.settings import read_config
    from AIM2026.Inference.vision.checkpoint import resolve, digest
    from AIM2026.UDP import UDPManager
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'AIM2026/config_local.yaml')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--drive', action='store_true', help='Send vehicle commands; default is observation only')
    parser.add_argument('--visualize', action='store_true', help='Write read-only RViz input logs')
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--speed-cap-kmh', type=float, default=60.)
    parser.add_argument('--sim-profile', choices=('normal','competition'), default=None,
                        help='Override the configured simulator steering direction')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('seconds must be finite and nonnegative')
    cfg = read_config(args.config)
    positive = cfg.get('Control', {}).get('positive_steer')
    if positive not in ('left', 'right'):
        parser.error('Control.positive_steer must be left or right')
    args.sim_profile = args.sim_profile or ('normal' if positive == 'right' else 'competition')
    UDPManager(cfg)._validate_ports()
    if args.drive and not cfg['Ego'].get('cmd_port'):
        parser.error('Set Ego.ip, cmd_port and cmd_source_port before driving')
    label = cfg['Inference']['vision']['checkpoint']
    weights, sha, head = resolve(label)
    if not weights.is_file() or digest(weights) != sha:
        raise ValueError('Missing or mismatched checkpoint: '+str(weights))
    if cfg['Inference']['vision']['signal_head'] and (head is None or not head.is_file()):
        raise ValueError('Missing signal head')
    output = (args.output or ROOT/'AIM2026/outputs/expert'/time.strftime('%Y%m%d-%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    context, controller, policy = prepare(args, output)
    generation = str(uuid.uuid4())
    child_cfg = publisher_config(cfg, output, generation)
    config_path = output/'inference.yaml'
    config_path.write_text(yaml.safe_dump(child_cfg, sort_keys=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), model=label, weights=str(weights),
                         sim_profile=args.sim_profile,
                         steering_sign=-1 if args.sim_profile == 'normal' else 1,
                         command_target=[cfg['Ego']['ip'],cfg['Ego']['cmd_port']],
                         mode='drive' if args.drive else 'shadow'), ensure_ascii=False), flush=True)
    if args.check:
        print('Configuration, model SHA256 and Expert route checked; no sockets opened.')
        return 0
    # Fail before launching/loading a model or sending control when an older
    # inference process still owns the sensor ports.
    UDPManager(child_cfg)._probe_ports()
    from inference.ve_object_tracking import VeObjectTracker
    from AIM2026.Expert.src.autonomous_driving import AutonomousDriving
    inputs = Inputs(generation, sha, VeObjectTracker(static_geometry_fusion=False))
    stop = []
    old_handlers = {sig: signal.signal(sig, lambda *_: stop.append(True))
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    process = sender = receiver = None
    logs = []
    try:
        expert_log = (output/'expert.log').open('w', encoding='utf-8', buffering=1); logs.append(expert_log)
        with contextlib.redirect_stdout(expert_log):
            driving = AutonomousDriving()
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            receiver.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        receiver.bind((cfg['Detection']['ip'], cfg['Detection']['port']))
        receiver.setblocking(False)
        if args.drive:
            sender = CmdSender(cfg['Ego']['ip'], cfg['Ego']['cmd_port'], source_port=cfg['Ego'].get('cmd_source_port'),
                               layout=cfg['Ego'].get('cmd_layout'))
            sender.send(Command(brake=1.))
        publisher_log = (output/'inference.log').open('w', encoding='utf-8', buffering=1); logs.append(publisher_log)
        control_log = (output/'control.jsonl').open('w', encoding='utf-8', buffering=1); logs.append(control_log)
        path_log = input_log = None
        if args.visualize:
            path_log = (output/'paths.jsonl').open('w', encoding='utf-8', buffering=1)
            input_log = (output/'inputs.jsonl').open('w', encoding='utf-8', buffering=1)
            logs.extend((path_log,input_log))
        child_options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        process = subprocess.Popen([sys.executable, '-u', '-B', str(ROOT/'AIM2026/Inference/main.py'), '--config', str(config_path)],
                                   cwd=ROOT, stdout=publisher_log, stderr=subprocess.STDOUT, **child_options)
        started = next_status = time.monotonic()
        next_snapshot = 0.
        last_steer = 0.
        # Loop pacing and the PID's control_dt use perf_counter: on Windows/Python 3.10
        # time.monotonic() only advances in 15.6 ms steps, which made dt jump between
        # 0, 16, 31 and 47 ms. Packet ages keep the shared monotonic_ns contract.
        timer = high_resolution_timer()
        timer_active = timer.__enter__()
        print('[Expert] 1 ms timer resolution: %s' % timer_active, flush=True)
        period = 1. / driving.realtime_sampling_rate
        last_tick = next_tick = time.perf_counter()
        while not stop and not (args.seconds and time.monotonic()-started >= args.seconds):
            if process.poll() is not None:
                raise RuntimeError('Inference exited; see '+str(output/'inference.log'))
            for _ in range(128):
                try:
                    data, _ = receiver.recvfrom(65536)
                except BlockingIOError:
                    break
                try:
                    result = inputs.accept(data)
                except (ValueError, KeyError, TypeError):
                    continue
                if result == 'invalid_detection':
                    policy.invalidate()
                if result == 'detection':
                    if input_log:
                        input_log.write(json.dumps(inputs.packet, allow_nan=False)+'\n')
                    if inputs.frames == 1:
                        p = inputs.packet['ego_pose']
                        policy.last_index = min(range(len(context.route_points)), key=lambda i:
                            (context.route_points[i][0]-p['x'])**2+(context.route_points[i][1]-p['y'])**2)
                    policy.output(time.monotonic())
                    policy.accept(inputs.packet, context.route_points, time.monotonic())
            now = time.monotonic()
            tick_time = time.perf_counter()
            control_dt = max(.001, tick_time - last_tick)
            fresh = inputs.ready()
            command = None
            code = policy.output(now)
            if fresh:
                if policy.reason == 'off_route':
                    raise RuntimeError('GPS pose outside route corridor')
                vehicle, objects = planner_inputs(inputs.packet, inputs.pose, position_sources=('GPS', 'GPS_dead_reckoning')
                    if (cfg.get('Localization') or {}).get('enabled', False) else ('GPS',))
                traffic = [policy.target_id, code, 0, now]
                with contextlib.redirect_stdout(expert_log):
                    tick = control_tick or (lambda d,*a,**kw: d.execute(*a,**kw))
                    command, path = tick(driving, vehicle, objects, traffic, sync_mode=False,
                        control_dt=control_dt, ego_header_seq=inputs.sequences[True],
                        ego_source_ns=inputs.pose['selected_samples']['vehicle_status'].get('source_stamp_ns'))
                if path_log:
                    points = [[float(p.x),float(p.y)] if hasattr(p,'x') else [float(p[0]),float(p[1])]
                              for p in list(path or [])[:80]]
                    path_log.write(json.dumps(dict(ego=[vehicle.position.x,vehicle.position.y,vehicle.yaw],
                                                   path=points), allow_nan=False)+'\n')
                    if now >= next_snapshot:
                        from AIM2026.Visualization.snapshot import write_snapshot
                        try:
                            write_snapshot(output/'visualization.json',
                                driving.lattice_planning_process.last_result, vehicle, objects, path)
                        except (OSError, TypeError, ValueError, AttributeError) as exc:
                            print('[RViz snapshot] '+str(exc), file=expert_log)
                        next_snapshot = now+.2
            encoded = wire_command(command, driving.max_steering, hold_steer=last_steer)
            last_steer = encoded.steer
            if sender:
                sender.send(encoded)
            row = dict(status='RUNNING' if fresh else 'WAITING', VE_frames=inputs.frames,
                       VE_held=inputs.held_frames,
                       control_pose_frames=inputs.pose_frames, reason=None if fresh else inputs.reason,
                       accel=encoded.accel, brake=encoded.brake, steer=encoded.steer, control_sent=bool(sender))
            # Record the exact pose used by this control tick for timing/jitter
            # diagnosis. This does not alter inputs, planning, or commands.
            control_log.write(json.dumps(dict(row, monotonic_ns=time.monotonic_ns(),
                control_dt_s=control_dt,
                # Signal decision of this tick (VE phase after policy; 0 = unknown -> stop).
                signal=dict(id=policy.target_id, code=code, head=getattr(policy, 'head_source', None),
                            reason=getattr(policy, 'reason', None), green_samples=policy.allow_samples,
                            phase_age_ms=round((now - policy.received) * 1e3, 1) if policy.received else None),
                control_pose=inputs.pose))+'\n')
            if now >= next_status:
                print(json.dumps(row), flush=True)
                next_status = now+2.
            last_tick = tick_time
            # Fixed schedule: a late tick does not shift every following tick.
            next_tick = max(next_tick + period, time.perf_counter())
            time.sleep(max(0., next_tick - time.perf_counter()))
        return 0
    finally:
        if 'timer' in locals():
            timer.__exit__(None, None, None)
        if sender:
            try:
                sender.send(Command(brake=1.))
            except OSError:
                pass
            finally:
                sender.close()
        if process and process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if receiver:
            receiver.close()
        for stream in logs:
            stream.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
