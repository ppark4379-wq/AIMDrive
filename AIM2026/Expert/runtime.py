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
for path in (ROOT, ROOT/'AIM2026/utils', ROOT/'VLDrive'):
    sys.path.insert(0, str(path))

from AIM2026.UDP.Detection import decode, NotReady
from AIM2026.UDP.Pose import decode_pose, SCHEMA as POSE_SCHEMA
from AIM2026.UDP.Cmd import CmdSender, Command
from AIM2026.Expert.adapter import prepare, planner_inputs
from AIM2026.Expert.static_memory import StaticMemory
from AIM2026.Expert.static_map import StaticObstacleMap
from AIM2026.Expert.tunnel_map import TunnelObstacleMap
from AIM2026.Expert.stationary_resolver import StationaryResolver
from AIM2026.Expert.box_dedup import drop_contained
from AIM2026.Expert.track_alias import TrackAliaser
from AIM2026.Expert.stationary_geometry import StationaryGeometry
from AIM2026.Inference.timer_resolution import high_resolution_timer

# VE not-ready reasons that must stop using the last detection at once.
HARD_NOT_READY = ('publisher_stopped', 'position_unavailable')


def telemetry_json(value, **options):
    """Serialize NumPy planner diagnostics without changing planner values."""
    import numpy as np

    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError('Object of type {} is not JSON serializable'.format(type(item).__name__))

    return json.dumps(value, default=convert, **options)


def validation_json_value(value):
    """Make optional diagnostics JSON-safe without changing planner values."""
    import numpy as np
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): validation_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [validation_json_value(item) for item in value]
    return None


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
    def __init__(self, generation, digest, tracker, memory=None, static_map=None, resolver=None, dedup=True, alias=None,
                 tunnel_map=None, ve_object_speed=False, ve_lead_memory=None):
        self.ve_lead_memory = ve_lead_memory
        self.ve_object_speed = ve_object_speed
        self.generation, self.digest, self.tracker = generation, digest, tracker
        self.dedup = dedup
        self.alias = alias
        self.geometry = None       # StationaryGeometry, set by main()
        self.resolver = resolver
        self.memory = memory
        self.static_map = static_map
        self.tunnel_map = tunnel_map
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
                if self.ve_lead_memory is not None:
                    self.ve_lead_memory.reset()
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
            if self.ve_lead_memory is not None:
                self.ve_lead_memory.reset()
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
        if self.dedup:
            row = dict(row, boxes=drop_contained(row['boxes'])[0])
        self.packet = self.tracker.update(row)
        if self.alias is not None:
            self.packet = self.alias.apply(self.packet)
        if self.resolver is not None:
            self.packet = self.resolver.apply(self.packet)
        if self.memory is not None:
            self.packet = self.memory.apply(self.packet)
        if self.geometry is not None:
            self.packet = self.geometry.apply(self.packet)
        if self.static_map is not None:
            self.packet = self.static_map.apply(self.packet)
        if self.tunnel_map is not None:
            self.tunnel_map.observe_packet(self.packet)
        if self.ve_object_speed:
            from AIM2026.Expert.ve_speed_experiment import use_ve_speed
            self.packet = use_ve_speed(self.packet)
        # This output is never fed back into the observed-only tracker/memories.
        if self.ve_lead_memory is not None:
            self.packet = self.ve_lead_memory.apply(self.packet)
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
        from AIM2026.Localization.service import create_localization
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
    parser.add_argument('--ve-object-speed', action='store_true',
                        help='Experiment: use raw VE speed for moving/uncertain objects; retain stationary handling')
    parser.add_argument('--no-lidar-rules', action='store_true',
                        help='Disable VE LiDAR refinement; bus/truck supplement has its own config switch')
    parser.add_argument('--detection-hold-s', type=float, default=2.,
                        help='Inference: hold missing large-vehicle detections for this many seconds')
    parser.add_argument('--lane-motion-tracking', action='store_true',
                        help='Inference: track only large-vehicle candidates along mapped highway lanes')
    parser.add_argument('--config', type=Path, default=ROOT/'AIM2026/config_local.yaml')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--drive', action='store_true', help='Send vehicle commands; default is observation only')
    parser.add_argument('--visualize', action='store_true', help='Write read-only RViz input logs')
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run-id', help='Scenario validation identity; enables controller-neutral pipeline logging')
    parser.add_argument('--speed-cap-kmh', type=float, default=None,
                        help='Optional additional speed ceiling; highway default comes from highway.py')
    parser.add_argument('--sim-profile', choices=('normal','competition'), default=None,
                        help='Override the configured simulator steering direction')
    parser.add_argument('--competition', action='store_true',
                        help='Competition configuration: no development signal ignores unless --ignore-signals is given')
    parser.add_argument('--ignore-signals', type=lambda text: [x for x in text.split(',') if x], default=None,
                        help='Comma-separated map signal IDs the Expert ignores (development default LCS01: no signal '
                             'stands at the tunnel entrance; none with --competition)')
    parser.add_argument('--scenario-context', type=Path,
                        help='Route/scenario configuration written by the scenario runner (ve_gps_scenario_context_v1)')
    parser.add_argument('--stop-file', type=Path,
                        help='Scenario harness requests graceful shutdown through this file')
    parser.add_argument('--static-geometry-fusion', action='store_true',
                        help='Tracker: publish fused footprints for stationary VE boxes')
    parser.add_argument('--moving-carry-s', type=float, default=0.75,
                        help='Tracker: seconds a confirmed mover keeps its speed (0 = off)')
    parser.add_argument('--slow-window-s', type=float, default=0.,
                        help='Tracker: slow-mover confirmation window in seconds (0 = off)')
    parser.add_argument('--no-static-map', action='store_true',
                        help='Plan the static-obstacle and Jamming sections on instantaneous VE boxes instead of the accumulated maps')
    parser.add_argument('--no-static-memory', action='store_true',
                        help='Do not remember small obstacles the detector reports on only some frames')
    parser.add_argument('--stationary-hold-s', type=float, default=4.,
                        help='Tracker: seconds a long-stopped object stays stationary through weak evidence (0 = off)')
    parser.add_argument('--no-box-dedup', action='store_true',
                        help='Keep a VE box that duplicates another box of the same frame (mostly inside it, or same size and centre)')
    parser.add_argument('--no-stationary-geometry', action='store_true',
                        help='Report the latest VE box of a parked object instead of the median of its recent boxes')
    parser.add_argument('--no-track-alias', action='store_true',
                        help='Keep separate ids for the piece and whole boxes VE alternates between on one parked object')
    parser.add_argument('--no-stationary-resolver', action='store_true',
                        help='Keep the tracker\'s uncertain label for parked objects (no 3 s position-history check)')
    parser.add_argument('--no-heading-gate', action='store_true',
                        help='Tracker: do not continue a vehicle track that moved along its own axis beyond the normal gate')
    parser.add_argument('--stale-hold-s', type=float, default=.3,
                        help='Coast (no throttle, last brake/steer) through input gaps up to this long before braking fully (0 = brake at once)')
    parser.add_argument('--history-gap-s', type=float, default=.45,
                        help='Tracker: detection gap that restarts the motion history (default 0.45 s)')
    parser.add_argument('--fitted-speed-above-mps', type=float, default=3.,
                        help='Tracker: report the position-fit speed instead of the agreeing VE speed from this speed on (0 = VE)')
    parser.add_argument('--no-fragment-gate', action='store_true',
                        help='Tracker: do not continue a track through a VE box that is only a piece of the same object')
    args = parser.parse_args(argv)
    if args.ignore_signals is None:
        args.ignore_signals = [] if args.competition else ['LCS01']
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('seconds must be finite and nonnegative')
    cfg = read_config(args.config)
    if not math.isfinite(args.detection_hold_s) or args.detection_hold_s < 0:
        parser.error('detection-hold-s must be finite and nonnegative')
    cfg['Inference']['vision']['large_vehicle_hold_s'] = args.detection_hold_s
    if args.no_lidar_rules:
        cfg['Inference']['vision'].update(
            lidar_heading=False, lidar_centers=False)
        print('[Perception] VE LiDAR refinements disabled; bus/truck supplement uses its own config', flush=True)
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
    if args.lane_motion_tracking:
        from AIM2026.Inference.vision.lane_tracking import LaneTracker
        geometry = LaneTracker.from_context(context)
        cfg['Inference']['vision']['large_vehicle_lanes'] = {
            name: lane.points for name, lane in geometry.lanes.items()}
    generation = str(uuid.uuid4())
    child_cfg = publisher_config(cfg, output, generation)
    config_path = output/'inference.yaml'
    config_path.write_text(yaml.safe_dump(child_cfg, sort_keys=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), model=label, weights=str(weights),
                         sim_profile=args.sim_profile,
                         object_speed_source=('lane_motion_tracking_with_fallback' if args.lane_motion_tracking
                                              else 've_raw_experiment' if args.ve_object_speed else 'tracked'),
                         detection_hold_s=args.detection_hold_s,
                         ve_lead_hold_s=cfg['Inference']['vision'].get('ve_lead_hold_s', 0.),
                         lane_motion_tracking=args.lane_motion_tracking,
                         ignored_signals=args.ignore_signals,
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
    from AIM2026.Inference.vision.ve_lead_memory import VeLeadMemory
    from AIM2026.Expert.src.autonomous_driving import AutonomousDriving
    inputs = Inputs(generation, sha, VeObjectTracker(
        static_geometry_fusion=args.static_geometry_fusion, moving_carry_s=args.moving_carry_s,
        slow_window_s=args.slow_window_s, stationary_hold_s=args.stationary_hold_s,
        fragment_gate=not args.no_fragment_gate, heading_gate=not args.no_heading_gate, history_gap_s=args.history_gap_s,
        fitted_speed_above_mps=args.fitted_speed_above_mps),
        memory=None if args.no_static_memory else StaticMemory(),
        resolver=None if args.no_stationary_resolver else StationaryResolver(), dedup=not args.no_box_dedup,
        alias=None if args.no_track_alias else TrackAliaser(),
        static_map=None if args.no_static_map else StaticObstacleMap(),
        tunnel_map=None if args.no_static_map else TunnelObstacleMap(),
        ve_object_speed=args.ve_object_speed,
        ve_lead_memory=VeLeadMemory(cfg['Inference']['vision'].get('large_vehicle_lanes'),
            cfg['Inference']['vision'].get('ve_lead_hold_s', 0.)))
    print('[Perception] VE same-lane lead hold=%ss (no LiDAR seed; after tracking)' %
          cfg['Inference']['vision'].get('ve_lead_hold_s', 0.), flush=True)
    stop = []
    old_handlers = {sig: signal.signal(sig, lambda *_: stop.append(True))
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    inputs.geometry = None if args.no_stationary_geometry else StationaryGeometry()
    process = sender = receiver = None
    pipeline = None
    logs = []
    try:
        expert_log = (output/'expert.log').open('w', encoding='utf-8', buffering=1); logs.append(expert_log)
        with contextlib.redirect_stdout(expert_log):
            driving = AutonomousDriving()
        if inputs.static_map is not None and hasattr(driving.lattice_planning_process, 'attach_static_map'):
            driving.lattice_planning_process.attach_static_map(inputs.static_map, inputs.tunnel_map,
                                                               getattr(driving, 'path', None))
        if os.environ.get('VE_CORRIDOR_TIMEOUT_S'):
            # Development experiment only: corridor search time budget of the recorded static planner.
            jam = getattr(driving.lattice_planning_process, 'recorded_jamming', None)
            if jam is not None:
                jam.local.corridor_search_timeout = float(os.environ['VE_CORRIDOR_TIMEOUT_S'])
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
        if args.run_id:
            from scenario_runner.validation.pipeline_log import PipelineWriter
            pipeline = PipelineWriter(output/'pipeline.jsonl', args.run_id, (
                Path(__file__), ROOT/'AIM2026/Expert/main.py',
                ROOT/'AIM2026/Expert/src/autonomous_driving.py',
                ROOT/'AIM2026/Expert/src/planning/planning_process.py',
                ROOT/'AIM2026/Expert/src/control/path_controller.py',
                ROOT/'AIM2026/Inference/main.py'))
        path_log = input_log = None
        if args.visualize:
            path_log = (output/'paths.jsonl').open('w', encoding='utf-8', buffering=1)
            input_log = (output/'inputs.jsonl').open('w', encoding='utf-8', buffering=1)
            logs.extend((path_log,input_log))
        child_options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        inference_command = [sys.executable, '-u', '-B', str(ROOT/'AIM2026/Inference/main.py'), '--config', str(config_path)]
        if args.run_id:
            inference_command += ['--run-id', args.run_id, '--audit-dir', str(output)]
        capped = set(filter(None, os.environ.get('AIM_EXPERT_BLAS_CAPPED', '').split(',')))
        inference_env = {k: v for k, v in os.environ.items() if k not in capped and k != 'AIM_EXPERT_BLAS_CAPPED'}
        process = subprocess.Popen(inference_command, env=inference_env,
                                   cwd=ROOT, stdout=publisher_log, stderr=subprocess.STDOUT, **child_options)
        started = next_status = time.monotonic()
        next_snapshot = 0.
        last_steer = 0.
        last_fresh = last_sent = None
        # Loop pacing and the PID's control_dt use perf_counter: on Windows/Python 3.10
        # time.monotonic() only advances in 15.6 ms steps, which made dt jump between
        # 0, 16, 31 and 47 ms. Packet ages keep the shared monotonic_ns contract.
        timer = high_resolution_timer()
        timer_active = timer.__enter__()
        print('[Expert] 1 ms timer resolution: %s' % timer_active, flush=True)
        period = 1. / driving.realtime_sampling_rate
        last_tick = next_tick = time.perf_counter()
        cycle_id = 0
        while (not stop and not (args.stop_file and args.stop_file.exists())
               and not (args.seconds and time.monotonic()-started >= args.seconds)):
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
                        input_log.write(json.dumps(
                            dict(inputs.packet, run_id=args.run_id) if args.run_id else inputs.packet,
                            allow_nan=False)+'\n')
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
            vehicle = objects = path = None
            code = policy.output(now)
            if fresh:
                if policy.reason == 'off_route':
                    raise RuntimeError('GPS pose outside route corridor')
                vehicle, objects = planner_inputs(inputs.packet, inputs.pose, position_sources=('GPS', 'GPS_dead_reckoning')
                    if (cfg.get('Localization') or {}).get('enabled', False) else ('GPS',))
                # Admission needs a recently observed gap, not merely a packet
                # that still passes the wider running/coasting expiry budget.
                observed_ns = inputs.packet.get('object_received_ns', inputs.packet['sensor_received_ns'])
                vehicle.object_observation_age_s = max(0., now - observed_ns / 1e9)
                traffic = [policy.target_id, code, 0, now]
                with contextlib.redirect_stdout(expert_log):
                    tick = control_tick or (lambda d,*a,**kw: d.execute(*a,**kw))
                    command, path = tick(driving, vehicle, objects, traffic, sync_mode=False,
                        control_dt=control_dt, ego_header_seq=inputs.sequences[True],
                        ego_source_ns=inputs.pose['selected_samples']['vehicle_status'].get('source_stamp_ns'))
                if path_log:
                    points = [[float(p.x),float(p.y)] if hasattr(p,'x') else [float(p[0]),float(p[1])]
                              for p in list(path or [])[:80]]
                    path_log.write(telemetry_json(dict(ego=[vehicle.position.x,vehicle.position.y,vehicle.yaw],
                                                   path=points), allow_nan=False)+'\n')
                    if now >= next_snapshot:
                        from AIM2026.Visualization.snapshot import write_snapshot
                        try:
                            write_snapshot(output/'visualization.json',
                                driving.lattice_planning_process.last_result, vehicle, objects, path,
                                signal=dict(id=policy.target_id, code=code, bound=policy.bound,
                                            reason=getattr(policy, 'reason', None)))
                        except (OSError, TypeError, ValueError, AttributeError) as exc:
                            print('[RViz snapshot] '+str(exc), file=expert_log)
                        next_snapshot = now+.1
            generated_ns = time.monotonic_ns()
            encoded = wire_command(command, driving.max_steering, hold_steer=last_steer)
            # A short input gap (a late pose or VE frame, 97 % of them under 0.3 s and up to 160 per full loop) used to
            # send full brake at once and the car lurched. Within --stale-hold-s of the last fresh tick it coasts
            # instead: no throttle, the last brake and steering kept; a longer gap still brakes fully.
            stale_hold = (command is None and args.stale_hold_s > 0 and last_fresh is not None
                          and last_sent is not None and now - last_fresh <= args.stale_hold_s)
            if stale_hold:
                encoded = Command(brake=last_sent.brake, steer=last_sent.steer)
            if fresh:
                last_fresh = now
            last_sent = encoded
            last_steer = encoded.steer
            if sender:
                sender.send(encoded)
            sent_ns = time.monotonic_ns() if sender else None
            row = dict(status='RUNNING' if fresh else 'WAITING', stale_hold=stale_hold, VE_frames=inputs.frames,
                       position_source=(inputs.pose or {}).get('ego_position_source', 'unavailable'),
                       VE_held=inputs.held_frames,
                       control_pose_frames=inputs.pose_frames, reason=None if fresh else inputs.reason,
                       accel=encoded.accel, brake=encoded.brake, steer=encoded.steer, control_sent=bool(sender),
                       # Traffic-signal decision as delivered to the Expert (why it may be holding at a light).
                       signal=dict(target=policy.target_id, code=code, bound=policy.bound,
                                   head_confidence=policy.confidence, binding=getattr(policy, 'reason', None),
                                   age_s=None if policy.received is None else round(now-policy.received, 3)))
            # Record the exact pose used by this control tick for timing/jitter
            # diagnosis. This does not alter inputs, planning, or commands.
            log_time_ns = time.monotonic_ns()
            plan_result = driving.lattice_planning_process.last_result if fresh else None
            plan_details = getattr(plan_result, 'debug_details', {})
            gate_audit = None
            if args.run_id:
                try:
                    gate_audit = policy.audit(now)
                except Exception:
                    pass  # Optional journal data cannot interrupt vehicle control.
            control_log.write(telemetry_json(dict(row, monotonic_ns=log_time_ns,
                **(dict(run_id=args.run_id, monotonic=log_time_ns / 1e9,
                        VE_sequence=inputs.packet.get('sequence') if fresh else None,
                        signal_code=code, signal_id=policy.target_id,
                        signal_head=policy.head_source, signal_gate=gate_audit)
                   if args.run_id else {}),
                control_dt_s=control_dt,
                inference_age_ms=(round((log_time_ns-inputs.packet['sensor_received_ns'])/1e6, 1)
                                  if inputs.packet else None),
                inference_sequence=(inputs.packet.get('sequence') if inputs.packet else None),
                planning=dict(reason=getattr(plan_result, 'reason', None),
                              target_speed=getattr(driving, 'previous_target_velocity', None),
                              target_lane=getattr(plan_result, 'target_link_id', None),
                              entry_merge=plan_details.get('entry_merge'),
                              second_merge_follow=(plan_details.get('merge_safety') or {}).get('second_follow'),
                              merge_safety={k: v for k, v in (plan_details.get('merge_safety') or {}).items()
                                            if k != 'predicted_paths'},
                              second_merge_preparation=plan_details.get('second_merge_preparation'),
                              merge_checks=plan_details.get('merge_checks'),
                              keep_lane_evidence=plan_details.get('keep_lane_evidence'),
                              dynamic_crossing=plan_details.get('dynamic_crossing'),
                              legacy_acc=plan_details.get('legacy_acc'),
                              speed_limits=plan_details.get('speed_limits'),
                              highway_brake_release=plan_details.get('highway_brake_release'),
                              lane_change_cruise=plan_details.get('lane_change_cruise')),
                # Signal decision of this tick (VE phase after policy; 0 = unknown -> stop).
                signal=dict(id=policy.target_id, code=code, head=getattr(policy, 'head_source', None),
                            reason=getattr(policy, 'reason', None), green_samples=policy.allow_samples,
                            phase_stop=getattr(policy, 'phase_stop', False),
                            phase_age_ms=round((now - policy.received) * 1e3, 1) if policy.received else None),
                control_pose=inputs.pose))+'\n')
            if pipeline is not None:
                # The journal is passive. A serialization or disk error is
                # recorded by PipelineWriter and cannot alter the UDP command.
                try:
                    packet = inputs.packet if fresh else None
                    pose = inputs.pose or {}
                    ego = pose.get('ego_pose') if isinstance(pose, dict) else None
                    points = ([[float(p.x), float(p.y)] if hasattr(p, 'x') else
                               [float(p[0]), float(p[1])] for p in list(path or [])[:80]]
                              if fresh else None)
                    boxes = packet.get('boxes') if packet else None
                    planning_ego = (dict(x=vehicle.position.x, y=vehicle.position.y,
                                         yaw_grid_rad=vehicle.yaw, speed_mps=vehicle.velocity)
                                    if vehicle is not None else None)
                    planning_objects = ([dict(track_id=obj.unique_id,
                        center_world=[obj.position.x, obj.position.y],
                        position_xy=[obj.position.x, obj.position.y],
                        planning_speed_mps=obj.velocity, length_m=obj.size_x,
                        width_m=obj.size_y, heading_deg=obj.heading_deg,
                        motion_state=obj.name.partition(':')[2] or None,
                        coordinate_frame='projected_grid',
                        is_static=obj.is_static,
                        speed_unknown=getattr(obj, 'speed_unknown', None),
                        motion_uncertain=getattr(obj, 'motion_uncertain', None))
                        for obj in objects] if objects is not None else None)
                    common = dict(cycle_id=cycle_id, controller='aim2026_integrated_udp',
                                  model_sha256=sha, coordinate_frame='projected_grid')
                    pipeline.write('perception', **common,
                        ve_fresh_max_s=0.5, pose_fresh_max_s=0.15,
                        ve_generation=generation, ve_sequence=packet.get('sequence') if packet else None,
                        ve_valid=fresh, ve_received_ns=packet.get('sensor_received_ns') if packet else None,
                        ve_inference_complete_ns=packet.get('inference_complete_ns') if packet else None,
                        objects=boxes, signal_head_output=packet.get('signal') if packet else None,
                        native_signal_head_output=packet.get('native_signal') if packet else None)
                    pipeline.write('planning', **common,
                        planning_input_source='aim2026_ve_udp' if fresh else None,
                        selected_planning_input_source='aim2026_ve_udp',
                        expert_planner_branch=getattr(driving, 'perception_source', None),
                        ve_sequence=packet.get('sequence') if packet else None,
                        ego_input=planning_ego, input_objects=planning_objects,
                        planned_trajectory=points,
                        target_velocity_mps=getattr(driving, 'previous_target_velocity', None) if fresh else None,
                        planner_name='AutonomousDriving',
                        planning_mode=getattr(plan_result, 'mode', None),
                        planning_reason=getattr(plan_result, 'reason', None),
                        debug_details=validation_json_value(plan_details),
                        signal_input_code=code if fresh else None,
                        signal_target_id=policy.target_id if fresh else None,
                        signal_gate=gate_audit)
                    # Same read-only gate snapshot as the control journal.
                    # This is not a second policy evaluation.
                    pipeline.write('control', **common,
                        planning_cycle_id=cycle_id if fresh else None,
                        ve_sequence=packet.get('sequence') if packet else None,
                        input_path=points,
                        target_velocity_mps=getattr(driving, 'previous_target_velocity', None) if fresh else None,
                        calculated_command=(dict(accel=command.accel, brake=command.brake,
                            front_steer_rad=command.front_steer) if command is not None else None),
                        final_wire_command=dict(accel=encoded.accel, brake=encoded.brake,
                                                steer_normalized=encoded.steer),
                        generated_ns=generated_ns, sent_ns=sent_ns,
                        sent=bool(sender), observed_ego=ego,
                        signal_gate=gate_audit)
                except (OSError, TypeError, ValueError, AttributeError) as journal_error:
                    pipeline.error = str(journal_error)
                    pipeline._report_error()
            cycle_id += 1
            if now >= next_status:
                print(telemetry_json(row), flush=True)
                # Lifecycle file for launchers (the scenario runner waits for DRIVING before it starts the course).
                state = ('DRIVING' if sender else 'SHADOW') if fresh else 'WAITING'
                (output/'status.json').write_text(telemetry_json(dict(row, status=state, elapsed_s=round(now-started, 1))),
                                                  encoding='utf-8')
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
        if pipeline is not None:
            pipeline.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
