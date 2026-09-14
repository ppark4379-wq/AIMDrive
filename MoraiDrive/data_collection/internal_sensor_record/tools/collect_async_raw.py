#!/usr/bin/env python3
"""Record all received ROS bytes on local storage, then seal. Never ticks MORAI."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT_ROOT)]
from async_capture.journal import RawWriter, write_json
from async_capture.live import RawSubscriptions
from async_capture.clock_health import probe as probe_clocks, snapshot as clock_snapshot, evaluate as evaluate_clocks
from async_capture.health_policy import SimulationPauseGrace
from async_capture.provenance import software_identity
from async_capture.sim_mode import FreeRunningModeContinuity, require_free_running_mode
from process_guard import process_lock
from ssd_storage import SSD_ROOT, require_ssd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensor-contract', type=Path, default=ROOT/'configs/encoder_mission_candidate_v1.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--max-gib', type=float, default=20)
    p.add_argument('--reserve-gib', type=float, default=50)
    p.add_argument('--ready-file', type=Path)
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--episode-metadata', type=Path)
    p.add_argument('--wait-for-motion',action='store_true')
    p.add_argument('--motion-timeout',type=float,default=300)
    p.add_argument('--sensor-ready-timeout',type=float,default=30)
    p.add_argument('--sim-pause-grace-seconds',type=float,default=8,
        help=('After capture starts, tolerate only a coherent /clock plus simulated-sensor '
              'pause for this bounded wall interval. Partial stream loss still fails immediately.'))
    p.add_argument('--time-mode',choices=('realtime','simulation_free_run'),default='realtime',
        help='Explicit diagnostic alternative; does not change MORAI mode or claim sensor simultaneity')
    a = p.parse_args()
    if not 1 <= a.sensor_ready_timeout <= 600:
        p.error('Sensor readiness wait must be bounded to 1..600 wall seconds')
    if not 1 <= a.seconds <= 3600 or not 0 < a.max_gib <= 500 or not 10 <= a.reserve_gib <= 500 or not 1 <= a.motion_timeout <= 600:
        p.error('Invalid bounded run duration/capacity')
    if not 3 <= a.sim_pause_grace_seconds <= 30:
        p.error('Simulation pause grace must be 3..30 wall seconds')
    output = a.output or SSD_ROOT/'runs'/datetime.datetime.now().strftime('raw_%Y%m%d_%H%M%S')
    storage = require_ssd(output, check_volume=True, reserve_bytes=int((a.reserve_gib+a.max_gib)*1024**3))
    clock_preflight=probe_clocks()
    if not clock_preflight['passed']:
        print(json.dumps(dict(state='CLOCK_PREFLIGHT_FAILED',clock_health=clock_preflight)),flush=True)
        raise RuntimeError('Local clock health failed; no capture started. Inspect CLOCK_MONOTONIC vs RAW and wall-clock steps.')
    contract = json.loads(a.sensor_contract.read_text())
    from native_capture.ground_truth import GrpcGroundTruth, default_grpc_target
    target=contract.get('capture',{}).get('actor_ground_truth',{}).get(
        'target',default_grpc_target())
    if target == 'auto':
        target = default_grpc_target()
    gt=GrpcGroundTruth(target,timeout=2)
    try:
        mode=gt.call(gt.simulation.GetSynchronousMode,gt.Empty())
    finally:
        gt.channel.close()
    mode_contract=require_free_running_mode(mode,a.time_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_LOG_DIR'] = str(output.parent/'ros_logs')
    os.environ['ROS_HOME'] = str(output.parent/'ros_home')
    episode = json.loads(a.episode_metadata.read_text()) if a.episode_metadata else None
    stopped = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stopped.append(signum))
    for path in (a.ready_file, a.stop_file):
        if path and path.exists():
            raise FileExistsError('Stale supervisor file: '+str(path))
    with process_lock('capture'):
        writer = RawWriter(output, dict(sensor_contract=contract, storage=storage,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            started_wall_ns=time.time_ns(), started_monotonic_ns=time.monotonic_ns(),
            episode=episode, capture_mode='passive_async', sync_mode_preflight=mode,time_mode_contract=mode_contract,
            clock_preflight=clock_preflight,
            software=software_identity(),
            health_continuity_policy=dict(
                name='bounded_coherent_simulation_pause_v1',
                grace_seconds=a.sim_pause_grace_seconds,
                partial_stream_loss='fail_immediately',
            ),
            topic_accounting='callbacks, not wire loss proof'),
            max_bytes=int(a.max_gib*1024**3),
            guard=lambda: require_ssd(output,reserve_bytes=int(a.reserve_gib*1024**3)))
        subscriptions = RawSubscriptions(writer, contract,time_mode=a.time_mode)
        mode_observer=GrpcGroundTruth(target,timeout=1) if a.time_mode=='simulation_free_run' else None
        mode_continuity=(FreeRunningModeContinuity(a.time_mode,3)
                         if mode_observer is not None else None)
        mode_observations=[]
        health_continuity=SimulationPauseGrace(a.sim_pause_grace_seconds)
        started = time.monotonic()
        ready_at = None
        recording_at = None
        recording_mono_ns = None
        resources=[]
        clock_samples=[clock_snapshot()]
        previous_cpu,previous_wall=time.process_time(),time.monotonic()
        success, reason, outcome = False, 'startup_failed', None
        try:
            subscriptions.start()
            next_check = 0
            while True:
                now = time.monotonic()
                if stopped:
                    reason = 'operator_interrupted'
                    break
                if writer.status()['errors']:
                    raise RuntimeError(str(writer.status()['errors']))
                if now >= next_check:
                    if mode_observer is not None:
                        # ROS SyncModeInfo.status is also true in free-running
                        # Simulation Time. Verify the zero external tick period
                        # over gRPC throughout this explicit diagnostic mode.
                        observation=mode_observer.call(mode_observer.simulation.GetSynchronousMode,
                            mode_observer.Empty())
                        continuity=mode_continuity.observe(observation)
                        mode_observations.append(dict(observation=observation,
                                                      continuity=continuity))
                    state = subscriptions.readiness()
                    require_ssd(output, reserve_bytes=int(a.reserve_gib*1024**3))
                    clock_samples.append(clock_snapshot())
                    # A long-run mean must not hide a new step or rate change.
                    if len(clock_samples)>=3:
                        clock_health=evaluate_clocks(clock_samples[-2:])
                        if not clock_health['passed']:
                            raise RuntimeError('Local clock changed during capture: '+str(clock_health))
                    cpu=time.process_time();wall=time.monotonic()
                    resources.append(dict(receive_monotonic_ns=time.monotonic_ns(),
                        cpu_percent=100*(cpu-previous_cpu)/max(.001,wall-previous_wall),
                        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                    previous_cpu,previous_wall=cpu,wall
                    print(json.dumps(dict(state='RECORDING' if recording_at else 'WAITING_FOR_MOTION' if ready_at else 'WAITING_FOR_SENSORS',
                        elapsed=now-started, storage=writer.status(), failures=state['failures'])), flush=True)
                    if ready_at is not None:
                        decision=health_continuity.observe(
                            state['failures'], time.monotonic_ns())
                        if decision['action']=='hold':
                            print(json.dumps(dict(
                                state='SIMULATION_PAUSE_GRACE',
                                elapsed=now-started,
                                continuity=decision,
                            )),flush=True)
                        elif decision['action']=='recovered':
                            print(json.dumps(dict(
                                state='SIMULATION_PAUSE_RECOVERED',
                                elapsed=now-started,
                                continuity=decision,
                            )),flush=True)
                        elif decision['action']=='fail':
                            raise RuntimeError(
                                'Stream/bridge health failed: '+json.dumps(
                                    decision,sort_keys=True))
                    if ready_at is None and state['ready']:
                        ready_at = now
                        if a.ready_file:
                            write_json(a.ready_file, dict(status='raw_capture_ready', raw_run=str(output)))
                    if ready_at is None and now-started > a.sensor_ready_timeout:
                        raise TimeoutError('Required ROS streams/bridge not ready')
                    if ready_at is not None and recording_at is None and (not a.wait_for_motion or state['motion_consistent']):
                        recording_at=now
                        recording_mono_ns=time.monotonic_ns()
                        print('MOTION_STARTED' if a.wait_for_motion else 'CAPTURE_STARTED',flush=True)
                    if ready_at is not None and recording_at is None and now-ready_at>a.motion_timeout:
                        raise TimeoutError('GPS/Ego motion gate timed out')
                    next_check = now + 2
                if a.stop_file and a.stop_file.exists():
                    outcome = json.loads(a.stop_file.read_text())
                    success = outcome.get('status') == 'completed'
                    reason = 'supervisor_'+str(outcome.get('status'))
                    break
                if recording_at is not None and now-recording_at >= a.seconds:
                    success = not bool(a.stop_file)
                    reason = 'duration_complete' if success else 'supervisor_deadline_exceeded'
                    break
                time.sleep(.02)
        except BaseException as exc:
            reason = type(exc).__name__+': '+str(exc)
            raise
        finally:
            # Stop the active acquisition window before unregister/drain/close.
            # Shutdown time is not a period in which sensors were subscribed.
            recording_end_mono_ns = time.monotonic_ns()
            if mode_observer is not None:mode_observer.close()
            subscriptions.stop()
            final = writer.close(success and not subscriptions.errors, reason,
                dict(topic_state=subscriptions.readiness(), episode_outcome=outcome,
                     resources=resources,recording_start_monotonic_ns=recording_mono_ns,
                     mode_observations=mode_observations,
                     mode_continuity=(dict(
                         max_consecutive_unavailable=mode_continuity.max_consecutive_unavailable,
                         final_consecutive_unavailable=mode_continuity.consecutive_unavailable,
                         total_unavailable=mode_continuity.total_unavailable)
                         if mode_continuity is not None else None),
                     health_continuity=health_continuity.snapshot(
                         recording_end_mono_ns),
                     clock_samples=clock_samples,
                     recording_end_monotonic_ns=recording_end_mono_ns,
                     recording_end_scope='capture_loop_exit_before_subscription_shutdown'))
            print(json.dumps(dict(state='RAW_CLOSED', status=final['status'], reason=reason,
                                  run=str(output), statistics=final['statistics'])), flush=True)
    if final['status'] != 'completed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
