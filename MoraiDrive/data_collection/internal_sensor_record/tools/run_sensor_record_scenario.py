"""Run the current team scenario/expert with local Sensor Record capture.

This adapter does not edit or copy scenario/expert source. Sensor Record covers
a bounded interior window; capture boundaries remain explicit in the journal.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=Path(os.environ.get('MORAI_PROJECT_ROOT',str(ROOT.parents[1]))).expanduser().resolve()
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(PROJECT_ROOT)]
from async_capture.internal_sensor_record_contract import (
    DEFAULT_ASSEMBLY, DEFAULT_SENSOR_SET, capture_preflight,
)
from async_capture.journal import digest, write_json
from async_capture.sensor_record_admission import build_capture_contract
from async_capture.sensor_record import inspect_directory
from async_capture.native_session import native_root_from_links
from async_capture.sim_mode import require_free_running_mode
from nas_storage import powershell
from native_capture.ground_truth import GrpcGroundTruth, default_grpc_target
from ssd_storage import SSD_ROOT, WINDOWS_ROOT, require_ssd, windows_path
from network_config import runtime_environment
from runtime_paths import (
    ANALYSIS_PYTHON, WINDOWS_SENSOR_DATA_ROOT,
    ensure_runtime_directory,
)
from run_sensor_record_probe import verify_native_targets
from python_environment import (
    build_analysis_environment, build_capture_environment,
    probe_analysis_environment,
)
from sensor_record_barrier import (
    prepare_coordination_directory, require_stop_acknowledgement,
    wait_for_native_quiescence,
)
from scenario_runner.scenarios.segment_modes.modes import PRIMARY_SCENARIOS


def quick_native_capture_summary(native, *, progress=None):
    """Metadata-only barrier used before redirecting Sensor Record to a new lap.

    The expensive byte hashes and image/point-cloud decodes belong to the
    offline validator.  This barrier still rejects an empty, malformed, or
    sensor-incomplete native session before the campaign advances.
    """
    rows = inspect_directory(native, progress=progress)
    counts = Counter(row['role'] for row in rows)
    bytes_by_role = Counter()
    keys_by_role = defaultdict(set)
    for row in rows:
        bytes_by_role[row['role']] += int(row['size_bytes'])
        keys_by_role[row['role']].add(tuple(row['filename_time_candidates_ns']))
    expected = {'camera_1', 'camera_2', 'camera_3', 'camera_4',
                'lidar', 'gps', 'imu'}
    failures = []
    if set(counts) != expected:
        failures.append('native sensor roles are incomplete')
    if not counts or any(counts.get(role, 0) == 0 for role in expected):
        failures.append('one or more native sensor streams are empty')
    if keys_by_role and any(keys_by_role.get(role, set()) != keys_by_role.get('lidar', set())
                            for role in expected):
        failures.append('native filename-time keys do not form complete seven-sensor bundles')
    summary = dict(
        schema='morai_native_capture_quick_barrier_v1',
        metadata_only=True,
        file_count=sum(counts.values()),
        total_bytes=sum(bytes_by_role.values()),
        counts=dict(counts),
        bytes_by_role=dict(bytes_by_role),
        complete_filename_bundles=len(keys_by_role.get('lidar', set())),
        failures=failures,
        passed=not failures,
        authoritative_byte_audit_deferred=True,
        training_visible=False,
        physical_simultaneity_verified=False,
    )
    if failures:
        raise RuntimeError('Quick native capture barrier failed: '+str(failures))
    return summary


def run_logged_stage(command, *, env, log, label, heartbeat_seconds=30):
    """Run one offline stage while keeping the operator informed."""
    started = time.monotonic()
    next_heartbeat = started + heartbeat_seconds
    process = subprocess.Popen(command, env=env, stdout=log,
                               stderr=subprocess.STDOUT)
    while process.poll() is None:
        time.sleep(1)
        now = time.monotonic()
        if now >= next_heartbeat:
            print('[REVIEW] %s running: %.0fs elapsed (details in supervisor log)' %
                  (label, now-started), flush=True)
            next_heartbeat = now + heartbeat_seconds
    return int(process.returncode)


def powershell_with_heartbeat(script, *, timeout, label, heartbeat_seconds=30):
    """Keep the terminal visibly alive while the Windows UI owner blocks."""
    finished = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not finished.wait(heartbeat_seconds):
            print(
                "[CAPTURE] %s: %.0fs elapsed; Sensor Record UI가 ON이며 "
                "상위 센서 감시가 계속 실행 중입니다." %
                (label, time.monotonic() - started),
                flush=True,
            )

    worker = threading.Thread(
        target=heartbeat,
        name="sensor-record-heartbeat",
        daemon=True,
    )
    worker.start()
    try:
        return powershell(script, timeout=timeout)
    finally:
        finished.set()
        worker.join(timeout=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--scenario', choices=PRIMARY_SCENARIOS, default='full_loop',
                   help='Main Scenario Runner mode to execute (default: full_loop)')
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--seconds',type=int,default=180)
    p.add_argument('--full-lap',action='store_true',help='Stop Sensor Record at selected-scenario route completion or the wall bound')
    p.add_argument('--seed',type=int,default=260910,
                   help='Deterministic per-lap seed for weather/time/obstacle selection')
    p.add_argument('--profile-state',type=Path,
                   help='Local-storage campaign state shared across laps so environment choices do not repeat silently')
    p.add_argument('--assembly',type=Path,default=DEFAULT_ASSEMBLY)
    p.add_argument('--sensor-set',type=Path,default=DEFAULT_SENSOR_SET)
    p.add_argument('--static-evidence',type=Path,
                   default=ROOT/'evidence/internal_sensor_record_contract_v1.json')
    p.add_argument('--raw-camera-transport',choices=('required','unrecorded','bridge_dropped','absent'),
                   default='bridge_dropped',
                   help=('Keep native Sensor Record cameras; by default discard duplicate ROS camera '
                         'payloads at the instrumented bridge before ROS publication'))
    p.add_argument('--defer-offline-review',action='store_true',
                   help=('After a metadata-only seven-sensor completion barrier, return so the next '
                         'lap can start. Full hashes, decodes, GT association and admission are then '
                         'performed by validate_sensor_record_campaign.py.'))
    p.add_argument('--check',action='store_true',help='Read-only storage/configuration preflight; no driving or UI action')
    a=p.parse_args()
    analysis_python=Path(os.environ.get(
        'MORAI_ANALYSIS_PYTHON',
        str(ANALYSIS_PYTHON),
    )).resolve()
    analysis_env=None
    analysis_environment_report=None
    if not a.defer_offline_review:
        analysis_env=build_analysis_environment(
            analysis_python, os.environ,
            OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
        )
        analysis_environment_report=probe_analysis_environment(
            analysis_python, analysis_env)
    max_seconds=3570 if a.full_lap else 930
    if (not a.name.replace('_','').isalnum() or len(a.name)>80
            or not 90<=a.seconds<=max_seconds):p.error('Simple unique name up to 80 characters, 90..%d wall seconds'%max_seconds)
    if not 0 <= a.seed < 2**31:
        p.error('Seed must be in [0, 2^31)')
    immutable_preflight=capture_preflight(a.assembly,a.sensor_set,a.static_evidence)
    if not immutable_preflight['ready']:
        raise RuntimeError('Pinned MORAI build/sensor-set preflight failed: '+str(immutable_preflight['failures']))
    base=SSD_ROOT/'sensor_record_auto_v1';run=base/a.name;ep=run/'episode_000'
    if a.profile_state is not None:
        a.profile_state=a.profile_state.expanduser().resolve()
        require_ssd(a.profile_state.parent,reserve_bytes=0)
        if a.profile_state.is_dir():
            raise ValueError('Campaign profile state must be a file path')
    session_plan_path=base/(a.name+'_native_session_plan.json')
    session_apply_path=base/(a.name+'_native_session_apply.json')
    preflight_ui_path=base/(a.name+'_preflight_ui')
    storage=require_ssd(run,check_volume=True)
    if any(path.exists() for path in (run,base/(a.name+'_ui'),preflight_ui_path,
                                      session_plan_path,session_apply_path,
                                      base/'native_sessions'/a.name)):
        raise FileExistsError('Run or native-session namespace exists')
    expected_volume=os.environ.get(
        'MORAI_STORAGE_VOLUME_ID',
        os.environ.get('MORAI_D_SSD_VOLUME_ID',''),
    )
    session_script='& {\n'+(ROOT/'tools/prepare_sensor_record_session.ps1').read_text()+(
        "\n} -SessionName '%s' -StorageRoot '%s' -SensorDataRoot '%s' "
        "-ExpectedVolumeId '%s' -ReserveBytes %d"%
        (a.name,WINDOWS_ROOT,WINDOWS_SENSOR_DATA_ROOT,expected_volume,
         storage['reserve_bytes']))
    session_plan=json.loads(powershell(session_script,timeout=30))
    if session_plan.get('bootstrap'):
        previous_links=[]
        previous_native=None
    else:
        previous_links=[dict(
            role=row['role'],source=row['source'],link_type='Junction',
            target=[row['old_target']],
        ) for row in session_plan['entries']]
        previous_native=native_root_from_links(previous_links,SSD_ROOT)
    # The old namespace is preserved by prepare_sensor_record_session.ps1. Do
    # not scan every old file; long histories otherwise make startup unbounded.
    previous_native_files=None
    env=build_capture_environment(
        dict(os.environ,**runtime_environment()))
    capture_environment_report=dict(
        schema='morai_capture_python_environment_v1',
        python='/usr/bin/python3',
        pythonpath=env['PYTHONPATH'].split(os.pathsep),
        pythonhome_removed='PYTHONHOME' not in env,
        no_user_site=env.get('PYTHONNOUSERSITE') == '1',
        bytecode_disabled=env.get('PYTHONDONTWRITEBYTECODE') == '1',
    )
    # Establish pre-reset receive evidence. A live gRPC Ego alone is insufficient
    # when the actual /Ego_topic vehicle-info connection has stopped sending.
    log_root=ensure_runtime_directory('sensor_record_preflight_logs')
    env['ROS_LOG_DIR']=str(log_root)
    preflight=subprocess.run(['/usr/bin/python3',str(ROOT/'tools/probe_sensor_record_readiness.py'),
        '--raw-camera-transport',a.raw_camera_transport],
        env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
    print(preflight.stdout,flush=True)
    if preflight.returncode:
        raise RuntimeError('Live sensor preflight failed before any scenario reset: '+preflight.stderr[-1000:])
    grpc_target=default_grpc_target()
    gt=GrpcGroundTruth(grpc_target,timeout=3)
    try:
        mode_observation=gt.call(gt.simulation.GetSynchronousMode,gt.Empty())
        observed_mode=require_free_running_mode(mode_observation,'simulation_free_run')
    finally:gt.close()
    coordination=ep/'sensor_record_coordination'
    command=['/usr/bin/python3',str(ROOT/'tools/run_proven_three_lap_scenario.py'),
        '--scenario',a.scenario,
        '--time-mode','simulation_free_run','--nearby-actors','--collision-policy','record',
        '--max-speed-kmh','60','--seconds',str(a.seconds),'--audit-scope','timing','--output',str(run),
        '--audit-python',str(analysis_python),
        '--sensor-record-coordination',str(coordination),
        '--raw-camera-transport',a.raw_camera_transport,'--seed',str(a.seed),
        '--grpc-target',grpc_target]
    if a.profile_state is not None:
        command += ['--profile-state',str(a.profile_state)]
    if a.full_lap:command.append('--full-lap')
    if a.check:
        subprocess.run(command+['--check'],env=env,check=True)
        print(json.dumps(dict(storage=storage,previous_links=previous_links,
                              previous_native_root=(str(previous_native)
                                                    if previous_native else None),
                              previous_native_files=previous_native_files,
                              previous_native_inventory_scanned=False,
                              proposed_native_session=session_plan,observed_mode=observed_mode,
                              capture_wall_seconds=a.seconds-30,scenario_command=command,
                              capture_environment=capture_environment_report,
                              immutable_capture_preflight=immutable_preflight)))
        return
    # Ctrl-C can terminate the WSL parent before the Windows UI owner finishes
    # its finally block.  Before changing any SensorData junction, observe the
    # real MORAI lamp and close a stale recording session if necessary.
    ensure_off_process=subprocess.run(
        [sys.executable,str(ROOT/'tools/recover_sensor_record_stop.py'),
         '--pid',str(a.pid),'--name',a.name+'_preflight_ui'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45,
    )
    if ensure_off_process.returncode:
        raise RuntimeError('Cannot establish Sensor Record OFF before native redirect: '
                           +ensure_off_process.stderr[-1000:])
    ensure_off_result=json.loads(ensure_off_process.stdout)
    if ensure_off_result.get('status') != 'off_verified':
        raise RuntimeError('Sensor Record OFF recovery returned an invalid result')
    write_json(session_plan_path,session_plan)
    windows_plan=windows_path(session_plan_path)
    apply_script='& {\n'+(ROOT/'tools/prepare_sensor_record_session.ps1').read_text()+(
        "\n} -SessionName '%s' -StorageRoot '%s' -SensorDataRoot '%s' "
        "-ExpectedVolumeId '%s' -ReserveBytes %d "
        "-Apply -PlanPath '%s' -SimulatorProcessId %d"%
        (a.name,WINDOWS_ROOT,WINDOWS_SENSOR_DATA_ROOT,expected_volume,
         storage['reserve_bytes'],windows_plan,a.pid))
    session_apply=json.loads(powershell(apply_script,timeout=90))
    write_json(session_apply_path,session_apply)
    links=verify_native_targets()
    native=native_root_from_links(links,SSD_ROOT)
    expected_native=base/'native_sessions'/a.name
    if native != expected_native:
        raise ValueError('New native junction set has an unexpected target')
    initial=inspect_directory(native)
    if initial:
        raise ValueError('New isolated native session is not empty')
    session_record=native/'SESSION.json'
    native_session_snapshot=dict(prepared_before_capture=True,session_name=a.name,
        native_root=str(native),previous_native_root=(str(previous_native)
                                                      if previous_native else None),
        source_directories_preserved=True,source_files_deleted=False,
        first_workstation_bootstrap=bool(session_plan.get('bootstrap')),
        session_plan=dict(path=str(session_plan_path),sha256=digest(session_plan_path)),
        session_apply=dict(path=str(session_apply_path),sha256=digest(session_apply_path)),
        session_record=dict(path=str(session_record),sha256=digest(session_record)))
    log_path=base/(a.name+'_supervisor.log')
    proc=None;error=None;scenario_warning=None;offline_stage_results=[]
    native_storage_quiescence=None
    postprocess_status='pending' if a.defer_offline_review else 'running'
    with log_path.open('x') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+65
            while not (ep/'ready.json').exists():
                if proc.poll() is not None or (ep/'outcome.json').exists():
                    supervisor_tail=(log_path.read_text(
                        encoding='utf-8',errors='replace')[-4000:]
                        if log_path.is_file() else '')
                    raise RuntimeError(
                        'Scenario ROS capture exited before readiness '
                        '(exit=%s). Supervisor tail:\n%s' %
                        (proc.poll(),supervisor_tail)
                    )
                if time.monotonic()>deadline:
                    raise TimeoutError(
                        'Scenario ROS capture did not become ready within 65 seconds; '
                        'inspect '+str(log_path)
                    )
                time.sleep(.2)
            plan_path=run/'plan.json'
            acquisition_path=run/'sensor_contract.json'
            plan=json.loads(plan_path.read_text())
            if not acquisition_path.is_file():
                raise RuntimeError('Scenario acquisition contract missing before Sensor Record')
            producer_paths = sorted({
                path.resolve()
                for pattern in (
                    '*.py', 'tools/*.py', 'tools/*.ps1', 'tools/*.sh',
                    'async_capture/*.py', 'native_capture/*.py',
                    'ros/*.py', 'bridge/*.py', 'morai_pretraining/*.py',
                )
                for path in ROOT.glob(pattern)
                if path.is_file()
            })
            snapshot_root=run/'software_snapshot'
            snapshot_root.mkdir(exist_ok=False)
            producer_hashes={}
            for path in producer_paths:
                relative=path.relative_to(ROOT)
                destination=snapshot_root/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                with path.open('rb') as source,destination.open('xb') as target:
                    shutil.copyfileobj(source,target)
                    target.flush();os.fsync(target.fileno())
                producer_hashes[str(relative)]=digest(destination)
            contract=build_capture_contract(immutable_preflight,plan,
                plan_sha256=digest(plan_path),
                producer_software=producer_hashes,
                producer_software_snapshot=dict(root=str(snapshot_root),files=producer_hashes),
                native_session=native_session_snapshot,
                acquisition_contract=dict(path=str(acquisition_path),sha256=digest(acquisition_path)))
            if not contract['ready']:
                raise RuntimeError('Runtime capture contract failed before Sensor Record: '+str(contract['failures']))
            write_json(run/'capture_contract.json',contract)
            write_json(run/'native_setup.json',dict(storage=storage,links=links,native_root=str(native),initial_files=initial,
                pre_reset_sensor_preflight=json.loads(preflight.stdout),command=command,
                capture_environment=capture_environment_report,
                raw_camera_transport=a.raw_camera_transport,
                capture_contract_sha256=digest(run/'capture_contract.json')))
            write_json(run/'native_files_before.json',inspect_directory(native))
            ui=windows_path(base/(a.name+'_ui'))
            windows_coord=windows_path(coordination)
            prepare_coordination_directory(coordination,ep)
            script='& {\n'+(ROOT/'tools/bounded_sensor_record.ps1').read_text()+(
                "\n} -SimulatorProcessId %d -OutputDirectory '%s' -AllowedRoot '%s' "
                "-WallSeconds %d -CoordinationDirectory '%s' -RestoreWindow"%
                (a.pid,ui,WINDOWS_ROOT,a.seconds-30,windows_coord))
            print('ROS ready; starting bounded automatic Sensor Record',flush=True)
            result=json.loads(powershell_with_heartbeat(
                script, timeout=a.seconds+10, label=a.name,
            ))
            result['coordination_stop_acknowledgement']=(
                require_stop_acknowledgement(coordination))
            write_json(run/'native_ui_result.json',result)
            proc.wait(timeout=1200 if a.full_lap else 300)
            if proc.returncode:
                raw_seal=ep/'raw/RAW_COMPLETE.json'
                raw_completed=(raw_seal.is_file() and
                    json.loads(raw_seal.read_text()).get('status')=='completed')
                quality_seal=ep/'quality/QUALITY_COMPLETE.json'
                if raw_completed and quality_seal.is_file():
                    scenario_warning='Scenario process exited %d after sealed raw/quality capture; preserve outcome separately'%proc.returncode
                else:
                    raise RuntimeError('Scenario/raw audit failed before sealed capture; inspect preserved logs')
            print('[BARRIER] Sensor Record stopped; waiting for native files to settle',
                  flush=True)
            native_storage_quiescence=wait_for_native_quiescence(native)
            print('[BARRIER] Native files stable for %.1fs; scanning metadata' %
                  native_storage_quiescence['stable_seconds'], flush=True)
            def native_scan_progress(role, phase, role_checked, role_files):
                if phase == 'listing':
                    print('[BARRIER] %s: listing files' % role, flush=True)
                elif phase == 'listed':
                    print('[BARRIER] %s: %d files found' %
                          (role, int(role_files or 0)), flush=True)
                else:
                    print('[BARRIER] %s: %d/%d files checked' %
                          (role, role_checked, int(role_files or 0)), flush=True)
            quick_summary=quick_native_capture_summary(
                native, progress=native_scan_progress)
            quick_summary['storage_quiescence']=native_storage_quiescence
            write_json(run/'native_files_after.json',quick_summary)
            print('Native capture barrier: %d complete seven-sensor bundles, %.2f GiB; byte audit %s'%(
                quick_summary['complete_filename_bundles'],
                quick_summary['total_bytes']/float(1024**3),
                'deferred to 03' if a.defer_offline_review else 'starting now'),flush=True)
            if scenario_warning is not None:
                outcome_path=ep/'outcome.json'
                outcome=(json.loads(outcome_path.read_text())
                         if outcome_path.is_file() else {})
                error=('Full-lap scenario did not complete after raw capture: '
                       +str(outcome.get('reason') or scenario_warning))
                postprocess_status='raw_captured_scenario_failed'

            # Drive and Sensor Record are already stopped. Complete the same
            # offline review without requiring another manual command/run.
            # Offline tools deserialize ROS1 wire payloads, so they need the
            # same Noetic/catkin message paths as the capture child even when
            # this supervisor was launched from a clean conda shell.
            stages=[] if a.defer_offline_review else [
                ('audit_sensor_record_probe.py',['--run',str(run),'--raw',str(ep/'raw'),'--output',str(ep/'native_audit')]),
                ('build_sensor_record_associations.py',['--episode',str(ep)]),
                ('build_sensor_record_gt_timeline.py',['--episode',str(ep),'--output',str(ep/'gt_timeline_v1')]),
                ('audit_sensor_record_direct_traffic.py',['--episode',str(ep),'--output',str(ep/'traffic_audit_v1')]),
                ('audit_sensor_record_actor_inventory.py',['--episode',str(ep),'--output',str(ep/'actor_inventory_audit_v1')]),
                ('audit_sensor_record_object_geometry.py',['--episode',str(ep),'--output',str(ep/'object_geometry_audit_v1')]),
                ('audit_sensor_record_observer_overhead.py',['--episode',str(ep),'--output',str(ep/'observer_overhead_v1')]),
                ('audit_sensor_record_route_signals.py',['--episode',str(ep),'--output',str(ep/'route_signal_audit_v1')]),
                ('audit_sensor_record_motion_time.py',['--episode',str(ep),'--output',str(ep/'motion_time_v1')]),
                ('audit_sensor_record_exposure.py',['--audit',str(ep/'native_audit'),'--quality',str(ep/'quality'),'--output',str(ep/'exposure_v1')]),
            ]
            if a.full_lap:
                if not a.defer_offline_review:
                    stages.append(('audit_scenario_interventions.py',[
                    '--episode',str(ep),
                    '--output',str(ep/'scenario_intervention_audit_v3'),
                    ]))
            if not a.defer_offline_review:
                stages.extend([
                ('summarize_sensor_record_review.py',['--episode',str(ep),'--output',str(ep/'review_closeout')]),
                ('admit_sensor_record_collection.py',['--episode',str(ep),'--capture-contract',str(run/'capture_contract.json'),
                    '--run-plan',str(plan_path),'--output',str(ep/'admission_v1')]),
                ])
            for tool,args in stages:
                print('Offline review: '+tool,flush=True)
                return_code=run_logged_stage(
                    [str(analysis_python),str(ROOT/'tools'/tool)]+args,
                    env=analysis_env,log=log,label=tool)
                offline_stage_results.append(dict(
                    tool=tool,return_code=return_code))
            failed_stages=[row for row in offline_stage_results
                           if row['return_code'] != 0]
            if failed_stages:
                # A scenario-specific audit can reject an incomplete lap while
                # the raw/native sensor evidence remains valid.  Run every
                # independent stage, including admission, before preserving a
                # fail-closed overall result so those two conclusions are not
                # conflated.
                raise RuntimeError('Offline review stage failures: '
                                   +str(failed_stages))
            postprocess_status='pending' if a.defer_offline_review else 'completed'
        except BaseException as exc:
            error=repr(exc)
            if isinstance(exc, KeyboardInterrupt) and proc is not None and proc.poll() is None:
                # Let the scenario owner run its cleanup (expert stop, Ego
                # brake, raw seal, owned-actor cleanup) instead of orphaning
                # the capture lock when the campaign launcher is interrupted.
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=75)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc is not None and proc.poll() is None:
                # Once the raw stop intent exists, allow the child to finish
                # its sealed journal even if the independent UI owner failed.
                if (ep/'stop.json').is_file():
                    try:proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:pass
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=45)
                    except subprocess.TimeoutExpired:error=(error or '')+'; scenario shutdown still pending'
            if run.exists():
                admission_path=ep/'admission_v1/report.json'
                admission=json.loads(admission_path.read_text()) if admission_path.exists() else None
                write_json(run/'NATIVE_CAPTURE_RESULT.json',dict(error=error,
                    scenario_name=a.scenario,
                    scenario_exit_code=(proc.poll() if proc is not None else None),
                    scenario_warning=scenario_warning,
                    capture_environment=capture_environment_report,
                    analysis_python=str(analysis_python),
                    analysis_dependencies=[
                        'cv2','numpy','pyproj','scipy','PIL','rospkg','rospy'],
                    analysis_environment=analysis_environment_report,
                    postprocess_status=postprocess_status,
                    quick_native_capture=(json.loads((run/'native_files_after.json').read_text())
                        if (run/'native_files_after.json').is_file() else None),
                    native_storage_quiescence=native_storage_quiescence,
                    offline_stage_results=offline_stage_results,
                    capture_contract_sha256=(digest(run/'capture_contract.json')
                        if (run/'capture_contract.json').exists() else None),admission=admission,
                    training_visible=False,physical_simultaneity_verified=False))
    print(json.dumps(dict(output=str(run),scenario_name=a.scenario,error=error)),flush=True)
    if error:raise SystemExit(2)


if __name__=='__main__':main()
