#!/usr/bin/env python3
"""Measure the normal VE sensor/publisher path without starting an Expert."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
for entry in (REPO, REPO/'VLDrive', REPO/'MoraiDrive/data_collection/internal_sensor_record'):
    sys.path.insert(0, str(entry))


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def findings(report):
    """Report faults, never equate a finished timer with good perception."""
    faults = {}
    if not report.get('accepted_outputs'):
        faults['no_accepted_outputs'] = True
    for key in ('jpeg_warning_occurrences', 'lidar_gate_counter_differs_from_selected_sample'):
        if report.get(key):
            faults[key] = report[key]
    repeated = report.get('input_selection', {}).get('lidar', {}).get('reused_selected_sample', 0)
    if repeated:
        faults['reused_lidar_scans'] = repeated
    losses = (report.get('transport_delivery') or {}).get('unreceived_within_observed_window', [])
    if losses:
        faults['output_udp_gaps'] = len(losses)
    for name, data in report.get('sensors', {}).items():
        counts = data.get('final_counters', {}).get('counters', {})
        errors = {k:v for k,v in counts.items() if v and k not in ('frames_started', 'end_markers')}
        if errors:
            faults[name] = errors
    gaps = (report.get('sender_output_rate') or {}).get('gaps_over_500ms', 0)
    if gaps:
        faults['output_gaps_over_500ms'] = gaps
    return faults


def process_rss_mib(process):
    if process is None:
        return None
    try:
        for line in Path('/proc/{}/status'.format(process.pid)).read_text().splitlines():
            if line.startswith('VmRSS:'):
                return round(int(line.split()[1])/1024, 1)
    except (OSError, AttributeError):
        pass
    return None


def wait_background_start(output, child, timeout=150.):
    """Do not tell someone going to sleep that a PID proves sensor readiness."""
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        path = output/'status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('phase') == 'measuring' and state.get('input_fresh'):
            print('[수신 시작] '+state.get('performance','VE 계측 시작'),flush=True)
            return 0
        if state.get('status') == 'COMPLETE':
            print('[계측 종료] '+str(output/'campaign_report.json'),flush=True)
            return 0
        if state.get('status') in ('FAILED','INTERRUPTED') or child.poll() is not None:
            print('[시작 실패] '+str(state.get('reason',output/'console.log')),flush=True)
            return 1
        time.sleep(.25)
    child.send_signal(signal.SIGINT)
    print('[시작 실패] 초기화 제한 시간 초과. 자식 종료 요청; '+str(output/'console.log'),flush=True)
    return 1


def campaign(output, checkpoint, seconds, repeat, session_factory, topology_check,
             is_shutdown, report_writer):
    status = dict(schema='ve_hz_campaign_v1', status='RUNNING', checkpoint=checkpoint,
        segment_seconds=seconds, requested_segments=repeat, completed_segments=0,
        segments=[], model_restarted_between_segments=True, control_commands_sent=0,
        training_updates=0, expert_callback_measurement=False)
    save(output/'status.json',status)
    session = None
    exit_code = 0
    try:
        for number in range(1,repeat+1):
            topology_check()
            folder = output if repeat == 1 else output/('segment_{:03d}'.format(number))
            folder.mkdir(exist_ok=True)
            session = session_factory(folder, checkpoint, observe_gt=False)
            status.update(active_segment=number, phase='model_and_sensors_starting',
                          active_directory=str(folder), updated_unix=time.time())
            save(output/'status.json',status)
            print('[회차 {}/{}] {}'.format(number,repeat,folder),flush=True)
            start = None
            completed = False
            try:
                session.start()
                session.await_ready()
                start = time.monotonic()
                last_heartbeat = 0.
                status['phase'] = 'measuring'
                while time.monotonic()-start < seconds:
                    if is_shutdown():
                        raise KeyboardInterrupt()
                    if session.error or session.sender.poll() is not None:
                        session.require_ready()
                    now = time.monotonic()
                    if now-last_heartbeat >= 10.:
                        topology_check(own_publisher=True)
                        if shutil.disk_usage(output).free < 1024**3:
                            raise RuntimeError('계측 저장 공간이 1GiB 미만입니다. 기록을 보존하고 종료합니다.')
                        log_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                        if log_bytes > 2*1024**3:
                            raise RuntimeError('계측 로그가 2GiB 한도에 도달했습니다. 기록을 보존하고 종료합니다.')
                        snapshot = session.snapshot()
                        status.update(elapsed_segment_s=round(now-start,1),
                            remaining_segment_s=round(max(0.,seconds-(now-start)),1),
                            updated_unix=time.time(), publisher_rss_mib=process_rss_mib(session.sender),
                            input_fresh=bool(snapshot['ve_fresh']), received_rejections=session.rejected,
                            log_bytes=log_bytes, performance=session.performance_line())
                        save(output/'status.json',status)
                        with (folder/'heartbeat.jsonl').open('a') as heartbeat:
                            heartbeat.write(json.dumps({k:status[k] for k in (
                                'updated_unix','elapsed_segment_s','publisher_rss_mib',
                                'input_fresh','received_rejections','log_bytes','performance')})+'\n')
                        print('[VE Hz] '+status['performance'],flush=True)
                        last_heartbeat=now
                    time.sleep(min(2.,max(0.,seconds-(time.monotonic()-start))))
                completed = True
            finally:
                measured = 0. if start is None else round(time.monotonic()-start,1)
                session.close()
                session = None
                path = report_writer(folder)
                report = json.loads(path.read_text())
                entry = dict(segment=number, report=str(path), duration_completed=completed,
                    measured_seconds=measured,
                    output_rate=report.get('sender_output_rate'),
                    distinct_lidar_rate=report.get('distinct_selected_lidar_output_rate'),
                    issues=findings(report), accepted_outputs=report.get('accepted_outputs',0))
                status['segments'].append(entry)
                status['completed_segments'] += int(completed)
                save(output/'status.json',status)
                print('[계측 보고서] '+str(path),flush=True)
        status['status'] = 'COMPLETE'
    except KeyboardInterrupt:
        status.update(status='INTERRUPTED', reason='사용자 종료 또는 ROS 종료')
        exit_code = 130
    except Exception as exc:
        status.update(status='FAILED', reason=str(exc))
        exit_code = 1
    finally:
        if session is not None:
            session.close()
        status.update(phase='finished', updated_unix=time.time(),
            segments_with_issues=sum(bool(x['issues']) for x in status['segments']),
            performance_passed_automatically=False,
            note='COMPLETE는 계측 종료입니다. 정지 장면의 입력 안정성 검사이며 주행 성능을 증명하지 않습니다.')
        save(output/'status.json',status)
        save(output/'campaign_report.json',status)
        print('[최종 보고서] '+str(output/'campaign_report.json'),flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--overnight', action='store_true', help='기본 30분씩 8회, 약 4시간')
    parser.add_argument('--background', action='store_true', help='터미널과 분리해 실행하고 로그 경로 출력')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', choices=('E3', 'E4'), default='E3')
    args = parser.parse_args()
    seconds = args.seconds if args.seconds is not None else (1800. if args.overnight else 60.)
    repeat = args.repeat if args.repeat is not None else (8 if args.overnight else 1)
    if not math.isfinite(seconds) or seconds <= 0 or not 1 <= repeat <= 48:
        parser.error('seconds must be finite and positive; repeat must be between 1 and 48')
    output = (args.output or REPO/'MoraiDrive/scenario_runner/.runtime/ve_objects'/('hz_'+str(time.time_ns()))).resolve()
    if (output/'status.json').exists():
        parser.error('기존 계측 결과가 있습니다. 새 출력 경로를 사용하세요.')
    output.mkdir(parents=True,exist_ok=True)
    if args.background:
        command = [sys.executable,'-u','-B',str(Path(__file__).resolve()),'--seconds',str(seconds),
            '--repeat',str(repeat),'--checkpoint',args.checkpoint,'--output',str(output)]
        with (output/'console.log').open('x') as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True)
        save(output/'launch.json',dict(pid=child.pid,command=command,created_unix=time.time()))
        print('[백그라운드 실행 요청] PID={}; 정상 센서 수신 여부는 status.json에서 확인'.format(child.pid),flush=True)
        print('[로그] '+str(output/'console.log'),flush=True)
        print('[진행 상태] '+str(output/'status.json'),flush=True)
        print('[최종 보고서] '+str(output/'campaign_report.json'),flush=True)
        try:
            return wait_background_start(output, child)
        except KeyboardInterrupt:
            child.send_signal(signal.SIGINT)
            return 130
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    from MoraiDrive.expert.tools.run_ve_udp_expert import require_ros_master, capture_guard
    from MoraiDrive.expert.tools.ve_udp_session import PerceptionSession
    from inference.ve_timing_audit import write_report
    import rospy
    def check_topics(own_publisher=False):
        code,reason,state=rospy.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(reason)
        pubs = dict(state[0])
        allowed = [rospy.get_name()] if own_publisher else []
        if pubs.get('/ctrl_cmd') or any(node not in allowed for node in pubs.get('/ve/Object_topic',[])):
            raise RuntimeError('기존 Expert/VE를 종료하세요. 차량 제어 없는 계측만 실행합니다.')
    try:
        require_ros_master()
        rospy.init_node('ve_hz_check', anonymous=True, disable_signals=True)
        capture_guard()
        check_topics()
    except Exception as exc:
        failed=dict(status='FAILED',phase='preflight',reason=str(exc),updated_unix=time.time())
        save(output/'status.json',failed)
        save(output/'campaign_report.json',failed)
        print('[실패] '+str(exc),flush=True)
        return 1
    files = ['VLDrive/inference/sensors/morai_udp.py','VLDrive/inference/sensors/live_frame.py',
             'VLDrive/inference/ve_objects_udp.py','VLDrive/inference/ve_timing_audit.py']
    save(output/'code_identity.json',{p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files})
    print('[계측] 차량 제어 없이 센서→VE→UDP→ROS 발행을 측정합니다.', flush=True)
    print('[로그] '+str(output), flush=True)
    print('[참고] Expert 콜백은 기존 시나리오 주행 시 별도로 자동 기록됩니다.', flush=True)
    return campaign(output,args.checkpoint,seconds,repeat,PerceptionSession,check_topics,
                    rospy.is_shutdown,write_report)


if __name__ == '__main__':
    raise SystemExit(main())
