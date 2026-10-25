#!/usr/bin/env python3
"""Manual driving: run P1 VE on the live sensors and record GT beside it. Never publishes vehicle control.

Drive the Ego yourself in MORAI. Press Ctrl+C when finished; the GT comparison and the speed analysis
(``analyze_ve_by_speed.py``) are produced from the recording.
"""
import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for entry in (REPO, REPO/'VLDrive', REPO/'MoraiDrive', REPO/'MoraiDrive/data_collection/internal_sensor_record'):
    sys.path.insert(0, str(entry))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=('E3', 'E4', 'P1'), default='P1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=0., help='0 = until Ctrl+C')
    parser.add_argument('--alongside-gt-expert', action='store_true',
                        help='Observe only while a GT-driven Expert (scenario runner without --ve-objects) drives: '
                             'allows an existing /ctrl_cmd publisher. VE output is never given to that Expert.')
    args = parser.parse_args()
    output = (args.output or REPO/'MoraiDrive/scenario_runner/.runtime/ve_objects'/('manual_'+str(time.time_ns()))).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output directory is not empty: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)

    stop = {'requested': False}
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__('requested', True))

    from MoraiDrive.scenario_runner.legacy_expert.run_ve_udp_expert import require_ros_master, capture_guard, stop_child
    from MoraiDrive.scenario_runner.legacy_expert.ve_udp_session import PerceptionSession, ANALYSIS_PYTHON, build_analysis_environment
    from inference.ve_timing_audit import write_report
    import rospy
    require_ros_master()
    rospy.init_node('ve_manual_record', anonymous=True, disable_signals=True)
    if args.alongside_gt_expert:
        # A GT-driven scenario runner is expected; the sensor-record collectors (data collection) still block this tool.
        _collector_guard()
    else:
        capture_guard()
    pubs = dict(rospy.get_master().getSystemState()[2][0])
    if pubs.get('/ve/Object_topic') or (pubs.get('/ctrl_cmd') and not args.alongside_gt_expert):
        raise SystemExit('기존 Expert/VE 실행이 남아 있습니다. 먼저 종료하세요 (rosnode list로 확인). '
                         'GT Expert 옆에서 관찰만 하려면 --alongside-gt-expert')

    session = PerceptionSession(output, args.checkpoint, observe_gt=False)
    observer = log = None
    try:
        session.start()
        session.await_ready()
        host, port = _grpc_target()
        log = (output/'gt_observer.log').open('w', buffering=1)
        observer = subprocess.Popen([sys.executable, '-u', '-B', str(REPO/'MoraiDrive/scenario_runner/legacy_expert/ve_gt_comparison.py'),
                                     '--record', str(output), '--geometry-target', '{}:{}'.format(host, port)],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+15.
        while not (output/'gt_observer_ready.json').is_file():
            if observer.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('GT 기록기 시작 실패: ' + str(output/'gt_observer.log'))
            time.sleep(.1)
        print('[준비 완료] 직접 주행하세요. 끝나면 Ctrl+C.  기록 폴더: ' + str(output), flush=True)
        start = last = time.monotonic()
        while not stop['requested'] and (not args.seconds or time.monotonic()-start < args.seconds):
            if session.error or session.sender.poll() is not None:
                session.require_ready()
            if time.monotonic()-last >= 10.:
                print('[VE Hz] ' + session.performance_line(), flush=True)
                last = time.monotonic()
            time.sleep(.2)
    finally:
        print('[종료] 기록을 정리하고 분석합니다...', flush=True)
        try:
            session.close()
        finally:
            if observer is not None:
                stop_child(observer)
            if log is not None:
                log.close()
        try:
            synthesize_expert_input(output)
            env = build_analysis_environment(ANALYSIS_PYTHON)
            for tool in ('ve_gt_comparison.py', 've_gt_geometry_audit.py'):
                args_ = ['--analyze', str(output)] if tool == 've_gt_comparison.py' else [str(output)]
                subprocess.run([str(ANALYSIS_PYTHON), '-B', str(REPO/'MoraiDrive/scenario_runner/legacy_expert'/tool), *args_],
                               env=env, timeout=300, stdout=subprocess.DEVNULL)
            write_report(output)
        except Exception as exc:
            print('[분석 실패] 원본 기록은 보존됨: ' + str(exc), flush=True)
        print('[기록 폴더] ' + str(output), flush=True)
        print('[속도별 분석] python3 MoraiDrive/scenario_runner/legacy_expert/analyze_ve_by_speed.py ' + str(output), flush=True)


def _collector_guard():
    tokens = ('collect_scenario_laps.sh', 'run_sensor_record_scenario.py')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if any(token in command for token in tokens):
            raise SystemExit('센서 수집기가 실행 중입니다. 수집과 동시에 VE를 돌리지 않습니다: ' + command[:100])


def synthesize_expert_input(output):
    """No Expert listens in manual driving. The existing analysis scores only packets an Expert received,
    so mark every published VE packet as received (one row per packet, flagged synthetic)."""
    rows = []
    for line in (output/'adapter.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        packet = json.loads(line)
        audit = packet.get('delivery_audit') or {}
        if 'ros_stamp_ns' not in audit:
            continue
        rows.append({'received_ns': audit.get('published_ns'), 'completed_ns': audit.get('published_ns'),
                     'ros_sequence': packet.get('sequence'), 'ros_stamp_ns': audit['ros_stamp_ns'],
                     'objects': len(packet.get('boxes') or []), 'synthetic_manual_drive': True})
    (output/'expert_input.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def _grpc_target():
    import os
    target = os.environ.get('MORAI_GRPC_TARGET', '').strip()
    if target:
        host, port = target.rsplit(':', 1)
        return host, port
    host = os.environ.get('MORAI_GRPC_HOST', 'auto')
    if host == 'auto':
        from MoraiDrive.scenario_runner.adapters.morai_grpc import _windows_host_gateway
        host = _windows_host_gateway()
    return host, os.environ.get('MORAI_GRPC_PORT', '7789')


if __name__ == '__main__':
    main()
