"""Optional VE object input for the ordinary scenario runner's own Expert."""
import json
import subprocess
import sys
import time
from pathlib import Path

SUPPORTED = ('full_loop_roundabout', 'full_loop_highway', 'full_loop_traffic_light', 'full_loop')
OBJECT_REMAP = '/Object_topic:=/ve/Object_topic'


def configure(cfg, scenario, checkpoint):
    if scenario not in SUPPORTED or cfg.get('sync_runtime', {}).get('enabled', True):
        raise ValueError('--ve-objects requires roundabout/highway/traffic_light and --no-sync-runtime')
    cfg['ve_objects_checkpoint'] = checkpoint
    cfg['expert_ros_remaps'] = [OBJECT_REMAP]


def configure_expert(path, audit_path=None):
    document = json.loads(Path(path).read_text())
    document['common']['perception_source'] = 'ros'
    document['common']['sync_grpc_inputs']['enabled'] = False
    if audit_path is not None:
        document['common']['ve_input_audit_path'] = str(audit_path)
    for key in ('jamming_scene_geometry', 'jamming_trial_contract', 'jamming_trial_diagnostics_path'):
        document['planning'].pop(key, None)
    Path(path).write_text(json.dumps(document, indent=2)+'\n')


def verify_topology(state, own_node, expert_pid=None, allow_starting=False):
    pubs, subs = dict(state[0]), dict(state[1])
    if pubs.get('/ve/Object_topic') != [own_node]:
        raise RuntimeError('VE 장애물 토픽 발행자가 중복되었거나 종료되었습니다')
    prefix = '/morai_standard_{}_'.format(expert_pid) if expert_pid is not None else None
    owners = pubs.get('/ctrl_cmd', [])
    if owners and (prefix is None or any(not n.startswith(prefix) for n in owners)):
        raise RuntimeError('다른 Expert/제어 노드가 실행 중입니다: '+repr(owners))
    expert_nodes = {n for nodes in subs.values() for n in nodes if prefix and n.startswith(prefix)}
    if expert_nodes.intersection(subs.get('/Object_topic', [])):
        raise RuntimeError('Expert에 GT 장애물 구독이 남아 있습니다')
    ready = bool(owners) and set(owners).issubset(subs.get('/ve/Object_topic', []))
    if expert_pid is not None and not ready and not allow_starting:
        raise RuntimeError('Expert의 VE 장애물 입력 연결이 끊겼습니다')
    return ready


class VeObjectInput:
    def __init__(self, context):
        from AIM2026.Expert.tools.ve_udp_session import REPO, PerceptionSession
        self.context = context
        self.output = REPO/'MoraiDrive/scenario_runner/.runtime/ve_objects'/str(time.time_ns())
        self.output.mkdir(parents=True)
        self.session = PerceptionSession(self.output, context.cfg['ve_objects_checkpoint'], observe_gt=False,
                                         signal_head=context.cfg.get('ve_signal_head'))
        self.signal = None
        self.unhealthy_since = None
        self.last_log = self.last_topology = 0.
        self.driving_started = False
        self.gt_observer = self.gt_log = None

    def start(self):
        ros = self.session.ros
        if not ros.core.is_initialized():
            ros.init_node('scenario_ve_objects', anonymous=True, disable_signals=True)
        code, reason, state = ros.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(reason)
        pubs = dict(state[0])
        if pubs.get('/ctrl_cmd') or pubs.get('/ve/Object_topic') or pubs.get('/ve/GetTrafficLightStatus'):
            raise RuntimeError('기존 Expert/VE 실행을 Ctrl+C로 종료한 뒤 실행하세요')
        if any(ros.get_param(k, False) for k in ('/morai_drive/sync_runtime_active', '/morai_drive/data_collector_ready')):
            raise RuntimeError('센서 수집/Sync Runtime이 실행 중입니다')
        print('[VE] 모델 준비 중. 로그: '+str(self.output/'publisher.log'), flush=True)
        if self.context.cfg.get('ve_signal_head'):
            from .ve_signal import SignalPublisher
            self.signal = SignalPublisher(self.context, self.session, self.output)
            self.session.signal_sink = self.signal
        self.session.start()
        self.session.await_ready()
        if self.context.cfg.get('ve_compare_gt'):
            self.start_gt_observer()
        print('[VE] 준비 완료. 기존 러너의 NPC·경로 사용; Expert 장애물 입력=/ve/Object_topic', flush=True)

    def start_gt_observer(self):
        from AIM2026.Expert.tools.ve_udp_session import REPO
        script = REPO/'AIM2026/Expert/tools/ve_gt_comparison.py'
        self.gt_log = (self.output/'gt_observer.log').open('w', buffering=1)
        self.gt_observer = subprocess.Popen([sys.executable, '-u', '-B', str(script),
            '--record', str(self.output), '--geometry-target', '{}:{}'.format(
                self.context.grpc.host, self.context.grpc.port)], stdout=self.gt_log, stderr=subprocess.STDOUT,
            start_new_session=True)
        deadline = time.monotonic()+10.
        while time.monotonic() < deadline:
            if self.gt_observer.poll() is not None:
                raise RuntimeError('GT 비교 기록기 시작 실패: '+str(self.output/'gt_observer.log'))
            if (self.output/'gt_observer_ready.json').is_file():
                print('[GT 비교] 별도 프로세스가 /Object_topic을 기록합니다. Expert 입력은 VE만 사용.', flush=True)
                return
            time.sleep(.05)
        raise RuntimeError('GT 비교 기록기 시작 시간 초과')

    def finish_gt_comparison(self):
        from AIM2026.Expert.tools.ve_udp_session import (REPO, ANALYSIS_PYTHON,
                                                           build_analysis_environment, stop_child)
        if self.gt_observer is None:
            return
        stop_child(self.gt_observer)
        self.gt_log.close()
        try:
            result = subprocess.run([str(ANALYSIS_PYTHON), '-B',
                str(REPO/'AIM2026/Expert/tools/ve_gt_comparison.py'), '--analyze', str(self.output)],
                env=build_analysis_environment(ANALYSIS_PYTHON), timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print('[GT 비교] 분석 실행 실패. 원본 기록 보존: '+str(exc), flush=True)
            return
        if result.returncode:
            print('[GT 비교] 분석 실패. gt_objects.jsonl과 adapter.jsonl은 보존했습니다.', flush=True)
        else:
            print('[GT 비교 보고서] '+str(self.output/'gt_comparison_report.json'), flush=True)
            if (self.output/'gt_geometry.jsonl').is_file():
                try:
                    audit = subprocess.run([str(ANALYSIS_PYTHON), '-B',
                        str(REPO/'AIM2026/Expert/tools/ve_gt_geometry_audit.py'), str(self.output)],
                        env=build_analysis_environment(ANALYSIS_PYTHON), timeout=120,
                        stdout=subprocess.DEVNULL)
                    if audit.returncode:
                        print('[GT 기준점 감사] 분석 실패. 원본 기록은 보존했습니다.', flush=True)
                    else:
                        print('[GT 기준점 감사] '+str(self.output/'gt_geometry_audit_report.json'), flush=True)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    print('[GT 기준점 감사] 실행 실패: '+str(exc), flush=True)

    def before_drive(self):
        from ..scenarios.segment_modes import tunnel_static_startup as stopping
        if self.signal is not None:
            self.signal.validate_route()
        configure_expert(self.context.expert_controller.runtime_config_path,
                         self.output/'expert_input.jsonl')
        stopping.hold_common_brake(self.context.grpc)
        try:
            since = time.monotonic()
            self.session.reset_tracks()
            self.session.await_ready(timeout=30., since=since)
        finally:
            stopping.release_common_brake(self.context.grpc)

    def after_start(self):
        self.driving_started = True
        deadline = time.monotonic()+15.
        while time.monotonic() < deadline:
            if self.check_topology(allow_starting=True):
                print('[PASS] Expert → VE 장애물 입력 연결 확인. GT 장애물 구독 없음.', flush=True)
                if self.signal is not None:
                    print('[PASS] Expert 신호 입력=/ve/GetTrafficLightStatus. GT 신호 구독 없음.', flush=True)
                return
            time.sleep(.05)
        raise RuntimeError('Expert의 VE 토픽 연결 시간 초과')

    def check_topology(self, allow_starting=False):
        ros = self.session.ros
        controller = self.context.expert_controller
        if not controller.is_running():
            raise RuntimeError('Expert 프로세스가 종료되었습니다')
        code, reason, state = ros.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(reason)
        ready = verify_topology(state, ros.get_name(), controller.process.pid, allow_starting)
        if self.signal is not None:
            from .ve_signal import verify_signal_topology
            ready = verify_signal_topology(state, ros.get_name(), controller.process.pid, allow_starting) and ready
        return ready

    def inspect(self):
        now = time.monotonic()
        if self.session.error or self.session.sender.poll() is not None:
            self.session.require_ready()
        data = self.session.snapshot()
        ready = data['ve_fresh'] and data['ego_age_s'] <= .5
        if ready:
            self.unhealthy_since = None
        else:
            self.unhealthy_since = now if self.unhealthy_since is None else self.unhealthy_since
            if now-self.unhealthy_since > 2.:
                raise RuntimeError('VE/자차 입력이 2초 이상 끊겼습니다: '+str(self.output/'publisher.log'))
        if now-self.last_topology > 1.:
            self.check_topology()
            self.last_topology = now
        if now-self.last_log > 2.:
            print('[VE] 입력={} 객체={} 이동={} | Expert=/ve/Object_topic'.format(
                '정상' if ready else '대기', len(data['ve']) if ready else '?',
                sum(b['motion_state'] == 'moving' for b in data['ve']) if ready else '?'), flush=True)
            print('[VE Hz] '+self.session.performance_line(), flush=True)
            self.last_log = now

    def close(self):
        # The runner stops its Expert before reaching this hook.
        from ..scenarios.segment_modes import tunnel_static_startup as stopping
        try:
            if self.driving_started:
                stopping.hold_common_brake(self.context.grpc)
                deadline = time.monotonic()+5.
                while time.monotonic() < deadline:
                    time.sleep(.1)
                    if self.context.grpc.get_ego_motion_state()['speed_mps'] < .15:
                        break
        finally:
            try:
                stopping.release_common_brake(self.context.grpc)
            finally:
                try:
                    self.session.close()
                    if self.signal is not None:
                        self.signal.close()
                finally:
                    self.finish_gt_comparison()
                from inference.ve_timing_audit import write_report
                try:
                    path = write_report(self.output)
                    print('[VE 수신·추론 계측] '+str(path), flush=True)
                except (OSError, ValueError) as exc:
                    print('[VE 계측 보고서 생성 실패] '+str(exc), flush=True)
