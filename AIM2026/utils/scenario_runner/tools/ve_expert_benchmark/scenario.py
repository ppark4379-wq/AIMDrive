"""Existing scenario lifecycle with one Expert and selectable object input."""
import hashlib
import json
import time
from pathlib import Path

from ...scenarios.segment_modes.scenario import MissionScenario
from ...scenarios.segment_modes import tunnel_static_startup as stopping
from ...utils.transform_utils import project_distance_on_polyline
from .runtime import write


def configure_expert(document):
    document['common']['perception_source']='ros'
    document['common']['sync_grpc_inputs']['enabled']=False
    # Do not let a previous measured scene insert GT footprints into VE planning.
    for key in ('jamming_scene_geometry','jamming_trial_contract','jamming_trial_diagnostics_path'):
        document['planning'].pop(key,None)
    return document


class EvaluationScenario(MissionScenario):
    def __init__(self,*args,session,spec,output,**kwargs):
        super().__init__(*args,**kwargs)
        self.session,self.spec,self.output=session,spec,output
        self.rows=[]
        self.goal_reached=False
        self.config_hash=None
        self.initial_scene=None
        self.last_sample=self.last_topology=-1.
        self.unhealthy_since=None
        self.trace=(output/'observations.jsonl').open('w',buffering=1)
        self.stopped_verified=False
        self.topology_checks=0

    def activate_expert_drive(self, **kwargs):
        original_hook=kwargs.get('after_expert_start')

        def started():
            if original_hook is not None:
                original_hook()
            deadline=time.monotonic()+15.
            while time.monotonic()<deadline:
                if not self.expert_controller.is_running():
                    raise RuntimeError('평가 Expert가 시작 중 종료되었습니다')
                if self.verify_topology(allow_wait=True):
                    return
                time.sleep(.05)
            raise RuntimeError('Expert 제어/장애물 토픽 연결 시간 초과')

        kwargs['after_expert_start']=started
        return super().activate_expert_drive(**kwargs)

    def setup_expert_controller(self):
        super().setup_expert_controller()
        path=Path(self.expert_controller.runtime_config_path)
        config=configure_expert(json.loads(path.read_text()))
        write(path,config)
        self.config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        # Model is already warm before setup; wait for fresh post-teleport inputs.
        stopping.hold_common_brake(self.grpc)
        try:
            since=time.monotonic()
            self.session.reset_tracks()
            data=self.session.await_ready(timeout=30.,since=since)
            self.initial_scene=dict(ego=[data['ego']['x'],data['ego']['y']],objects=data['gt'])
            write(self.output/'initial_scene.json',self.initial_scene)
        finally:
            stopping.release_common_brake(self.grpc)

    def record_ego_trace(self,elapsed,ego_state):
        now=time.monotonic()
        if now-self.last_sample < .1 and not ego_state.get('collision_objects'):
            return
        self.last_sample=now
        if self.session.sender.poll() is not None or self.session.error:
            raise RuntimeError('인지 프로세스 중단: '+str(self.session.error))
        data=self.session.snapshot()
        if not (data['gt_fresh'] and data['ve_fresh'] and data['ego_age_s']<=.5):
            self.unhealthy_since=now if self.unhealthy_since is None else self.unhealthy_since
            if now-self.unhealthy_since>1.:
                raise RuntimeError('GT/VE/Ego 평가 입력이 1초 이상 끊김')
        else:
            self.unhealthy_since=None
        progress=project_distance_on_polyline(self.route_points,ego_state['x'],ego_state['y'])
        row=dict(t=elapsed,ego=ego_state,progress_s=progress,remaining_s=self.route_length_m-progress,
                 gt=data['gt'],ve=data['ve'],gt_fresh=data['gt_fresh'],ve_fresh=data['ve_fresh'])
        self.rows.append(row)
        self.trace.write(json.dumps(row,allow_nan=False)+'\n')
        if now-self.last_topology>1.:
            self.verify_topology()
            self.last_topology=now

    def verify_topology(self, allow_wait=False):
        ros=self.session.ros
        code,reason,state=ros.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(reason)
        pubs,subs=dict(state[0]),dict(state[1])
        prefix='/morai_standard_'+str(self.expert_controller.process.pid)+'_'
        owners=pubs.get('/ctrl_cmd',[])
        if not owners and allow_wait:
            return False
        if not owners or any(not n.startswith(prefix) for n in owners):
            raise RuntimeError('평가 Expert 외 제어 노드 감지: '+repr(owners))
        selected='/ve/Object_topic' if self.spec['arm']=='VE' else '/Object_topic'
        if not any(n.startswith(prefix) for n in subs.get(selected,[])):
            if allow_wait:
                return False
            raise RuntimeError('Expert 장애물 입력 연결 불일치: '+selected)
        if self.spec['arm']=='VE' and any(n.startswith(prefix) for n in subs.get('/Object_topic',[])):
            raise RuntimeError('VE Expert가 GT 장애물을 구독함')
        if pubs.get('/ve/Object_topic') != [ros.get_name()]:
            raise RuntimeError('VE 객체 토픽 발행자가 중복됨')
        self.topology_checks+=1
        return True

    def on_goal_reached_before_stop(self,**kwargs):
        super().on_goal_reached_before_stop(**kwargs)
        self.goal_reached=True

    def _wait_stopped(self):
        deadline=time.monotonic()+15.
        stable=None
        while time.monotonic()<deadline:
            state=self.grpc.get_ego_motion_state()
            if state['speed_mps']<.15:
                stable=time.monotonic() if stable is None else stable
                if time.monotonic()-stable>.5:
                    return
            else:
                stable=None
            time.sleep(.1)
        raise RuntimeError('회차 종료 후 실제 정지 확인 실패')

    def cleanup(self):
        has_world=getattr(self.grpc,'world',None) is not None
        try:
            # Stop control and brake BEFORE actor teardown, which can take
            # seconds. The last driving throttle must not remain active.
            self.stop_expert_controller()
            if has_world:
                stopping.hold_common_brake(self.grpc)
                self._wait_stopped()
            super().cleanup()
            if has_world:
                stopping.release_common_brake(self.grpc)
                stopping._require_no_controller()
                self._wait_stopped()
                self.stopped_verified=True
        finally:
            stopping.release_common_brake(self.grpc)
            self.trace.close()
