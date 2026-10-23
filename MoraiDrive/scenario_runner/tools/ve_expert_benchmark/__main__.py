"""Offline prepare by default; --run performs the paired MORAI trials."""
import argparse
import copy
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .runtime import REPO, PerceptionSession, write
from .metrics import score, compare_pair
from inference.realtime_ve_objects import CHECKPOINTS
from AIM2026.Expert.tools.run_ve_udp_expert import capture_guard, require_ros_master

CASES={
    'empty':dict(mission_mode='full_loop_tunnel_normal'),
    'slow_lead':dict(mission_mode='full_loop_tunnel_vehicle',behavior='vehicle_slow'),
    'stop_restart':dict(mission_mode='full_loop_tunnel_vehicle',behavior='vehicle_sudden_brake'),
    'roundabout':dict(mission_mode='full_loop_roundabout'),
    'static':dict(mission_mode='full_loop_static'),
    'oncoming':dict(mission_mode='full_loop_tunnel_vehicle',behavior='vehicle_oncoming'),
    'cut_in':dict(mission_mode='full_loop_tunnel_vehicle',behavior='vehicle_cut_in'),
}
DEFAULT_CASES=['empty','slow_lead','stop_restart','roundabout','static']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_sources(spec):
    for relative, expected in spec['source_sha256'].items():
        path=(REPO/relative).resolve()
        if REPO not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('평가 준비 후 코드/설정이 변경되었습니다: '+relative)


def trial_config(base,spec,output):
    cfg=copy.deepcopy(base)
    cfg.update(mission_mode=CASES[spec['case']]['mission_mode'],random_seed=spec['seed'],
        auto_repeat=False,max_laps=1,clean_road_mode=False,timeout_sec=180.,route_setup_attempts=1,
        forced_weather=spec['weather'],forced_time_hour=spec['hour'],route_debug_enabled=False,
        reset_ego_motion_on_start=True,expert_python_executable='/usr/bin/python3',
        profile_state_path=str(output/'profile.json'),expert_is_closed_path=False,
        expert_ros_remaps=['/Object_topic:=/ve/Object_topic'] if spec['arm']=='VE' else [],
        progress_watchdog=dict(enabled=True,timeout_sec=45.,startup_grace_sec=30.,minimum_progress_m=2.),
        off_route_timeout_sec=3.,off_route_grace_sec=3.)
    cfg['sync_runtime']['enabled']=False
    if 'behavior' in CASES[spec['case']]:
        cfg['tunnel_vehicle_dynamic'].update(enabled=True,behavior=CASES[spec['case']]['behavior'],
                                             random_behavior=False,additional_behaviors=[])
    return cfg


def prepare(args,output):
    from ...main_runner import load_configuration, MGeoMapLoader, PACKAGE_ROOT
    from ..verify_separation import verify_configuration, verify_route
    from ..verify_segment_modes import verify_mission_mode
    global_cfg,_,base=load_configuration()
    checkpoint=CHECKPOINTS[args.checkpoint_label]
    if not Path(checkpoint['path']).is_file():
        raise FileNotFoundError(checkpoint['path'])
    source_files=[REPO/'VLDrive/inference/ve_object_tracking.py',REPO/'VLDrive/inference/ve_objects_udp.py',
                  REPO/'VLDrive/inference/realtime_ve_objects.py',
                  REPO/'AIM2026/Expert/src/expert/src/config/config.json',Path(base['route_definition_path']),
                  Path(base['global_path']),PACKAGE_ROOT/'config/full_loop.yaml',PACKAGE_ROOT/'config/global.yaml']
    if (PACKAGE_ROOT/'config/local.yaml').is_file():
        source_files.append(PACKAGE_ROOT/'config/local.yaml')
    source_files+=list(Path(__file__).parent.glob('*.py'))
    source_files+=[REPO/'AIM2026/Expert/tools/ve_udp_session.py',
                   REPO/'VLDrive/inference/sensors/live_frame.py']
    source_files+=list((REPO/'AIM2026/Expert/src/expert/src').rglob('*.py'))
    source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    specs=[]
    for repeat in range(args.repeats):
        for index,case in enumerate(args.cases):
            contract=dict(case=case,seed=args.seed+repeat*100+index,weather=args.weather,
                          hour=args.hour,checkpoint=checkpoint,source_sha256=source_hashes)
            # Alternate order across repetitions to avoid always warming GT first.
            for arm in (('GT','VE') if repeat%2==0 else ('VE','GT')):
                spec=dict(contract,arm=arm,repeat=repeat+1,checkpoint_label=args.checkpoint_label,
                          pair_contract_sha256=fingerprint(contract))
                specs.append(spec)
    with (output/'offline_check.log').open('w') as log:
        import contextlib
        with contextlib.redirect_stdout(log):
            for case in args.cases:
                spec=next(s for s in specs if s['case']==case)
                cfg=trial_config(base,spec,output/('offline_'+case))
                verify_configuration(PACKAGE_ROOT,global_cfg,cfg)
                verify_route(PACKAGE_ROOT,global_cfg,cfg)
                print(json.dumps(verify_mission_mode(cfg,cfg['mission_mode']),indent=2))
                if 'behavior' in CASES[case]:
                    import random
                    import yaml
                    from types import SimpleNamespace
                    from ...utils.transform_utils import polyline_length
                    from ...scenarios.full_loop_parts.tunnel_vehicle_dynamic import prepare_tunnel_vehicle
                    definition=yaml.safe_load(Path(cfg['route_definition_path']).read_text())[cfg['route_definition_key']]
                    points=[tuple(map(float,line.split()[:3])) for line in Path(cfg['global_path']).read_text().splitlines()
                            if line.strip() and not line.lstrip().startswith('#')]
                    zone=definition['zones']['tunnel']
                    checkpoints={str(c['id']):int(c['path_index']) for c in definition['checkpoints']}
                    context=SimpleNamespace(cfg=cfg,scenario_name=cfg['mission_mode'],clean_road_mode=False,
                        map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),route_points=points,
                        route_definition=definition,run_seed=spec['seed'],random=random.Random(spec['seed']),
                        tunnel_entry_s=polyline_length(points[:checkpoints[str(zone['entry_checkpoint'])]+1]),
                        tunnel_exit_s=polyline_length(points[:checkpoints[str(zone['exit_checkpoint'])]+1]))
                    prepare_tunnel_vehicle(context)
                    print(json.dumps(context.tunnel_vehicle_plan,indent=2))
                if cfg['sync_runtime']['enabled'] or cfg['max_laps']!=1 or cfg['timeout_sec']!=180.:
                    raise ValueError('Bounded benchmark lifecycle contract violated')
    plan=dict(schema='ve_expert_paired_benchmark_v1',status='PREPARED',specs=specs,
        trials=len(specs),scenario_drive_timeout_s=180,trial_wall_timeout_s=360,
        total_drive_budget_minutes=len(specs)*3,total_wall_budget_minutes=len(specs)*6,
        initial_scope=DEFAULT_CASES,optional_cases=['oncoming','cut_in'],
        deferred_cases=['dedicated adjacent same-direction vehicle pass'],
        model_loaded=False,training_updates=0,vehicle_commands_sent=0,
        note='Runtime model availability, sensor timing and scenario validity are checked during execution.')
    write(output/'plan.json',plan)
    return plan


def run_trial(spec_path):
    from ...main_runner import load_configuration, MoraiGrpcClient, MGeoMapLoader
    from .scenario import EvaluationScenario
    import rospy
    spec=json.loads(spec_path.read_text())
    output=spec_path.parent
    verify_sources(spec)
    capture_guard()
    require_ros_master()
    rospy.init_node('ve_benchmark_observer',anonymous=True,disable_signals=True)
    code,why,state=rospy.get_master().getSystemState()
    if code!=1:
        raise RuntimeError(why)
    pubs,subs=dict(state[0]),dict(state[1])
    for topic in ('/ctrl_cmd','/ve/Object_topic'):
        if pubs.get(topic):
            raise RuntimeError('기존 실행을 Ctrl+C로 종료하세요: '+topic+' '+repr(pubs[topic]))
    if not subs.get('/ctrl_cmd'):
        raise RuntimeError('MORAI /ctrl_cmd 제어 연결이 없습니다')
    if any(rospy.get_param(k,False) for k in ('/morai_drive/sync_runtime_active','/morai_drive/data_collector_ready')):
        raise RuntimeError('수집/Sync Runtime이 실행 중입니다')
    def interrupt(*_):
        raise KeyboardInterrupt('자동 평가 중단 요청')
    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)
    global_cfg,_,base=load_configuration()
    global_cfg['paths']['runtime_root']=str(output/'runtime')
    cfg=trial_config(base,spec,output)
    write(output/'scenario_config.json',cfg)
    session=PerceptionSession(output,spec['checkpoint_label'])
    scenario=None
    client=None
    error=None
    started=time.monotonic()
    try:
        # Warm identical VE perception in BOTH arms before any NPC spawn.
        # Only the VE arm remaps the Expert object input to its output.
        session.start()
        session.await_ready()
        client=MoraiGrpcClient(global_cfg)
        client.connect()
        scenario=EvaluationScenario(grpc_client=client,map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),
            global_cfg=global_cfg,scenario_cfg=cfg,session=session,spec=spec,output=output)
        scenario.run()
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        print('[회차 종료] '+error,flush=True)
    finally:
        try:
            session.close()
        except Exception as exc:
            error=(error+'; ' if error else '')+'Perception cleanup: '+str(exc)
            if scenario is not None:
                scenario.stopped_verified=False
        finally:
            if client is not None:
                try:
                    client.stop()
                except Exception as exc:
                    error=(error+'; ' if error else '')+'MORAI cleanup: '+str(exc)
                    if scenario is not None:
                        scenario.stopped_verified=False
    rows=scenario.rows if scenario else []
    metrics=score(rows,spec['case'],bool(scenario and scenario.goal_reached),error)
    event=getattr(scenario,'tunnel_vehicle_result',None)
    if event is not None:
        metrics['scenario_event_completed']=bool(event.get('interaction_complete') and event.get('observed_moving'))
        metrics['basic_trial_passed'] &= metrics['scenario_event_completed']
        write(output/'vehicle_event.json',event)
    baseline=getattr(scenario,'tunnel_baseline_report',None)
    if baseline is not None and not baseline.get('final_stop_verified',False):
        scenario.stopped_verified=False
    report=dict(schema='ve_expert_trial_v1',status='COMPLETE' if error is None else 'FAILED',spec=spec,
        pair_contract_sha256=spec['pair_contract_sha256'],expert_config_sha256=scenario.config_hash if scenario else None,
        initial_scene=scenario.initial_scene if scenario else None,metrics=metrics,
        stopped_verified=bool(scenario and scenario.stopped_verified),
        topology_checks=scenario.topology_checks if scenario else 0,
        GT_obstacles_used_for_VE_control=False,VE_waypoints_used=False,training_updates=0,
        source_synchrony_verified=False,elapsed_wall_s=time.monotonic()-started)
    write(output/'report.json',report)
    return 0 if report['stopped_verified'] and rows else 2


def render(output,reports,status):
    pairs={}
    for report in reports:
        spec=report['spec']
        pairs.setdefault((spec['case'],spec['repeat']),{})[spec['arm']]=report
    comparisons={case+'_'+str(repeat):compare_pair(arms['GT'],arms['VE'])
                 for (case,repeat),arms in pairs.items() if set(arms)=={'GT','VE'}}
    result=dict(schema='ve_expert_benchmark_result_v1',status=status,trials=len(reports),
        passed_trials=sum(r['metrics']['basic_trial_passed'] for r in reports),
        failed_or_unqualified_trials=sum(not r['metrics']['basic_trial_passed'] for r in reports),
        comparisons=comparisons,training_updates=0,automatic_safety_certification=False,
        reports=[str(Path(r['output'])/'report.json') for r in reports])
    write(output/'report.json',result)
    lines=[]
    for r in reports:
        m,s=r['metrics'],r['spec']
        lines.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (s['case'],s['repeat'],s['arm'],m['basic_trial_passed'],m['goal_reached'],m['collision_observed'],
             m['min_estimated_footprint_distance_m'],round(m['stopped_s'],1),m['center_match_2m_rate'],m['error']))+'</tr>')
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>VE / Expert 자동 평가</title>'
        '<style>body{font:16px sans-serif;margin:24px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style>'
        '<h1>GT 입력 / VE 입력 자동 주행 비교</h1><p>동일한 seed·날씨·경로 정책을 사용합니다. '
        '비동기 주행의 실제 NPC 궤적은 달라질 수 있습니다. 거리 값은 ROS 크기로 계산한 근사치이며, '
        '충돌 여부는 MORAI 충돌 관측입니다. 검출률은 BEV 내 GT 중심과 2m 이내 일대일 매칭 비율입니다.</p>'
        '<table><tr>'+''.join('<th>'+x+'</th>' for x in ('장면','반복','입력','기본 기준 통과','완주','충돌 관측',
            '최소 박스 거리(근사 m)','정지 s','중심 매칭 비율','오류'))+'</tr>'+''.join(lines)+'</table>',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='실제 MORAI 자동 주행 실행')
    parser.add_argument('--check',action='store_true',help='오프라인 구성 검사만 실행 (기본값)')
    parser.add_argument('--cases',nargs='+',choices=tuple(CASES),default=DEFAULT_CASES)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--seed',type=int,default=26101400)
    parser.add_argument('--weather',choices=('SUNNY','FOGGY'),default='SUNNY')
    parser.add_argument('--hour',type=int,choices=(11,13,15),default=15)
    parser.add_argument('--checkpoint-label',choices=('E3','E4'),default='E3')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--trial-spec',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.trial_spec:
        return run_trial(args.trial_spec.resolve())
    if not 1<=args.repeats<=20 or len(set(args.cases))!=len(args.cases) or (args.run and args.check):
        parser.error('repeats=1..20, 중복 없는 cases, --run/--check 중 하나를 사용하세요')
    runtime_root=REPO/'MoraiDrive/scenario_runner/.runtime/ve_expert_benchmark'
    root=args.output or runtime_root/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root=root.resolve()
    if runtime_root.resolve() not in root.parents:
        parser.error('--output은 scenario_runner/.runtime/ve_expert_benchmark 아래 새 폴더여야 합니다')
    root.mkdir(parents=True,exist_ok=False)
    plan=prepare(args,root)
    print(json.dumps(dict(status='PREPARED',plan=str(root/'plan.json'),trials=plan['trials']),ensure_ascii=False),flush=True)
    if not args.run:
        return 0
    reports=[]
    status='RUNNING'
    child=None
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for index,spec in enumerate(plan['specs']):
            folder=root/('%02d_%s_%s'%(index+1,spec['case'],spec['arm']))
            folder.mkdir()
            write(folder/'spec.json',spec)
            print('[평가 %d/%d] %s / %s · 로그: %s'%(index+1,len(plan['specs']),spec['case'],spec['arm'],folder/'console.log'),flush=True)
            with (folder/'console.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'-u','-B','-m','scenario_runner.tools.ve_expert_benchmark',
                                        '--trial-spec',str(folder/'spec.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    rc=child.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        # Do not silently continue with an orphaned vehicle controller.
                        status='BLOCKED_CHILD_CLEANUP'
                        raise RuntimeError('회차 정리가 끝나지 않았습니다. 다음 회차는 시작하지 않습니다: '+str(folder))
                    rc=2
            path=folder/'report.json'
            if path.is_file():
                r=json.loads(path.read_text())
                r['output']=str(folder)
                reports.append(r)
            else:
                reports.append(dict(spec=spec,output=str(folder),status='FAILED',
                    pair_contract_sha256=spec['pair_contract_sha256'],expert_config_sha256=None,
                    initial_scene=None,metrics=score([],spec['case'],False,'회차 보고서 없음: console.log 확인')))
            render(root,reports,'RUNNING')
            if rc!=0:
                status='BLOCKED'
                break
        else:
            status='COMPLETE'
    except KeyboardInterrupt:
        status='INTERRUPTED'
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                status='BLOCKED_CHILD_CLEANUP'
    except Exception as exc:
        if status!='BLOCKED_CHILD_CLEANUP':
            status='FAILED'
        print('[평가 중단] '+str(exc),flush=True)
    finally:
        result=render(root,reports,status)
    print(json.dumps(dict(status=status,report=str(root/'report.json'),review=str(root/'review.html'),
        passed_trials=result['passed_trials'],failed_or_unqualified_trials=result['failed_or_unqualified_trials']),ensure_ascii=False))
    return 0 if status=='COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
