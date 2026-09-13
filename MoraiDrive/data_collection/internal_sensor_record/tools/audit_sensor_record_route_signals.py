#!/usr/bin/env python3
"""Bind exact-route stop lines to direct gRPC light observations at sensor time."""
import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from async_capture.journal import digest,write_json
from async_capture.route_signal_binding import bind_route_progress
from data_collection.mgeo_navigation import MGeoMap,MGeoRouteTracker,load_official_route
from ssd_storage import require_ssd


def ego_at(rows,stamp,max_span_ns=30_000_000):
    times=[row['source_timestamp_ns'] for row in rows];i=bisect_left(times,stamp)
    if i<len(rows) and times[i]==stamp:return dict(rows[i]['signal'],record_ids=[rows[i]['record_id']])
    if i==0 or i==len(rows) or times[i]-times[i-1]>max_span_ns:return None
    a,b=rows[i-1],rows[i];f=(stamp-times[i-1])/(times[i]-times[i-1])
    pa,pb=a['signal']['position'],b['signal']['position'];ya,yb=a['signal']['yaw'],b['signal']['yaw']
    dy=(yb-ya+math.pi)%(2*math.pi)-math.pi
    return dict(position=[(1-f)*pa[k]+f*pb[k] for k in range(3)],yaw=ya+f*dy,
                speed=(1-f)*a['signal']['speed']+f*b['signal']['speed'],
                record_ids=[a['record_id'],b['record_id']])


def verify(folder,seal_name):
    seal=json.loads((folder/seal_name).read_text())
    for name,sha in seal['files'].items():
        if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError('Changed sealed evidence')
    return seal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    scenario=ROOT.parents[1]/'scenario_runner'
    p.add_argument('--map-root',type=Path,default=scenario/'data/R_KR_PR_K-city_2025')
    p.add_argument('--route-definition',type=Path,default=scenario/'config/full_loop_links.yaml')
    p.add_argument('--route-key',default='full_loop')
    p.add_argument('--actor-audit-dir',default='actor_inventory_audit_v1')
    a=p.parse_args()
    for path in (a.episode,a.output):require_ssd(path,reserve_bytes=0)
    qseal=verify(a.episode/'quality','QUALITY_COMPLETE.json');tseal=verify(a.episode/'gt_timeline_v1','COMPLETE.json')
    traffic_seal=verify(a.episode/'traffic_audit_v1','COMPLETE.json')
    actor_dir=a.episode/a.actor_audit_dir
    actor_seal=verify(actor_dir,'COMPLETE.json')
    index=json.loads((a.episode/'quality/INDEX_COMPLETE.json').read_text())
    traffic=json.loads((a.episode/'traffic_audit_v1/report.json').read_text())
    if traffic['raw_manifest_sha256']!=index['raw_manifest_sha256'] or not traffic['structural_valid']:
        raise ValueError('Invalid or mixed direct traffic evidence')
    contract=json.loads((a.episode.parent/'sensor_contract.json').read_text())
    lights=contract['capture']['traffic_light_ground_truth']['light_ids']
    world=MGeoMap(str(a.map_root/'link_set.json'),node_set_path=str(a.map_root/'node_set.json'))
    route=load_official_route(str(a.route_definition),str(a.map_root/'2026_molit_comp_global_path.txt'),a.route_key)
    tracker=MGeoRouteTracker(world,route)
    active=json.loads((a.episode/'quality/active_signals.json').read_text())
    egos=sorted((row for row in active if row['role']=='ego'),key=lambda row:(row['source_timestamp_ns'],row['record_id']))
    frames=json.loads((a.episode/'gt_timeline_v1/frames.json').read_text());output=[]
    actor_frames=json.loads((actor_dir/'frame_brackets.json').read_text())
    if len(actor_frames)!=len(frames):raise ValueError('Actor frame brackets differ from sensor frames')
    last_route=-1
    for frame,actor_frame in zip(frames,actor_frames):
        ego=ego_at(egos,frame['lidar_source_ns']);reasons=[]
        if ego is None:
            row=dict(index=frame['index'],sensor_source_ns=frame['lidar_source_ns'],valid=False,
                     training_label=-1,reason='ego_not_source_bracketed')
        else:
            progress=tracker.update(ego['position'][0],ego['position'][1],ego['yaw'])
            row=bind_route_progress(progress,tracker._point_heading(progress.route_index),ego['yaw'],
                frame['traffic_light_gt'].get('values',[]),lights)
            rpc_links={actor_frame.get('ego_current_link_before'),actor_frame.get('ego_current_link_after')}
            rpc_links.discard(None)
            agreement=(len(rpc_links)==1 and progress.current_link_id in rpc_links)
            row.update(index=frame['index'],sensor_source_ns=frame['lidar_source_ns'],
                       ego_source_record_ids=ego['record_ids'],rpc_ego_current_links=sorted(rpc_links),
                       route_rpc_current_link_agreement=agreement)
            if row['valid'] and not agreement:reasons.append('route_and_rpc_current_link_disagree')
            if progress.route_index<last_route:reasons.append('route_progress_reversed')
            last_route=max(last_route,progress.route_index)
        if reasons:
            row['valid']=False;row['training_label']=-1;row['reason']='+'.join(reasons)
        row['training_visible']=False;output.append(row)
    statuses=Counter(row['reason'] or ('red' if row['training_label']==0 else 'green') for row in output)
    candidates=[row for row in output if row['valid']]
    target=[row for row in output if row.get('light_id') and row.get('distance_to_stopline_m') is not None
            and row['distance_to_stopline_m']<=45]
    report=dict(schema='morai_sensor_record_route_signal_binding_audit_v1',frames=len(output),
        exact_route_source=route['route_source'],route_definition_sha256=digest(a.route_definition),
        global_path_sha256=digest(a.map_root/'2026_molit_comp_global_path.txt'),
        route_light_ids=lights,status_counts=dict(statuses),stopline_lookahead_frames=len(target),
        bound_pure_color_frames=len(candidates),red_frames=sum(row['training_label']==0 for row in candidates),
        green_frames=sum(row['training_label']==1 for row in candidates),
        route_binding_evidence_present=bool(target),route_signal_training_label_candidate=bool(candidates),
        both_binary_classes_observed=all(any(row['training_label']==value for row in candidates) for value in (0,1)),
        route_rpc_current_link_agreement_frames=sum(row.get('route_rpc_current_link_agreement') is True for row in output),
        route_rpc_current_link_disagreement_frames=sum(row.get('route_rpc_current_link_agreement') is False for row in output),
        physical_sensor_gt_atomicity_verified=False,training_visible=False,
        limitation='Exact-route/source-time binding candidate. MORAI Sensor Record scene contract is separate from a physical millisecond exposure guarantee.',
        raw_manifest_sha256=index['raw_manifest_sha256'])
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'report.json',report);write_json(a.output/'frames.json',output)
    write_json(a.output/'COMPLETE.json',dict(schema=report['schema'],files={name:digest(a.output/name) for name in ('report.json','frames.json')},
        quality_seal_sha256=digest(a.episode/'quality/QUALITY_COMPLETE.json'),gt_timeline_seal_sha256=digest(a.episode/'gt_timeline_v1/COMPLETE.json'),
        traffic_audit_seal_sha256=digest(a.episode/'traffic_audit_v1/COMPLETE.json'),
        actor_inventory_seal_sha256=digest(actor_dir/'COMPLETE.json'),
        script_sha256=digest(Path(__file__)),training_visible=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
