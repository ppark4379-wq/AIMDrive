import json, collections, math, statistics
from pathlib import Path
BASE=Path(__file__).parent
NEW=sorted(Path('/mnt/c/AIMDrive/competition_train/data').glob('*19_48_00*'))[0]
def summarize(root, frames):
    frames=sorted(frames,key=lambda f:f['frame_id'])
    c=collections.Counter
    ticks=[f['native_sync_frame'] for f in frames]
    gaps=c(b-a for a,b in zip(ticks,ticks[1:]))
    inc=[json.loads(s) for s in (root/'incomplete_frames.jsonl').read_text().splitlines() if s.strip()]
    prov=[json.loads(s) for s in (root/'capture_provenance.jsonl').read_text().splitlines() if s.strip()]
    affected=[f for f in frames if f.get('affected_light_id',-1)!=-1]
    obs=[f for f in affected if f.get('navigation',{}).get('affected_light_state_raw') is not None]
    dist=[f['navigation']['distance_to_link_m'] for f in frames if f.get('navigation',{}).get('distance_to_link_m') is not None]
    slow=[f for f in frames if f['speed']<0.1]
    return dict(frames=len(frames),keys=sorted(set().union(*(f.keys() for f in frames))),navigation_keys=sorted(set().union(*(f.get('navigation',{}).keys() for f in frames))),tick_range=[min(ticks),max(ticks)],tick_deltas=dict(gaps),sim_span_sec=(max(ticks)-min(ticks))*.02,unique_ticks=len(set(ticks)),provenance=len(prov),incomplete=len(inc),incomplete_reasons=dict(c(x.get('timeout_reason') for x in inc)),incomplete_before_first=sum(x['native_sync_frame']<min(ticks) for x in inc),incomplete_after_last=sum(x['native_sync_frame']>max(ticks) for x in inc),incomplete_between=sum(min(ticks)<=x['native_sync_frame']<=max(ticks) for x in inc),tick_match=sum(f.get('grpc_label_tick_raw')==f['native_sync_frame'] for f in frames),ego_tick_match=sum(f.get('ego_state_source_frame')==f['native_sync_frame'] for f in frames),next_tick_match=sum(f.get('next_ego_state_source_frame')==f['native_sync_frame']+1 for f in frames),control_modes=dict(c(f.get('control_mode') for f in frames)),control_reused=sum(bool(f.get('control_reused')) for f in frames),traffic_light=dict(c(str(f.get('traffic_light')) for f in frames)),affected=len(affected),matched_light=len(obs),light_ids=sorted(set(f['navigation']['affected_light_id_raw'] for f in affected)),red=sum(bool(f.get('is_red_light_present')) for f in frames),route_valid=dict(c(str(f.get('navigation',{}).get('route_valid')) for f in frames)),route_distance_max=max(dist),route_distance_median=statistics.median(dist),stationary=len(slow),distance_travelled_m=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(frames,frames[1:])),actor_frames=sum(bool(f.get('actors_data')) for f in frames),gt_available=sum(bool(f.get('gt_objects_available')) for f in frames),target_speed=dict(c(f.get('target_speed') for f in frames)),commands=dict(c(f.get('command') for f in frames)),short_waypoints=sum(len(f.get('future_waypoints',[]))<50 for f in frames),last={k:frames[-1].get(k) for k in ['frame_id','speed','navigation','future_waypoints']},example_light=next(({k:f.get(k) for k in ['frame_id','traffic_lights','traffic_light','navigation']} for f in obs),None))
if __name__=='__main__':
    old=BASE/'old'
    oldframes=[json.loads(p.read_text()) for p in sorted((old/'measurements_full').glob('*.json'))]
    newframes=json.loads((NEW/'measurements_all.json').read_text())
    output={'old':summarize(old,oldframes),'new':summarize(NEW,newframes)}
    inv={}
    for d in NEW.iterdir():
        if d.is_dir():
            fs=[p for p in d.iterdir() if p.is_file()]
            inv[d.name]={'count':len(fs),'zero':sum(p.stat().st_size==0 for p in fs),'bytes':sum(p.stat().st_size for p in fs)}
    output['new_inventory']=inv
    output['old_inventory']=json.loads((old/'inventory.json').read_text(encoding='utf-8-sig'))
    (BASE/'summary.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print(json.dumps(output,ensure_ascii=False))
