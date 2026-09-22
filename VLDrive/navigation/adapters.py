"""Input adapters for audited Sensor Record, legacy routes, and portable episodes."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .mgeo import route_from_mgeo


def json_array_items(path):
    """Stream the large audited signal array without retaining camera/object rows."""
    decoder=json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        buffer='';started=False;done=False
        while True:
            block=stream.read(1024*1024)
            buffer+=block
            position=0
            while True:
                while position<len(buffer) and buffer[position].isspace(): position+=1
                if not started:
                    if position==len(buffer): break
                    if buffer[position]!='[': raise ValueError('Expected signal JSON array')
                    started=True;position+=1;continue
                if position==len(buffer): break
                if buffer[position]==',': position+=1;continue
                if buffer[position]==']':
                    done=True;position+=1
                    break
                try: item,end=decoder.raw_decode(buffer,position)
                except json.JSONDecodeError:
                    if not block: raise
                    break
                yield item
                position=end
            buffer=buffer[position:]
            if done:
                if buffer.strip() or stream.read().strip(): raise ValueError('Trailing signal data')
                return
            if not block: raise ValueError('Incomplete signal array')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def portable(path, policy):
    document = load(path)
    if document.get('schema') != 'navigation-episode-v1':
        raise ValueError('Expected navigation-episode-v1: '+str(path))
    if not isinstance(document.get('episode_id'), str) or not document['episode_id']:
        raise ValueError('Portable episode needs a stable episode_id')
    document['frames']=recorded_action_labels(document['frames'], policy)
    return document


def recorded_action_labels(frames, policy):
    """Offline weak instruction labels from actual start/braking outcomes.

    Future samples are used only to classify a completed stop for supervision,
    never by the online navigation engine or as a route/trajectory input.
    """
    frames=[dict(f) for f in frames]
    stopped_speed=policy['stopped_speed_mps'];moving_speed=policy['moving_speed_mps']
    brake=policy['brake_threshold'];gap=policy['max_sensor_gap_ms']*1e6
    for f in frames:
        for key in ('speed_mps','brake'):
            if f.get(key) is None: f.pop(key,None)
            elif type(f[key]) not in (int,float,bool) or not math.isfinite(f[key]):
                raise ValueError('Invalid recorded '+key)
    for i in range(1,len(frames)):
        before,current=frames[i-1],frames[i]
        if (before.get('speed_mps') is not None and current.get('speed_mps') is not None
                and 0<=before['speed_mps']<=stopped_speed and current['speed_mps']>moving_speed
                and 0<current['sensor_source_ns']-before['sensor_source_ns']<=gap):
            current.setdefault('requested_action','start')
            current.setdefault('action_id','recorded_start:'+str(current['index']))
            current['action_evidence']='recorded_stationary_to_motion'
    i=0
    while i<len(frames):
        if not (frames[i].get('brake',0)>brake and frames[i].get('speed_mps',0)>moving_speed):
            i+=1;continue
        start=i;end=i
        while end+1<len(frames) and frames[end+1].get('brake',0)>brake and (
                0<frames[end+1]['sensor_source_ns']-frames[end]['sensor_source_ns']<=gap):
            end+=1
        stopped=any(f.get('speed_mps',float('inf'))<=stopped_speed for f in frames[start:end+1])
        slowing=frames[end].get('speed_mps',float('inf'))<frames[start]['speed_mps']-moving_speed
        if stopped or slowing:
            for f in frames[start:end+1]:
                f.setdefault('requested_action','stop' if stopped else 'slow')
                f.setdefault('action_id','recorded_braking:'+str(frames[start]['index']))
                f['action_evidence']='offline_recorded_brake_and_motion_outcome'
        i=end+1
    return frames


def sensor_record(episode, config, policy=None):
    from tools.build_instructions import sensor_record_route, indexed_timeline
    import yaml
    args = SimpleNamespace(map_root=config.get('map_root'), route_definition=config.get('route_definition'))
    basic, point_links, point_link_indices, _, sources = sensor_record_route(episode, args)
    definition = yaml.safe_load(Path(sources['route_definition']['path']).read_text(encoding='utf-8-sig'))
    key = load(episode/'route_signal_audit_v1/report.json')['route_key']
    definition = definition[key]
    links = {str(x['idx']):x for x in load(sources['link_set']['path'])}
    nodes = {str(x['idx']):x for x in load(sources['node_set']['path'])}
    route = route_from_mgeo(basic['nav_waypoints'], definition['route_links'], links, nodes,
                            definition, sources['global_path']['recorded_sha256'],policy)
    timelines = [indexed_timeline(episode/path) for path in (
        'route_signal_audit_v1/frames.json','gt_timeline_v1/frames.json','native_associations/associations.json')]
    if len({len(rows) for rows in timelines}) != 1:
        raise ValueError('Sensor Record timeline lengths disagree')
    needed={i for row in timelines[0] for i in row.get('ego_source_record_ids',[])}
    ego={}
    signal_path=episode/'quality/active_signals.json'
    if signal_path.is_file():
        from tools.build_instructions import sha256
        for row in json_array_items(signal_path):
            if row.get('record_id') in needed and row.get('role')=='ego': ego[row['record_id']]=row
        sources['ego_signals']={'path':str(signal_path),'sha256':sha256(signal_path)}
    frames = []
    for signal, gt, assoc in zip(*timelines):
        stamp = signal['sensor_source_ns']
        if stamp != gt['lidar_source_ns'] or stamp != assoc['scene_time_hypothesis_ns']:
            raise ValueError('Sensor Record scene timestamp mismatch')
        xy = gt.get('ego_xy')
        frame = dict(index=signal['index'],sensor_source_ns=stamp,
                     x=xy[0] if xy else 0,y=xy[1] if xy else 0,
                     route_index=signal.get('route_index'),
                     valid=bool(xy) and signal.get('route_index') is not None,
                     reason='missing_audited_route_position',
                     camera_files=assoc['camera_files'],lidar_file=assoc['lidar_file'],
                     native_filename_time_ns=assoc['native_filename_time_ns'])
        if frame['valid']:
            idx=frame['route_index']
            # Original audit is still tied to exactly the verified map.
            if (type(idx) is not int or not 0<=idx<len(point_links) or
                    signal['route_link_id'] != point_links[idx] or
                    signal['route_link_index'] != point_link_indices[idx]):
                raise ValueError('Audited route link mismatch')
            # Use a causal, fresh source sample for heading; do not estimate
            # ego heading from the planned route or from a future trajectory.
            candidates=[ego[i] for i in signal.get('ego_source_record_ids',[]) if i in ego and
                        ego[i].get('source_timestamp_ns') is not None and
                        0<=stamp-ego[i]['source_timestamp_ns']<=50_000_000]
            if candidates:
                current=max(candidates,key=lambda r:r['source_timestamp_ns'])
                values=current.get('signal') or {}
                if values.get('yaw') is not None: frame['yaw_rad']=float(values['yaw'])
                if values.get('speed') is not None: frame['speed_mps']=float(values['speed'])
                frame['ego_signal_record_id']=current['record_id']
        frames.append(frame)
    return dict(schema='navigation-episode-v1', episode_id=episode.parent.name+'/'+episode.name,
                route=route,frames=frames,sources=sources,source_path=str(episode))


def legacy(root, policy):
    from tools.build_instructions import dataset_routes, sha256
    _, routes = dataset_routes(root)
    for identity, folder, count in routes:
        path=folder/'route.json'; data=load(path)
        points=data['nav_waypoints']; options=data['nav_road_options']
        if len(points)!=len(options): raise ValueError('Legacy route options do not align')
        events=[]; start=0
        for end in range(1,len(options)+1):
            if end<len(options) and options[end]==options[start]: continue
            option=options[start]
            if option in (1,2,3,5,6):
                events.append(dict(id='command:'+str(start),start_index=start,end_index=end-1,
                    kind='lane_change' if option in (5,6) else 'turn',
                    direction={1:'left',2:'right',3:'straight',5:'left',6:'right'}[option],
                    junction=option==3, evidence='recorded_road_option'))
            start=end
        frames=[]
        for i in range(count):
            m=load(folder/'measurements_full'/f'{i:04d}.json')
            nav=m.get('navigation',{})
            frames.append(dict(index=i,sensor_source_ns=i*100_000_000,x=m['gps_x'],y=m['gps_y'],
                               route_index=nav.get('route_index'),valid=nav.get('route_valid',True),
                               native_sync_frame=m.get('native_sync_frame'),
                               speed_mps=m.get('speed'),brake=float(m.get('brake',0)),
                               clock='legacy_index_10hz'))
        yield dict(schema='navigation-episode-v1',episode_id=identity,frames=recorded_action_labels(frames,policy),
                   route=dict(points=points,events=events,closed=math.dist(points[0][:2],points[-1][:2])<=.001,
                              route_sha256=sha256(path)),source_path=str(folder))


def episodes(config):
    seen=set()
    for source in config['datasets']:
        root=Path(source['root'])
        kind=source.get('format','auto')
        if kind=='auto':
            kind='portable' if root.is_file() or (root/'navigation_episode.json').is_file() else (
                'legacy' if (root/'dataset_index.txt').is_file() else 'sensor_record')
            if kind=='sensor_record' and root.is_dir():
                has_portable=next(root.rglob('navigation_episode.json'),None) is not None
                has_sensor_record=next(root.rglob('episode.json'),None) is not None
                if has_portable and has_sensor_record:
                    raise ValueError('Mixed dataset formats under one root; configure separate dataset entries')
                if has_portable: kind='portable'
        if kind=='portable':
            paths=[root] if root.is_file() else sorted(root.rglob('navigation_episode.json'))
            items=(portable(p,config['policy']) for p in paths)
        elif kind=='legacy': items=legacy(root,config['policy'])
        elif kind=='sensor_record':
            paths=[root] if (root/'episode.json').is_file() else sorted(p.parent for p in root.rglob('episode.json'))
            items=(sensor_record(p, source,config['policy']) for p in paths)
        else: raise ValueError('Unsupported dataset format: '+kind)
        count=0
        for episode in items:
            identity=source.get('namespace','')
            if identity: episode['episode_id']=identity+'/'+episode['episode_id']
            if episode['episode_id'] in seen: raise ValueError('Duplicate episode ID; set dataset namespace')
            seen.add(episode['episode_id']);count+=1
            yield episode
        if not count: raise ValueError('No supported episodes: '+str(root))
