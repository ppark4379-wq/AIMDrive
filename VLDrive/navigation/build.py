"""Generate reproducible candidate instruction episodes and online replay traces."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from .adapters import episodes
from .catalog import CATALOG_SHA256, CORRECTIONS, coverage
from .engine import DEFAULT_POLICY, NavigationEngine

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = 'lmdrive-navigation-build-v1'
LABEL_SCHEMA = 'lmdrive-navigation-labels-v1'


def read_config(path):
    path=Path(path).resolve()
    from utils.config.config_files import load_config
    config=load_config(path, 'navigation')
    if config.get('schema')!=CONFIG_SCHEMA or set(config)!={'schema','datasets','output','policy'}:
        raise ValueError('Expected strict '+CONFIG_SCHEMA+' config')
    if not isinstance(config['datasets'],list) or not config['datasets']:
        raise ValueError('datasets must be a nonempty list')
    def resolve(value):
        if not isinstance(value,str) or not value.strip(): raise ValueError('Expected nonempty path')
        return str((ROOT/Path(value).expanduser()).resolve())
    config['output']=resolve(config['output'])
    for dataset in config['datasets']:
        if not isinstance(dataset,dict) or 'root' not in dataset or set(dataset)-{'root','format','namespace','map_root','route_definition','exclude_episodes'}:
            raise ValueError('Invalid dataset adapter config')
        for key in ('root','map_root','route_definition'):
            if dataset.get(key) is not None: dataset[key]=resolve(dataset[key])
        if dataset.get('format','auto') not in ('auto','portable','sensor_record','legacy'):
            raise ValueError('Invalid dataset format')
        source=Path(dataset['root']);out=Path(config['output'])
        if out==source or source in out.parents:
            raise ValueError('Generated output must be outside source datasets')
    if not isinstance(config['policy'],dict) or set(config['policy'])!=set(DEFAULT_POLICY):
        raise ValueError('Config must explicitly specify all navigation policy keys')
    # Include resolved shared settings in the provenance fingerprint.
    from utils.config.config_files import read_document, document_bytes
    source=read_document(path)
    from utils.config.config_files import read_document, document_bytes
    payload=json.dumps(load_config(path, 'navigation'),sort_keys=True).encode() if 'preprocessing' in source else document_bytes(path)
    config['config_sha256']=hashlib.sha256(payload).hexdigest()
    return config


def build_episode(episode, policy):
    engine=NavigationEngine(episode['route'],policy,episode['episode_id'])
    records={};trace=[];previous=None;seen=set();indices=set();excluded=Counter()
    for frame in episode['frames']:
        stamp=frame['sensor_source_ns'];idx=frame['index']
        if type(idx) is not int or idx<0 or idx in indices or stamp in seen:
            raise ValueError('Duplicate or invalid source frame identity')
        if previous is not None and (stamp<=previous['sensor_source_ns'] or idx<=previous['index']):
            raise ValueError('Offline frame order must be strictly increasing')
        if previous is not None and idx!=previous['index']+1:
            engine.instances.clear();engine.active=None
        seen.add(stamp);indices.add(idx)
        result=engine.update(frame)
        if not result['valid']: excluded[result['reason']]+=1
        for candidate in result['candidates']:
            key=candidate['instruction_instance_id']
            if key not in records:
                records[key]=dict(candidate,episode_id=episode['episode_id'],start_frame=idx,stop_frame=idx,
                                  start_sensor_source_ns=stamp,stop_sensor_source_ns=stamp,
                                  schema=LABEL_SCHEMA,route_sha256=engine.route.identity)
            records[key].update(stop_frame=idx,stop_sensor_source_ns=stamp)
        trace.append(dict(frame=frame, valid=result['valid'],reason=result.get('reason'),
                          progress_m=result.get('progress_m'),lap=result.get('lap'),
                          active_instruction_instance_id=result['instruction']['instruction_instance_id'] if result['instruction'] else None,
                          candidate_ids=[c['instruction_instance_id'] for c in result['candidates']]))
        previous=frame
    if not trace: raise ValueError('Episode has no frames')
    return list(records.values()),trace,dict(excluded)


def dump(value): return json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
def jsonl(rows): return ''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows)


def build(config_path):
    config=read_config(config_path)
    files={};all_records=[];reports=[];manifest=[]
    for episode in episodes(config):
        records,trace,excluded=build_episode(episode,config['policy'])
        identity=episode['episode_id'];stem=hashlib.sha256(identity.encode()).hexdigest()[:20]
        route_file='episodes/'+stem+'.json'
        trace_file='traces/'+stem+'.jsonl'
        files[route_file]=dump(episode);files[trace_file]=jsonl(trace)
        manifest.append(dict(episode_id=identity,episode=route_file,trace=trace_file,
                             episode_sha256=hashlib.sha256(files[route_file].encode()).hexdigest(),
                             trace_sha256=hashlib.sha256(files[trace_file].encode()).hexdigest()))
        reports.append(dict(episode_id=identity,frames=len(trace),valid_frames=sum(t['valid'] for t in trace),
                            excluded=excluded,instruction_instances=len(records),coverage=coverage(records),
                            diagnostics=episode['route'].get('diagnostics',[])))
        all_records.extend(records)
    if not all_records: raise ValueError('No valid instruction instances in dataset')
    files['instructions.jsonl']=jsonl(all_records)
    files['manifest.json']=dump(dict(schema=LABEL_SCHEMA,episodes=manifest,policy=config['policy'],
                                     catalog_sha256=CATALOG_SHA256,corrections=CORRECTIONS,
                                     instruction_sha256=hashlib.sha256(files['instructions.jsonl'].encode()).hexdigest()))
    files['report.json']=dump(dict(schema='lmdrive-navigation-build-report-v1',config=config,
        implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        coverage=coverage(all_records),episodes=reports,
        semantics='LMDrive taxonomy and templates; MORAI route-aware rules, not CARLA frame-rule bitwise parity',
        physical_manual_drive_verified=False))
    output=Path(config['output'])
    for name,content in files.items():
        path=output/name
        if path.exists() and path.read_text(encoding='utf-8')!=content:
            raise FileExistsError('Output differs; choose a new config output directory: '+str(path))
    existed=all((output/name).exists() for name in files)
    for name,content in files.items():
        path=output/name
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8',newline='\n') as stream: stream.write(content)
    summary=dict(output=str(output),episodes=len(reports),instruction_instances=len(all_records),
                 observed_types=sum(v['instances']>0 for v in coverage(all_records).values()),
                 valid_frames=sum(e['valid_frames'] for e in reports),
                 status='verified_existing' if existed else 'created')
    print(dump(summary))
    return summary
