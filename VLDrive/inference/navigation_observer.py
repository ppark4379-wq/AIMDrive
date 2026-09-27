#!/usr/bin/env python3
"""Observe manual MORAI driving, or replay recorded poses, using the same engine."""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from navigation.engine import NavigationEngine
from navigation.build import dump


from navigation.route_source import load_live_route


def validate_udp(udp):
    if not isinstance(udp,dict) or set(udp)!={'bind','gps_port','vehicle_status_port'}:
        raise ValueError('udp must specify bind, gps_port, vehicle_status_port')
    if not isinstance(udp['bind'],str) or not udp['bind'].strip():
        raise ValueError('udp.bind must be a local interface address')
    ports=[udp['gps_port'],udp['vehicle_status_port']]
    if any(type(port) is not int or not 1<=port<=65535 for port in ports) or ports[0]==ports[1]:
        raise ValueError('GPS and Vehicle Status require distinct integer UDP ports in 1..65535')


def load_bundle(folder, episode_id=None):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    entries=[row for row in manifest['episodes'] if episode_id is None or row['episode_id']==episode_id]
    if len(entries)!=1: raise ValueError('Select one episode_id for the live planned route')
    entry=entries[0]
    path=(folder/entry['episode']).resolve()
    if folder.resolve() not in path.parents: raise ValueError('Unsafe episode path')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['episode_sha256']:
        raise ValueError('Navigation route bundle changed')
    return json.loads(path.read_text()),manifest,entry


def reception(receiver, sample, age, max_age):
    packets=getattr(receiver,'packets',0)
    state=('no_valid_sample' if packets else 'no_packets') if sample is None else (
        'stale' if age>max_age else 'ok')
    return dict(port=receiver.port,state=state,packets=packets,
                samples=getattr(receiver,'samples',0),age_s=age if math.isfinite(age) else None)


def reception_text(diagnostics):
    parts=[]
    for name in ('gps','status'):
        item=diagnostics[name]
        age='--' if item['age_s'] is None else f"{item['age_s']*1000:.0f}ms"
        parts.append(f"{name.upper()} :{item['port']} {item['state'].upper()} | "
                     f"packets {item['packets']} parsed {item['samples']} age {age}")
    if diagnostics.get('startup_wait_exceeded'):
        parts.append('Still waiting. Check MORAI destination IP/ports and sensor output; automatic retry is active.')
    return '\n'.join(parts)


class Display:
    def __init__(self, route):
        import tkinter as tk
        self.root=tk.Tk();self.root.title('MORAI navigation observer')
        self.label=tk.Label(self.root,text='Waiting for position...',wraplength=820,font=('Arial',20))
        self.label.pack(padx=16,pady=12)
        self.info=tk.Label(self.root,text='',font=('Arial',11));self.info.pack()
        self.network=tk.Label(self.root,text='',font=('Arial',10));self.network.pack()
        self.canvas=tk.Canvas(self.root,width=840,height=540,bg='white');self.canvas.pack()
        points=route['points'];xs=[p[0] for p in points];ys=[p[1] for p in points]
        self.x0=min(xs);self.y0=min(ys);self.scale=min(800/max(1,max(xs)-min(xs)),500/max(1,max(ys)-min(ys)))
        vertices=[v for p in points for v in self.xy(p[0],p[1])]
        self.canvas.create_line(*vertices,fill='#9aa5b1',width=2)
        self.car=self.canvas.create_oval(0,0,10,10,fill='#1565c0',state='hidden')
        self.closed=False
        self.root.protocol('WM_DELETE_WINDOW',self.close)
    def xy(self,x,y): return 20+(x-self.x0)*self.scale,520-(y-self.y0)*self.scale
    def close(self): self.closed=True;self.root.destroy()
    def update(self,frame,result):
        if self.closed: return
        ins=result.get('instruction')
        self.label.config(text=ins['text'] if ins else 'UNAVAILABLE: '+result.get('reason','waiting'))
        self.info.config(text=(f"{ins['rule']}  variant {ins['variant_id']}  " if ins else '')+
                         f"route {result.get('progress_m',0):.1f} m | source {frame['sensor_source_ns']}")
        self.network.config(text=reception_text(result['reception']) if 'reception' in result else '')
        self.canvas.itemconfigure(self.car,state='normal' if result['valid'] else 'hidden')
        if result['valid']:
            x,y=self.xy(frame['x'],frame['y'])
            self.canvas.coords(self.car,x-5,y-5,x+5,y+5)
        self.root.update_idletasks();self.root.update()


def replay(episode, policy):
    engine=NavigationEngine(episode['route'],policy,episode['episode_id'])
    previous=None
    for frame in episode['frames']:
        if previous is not None and frame['index']!=previous+1:
            engine.instances.clear();engine.active=None
        previous=frame['index']
        yield frame,engine.update(frame)


def live(config, episode, policy):
    from inference.morai_udp import GpsReceiver, VehicleStatusReceiver
    from pyproj import Transformer
    runtime=config['runtime'];udp=config['udp'];position=config['position']
    gps=status=None
    try:
        gps=GpsReceiver(udp['gps_port'],bind=udp['bind'])
        status=VehicleStatusReceiver(udp['vehicle_status_port'],bind=udp['bind'])
        converter=Transformer.from_crs('EPSG:4326',position['map_crs'],always_xy=True)
        engine=NavigationEngine(episode['route'],policy,'manual:'+episode['episode_id'])
        began=time.monotonic();count=0;last_stamp=None;last_result=None;last_frame=None
        while True:
            fix,gps_age=gps.latest();vehicle,status_age=status.latest()
            diagnostics=dict(gps=reception(gps,fix,gps_age,runtime['max_age_s']),
                             status=reception(status,vehicle,status_age,runtime['max_age_s']))
            valid=(fix is not None and vehicle is not None and
                   max(gps_age,status_age)<=runtime['max_age_s'] and
                   abs(gps_age-status_age)<=runtime['max_pair_skew_s'])
            if not valid:
                failures=[name+'_'+item['state'] for name,item in diagnostics.items() if item['state']!='ok']
                reason='+'.join(failures) if failures else 'gps_status_receive_time_skew'
                diagnostics['startup_wait_exceeded']=(last_stamp is None and
                    time.monotonic()-began>runtime['startup_timeout_s'])
                frame=dict(index=count,sensor_source_ns=time.monotonic_ns(),x=0,y=0,valid=False,
                           reason=reason,clock='host_monotonic')
                # Do not retain a stale instruction on the screen.
                engine.active=None;engine.instances.clear()
                last_frame=frame
                last_result=dict(valid=False,reason=frame['reason'],instruction=None,candidates=[],reception=diagnostics)
                yield frame,last_result
            else:
                stamp=round(vehicle['sim_time']*1e9)
                if stamp!=last_stamp:
                    east,north=converter.transform(fix['longitude'],fix['latitude'])
                    frame=dict(index=count,sensor_source_ns=stamp,x=east-position['east_offset_m'],
                               y=north-position['north_offset_m'],yaw_rad=vehicle['yaw'],
                               speed_mps=vehicle['speed'],brake=vehicle['brake'],clock='morai_status_simulation_time')
                    if not all(math.isfinite(frame[k]) for k in ('x','y','yaw_rad','speed_mps')):
                        frame.update(valid=False,reason='nonfinite_live_pose')
                    last_result=engine.update(frame);last_frame=frame
                    last_result['reception']=diagnostics
                    yield frame,last_result
                    last_stamp=stamp;count+=1
                elif last_result is not None:
                    yield dict(last_frame,display_only=True),dict(last_result,reception=diagnostics)
            time.sleep(1/runtime['hz'])
    finally:
        if gps: gps.close()
        if status: status.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    from utils.config_files import read_document, document_bytes
    config=read_document(args.config)
    required={'schema','mode','bundle','episode_id','output','window','runtime','udp','position'}
    if set(config)-{'route'}!=required or config['schema']!='morai-navigation-observer-v1':
        raise ValueError('Invalid observer config')
    if config['mode'] not in ('replay','udp'): raise ValueError('mode must be replay or udp')
    runtime=config['runtime']
    if set(runtime)!={'hz','max_age_s','max_pair_skew_s','startup_timeout_s','replay_realtime'}:
        raise ValueError('Invalid observer runtime parameters')
    for k in ('hz','max_age_s','max_pair_skew_s','startup_timeout_s'):
        if type(runtime[k]) not in (int,float) or not math.isfinite(runtime[k]) or runtime[k]<=0:
            raise ValueError(k+' must be finite and positive')
    validate_udp(config['udp'])
    if config.get('route') is not None:
        if config['mode']!='udp' or config['bundle'] is not None or config['episode_id'] is not None:
            raise ValueError('Explicit route is for UDP mode; set bundle and episode_id to null')
        episode,manifest,entry=load_live_route(config['route'])
        folder=None
    else:
        folder=ROOT/config['bundle']
        episode,manifest,entry=load_bundle(folder,config['episode_id'])
    output=ROOT/config['output']/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(dump(config),encoding='utf-8')
    (output/'planned_route.json').write_text(dump(dict(episode_id=episode['episode_id'],
        route=episode['route'],sources=episode.get('sources',{}),policy=manifest['policy'])),encoding='utf-8')
    view=Display(episode['route']) if config['window'] else None
    stream=live(config,episode,manifest['policy']) if config['mode']=='udp' else replay(episode,manifest['policy'])
    rows=0;changes=0;invalid=0;last=None;expected=[];differences=[]
    if config['mode']=='replay':
        trace_path=folder/entry['trace']
        if hashlib.sha256(trace_path.read_bytes()).hexdigest()!=entry['trace_sha256']:
            raise ValueError('Recorded replay trace changed')
        expected=[json.loads(line) for line in trace_path.read_text().splitlines()]
    try:
        with (output/'navigation.jsonl').open('x',encoding='utf-8') as log:
            for frame,result in stream:
                if frame.get('display_only'):
                    if view:
                        view.update(frame,result)
                        if view.closed: break
                    continue
                instruction=result.get('instruction')
                key=instruction['instruction_instance_id'] if instruction else result.get('reason')
                if key!=last:
                    print(instruction['text'] if instruction else 'UNAVAILABLE: '+str(key),flush=True)
                    changes+=1;last=key
                if not result['valid']: invalid+=1
                if expected:
                    current=expected[rows]
                    if (current['valid']!=result['valid'] or current['active_instruction_instance_id']!=
                            (instruction['instruction_instance_id'] if instruction else None) or
                            current['candidate_ids']!=[c['instruction_instance_id'] for c in result['candidates']]):
                        differences.append(frame['index'])
                log.write(json.dumps(dict(frame=frame,result=result),allow_nan=False)+'\n');log.flush()
                rows+=1
                if view:
                    view.update(frame,result)
                    if view.closed: break
                if config['mode']=='replay' and runtime['replay_realtime']: time.sleep(1/runtime['hz'])
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        if view and not view.closed: view.close()
    summary=dict(mode=config['mode'],frames=rows,changes=changes,unavailable_frames=invalid,
                 replay_difference_frames=differences,
                 replay_complete=(len(expected)==rows) if expected else None,
                 physical_manual_drive_verified=False,
                 note='Live trace is evidence for human review, not an automatic driving-quality certification.')
    (output/'summary.json').write_text(dump(summary),encoding='utf-8')
    if config['mode']=='udp':
        # Portable pose-only episodes let another manual run use the same
        # offline generator. These are not camera/LiDAR training publications.
        segments=[];current=[]
        with (output/'navigation.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                frame=json.loads(line)['frame']
                if frame.get('clock')!='morai_status_simulation_time': continue
                if current and frame['sensor_source_ns']<=current[-1]['sensor_source_ns']:
                    segments.append(current);current=[]
                current.append(dict(frame,index=len(current)))
        if current: segments.append(current)
        for i,frames in enumerate(segments):
            folder=output/f'episode_{i:03d}';folder.mkdir()
            (folder/'navigation_episode.json').write_text(dump(dict(schema='navigation-episode-v1',
                episode_id='manual/'+output.name+f'/episode_{i:03d}',route=episode['route'],frames=frames,
                source='manual_observer_pose_log',contains_training_sensors=False)),encoding='utf-8')
    print(dump(dict(output=str(output),**summary)))
    if differences or (expected and rows!=len(expected)): raise SystemExit(1)


if __name__=='__main__': main()
