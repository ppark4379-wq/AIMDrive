#!/usr/bin/env python3
"""Replay allowed sensors at recorded receipt times. No vehicle control or GT pose input."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from gps_jamming.localization.gps_blackout_hold import GpsBlackoutHold, SectionZone
from gps_jamming.localization.gps_aligned_hold import AlignedGpsBlackoutHold
from localization_metrics import rows, pose_rows, sha256
from analyze_localization import write_analysis


class History:
    def __init__(self):
        self.history,self.history_lock=deque(maxlen=200),threading.Lock()


class ReplayGps:
    def __init__(self,clock):
        self.clock,self.fix,self.identity=clock,None,{}
        self.packets=self.samples=0

    def latest_snapshot(self):
        return (self.fix,(self.clock[0]-self.identity['received_ns'])/1e9,self.identity) if self.fix else (None,float('inf'),{})

    def diagnostics(self):
        return {}


def allowed_value(record):
    """Whitelist estimator inputs; even status packet pose/yaw fields are excluded."""
    value=record['value']
    keys={'gps':('latitude','longitude','altitude'),
          'status':('signed_speed','sim_time'),
          'imu':('yaw','quaternion','angular_velocity','linear_acceleration','sim_time')}[record['sensor']]
    return {key:value[key] for key in keys if key in value}


def replay(recording,output,method='aligned'):
    recording,output=Path(recording),Path(output)
    if (output/'poses.jsonl').exists():
        raise FileExistsError('choose a new output directory; poses.jsonl already exists')
    plan=json.loads((recording/'plan.json').read_text())
    recorded=rows(recording/'sensor_audit.jsonl')
    if any(not isinstance(r.get('received_ns'),int) or isinstance(r.get('received_ns'),bool)
           or r['received_ns'] <= 0 for r in recorded):
        raise ValueError('positive integer sensor receipt timestamps required')
    controls=pose_rows(recording)
    batch_format=any('observed_poll_id' in row for row in recorded) or any('poll_id' in row for row in controls)
    if batch_format:
        if not all('observed_poll_id' in row for row in recorded) or not all('poll_id' in row for row in controls) or not controls:
            raise ValueError('batch recordings require observed_poll_id on all sensors and poll_id on all poses')
        record_polls=[row['observed_poll_id'] for row in recorded]
        poll_ids=[row['poll_id'] for row in controls]
        if any(not isinstance(value,int) or isinstance(value,bool) or value <= 0 for value in record_polls+poll_ids):
            raise ValueError('positive integer poll IDs required')
        if any(b<=a for a,b in zip(poll_ids,poll_ids[1:])):
            raise ValueError('pose poll IDs must be strictly monotonic')
        records=sorted(enumerate(recorded),key=lambda item:(item[1]['observed_poll_id'],item[1]['received_ns'],item[0]))
    else:
        poll_ids=None
        records=sorted(enumerate(recorded),key=lambda item:(item[1]['received_ns'],item[0]))
    ticks=[row.get('monotonic_ns',round(row.get('monotonic',0)*1e9)) for row in controls]
    if not ticks:
        ticks=sorted({record['received_ns'] for _,record in records})
    if any(not isinstance(tick,int) or isinstance(tick,bool) or tick <= 0 for tick in ticks):
        raise ValueError('positive integer output ticks required')
    if any(b<a for a,b in zip(ticks,ticks[1:])):
        raise ValueError('recorded output ticks must be monotonic')
    status=type('ReplayStatus',(),{})();status.speed,status.imu=History(),History()
    clock=[0];gps=ReplayGps(clock)
    options=dict(simulate_zone=SectionZone(**{key:value for key,value in plan['simulate_zone'].items()
                                             if key in ('start_xy','end_xy','half_width_m','margin_m')})) if plan.get('simulate_zone') else {}
    options.update({key:plan[key] for key in ('max_hold_s','max_distance_m') if key in plan})
    anchor_window=0. if method=='aligned_no_filter' else plan.get('gps_anchor_window_s',plan.get('anchor_window_s',1.))
    estimator=(GpsBlackoutHold(gps,status,plan['map_basis'],**options) if method=='legacy' else
        AlignedGpsBlackoutHold(gps,status,plan['map_basis'],anchor_window_s=anchor_window,**options))
    output.mkdir(parents=True,exist_ok=True)
    index=0;counts=dict(gps=0,status=0,imu=0);seen={key:set() for key in counts};epochs={}
    replay_plan=dict(schema='gps_jamming_localization_replay_v1',method=method,map_basis=plan['map_basis'],
        simulate_zone=plan.get('simulate_zone'),source=str(recording.resolve()),scope='same_recorded_inputs_not_new_live_trial',
        input_sha256={name:sha256(recording/name) for name in ('plan.json','sensor_audit.jsonl')},
        sample_identity='per_stream_sample_id_or_per_stream_record_order',anchor_window_s=anchor_window,
        replay_visibility='observed_poll_and_receipt' if batch_format else 'receipt_only_legacy',
        max_hold_s=options.get('max_hold_s',120.),max_distance_m=options.get('max_distance_m',250.))
    with (output/'poses.jsonl').open('w',encoding='utf-8') as stream,patch('time.monotonic_ns',side_effect=lambda:clock[0]):
        for tick_index,tick in enumerate(ticks):
            clock[0]=tick
            poll_id=poll_ids[tick_index] if batch_format else None
            while (index<len(records) and records[index][1]['received_ns']<=tick
                   and (not batch_format or records[index][1]['observed_poll_id']<=poll_id)):
                _,record=records[index];index+=1
                sensor=record['sensor']
                if sensor not in counts:raise ValueError('unknown sensor: '+str(sensor))
                counts[sensor]+=1
                identity_id=record.get('sample_id',counts[sensor])
                if not isinstance(identity_id,int) or isinstance(identity_id,bool) or identity_id <= 0:
                    raise ValueError('positive integer per-stream sample IDs required')
                epoch=record.get('clock_epoch',0)
                identity_key=(epoch,identity_id)
                if identity_key in seen[sensor]:continue
                seen[sensor].add(identity_key)
                if sensor in epochs and epoch != epochs[sensor] and sensor != 'gps':
                    getattr(status,'speed' if sensor=='status' else 'imu').history.clear()
                epochs[sensor]=epoch
                source=record.get('source_stamp_ns')
                if source is None:source=round(record['value'].get('sim_time',0)*1e9)
                identity=dict(received_ns=record['received_ns'],source_stamp_ns=source,sample_id=identity_id,clock_epoch=epoch)
                value=allowed_value(record)
                if sensor=='gps':gps.fix,gps.identity=value,identity
                else:getattr(status,'speed' if sensor=='status' else 'imu').history.append((value,identity))
            fix,age,meta=estimator.latest_snapshot()
            pose=None
            if fix:
                point=estimator._grid(fix)
                imu_value=status.imu.history[-1][0] if status.imu.history else None
                speed_value=status.speed.history[-1][0] if status.speed.history else None
                if imu_value and speed_value:
                    import math
                    pose=dict(x=point[0],y=point[1],yaw_grid_rad=imu_value['yaw']+math.radians(estimator.core.convergence_deg(point)),
                              speed_mps=speed_value['signed_speed']/3.6)
            row=dict(monotonic_ns=tick,position=pose,selected_samples=dict(gps=meta))
            if batch_format:row['poll_id']=poll_id
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    # Optional references are copied for post-estimation scoring only, never fed above.
    (output/'sensor_audit.jsonl').write_bytes((recording/'sensor_audit.jsonl').read_bytes())
    replay_plan['source_code_sha256']={str(path.relative_to(PACKAGE)):sha256(path) for path in
        (Path(__file__).resolve(),Path(__file__).resolve().with_name('localization_metrics.py'),
         PACKAGE/'gps_jamming/localization/gps_blackout_hold.py',PACKAGE/'gps_jamming/localization/gps_aligned_hold.py',
         PACKAGE/'gps_jamming/localization/map_geometry.py')}
    (output/'plan.json').write_text(json.dumps(replay_plan,indent=2,allow_nan=False)+'\n')
    if (recording/'recording_status.json').is_file():
        (output/'recording_status.json').write_bytes((recording/'recording_status.json').read_bytes())
    analysis,folder=write_analysis(output)
    return analysis,folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--method',choices=('legacy','aligned_no_filter','aligned'),default='aligned')
    args=parser.parse_args(argv)
    result,folder=replay(args.recording,args.output_dir,args.method)
    print(json.dumps(dict(scope='same_recorded_inputs_not_new_live_trial',method=args.method,
        metrics=result['metrics'],unique_gps_comparisons=result['unique_gps_comparisons'],output=str(folder.resolve())),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
