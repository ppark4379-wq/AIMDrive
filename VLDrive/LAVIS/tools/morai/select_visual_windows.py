#!/usr/bin/env python3
"""Inspect existing contracts and select a small, declared speed-band fit probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
import os
sys.path.insert(0, str(ROOT))
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, add_path_arguments, resolve_settings
configure_runtime()


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    add_path_arguments(parser)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('Use a new output inside LAVIS')
    output.mkdir(parents=True)
    import numpy as np
    import torch
    from lavis.datasets.datasets.morai.training_dataset import MoraiTrainingDataset
    from lavis.runners.morai_trainer import subset_indices
    torch.set_num_threads(4)
    config=resolve_settings(json.loads(args.config.read_text()), args)
    root=ROOT/config['dataset']
    def stats(values):
        a=np.asarray(values,dtype=float)
        return dict(min=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))
    speed_max=config['normal_max_speed_kmh']/3.6
    count=config['train_windows']
    summary=dict(schema='morai-visual-contract-v1',user_normal_max_speed_kmh=config['normal_max_speed_kmh'],
        high_speed_section_max_speed_kmh=None,section_labels_available=False,
        note='Speed-band selection does not identify normal/high-speed map sections and is not a compliance judgement.',
        actual_route_contract=dict(points=41,spacing_m=1.,normalization_m=40.),
        teacher_horizons_s=[.5,1.,1.5,2.,2.5],sequence_length=config['sequence_length'],
        cross_sensor_clock='receive_monotonic_ns only; no cross-sensor header subtraction',
        speed_kinematics=[dict(speed_kmh=v,distance_in_2_5s_m=v/3.6*2.5,seconds_covered_by_40m=40/(v/3.6)) for v in (30,40,60,80,100)],
        screening=dict(current_and_future_max_speed_kmh=config['normal_max_speed_kmh'],route_start_distance_m_max=2.,
            purpose='small trainability diagnostic only; preserve every original row'),splits={})
    selection=dict(schema='morai-visual-probe-selection-v1',train=[],val=[],val_regular=[],sample_ids={})
    all_details=[]
    for split in ('train','val'):
        dataset=MoraiTrainingDataset(root,split_file=root/config['split_file'],split=split,
            sequence_length=config['sequence_length'],input_mode='raw')
        manifests={}
        freshness={k:[] for k in ('front','left','right','rear','gps','imu','ego_teacher')}
        for run in dataset.base._runs:
            rows=[json.loads(line) for line in (run['path']/'manifest_training.jsonl').read_text().splitlines()]
            manifests[run['name']]={row['sample_id']:row for row in rows}
            for row in rows:
                anchor=row['anchor_receive_monotonic_ns']
                refs=row['sensor_refs']
                selected=dict(refs['cameras'],gps=refs['gps'],imu=refs['imu'],ego_teacher=row['privileged_metadata']['ego_teacher'])
                for name,ref in selected.items():
                    age=(anchor-ref['receive_monotonic_ns'])/1e6
                    if age<0:raise ValueError('Noncausal '+name)
                    freshness[name].append(age)
        details=[]
        for index,(run_index,begin,end) in enumerate(dataset.base.windows):
            run=dataset.base._runs[run_index];entries=run['entries'][begin:end];entry=entries[-1]
            row=manifests[run['name']][entry['sample_id']]
            with np.load(run['path']/entry['target']['path'],allow_pickle=False) as data:
                xy=data['trajectory_xy_forward_left'].astype(float);speed=data['future_speed_mps'].astype(float)
            with np.load(run['path']/entry['inputs']['route']['path'],allow_pickle=False) as data:
                route=data['ego_frame_xy'].astype(float)
            arc=float(np.linalg.norm(np.diff(np.vstack([np.zeros((1,2)),xy]),axis=0),axis=-1).sum())
            coverage=float(np.linalg.norm(np.diff(route,axis=0),axis=-1).sum())
            offset=float(np.linalg.norm(route[0]));current=float(row['privileged_metadata']['ego_teacher']['speed_xy_mps'])
            allowed=max(current,float(speed.max()))<=speed_max+1e-6
            eligible=allowed and offset<=2.
            details.append(dict(split=split,index=index,sample_id=entry['sample_id'],eligible=eligible,
                current_speed_mps=current,future_max_speed_mps=float(speed.max()),future_mean_speed_mps=float(speed.mean()),
                speed_band_pass=allowed,route_start_distance_m=offset,teacher_polyline_m=arc,route_coverage_m=coverage,
                history_receive_s=(entries[-1]['timestamp']-entries[0]['timestamp'])/1e9))
        eligible=[d for d in details if d['eligible']]
        if len(eligible)<count:raise ValueError('Not enough speed-band/path-close windows for declared probe')
        chosen=[eligible[i]['index'] for i in subset_indices(len(eligible),count)]
        if split=='train':selection['train']=chosen
        else:
            selection['val_regular']=chosen
            selection['val']=subset_indices(len(dataset),config['val_windows'])
        for name in ('train',) if split=='train' else ('val','val_regular'):
            selection['sample_ids'][name]=[details[i]['sample_id'] for i in selection[name]]
        summary['splits'][split]=dict(windows=len(dataset),eligible_windows=len(eligible),
            current_and_future_speed_band_fraction=float(np.mean([d['speed_band_pass'] for d in details])),
            history_receive_s=stats([d['history_receive_s'] for d in details]),
            teacher_polyline_m=stats([d['teacher_polyline_m'] for d in details]),
            future_max_speed_mps=stats([d['future_max_speed_mps'] for d in details]),
            receive_age_ms={k:stats(v) for k,v in freshness.items()},
            geometric_coverage_exceeded_fraction={str(length):float(np.mean([d['teacher_polyline_m']>length for d in details]))
                for length in (40,50,60,80,100)},
            selected_trainability_or_regular_indices=chosen,
            source_dataset=dataset.describe())
        all_details.extend(details)
    summary['normal_speed_2_5s_minimum_path_m']=speed_max*2.5
    summary['candidate_normal_lookahead']=dict(length_m=50.,points=41,spacing_m=1.25,
        constant_speed_distance_margin_m=50-speed_max*2.5,status='candidate only, not written into existing dataset or model')
    write(output/'contract.json',summary);write(output/'selection.json',selection)
    with (output/'window_details.jsonl').open('x') as stream:
        for row in all_details:stream.write(json.dumps(row)+'\n')
    print(json.dumps(dict(selection=selection,history={k:v['history_receive_s'] for k,v in summary['splits'].items()},
        eligible={k:v['eligible_windows'] for k,v in summary['splits'].items()}),indent=2),flush=True)


if __name__=='__main__':main()
