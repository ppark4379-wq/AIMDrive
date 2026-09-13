#!/usr/bin/env python3
"""Train/verify a small raw-sensor visual diagnostic using existing contracts."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[0]
import os
sys.path.insert(0, str(ROOT))
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, add_path_arguments, resolve_settings
configure_runtime()


def write(path,value):
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--policy',choices=('frozen','fusion','all_features'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('train','verify'),default='train')
    parser.add_argument('--experiment',type=Path)
    parser.add_argument('--updates',type=int,help='Override optimizer update count')
    add_path_arguments(parser)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('Use a new output inside LAVIS')
    output.mkdir(parents=True)
    settings=resolve_settings(json.loads(args.config.read_text()), args);settings['encoder_policy']=args.policy
    if args.updates is not None: settings['updates']=args.updates
    write(output/'resolved_config.json',settings)
    from lavis.common.morai_runtime import bootstrap,sha256
    backend=bootstrap(ROOT.parent)
    import numpy as np
    import torch
    from lavis.datasets.datasets.morai.training_dataset import MoraiTrainingDataset
    from lavis.runners.morai_trainer import (seed_all,source_hashes,make_optimizer,evaluate,train,
        parameter_hash,save_checkpoint,restore_checkpoint)
    from lavis.models.vldrive.visual_trajectory import VisualTrajectoryModel, VisualTrajectoryConfig
    from lavis.runners.morai_visual_trainer import visual_preflight
    torch.set_num_threads(4);seed_all(settings['seed'])
    if not torch.cuda.is_available():raise RuntimeError('This actual raw-sensor probe requires CUDA')
    selection=json.loads(args.selection.read_text())
    dataset_root=ROOT/settings['dataset']
    print('Validating unchanged raw/teacher dataset and receive-causal windows...',flush=True)
    datasets={split:MoraiTrainingDataset(dataset_root,split_file=dataset_root/settings['split_file'],split=split,
        sequence_length=settings['sequence_length'],input_mode='raw') for split in ('train','val')}
    indices={key:selection[key] for key in ('train','val','val_regular')}
    for split,selected in indices.items():
        base=datasets['train' if split=='train' else 'val'].base
        ids=[]
        for index in selected:
            ri,begin,end=base.windows[index]
            ids.append(base._runs[ri]['entries'][end-1]['sample_id'])
        if ids!=selection['sample_ids'][split]:raise ValueError('Selection no longer identifies the same anchors')
    provenance=dict(backend=backend,source=source_hashes(ROOT),
        selection_sha256=sha256(args.selection),datasets={k:v.describe() for k,v in datasets.items()})
    write(output/'provenance.json',provenance);write(output/'window_indices.json',indices)
    config=VisualTrajectoryConfig(encoder_policy=args.policy,visual_checkpoint=str(ROOT/settings['visual_checkpoint']),
        visual_sha256=datasets['train'].base.checkpoint_sha256,precision=settings['precision'])
    model=VisualTrajectoryModel(config,device='cuda');optimizer=make_optimizer(model,**settings['optimizer'])
    write(output/'model.json',model.describe())
    def evaluate_split(split,destination):
        def progress(completed,total):
            write(output/'evaluation_progress.json',dict(split=split,completed=completed,total=total))
        return evaluate(model,datasets['train' if split=='train' else 'val'],indices[split],'cuda',
            save_predictions=destination,on_progress=progress)
    if args.mode=='verify':
        if not args.experiment:raise ValueError('Verification requires --experiment')
        expected=json.loads((args.experiment/'report.json').read_text())
        saved=restore_checkpoint(args.experiment/'checkpoint.pt',model,optimizer,provenance=provenance)
        actual=evaluate_split('val',output/'val_predictions.json')
        old=json.loads((args.experiment/'val_predictions.json').read_text())
        new=json.loads((output/'val_predictions.json').read_text())
        assert [r['sample_id'] for r in old]==[r['sample_id'] for r in new]
        deltas={k:max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for a,b in zip(old,new))
            for k in ('trajectory','speed')}
        matched=parameter_hash(model,trainable=True)==expected['trainable_hash']
        hashes=[]
        for attempt in range(2):
            restore_checkpoint(args.experiment/'checkpoint.pt',model,optimizer,provenance=provenance)
            train(model,optimizer,datasets['train'],indices['train'],start_update=saved['updates'],updates=saved['updates']+1,
                seed=settings['seed'],device='cuda',log_path=output/('restore_step_'+str(attempt)+'.jsonl'))
            hashes.append(parameter_hash(model,trainable=True))
        report=dict(mode='verify',prediction_max_delta=deltas,trainable_hash_matches=matched,
            exact_next_update=hashes[0]==hashes[1],next_update_hashes=hashes,validation=actual,
            passed=matched and max(deltas.values())<=1e-5 and hashes[0]==hashes[1])
        write(output/'report.json',report)
        if not report['passed']:raise RuntimeError('Fresh-process restoration failed')
    else:
        preflight=visual_preflight(model,datasets['train'][indices['train'][0]],'cuda')
        write(output/'preflight.json',preflight)
        if not preflight['passed']:raise RuntimeError('Visual diagnostic preflight failed')
        original=parameter_hash(model,trainable=True);frozen=parameter_hash(model,trainable=False)
        report=dict(schema='morai-visual-probe-report-v2',preflight=preflight,fit_gate=settings['fit_gate'],
            before={split:evaluate_split(split,output/(split+'_predictions_before.json')) for split in indices})
        write(output/'progress.json',dict(phase='training',completed_updates=0))
        def on_update(step):
            if step%100==0:write(output/'progress.json',dict(phase='training',completed_updates=step))
            if step%300==0 and step<settings['updates']:
                save_checkpoint(output/('checkpoint_'+str(step)+'.pt'),model,optimizer,updates=step,
                    provenance=provenance,settings=settings)
        train(model,optimizer,datasets['train'],indices['train'],start_update=0,updates=settings['updates'],
            seed=settings['seed'],device='cuda',log_path=output/'training.jsonl',on_update=on_update)
        save_checkpoint(output/'checkpoint.pt',model,optimizer,updates=settings['updates'],provenance=provenance,settings=settings)
        write(output/'progress.json',dict(phase='evaluating',completed_updates=settings['updates']))
        report['after']={split:evaluate_split(split,output/(split+'_predictions.json')) for split in indices}
        report.update(optimizer_updates=settings['updates'],trainable_hash=parameter_hash(model,trainable=True),
            frozen_hash=parameter_hash(model,trainable=False),peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        report['checks']=dict(trainable_changed=report['trainable_hash']!=original,frozen_unchanged=report['frozen_hash']==frozen,
            finite_metrics=all(np.isfinite(r['objective']) for r in report['after'].values()),
            small_train_objective_decreased=report['after']['train']['objective']<report['before']['train']['objective'])
        report['fit_gate_passed']=all(report['after']['train'][k]<=v for k,v in settings['fit_gate'].items())
        if not all(report['checks'].values()):raise RuntimeError('Training integrity check failed: '+str(report['checks']))
        report['completed_wall_time']=time.time()
        write(output/'report.json',report)
        write(output/'progress.json',dict(phase='complete',completed_updates=settings['updates']))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
