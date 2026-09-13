#!/usr/bin/env python3
"""Isolated corrected-RoPE A/B: freeze policy alone changes, never production weights.

A = original MDP only; B = ALL originally trainable parameters, including the
used TinyLlama decoder/text embeddings (not just DDIA attention projections).
Both execute the same training-mode Gumbel/dropout/auxiliary forward graph.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('LAVIS_IMPORT_PROFILE', 'morai')
OLD = ROOT/'outputs/morai_paper_last_encoder_20260910T055054Z'
AUDIT = ROOT/'outputs/diagnosis_controlled_experiments_20260911'
ORIGINAL_SHA = 'a63a619b2565d2bb0fba13ddfe305779f344bf67d0fbd9aafa7f6476c9f14b9a'
SELECTION_SHA = '192f5afde3ef43c3d3dd861995a7e1506e2e2e50c97a034edd00b3b7edc3df51'
REFERENCE_SHA = '310d04713897cf850f2b3d7a86b32c9f830a83fa633f196814991d2838dd4dc5'
MDP = 'language_model.waypoints_predictor.'
GROUPS = dict(mdp=MDP, selector='connector.ccdp.', mefa='connector.temporal_encoder.',
    qformer='connector.qformer.', projection='connector.qformer_to_language.',
    reconstruction='connector.reconstruction_transformer.',
    decoder='language_model.tinyllama.model.layers.', embeddings='language_model.tinyllama.model.embed_tokens.')


def write(path, value):
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    pending.replace(path)


def order_for(indices, steps, seed=607):
    import torch
    return [indices[torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed+s//len(indices))).tolist()[s%len(indices)]]
            for s in range(steps)]


def set_policy(model, arm):
    if arm not in ('A','B'):
        raise ValueError('Unknown arm')
    original={name for name,p in model.named_parameters() if p.requires_grad}
    if not any(name.startswith(MDP) for name in original):
        raise ValueError('Original policy has no MDP')
    if arm=='A':
        for name,p in model.named_parameters():
            p.requires_grad_(name in original and name.startswith(MDP))
    # Never call connector.eval() here: freeze is a gradient policy, not a mode.
    return original


def manifest(model):
    values=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype),numel=p.numel(),trainable=p.requires_grad)
            for n,p in model.named_parameters()]
    return dict(parameters=values,trainable_tensors=sum(v['trainable'] for v in values),
                trainable_numel=sum(v['numel'] for v in values if v['trainable']),
                trainable_name_sha256=hashlib.sha256('\n'.join(v['name'] for v in values if v['trainable']).encode()).hexdigest())


def gradient_groups(model):
    import torch
    totals={key:None for key in GROUPS}; counts=Counter()
    for name,p in model.named_parameters():
        if p.grad is None:
            continue
        key=next((k for k,prefix in GROUPS.items() if name.startswith(prefix)),'other')
        value=p.grad.detach().reshape(-1)
        square=value.new_zeros((),dtype=torch.float32)
        for chunk in value.split(1048576):
            square.add_(chunk.float().square().sum())
        totals[key]=square if totals.get(key) is None else totals[key]+square
        counts[key]+=1
    return {key:dict(l2=0. if value is None else float(value.sqrt()),gradient_tensors=counts[key])
            for key,value in totals.items()}


def sample_indices(length, limit=4096):
    import torch
    return torch.linspace(0,length-1,min(length,limit),dtype=torch.long)


def representatives(model):
    named=dict(model.named_parameters())
    chosen=[]
    for prefix in (MDP+'0.weight',MDP+'2.weight',MDP+'2.bias','connector.ccdp.score_mlp.2.bias',
                   'connector.qformer_to_language.','connector.temporal_encoder.',
                   'language_model.tinyllama.model.layers.0.self_attn.q_proj.',
                   'language_model.tinyllama.model.embed_tokens.'):
        candidates=[name for name,p in named.items() if name.startswith(prefix) and p.requires_grad]
        if candidates:
            chosen.append(candidates[0])
    return list(dict.fromkeys(chosen))


def sampled_values(model, names, *, optimizer=None, gradients=False):
    import torch
    named=dict(model.named_parameters())
    master_by_id={} if optimizer is None else {id(p):m for p,m in zip(optimizer.parameters,optimizer.masters)}
    result={}
    for name in names:
        p=named[name]
        value=(p.grad if gradients else master_by_id.get(id(p),p))
        if value is None:
            result[name]=None
        else:
            indices=sample_indices(value.numel()).to(value.device)
            result[name]=value.detach().reshape(-1)[indices].float().cpu().clone()
    return result


def changes(before, after):
    import torch
    return {name:dict(sampled_elements=a.numel(),delta_l2=float(torch.linalg.vector_norm(after[name]-a)),
                      max_abs=float((after[name]-a).abs().max()),changed_elements=int(torch.count_nonzero(after[name]-a)))
            for name,a in before.items()}


def same_initial(reference, current):
    keys=('common','initial_original_trainable_hash','initial_original_frozen_hash','before','before_all_train','train_forward')
    mismatch=[key for key in keys if key not in reference or key not in current or reference[key]!=current[key]]
    if mismatch:
        raise ValueError('Paired initial contract/forward mismatch: '+', '.join(mismatch))


def same_actual_training_forward(reference, losses, signature):
    keys=('loss','waypoint_loss','pruning_loss','reconstruction_loss','sampled_retention',
          'soft_retention','saturated_keep_fraction','retention_logit_loss')
    if any(key not in reference or key not in losses or reference[key]!=losses[key] for key in keys):
        raise ValueError('Paired first grad-enabled training loss/selector values differ')
    if reference.get('actual_first_training_forward')!=signature:
        raise ValueError('Paired first grad-enabled trajectory/selected-count differs')


def check_rope():
    import torch
    from lavis.models.vldrive.ddia_attention import apply_rope
    x=torch.ones(1,1,2,64,dtype=torch.bfloat16)
    positions=torch.tensor([[1200,1201]])
    frequency=1./(10000.**(torch.arange(0,64,2).float()/64))
    angle=positions.float()[...,None]*frequency
    angle=torch.cat((angle,angle),-1)[:,None]
    half=torch.cat((-x[...,32:],x[...,:32]),-1)
    expected=x*angle.cos().to(x.dtype)+half*angle.sin().to(x.dtype)
    actual=apply_rope(x,positions)
    if not torch.equal(actual,expected) or torch.equal(actual[:,:,0],actual[:,:,1]):
        raise ValueError('Expected fixed FP32-trigonometry RoPE at long positions')
    return dict(positions=[1200,1201],fp32_trigonometry_then_bf16_cast=True,exact_reference_match=True)


def dataset_contract(settings):
    import numpy as np
    import torch
    from lavis.common.morai_runtime import sha256
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    from lavis.runners.vldrive_trainer import stride_indices
    dataset=MoraiExpertVLDriveDataset(settings['dataset'],cache_root=settings['feature_cache'],
        split_file=settings['split_file'],instructions=settings['instructions'],split='train',
        sequence_length=11,instruction_boundary_policy='exclude')
    metadata=json.loads((AUDIT/'all_frame_readout.json').read_text())
    if dataset.describe()!=metadata['original_dataset_description']:
        raise ValueError('Current dataset/cache differs from the original 636-window contract')
    if sha256(AUDIT/'audit_selection.json')!=SELECTION_SHA or sha256(AUDIT/'all_frame_readout.npz')!=REFERENCE_SHA:
        raise ValueError('Independent selection/GT reference hash mismatch')
    for entry in dataset.entries:
        dataset._array(entry['visual_feature'],(106,256),'visual feature')
        dataset._array(entry['future_trajectory'],(5,2),'actual future trajectory')
    # Historical LLM features/predictions are NOT loaded or used as a baseline.
    with np.load(AUDIT/'all_frame_readout.npz',allow_pickle=False) as values:
        expected={k:values[k] for k in ('target','frame_ids','anchor_frames','native_sync_frames')}
    evaluation=stride_indices(len(dataset),5)
    if len(evaluation)!=636:
        raise ValueError('Expected exactly 636 original training windows')
    for row,index in enumerate(evaluation):
        begin,end=dataset.windows[index]; entries=dataset.entries[begin:end]
        gt=torch.stack([dataset._array(e['future_trajectory'],(5,2),'GT') for e in entries]).numpy()
        if (not np.array_equal(gt,expected['target'][row])
            or [e['frame_id'] for e in entries]!=expected['frame_ids'][row].tolist()
            or [e['native_sync_frame'] for e in entries]!=expected['native_sync_frames'][row].tolist()):
            raise ValueError('Original 6996-frame GT/context alignment mismatch')
    selection=json.loads((AUDIT/'audit_selection.json').read_text())
    selected=[evaluation[i] for i in selection['indices']]
    if selected!=selection['dataset_window_indices'] or expected['anchor_frames'][selection['indices']].tolist()!=selection['anchor_frames']:
        raise ValueError('Audited 32-window index mapping mismatch')
    return dataset,evaluation,selected,selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=('A','B'),required=True)
    parser.add_argument('--steps',type=int,choices=(2,1280),default=1280)
    parser.add_argument('--eval-limit',type=int,default=0,help='Only steps=2 smoke: cap selected/final evaluation')
    parser.add_argument('--paired-reference',type=Path,help='B requires A output directory from this same bounded budget')
    parser.add_argument('--prepare-only',action='store_true',help='CPU/source/data/RoPE checks; zero GPU use and optimizer updates')
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    smoke=args.steps==2
    if (args.eval_limit and (not smoke or not 1<=args.eval_limit<=8)) or (args.arm=='B' and not args.paired_reference and not args.prepare_only):
        raise ValueError('Only labeled 2-step smoke may limit evaluation; B needs paired A reference')
    output=args.output.resolve()
    if ROOT/'outputs' not in output.parents or output.exists():
        raise ValueError('Use a NEW output directory below LAVIS/outputs')
    output.mkdir(parents=True)
    from lavis.common.morai_runtime import configure_runtime,bootstrap,sha256
    configure_runtime()
    import numpy as np
    import torch
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig,VLDriveTrainingModel
    from lavis.runners.vldrive_trainer import (make_optimizer,warm_start_model_checkpoint,
        save_model_checkpoint,preserve_rng_state,parameter_hash,seed_all,source_hashes)
    from lavis.runners.morai_trainer import batch,autocast
    # Stable metric functions from the preceding immutable diagnostics.
    sys.path.insert(0,str(AUDIT))
    from model_readout_train import summarize,gate
    torch.set_num_threads(4)
    seed_all(607)
    started=time.monotonic()
    def status(phase,**values):
        row=dict(phase=phase,arm=args.arm,steps=args.steps,smoke=smoke,
                 elapsed_s=time.monotonic()-started,utc=datetime.now(timezone.utc).isoformat(),**values)
        write(output/'progress.json',row); print(json.dumps(row),flush=True)
    status('validating')
    settings=json.loads((OLD/'resolved_config.json').read_text())
    settings['split_file']=str((ROOT/settings['split_file']).resolve())
    config=VLDriveTrainingConfig(**settings['model']); config.validate()
    if (config.training_variant!='paper_independent' or config.selector_task_gradient_mode!='legacy'
        or config.inference_selection_mode!='argmax' or config.retention_logit_weight!=0.
        or config.reconstruction_reduction!='sum' or config.detach_reconstruction_loss_mask
        or config.input_mode!='cached' or config.precision!='bf16'):
        raise ValueError('Unmodified original paper-objective configuration required')
    rope=check_rope()
    dataset,evaluation,selected,selection=dataset_contract(settings)
    if sha256(Path(config.visual_checkpoint))!=dataset.checkpoint_sha256 or sha256(OLD/'model-last.pt')!=ORIGINAL_SHA:
        raise ValueError('Original vision/downstream checkpoint hash mismatch')
    source=source_hashes(ROOT)
    diagnostic_paths=[Path(__file__).resolve(),ROOT/'tests/vldrive/test_matched_upstream_probe.py',
        AUDIT/'model_readout_train.py',AUDIT/'audit_selection.json',OLD/'resolved_config.json']
    diagnostic_hashes={str(p):sha256(p) for p in diagnostic_paths}
    order=order_for(selected,args.steps)
    if not smoke and (len(Counter(order))!=32 or set(Counter(order).values())!={40}):
        raise ValueError('Full pilot requires exactly 40 exposures per window')
    common=dict(steps=args.steps,seed=607,per_step_stochastic_seed='607 + zero_based_update',
        lr=1e-5,lr_schedule='constant',weight_decay=.06,clip_norm=10.,optimizer='CPU FP32-master AdamW; fresh moments',
        batch_windows=1,frames_per_window=11,loss='original all-frame L1 + 10*pruning + reconstruction',
        training_forward_mode='model.train in BOTH arms; Gumbel and all dropout remain enabled',
        evaluation_mode='model.eval dynamic argmax',original_checkpoint_sha256=ORIGINAL_SHA,
        original_optimizer_restored=False,selector_reset=False,bert_reinitialization=False,
        visual_cache_sha256=dataset.cache_meta_sha256,visual_checkpoint_sha256=dataset.checkpoint_sha256,
        selected_indices=selected,anchor_frames=selection['anchor_frames'],sample_order=order,
        evaluation_indices=evaluation[:args.eval_limit] if args.eval_limit else evaluation,
        selected_evaluation_indices=selected[:args.eval_limit] if args.eval_limit else selected,
        source=source,diagnostic_source=diagnostic_hashes,rope=rope,smoke=smoke,
        validation_used=False,production_changed=False,
        inference_gate='selected all352: L1<.05, turnFDE<.5, turnDirection>=.95, stopFDE<.1',
        limitations=['Failure at 1280 updates does not prove impossible fitting.',
            'Global clip denominator changes with trainable scope; log MDP group gradients and actual updates.',
            'Equal stochastic seeds do not guarantee identical masks after learned tensor lengths diverge.',
            'All6996 are original train occurrences, not validation.',
            'This original-checkpoint warm start keeps its learned selector state; no reset or architecture change.'])
    write(output/'protocol.json',common)
    if args.prepare_only:
        status('cpu_prepared',optimizer_updates=0,verified_original_occurrences=6996)
        return
    import fcntl
    lock_path=ROOT/'.runtime/training_runs/last_encoder.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    training_lock=lock_path.open('a') if args.lock_fd is None else os.fdopen(os.dup(args.lock_fd),'a')
    held,expected=os.fstat(training_lock.fileno()),lock_path.stat()
    if (held.st_dev,held.st_ino)!=(expected.st_dev,expected.st_ino):
        raise ValueError('Inherited lock does not match product training lock')
    fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend=bootstrap(ROOT.parent)
    provenance=dict(backend=backend,source=source,datasets={'train':dataset.describe()},diagnostic_source=diagnostic_hashes)
    status('loading_original_model')
    seed_all(607)
    model=VLDriveTrainingModel(config,device='cuda')
    original_manifest=manifest(model)
    # Restore BEFORE A freezes upstream, so source trainable keys and frozen
    # pretrained/random assets undergo the complete original-policy validation.
    transfer=warm_start_model_checkpoint(OLD/'model-last.pt',model,provenance=provenance,reset_prefixes=(),
        transfer_reason='Authorized diagnostic: corrected RoPE common to A/B; original data/model policy; new optimizer, no reset.')
    initial_original_trainable=parameter_hash(model,trainable=True)
    initial_original_frozen=parameter_hash(model,trainable=False)
    original_metadata=torch.load(OLD/'model-last.pt',map_location='cpu',weights_only=False,mmap=True)
    if (initial_original_trainable!=original_metadata['trainable_hash']
        or initial_original_frozen!=original_metadata['frozen_hash']):
        raise RuntimeError('Transferred full original state does not match recorded source hashes')
    del original_metadata
    set_policy(model,args.arm)
    arm_manifest=manifest(model)
    frozen_start=parameter_hash(model,trainable=False)
    write(output/'model_policy.json',dict(original=original_manifest,active=arm_manifest,transfer=transfer))
    selected_eval=common['selected_evaluation_indices']

    @torch.no_grad()
    @preserve_rng_state()
    def evaluate(indices,label):
        model.eval(); predictions=[]; targets=[]; frames=[]; counts=[]
        for index in indices:
            sample=dataset[index]; item=batch(sample,'cuda')
            with autocast(model):
                result=model.predict(item['model_inputs'],compute_auxiliary=False)
            predictions.append(result['trajectory'][0].float().cpu().numpy())
            targets.append(item['training_targets']['future_trajectory'][0].float().cpu().numpy())
            frames.append(sample['metadata']['frame_ids'].numpy())
            counts.append(result['selected_count'][0].cpu().numpy())
            if len(predictions)%32==0 or len(predictions)==len(indices):
                status('evaluating',evaluation=label,completed_windows=len(predictions),total_windows=len(indices))
        predictions,targets,frames,counts=map(np.stack,(predictions,targets,frames,counts))
        metrics=summarize(predictions,targets)
        np.savez_compressed(output/(label+'_predictions.npz'),prediction=predictions,target=targets,
            frame_ids=frames,selected_count=counts,dataset_window_indices=np.asarray(indices))
        write(output/(label+'_metrics.json'),metrics)
        model.train()
        return metrics

    # Deterministic selected right-turn anchor2850, independent of prediction.
    probe_index=selection['dataset_window_indices'][selection['anchor_frames'].index(2850)]
    @preserve_rng_state()
    def forward_signature():
        seed_all(607); model.train()
        with torch.no_grad(),autocast(model):
            result=model.predict(batch(dataset[probe_index],'cuda')['model_inputs'],compute_auxiliary=True)
        return dict(trajectory=result['trajectory'].float().cpu().tolist(),
            selected_count=result['selected_count'].cpu().tolist(),
            scalar={k:float(result[k]) for k in ('pruning_loss','reconstruction_loss','cycle_loss')})

    status('initial_selected_evaluation')
    before=evaluate(selected_eval,'before')
    before_all=evaluate(common['evaluation_indices'],'before_all_train')
    initial=dict(common=common,initial_original_trainable_hash=initial_original_trainable,
        initial_original_frozen_hash=initial_original_frozen,before=before,before_all_train=before_all,
        train_forward=forward_signature())
    if args.arm=='B':
        same_initial(json.loads((args.paired_reference/'initial.json').read_text()),initial)
        for label in ('before','before_all_train'):
            with np.load(args.paired_reference/(label+'_predictions.npz')) as ref,np.load(output/(label+'_predictions.npz')) as cur:
                if set(ref.files)!=set(cur.files) or any(not np.array_equal(ref[k],cur[k]) for k in ref.files):
                    raise ValueError('Paired corrected-RoPE zero-step arrays differ: '+label)
    write(output/'initial.json',initial)
    names=representatives(model)

    @preserve_rng_state()
    def loss_probe(label):
        components={}; vectors={}
        for component in ('waypoint_loss','pruning_loss','reconstruction_loss','loss'):
            seed_all(607); model.train(); model.zero_grad(set_to_none=True)
            with autocast(model):
                losses=model(batch(dataset[probe_index],'cuda'))
                objective=losses[component]*(10. if component=='pruning_loss' else 1.)
            row=dict(value=float(objective.detach()),requires_grad=objective.requires_grad)
            if objective.requires_grad:
                objective.backward()  # Reentrant HF checkpointing supports backward, not autograd.grad(inputs).
                row['gradient_groups']=gradient_groups(model)
                vectors[component]=sampled_values(model,names,gradients=True)
            else:
                row['gradient_groups']={}
                vectors[component]={n:None for n in names}
            components[component]=row
        waypoint=components['waypoint_loss']['gradient_groups']
        if not waypoint.get('mdp',{}).get('l2',0)>0:
            raise RuntimeError('Actual waypoint backward did not reach MDP')
        if args.arm=='B' and not all(waypoint.get(k,{}).get('l2',0)>0 for k in ('decoder','qformer')):
            raise RuntimeError('Actual waypoint backward did not reach original upstream trainables')
        if any(not math.isfinite(v['l2']) for c in components.values() for v in c['gradient_groups'].values()):
            raise RuntimeError('Nonfinite component gradient')
        cosines={}
        for other in ('pruning_loss','reconstruction_loss','loss'):
            cosines[other]={}
            for name,a in vectors['waypoint_loss'].items():
                b=vectors[other][name]
                if a is not None and b is not None and float(a.norm())>0 and float(b.norm())>0:
                    cosines[other][name]=float(torch.dot(a,b)/(a.norm()*b.norm()))
        result=dict(probe_anchor=2850,components=components,waypoint_cosines_on_sampled_parameter_coordinates=cosines,
                    note='Per-component real loss.backward on the same seeded training forward; zero optimizer steps.')
        model.zero_grad(set_to_none=True); write(output/(label+'_gradient_probe.json'),result)
        return result

    status('actual_backward_preflight')
    loss_probe('initial')
    optimizer=make_optimizer(model,lr=1e-5,weight_decay=.06,state_device='cpu')
    curve=[]
    status('training',update=0)
    training_start=time.monotonic()
    with (output/'training.jsonl').open('x') as log:
        for step0,index in enumerate(order):
            step=step0+1
            seed_all(607+step0); model.train(); optimizer.zero_grad(set_to_none=True)
            actual_first={}; handles=[]
            if step==1:
                def capture_connector(module,inputs,result):
                    actual_first['selected_count']=result['selected_count_per_frame'].detach().cpu().tolist()
                def capture_language(module,inputs,result):
                    actual_first['trajectory']=result.predicted_waypoints.detach().reshape(1,11,5,2).float().cpu().tolist()
                handles=[model.connector.register_forward_hook(capture_connector),
                         model.language_model.register_forward_hook(capture_language)]
            try:
                with autocast(model):
                    losses=model(batch(dataset[index],'cuda'))
            finally:
                for handle in handles:
                    handle.remove()
            if not losses['loss'].requires_grad or not bool(torch.isfinite(losses['loss'])):
                raise RuntimeError('Loss must be finite and connected to actual trainables')
            if step==1 and args.arm=='B':
                with (args.paired_reference/'training.jsonl').open() as reference_log:
                    reference_first=json.loads(reference_log.readline())
                same_actual_training_forward(reference_first,{k:float(v.detach()) for k,v in losses.items()},actual_first)
            losses['loss'].backward()
            observe=step==1 or step%32==0 or step==args.steps
            groups=gradient_groups(model) if observe else None
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.,error_if_nonfinite=True)
            if observe:
                before_parameters=sampled_values(model,names)
                before_masters=sampled_values(model,names,optimizer=optimizer)
                after_clip=gradient_groups(model)
            optimizer.step()
            row=dict(update=step,index=index,lr=1e-5,stochastic_seed=607+step0,backward_calls=1,
                preclip_global_norm=float(norm),clip_coefficient=float(torch.clamp(10./(norm+1e-6),max=1.)),
                elapsed_training_s=time.monotonic()-training_start,**{k:float(v.detach()) for k,v in losses.items()})
            if step==1:
                row['actual_first_training_forward']=actual_first
                row['paired_grad_enabled_forward_exact_before_optimizer']=(args.arm=='B')
            if observe:
                row.update(preclip_gradient_groups=groups,postclip_gradient_groups=after_clip,
                    actual_model_sampled_update=changes(before_parameters,sampled_values(model,names)),
                    actual_fp32_master_sampled_update=changes(before_masters,sampled_values(model,names,optimizer=optimizer)))
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
            del losses
            optimizer.zero_grad(set_to_none=True)
            if step%5==0 or observe:
                status('training',update=step,waypoint_loss=row['waypoint_loss'],sampled_retention=row.get('sampled_retention'))
            if step%160==0 or step==args.steps:
                metrics=evaluate(selected_eval,'after_%04d'%step)
                curve.append(dict(update=step,metrics=metrics,gate=gate(metrics) if not smoke else None))
                write(output/'curve.json',curve)
                status('evaluated',update=step,metrics=metrics)
    status('final_full_original_train_evaluation',update=args.steps)
    all_metrics=evaluate(common['evaluation_indices'],'final_all_train')
    loss_probe('final')
    final_trainable_hash=parameter_hash(model,trainable=True)
    if parameter_hash(model,trainable=False)!=frozen_start:
        raise RuntimeError('Frozen upstream/unused model weights changed')
    counts=sorted({int(s['step']) for s in optimizer.optimizer.state.values()})
    master_match=all(torch.equal(p.detach().cpu(),m.detach().to(p.dtype)) for p,m in zip(optimizer.parameters,optimizer.masters))
    parameters_by_id={id(p):n for n,p in model.named_parameters()}
    no_optimizer_state=[parameters_by_id[id(p)] for p,m in zip(optimizer.parameters,optimizer.masters)
                        if m not in optimizer.optimizer.state]
    if counts!=[args.steps] or not master_match:
        raise RuntimeError('Optimizer step count or FP32-master/BF16 contract failed')
    save_model_checkpoint(output/'model-final-copy.pt',model,updates=args.steps,provenance=provenance,
        settings=dict(common,arm=args.arm,base_original_checkpoint=str(OLD/'model-last.pt')),
        metrics=curve[-1]['metrics'])
    for path,digest in diagnostic_hashes.items():
        if sha256(Path(path))!=digest:
            raise RuntimeError('Diagnostic source changed during execution')
    if source_hashes(ROOT)!=source or sha256(OLD/'model-last.pt')!=ORIGINAL_SHA:
        raise RuntimeError('Production source or original checkpoint changed')
    dataset._verified.clear()
    for entry in dataset.entries:
        dataset._array(entry['visual_feature'],(106,256),'visual feature')
        dataset._array(entry['future_trajectory'],(5,2),'actual future trajectory')
    report=dict(complete=True,arm=args.arm,steps=args.steps,smoke=smoke,validation_used=False,
        selected_all_frames=curve[-1]['metrics'],gate=curve[-1]['gate'],all_original_train=all_metrics,
        initial_original_trainable_hash=initial_original_trainable,final_active_trainable_hash=final_trainable_hash,
        optimizer_step_counts=counts,master_cast_equals_model=master_match,frozen_parameters_unchanged=True,
        active_tensors_without_optimizer_state=no_optimizer_state,
        paired_grad_enabled_first_forward_exact=(args.arm=='B'),
        source_and_data_unchanged=True,production_installed=False,elapsed_s=time.monotonic()-started,
        all_original_train_windows=len(common['evaluation_indices']),
        warning='Diagnostic copy only. A saves MDP deltas relative to original checkpoint, not a standalone full driving model.')
    write(output/'report.json',report)
    status('complete',update=args.steps,gate=report['gate'])


if __name__=='__main__':
    main()
