"""Frozen native VE -> cached light tokens -> fixed-budget three-state head.

Run from VLDrive with the same environment as the MORAI perception recipe.
No simulator connection or control publishing. Only the supported, admitted
1/16/33 labels are used; held-out laps are never used for gradient updates.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Subset

from data.src.loaders.morai_dataloader import make_morai_loader
from data.src.loaders.morai_sensor_record_dataset import PathResolver, SensorRecordDataset, load_publications
from data.src.loaders.morai_route_permission import RoutePermissionDataset
from models.ve.signal_state import (SIGNAL_CODES, SignalStateHead, extract_signal_feature,
                                    encode_checkpoint, decode_checkpoint, decode_signal)
from training.vision_encoder.recipes.route_permission import write, save
from training.vision_encoder.utils.morai_runtime_state import nested_state_digest
from training.vision_encoder.utils.morai_transfer_metrics import to_device
from utils.common.runtime import deterministic_pointpillar_reductions
from utils.evaluation.morai_transfer_evaluation import _load_model


def signal_target(meta):
    bits = meta.get('raw_signal_bits')
    return SIGNAL_CODES.index(bits) if meta.get('valid') and type(bits) is int and bits in SIGNAL_CODES else -1


def metrics(logits, labels, metadata):
    pred = logits.argmax(1).cpu().tolist(); truth = labels.cpu().tolist()
    codes = decode_signal(logits)
    def subset(indices):
        cm = [[0]*3 for _ in range(3)]
        for i in indices: cm[truth[i]][pred[i]] += 1
        denied = [i for i in indices if truth[i] != 2]
        accepted = [i for i in indices if codes[i]['code'] is not None]
        return dict(frames=len(indices), confusion_gt_rows_pred_columns=cm,
            accuracy=sum(truth[i] == pred[i] for i in indices)/len(indices) if indices else None,
            denied_frames=len(denied), false_left_allow_argmax=sum(pred[i] == 2 for i in denied),
            false_left_allow_confidence_0_9=sum(codes[i]['code'] == 33 for i in denied),
            accepted_confidence_0_9=len(accepted),
            accepted_accuracy=sum(truth[i] == pred[i] for i in accepted)/len(accepted) if accepted else None,
            left_allow_frames=sum(truth[i] == 2 for i in indices),
            left_allow_accepted_correct=sum(truth[i] == 2 and codes[i]['code'] == 33 for i in indices))
    groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        distance = meta.get('stopline_distance_m')
        band = ('unknown' if distance is None else '0_10m' if distance < 10 else
                '10_20m' if distance < 20 else '20_30m' if distance < 30 else '30m_plus')
        groups['distance:'+band].append(i)
        groups['state_distance:'+str(meta['raw_signal_bits'])+':'+band].append(i)
        groups['manifest:'+meta['manifest_path']].append(i)
    return dict(signal_codes=list(SIGNAL_CODES), overall=subset(list(range(len(truth)))),
                by_group={key:subset(indices) for key,indices in sorted(groups.items())})


def run(args):
    if args.output.exists(): raise FileExistsError('Use a fresh output directory')
    if args.epochs < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Invalid training budget')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True)
    start = time.monotonic()
    write(args.output/'status.json',dict(phase='dataset_admission',training_updates=0))
    resolver = PathResolver(config.get('path_mappings', []))
    index, manifests = load_publications(config['manifest_indices'],resolver)
    datasets = {split: RoutePermissionDataset(SensorRecordDataset(manifests[split],resolver,
                velocity_source='ego_topic'),index['manifest_scenarios']) for split in ('train','val')}
    metadata = {split: dataset.metadata for split,dataset in datasets.items()}
    selected = {split:[i for i,m in enumerate(metadata[split]) if signal_target(m)>=0] for split in datasets}
    for split in datasets:
        if {signal_target(metadata[split][i]) for i in selected[split]} != {0,1,2}:
            raise ValueError('Every split must contain all three supported states')
    identities = {split:{(m['manifest_path'],m['frame_id']) for m in metadata[split]} for split in datasets}
    train_paths={m['manifest_path'] for m in metadata['train']}
    val_paths={m['manifest_path'] for m in metadata['val']}
    if identities['train'] & identities['val'] or train_paths & val_paths:
        raise ValueError('Training and held-out manifests overlap')
    base,payload,sha = _load_model(resolver(config['checkpoint']),config['checkpoint_sha256'],args.device)
    contract = payload.get('morai_encoder_contract',{})
    if (contract.get('velocity_input')!='causal_ego_topic_speed_mps_50ms'
            or any(ds.base_dataset.map_basis!=contract.get('map_basis') for ds in datasets.values())):
        raise ValueError('Signal data/base encoder contract mismatch')
    del payload
    base.requires_grad_(False).eval()
    before=nested_state_digest(base.state_dict())
    provenance=dict(schema='morai_signal_state_pilot_v1',base_checkpoint=config['checkpoint'],
        base_checkpoint_sha256=sha,encoder_frozen=True,signal_codes=list(SIGNAL_CODES),
        args=vars(args),publications=index['publications'],
        coverage={s:dict(admitted=len(datasets[s]),selected=len(selected[s]),
             labels=dict(Counter(str(metadata[s][i]['raw_signal_bits']) for i in selected[s]))) for s in datasets},
        feature_input='native traffic-light token from unchanged sensor/ego/route forward',
        forbidden_inputs=['GT signal bits','GT light ID','GT stopline distance','GT permission'],
        selection_policy='fixed epoch budget; final epoch only; no validation-based model selection',
        live_expert_connected=False)
    write(args.output/'config.json',provenance)
    cached={}; forwards=0; probes=[]
    with torch.no_grad(),deterministic_pointpillar_reductions():
        for split,ds in datasets.items():
            features=[]; metas=[]; targets=[]; checked=set()
            data=make_morai_loader(Subset(ds.base_dataset,selected[split]),workers=args.workers,
                                  shuffle=False,persistent_workers=False,
                                  generator=torch.Generator().manual_seed(args.seed))
            for pos,batch in enumerate(data):
                inputs,gt=to_device(batch,args.device); meta=metadata[split][selected[split][pos]]
                if int(gt['frame_id'].item())!=meta['frame_id'] or int(gt['sensor_time_ns'].item())!=meta['sensor_time_ns']:
                    raise ValueError('Cached feature/label identity mismatch')
                token,native=extract_signal_feature(base,inputs); forwards+=1
                label=signal_target(meta)
                if label not in checked:
                    check=base(inputs);forwards+=1
                    errors=[float((a-b).abs().max()) for a,b in zip(native,check)]
                    if len(native)!=len(check) or any(errors):
                        write(args.output/'native_repeat_failure.json',dict(errors=errors,split=split,metadata=meta))
                        raise ValueError('Native repeat differed: '+str(errors))
                    checked.add(label);probes.append(dict(split=split,code=SIGNAL_CODES[label],native_max_abs=errors))
                if len(token)!=1 or not torch.isfinite(token).all():raise ValueError('Invalid cached token')
                features.append(token.cpu()); targets.append(label); metas.append(meta)
                if (pos+1)%100==0 or pos+1==len(selected[split]):
                    status=dict(phase='feature_extraction',split=split,completed=pos+1,total=len(selected[split]),
                                model_forwards=forwards,elapsed_seconds=time.monotonic()-start,training_updates=0)
                    write(args.output/'status.json',status);print(json.dumps(status),flush=True)
            cached[split]=(torch.cat(features),torch.tensor(targets),metas)
            save(args.output/(split+'_features.pt'),dict(features=cached[split][0],labels=cached[split][1],
                 base_checkpoint_sha256=sha,metadata=metas))
    if before != nested_state_digest(base.state_dict()):raise ValueError('Native encoder changed')
    del base;torch.cuda.empty_cache()
    # Small head is deliberately trained on CPU: cached tokens need no GPU.
    head=SignalStateHead(feature_dim=cached['train'][0].shape[1]); optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr)
    x,y,_=cached['train']; weights=len(y)/(3*torch.bincount(y,minlength=3).float()); updates=0
    with (args.output/'training.jsonl').open('x',buffering=1) as log:
        for epoch in range(1,args.epochs+1):
            head.train(); order=torch.randperm(len(y)); total=0.
            for ids in order.split(64):
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(head(x[ids]),y[ids],weight=weights)
                if not torch.isfinite(loss):raise ValueError('Nonfinite signal loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1;total+=float(loss)*len(ids)
            row=dict(epoch=epoch,updates=updates,train_weighted_ce=total/len(y))
            log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
    head.eval();sidecar=encode_checkpoint(head,sha); restored=decode_checkpoint(sidecar,sha)
    result={}
    with torch.no_grad():
        for split,(x,y,metas) in cached.items():
            logits=head(x);torch.testing.assert_close(logits,restored(x),rtol=0,atol=0)
            result[split]=metrics(logits,y,metas)
            with (args.output/(split+'_predictions.jsonl')).open('x') as log:
                for meta,label,probability,decoded in zip(metas,y.tolist(),logits.softmax(1).tolist(),decode_signal(logits)):
                    log.write(json.dumps(dict(metadata=meta,gt_code=SIGNAL_CODES[label],
                              probabilities=probability,decoded=decoded),ensure_ascii=False)+'\n')
    save(args.output/'signal_head_final.pt',sidecar)
    write(args.output/'report.json',dict(status='COMPLETE',training_updates=updates,encoder_training_updates=0,
        encoder_digest_before=before,encoder_unchanged=True,base_checkpoint_sha256=sha,model_forwards=forwards,
        probes=probes,metrics=result,elapsed_seconds=time.monotonic()-start,live_expert_connected=False,
        deployment_ready=False,limitations=['One protected-left intersection and three known phases only.',
        'Same signal schedule may correlate state with speed/stopline distance; no visual causality claim.',
        'Softmax threshold 0.9 is diagnostic, not calibrated unknown-phase rejection.',
        'No live signal-ID association or Expert control integration in this offline experiment.']))
    write(args.output/'status.json',dict(status='COMPLETE',report=str(args.output/'report.json')))
    print(json.dumps(dict(status='COMPLETE',report=str(args.output/'report.json'),val=result['val']['overall'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=42);run(p.parse_args())


if __name__=='__main__':main()
