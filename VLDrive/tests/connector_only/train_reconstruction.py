"""Isolate reconstruction training with a frozen selector/MEFA/Q-Former."""
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.connector_only.run import (
    ROOT, torch, np, QFormerMEFAConfig, QFormerMEFAConnector, load_records,
    save_checkpoint, sha256, write_json, emit,
)

PREFIXES = ('reconstruction_align.', 'reconstruction_transformer.',
            'mask_embedding', 'reconstruction_position_embedding')


def pack(rows, device):
    count = max(r['tokens'].shape[1] for r in rows)
    tokens = torch.zeros(len(rows), count, rows[0]['tokens'].shape[-1], device=device)
    ids = torch.full((len(rows),count),-1,dtype=torch.long,device=device)
    valid = torch.zeros(len(rows),count,dtype=torch.bool,device=device)
    for i,r in enumerate(rows):
        n = r['tokens'].shape[1]
        tokens[i,:n] = r['tokens'][0].to(device)
        ids[i,:n] = r['ids'][0].to(device)
        valid[i,:n] = r['valid'][0].to(device)
    return dict(reconstruction_features=torch.cat([r['target'] for r in rows]).to(device),
                language_tokens_flat=tokens,selected_indices_flat=ids,selected_valid_flat=valid,
                full_selection_mask_flat=torch.cat([r['mask'] for r in rows]).to(device))


def prepare(model, rows):
    packed=[]
    with torch.inference_mode():
        for row in rows:
            x=row['tensor'].cuda()
            output=model(x,compute_reconstruction=False)
            # Clone outside inference_mode below: ordinary tensors needed for backward.
            packed.append(dict(episode=row['episode'],split=row['split'],
                tokens=output['language_tokens'][:,-1].cpu(),ids=output['selected_indices'][:,-1].cpu(),
                valid=output['selected_valid_mask'][:,-1].cpu(),mask=output['selection_mask'][:,-1].cpu(),
                target=x[:,-1].cpu()))
    for r in packed:
        for k in ('tokens','ids','valid','mask','target'):
            r[k]=r[k].clone()
        if r['mask'].sum() >= r['mask'].numel():
            raise ValueError('No dropped reconstruction target')
    return packed


def evaluate(model, rows, mean_feature, corrupt=False):
    model.eval()
    results=[]
    with torch.inference_mode():
        for start in range(0,len(rows),4):
            group=rows[start:start+4]
            args=pack(group,'cuda')
            out=model._reconstruct(**args)
            target=args['reconstruction_features']
            dropped=1-args['full_selection_mask_flat']
            baseline=((target-mean_feature).square()*dropped.unsqueeze(-1)).sum((1,2))/(dropped.sum(1)*target.shape[-1])
            zero=None
            if corrupt:
                altered=dict(args,language_tokens_flat=torch.zeros_like(args['language_tokens_flat']))
                zero=model._reconstruct(**altered)['reconstruction_loss_per_frame']
            for j,r in enumerate(group):
                results.append(dict(episode=r['episode'],lrec=float(out['reconstruction_loss_per_frame'][j]),
                    mean_baseline=float(baseline[j]),zero_input_lrec=float(zero[j]) if zero is not None else None))
    return dict(lrec=float(np.mean([r['lrec'] for r in results])),
                mean_baseline=float(np.mean([r['mean_baseline'] for r in results])),
                beats_mean_frames=sum(r['lrec']<r['mean_baseline'] for r in results),frames=len(results),
                zero_input_lrec=float(np.mean([r['zero_input_lrec'] for r in results])) if corrupt else None,
                rows=results)


def main():
    source=ROOT/'outputs/logs/connector_mefa_dynamic_anneal'
    dest=ROOT/'outputs/logs/reconstruction_finetune_v1'
    dest.mkdir(exist_ok=False)
    torch.manual_seed(607); random.seed(607); torch.set_num_threads(4)
    state=torch.load(source/'model-last.pt',map_location='cpu',weights_only=True)
    cfg=dict(state['config'],epochs=200,learning_rate=3e-4,weight_decay=.01,
             output=str(dest.relative_to(ROOT)),experiment='frozen_frontend_reconstruction_only',
             trainable_prefixes=list(PREFIXES),batch_size=4,source_checkpoint_sha256=sha256(source/'model-last.pt'))
    write_json(dest/'experiment.json',cfg)
    model=QFormerMEFAConnector(QFormerMEFAConfig(**state['connector_config']))
    model.load_state_dict(state['state_dict'],strict=True)
    model.cuda().eval()
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(PREFIXES))
    for name in ('manifest.json','cache_source.json'):
        shutil.copy2(source/name,dest/name)
    shutil.copytree(source/'features',dest/'features')
    records=load_records(source)
    features=prepare(model,records)
    train=[r for r in features if r['split']=='train']
    validation=[r for r in features if r['split']=='validation']
    mean_feature=torch.cat([r['target'] for r in train]).mean(0).cuda()
    initial=evaluate(model,validation,mean_feature,True)
    write_json(dest/'evaluation-initial.json',initial)
    emit('reconstruction_initial',**{k:v for k,v in initial.items() if k!='rows'})
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best_loss=initial['lrec'];best_epoch=0;best=None;history=[];started=time.monotonic()
    for epoch in range(1,cfg['epochs']+1):
        order=list(range(len(train)));random.shuffle(order)
        losses=[]
        # Dropout is zero; keep the frozen front end in eval for all training.
        for i in range(0,len(order),4):
            opt.zero_grad(set_to_none=True)
            out=model._reconstruct(**pack([train[j] for j in order[i:i+4]],'cuda'))
            loss=out['reconstruction_loss']
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        if epoch%5==0 or epoch==1:
            val=evaluate(model,validation,mean_feature)
            row=dict(epoch=epoch,train_lrec=float(np.mean(losses)),validation_lrec=val['lrec'],
                     baseline=val['mean_baseline'],beats_mean_frames=val['beats_mean_frames'],elapsed_s=round(time.monotonic()-started,1))
            history.append(row)
            with (dest/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            if val['lrec']<best_loss:
                best_loss=val['lrec'];best_epoch=epoch
                best={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            emit('reconstruction_epoch',**row)
    save_checkpoint(dest/'model-last.pt',model,opt,cfg['epochs'],cfg,dest)
    if best is None:raise RuntimeError('No improvement; last checkpoint retained')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in best:p.copy_(best[n])
            else:
                if not torch.equal(p.cpu(),state['state_dict'][n]):raise AssertionError('Frozen parameter changed: '+n)
    # Best checkpoint is inference-only; optimizer state belongs to last epoch.
    best_state=dict(schema='connector-reconstruction-v1',epoch=best_epoch,connector_config=state['connector_config'],
        config=cfg,manifest_sha256=sha256(dest/'manifest.json'),cache_source_sha256=sha256(dest/'cache_source.json'),
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},scope='best validation reconstruction; inference only')
    torch.save(best_state,dest/'model-best.pt')
    validation_result=evaluate(model,validation,mean_feature,True)
    train_result=evaluate(model,train,mean_feature)
    # Already inspected earlier for MEFA; explicitly a secondary descriptive set,
    # never used to select the epoch in this reconstruction experiment.
    secondary=prepare(model,load_records(ROOT/'outputs/logs/mefa_history_validation_20_drives',evaluation_only=True))
    secondary_result=evaluate(model,secondary,mean_feature,True)
    result=dict(best_epoch=best_epoch,frozen_parameters_unchanged=True,initial=initial,
        validation=validation_result,training=train_result,secondary_previously_seen=secondary_result,
        source_checkpoint_sha256=cfg['source_checkpoint_sha256'],best_checkpoint_sha256=sha256(dest/'model-best.pt'),
        note='Existing 2 validation drives choose checkpoint; 20 previously inspected drives are secondary, not an untouched test.',
        source_sha256=sha256(Path(__file__)))
    write_json(dest/'result.json',result)
    emit('reconstruction_complete',best_epoch=best_epoch,validation={k:v for k,v in validation_result.items() if k!='rows'},
         secondary={k:v for k,v in secondary_result.items() if k!='rows'})


if __name__=='__main__':main()
