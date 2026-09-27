#!/usr/bin/env python3
"""Read-only model audit: isolate loss gradients on the actual cached inputs.

No optimizer is created, no checkpoint is modified, and no inference/training
policy is changed. Reports distinguish connectivity from useful task gradients.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from utils.common.runtime import configure_runtime, sha256
configure_runtime()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
    from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
    from navigation.token_cache import CachedNavigationTokenizer
    from utils.training.training_helper import batch, autocast, seed_all, source_hashes
    cfg=json.loads(args.config.read_text())
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    report_path=output/'model_audit.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    torch.set_num_threads(cfg.get('cpu_threads',4))
    seed_all(cfg['seed'])
    language=cfg['navigation_sampling']
    dataset=NavigationCacheDataset(ROOT/cfg['feature_cache'],ROOT/cfg['instructions'],'train',
        cfg['sequence_length'],cfg['seed'],train_variants=language['train_variants'],
        validation_variants=language['validation_variants'],
        frame_sampling_interval=cfg.get('frame_sampling_interval',1))
    model_cfg=dict(cfg['model'])
    for key in ('visual_checkpoint','language_checkpoint','qformer_checkpoint'):
        model_cfg[key]=str(ROOT/model_cfg[key])
    print('Loading exact configured architecture and pretrained assets...',flush=True)
    model=VLDriveTrainingModel(VLDriveTrainingConfig(**model_cfg),device=cfg['device'])
    model.language_model.tokenizer=CachedNavigationTokenizer(model.language_model.tokenizer,
        ROOT/language['token_cache'],sha256(ROOT/cfg['instructions']/'manifest.json'),
        sha256(ROOT/cfg['feature_cache']/'manifest.jsonl'))
    sample=dataset[0]
    item=batch(sample,cfg['device'])
    report=dict(config_sha256=sha256(args.config),source_hashes=source_hashes(ROOT),
        optimizer_updates=0,scope='fresh configured model, actual basic_drive cache; diagnostic only',
        sample=sample['metadata'],components={},model_config=model_cfg)
    parameters=dict(model.named_parameters())
    selector=[(n,p) for n,p in parameters.items() if n.startswith('connector.ccdp.')]
    def grad_norm(named):
        gradients=[p.grad.detach().float() for _,p in named if p.grad is not None]
        return math.sqrt(sum(float(g.square().sum()) for g in gradients)) if gradients else None
    def norm(value):
        return float(value.detach().float().norm()) if value is not None else None
    captured={}
    def capture(module, inputs, result):
        captured.clear()
        captured.update(result)
        if result['selection_mask'].requires_grad:
            result['selection_mask'].retain_grad()
    handle=model.connector.register_forward_hook(capture)
    model.train()
    for name in ('waypoint_loss','pruning_weighted','reconstruction_loss','loss'):
        model.zero_grad(set_to_none=True)
        seed_all(902)
        with autocast(model):
            losses=model(item)
        value=10*losses['pruning_loss'] if name=='pruning_weighted' else losses[name]
        value.backward()
        record=dict(value=float(value.detach()),selector_gradient_l2=grad_norm(selector),
                    mask_gradient_l2=norm(captured['selection_mask'].grad),
                    mask_view_gradient_present=captured['selection_mask'].grad is not None,
                    mask_gradient_interpretation='Reference only: the returned view is not an ancestor of every loss. Judge connectivity using selector parameter gradients.',
                    selector_parameters_with_gradient=[n for n,p in selector if p.grad is not None],
                    train_mean_selected=float(captured['selected_count_per_frame'].float().mean()),
                    all_gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in parameters.values() if p.grad is not None))
        if name=='loss':
            unused={n:p.numel() for n,p in parameters.items() if p.requires_grad and p.grad is None}
            report['trainable_parameters_without_gradient']=unused
            report['unused_trainable_parameter_count']=sum(unused.values())
            report['trainable_parameter_count']=sum(p.numel() for p in parameters.values() if p.requires_grad)
            record['loss_values']={k:float(v.detach()) for k,v in losses.items() if v.ndim==0}
            raw=item['model_inputs']['visual_feature']
            normalized=captured['normalized_features']
            target=captured['reconstruction_target'].reshape_as(raw)
            report['reconstruction_target']=dict(actual=model.config.reconstruction_feature_space,
                target_requires_grad=target.requires_grad,
                target_vs_raw_max_abs=float((target-raw).detach().abs().max()),
                raw_rms=float(raw.float().square().mean().sqrt()),
                normalized_rms=float(normalized.detach().float().square().mean().sqrt()),
                raw_vs_normalized_rms=float((raw-normalized.detach()).float().square().mean().sqrt()))
        report['components'][name]=record
        print(name,json.dumps(record),flush=True)
        del losses,value
        captured.clear()
    model.zero_grad(set_to_none=True)
    handle.remove()
    layer=model.connector.qformer.bert.encoder.layer[0]
    report['qformer_query_initialization']=dict(
        query_ffn_equals_loaded_text_ffn=bool(torch.equal(layer.intermediate_query.dense.weight,layer.intermediate.dense.weight)),
        query_text_ffn_weight_difference_l2=norm(layer.intermediate_query.dense.weight-layer.intermediate.dense.weight),
        interpretation='query-only FFN is random; normal text FFN is loaded but unused for query-only inputs')
    model.eval()
    retention=[]
    # Cover several train windows; this evaluates initialization, not convergence.
    with torch.inference_mode(),autocast(model):
        for index in sorted({0,len(dataset)//4,len(dataset)//2,3*len(dataset)//4,len(dataset)-1}):
            current=batch(dataset.evaluation_sample(index),cfg['device'])
            result=model.connector(current['model_inputs']['visual_feature'],compute_reconstruction=False)
            retention.extend(result['selected_count_per_frame'].flatten().tolist())
    report['untrained_eval_retention']=dict(frames=len(retention),minimum=min(retention),maximum=max(retention),
        mean=sum(retention)/len(retention),one_token_fraction=sum(n==1 for n in retention)/len(retention),
        full_token_fraction=sum(n==106 for n in retention)/len(retention),target_ratio=.3,
        scope='initialization diagnostic only; not a trained collapse measurement')
    # Explicitly record clip spacing independently of window stride.
    times=[r['sensor_source_ns'] for r in dataset.windows[0]]
    report['sampling']=dict(sequence_length=len(times),source_interval_ms=[(b-a)/1e6 for a,b in zip(times,times[1:])],
        frame_sampling_interval=dataset.frame_sampling_interval,
        memory_span_s=(times[-1]-times[0])/1e9,train_window_stride=cfg['train_window_stride'])
    report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3 if cfg['device']=='cuda' else None
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved',report_path,flush=True)


if __name__=='__main__':
    main()
