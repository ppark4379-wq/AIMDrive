"""Audit reconstruction quality and the actual temporal/Q-Former tensor path."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.connector_only.run import (
    ROOT, torch, np, sha256, load_records, QFormerMEFAConfig, QFormerMEFAConnector, write_json, emit,
)


def main():
    folder = ROOT / 'outputs/logs/mefa_history_validation_20_drives'
    result = json.loads((folder/'mefa_result.json').read_text())
    checkpoint = Path(result['plan']['checkpoint'])
    assert sha256(checkpoint) == result['plan']['checkpoint_sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = QFormerMEFAConnector(QFormerMEFAConfig(**state['connector_config']))
    model.load_state_dict(state['state_dict'], strict=True)
    del state
    model.cuda().eval().requires_grad_(False)
    torch.set_num_threads(4)
    records = load_records(folder, evaluation_only=True)
    training = [r for r in load_records(checkpoint.parent) if r['split']=='train']
    assert not {r['episode'] for r in records} & {r['episode'] for r in training}
    mean_feature = torch.stack([r['tensor'][0,-1] for r in training]).mean(0).cuda()
    rows, failures = [], []
    captured = {}

    def require(condition, name):
        if not bool(condition):
            failures.append(name)

    def capture_bert(module, args, kwargs):
        captured['query'] = kwargs['query_embeds'].detach().clone()
        captured['memory'] = kwargs['encoder_hidden_states'].detach().clone()
        captured['query_mask'] = kwargs['attention_mask'].detach().clone()

    handles = [model.qformer.bert.register_forward_pre_hook(capture_bert, with_kwargs=True)]
    for i, layer in enumerate(model.qformer.bert.encoder.layer):
        if not layer.has_cross_attention:
            continue
        for name in ('key','value'):
            def capture(module, args, index=i, field=name):
                captured[(index,field)] = args[0].detach().clone()
            handles.append(getattr(layer.crossattention.self,name).register_forward_pre_hook(capture))

    with torch.inference_mode():
        # Test window boundary, age limit and causality on an explicit 13-frame sequence.
        synthetic = torch.arange(13,device='cuda',dtype=torch.float32).view(1,13,1,1).expand(1,13,106,256).clone()
        temporal = model.temporal_encoder
        h = temporal.causal_history_average(synthetic)
        expected = torch.zeros_like(h)
        for t in range(1,13):
            expected[:,t] = synthetic[:,max(0,t-10):t].mean(1)
        require(torch.equal(h,expected), 'history_recent_10_excludes_current')
        changed = synthetic.clone(); changed[:,0] += 100
        require(torch.equal(temporal(changed)['enhanced_features'][:,-1],temporal(synthetic)['enhanced_features'][:,-1]), 'expired_frame_not_used')
        changed = synthetic.clone(); changed[:,-1] += 100
        require(torch.equal(temporal(changed)['enhanced_features'][:,:-1],temporal(synthetic)['enhanced_features'][:,:-1]), 'future_not_used')
        require(torch.count_nonzero(temporal(synthetic[:,-1:])['history_average'])==0,'standalone_call_has_no_persistent_bank')

        for index, row in enumerate(records):
            x = row['tensor'].cuda()
            normal = model(x, compute_reconstruction=True)
            memory = normal['enhanced_memory'].flatten(0,1)
            valid = normal['selected_valid_mask'].flatten(0,1)
            ids = normal['selected_indices'].flatten(0,1)
            norm = normal['normalized_features'].flatten(0,1)
            gathered = norm.gather(1,ids.clamp_min(0).unsqueeze(-1).expand(-1,-1,256))
            expected_query = model.retained_query_projection(gathered)
            require(torch.allclose(captured['query'][valid],expected_query[valid],atol=1e-6,rtol=1e-5), 'query_from_ccdp_selected_current_features')
            require(torch.equal(captured['query_mask'].bool(),valid), 'query_padding_mask')
            require(torch.equal(captured['memory'],memory), 'bert_memory_equals_enhanced_full_features')
            for key,value in captured.items():
                if isinstance(key,tuple):
                    require(torch.equal(value,memory), f'crossattention_{key}_receives_enhanced_memory')
            reference = torch.zeros_like(normal['history_average'])
            for t in range(1,x.shape[1]):
                reference[:,t] = normal['normalized_features'][:,max(0,t-10):t].mean(1)
            require(torch.equal(reference,normal['history_average']), 'real_history_average')
            require(torch.equal(memory,(normal['normalized_features']+normal['temporal_encoding']).flatten(0,1)), 'memory_residual_addition')
            original_query = captured['query'].clone()
            zero_history = model(x,hard_mask=normal['selection_mask'],history_mode='zero',compute_reconstruction=False)
            require(torch.equal(original_query,captured['query']), 'history_ablation_keeps_queries_fixed')
            current_valid = normal['selected_valid_mask'][0,-1]
            hidden_delta = float((normal['qformer_hidden'][0,-1,current_valid]-zero_history['qformer_hidden'][0,-1,current_valid]).abs().mean())
            output_delta = float((normal['language_tokens'][0,-1,current_valid]-zero_history['language_tokens'][0,-1,current_valid]).abs().mean())
            require(hidden_delta>0 and output_delta>0, 'history_reaches_selected_output_tokens')
            require(torch.count_nonzero(zero_history['history_average'])==0, 'zero_history_is_zero')
            mask = normal['selection_mask'][:,-1]
            target = x[:,-1]
            tokens = normal['language_tokens'][:,-1]
            indices = normal['selected_indices'][:,-1]
            validity = normal['selected_valid_mask'][:,-1]
            def reconstruct(values, features=target):
                return model._reconstruct(features,values,indices,validity,mask)
            direct = reconstruct(tokens)
            require(torch.allclose(direct['reconstructed_features'][0],normal['reconstructed_features'][-1],atol=1e-5,rtol=1e-5), 'reconstruction_path_reproduces_forward')
            drop = mask[0]==0
            if not drop.any():
                continue
            zero_input = reconstruct(torch.zeros_like(tokens))
            # Opposite category donor, sampled deterministically. Repack donor
            # using recipient indices so query count, positions and target stay fixed.
            donor = records[(index+len(records)//2)%len(records)]
            require(donor['category']!=row['category'],'donor_different_category')
            donor_out = model(donor['tensor'].cuda(),hard_mask=normal['selection_mask'],compute_reconstruction=False)
            wrong_input = reconstruct(donor_out['language_tokens'][:,-1])
            wrong_target = target.clone(); wrong_target[:,drop] += 10
            leak_probe = reconstruct(tokens,wrong_target)
            require(torch.equal(leak_probe['reconstructed_features'],direct['reconstructed_features']), 'dropped_target_not_leaked_to_reconstruction')
            rec = float(direct['reconstruction_loss_per_frame'][0])
            zero_rec = float(zero_input['reconstruction_loss_per_frame'][0])
            wrong_rec = float(wrong_input['reconstruction_loss_per_frame'][0])
            baseline = float((target[0,drop]-mean_feature[drop]).square().mean())
            rows.append(dict(episode=row['episode'],category=row['category'],time_ns=row['times_ns'][-1],
                normal_lrec=rec, zero_connector_lrec=zero_rec, wrong_scene_lrec=wrong_rec,
                training_position_mean_lrec=baseline, qformer_hidden_abs_change=hidden_delta,
                selected_output_abs_change=output_delta,
                kv_abs_change=float((normal['enhanced_memory'][:,-1]-zero_history['enhanced_memory'][:,-1]).abs().mean())))
            if (index+1)%20==0:
                emit('path_audit',frames=index+1,total=len(records),failed_checks=len(failures))
    for h in handles:
        h.remove()
    averages = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('episode','category','time_ns')}
    comparisons = {}
    for field in ('zero_connector_lrec','wrong_scene_lrec','training_position_mean_lrec'):
        differences = [r[field]-r['normal_lrec'] for r in rows]
        comparisons[field] = dict(delta_vs_normal=float(np.mean(differences)),normal_better_frames=sum(v>0 for v in differences),frames=len(rows))
    payload = dict(checkpoint_sha256=sha256(checkpoint), checkpoint_unchanged=sha256(checkpoint)==result['plan']['checkpoint_sha256'],
        failed_checks=failures, tensor_path_checks_passed=not failures,frames=rows,means=averages,comparisons=comparisons,
        memory_implementation='Stateless rolling history average over supplied [B,T,N,C]; no persistent memory bank between calls.',
        qkv_semantics='Selected current features are projected to query embeddings. Each cross-attention K/V projection receives full current features enhanced by the causal history MLP. Internal attention queries also pass through self-attention/normalization.',
        scope='Offline connector tensor path; not a live sensor streaming memory-buffer audit.',
        mean_baseline='Per-position feature mean from 32 training-window current frames only.',
        sources={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),ROOT/'models/connector/qformer_mefa.py',ROOT/'models/connector/qformer.py')})
    write_json(folder/'reconstruction_path_audit.json',payload)
    emit('audit_complete',means=averages,comparisons=comparisons,failed_checks=failures)


if __name__=='__main__':
    main()
