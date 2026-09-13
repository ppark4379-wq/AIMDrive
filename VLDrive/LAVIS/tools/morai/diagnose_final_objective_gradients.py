#!/usr/bin/env python3
"""Saved final-model loss-specific gradients. No optimizer or weight update."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lavis.common.morai_runtime import configure_runtime, bootstrap, sha256
configure_runtime()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if ROOT / 'outputs' not in destination.parents or destination.exists():
        raise ValueError('Use a new diagnostic JSON within LAVIS/outputs')
    destination.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (ROOT / '.runtime/training_runs/last_encoder.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import torch
    bootstrap(ROOT.parent)
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    import lavis.models.vldrive.ddia_attention as attention
    from lavis.runners.vldrive_trainer import parameter_hash, source_hashes, seed_all
    from lavis.runners.morai_trainer import autocast
    torch.set_num_threads(2)
    run = ROOT / 'outputs/morai_paper_last_encoder_20260910T055054Z'
    settings = read(run/'resolved_config.json')
    checkpoint = run/'model-last.pt'
    identity = read(checkpoint.with_suffix('.pt.sha256.json'))
    assert sha256(checkpoint) == identity['sha256']
    sources = source_hashes(ROOT)
    seed_all(settings['seed'])
    print('LOADING_FINAL_MODEL_FOR_GRADIENTS', flush=True)
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert asdict(VLDriveTrainingConfig(**state['model_config'])) == asdict(model.config)
    frozen = parameter_hash(model, trainable=False)
    assert frozen == state['frozen_hash'] and state['updates'] == 9540
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    assert expected == set(state['trainable_state'])
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in expected:
                p.copy_(state['trainable_state'][name])
    trainable = parameter_hash(model, trainable=True)
    assert trainable == state['trainable_hash']
    del state
    cache = Path(settings['feature_cache'])
    rows = {r['frame_id']: r for r in map(json.loads, (cache/'manifest.jsonl').read_text().splitlines()) if r['split']=='train'}
    instructions = list(map(json.loads, Path(settings['instructions']).read_text().splitlines()))
    saved = {int(r['sample_id'].rsplit('/',1)[1]): r for r in read(run/'train_predictions.json')}
    previous_rope = attention.apply_rope
    def historical_rope(x, position_ids, rope_theta=10000.0):
        d=x.shape[-1]
        freq=1.0/(rope_theta**(torch.arange(0,d,2,device=x.device).float()/d))
        angle=position_ids.float().unsqueeze(-1)*freq
        angle=torch.cat((angle,angle),-1).unsqueeze(1).to(x.dtype)
        return x*angle.cos()+attention.rotate_half(x)*angle.sin()
    groups={'selector':'connector.ccdp.', 'input_norm':'connector.visual_input_norm.',
            'mefa':'connector.temporal_encoder.', 'qformer':'connector.qformer.',
            'projection':'connector.qformer_to_language.', 'reconstruction':'connector.reconstruction_',
            'llm':'language_model.tinyllama.model.', 'waypoint_head':'language_model.waypoints_predictor.'}
    result=dict(optimizer_updates=0, original_updates=9540, checkpoint_sha256=identity['sha256'],
                diagnostic_script_sha256=sha256(Path(__file__)), source_sha256=sources,
                scope='three actual train clips, historical RoPE, no optimizer', probes=[])
    try:
        attention.apply_rope=historical_rope
        for frame in (585,2865,2970):
            x=[]; y=[]
            for f in range(frame-10,frame+1):
                for key, buffer in [('visual_feature', x),('future_trajectory', y)]:
                    path=cache/rows[f][key]['path']
                    assert sha256(path)==rows[f][key]['sha256']
                    buffer.append(np.load(path, allow_pickle=False))
            features=torch.from_numpy(np.stack(x).astype(np.float32))[None].cuda()
            target=torch.from_numpy(np.stack(y).astype(np.float32))[None].cuda()
            text=next(r['text'] for r in instructions if r['start_frame']<=frame<=r['stop_frame'])
            inputs=dict(visual_feature=features,instruction=[text])
            model.eval()
            with torch.no_grad(),autocast(model):
                eval_output=model.predict(inputs,compute_auxiliary=False)
            assert np.array_equal(eval_output['trajectory'][0,-1].float().cpu().numpy(),np.array(saved[frame]['trajectory']))
            del eval_output
            seed_all(1307+frame)
            model.train()
            model.zero_grad(set_to_none=True)
            with autocast(model):
                prediction=model.predict(inputs,compute_auxiliary=True)
                waypoint=model.loss(prediction,dict(future_trajectory=target))['waypoint_loss']
            losses={'waypoint':waypoint,'weighted_pruning':10*prediction['pruning_loss'],
                    'weighted_reconstruction':prediction['reconstruction_loss']}
            probe=dict(frame=frame, actual_history_frames=10,
                selected_count=prediction['selected_count'].cpu().tolist(),
                saturated_keep_fraction=float(prediction['saturated_keep_fraction']),
                soft_retention=float(prediction['soft_retention'].mean()),
                loss={k:float(v.detach()) for k,v in losses.items()}, gradients={})
            for number,(name,loss) in enumerate(losses.items()):
                model.zero_grad(set_to_none=True)
                loss.backward(retain_graph=number<len(losses)-1)
                norms={key:[] for key in groups}; total=[]
                for param_name,param in model.named_parameters():
                    if param.grad is None:
                        continue
                    assert torch.isfinite(param.grad).all()
                    square=param.grad.detach().float().square().sum()
                    total.append(square)
                    for key,prefix in groups.items():
                        if param_name.startswith(prefix):
                            norms[key].append(square)
                probe['gradients'][name]={key:float(torch.stack(parts).sum().sqrt()) if parts else 0.
                                          for key,parts in norms.items()}
                probe['gradients'][name]['total']=float(torch.stack(total).sum().sqrt())
            model.zero_grad(set_to_none=True)
            del prediction,losses,waypoint,loss
            result['probes'].append(probe)
            print(json.dumps(probe),flush=True)
    finally:
        attention.apply_rope=previous_rope
    assert source_hashes(ROOT)==sources
    assert parameter_hash(model,trainable=True)==trainable
    assert parameter_hash(model,trainable=False)==frozen
    result.update(complete=True,trainable_hash_unchanged=True,frozen_hash_unchanged=True,
                  production_sources_unchanged=True,apply_rope_restored=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('FINAL_OBJECTIVE_GRADIENTS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
