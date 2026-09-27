#!/usr/bin/env python3
"""Cache audited Sensor Record publications for the navigation Stage 2 loader."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def stage2_trajectory(waypoints, contract):
    import numpy as np
    if contract.get('waypoint_output_axes')!='right_negative_forward':
        raise ValueError('Expected audited Sensor Record [right, -forward] waypoint axes')
    target=np.asarray(waypoints,dtype=np.float32).copy()
    if target.shape!=(5,2) or not np.isfinite(target).all():
        raise ValueError('Invalid Sensor Record future trajectory')
    # Preserve existing VLDrive Stage 2 [left, -forward] coordinates explicitly.
    target[:,0]*=-1
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if cfg.get('schema') == 'native-navigation-token-cache-config-v1':
        from tools.cache_sensor_record_tokens import main as cache_native
        return cache_native()
    if set(cfg)!={'schema','manifest_indices','path_mappings','instructions','checkpoint','checkpoint_sha256','output','device'} or cfg['schema']!='navigation-token-cache-config-v1':
        raise ValueError('Invalid navigation token cache config')
    if not cfg['manifest_indices']: raise ValueError('Configure admitted train/validation publication indices')
    checkpoint=(ROOT/cfg['checkpoint']).resolve()
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha!=cfg['checkpoint_sha256']: raise ValueError('Checkpoint SHA256 mismatch')
    output=(ROOT/cfg['output']).resolve()
    if output.exists(): raise FileExistsError('Choose a new cache output directory')
    from utils.common.runtime import configure_runtime
    configure_runtime()
    import numpy as np
    import torch
    from datasets.morai_sensor_record_dataset import PathResolver, SensorRecordDataset, load_publications
    from models.vision_encoder import create_vision_encoder
    from navigation.sampling import NavigationLabels
    labels=NavigationLabels(ROOT/cfg['instructions'])
    resolver=PathResolver(cfg['path_mappings'])
    publication,manifests=load_publications([ROOT/p for p in cfg['manifest_indices']],resolver)
    model=create_vision_encoder({'name':'memfuser_baseline_e1d3_return_feature'},pretrained_backbone=False)
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=blob.get('state_dict',blob)
    model.load_state_dict(state,strict=True)
    model.to(cfg['device']).eval()
    # No defaulting to an unverified partial checkpoint or a different encoder.
    records=[]
    output.mkdir(parents=True)
    for split in ('train','val'):
        for path,document in manifests[split]:
            source=resolver(document['episode'])
            suffix=source.parent.name+'/'+source.name
            matching=[key for key in labels.routes if key==suffix or key.endswith('/'+suffix)]
            if len(matching)!=1: raise ValueError('No unique navigation episode for '+suffix)
            identity=matching[0]
            dataset=SensorRecordDataset([(path,document)],resolver,velocity_source='gps')
            for i,(_,entry,_,_) in enumerate(dataset.rows):
                stamp=entry['sensor_time_ns'];scene=labels.frames.get((identity,stamp))
                if scene is None: raise ValueError('Published sensor time has no navigation association')
                if not scene['valid']: continue
                inputs,targets=dataset[i]
                with torch.inference_mode():
                    visual=model({k:v.unsqueeze(0).to(cfg['device']) for k,v in inputs.items()})[0].float().cpu().numpy()
                target=stage2_trajectory(targets['waypoints'].numpy(),document['contract'])
                if visual.shape!=(106,256) or target.shape!=(5,2) or not np.isfinite(visual).all() or not np.isfinite(target).all():
                    raise ValueError('Feature/target tensor contract mismatch')
                key=hashlib.sha256(f'{identity}|{stamp}'.encode()).hexdigest()[:24]
                references={}
                for name,value in [('visual_feature',visual.astype(np.float16)),('future_trajectory',target)]:
                    file=output/name/(key+'.npy');file.parent.mkdir(exist_ok=True)
                    with file.open('xb') as stream: np.save(stream,value,allow_pickle=False)
                    references[name]=dict(path=file.relative_to(output).as_posix(),shape=list(value.shape),
                                          sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                references['future_trajectory'].update(source='future_synchronized_ego_positions',horizons_s=[.5,1,1.5,2,2.5])
                records.append(dict(episode_id=identity,sensor_source_ns=stamp,
                                    split='validation' if split=='val' else 'train',**references))
                if len(records)%100==0: print('Cached frames:',len(records),flush=True)
    if not records: raise ValueError('No admitted frames with valid navigation')
    manifest=output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in records),encoding='utf-8')
    meta=dict(schema='navigation-token-cache-v1',checkpoint_sha256=checkpoint_sha,
              trajectory_coordinate_convention='lmdrive_left_negative_forward',
              manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),publication=publication,
              labels_manifest_sha256=hashlib.sha256((labels.root/'manifest.json').read_bytes()).hexdigest(),
              config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    (output/'cache_meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),frames=len(records)),indent=2))


if __name__=='__main__': main()
