"""Stage 2 windows joined by episode and native sensor time, with LMDrive text sampling."""
import hashlib
import json
from pathlib import Path

from navigation.sampling import NavigationLabels


class NavigationCacheDataset:
    trajectory_coordinate_convention='lmdrive_left_negative_forward'
    def __init__(self, cache_root, instructions, split, sequence_length=11, seed=42,
                 train_variants=tuple(range(8)), validation_variants=tuple(range(8)),
                 frame_sampling_interval=1, include_validation_in_train=False, path_feature_cache=None):
        self.root=Path(cache_root).resolve()
        self.labels=NavigationLabels(instructions)
        self.path_features = None
        if path_feature_cache:
            from instruction.global_path.encoder import PathFeatureCache
            self.path_features = PathFeatureCache(path_feature_cache,
                hashlib.sha256((self.labels.root/'manifest.json').read_bytes()).hexdigest())
        self.split='validation' if split in ('val','validation') else split
        if self.split not in ('train','validation'): raise ValueError('Invalid cache split')
        self.include_validation_in_train=include_validation_in_train
        if include_validation_in_train and self.split != "train":
            raise ValueError("Full-route mode is training-only")
        self.seed=seed;self.epoch=0
        self.variants=train_variants if self.split=='train' else validation_variants
        from navigation.catalog import variant_for
        variant_for(seed,'validate',variants=self.variants)
        if type(sequence_length) is not int or sequence_length<1: raise ValueError('Invalid sequence_length')
        if type(frame_sampling_interval) is not int or frame_sampling_interval<1:
            raise ValueError('Invalid frame_sampling_interval')
        self.sequence_length=sequence_length
        self.frame_sampling_interval=frame_sampling_interval
        meta=json.loads((self.root/'cache_meta.json').read_text())
        if meta.get('schema')!='navigation-token-cache-v1': raise ValueError('Wrong navigation token cache schema')
        if meta.get('trajectory_coordinate_convention')!=self.trajectory_coordinate_convention:
            raise ValueError('Wrong Stage 2 trajectory coordinate convention')
        if meta.get('labels_manifest_sha256') != hashlib.sha256((self.labels.root/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Cache and navigation labels differ')
        path=self.root/'manifest.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['manifest_sha256']:
            raise ValueError('Token cache manifest changed')
        self.checkpoint_sha256=meta['checkpoint_sha256'];self.cache_meta=meta
        all_rows=[json.loads(s) for s in path.read_text().splitlines()]
        seen=set();by_episode={}
        for row in all_rows:
            key=(row['episode_id'],row['sensor_source_ns'])
            if key in seen: raise ValueError('Duplicate cache scene across splits')
            seen.add(key)
            if row['split'] not in ('train','validation'): raise ValueError('Invalid cache row split')
            if row['future_trajectory'].get('source')!='future_synchronized_ego_positions' or row['future_trajectory'].get('horizons_s')!=[.5,1,1.5,2,2.5]:
                raise ValueError('Trajectory must contain actual future ego positions')
            by_episode.setdefault(row['episode_id'],[]).append(row)
        # Reject adjacent split blocks that share the future target horizon.
        for rows in by_episode.values():
            rows.sort(key=lambda r:r['sensor_source_ns'])
            for a,b in zip(rows,rows[1:]):
                if a['split']!=b['split'] and b['sensor_source_ns']-a['sensor_source_ns']<=2_500_000_000:
                    raise ValueError('Train/validation boundary leaks future target horizon')
        self.windows=[]
        span=(sequence_length-1)*frame_sampling_interval+1
        for identity,rows in by_episode.items():
            rows=[r for r in rows if self.include_validation_in_train or r['split']==self.split]
            for end in range(span,len(rows)+1):
                continuous=rows[end-span:end]
                # Validate the entire source span before subsampling. A missing
                # cached frame must not be mistaken for a larger sample interval.
                eligible = self.path_features.eligible if self.path_features is not None else self.labels.eligible
                if eligible(identity,[r['sensor_source_ns'] for r in continuous]):
                    self.windows.append(continuous[::frame_sampling_interval])
        if not self.windows: raise ValueError('No valid navigation cache windows')
        self._verified=set()

    def set_epoch(self, epoch): self.epoch=int(epoch) if self.split=='train' else 0
    def __len__(self): return len(self.windows)

    def _array(self, reference, shape):
        import numpy as np
        import torch
        path=(self.root/reference['path']).resolve()
        if self.root not in path.parents: raise ValueError('Unsafe token cache path')
        if path not in self._verified:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
                raise ValueError('Cached tensor changed')
            self._verified.add(path)
        value=np.load(path,allow_pickle=False)
        if value.shape!=shape or not np.isfinite(value).all(): raise ValueError('Invalid cached tensor')
        return torch.from_numpy(value.astype('float32'))

    def __getitem__(self,index):
        return self._sample(index,self.epoch)

    def evaluation_sample(self,index):
        return self._sample(index,0)

    def _sample(self,index,epoch):
        import torch
        rows=self.windows[index];identity=rows[0]['episode_id'];times=[r['sensor_source_ns'] for r in rows]
        path_inputs = {}
        if self.path_features is None:
            instruction=self.labels.sample(identity,times,self.seed,epoch,self.variants,
                                           sampling_interval=self.frame_sampling_interval)
        else:
            selection, features = self.path_features.sample(identity, times)
            path_inputs['path_features'] = {key: torch.tensor(value, dtype=torch.bool if key == 'path_valid' else torch.float32)
                                            for key, value in features.items()}
            instruction = dict(text=selection['text'], instruction_id=selection['kind'], variant_id=0,
                               instruction_instance_id=identity + '/' + str(times[0]) + '/' + selection['event_id'])
        return {'model_inputs':{**path_inputs,'visual_feature':torch.stack([self._array(r['visual_feature'],(106,256)) for r in rows]),
                                'instruction':instruction['text']},
                'training_targets':{'future_trajectory':torch.stack([self._array(r['future_trajectory'],(5,2)) for r in rows])},
                'metadata':{'run_id':identity,'sample_ids':[identity+'/'+str(t) for t in times],
                            'instruction_instance_id':instruction['instruction_instance_id'],
                            'instruction_id':instruction['instruction_id'],'variant_id':instruction['variant_id'],
                            'input_mode':'cached','purpose':self.split}}

    def describe(self):
        result = dict(schema='navigation-cache-dataset-v1',windows=len(self),split=self.split,
                    sequence_length=self.sequence_length,frame_sampling_interval=self.frame_sampling_interval,
                    include_validation_in_train=self.include_validation_in_train,
                    checkpoint_sha256=self.checkpoint_sha256,cache_meta=self.cache_meta,
                    labels_manifest_sha256=hashlib.sha256((self.labels.root/'manifest.json').read_bytes()).hexdigest(),
                    template_variants=list(self.variants),seed=self.seed)
        if self.path_features is not None:
            result['path_feature_cache_sha256'] = self.path_features.sha256
            result['instruction_policy'] = 'first-frame-planned-route-v1'
        return result
