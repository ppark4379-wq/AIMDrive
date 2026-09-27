"""Actual tensor smoke tests for the Stage 2 navigation join and split checks."""
import hashlib
import json

import numpy as np
import pytest

from navigation.engine import DEFAULT_POLICY
from navigation.build import build
from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cache(tmp_path):
    sources=[]
    for name in ('train_episode','validation_episode'):
        path=tmp_path/(name+'.json')
        path.write_text(json.dumps(dict(schema='navigation-episode-v1',episode_id=name,
            route=dict(points=[[i,0] for i in range(100)],events=[],closed=False),
            frames=[dict(index=i,sensor_source_ns=i*100_000_000,x=i,y=0,yaw_rad=0) for i in range(12)])))
        sources.append(dict(root=str(path),format='portable'))
    config=tmp_path/'build.json';labels=tmp_path/'labels'
    config.write_text(json.dumps(dict(schema='lmdrive-navigation-build-v1',datasets=sources,
                                      output=str(labels),policy=DEFAULT_POLICY)))
    build(config)
    folder=tmp_path/'cache';folder.mkdir()
    rows=[]
    for split in ('train','validation'):
        for i in range(12):
            refs={}
            for key,shape in [('visual_feature',(106,256)),('future_trajectory',(5,2))]:
                path=folder/f'{split}_{i}_{key}.npy'
                np.save(path,np.ones(shape,dtype='float32')*i)
                refs[key]=dict(path=path.name,sha256=digest(path),shape=list(shape))
            refs['future_trajectory'].update(source='future_synchronized_ego_positions',horizons_s=[.5,1,1.5,2,2.5])
            rows.append(dict(episode_id=split+'_episode',split=split,sensor_source_ns=i*100_000_000,**refs))
    manifest=folder/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    meta=dict(schema='navigation-token-cache-v1',checkpoint_sha256='0'*64,
              trajectory_coordinate_convention='lmdrive_left_negative_forward',
              manifest_sha256=digest(manifest),labels_manifest_sha256=digest(labels/'manifest.json'))
    (folder/'cache_meta.json').write_text(json.dumps(meta))
    return folder,labels,rows,meta


def test_real_stage2_tensor_contract_and_language_augmentation(cache):
    folder,labels,_,_=cache
    dataset=NavigationCacheDataset(folder,labels,'train',sequence_length=3)
    item=dataset[0]
    assert tuple(item['model_inputs']['visual_feature'].shape)==(3,106,256)
    assert tuple(item['training_targets']['future_trajectory'].shape)==(3,5,2)
    assert item['model_inputs']['instruction']
    assert len(item['metadata']['sample_ids'])==3
    texts=set()
    fixed=dataset.evaluation_sample(0)['model_inputs']['instruction']
    for epoch in range(25):
        dataset.set_epoch(epoch);texts.add(dataset[0]['model_inputs']['instruction'])
    assert len(texts)>8
    assert dataset.evaluation_sample(0)['model_inputs']['instruction']==fixed
    validation=NavigationCacheDataset(folder,labels,'val',sequence_length=3,validation_variants=[6,7])
    before=validation[0]['model_inputs']['instruction'];validation.set_epoch(5)
    assert validation[0]['model_inputs']['instruction']==before
    assert validation[0]['metadata']['variant_id'] in (6,7)


def test_exhaustive_text_cache_covers_500_epochs_and_full_route(cache):
    folder,labels,_,_=cache
    dataset=NavigationCacheDataset(folder,labels,'train',sequence_length=3,
                                   include_validation_in_train=True)
    identities=set()
    for window in dataset.windows:
        identity=window[0]['episode_id'];identities.add(identity)
        stamps=[r['sensor_source_ns'] for r in window]
        possible=dataset.labels.possible_texts(identity,stamps,dataset.variants)
        for epoch in range(500):
            assert dataset.labels.sample(identity,stamps,dataset.seed,epoch,dataset.variants)['text'] in possible
    assert identities=={'train_episode','validation_episode'}


def test_split_overlap_and_target_horizon_leak_are_rejected(cache):
    folder,labels,rows,meta=cache
    rows=[r for r in rows if r['split']=='train']
    for row in rows[6:]: row['split']='validation'
    manifest=folder/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    meta['manifest_sha256']=digest(manifest);(folder/'cache_meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='leaks future'):
        NavigationCacheDataset(folder,labels,'train',sequence_length=3)


def test_changed_feature_is_detected(cache):
    folder,labels,rows,_=cache
    path=folder/rows[0]['visual_feature']['path'];path.write_bytes(b'changed')
    dataset=NavigationCacheDataset(folder,labels,'train',sequence_length=3)
    with pytest.raises(ValueError,match='Cached tensor changed'): dataset[0]


def test_sensor_record_right_axis_is_converted_to_stage2_left_axis():
    from tools.cache.cache_navigation_tokens import stage2_trajectory
    source=np.asarray([[-2,-5]]*5,dtype='float32')
    result=stage2_trajectory(source,{'waypoint_output_axes':'right_negative_forward'})
    assert np.array_equal(result,np.asarray([[2,-5]]*5))
    assert source[0,0]==-2
    with pytest.raises(ValueError,match='waypoint axes'):
        stage2_trajectory(source,{'waypoint_output_axes':'unknown'})


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_interval_two_preserves_scene_feature_target_and_instruction_join(cache, split):
    folder,labels,_,_=cache
    dataset=NavigationCacheDataset(folder,labels,split,sequence_length=3,frame_sampling_interval=2)
    assert len(dataset)==8  # Twelve source frames, five-frame span per window.
    item=dataset[0]
    times=[0,200_000_000,400_000_000]
    assert item['metadata']['sample_ids']==[split+'_episode/'+str(t) for t in times]
    assert item['model_inputs']['visual_feature'][:,0,0].tolist()==[0,2,4]
    assert item['training_targets']['future_trajectory'][:,0,0].tolist()==[0,2,4]
    eligible=dataset.labels.eligible(split+'_episode',times,sampling_interval=2)
    assert item['metadata']['instruction_instance_id'] in eligible
    assert dataset.describe()['frame_sampling_interval']==2
    dataset.set_epoch(2)
    assert dataset[0]['model_inputs']['instruction']


@pytest.mark.parametrize('interval', [0,-1,1.5,True])
def test_bad_frame_sampling_interval_rejected(cache, interval):
    folder,labels,_,_=cache
    with pytest.raises(ValueError,match='frame_sampling_interval'):
        NavigationCacheDataset(folder,labels,'train',sequence_length=3,frame_sampling_interval=interval)


def test_stride_cannot_hide_missing_cached_source_frame(cache):
    folder,labels,rows,meta=cache
    rows=[r for r in rows if not (r['split']=='train' and r['sensor_source_ns']==100_000_000)]
    manifest=folder/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    meta['manifest_sha256']=digest(manifest)
    (folder/'cache_meta.json').write_text(json.dumps(meta))
    dataset=NavigationCacheDataset(folder,labels,'train',sequence_length=3,frame_sampling_interval=2)
    assert all(window[0]['sensor_source_ns']>=200_000_000 for window in dataset.windows)


@pytest.mark.parametrize('problem', ['invalid','instruction_gap','missing','time_gap'])
def test_stride_checks_unsampled_source_scenes(cache, problem):
    folder,labels,_,_=cache
    dataset=NavigationCacheDataset(folder,labels,'train',sequence_length=3,frame_sampling_interval=2)
    samples=dataset.labels
    times=[0,200_000_000,400_000_000]
    assert samples.eligible('train_episode',times,sampling_interval=2)
    skipped=samples.frame_indices[('train_episode',1)]
    if problem=='invalid': skipped['valid']=False
    elif problem=='instruction_gap': skipped['candidate_ids']=[]
    elif problem=='missing': del samples.frame_indices[('train_episode',1)]
    else: skipped['frame']['sensor_source_ns']=10_000_000_000
    assert not samples.eligible('train_episode',times,sampling_interval=2)
    with pytest.raises(ValueError,match='causally valid'):
        samples.sample('train_episode',times,sampling_interval=2)


def test_full_route_includes_validation_as_training(cache):
    folder, labels, _, _ = cache
    original = NavigationCacheDataset(folder, labels, 'train', sequence_length=3)
    full = NavigationCacheDataset(folder, labels, 'train', sequence_length=3,
                                  include_validation_in_train=True)
    assert len(full) == 2 * len(original)
    assert {r['split'] for w in full.windows for r in w} == {'train','validation'}
    assert full[-1]['metadata']['purpose'] == 'train'
    with pytest.raises(ValueError, match='training-only'):
        NavigationCacheDataset(folder, labels, 'val', include_validation_in_train=True)
