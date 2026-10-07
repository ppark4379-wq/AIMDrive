"""Recorded IDs -> learned Zone token -> causal waypoint supervision."""
import json
from unittest.mock import patch

import pytest
import torch

from instruction.zone.encoding import recorded_zone_id, UNKNOWN_ZONE
from VLDrive.data.src.loaders.situation_instructions import SituationInstructionIndex
from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
from models.vldrive import VLDriveTrainingConfig
from training.common.runtime.batching import batch_samples
from training.common.checkpoints import save_model_checkpoint, restore_model_checkpoint, warm_start_model_checkpoint
from training.common.logging import gradient_report
from .test_navigation_cache import cache, digest
from .test_situation_instruction import model, write_labels


def test_recorded_zone_zero_and_unknown_are_distinct():
    assert recorded_zone_id(dict(zone=None,zone_instruction=0)) == 0
    assert recorded_zone_id(dict(zone='normal',zone_instruction=0)) == 0
    assert recorded_zone_id(dict(zone='static_obstacle',zone_instruction=1)) == 1
    assert recorded_zone_id({}) == UNKNOWN_ZONE
    for reason in ('missing_ego_xy','invalid_ego_xy'):
        assert recorded_zone_id(dict(zone=None,zone_instruction=0,frame_invalid_reasons=[reason])) == UNKNOWN_ZONE
    # Missing object GT does not invalidate a known map zone.
    assert recorded_zone_id(dict(zone=None,zone_instruction=0,frame_valid=False,
                                 frame_invalid_reasons=['incomplete_object_inventory'])) == 0
    for row in (dict(zone=None,zone_instruction=1),dict(zone='highway',zone_instruction=1),
                dict(zone_instruction=True),dict(zone_instruction=6)):
        with pytest.raises(ValueError,match='Zone'):
            recorded_zone_id(row)


@pytest.mark.parametrize('situation_enabled', [False, True])
def test_dataset_reads_zones_without_requiring_situation_text(cache, tmp_path, situation_enabled):
    folder, bundle, cached_rows, meta = cache
    specs = write_labels(tmp_path/'raw')
    path = tmp_path/'raw/episode/situation_labels/labeled_frames.jsonl'
    labels = [json.loads(line) for line in path.read_text().splitlines()]
    for i in (0,2):
        labels[i].update(zone=None,zone_instruction=0)
        labels[i]['labeling'].update(zone_id=None,instruction_id=None,instruction=None)
    labels[2]['frame_invalid_reasons'] = ['invalid_ego_xy']
    path.write_text(''.join(json.dumps(row)+'\n' for row in labels))
    path.with_name('COMPLETE.json').write_text(json.dumps({'files':{'labeled_frames.jsonl':digest(path)}}))
    for row in cached_rows:
        row['frame_index'] = row['sensor_source_ns']//100_000_000
    manifest=folder/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in cached_rows))
    meta['manifest_sha256']=digest(manifest)
    (folder/'cache_meta.json').write_text(json.dumps(meta))
    def index(*args, **kwargs):
        value = SituationInstructionIndex(*args,**kwargs)
        value.frames = {('train_episode',stamp):r for (_,stamp),r in value.frames.items()}
        value.frame_times = {('train_episode',frame):stamp for (_,frame),stamp in value.frame_times.items()}
        value.zones = {('train_episode',stamp):z for (_,stamp),z in value.zones.items()}
        return value
    with patch('VLDrive.data.src.loaders.situation_instructions.SituationInstructionIndex',side_effect=index) as build:
        dataset=NavigationCacheDataset(folder,bundle,'train',sequence_length=3,frame_sampling_interval=2,
            situation_datasets=specs if situation_enabled else None,zone_datasets=specs)
        assert build.call_count == 1  # Shared Situation+Zone labels are read only once.
        unlabelled=NavigationCacheDataset(folder,bundle,'validation',sequence_length=3,
            frame_sampling_interval=2,zone_datasets=specs)
    assert len(dataset)==len(unlabelled)==8
    sample=dataset[0]
    assert sample['model_inputs']['zone_ids'].tolist()==[0,-1,1]
    assert ('situation_instructions' in sample['model_inputs']) == situation_enabled
    if situation_enabled:
        assert sample['model_inputs']['situation_instructions'][:2]==[None,None]
    batched=batch_samples([sample,sample], 'cpu')
    assert batched['model_inputs']['zone_ids'].tolist()==[[0,-1,1]]*2
    assert unlabelled[0]['model_inputs']['zone_ids'].tolist()==[-1]*3
    assert dataset.describe()['zone_instruction']['schema']=='zone_v2_no_traffic_light'


@pytest.mark.parametrize('path_enabled', [False, True])
@pytest.mark.parametrize('situation_enabled', [False, True])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_eleven_frames_zone_gradients_and_future_isolation(monkeypatch,path_enabled,situation_enabled,checkpointed):
    torch.manual_seed(17)
    net=model(monkeypatch,path=path_enabled,situation=situation_enabled,zone=True)
    if checkpointed:
        net.tinyllama.model.gradient_checkpointing_enable()
    visual=torch.randn(2,11,4,32)
    zones=torch.tensor([[0,1,2,3,4,5,0,1,2,3,4],[-1]*11])
    args=dict(zone_ids=zones,predict_per_frame=True)
    if path_enabled:
        args['path_features']=dict(ego_relation=torch.randn(2,11,3),path_context=torch.randn(2,11,5),
                                   path_valid=torch.ones(2,11,dtype=torch.bool))
    if situation_enabled:
        args['situation_instructions']=[['Yield to pedestrians.',None]*5+[None],[None]*11]
    net.train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
    output=net(['Follow.']*2,visual,**args)
    target=torch.randn_like(output.predicted_waypoints)
    torch.nn.functional.l1_loss(output.predicted_waypoints,target).backward()
    table=net.zone_embedding.zone_embedding.weight
    assert table.grad.abs().sum(1).gt(0).all()  # Includes general-driving ID 0.
    assert net.zone_embedding.projection.weight.grad.abs().sum()>0
    previous=table.detach().clone()
    optimizer.step()
    assert not torch.equal(previous,table)
    net.eval()
    with torch.no_grad():
        baseline=net(['Follow.']*2,visual,**args)
        changed=zones.clone(); changed[:,-1]=(changed[:,-1]+1)%6
        future=net(['Follow.']*2,visual,**dict(args,zone_ids=changed))
    torch.testing.assert_close(baseline.predicted_waypoints[:,:-1],future.predicted_waypoints[:,:-1],atol=1e-6,rtol=0)
    assert not torch.allclose(baseline.predicted_waypoints[:,-1],future.predicted_waypoints[:,-1])
    # A known Zone contributes exactly one active token per frame.
    counts=baseline.sequence.attention_mask.sum(1)
    if not situation_enabled:
        assert int(counts[0]-counts[1])==11


def test_unknown_zone_is_masked_and_not_normal(monkeypatch):
    torch.manual_seed(29)
    net=model(monkeypatch,path=True,situation=False,zone=True).eval()
    baseline=model(monkeypatch,path=True,situation=False).eval()
    baseline.load_state_dict({k:v for k,v in net.state_dict().items() if not k.startswith('zone_embedding.')})
    visual=torch.randn(1,3,4,32)
    args=dict(path_features=dict(ego_relation=torch.randn(1,3,3),path_context=torch.randn(1,3,5),
                                path_valid=torch.ones(1,3,dtype=torch.bool)),predict_per_frame=True)
    unknown=net(['Follow.'],visual,zone_ids=torch.full((1,3),-1),**args)
    original=baseline(['Follow.'],visual,**args)
    torch.testing.assert_close(unknown.predicted_waypoints,original.predicted_waypoints,atol=1e-6,rtol=0)
    normal=net(['Follow.'],visual,zone_ids=torch.zeros(1,3,dtype=torch.long),**args)
    assert int(normal.sequence.attention_mask.sum()-unknown.sequence.attention_mask.sum())==3
    assert not torch.allclose(normal.predicted_waypoints,unknown.predicted_waypoints)
    unknown.predicted_waypoints.sum().backward()
    assert net.zone_embedding.zone_embedding.weight.grad.count_nonzero()==0


def test_zone_input_contract(monkeypatch):
    net=model(monkeypatch,path=False,situation=False,zone=True)
    visual=torch.randn(1,3,4,32)
    for value in (None,torch.zeros(1,3),torch.zeros(1,2,dtype=torch.long),
                  torch.full((1,3),6),torch.full((1,3),-2)):
        with pytest.raises(ValueError,match='Zone'):
            net(['Follow.'],visual,zone_ids=value)
    disabled=model(monkeypatch,path=False,situation=False)
    with pytest.raises(ValueError,match='disabled'):
        disabled(['Follow.'],visual,zone_ids=torch.zeros(1,3,dtype=torch.long))


def test_zone_warm_start_and_checkpoint_roundtrip(monkeypatch,tmp_path):
    def wrapper(enabled):
        torch.manual_seed(41)
        value=torch.nn.Module()
        value.config=VLDriveTrainingConfig(path_navigation=True,path_fusion='mlp',zone_instruction=enabled)
        value.checkpoint_schema='vldrive-training-v1'
        value.language_model=model(monkeypatch,path=True,situation=False,zone=enabled)
        value.language_model.end_predictor.requires_grad_(False)
        value.language_model.tinyllama.lm_head.requires_grad_(False)
        return value
    old,new=wrapper(False),wrapper(True)
    source=tmp_path/'old.pt'
    save_model_checkpoint(source,old,updates=1,provenance={},settings={},metrics={})
    report=warm_start_model_checkpoint(source,new,provenance={},
        new_prefixes=['language_model.zone_embedding.'],transfer_reason='Add learned Zone conditioning')
    assert len(report['initialized_new_keys'])==3
    assert all(k.startswith('language_model.zone_embedding.') for k in report['initialized_new_keys'])
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    saved=tmp_path/'zone.pt'
    save_model_checkpoint(saved,new,updates=2,provenance={},settings={},metrics={})
    with torch.no_grad():
        new.language_model.zone_embedding.zone_embedding.weight.add_(1)
    restored=restore_model_checkpoint(saved,new,provenance={})
    assert torch.equal(new.language_model.zone_embedding.zone_embedding.weight,
                       restored['trainable_state']['language_model.zone_embedding.zone_embedding.weight'])


def test_gradient_check_distinguishes_unknown_zone_from_broken_connection():
    net = torch.nn.Module()
    net.zone = torch.nn.Linear(2,2)
    net.gradient_groups = lambda: {'zone_embedding':'zone.'}
    known = dict(zone_ids=torch.tensor([[0,1]]))
    assert not gradient_report(net,inputs=known)['passed']
    unknown = gradient_report(net,inputs=dict(zone_ids=torch.tensor([[-1,-1]])))
    assert unknown['passed']
    assert unknown['groups']['zone_embedding']['skipped']=='no_known_zone_in_sample'
    net.zone(torch.ones(1,2)).sum().backward()
    assert gradient_report(net,inputs=known)['passed']
