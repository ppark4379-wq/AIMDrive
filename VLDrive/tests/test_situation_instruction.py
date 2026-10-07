"""GT/prediction text parity, exact frame joins and causal LLM conditioning."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from instruction.situation.text import ground_truth_text, load_catalog, prompt_text
from VLDrive.data.src.loaders.situation_instructions import SituationInstructionIndex
from VLDrive.training.situation.predict import decode_prediction
from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
from data.src.processing.build_path_features import build_path_features
from models.llm import language_model
from models.llm.ddia_attention import install_ddia_attention
from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
from training.common.runtime.batching import batch_samples
from .test_navigation_cache import cache, digest


def label(identifier='static_normal', variables=None):
    catalog = load_catalog()
    variables = variables or {}
    return dict(zone_id='static_obstacle', instruction_id=identifier,
                instruction=catalog['static_obstacle'][identifier]['instruction'].format(**variables),
                instruction_variables=variables, instruction_consistency='consistent')


def test_gt_and_prediction_share_text_and_tokenizer_input():
    gt = label('follow_avoid', dict(position='to the left of', object='an obstacle', direction='right'))
    vocab = dict(instruction=['static_obstacle:follow_avoid'], position=['to the left of'],
                 object=['an obstacle'], direction=['right'])
    prediction = decode_prediction({key: torch.zeros(1, 1) for key in vocab}, vocab, load_catalog())
    assert prediction['text'] == ground_truth_text(gt, load_catalog())
    assert prompt_text(prediction['text']) == prompt_text(gt['instruction'])
    del vocab['direction']
    unavailable = decode_prediction({key: torch.zeros(1, 1) for key in vocab}, vocab, load_catalog())
    assert unavailable['text'] is None  # No GT fallback for an unpredicted slot.
    assert prompt_text(unavailable['text']) is None
    assert prompt_text('  ') is None
    gt['instruction'] = 'Turn left.'
    with pytest.raises(ValueError, match='differs'):
        ground_truth_text(gt, load_catalog())


def write_labels(root, missing=None, inconsistent=False):
    rows = []
    for i in range(12):
        gt = label('normal' if i % 2 else 'static_normal')
        gt['index'] = i
        if i % 3 == 1:
            gt = dict(label('follow_avoid_continue', dict(direction='left')), index=i)
        rows.append(dict(frame_id=i, sensor_time_ns=i*100_000_000, zone='static_obstacle',
                         zone_instruction=1, frame_valid=i != missing, labeling=gt))
    if inconsistent:
        rows[0]['labeling']['instruction'] = 'Incorrect sentence.'
    folder = root / 'episode/situation_labels'
    folder.mkdir(parents=True)
    path = folder / 'labeled_frames.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    (folder/'COMPLETE.json').write_text(json.dumps({'files':{'labeled_frames.jsonl':digest(path)}}))
    return [dict(root=str(root), namespace='static_obstacle')]


def test_index_keeps_missing_gt_but_rejects_conflicting_frame_identity(tmp_path):
    specs = write_labels(tmp_path/'raw', missing=3)
    index = SituationInstructionIndex(specs)
    row = dict(frame_index=0, sensor_source_ns=0)
    assert index.sample('static_obstacle/episode', [row]) == ['Continue driving.']
    assert index.sample('another/episode', [row]) == [None]
    assert index.sample('static_obstacle/episode', [dict(frame_index=99, sensor_source_ns=9_900_000_000)]) == [None]
    assert index.sample('static_obstacle/episode', [dict(frame_index=3, sensor_source_ns=300_000_000)]) == [None]
    with pytest.raises(ValueError, match='timestamps'):
        index.sample('static_obstacle/episode', [dict(row, sensor_source_ns=1)])
    with pytest.raises(ValueError, match='indices'):
        index.sample('static_obstacle/episode', [dict(row, frame_index=1)])
    bad = write_labels(tmp_path/'bad', inconsistent=True)
    with pytest.raises(ValueError, match='differs'):
        SituationInstructionIndex(bad)


def test_dataset_batch_join_never_copies_last_gt(cache, tmp_path):
    folder, bundle, rows, meta = cache
    specs = write_labels(tmp_path/'raw', missing=5)
    # Explicit namespace mapping in this synthetic fixture, never fuzzy matching.
    index = SituationInstructionIndex(specs)
    index.frames = {('train_episode', stamp): value for (_, stamp), value in index.frames.items()}
    index.frame_times = {('train_episode', frame): stamp for (_, frame), stamp in index.frame_times.items()}
    for row in rows:
        row['frame_index'] = row['sensor_source_ns']//100_000_000
    manifest = folder/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    meta['manifest_sha256'] = digest(manifest)
    (folder/'cache_meta.json').write_text(json.dumps(meta))
    path_cache = tmp_path/'path.json'
    build_path_features(bundle, path_cache)
    from unittest.mock import patch
    with patch('VLDrive.data.src.loaders.situation_instructions.SituationInstructionIndex', return_value=index):
        dataset = NavigationCacheDataset(folder, bundle, 'train', sequence_length=3,
            frame_sampling_interval=2, path_feature_cache=path_cache, situation_datasets=specs)
    baseline = NavigationCacheDataset(folder, bundle, 'train', sequence_length=3,
        frame_sampling_interval=2, path_feature_cache=path_cache)
    assert dataset.windows == baseline.windows
    assert any(any(r['frame_index']==5 for r in window) for window in dataset.windows)
    assert all(window[-1]['frame_index']-window[0]['frame_index']==4 for window in dataset.windows)
    sample = dataset[0]
    assert sample['model_inputs']['situation_instructions'] == [
        'Continue driving.', 'Continue driving.', 'Continue the avoidance maneuver to the left.']
    other = copy.deepcopy(sample)
    other['model_inputs']['situation_instructions'] = ['Frame A.', 'Frame B.', 'Frame C.']
    data = batch_samples([sample, other], 'cpu')
    assert data['model_inputs']['situation_instructions'] == [sample['model_inputs']['situation_instructions'],
                                                            other['model_inputs']['situation_instructions']]
    assert data['model_inputs']['visual_feature'].shape == (2,3,106,256)
    assert dataset.describe()['situation_instruction']['conditioning']=='ground_truth'
    assert dataset.situation_window_counts['partially_labeled'] > 0
    for i, window in enumerate(dataset.windows):
        if any(r['frame_index']==5 for r in window):
            for row, text in zip(window, dataset[i]['model_inputs']['situation_instructions']):
                assert (text is None) == (row['frame_index']==5)
    # A whole episode without Situation annotations still trains on its trajectory.
    unlabeled = NavigationCacheDataset(folder, bundle, 'validation', sequence_length=3,
        frame_sampling_interval=2, situation_datasets=specs)
    assert len(unlabeled) == 8
    assert unlabeled[0]['model_inputs']['situation_instructions'] == [None]*3
    assert unlabeled.situation_window_counts['without_text'] == len(unlabeled)


def test_batch_order_when_batch_equals_frame_count():
    samples = [dict(model_inputs=dict(visual_feature=torch.zeros(3,1,2),
                    instruction='Follow.', situation_instructions=[f'b{b}t{t}' if t != 1 else None for t in range(3)]))
               for b in range(3)]
    result = batch_samples(samples, 'cpu')['model_inputs']['situation_instructions']
    assert result == [[f'b{b}t{t}' if t != 1 else None for t in range(3)] for b in range(3)]
    mixed = copy.deepcopy(samples)
    del mixed[1]['model_inputs']['situation_instructions']
    with pytest.raises(ValueError, match='Mixed'):
        batch_samples(mixed, 'cpu')


class VariableTokenizer:
    def __call__(self, texts, **kwargs):
        ids = [[1]+[2+sum(word.encode())%125 for word in text.split()] for text in texts]
        width = max(map(len, ids))
        return dict(input_ids=torch.tensor([row+[0]*(width-len(row)) for row in ids]),
                    attention_mask=torch.tensor([[1]*len(row)+[0]*(width-len(row)) for row in ids]))


def model(monkeypatch, path=True, situation=True, zone=False):
    def load(*args):
        cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                          max_position_embeddings=512)
        return SimpleNamespace(tokenizer=VariableTokenizer(), model=install_ddia_attention(LlamaForCausalLM(cfg)))
    monkeypatch.setattr(language_model, 'load_tinyllama', load)
    return language_model.VLDriveModel(visual_feature_dim=32, precision='fp32', path_navigation=path,
        path_fusion='mlp', path_width=16, situation_instruction=situation, max_situation_tokens=48,
        zone_instruction=zone)


@pytest.mark.parametrize('path_enabled', [False, True])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('frames', [3, 11])
def test_frame_text_forward_backward_and_no_future_leak(monkeypatch, path_enabled, checkpointed, frames):
    torch.manual_seed(11)
    net = model(monkeypatch, path=path_enabled)
    if checkpointed:
        net.tinyllama.model.gradient_checkpointing_enable()
    visual = torch.randn(2, frames, 4, 32)
    sentences = ['Continue driving.', None, 'Remain stopped.']
    texts = [[sentences[i % 3] for i in range(frames)], [None]*frames]
    args = dict(situation_instructions=texts, predict_per_frame=True)
    if path_enabled:
        args['path_features'] = dict(ego_relation=torch.randn(2,frames,3), path_context=torch.randn(2,frames,5),
                                    path_valid=torch.ones(2,frames,dtype=torch.bool))
    net.train()
    output = net(['Follow the path.']*2, visual, **args)
    assert output.predicted_waypoints.shape==(2,frames,10)
    output.predicted_waypoints.square().mean().backward()
    assert net.tinyllama.get_input_embeddings().weight.grad.abs().sum()>0
    if path_enabled:
        assert net.navigation_mlp.network[0].weight.grad.abs().sum()>0
    net.eval()
    with torch.no_grad():
        original = net(['Follow the path.']*2, visual, **args)
        changed = copy.deepcopy(args)
        changed['situation_instructions'] = [row[:-1]+['There is an obstacle ahead. Avoid it to the right.'] for row in texts]
        future = net(['Follow the path.']*2, visual, **changed)
        torch.testing.assert_close(original.predicted_waypoints[:,:-1], future.predicted_waypoints[:,:-1], atol=1e-6,rtol=0)
        assert not torch.allclose(original.predicted_waypoints[:,-1], future.predicted_waypoints[:,-1])
        isolated = net(['Follow the path.'], visual[:1], **{
            k: ({n:v[:1] for n,v in value.items()} if isinstance(value,dict) else value[:1]
                if isinstance(value,list) else value) for k,value in args.items()})
        torch.testing.assert_close(original.predicted_waypoints[:1],isolated.predicted_waypoints,atol=1e-6,rtol=0)


def test_malformed_text_and_overlong_context_rejected(monkeypatch):
    net = model(monkeypatch, path=False)
    visual = torch.randn(1,3,4,32)
    for texts in (None, ['one sentence'], [['one','two']], [['one',17,'three']]):
        with pytest.raises(ValueError, match='Situation'):
            net(['Follow.'],visual,situation_instructions=texts)
    with pytest.raises(ValueError,match='token limit'):
        net(['Follow.'],visual,situation_instructions=[['word '*60]*3])
    net.tinyllama.config.max_position_embeddings=20
    with pytest.raises(ValueError,match='context length'):
        net(['Follow.'],visual,situation_instructions=[['Continue driving.']*3])
    legacy = model(monkeypatch,path=False,situation=False)
    with pytest.raises(ValueError,match='disabled'):
        legacy(['Follow.'],visual,situation_instructions=[['Continue driving.']*3])


def test_absent_situation_has_no_tokens_and_keeps_waypoint_gradients(monkeypatch):
    net = model(monkeypatch, path=True)
    baseline = model(monkeypatch, path=True, situation=False)
    baseline.load_state_dict(net.state_dict())
    visual = torch.randn(1,11,4,32, requires_grad=True)
    path = dict(ego_relation=torch.randn(1,11,3), path_context=torch.randn(1,11,5),
                path_valid=torch.ones(1,11,dtype=torch.bool))
    net.eval(); baseline.eval()
    result = net(['Follow.'], visual, path_features=path,
                 situation_instructions=[[None,'', '  ']*3+[None,None]], predict_per_frame=True)
    expected = baseline(['Follow.'], visual, path_features=path, predict_per_frame=True)
    torch.testing.assert_close(result.predicted_waypoints, expected.predicted_waypoints, atol=1e-6, rtol=0)
    assert torch.equal(result.sequence.attention_mask, expected.sequence.attention_mask)
    assert result.sequence.inputs_embeds.shape == expected.sequence.inputs_embeds.shape
    result.predicted_waypoints.square().mean().backward()
    assert visual.grad.abs().sum() > 0


@pytest.mark.parametrize('zone_enabled', [False, True])
def test_stage2_input_forwarding_and_waypoint_gt_only(monkeypatch, zone_enabled):
    net = VLDriveTrainingModel.__new__(VLDriveTrainingModel)
    torch.nn.Module.__init__(net)
    net.config = VLDriveTrainingConfig(situation_instruction=True, zone_instruction=zone_enabled)
    net.language_model = model(monkeypatch,path=False,zone=zone_enabled)
    class Connector(torch.nn.Module):
        def forward(self,x,**kw):
            return dict(language_tokens=x,selected_valid_mask=torch.ones(x.shape[:3],dtype=torch.bool),
                selected_count_per_frame=torch.full(x.shape[:2],x.shape[2]),pruning_loss=x.sum()*0,
                reconstruction_loss=x.sum()*0,cycle_loss=x.sum()*0,retention_logit_loss=x.sum()*0,
                selector=dict(retention_per_frame=torch.ones(x.shape[:2]),soft_retention_per_frame=torch.ones(x.shape[:2]),
                              probabilities=torch.zeros(*x.shape[:3],2)))
    net.connector=Connector()
    inputs=dict(visual_feature=torch.randn(1,3,4,32),instruction=['Follow.'],
                situation_instructions=[['Continue driving.',None,'Remain stopped.']])
    if zone_enabled:
        inputs['zone_ids'] = torch.tensor([[0,1,2]])
    output=net.predict(inputs)
    assert output['trajectory'].shape==(1,3,5,2)
    loss=net.loss(output,{'future_trajectory':torch.zeros_like(output['trajectory'])})['loss']
    loss.backward()
    if zone_enabled:
        assert net.language_model.zone_embedding.zone_embedding.weight.grad.abs().sum()>0
    assert not hasattr(net,'situation_head')
    with pytest.raises(ValueError,match='allowlist'):
        net.predict(dict(inputs,situation_embedding=torch.zeros(1,1,2048)))
