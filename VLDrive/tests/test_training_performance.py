"""I/O optimizations preserve inputs, updates, checksums and resume state."""
import copy
from types import SimpleNamespace

import pytest
import torch

from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
from .test_navigation_cache import cache
from training.connector_llm.training import evaluate


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    else:
        assert a == b


def test_array_reuse_is_exact_bounded_and_cannot_be_mutated(cache, monkeypatch):
    import numpy as np
    folder, labels, rows, _ = cache
    baseline = NavigationCacheDataset(folder, labels, 'train', sequence_length=3)
    reused = NavigationCacheDataset(folder, labels, 'train', sequence_length=3, array_cache_bytes=400_000)
    for epoch in (0, 3):
        baseline.set_epoch(epoch); reused.set_epoch(epoch)
        for i in (0, 0, 1, 5, 1):
            same(baseline[i], reused[i])
            assert reused._array_cache_size <= reused._array_cache_limit
    reference = rows[0]['visual_feature']
    expected = reused._array(reference, (106, 256)).clone()
    def cannot_load(*args, **kwargs): raise AssertionError('LRU hit reopened the array')
    monkeypatch.setattr(np, 'load', cannot_load)
    result = reused._array(reference, (106, 256)); result.add_(100)
    assert torch.equal(reused._array(reference, (106, 256)), expected)


def test_array_reuse_still_rejects_changed_files_and_references(cache):
    folder, labels, rows, _ = cache
    data = NavigationCacheDataset(folder, labels, 'train', sequence_length=3, array_cache_bytes=400_000)
    ref = rows[0]['visual_feature']
    data._array(ref, (106, 256))
    with pytest.raises(ValueError, match='Cached tensor changed'):
        data._array(dict(ref, sha256='f'*64), (106, 256))
    data._array(ref, (106, 256))
    (folder/ref['path']).write_bytes(b'changed after first read')
    with pytest.raises(ValueError, match='Cached tensor changed'):
        data._array(ref, (106, 256))
    with pytest.raises(ValueError, match='Unsafe'):
        data._array(dict(ref, path='../outside.npy'), (106, 256))


def fast_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({'[PAD]':0, '[UNK]':1, 'Follow':2, 'Continue':3,
        'driving':4, '.':5, 'Remain':6, 'stopped':7, 'Situation':8, ':':9}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_token_ids_masks_and_eviction_match_direct_tokenization(monkeypatch, side):
    from .test_situation_instruction import model
    net = model(monkeypatch, path=False)
    net.tokenizer = fast_tokenizer(); net.tokenizer.padding_side = side
    net.token_cache_size = 2
    for texts in (['Follow.', 'Continue driving.'], ['Remain stopped.'],
                  ['Follow.', 'Continue driving.', 'Remain stopped.', 'Follow.']):
        expected = dict(net.tokenizer(texts, padding=True, truncation=False, return_tensors='pt'))
        rng = torch.get_rng_state().clone()
        same(expected, net._tokenize(texts, 'cpu'))
        assert torch.equal(rng, torch.get_rng_state())
        assert len(net._token_id_cache) <= 2
    with pytest.raises(ValueError, match='token limit'):
        net._tokenize(['Continue driving.'], 'cpu', max_length=1)
    replacement = fast_tokenizer(); replacement.add_tokens(['Continue'])
    net.tokenizer = replacement
    same(dict(replacement(['Follow.'], padding=True, truncation=False, return_tensors='pt')),
         net._tokenize(['Follow.'], 'cpu'))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cached_text_preserves_loss_gradients_updates_and_live_embeddings(monkeypatch, dtype):
    from .test_situation_instruction import model
    from training.common.optimizers import CPUAdamW
    torch.manual_seed(5)
    old = model(monkeypatch, path=False, zone=True).to(dtype=dtype)
    old.tokenizer = fast_tokenizer()
    new = copy.deepcopy(old); new.token_cache_size = 8
    visual = torch.randn(2, 11, 4, 32)
    args = dict(situation_instructions=[[None,'Continue driving.',None,'Remain stopped.']*2+[None]*3,
                                        ['Remain stopped.']*11],
                zone_ids=torch.tensor([[0,1,2,0,1,2,0,1,2,0,1]]*2), predict_per_frame=True)
    optimizers = [CPUAdamW(net.parameters(), lr=1e-3) for net in (old, new)]
    initial = new.tinyllama.get_input_embeddings().weight.detach().clone()
    for _ in range(2):
        results = []
        for net, optimizer in zip((old,new), optimizers):
            optimizer.zero_grad()
            result = net(['Follow.','Follow.'], visual, **args)
            loss = result.predicted_waypoints.square().mean()
            loss.backward()
            results.append((result, loss.detach()))
        same(results[0][1], results[1][1])
        for key in ('inputs_embeds', 'attention_mask', 'position_ids', 'frame_readout_indices'):
            same(getattr(results[0][0].sequence, key), getattr(results[1][0].sequence, key))
        for (name,a), (_,b) in zip(old.named_parameters(),new.named_parameters()):
            if a.grad is not None: same(a.grad,b.grad)
        for optimizer in optimizers: optimizer.step()
        for a,b in zip(old.parameters(),new.parameters()): same(a,b)
    assert not torch.equal(initial, new.tinyllama.get_input_embeddings().weight)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_snapshot_checksum_is_identical_to_model_checksum(dtype):
    from training.common.checkpoints import _snapshot_hash, parameter_hash
    from .test_vldrive_trainer_utilities import _Model
    net = _Model().to(dtype=dtype)
    snapshot = {name:p.detach().cpu().clone() for name,p in net.named_parameters() if p.requires_grad}
    assert _snapshot_hash(snapshot) == parameter_hash(net, trainable=True)


def test_batched_evaluation_weights_short_last_batch_and_preserves_rng(tmp_path):
    class Dataset:
        def __getitem__(self,index):
            return dict(model_inputs={'x':torch.tensor(float(index+1))},
                training_targets={'future_trajectory':torch.zeros(1,5,2)},
                metadata={'run_id':'route','sample_ids':['route/'+str(index)]})
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.config=SimpleNamespace(precision='fp32')
        def predict(self,inputs,**kwargs):
            b=len(inputs['x'])
            return dict(trajectory=inputs['x'][:,None,None,None].expand(b,1,5,2),selected_count=torch.ones(b,1,dtype=torch.long))
    net,data=Model(),Dataset()
    first=evaluate(net,data,list(range(5)),'cpu',batch_size=1)
    rng=torch.get_rng_state().clone();progress=[]
    second=evaluate(net,data,list(range(5)),'cpu',batch_size=4,progress_callback=progress.append,
                    save_predictions=tmp_path/'predictions.json')
    assert torch.equal(rng,torch.get_rng_state())
    assert second['waypoint_l1']==3 and second['windows']==5
    assert first['ade_m']==second['ade_m'] and first['fde_m']==second['fde_m']
    assert progress[-1]['completed']==progress[-1]['total']==5
    import json
    assert [r['sample_id'] for r in json.loads((tmp_path/'predictions.json').read_text())]==['route/'+str(i) for i in range(5)]
