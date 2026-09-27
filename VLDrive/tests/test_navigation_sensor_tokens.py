"""Regression checks for actual-ego targets, split leakage, and cached text IDs."""
import hashlib
import json

import numpy as np
import pytest

from data.loaders.navigation_sensor_record import future_target, temporal_split
from navigation.token_cache import CachedNavigationTokenizer, tokenizer_fingerprint


def test_targets_use_current_ego_axes_and_real_future_positions():
    times=list(range(0,3_000_000_001,100_000_000))
    # Driving north and bending west: forward negative; left positive.
    positions=[[ -2*(t/1e9)**2, 4*t/1e9] for t in times]
    result=future_target(times,positions,0,np.pi/2,150_000_000)
    expected=[[2*t*t,-4*t] for t in (.5,1,1.5,2,2.5)]
    assert np.allclose(result,expected)
    assert future_target(times,positions,10,np.pi/2,150_000_000) is None
    assert future_target(times,positions,0,None,150_000_000) is None
    positions[12]=None
    assert future_target(times,positions,0,0,150_000_000) is None


def test_interpolation_rejects_missing_intermediate_sensor_intervals():
    times=list(range(0,3_000_000_001,100_000_000))
    positions=[[t/1e9,0] for t in times]
    del times[7:9]
    del positions[7:9]
    assert future_target(times,positions,0,0,150_000_000) is None


def test_split_purges_even_the_target_exactly_on_boundary():
    boundary=10_000_000_000
    assert temporal_split(7_400_000_000,boundary)=='train'
    assert temporal_split(7_500_000_000,boundary) is None
    assert temporal_split(9_900_000_000,boundary) is None
    assert temporal_split(boundary,boundary)=='validation'


class FakeTokenizer:
    class Backend:
        def to_str(self): return '{"model":{"vocab":{"left":3}},"padding":null}'
    backend_tokenizer=Backend()
    special_tokens_map={'bos_token':'<s>','pad_token':'</s>'}
    padding_side='right'
    add_bos_token=True
    add_eos_token=False
    def __len__(self): return 20
    def __call__(self,text,**kwargs): return {'input_ids':[1,5]}
    def pad(self,rows,**kwargs): return rows


def test_cached_text_ids_are_consumed_and_new_instructions_remain_supported(tmp_path):
    tokenizer=FakeTokenizer()
    content=json.dumps(dict(text='Turn left.',input_ids=[1,3]))+'\n'
    (tmp_path/'tokens.jsonl').write_text(content)
    meta=dict(schema='navigation-text-token-cache-v1',tokenizer_sha256=tokenizer_fingerprint(tokenizer),
              labels_manifest_sha256='labels',visual_manifest_sha256='visual',unique_texts=1,
              tokens_sha256=hashlib.sha256((tmp_path/'tokens.jsonl').read_bytes()).hexdigest())
    (tmp_path/'token_meta.json').write_text(json.dumps(meta))
    wrapped=CachedNavigationTokenizer(tokenizer,tmp_path,'labels','visual')
    rows=wrapped(['Turn left.','New instruction'],padding=True,truncation=False,return_tensors='pt')
    assert rows[0]['input_ids']==[1,3] and rows[1]['input_ids']==[1,5]
    assert wrapped.hits==1 and wrapped.misses==1
    with pytest.raises(ValueError,match='mismatch'):
        CachedNavigationTokenizer(tokenizer,tmp_path,'changed labels','visual')
    (tmp_path/'tokens.jsonl').write_text(content+'\n')
    with pytest.raises(ValueError,match='mismatch'):
        CachedNavigationTokenizer(tokenizer,tmp_path,'labels','visual')
