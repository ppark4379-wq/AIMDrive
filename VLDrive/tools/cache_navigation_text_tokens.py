#!/usr/bin/env python3
"""Tokenize every navigation text sampled by the configured Stage 2 schedule."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from utils.runtime import configure_runtime, sha256
configure_runtime()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True,help='VLDrive Stage 2 training config')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    from transformers import AutoTokenizer
    from datasets.navigation_cache_dataset import NavigationCacheDataset
    from navigation.token_cache import tokenizer_fingerprint
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/cfg['model']['language_checkpoint']),
                                          local_files_only=True,use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    fingerprint=tokenizer_fingerprint(tokenizer)
    language=cfg['navigation_sampling']
    folder=ROOT/language['token_cache']
    folder.mkdir(parents=True,exist_ok=True)
    texts=set();counts={};examples=[]
    for split in ('train','val'):
        dataset=NavigationCacheDataset(ROOT/cfg['feature_cache'],ROOT/cfg['instructions'],split,
            cfg['sequence_length'],cfg['seed'],train_variants=language['train_variants'],
            validation_variants=language['validation_variants'],
            frame_sampling_interval=cfg.get('frame_sampling_interval',1))
        counts[split]=len(dataset)
        # Enumerate text only, without rereading image-feature arrays for every epoch.
        for epoch in range(cfg['epochs'] if split=='train' else 1):
            for index, rows in enumerate(dataset.windows):
                times=[r['sensor_source_ns'] for r in rows]
                sample=dataset.labels.sample(rows[0]['episode_id'],times,cfg['seed'],epoch,dataset.variants,
                                             sampling_interval=dataset.frame_sampling_interval)
                texts.add(sample['text'])
                if epoch==0:
                    examples.append(dict(split=split,window_index=index,episode_id=rows[0]['episode_id'],
                                         sensor_source_ns=times,**sample))
        print('Prepared text windows:',split,len(dataset),'unique texts:',len(texts),flush=True)
    tokens=[]
    maximum=0
    ordered=sorted(texts)
    for begin in range(0,len(ordered),256):
        batch=ordered[begin:begin+256]
        encoded=tokenizer(batch,padding=False,truncation=False)['input_ids']
        for text,ids in zip(batch,encoded):
            maximum=max(maximum,len(ids))
            if len(ids)>cfg['model']['max_instruction_tokens']:
                raise ValueError('Navigation exceeds configured context; refusing truncation')
            tokens.append(dict(text=text,input_ids=ids))
    lookup={row['text']:row['input_ids'] for row in tokens}
    for row in examples:
        row['input_ids']=lookup[row['text']]
    for name, rows in [('tokens.jsonl',tokens),('window_instructions.jsonl',examples)]:
        path=folder/name
        pending=path.with_suffix('.pending')
        pending.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        pending.replace(path)
    meta=dict(schema='navigation-text-token-cache-v1',tokenizer_sha256=fingerprint,
        tokens_sha256=sha256(folder/'tokens.jsonl'),unique_texts=len(tokens),max_token_length=maximum,
        epochs=cfg['epochs'],seed=cfg['seed'],windows=counts,config_sha256=sha256(args.config),
        sequence_length=cfg['sequence_length'],frame_sampling_interval=cfg.get('frame_sampling_interval',1),
        labels_manifest_sha256=sha256(ROOT/cfg['instructions']/'manifest.json'),
        visual_manifest_sha256=sha256(ROOT/cfg['feature_cache']/'manifest.jsonl'))
    (folder/'token_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    main()
