"""Train only static-zone Situation Head from verified, frozen VE cache."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...data.src.loaders.situation_dataset import SituationCacheDataset, load_config, project_path
from ...models.llm.situation_head import SituationHead
from .trainer import SituationTrainingConfig, fit, _to_device
from .loss import situation_loss
from .labels import vocabulary

CONFIG=Path(__file__).with_name('config.static_gt.json')
LOGS=project_path('outputs/logs/situation_static_gt_20261008')


def status(state, **kwargs):
    tmp=LOGS/'status.tmp'
    tmp.write_text(json.dumps(dict(state=state,**kwargs),ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(LOGS/'status.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();LOGS.mkdir(parents=True,exist_ok=True)
    if args.launch:
        with (LOGS/'train.log').open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','VLDrive.training.situation.train_static'],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        print(json.dumps(dict(pid=child.pid,log=str(LOGS/'train.log'))));return
    import msvcrt
    with (LOGS/'train.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            status('loading',pid=os.getpid())
            torch.set_num_threads(2);cfg=load_config(CONFIG)
            torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
            dataset=SituationCacheDataset(cfg)
            before=len(dataset)
            dataset.windows=[(window,records) for window,records in dataset.windows
                if all(r.get('zone')=='static_obstacle' and r['zone_instruction']==1 for r in window)]
            assert dataset.windows, 'No static-only windows'
            dataset.vocab=vocabulary([window[-1]['labeling'] for window,_ in dataset.windows])
            dataset.counts=Counter(windows=len(dataset),zone_1=len(dataset),excluded_nonstatic_windows=before-len(dataset))
            # Validate the entire temporal window, not only its anchor or folder.
            assert all(all(r.get('zone')=='static_obstacle' and r['zone_instruction']==1 for r in window)
                       for window,_ in dataset.windows), 'Non-static zone in training samples'
            assert not dataset.feature_source.get('diagnostic_frame_limit'), 'Partial cache'
            assert 'to the left and right of' in dataset.vocab['position']
            settings=SituationTrainingConfig(**cfg['training']);settings.validate()
            model=SituationHead({k:len(v) for k,v in dataset.vocab.items()},**cfg['model']).to(settings.device)
            loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,pin_memory=True,
                              generator=torch.Generator().manual_seed(cfg['seed']))
            model.eval();batch=_to_device(next(iter(loader)),settings.device)
            with torch.no_grad():loss,metrics=situation_loss(model(**batch['inputs']),batch['targets'])
            assert metrics and torch.isfinite(loss)
            selected={records[-1]['episode_id'] for _,records in dataset.windows}
            source=dict(dataset.feature_source)
            source['labels_sha256']={k:v for k,v in source['labels_sha256'].items() if k in selected}
            source['training_selection']=dict(zone_id=1,all_window_frames_static=True,samples=len(dataset),
                episodes=len(selected),initialization='fresh_head',vision_encoder_frozen=True)
            info=dict(pid=os.getpid(),samples=len(dataset),episodes=len(selected),steps_per_epoch=len(loader),
                      epochs=settings.epochs,vocabulary=dataset.vocab,initial_loss=loss.item(),counts=dict(dataset.counts))
            print(json.dumps(info,ensure_ascii=False),flush=True)
            if args.check_only:status('check_passed',**info);return
            destination=project_path(cfg['output_root'])/datetime.now().strftime('situation_%Y%m%d_%H%M%S_%f')
            info['output']=str(destination);status('training',**info)
            fit(model,loader,dataset.vocab,destination,settings,source,dict(enabled=False))
            status('complete',**info)
        except BaseException as exc:
            status('failed',error=str(exc),pid=os.getpid());raise


if __name__=='__main__':main()
