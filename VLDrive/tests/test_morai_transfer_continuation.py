"""Exercise the real transfer control flow, replacing only expensive data/model IO."""
import argparse
import contextlib
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from training.vision_encoder.recipes import transfer
from training.vision_encoder.utils.morai_runtime_state import EpochOrderSampler,nested_state_digest


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.5]));self.return_feature=False
    def forward(self,inputs):
        if self.return_feature:return self.weight.expand(1,106,256)
        return self.weight*inputs['x']


class TinyDataset(Dataset):
    def __init__(self,*args):
        adapter=type('Adapter',(),{'config':{}})()
        self.rows=[(None,{'object_positive':True,'traffic_light':i%2},adapter,0.) for i in range(4)]
        self.coverage=Counter(frames=4,object_positive=4,object_valid=4,traffic_light_0=2,traffic_light_1=2)
        self.excluded=Counter();self.manifest_hashes={'manifest':'same'}
    def __len__(self):return 4
    def __getitem__(self,i):return {'x':torch.tensor([float(i+1)])},{'dummy':torch.tensor(i)}


def loss(outputs,targets,**kw):
    del targets,kw
    value=outputs.square().mean()
    return value,dict(detection=value,waypoint_l1=value*0,object_speed=value*0,traffic_light_ce=value*0)


class TransferContinuationTests(unittest.TestCase):
    def execute(self,tmp,name,resume=None,updates=0):
        args=transfer.add_arguments(argparse.ArgumentParser()).parse_args([
            '--manifest-index',str(tmp/'index.json'),'--checkpoint',str(tmp/'base.pt'),
            '--checkpoint-sha256','base','--output',str(tmp/name),'--epochs','1',
            '--initialization','morai_checkpoint','--device','cpu','--velocity-source','ego_topic',
            '--stop-after-updates',str(updates),'--learning-rate','0.000005',
            *(['--resume-from',str(resume)] if resume else [])])
        index={'publications':[{'path':'index','sha256':'same'}]}
        doc={'navigation_contract':{'global_path_sha256':'route'},'entries':[{} for _ in range(4)]}
        manifests={'train':[(tmp/'manifest.json',doc)],'val':[(tmp/'val.json',doc)]}
        def evaluate(model,*a,**kw):
            # Updating a positive scalar downwards improves this test score.
            score=1-float(model.weight.detach())
            return dict(frames=4,loss=score,detection_cell_average_precision=score,
                        detection_cell_best_f1=score,detection_cell_best_threshold=.5)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(transfer,'load_publications',return_value=(index,manifests)))
            stack.enter_context(patch.object(transfer,'SensorRecordDataset',TinyDataset))
            stack.enter_context(patch('data.src.loaders.morai_map_geometry.manifest_map_basis',return_value={'grid':True}))
            stack.enter_context(patch.object(transfer,'sensor_record_encoder_contract',return_value={'grid':True}))
            stack.enter_context(patch.object(transfer,'navigation_recipe',return_value={'recipe':'same'}))
            stack.enter_context(patch.object(transfer,'make_model',side_effect=lambda *a:TinyModel()))
            stack.enter_context(patch.object(transfer,'configure_trainable_scope',return_value=transfer.GROUPS))
            stack.enter_context(patch.object(transfer,'train_mode',side_effect=lambda *a:None))
            stack.enter_context(patch.object(transfer,'gradients',return_value={k:True for k in transfer.GROUPS}))
            stack.enter_context(patch.object(transfer,'multitask_loss',side_effect=loss))
            stack.enter_context(patch.object(transfer,'evaluate',side_effect=evaluate))
            stack.enter_context(patch('training.vision_encoder.utils.morai_runtime_state.install_stop_handlers',return_value={'requested':False}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return transfer.run(args)

    def test_actual_trainer_continues_remaining_updates_and_keeps_original_anchor(self):
        with tempfile.TemporaryDirectory() as name:
            tmp=Path(name);torch.save({'state_dict':TinyModel().state_dict()},tmp/'base.pt')
            first=self.execute(tmp,'first',updates=2)
            self.assertEqual(first['steps'],2)
            resumed=self.execute(tmp,'resumed',resume=Path(first['checkpoint']))
            self.assertEqual(resumed['optimizer_updates'],4)
            self.assertEqual(resumed['optimizer_updates_this_run'],2)
            self.assertTrue(resumed['resume']['weights_identical'])
            self.assertTrue(resumed['resume']['optimizer_state_identical'])
            baseline=json.loads((tmp/'resumed/baseline.json').read_text())
            self.assertEqual(baseline['detection_cell_average_precision'],.5)
            source=torch.load(first['checkpoint'],weights_only=True)
            continued=torch.load(tmp/'resumed/checkpoint-last.pth.tar',weights_only=True)
            self.assertEqual(continued['training_state']['cursor'],4)
            self.assertEqual(continued['training_state']['order'],source['training_state']['order'])
            self.assertTrue((tmp/'resumed/handoff_confirmed.json').is_file())

    def test_epoch_boundary_resume_evaluates_without_repeating_training(self):
        with tempfile.TemporaryDirectory() as name:
            tmp=Path(name);torch.save({'state_dict':TinyModel().state_dict()},tmp/'base.pt')
            first=self.execute(tmp,'first',updates=4)
            source=torch.load(first['checkpoint'],weights_only=True)
            resumed=self.execute(tmp,'resumed',resume=Path(first['checkpoint']))
            self.assertEqual(resumed['optimizer_updates_this_run'],0)
            final=torch.load(tmp/'resumed/checkpoint-last.pth.tar',weights_only=True)
            self.assertEqual(nested_state_digest(source['state_dict']),nested_state_digest(final['state_dict']))
            self.assertEqual(nested_state_digest(source['optimizer']),nested_state_digest(final['optimizer']))

    def test_prefetch_iteration_does_not_advance_consumed_cursor(self):
        sampler=EpochOrderSampler([2,0,1,3],1)
        self.assertEqual(list(sampler),[0,1,3])
        self.assertEqual(sampler.cursor,1)
        sampler.set_order([2,0,1,3],2)
        self.assertEqual(list(sampler),[1,3])

if __name__=='__main__':unittest.main()
