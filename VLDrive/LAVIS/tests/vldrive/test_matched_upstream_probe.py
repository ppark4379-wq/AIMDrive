"""CPU invariants for the isolated actual-graph A/B runner."""
from pathlib import Path
import importlib.util
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('matched_upstream_probe',ROOT/'tools/morai/train_matched_upstream_probe.py')
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.connector=nn.Sequential(nn.Linear(4,4),nn.Dropout(.2))
        self.language_model=nn.Module()
        self.language_model.waypoints_predictor=nn.Linear(4,2)
        self.unused=nn.Parameter(torch.zeros(1),requires_grad=False)

    def forward(self,x):
        return self.language_model.waypoints_predictor(self.connector(x))


class Tests(unittest.TestCase):
    def test_same_forward_freeze_only(self):
        torch.manual_seed(607); a=Tiny(); b=Tiny(); b.load_state_dict(a.state_dict())
        original=probe.set_policy(a,'A'); self.assertEqual(probe.set_policy(b,'B'),original)
        a.train(); b.train(); x=torch.ones(1,11,4)
        torch.manual_seed(607); ya=a(x)
        torch.manual_seed(607); yb=b(x)
        self.assertTrue(torch.equal(ya,yb))
        self.assertTrue(a.connector.training)
        self.assertTrue(a.connector[1].training)
        ya.square().mean().backward(); yb.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in a.connector.parameters()))
        self.assertTrue(all(p.grad is not None for p in b.connector.parameters()))
        self.assertFalse(a.unused.requires_grad); self.assertFalse(b.unused.requires_grad)

    def test_exact_repetitions(self):
        from collections import Counter
        order=probe.order_for(list(range(32)),1280)
        self.assertEqual(Counter(order),Counter({i:40 for i in range(32)}))
        self.assertEqual(order[:2],probe.order_for(list(range(32)),2))

    def test_loss_uses_all_history_coordinates(self):
        p=torch.zeros(1,11,5,2); gt=p.clone(); gt[0,0,0,0]=110.
        self.assertEqual(float(torch.nn.functional.l1_loss(p,gt)),1.)

    def test_comparison_rejects_scope_independent_drift(self):
        item={k:{} for k in ('common','initial_original_trainable_hash','initial_original_frozen_hash','before','before_all_train','train_forward')}
        probe.same_initial(item,dict(item,arm='B'))
        with self.assertRaises(ValueError):
            probe.same_initial(item,dict(item,train_forward={'wrong':True}))

    def test_actual_training_forward_gate(self):
        values={k:1. for k in ('loss','waypoint_loss','pruning_loss','reconstruction_loss','sampled_retention',
            'soft_retention','saturated_keep_fraction','retention_logit_loss')}
        signature={'selected_count':[[106]*11],'trajectory':[[[0.,0.]]]}
        reference=dict(values,actual_first_training_forward=signature)
        probe.same_actual_training_forward(reference,values,signature)
        with self.assertRaises(ValueError):
            probe.same_actual_training_forward(reference,dict(values,waypoint_loss=2.),signature)
        with self.assertRaises(ValueError):
            probe.same_actual_training_forward(reference,values,dict(signature,selected_count=[[1]*11]))

    def test_update_samples_measure_real_change(self):
        before={'head':torch.tensor([1.,2.])}; after={'head':torch.tensor([1.,3.])}
        self.assertEqual(probe.changes(before,after)['head']['changed_elements'],1)
        self.assertEqual(probe.changes(before,after)['head']['delta_l2'],1.)

    def test_original_requires_grad_subset_not_new_unfreeze(self):
        model=Tiny(); model.connector[0].bias.requires_grad_(False)
        before={n:p.requires_grad for n,p in model.named_parameters()}
        probe.set_policy(model,'B')
        self.assertEqual(before,{n:p.requires_grad for n,p in model.named_parameters()})


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
