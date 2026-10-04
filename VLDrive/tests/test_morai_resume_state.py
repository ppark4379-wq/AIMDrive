"""CPU-only continuation equivalence and fail-before-mutation guarantees."""
import copy
import io
import random
import unittest

import numpy as np
import torch

from training.vision_encoder.utils.morai_resume_state import (
    LEGACY_TRAINER_SHA256, PROTECTED_IMPLEMENTATIONS, epoch_order,
    capture_progress, restore_progress, apply_rng_state,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(3, 3)
        self.drop = torch.nn.Dropout(.4)
        self.b = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.b(self.drop(self.a(x)))


def provenance():
    return dict(schema='morai_visual_transfer_v1', args={'seed':42},
                publications=[{'path':'index','sha256':'sealed'}],
                coverage={'train':{'frames':7}}, trainable_scope='full', loss='fixed LAV',
                batch_size=1, frame_sampling={'policy':'all','selected_frames':7},
                augmentation='none', velocity_source='ego_topic', velocity_contract='causal',
                waypoint_source='sealed', target_point='route25m', batchnorm='frozen stats',
                stop_sign='frozen', encoder_contract={'map_basis':'projected_grid'},
                optimizer={'name':'AdamW','learning_rate':5e-6,'weight_decay':.01,'schedule':'constant','restored':False},
                implementation_sha256={**{k:'same' for k in PROTECTED_IMPLEMENTATIONS},
                    'training/vision_encoder/recipes/transfer.py':LEGACY_TRAINER_SHA256})


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=.01)


def update(model, opt, value):
    opt.zero_grad(set_to_none=True)
    # Exercise all three model-side generators, not only torch dropout.
    scale = random.random() + float(np.random.random()) + 1.
    loss = model(torch.tensor([[value, value + .25, 1.]])).square().mean() * scale
    loss.backward()
    opt.step()
    return float(loss.detach())


def payload(model, opt, prov, state=None, steps=2):
    out = dict(state_dict=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
               pretraining=copy.deepcopy(prov), morai_encoder_contract=prov['encoder_contract'],
               epoch=1, steps=steps)
    if state is not None:
        out['training_state'] = state
    return out


class ResumeStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91); np.random.seed(92); random.seed(93)
        self.p = provenance()
        self.model = Model()
        self.opt = optimizer(self.model)
        self.order = epoch_order(7, 42, 0)
        for index in self.order[:2]:
            update(self.model, self.opt, index)

    def restore(self, source, model=None, opt=None, **kwargs):
        model = model or Model()
        opt = opt or optimizer(model)
        result = restore_progress(source, model, opt, self.p, self.p['encoder_contract'],
                                  dataset_size=7, seed=42, **kwargs)
        return model, opt, result

    def snapshot(self):
        return capture_progress(self.model,self.opt,self.p,self.p['encoder_contract'],
                                epoch_index=0,order=self.order,cursor=2,global_step=2,
                                include_cuda=False)

    def test_exact_future_continuation_matches_dropout_and_adam_update(self):
        state = self.snapshot()
        source = payload(self.model,self.opt,self.p,state)
        expected_loss = update(self.model,self.opt,self.order[2])
        expected_model = copy.deepcopy(self.model.state_dict())
        expected_opt = copy.deepcopy(self.opt.state_dict())
        model, opt, restored = self.restore(source)
        # Loader setup or imports may consume RNG before the explicit restore.
        random.random(); np.random.random(); torch.rand(3)
        apply_rng_state(restored['pending_rng_state'])
        actual_loss = update(model,opt,restored['order'][restored['cursor']])
        self.assertEqual(actual_loss,expected_loss)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,expected_model[name],rtol=0,atol=0)
        for pid,moments in opt.state_dict()['state'].items():
            for key,value in moments.items():
                torch.testing.assert_close(value,expected_opt['state'][pid][key],rtol=0,atol=0)
        self.assertTrue(restored['exact_next_batch'])

    def test_training_state_loads_with_weights_only(self):
        buffer=io.BytesIO();torch.save(payload(self.model,self.opt,self.p,self.snapshot()),buffer)
        buffer.seek(0)
        source=torch.load(buffer,weights_only=True)
        self.assertEqual(self.restore(source)[2]['cursor'],2)

    def test_legacy_preserves_moments_steps_and_remaining_budget(self):
        source=payload(self.model,self.opt,self.p)
        model,opt,restored=self.restore(source,allow_legacy_sampler_reset=True)
        self.assertEqual(restored['global_step'],2)
        self.assertEqual(restored['epoch_index'],0)
        self.assertEqual(restored['remaining_epoch_samples'],5)
        self.assertIsNone(restored['pending_rng_state'])
        self.assertFalse(restored['exact_next_batch'])
        self.assertIn('repeat or omit',restored['continuity']['caveat'])
        for pid,moments in opt.state_dict()['state'].items():
            for key,value in moments.items():
                torch.testing.assert_close(value,source['optimizer']['state'][pid][key],rtol=0,atol=0)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,source['state_dict'][name],rtol=0,atol=0)

    def test_legacy_requires_explicit_reset_and_audited_parameter_order(self):
        source=payload(self.model,self.opt,self.p)
        with self.assertRaisesRegex(ValueError,'sampler/RNG'):
            self.restore(source)
        source['pretraining']['implementation_sha256']['training/vision_encoder/recipes/transfer.py']='unknown'
        with self.assertRaisesRegex(ValueError,'not audited'):
            self.restore(source,allow_legacy_sampler_reset=True)

    def test_wrong_mapping_rejected_before_any_weight_optimizer_rng_mutation(self):
        source=payload(self.model,self.opt,self.p,self.snapshot())
        # a.weight and b.weight have identical shapes; shape-only guards fail here.
        source['training_state']['optimizer_parameter_names'][0][0]='b.weight'
        model=Model();opt=optimizer(model);weights=copy.deepcopy(model.state_dict())
        rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,'mapping'):
            self.restore(source,model,opt)
        self.assertFalse(opt.state)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,weights[name]))

    def test_legacy_same_shape_parameter_reordering_is_rejected(self):
        source=payload(self.model,self.opt,self.p)
        model=Model();parameters=list(model.parameters())
        parameters[0],parameters[2]=parameters[2],parameters[0]
        opt=torch.optim.AdamW(parameters,lr=5e-6,weight_decay=.01)
        with self.assertRaisesRegex(ValueError,'parameter order'):
            self.restore(source,model,opt,allow_legacy_sampler_reset=True)

    def test_changed_lr_or_data_or_scope_is_rejected(self):
        for kind in ('lr','data','scope','map','loss'):
            source=payload(self.model,self.opt,self.p,self.snapshot())
            model=Model();opt=optimizer(model)
            if kind=='lr':opt.param_groups[0]['lr']=1e-4
            if kind=='data':source['pretraining']['publications'][0]['sha256']='changed'
            if kind=='scope':source['pretraining']['trainable_scope']='object_head'
            if kind=='map':source['morai_encoder_contract']={'map_basis':'true_enu'}
            if kind=='loss':source['pretraining']['loss']='different objective'
            with self.assertRaises(ValueError):self.restore(source,model,opt)
            self.assertFalse(opt.state)

    def test_changed_or_missing_seed_rejected_before_mutation(self):
        for kind in ('source', 'argument', 'missing'):
            source=payload(self.model,self.opt,self.p,self.snapshot())
            model=Model();opt=optimizer(model);before=copy.deepcopy(model.state_dict())
            rng=torch.get_rng_state().clone()
            if kind=='source':source['pretraining']['args']['seed']=43
            if kind=='missing':del source['pretraining']['args']['seed']
            with self.assertRaises(ValueError):
                restore_progress(source,model,opt,self.p,self.p['encoder_contract'],
                                 dataset_size=7,seed=43 if kind=='argument' else 42)
            self.assertFalse(opt.state)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,before[name]))

    def test_loader_and_evaluation_changes_do_not_reset_training(self):
        source=payload(self.model,self.opt,self.p,self.snapshot())
        source['pretraining']['implementation_sha256']['data/src/loaders/morai_sensor_record_dataset.py']='old IO'
        source['pretraining']['implementation_sha256']['utils/evaluation/morai_object_metrics.py']='old eval'
        _,_,restored=self.restore(source)
        self.assertTrue(restored['optimizer_restored'])

    def test_malformed_cursor_or_rng_or_moments_rejected(self):
        for kind in ('order','cursor','rng','moment'):
            source=payload(self.model,self.opt,self.p,self.snapshot())
            if kind=='order':source['training_state']['order'][0]=source['training_state']['order'][1]
            if kind=='cursor':source['training_state']['cursor']=3
            if kind=='rng':source['training_state']['rng_state']['torch_cpu']=torch.tensor([1.])
            if kind=='moment':source['optimizer']['state'][0]['exp_avg']=torch.zeros(9)
            with self.assertRaises(ValueError):self.restore(source)

    def test_epoch_boundary_keeps_empty_remainder_for_pending_validation(self):
        source=payload(self.model,self.opt,self.p,steps=7)
        _,_,state=self.restore(source,allow_legacy_sampler_reset=True)
        self.assertEqual((state['epoch_index'],state['cursor'],state['remaining_epoch_samples']),(0,7,0))

    def test_real_adapter_tuple_list_contract_is_equivalent_but_values_are_not(self):
        # Same four sequence paths as the live checkpoint: raw adapter dataclass
        # tuples versus provenance written through JSON's arrays.
        adapter = dict(lidar_translation_xyz=[1., 0., 1.4800000190734863],
                       lidar_rotation_xyzw=[0., 0., 0., 1.])
        self.p['encoder_contract'] = dict(schema='morai_sensor_record_encoder_v2',
            adapter=copy.deepcopy(adapter), native_adapter_configs=[copy.deepcopy(adapter)],
            map_basis={'orientation_frame':'projected_grid','origin_xyz':[302595.,4124145.,0.]})
        source = payload(self.model,self.opt,self.p)
        source['morai_encoder_contract'] = copy.deepcopy(self.p['encoder_contract'])
        for item in [source['morai_encoder_contract']['adapter'],
                     *source['morai_encoder_contract']['native_adapter_configs']]:
            for key in ('lidar_translation_xyz','lidar_rotation_xyzw'):
                item[key] = tuple(item[key])
        _, opt, result = self.restore(source,allow_legacy_sampler_reset=True)
        self.assertTrue(result['optimizer_restored'])
        self.assertTrue(opt.state)
        source['morai_encoder_contract']['adapter']['lidar_translation_xyz']=(1.,0.,1.5)
        model=Model();opt=optimizer(model);before=copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError,'contradicts'):
            self.restore(source,model,opt,allow_legacy_sampler_reset=True)
        self.assertFalse(opt.state)
        for name,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,before[name]))

    def test_permutation_isolated_from_global_model_rng(self):
        before=torch.get_rng_state().clone()
        self.assertEqual(epoch_order(7,42,0),self.order)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':
    unittest.main()
