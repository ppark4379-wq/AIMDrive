"""Check the opt-in bbox objective through the real multitask loss and CLI."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import torch
from data.src.loaders.morai_transfer_targets import render_object_target
from training.vision_encoder.losses.morai_bbox_auxiliary import relative_bbox_loss
from training.vision_encoder.losses.morai_transfer import multitask_loss
from training.vision_encoder.recipes.transfer import add_arguments, resolve_args


class BboxTrainingIntegrationTests(unittest.TestCase):
    def sample(self):
        obj=dict(x=10., y=0., heading_deg=35., length=.6, width=.4, height=1.,
                 vel_x=0., vel_y=0., geometry_valid=True)
        target,valid,_,ids=render_object_target(
            {'ego_raw':dict(x=0.,y=0.,yaw=0.)},[obj],
            target_policy='small_object_support_v1',return_instance_ids=True)
        self.assertTrue(valid)
        t=torch.tensor(target)[None]
        det=t.clone();det[:,:,1]+=.1;det[:,:,5]+=.05;det.requires_grad_(True)
        wp=torch.ones(1,5,2,requires_grad=True)
        light=torch.zeros(1,2,requires_grad=True)
        targets=dict(detection=t,object_instance_ids=torch.tensor(ids)[None],
                     object_valid=torch.tensor([True]),waypoint_valid=torch.tensor([True]),
                     waypoints=torch.zeros_like(wp),traffic_light=torch.tensor([1]))
        return [det,wp,light],targets

    def test_auxiliary_changes_only_the_explicitly_weighted_objective(self):
        outputs,t=self.sample()
        base,tasks=multitask_loss(outputs,t,detection_loss_policy='lmdrive_lav_v1')
        enhanced,enhanced_tasks=multitask_loss(outputs,t,detection_loss_policy='lmdrive_lav_v1',geometry_aux_weight=.2)
        auxiliary,_=relative_bbox_loss(outputs[0],t['detection'],t['object_instance_ids'],t['object_valid'])
        torch.testing.assert_close(enhanced-base,.2*auxiliary)
        for key in tasks:torch.testing.assert_close(tasks[key],enhanced_tasks[key],rtol=0,atol=0)
        self.assertNotIn('bbox_auxiliary',tasks)
        base_grads=torch.autograd.grad(base,outputs,retain_graph=True)
        enhanced_grads=torch.autograd.grad(enhanced,outputs)
        torch.testing.assert_close(base_grads[1],enhanced_grads[1],rtol=0,atol=0)
        torch.testing.assert_close(base_grads[2],enhanced_grads[2],rtol=0,atol=0)
        self.assertFalse(torch.equal(base_grads[0],enhanced_grads[0]))

    def test_missing_instance_assignment_fails_only_when_opted_in(self):
        outputs,t=self.sample();del t['object_instance_ids']
        multitask_loss(outputs,t,detection_loss_policy='lmdrive_lav_v1')
        with self.assertRaisesRegex(ValueError,'object_instance_ids'):
            multitask_loss(outputs,t,detection_loss_policy='lmdrive_lav_v1',geometry_aux_weight=.2)

    def test_masked_geometry_does_not_add_loss(self):
        outputs,t=self.sample();t['object_valid'][:]=False
        base,_=multitask_loss(outputs,t,detection_loss_policy='lmdrive_lav_v1')
        improved,tasks=multitask_loss(outputs,t,detection_loss_policy='lmdrive_lav_v1',geometry_aux_weight=.2)
        torch.testing.assert_close(base,improved,rtol=0,atol=0)
        self.assertEqual(float(tasks['bbox_auxiliary']),0.)

    def test_invalid_auxiliary_config_rejected(self):
        outputs,t=self.sample()
        for weight in [-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                multitask_loss(outputs,t,detection_loss_policy='lmdrive_lav_v1',geometry_aux_weight=weight)
        with self.assertRaisesRegex(ValueError,'LAV'):
            multitask_loss(outputs,t,detection_loss_policy='mvt_l1_v1',geometry_aux_weight=.2)

    def test_cli_defaults_preserve_old_targets_and_opt_in_is_explicit(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'config.json'
            path.write_text(json.dumps(dict(manifest_indices=[str(Path(name)/'index.json')],checkpoint=str(Path(name)/'base.pt'))))
            parser=add_arguments(argparse.ArgumentParser())
            legacy=parser.parse_args(['--config',str(path)])
            resolve_args(legacy)
            self.assertEqual(legacy.object_target_policy,'footprint_v1')
            self.assertEqual(legacy.geometry_aux_weight,0.)
            modern=parser.parse_args(['--config',str(path),'--object-target-policy','small_object_support_v1','--geometry-aux-weight','.2'])
            resolve_args(modern)
            self.assertEqual(modern.geometry_aux_weight,.2)
            invalid=parser.parse_args(['--config',str(path),'--object-target-policy','small_object_support_v1','--detection-loss-policy','mvt_l1_v1'])
            with self.assertRaisesRegex(ValueError,'LAV'):resolve_args(invalid)

if __name__=='__main__':unittest.main()
