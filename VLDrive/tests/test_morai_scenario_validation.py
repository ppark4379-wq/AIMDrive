import argparse
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from training.vision_encoder.utils.morai_scenario_validation import selection_decision, SCENARIOS, scenario_groups
from training.vision_encoder.utils.morai_transfer_metrics import evaluate
from utils.evaluation.morai_object_metrics import ObjectMetricsAccumulator
from training.vision_encoder.recipes.transfer import add_arguments, resolve_args, evaluate_and_checkpoint, preflight_manifest_subset, retain_checkpoint, save


def report():
    row=dict(frames=100,detection_observed_frames=99,detection_cell_average_precision=.7,
             detection_cell_recall=.6,box_center_l2_m=.4,box_length_mae_m=.3,box_width_mae_m=.2,
             ade_m=.3,fde_m=.6,traffic_light_accuracy=.99,
             object_detection={'policy':{'fixed':True},'gt_boxes':30,
                 'thresholds':{k:dict(recall=.5,precision=.8) for k in ('0.3','0.5')},
                 'false_positive_boxes_per_empty_frame':.01})
    return dict(per_scenario={name:copy.deepcopy(row) for name in SCENARIOS.values()},
                macro_detection_cell_average_precision=.7,macro_detection_cell_recall=.6)


class GuardedSelectionTests(unittest.TestCase):
    def test_baseline_and_unchanged_are_eligible(self):
        b=report();self.assertTrue(selection_decision(b,b)['eligible'])
    def test_average_improvement_cannot_hide_dynamic_regression(self):
        b=report();c=copy.deepcopy(b);c['macro_detection_cell_average_precision']=.8
        c['per_scenario']['dynamic_obstacle']['detection_cell_recall']=.55
        d=selection_decision(c,b);self.assertFalse(d['eligible']);self.assertEqual(d['violations'][0]['scenario'],'dynamic_obstacle')
    def test_bbox_regression_and_extra_false_objects_block_selection(self):
        for change in ('size','precision','false_positive'):
            b=report();c=copy.deepcopy(b);r=c['per_scenario']['static_obstacle']
            if change=='size':r['box_width_mae_m']=.5
            if change=='precision':r['object_detection']['thresholds']['0.5']['precision']=.6
            if change=='false_positive':r['object_detection']['false_positive_boxes_per_empty_frame']=.04
            self.assertFalse(selection_decision(c,b)['eligible'])
    def test_missing_candidate_metric_is_not_improvement(self):
        b=report();c=copy.deepcopy(b);c['per_scenario']['highway']['ade_m']=None
        self.assertFalse(selection_decision(c,b)['eligible'])
        b['per_scenario']['highway']['traffic_light_accuracy']=None
        c=copy.deepcopy(b);self.assertTrue(selection_decision(c,b)['eligible'])
    def test_removing_all_false_objects_from_empty_scene_is_improvement(self):
        b=report()
        row=b['per_scenario']['basic_driving']['object_detection']
        row['gt_boxes']=0
        for metric in row['thresholds'].values():metric.update(precision=0.,recall=None)
        c=copy.deepcopy(b)
        row=c['per_scenario']['basic_driving']['object_detection']
        for metric in row['thresholds'].values():metric['precision']=None
        row['false_positive_boxes_per_empty_frame']=0.
        self.assertTrue(selection_decision(c,b)['eligible'])
    def test_different_coverage_or_object_policy_is_rejected(self):
        for key in ('frames','policy'):
            b=report();c=copy.deepcopy(b)
            if key=='frames':c['per_scenario']['dynamic_obstacle']['frames']=99
            else:c['per_scenario']['dynamic_obstacle']['object_detection']['policy']={}
            with self.assertRaises(ValueError):selection_decision(c,b)
    def test_legacy_float32_baseline_cannot_select_float64_candidate(self):
        baseline = report()
        policy = ObjectMetricsAccumulator().compute()['policy']
        for row in baseline['per_scenario'].values():
            row['object_detection']['policy'] = copy.deepcopy(policy)
        candidate = copy.deepcopy(baseline)
        candidate['macro_detection_cell_average_precision'] = .9
        for row in baseline['per_scenario'].values():
            row['object_detection']['policy']['schema'] = 'morai_oriented_bev_object_diagnostic_v1'
            row['object_detection']['policy'].pop('intersection_implementation')
        with self.assertRaisesRegex(ValueError, 'identical object diagnostic policy'):
            selection_decision(candidate, baseline)

    def test_recomputed_float64_anchor_is_comparable_without_relaxing_guards(self):
        baseline = report()
        policy = ObjectMetricsAccumulator().compute()['policy']
        for row in baseline['per_scenario'].values():
            row['object_detection']['policy'] = copy.deepcopy(policy)
        candidate = copy.deepcopy(baseline)
        self.assertTrue(selection_decision(candidate, baseline)['eligible'])
        candidate['per_scenario']['dynamic_obstacle']['object_detection']['thresholds']['0.5']['recall'] = .4
        decision = selection_decision(candidate, baseline)
        self.assertFalse(decision['eligible'])
        self.assertTrue(any(row['metric'] == 'object_recall_iou_0.5' for row in decision['violations']))

    def test_scenarios_bind_publication_metadata_not_filename(self):
        ds=type('DS',(),{'row_manifest_paths':[str(i) for i in range(5)]})()
        mapping={str(i):name for i,name in enumerate(SCENARIOS)}
        self.assertEqual(len(scenario_groups(ds,mapping)),5)
        mapping['0']='unknown'
        with self.assertRaises(ValueError):scenario_groups(ds,mapping)
    def test_eval_failure_still_preserves_checkpoint_for_custom_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'weights.pt'
            def broken(*a,**kw):
                self.assertTrue(p.exists());raise RuntimeError('grouped eval failure')
            with self.assertRaises(RuntimeError):
                evaluate_and_checkpoint(p,{'steps':3,'state_dict':{}},None,None,'cpu',evaluation_fn=broken)
            self.assertFalse(torch.load(p,weights_only=True)['validation_completed'])
    def test_retained_epoch_survives_later_last_checkpoint_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            last,epoch=Path(tmp)/'last.pt',Path(tmp)/'epoch.pt'
            save(last,{'steps':1});retain_checkpoint(last,epoch)
            save(last,{'steps':2})
            self.assertEqual(torch.load(epoch,weights_only=True)['steps'],1)
            self.assertEqual(torch.load(last,weights_only=True)['steps'],2)
    def test_bounded_preflight_preserves_one_intact_episode_per_publication_split(self):
        def doc(objects,light):return {'entries':[dict(object_positive=objects,traffic_light=light)]}
        rows=[(Path('/pub/a/1.json'),doc(False,-1)),(Path('/pub/a/2.json'),doc(True,0)),
              (Path('/pub/b/1.json'),doc(True,1))]
        chosen=preflight_manifest_subset({'train':rows,'val':rows})
        self.assertEqual(len(chosen['train']),2)
        self.assertIs(chosen['train'][0],rows[1])
        self.assertIs(chosen['val'][1],rows[2])
    def test_guarded_cli_forbids_skip_and_partial_validation(self):
        for extra in (['--skip-baseline'],['--diagnostic-limit','5']):
            args=add_arguments(argparse.ArgumentParser()).parse_args([
                '--manifest-index','/tmp/index.json','--checkpoint','/tmp/model.pt',
                '--checkpoint-selection','scenario_guarded']+extra)
            with self.assertRaisesRegex(ValueError,'full baseline'):resolve_args(args)


class ObjectMetadataEvaluationTests(unittest.TestCase):
    def test_binding_disabled_and_wrong_frame_fail(self):
        det=torch.zeros(2500,8);det[:,0]=-10
        target=dict(detection=det,object_valid=torch.tensor(True),waypoint_valid=torch.tensor(True),
                    waypoints=torch.zeros(5,2),traffic_light=torch.tensor(-1),
                    frame_id=torch.tensor(1),sensor_time_ns=torch.tensor(123))
        class Model:
            def eval(self):pass
            def __call__(self,x):return det.unsqueeze(0),torch.zeros(1,5,2),torch.zeros(1,2)
        loader=DataLoader([({},target)],batch_size=1)
        entry=dict(frame_id=1,sensor_time_ns=123,object_valid=True,objects=[],ego_pose={'x':0.,'y':0.,'yaw':0.})
        r=evaluate(Model(),loader,'cpu',detection_loss_policy='lmdrive_lav_v1',object_entries=[entry])
        self.assertEqual(r['object_detection']['empty_frames'],1)
        with self.assertRaisesRegex(ValueError,'enabled'):
            evaluate(Model(),loader,'cpu',include_objects=False,object_entries=[entry])
        with self.assertRaisesRegex(ValueError,'frame/time'):
            evaluate(Model(),loader,'cpu',detection_loss_policy='lmdrive_lav_v1',object_entries=[dict(entry,frame_id=2)])

if __name__=='__main__':unittest.main()
