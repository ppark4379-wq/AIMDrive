"""Causal input, split isolation and original masked multitask loss checks."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import torch
from data.src.loaders.morai_gps_speed import causal_gps_speed, parse_native_gps
from data.src.loaders.morai_sensor_record_dataset import (
    child, digest, PathResolver, load_publications, validate_navigation_target,
    verify_admission_recovery, verify_subset_selection,
)
from training.vision_encoder.losses.morai_transfer import multitask_loss
from training.vision_encoder.recipes.transfer import object_selection_score, selected_training_indices, is_static_avoidance, classify_situation, photometric_jitter

class TransferTests(unittest.TestCase):
    def test_subset_loader_accepts_only_admitted_whole_laps(self):
        index={'train':[{'lap':1}],'val':[{'lap':2}]}
        report={
            'subset_policy':'whole_lap_fail_closed_exclusion_with_full_variation_preservation_v1',
            'subset_coverage_preserved':True,'all_captured_laps_valid':False,
            'selected_laps':2,
            'laps':[{'lap':1,'valid':True,'training_usable':True},
                    {'lap':2,'valid':True,'training_usable':True}],
            'excluded_laps':[{'lap':3,'valid':False,'capture_status':'succeeded'}],
        }
        verify_subset_selection(index,report)
        capture_only=dict(report,all_captured_laps_valid=True,
                          excluded_laps=[{'lap':3,'valid':False,'capture_status':'failed'}])
        verify_subset_selection(index,capture_only)
        with_test=dict(capture_only,selected_laps=3,
                       laps=report['laps']+[{'lap':4,'valid':True,'training_usable':True}])
        verify_subset_selection(dict(index,test=[{'lap':4}]),with_test)
        with self.assertRaisesRegex(ValueError,'Subset validation'):
            verify_subset_selection(index,dict(report,all_captured_laps_valid=True))
        with self.assertRaisesRegex(ValueError,'Subset validation'):
            verify_subset_selection({'train':[{'lap':1},{'lap':3}], 'val':[{'lap':2}]},report)
        with self.assertRaisesRegex(ValueError,'Subset validation'):
            verify_subset_selection(index,dict(report,excluded_laps=[{'lap':3,'valid':True}]))

    def test_object_checkpoint_selection_prefers_ap_then_f1(self):
        lower_ap = {'detection_cell_average_precision': .20,
                    'detection_cell_best_f1': .90}
        higher_ap = {'detection_cell_average_precision': .21,
                     'detection_cell_best_f1': .10}
        tied_ap_higher_f1 = {'detection_cell_average_precision': .21,
                             'detection_cell_best_f1': .20}
        self.assertLess(object_selection_score(lower_ap), object_selection_score(higher_ap))
        self.assertLess(object_selection_score(higher_ap), object_selection_score(tied_ap_higher_f1))

    def test_loader_verifies_additive_admission_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';episode=run/'episode_000'
            original=episode/'admission_v1';corrected=episode/'admission_portable_v2'
            for folder,report in (
                (original,{'raw_manifest_sha256':'raw'}),
                (corrected,{
                    'raw_manifest_sha256':'raw','capture_contract_valid':True,
                    'runtime_streams_valid':True,'sensor_capture_status':'admitted',
                    'teacher_transport_status':'admitted','source_timestamps_modified':False,
                    'training_visible':False,
                }),
            ):
                folder.mkdir(parents=True)
                (folder/'report.json').write_text(json.dumps(report))
                (folder/'frames.json').write_text('[]')
                (folder/'COMPLETE.json').write_text(json.dumps({'files':{
                    'report.json':digest(folder/'report.json'),
                    'frames.json':digest(folder/'frames.json'),
                }}))
            recovery={
                'schema':'morai_sensor_record_admission_recovery_v1',
                'status':'completed','recovery_is_additive':True,
                'source_timestamps_modified':False,'training_visible':False,
                'original_admission':{'sha256':digest(original/'report.json')},
                'corrected_admission':{
                    'path':'episode_000/admission_portable_v2/report.json',
                    'sha256':digest(corrected/'report.json'),
                    'seal_sha256':digest(corrected/'COMPLETE.json'),
                },
            }
            (run/'ADMISSION_RECOVERY.json').write_text(json.dumps(recovery))
            evidence={
                'admission_recovery':digest(run/'ADMISSION_RECOVERY.json'),
                'effective_admission':digest(corrected/'COMPLETE.json'),
            }
            self.assertEqual(verify_admission_recovery(episode,evidence),corrected)
            recovery['corrected_admission']['sha256']='bad'
            (run/'ADMISSION_RECOVERY.json').write_text(json.dumps(recovery))
            evidence['admission_recovery']=digest(run/'ADMISSION_RECOVERY.json')
            with self.assertRaisesRegex(ValueError,'corrected admission'):
                verify_admission_recovery(episode,evidence)

    def test_object_balanced_sampling_uses_positive_and_verified_empty_rows(self):
        class Dataset:
            rows = [(None, {'object_valid': True, 'object_positive': True}, None, 0.),
                    (None, {'object_valid': True, 'object_positive': False}, None, 0.),
                    (None, {'object_valid': True, 'object_positive': False}, None, 0.),
                    (None, {'object_valid': False, 'object_positive': False}, None, 0.)]
            def __len__(self): return len(self.rows)
        args=type('Args',(),dict(frame_sampling='object_balanced',seed=42,
                                 object_negative_ratio=1.,diagnostic_limit=0))()
        indices,summary=selected_training_indices(Dataset(),args)
        self.assertEqual(len(indices),2)
        self.assertIn(0,indices)
        self.assertNotIn(3,indices)
        self.assertEqual(summary['object_positive_frames'],1)
        self.assertEqual(summary['selected_verified_empty_frames'],1)

    def test_static_avoidance_oversamples_only_route_leaving_static_frames(self):
        import math
        def entry(lateral, obstacle_y=8.):
            # Ego faces +y, so world +x is the ego's right.
            return dict(ego_pose=dict(x=0., y=0., yaw=math.pi/2), navigation=dict(target_x=0., target_y=25.),
                        waypoints=[[0., -2.], [0., -4.], [0., -6.], [0., -8.], [lateral, -10.]],
                        objects=[dict(x=.5, y=obstacle_y)])
        static='/data/static_obstacle__lap/lap_002_train_manifest.json'
        self.assertTrue(is_static_avoidance(entry(1.2), static))
        self.assertFalse(is_static_avoidance(entry(.1), static))
        self.assertFalse(is_static_avoidance(entry(1.2, obstacle_y=40.), static))
        self.assertFalse(is_static_avoidance(entry(1.2), '/data/roundabout__lap/lap_002_train_manifest.json'))
        class Dataset:
            rows=[(None,entry(1.2),None,0.),(None,entry(.1),None,0.),(None,entry(.1),None,0.)]
            row_manifest_paths=[static]*3
            def __len__(self): return len(self.rows)
        args=type('Args',(),dict(frame_sampling='static_avoidance',seed=42,avoidance_repeat=3,
                                 avoidance_other_frames=1,diagnostic_limit=0))()
        indices,summary=selected_training_indices(Dataset(),args)
        self.assertEqual(indices.count(0),3)
        self.assertEqual(len(indices),4)
        self.assertEqual((summary['avoidance_frames'],summary['other_frames']),(1,1))

    def test_photometric_jitter_is_shared_by_cameras_and_keeps_shapes(self):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        torch.manual_seed(0)
        inputs = {'rgb_front': torch.randn(4, 3, 8, 8), 'rgb_left': torch.randn(4, 3, 4, 4),
                  'lidar': torch.randn(4, 10, 4), 'target_point': torch.zeros(4, 2)}
        same = photometric_jitter(inputs, torch.Generator().manual_seed(1), 0.0, mean, std)
        self.assertIs(same, inputs)
        out = photometric_jitter(inputs, torch.Generator().manual_seed(1), 1.0, mean, std)
        self.assertEqual(out['rgb_front'].shape, inputs['rgb_front'].shape)
        self.assertIs(out['lidar'], inputs['lidar'])
        self.assertFalse(torch.equal(out['rgb_front'], inputs['rgb_front']))
        m = torch.tensor(mean).view(1, 3, 1, 1); sd = torch.tensor(std).view(1, 3, 1, 1)
        for key in ('rgb_front', 'rgb_left'):
            image = out[key] * sd + m
            self.assertGreaterEqual(float(image.min()), -1e-5)
            self.assertLessEqual(float(image.max()), 1 + 1e-5)
        # Deterministic for a seed.
        again = photometric_jitter(inputs, torch.Generator().manual_seed(1), 1.0, mean, std)
        self.assertTrue(torch.equal(out['rgb_left'], again['rgb_left']))

    def test_situation_weighted_repeats_hard_frames_and_thins_cruise(self):
        import math
        def entry(waypoints, obstacle_y=None):
            # Ego faces +y, so world +x is the ego's right; route target straight ahead.
            return dict(ego_pose=dict(x=0., y=0., yaw=math.pi/2), navigation=dict(target_x=0., target_y=25.),
                        waypoints=waypoints, objects=[] if obstacle_y is None else [dict(x=.5, y=obstacle_y)])
        cruise=[[0., -2.*k] for k in range(1, 6)]
        rows={'cruise': entry(cruise), 'curve': entry([[0., -2.], [.3, -4.], [.8, -6.], [1.6, -8.], [2.6, -10.]]),
              'decel': entry([[0., -2.], [0., -3.], [0., -3.5], [0., -3.8], [0., -4.]]),
              'hold_object': entry([[0., 0.]]*5, obstacle_y=6.), 'launch': entry([[0., -.2], [0., -.8], [0., -1.6], [0., -2.6], [0., -4.]]),
              'avoid': entry(cruise[:4]+[[1.2, -10.]], obstacle_y=8.), 'object_ahead': entry(cruise, obstacle_y=15.)}
        for name, e in rows.items():
            self.assertEqual(classify_situation(e), name)
        names=list(rows)+['cruise']*3
        class Dataset:
            rows=[(None,rows[n],None,0.) for n in names]
            def __len__(self): return len(self.rows)
        args=type('Args',(),dict(frame_sampling='situation_weighted',seed=42,situation_hard_repeat=2,
                                 situation_curve_fraction=1.,situation_cruise_fraction=.5,diagnostic_limit=0))()
        indices,summary=selected_training_indices(Dataset(),args)
        self.assertEqual(indices.count(names.index('decel')),2)
        self.assertEqual(indices.count(names.index('curve')),1)
        self.assertEqual(sum(names[i]=='cruise' for i in indices),2)
        self.assertEqual(summary['situation_frames']['cruise'],4)

    def test_static_object_sampling_repeats_rotated_frames_and_keeps_replay(self):
        import math
        def entry(heading_deg, positive=True):
            return dict(ego_pose=dict(x=0., y=0., yaw=math.pi/2), object_valid=True, object_positive=positive,
                        objects=[dict(x=0., y=10., heading_deg=heading_deg)])
        # (manifest path, entry): ego axis is 90 deg, so heading 90/0/180 are aligned and 60 is rotated 30 deg.
        listing = [('static_obstacle/a', entry(90.)), ('static_obstacle/b', entry(60.)),
                ('static_obstacle/c', entry(0., positive=False)), ('highway/h', entry(90.)),
                ('highway/i', entry(90.)), ('roundabout/r', entry(90.))]
        class Dataset:
            rows = [(None, e, None, 0.) for _, e in listing]
            row_manifest_paths = [p for p, _ in listing]
            def __len__(self): return len(self.rows)
        args = type('Args', (), dict(frame_sampling='static_object', seed=42, avoidance_repeat=3,
                                     rotated_extra_repeat=2, avoidance_other_frames=2, diagnostic_limit=0))()
        indices, summary = selected_training_indices(Dataset(), args)
        self.assertEqual(indices.count(0), 3)            # aligned static frame
        self.assertEqual(indices.count(1), 3 + 2)        # rotated static frame
        replay = [i for i in indices if i not in (0, 1)]
        self.assertEqual(len(replay), 2)
        self.assertTrue(set(replay) <= {2, 3, 4, 5})
        self.assertEqual(summary['rotated_frames'], 1)

    def test_partial_scopes_train_only_selected_groups(self):
        from training.vision_encoder.setup import configure_trainable_scope, train_mode
        import torch.nn as nn
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.rgb_backbone = nn.Linear(2, 2); self.lidar_backbone = nn.Linear(2, 2)
                self.decoder = nn.Linear(2, 2); self.traffic_pred_head = nn.Linear(2, 2)
                self.traffic_light_pred_head = nn.Linear(2, 2); self.stop_sign_head = nn.Linear(2, 2)
        for scope, expected in (('lidar_head', {'lidar_backbone', 'traffic_pred_head'}),
                                ('decoder_head', {'lidar_backbone', 'decoder', 'traffic_pred_head'})):
            model = Tiny()
            configure_trainable_scope(model, scope)
            trainable = {n.split('.')[0] for n, p in model.named_parameters() if p.requires_grad}
            self.assertEqual(trainable, expected)
            model.train(); train_mode(model, scope)
            self.assertFalse(model.training)

    def test_distillation_is_zero_on_reference_and_ignores_geometry_channels(self):
        import torch
        from training.vision_encoder.losses.morai_distill import distillation_loss
        det = torch.randn(2, 2500, 8); light = torch.randn(2, 2)
        reference = (det, None, light)
        loss, _ = distillation_loss((det.clone(), None, light.clone()), reference)
        self.assertAlmostEqual(float(loss), 0., places=6)
        moved = det.clone(); moved[..., 1:7] += 5.      # geometry may change freely
        loss, _ = distillation_loss((moved, None, light.clone()), reference)
        self.assertAlmostEqual(float(loss), 0., places=6)
        moved = det.clone(); moved[..., 0] += 1.        # confidence logits may not
        loss, terms = distillation_loss((moved, None, light.clone()), reference, torch.tensor([True, False]))
        self.assertAlmostEqual(float(terms['distill_logit']), 1., places=5)

    def test_geometry_distillation_only_on_anchored_frames_weighted_by_reference(self):
        import torch
        from training.vision_encoder.losses.morai_distill import distillation_loss
        det = torch.zeros(2, 2500, 8); det[..., 0] = -20.; det[:, 7, 0] = 20.   # reference sees one object
        light = torch.zeros(2, 2)
        moved = det.clone(); moved[:, 7, 1:7] += 1.; moved[:, 9, 1:7] += 9.    # cell 9 is background
        _, terms = distillation_loss((moved, None, light), (det, None, light), None, torch.tensor([True, False]))
        self.assertAlmostEqual(float(terms['distill_geometry']), 1., places=4)
        _, terms = distillation_loss((moved, None, light), (det, None, light), None, torch.tensor([False, False]))
        self.assertEqual(float(terms['distill_geometry']), 0.)
        _, terms = distillation_loss((moved, None, light), (det, None, light))
        self.assertNotIn('distill_geometry', terms)

    def test_speed_frames_hold_speed_where_logits_are_freed(self):
        import torch
        from training.vision_encoder.losses.morai_distill import distillation_loss
        det = torch.zeros(2, 2500, 8); det[..., 0] = -20.; det[:, 7, 0] = 20.   # reference detects one object
        light = torch.zeros(2, 2)
        moved = det.clone(); moved[:, 7, 7] += .5                              # speed of that object drifts
        freed = torch.tensor([False, True])                                     # frame 0 holds a skipped (new) model
        _, terms = distillation_loss((moved, None, light), (det, None, light), freed)
        self.assertAlmostEqual(float(terms['distill_speed']), .5, places=4)     # legacy: only frame 1 holds speed
        _, terms = distillation_loss((moved, None, light), (det, None, light), freed,
                                     speed_frames=torch.tensor([True, True]))
        self.assertAlmostEqual(float(terms['distill_speed']), .5, places=4)     # both frames now hold it
        moved[0, 7, 7] += 1.                                                     # drift only on the freed frame
        _, legacy = distillation_loss((moved, None, light), (det, None, light), freed)
        _, held = distillation_loss((moved, None, light), (det, None, light), freed,
                                    speed_frames=torch.tensor([True, True]))
        self.assertAlmostEqual(float(legacy['distill_speed']), .5, places=4)    # invisible to the legacy mask
        self.assertAlmostEqual(float(held['distill_speed']), 1., places=4)      # (1.5 + .5) / 2

    def test_negative_speed_penalty_only_where_confident(self):
        import torch
        from training.vision_encoder.losses.morai_distill import distillation_loss
        det = torch.zeros(1, 2500, 8); det[..., 0] = -30.; det[0, 3, 0] = 30.
        light = torch.zeros(1, 2)
        student = det.clone(); student[0, 3, 7] = -.25; student[0, 9, 7] = -5.  # cell 9 is not confident
        loss0, terms0 = distillation_loss((student, None, light), (student, None, light))
        self.assertNotIn('distill_negative_speed', terms0)
        loss1, terms1 = distillation_loss((student, None, light), (student, None, light), negative_speed_weight=2.)
        self.assertAlmostEqual(float(terms1['distill_negative_speed']), .25, places=3)
        self.assertAlmostEqual(float(loss1 - loss0), .5, places=3)
        student[0, 3, 7].requires_grad_(False)
        grad_input = student.clone().requires_grad_(True)
        loss, _ = distillation_loss((grad_input, None, light), (det, None, light), negative_speed_weight=1.)
        loss.backward()
        self.assertLess(float(grad_input.grad[0, 3, 7]), 0.)                    # pushes the negative speed up

    def test_geometry_weight_scales_only_the_geometry_term(self):
        import torch
        from training.vision_encoder.losses.morai_distill import distillation_loss
        det = torch.zeros(1, 2500, 8); det[..., 0] = -20.; det[0, 7, 0] = 20.
        light = torch.zeros(1, 2)
        moved = det.clone(); moved[0, 7, 1:7] += 1.
        loss1, terms1 = distillation_loss((moved, None, light), (det, None, light), None, torch.tensor([True]))
        loss10, terms10 = distillation_loss((moved, None, light), (det, None, light), None, torch.tensor([True]),
                                            geometry_weight=10.)
        self.assertAlmostEqual(float(terms1['distill_geometry']), float(terms10['distill_geometry']), places=6)
        self.assertAlmostEqual(float(loss10), 10. * float(loss1), places=3)

    def test_geometry_anchor_skips_static_scenes_and_listed_paths(self):
        from training.vision_encoder.recipes.transfer import GeometryAnchorDataset
        class Base:
            row_manifest_paths = ['/x/heavy_vehicle_bbox_20261015/training_map/lap_001_train_manifest.json',
                                  '/x/static_obstacle__static_shapes/lap_002.json',
                                  '/x/dynamic_obstacle__lap100/lap_003.json']
            def __len__(self): return 3
            def __getitem__(self, i): return {}, {}
        self.assertEqual(GeometryAnchorDataset(Base()).anchor, [True, False, True])
        self.assertEqual(GeometryAnchorDataset(Base(), ('heavy_vehicle_bbox',)).anchor, [False, False, True])

    def test_distill_mask_frees_frames_with_skipped_models_anywhere(self):
        from training.vision_encoder.recipes.transfer import DistillMaskDataset, contains_model
        truck = {'model_name': '2013_Hyundai_Xcient(P520)', 'x': -12., 'y': 0., 'geometry_valid': False}
        car = {'model_name': '2014_Kia_K7', 'x': 10., 'y': 0., 'geometry_valid': True}
        names = {'2013_Hyundai_Xcient(P520)'}
        self.assertTrue(contains_model({'objects': [car, truck]}, names))   # behind ego, invalid geometry: still freed
        self.assertFalse(contains_model({'objects': [car]}, names))
        self.assertFalse(contains_model({'objects': None}, names))
        class Base:
            def __len__(self): return 2
            def __getitem__(self, i): return i, {'object_valid': torch.tensor(1.)}
        data = DistillMaskDataset(Base(), [True, False])
        self.assertTrue(bool(data[0][1]['distill_keep'])); self.assertFalse(bool(data[1][1]['distill_keep']))
        self.assertEqual(data[1][0], 1)
        with self.assertRaises(ValueError):
            DistillMaskDataset(Base(), [True])

    def test_geometry_anchor_marks_non_static_scenes(self):
        from training.vision_encoder.recipes.transfer import GeometryAnchorDataset
        class Base:
            row_manifest_paths = ['x/static_obstacle__a/m.json', 'x/highway__b/m.json', 'x/tunnel_static__c/m.json']
            def __len__(self): return 3
            def __getitem__(self, i): return {}, {}
        ds = GeometryAnchorDataset(Base())
        self.assertEqual([bool(ds[i][1]['geometry_anchor']) for i in range(3)], [False, True, False])

    def test_gps_speed_is_causal_interval(self):
        a=(1000000000,37.,127.)
        self.assertEqual(causal_gps_speed(a,(1100000000,37.,127.)),0)
        self.assertAlmostEqual(causal_gps_speed(a,(1100000000,37.00001,127.)),11.11949,places=3)
        for b in [(1000000000,37.,127.),(900000000,37.,127.),(1300000000,37.,127.),(1100000000,float('nan'),127.),(1100000000,40.,127.)]:
            with self.assertRaises(ValueError): causal_gps_speed(a,b)
        self.assertEqual(parse_native_gps('Latitude : 37\nLongitude : 127\nAltitude : 4'),(37.,127.))

    def test_no_path_escape_and_longest_prefix(self):
        for name in ('../escape','/absolute'):
            with self.assertRaises(ValueError): child('/tmp/transfer',name)
        resolver=PathResolver([('/old','/tmp/first'),('/old/data','/tmp/second')])
        self.assertEqual(resolver('/old/data/a'),Path('/tmp/second/a'))
        self.assertEqual(resolver('/older/a'),Path('/older/a'))

    def test_requested_new_campaign_cannot_be_omitted(self):
        import argparse,json
        from training.vision_encoder.recipes.transfer import add_arguments,resolve_args
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            config.write_text(json.dumps({'manifest_indices':['/old/index.json'],
                'manifest_campaigns':[str(Path(tmp)/'pending_campaign')],
                'checkpoint':'/weights/base.pth.tar'}))
            parser=argparse.ArgumentParser();add_arguments(parser)
            args=parser.parse_args(['--config',str(config)])
            with self.assertRaisesRegex(ValueError,'Additional campaign is not ready'):
                resolve_args(args)
            # Explicit CLI selection replaces local configuration for diagnostics.
            args=parser.parse_args(['--config',str(config),'--manifest-index','/explicit/index.json'])
            resolve_args(args)
            self.assertEqual(args.manifest_index,[Path('/explicit/index.json')])

    def test_loss_mask_has_no_unknown_gt_gradient(self):
        det=torch.randn(2,2500,8,requires_grad=True)
        wp=torch.randn(2,5,2,requires_grad=True)
        light=torch.randn(2,2,requires_grad=True)
        target=torch.zeros_like(det); target[0,100,0]=1
        labels=dict(detection=target,object_valid=torch.tensor([True,False]),
                    waypoint_valid=torch.tensor([True,True]),waypoints=torch.zeros_like(wp),
                    traffic_light=torch.tensor([0,-1]))
        loss,_=multitask_loss((det,wp,light),labels,detection_loss_policy='lmdrive_lav_v1')
        loss.backward()
        self.assertEqual(det.grad[1].abs().sum().item(),0)
        self.assertEqual(light.grad[1].abs().sum().item(),0)
        self.assertGreater(det.grad[0].abs().sum().item(),0)
        self.assertGreater(wp.grad[1].abs().sum().item(),0)

    def test_merge_preserves_holdout_and_rejects_duplicate_scene(self):
        roles={r:r for r in ('camera_1','camera_2','camera_3','camera_4','lidar')}
        def doc(episode,tag):
            return {'episode':episode,'entries':[{'asset_roles':roles,'assets':{r:tag+r for r in roles}}]}
        rows1={'train':[(Path('/t1'),doc('/episode1','a'))],'val':[(Path('/v1'),doc('/episode2','b'))]}
        rows2={'train':[(Path('/t2'),doc('/episode3','c'))],'val':[(Path('/v2'),doc('/episode4','d'))]}
        module='data.src.loaders.morai_sensor_record_dataset'
        with patch(module+'.digest',side_effect=lambda p:str(p)), patch(module+'.load_publication',side_effect=[({},rows1),({},rows2)]):
            _,rows=load_publications(['/i1','/i2'],PathResolver())
            self.assertEqual(len(rows['train']),2);self.assertEqual(len(rows['val']),2)
        rows2['train'][0][1]['entries']=rows1['val'][0][1]['entries']
        with patch(module+'.digest',side_effect=lambda p:str(p)), patch(module+'.load_publication',side_effect=[({},rows1),({},rows2)]):
            with self.assertRaisesRegex(ValueError,'bundle duplicated'):load_publications(['/i1','/i2'],PathResolver())

    def test_navigation_target_allows_only_proven_closed_loop_wrap(self):
        nav = {
            'route_target_valid': True,
            'target_source': 'official_route_lookahead',
            'route_index': 4404,
            'target_route_index': 26,
            'target_x': 1.0,
            'target_y': 2.0,
            'distance_to_route_m': 0.2,
        }
        closed = {'route_topology': 'closed_loop', 'route_point_count': 4406}
        validate_navigation_target(nav, closed)
        for contract in ({}, {'route_topology': 'open', 'route_point_count': 4406}):
            with self.assertRaisesRegex(ValueError, 'Invalid official navigation target'):
                validate_navigation_target(nav, contract)
        with self.assertRaisesRegex(ValueError, 'Invalid official navigation target'):
            validate_navigation_target(dict(nav, route_index=100, target_route_index=90), closed)
        with self.assertRaisesRegex(ValueError, 'Invalid official navigation target'):
            validate_navigation_target(dict(nav, target_route_index=4406), closed)

if __name__=='__main__':unittest.main()
