"""Causal input, split isolation and original masked multitask loss checks."""
import argparse
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import torch
from datasets.morai_gps_speed import causal_gps_speed, parse_native_gps
from datasets.morai_sensor_record_dataset import (
    child, digest, PathResolver, load_publications, validate_navigation_target,
    verify_admission_recovery, verify_publication_source,
)
from utils.morai_transfer_objective import multitask_loss
from utils.morai_transfer_training import (
    add_arguments, long_stationary_duplicate_indices, object_selection_score,
    resolve_args, selected_training_indices,
)

class TransferTests(unittest.TestCase):
    def test_manifest_campaign_discovers_direct_dataset_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publication = (root/'campaign'/'_campaign'/'validations'/'validation_1'/
                           'training_dataset_v1'/'manifest_index.json')
            publication.parent.mkdir(parents=True)
            publication.write_text('{}')
            checkpoint = root/'checkpoint.pth.tar'
            checkpoint.write_bytes(b'x')
            config = root/'config.json'
            config.write_text('{}')
            parser = add_arguments(argparse.ArgumentParser())
            args = parser.parse_args([
                '--config', str(config), '--manifest-campaign', str(root/'campaign'),
                '--checkpoint', str(checkpoint), '--device', 'cpu',
            ])
            resolve_args(args)
            self.assertEqual(args.manifest_index, [publication.resolve()])

    def test_publication_accepts_strict_and_fail_closed_subset_schemas(self):
        strict = 'morai_sensor_record_campaign_validation_v1'
        subset = 'morai_sensor_record_campaign_validation_subset_v1'
        verify_publication_source(
            {'source_schema': strict},
            {'schema': strict, 'valid': True},
            {'schema': strict, 'status': 'passed'},
        )
        subset_report = {
            'schema': subset,
            'valid': True,
            'selected_laps': 2,
            'laps': [{}, {}],
            'excluded_laps': [{}],
            'subset_coverage_preserved': True,
            'subset_policy': (
                'whole_lap_fail_closed_exclusion_with_full_variation_preservation_v1'),
        }
        verify_publication_source(
            {'source_schema': subset}, subset_report,
            {'schema': subset, 'status': 'passed'},
        )
        subset_report['subset_coverage_preserved'] = False
        with self.assertRaisesRegex(ValueError, 'fail-closed'):
            verify_publication_source(
                {'source_schema': subset}, subset_report,
                {'schema': subset, 'status': 'passed'},
            )

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

    def test_long_stop_sampling_thins_only_continuous_segment_interiors(self):
        def row(native, frame, speed, light=0):
            return (Path(native), {
                'sensor_time_ns': frame * 100_000_000,
                'ego_pose': {'speed': speed},
                'traffic_light': light,
                'object_valid': True,
                'object_positive': False,
            }, None, 0.)
        # Seven seconds at 10 Hz is thinned.  A separate three-second episode
        # is retained in full and cannot be joined across the episode boundary.
        rows = [row('/run/a', i, 0.) for i in range(71)]
        rows += [row('/run/b', i, 0., -1) for i in range(31)]
        rows += [row('/run/b', 31, 2., -1)]
        removed, summary = long_stationary_duplicate_indices(rows)
        kept = set(range(len(rows))) - removed
        self.assertEqual(summary['long_stop_segments'], 1)
        self.assertEqual(summary['long_stop_episodes'], 1)
        self.assertGreater(summary['long_stop_removed_frames'], 0)
        self.assertEqual(summary['long_stop_traffic_light_counts'][0], 71)
        self.assertTrue(set(range(71, len(rows))).issubset(kept))
        # Entry and release boundaries of the long stop are always retained.
        self.assertTrue(set(range(0, 11)).issubset(kept))
        self.assertTrue(set(range(60, 71)).issubset(kept))

    def test_object_balance_runs_after_long_stop_thinning(self):
        def row(frame, speed, positive):
            return (Path('/run/a'), {
                'sensor_time_ns': frame * 100_000_000,
                'ego_pose': {'speed': speed}, 'traffic_light': 0,
                'object_valid': True, 'object_positive': positive,
            }, None, 0.)
        class Dataset:
            rows = [row(i, 0., i == 0) for i in range(71)]
            rows += [row(71, 2., True), row(72, 2., False)]
            def __len__(self): return len(self.rows)
        args=type('Args',(),dict(
            frame_sampling='object_balanced', seed=42,
            object_negative_ratio=1., diagnostic_limit=0,
            long_stop_sampling='thin', long_stop_speed_threshold=.2,
            long_stop_min_seconds=5., long_stop_edge_seconds=1.,
            long_stop_stride_seconds=1.))()
        indices,summary=selected_training_indices(Dataset(),args)
        self.assertEqual(len(indices),4)
        self.assertEqual(summary['object_positive_frames'],2)
        self.assertEqual(summary['selected_verified_empty_frames'],2)
        self.assertGreater(summary['long_stop_removed_frames'],0)

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
        from utils.morai_transfer_training import add_arguments,resolve_args
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
        module='datasets.morai_sensor_record_dataset'
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
