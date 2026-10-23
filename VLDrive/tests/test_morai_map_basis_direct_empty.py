"""The direct-record branch must prove empty actors instead of inventing poses."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.src.loaders.morai_map_derivation import (
    correct_entry, load_map_derivation_inputs, verify_map_derivation)
from data.src.loaders.morai_sensor_record_dataset import PathResolver, digest, read_json
from data.src.processing.correct_morai_map_basis import main


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(folder, **metadata):
    value = dict(metadata, files={p.name: digest(p) for p in folder.iterdir()
                                 if p.is_file() and p.name != 'COMPLETE.json'})
    write(folder/'COMPLETE.json', value)


class DirectEmptyMapBasisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.resolver = PathResolver()
        self.source_folder = self.root/'source'/'manifests'
        self.source_folder.mkdir(parents=True)
        self.episode = self.root/'capture'/'lap1'/'episode_000'
        self.source = self.record(self.episode, 1)
        self.source_path = self.source_folder/'lap1.json'
        write(self.source_path, self.source)

    def record(self, episode, lap):
        write(episode/'raw/raw_manifest.json', {'lap': lap})
        native = episode.parent
        native.mkdir(exist_ok=True)
        gps = native/'GPS.txt'
        gps.write_text('eastOffset: 302595\nnorthOffset: 4124145\n')
        entries, timelines, geometries, signals = [], [], [], []
        for i in range(2):
            stamp = 1000+i
            permission = dict(valid=True, target=bool(i), raw_signal_bits=33 if i else 1,
                              signal_movement='left', separate_from_color_head=True)
            entries.append(dict(frame_id=i, sensor_time_ns=stamp,
                ego_pose=dict(x=10., y=0., yaw=1.5, speed=0.),
                waypoints=[[0., -float(j)] for j in range(1, 6)],
                objects=[], object_valid=True, object_positive=False,
                traffic_light=-1 if i else 0, route_permission=permission,
                traffic_signal=dict(permission_valid=True, signal_permission=bool(i)),
                navigation=dict(target_x=20., target_y=30.),
                asset_roles=dict(gps='GPS.txt', camera_1='a', camera_2='b',
                                 camera_3='c', camera_4='d', lidar='l'),
                assets={
                    'GPS.txt': digest(gps), **{k: '%d-%s' % (lap, k) for k in 'abcdl'}},
                asset_sizes={'GPS.txt': gps.stat().st_size}))
            timelines.append(dict(index=i, lidar_source_ns=stamp, object_gt=dict(
                objects=[], source='grpc_full_inventory', complete_inventory=True,
                transport_timing_candidate=True)))
            geometries.append(dict(index=i, lidar_source_ns=stamp, objects=[],
                                   object_count=0, api_box_geometry_candidate=True))
            signals.append(dict(index=i, sensor_source_ns=stamp,
                                training_label=-1 if i else 0, heading_error_deg=0.))
        write(episode/'gt_timeline_v1/frames.json', timelines)
        seal(episode/'gt_timeline_v1', raw_manifest_sha256=digest(episode/'raw/raw_manifest.json'))
        write(episode/'object_geometry_audit_v1/frames.json', geometries)
        write(episode/'object_geometry_audit_v1/report.json', dict(
            schema='morai_sensor_record_object_geometry_audit_v1', structural_valid=True,
            complete_observed_geometry_scope=True,
            raw_manifest_sha256=digest(episode/'raw/raw_manifest.json')))
        seal(episode/'object_geometry_audit_v1',
             schema='morai_sensor_record_object_geometry_audit_v1',
             gt_timeline_seal_sha256=digest(episode/'gt_timeline_v1/COMPLETE.json'))
        write(episode/'route_signal_audit_v1/frames.json', signals)
        seal(episode/'route_signal_audit_v1')
        return dict(entries=entries,
            episode='../../capture/lap%d/episode_000' % lap,
            native_root='../../capture/lap%d' % lap, route='../../capture/lap%d/episode_000' % lap,
            evidence_seal_sha256=dict(
                gt_timeline=digest(episode/'gt_timeline_v1/COMPLETE.json'),
                object_geometry=digest(episode/'object_geometry_audit_v1/COMPLETE.json'),
                route_signal=digest(episode/'route_signal_audit_v1/COMPLETE.json')),
            navigation_contract=dict(route_source_resolution=dict(recorded_sha256=dict(node_set='node-hash'))))

    def inputs(self):
        return load_map_derivation_inputs(self.source, self.resolver,
                                          self.source_folder, self.episode)

    def reseal(self):
        seal(self.episode/'gt_timeline_v1',
             raw_manifest_sha256=digest(self.episode/'raw/raw_manifest.json'))
        seal(self.episode/'object_geometry_audit_v1',
             schema='morai_sensor_record_object_geometry_audit_v1',
             gt_timeline_seal_sha256=digest(self.episode/'gt_timeline_v1/COMPLETE.json'))
        self.source['evidence_seal_sha256'].update(
            gt_timeline=digest(self.episode/'gt_timeline_v1/COMPLETE.json'),
            object_geometry=digest(self.episode/'object_geometry_audit_v1/COMPLETE.json'))

    def test_direct_empty_inputs_and_signal_fields_preserved(self):
        timeline, geometry, pivots, policy = self.inputs()
        self.assertEqual(pivots, {})
        self.assertEqual(policy, 'sealed_complete_empty_inventory_v1')
        basis = dict(schema='morai_projected_grid_basis_v1', crs='EPSG:32652',
            origin_xyz=[302595., 4124145., 0.], position_frame='projected_grid',
            source_orientation_frame='true_enu', orientation_frame='projected_grid')
        for before in self.source['entries']:
            i = before['frame_id']
            signals = read_json(self.episode/'route_signal_audit_v1/frames.json')
            after, _ = correct_entry(before, timeline[i], geometry[i], basis, signals[i], pivots)
            for key in ('route_permission', 'traffic_signal', 'sensor_time_ns', 'navigation'):
                self.assertEqual(after[key], before[key])

    def test_nonempty_unapproved_and_missing_selected_frames_rejected(self):
        for which, field, value in [
                ('entry', 'objects', [{'id': 'actor'}]),
                ('entry', 'object_valid', False),
                ('entry', 'object_positive', True),
                ('inventory', 'objects', [{'id': 'actor'}]),
                ('inventory', 'complete_inventory', False),
                ('inventory', 'transport_timing_candidate', False),
                ('geometry', 'objects', [{'id': 'actor'}]),
                ('geometry', 'api_box_geometry_candidate', False),
                ('geometry', 'lidar_source_ns', 999),
                ('timeline', 'index', 9)]:
            with self.subTest(which=which, field=field):
                saved = copy.deepcopy(self.source)
                tp, gp = (self.episode/'gt_timeline_v1/frames.json',
                          self.episode/'object_geometry_audit_v1/frames.json')
                t, g = read_json(tp), read_json(gp)
                original_t, original_g = copy.deepcopy(t), copy.deepcopy(g)
                target = {'entry': self.source['entries'][0], 'inventory': t[0]['object_gt'],
                          'geometry': g[0], 'timeline': t[0]}[which]
                target[field] = value
                write(tp, t); write(gp, g); self.reseal()
                with self.assertRaises(ValueError):
                    self.inputs()
                write(tp, original_t); write(gp, original_g)
                self.source = saved
                self.reseal()

    def test_tampered_seal_raw_binding_and_timeline_binding_rejected(self):
        timeline = self.episode/'gt_timeline_v1/frames.json'
        timeline.write_text(timeline.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'Changed sealed evidence'):
            self.inputs()
        self.reseal()
        report = self.episode/'object_geometry_audit_v1/report.json'
        value = read_json(report); value['raw_manifest_sha256'] = 'other-recording'
        write(report, value); self.reseal()
        with self.assertRaisesRegex(ValueError, 'empty-inventory evidence'):
            self.inputs()
        value['raw_manifest_sha256'] = digest(self.episode/'raw/raw_manifest.json')
        write(report, value); self.reseal()
        gp = self.episode/'object_geometry_audit_v1/COMPLETE.json'
        value = read_json(gp); value['gt_timeline_seal_sha256'] = 'other-timeline'
        write(gp, value)
        self.source['evidence_seal_sha256']['object_geometry'] = digest(gp)
        with self.assertRaisesRegex(ValueError, 'empty-inventory evidence'):
            self.inputs()

    def test_existing_fullpose_branch_keeps_pivot_evidence(self):
        geometry_dir = self.root/'fullpose'
        write(geometry_dir/'frames.json', [dict(index=0, objects=[{'id': 'actor'}])])
        write(geometry_dir/'pivot_evidence.json', {'instances': {'actor': {'valid': True}}})
        self.source['geometry_reprocessing'] = dict(
            timeline_directory=str(self.episode/'gt_timeline_v1'), geometry_directory=str(geometry_dir))
        with patch('data.src.loaders.morai_sensor_record_dataset.verify_geometry_derivation') as verify:
            _, geometry, pivots, policy = self.inputs()
        verify.assert_called_once_with(self.source, self.resolver, self.source_folder, self.episode)
        self.assertEqual(policy, 'sealed_fullpose_geometry_v1')
        self.assertEqual(geometry[0]['objects'], [{'id': 'actor'}])
        self.assertEqual(pivots, {'actor': {'valid': True}})

    def test_writer_missing_test_split_relocates_and_verifies_paths(self):
        second_episode = self.root/'capture/lap2/episode_000'
        second = self.record(second_episode, 2)
        report = self.root/'source/report.json'
        write(report, dict(valid=True, schema='morai_sensor_record_campaign_validation_v1'))
        seal(report.parent, status='passed')
        rows = []
        for lap, source in [(1, self.source), (2, second)]:
            source['campaign_validation_sha256'] = digest(report)
            p = self.source_folder/('lap%d.json' % lap); write(p, source)
            rows.append(dict(path=p.name, sha256=digest(p), frames=2, lap=lap))
        index = self.source_folder/'manifest_index.json'
        write(index, dict(schema='morai_visual_manifest_index_v1', training_ready=True,
            source_schema='morai_sensor_record_campaign_validation_v1', source_validation='../report.json',
            source_validation_sha256=digest(report), train=[rows[0]], val=[rows[1]],
            coverage=dict(train=dict(raw_frames=3, excluded_frames=1),
                          val=dict(raw_frames=3, excluded_frames=1))))
        seal(self.source_folder, status='passed', training_visible=True,
             schema='morai_sensor_record_training_publication_v1')
        config = self.root/'config.json'; write(config, {'manifest_indices': [str(index)]})
        metadata = self.root/'map.json'
        write(metadata, dict(global_coordinate_system='EPSG:32652',
            local_origin_in_global=[302595., 4124145., 0.], mgeo_file_hash="{'node_set.json': 'node-hash'}"))
        output = self.root/'derived'
        with patch('sys.argv', ['correct_map', '--config', str(config),
                               '--map-metadata', str(metadata), '--output', str(output)]):
            main()
        derived_folder = output/'manifests'
        result = read_json(derived_folder/'manifest_index.json')
        self.assertEqual(result['counts'], {'train': 2, 'val': 2})
        self.assertEqual(result['test'], [])
        self.assertEqual(result['coverage']['val']['route_permission_frames'], 2)
        self.assertEqual(result['coverage']['val']['route_permission_0'], 1)
        self.assertEqual(result['coverage']['val']['route_permission_1'], 1)
        self.assertEqual(result['coverage']['val']['excluded_frames'], 1)
        derived = read_json(derived_folder/'lap1.json')
        verify_map_derivation(derived, self.resolver, derived_folder, self.episode)
        for key in ('episode', 'native_root', 'route'):
            self.assertTrue(Path(derived[key]).is_absolute())
            tampered = copy.deepcopy(derived); tampered[key] = str(self.root/'redirected')
            with self.subTest(path=key), self.assertRaises(ValueError):
                verify_map_derivation(tampered, self.resolver, derived_folder, self.episode)
        self.assertEqual(digest(self.source_path), rows[0]['sha256'])


if __name__ == '__main__':
    unittest.main()
