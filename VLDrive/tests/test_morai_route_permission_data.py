"""Protected-turn targets stay separate from color and never enter inputs."""
import copy
import unittest

import torch
from torch.utils.data import DataLoader

from data.src.loaders.morai_route_permission import (
    PHASE_PERMISSIONS, SCHEMA, RoutePermissionDataset, permission_target,
)


def entry(raw=33, movement='left'):
    permissions = PHASE_PERMISSIONS[raw]
    target = permissions[movement]
    return dict(frame_id=7, sensor_time_ns=1_000_000_000,
                traffic_light={1: 0, 16: 1}.get(raw, -1),
                route_permission=dict(schema=SCHEMA, valid=True, target=target,
                    raw_signal_bits=raw, light_id='C1256W000052', signal_movement=movement,
                    source_record_ids=[20, 22], source_bracket_ns=[990_000_000, 1_020_000_000],
                    separate_from_color_head=True),
                traffic_signal=dict(schema=SCHEMA, permission_valid=True,
                    raw_signal_bits=raw, light_id='C1256W000052', signal_movement=movement,
                    signal_permission=target, straight_permitted=permissions['straight'],
                    left_permitted=permissions['left'], distance_to_stopline_m=12.5,
                    signal_route_link_id='A2256W000083', signal_stop_node_id='A1256W000742',
                    light_source_record_ids=[20, 22], light_source_bracket_ns=[990_000_000, 1_020_000_000]))


class FakeAdmittedDataset:
    def __init__(self, entries):
        self.rows = [('native', item, None, 0.) for item in entries]
        self.row_manifest_paths = ['manifest'] * len(entries)
        self.samples = [(dict(rgb_front=torch.zeros(3, 2, 2), velocity=torch.zeros(1)),
                         dict(frame_id=torch.tensor(item['frame_id'], dtype=torch.int64),
                              sensor_time_ns=torch.tensor(item['sensor_time_ns'], dtype=torch.int64),
                              traffic_light=torch.tensor(item['traffic_light'], dtype=torch.int64)))
                        for item in entries]
        self.reads = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.reads += 1
        return self.samples[index]


class PermissionParserTests(unittest.TestCase):
    def test_red_left_arrow_is_permitted_despite_unknown_color_label(self):
        row = entry(33)
        before = copy.deepcopy(row)
        result = permission_target(row)
        self.assertEqual(row['traffic_light'], -1)
        self.assertEqual((result['target'], result['valid'], result['movement_index']), (1, True, 1))
        self.assertEqual(result['stopline_distance_m'], 12.5)
        self.assertEqual(result['provenance']['frame_id'], 7)
        self.assertEqual(row, before)

    def test_straight_green_is_denied_for_left_and_allowed_for_straight(self):
        self.assertEqual(permission_target(entry(16, 'left'))['target'], 0)
        self.assertEqual(permission_target(entry(16, 'straight'))['target'], 1)

    def test_every_supported_phase_is_checked_by_movement(self):
        for raw, permissions in PHASE_PERMISSIONS.items():
            for movement, permitted in permissions.items():
                result = permission_target(entry(raw, movement))
                self.assertEqual(result['target'], int(permitted))

    def test_missing_false_or_nonboolean_mask_never_falls_back_to_color(self):
        for color in (-1, 0, 1):
            row = entry()
            row['traffic_light'] = color
            del row['route_permission']
            self.assertEqual(permission_target(row)['target'], -1)
        for mask in (None, False, 1, 'true'):
            row = entry()
            row['route_permission']['valid'] = mask
            result = permission_target(row)
            self.assertEqual(result['target'], -1)
            self.assertFalse(result['valid'])
        self.assertEqual(permission_target({})['movement_index'], -1)

    def test_claimed_valid_contradictions_raise(self):
        mutations = [
            ('route_permission', 'target', False),
            ('route_permission', 'target', 1),
            ('route_permission', 'schema', 'legacy'),
            ('route_permission', 'separate_from_color_head', False),
            ('route_permission', 'raw_signal_bits', 49),
            ('route_permission', 'raw_signal_bits', True),
            ('route_permission', 'light_id', 'another light'),
            ('route_permission', 'signal_movement', 'right'),
            ('route_permission', 'signal_movement', ['left']),
            ('route_permission', 'source_record_ids', [10, 22]),
            ('route_permission', 'source_bracket_ns', [990_000_000, 1_001_000_000]),
            ('traffic_signal', 'permission_valid', False),
            ('traffic_signal', 'signal_movement', 'straight'),
            ('traffic_signal', 'signal_permission', False),
            ('traffic_signal', 'straight_permitted', True),
            ('traffic_signal', 'left_permitted', False),
            ('traffic_signal', 'signal_route_link_id', None),
            ('traffic_signal', 'signal_stop_node_id', ''),
            ('traffic_signal', 'distance_to_stopline_m', float('nan')),
            ('traffic_signal', 'distance_to_stopline_m', -1.),
        ]
        for where, key, value in mutations:
            row = entry()
            row[where][key] = value
            with self.subTest(where=where, key=key, value=value):
                with self.assertRaises(ValueError):
                    permission_target(row)

    def test_source_bracket_must_contain_the_exact_sensor_time(self):
        row = entry()
        row['sensor_time_ns'] = 2_000_000_000
        with self.assertRaisesRegex(ValueError, 'sensor time'):
            permission_target(row)
        row = entry()
        row['frame_id'] = True
        with self.assertRaisesRegex(ValueError, 'frame identity'):
            permission_target(row)


class PermissionDatasetTests(unittest.TestCase):
    def test_rows_and_masks_retained_without_reading_sensors_during_wrapping(self):
        invalid = entry(16)
        invalid['frame_id'] = 8
        del invalid['route_permission']
        base = FakeAdmittedDataset([entry(), invalid])
        dataset = RoutePermissionDataset(base, {'manifest': 'full_loop_traffic_light'})
        self.assertEqual(base.reads, 0)
        self.assertEqual(len(dataset), 2)
        self.assertIs(dataset.rows, base.rows)
        self.assertEqual([row['frame_id'] for row in dataset.metadata], [7, 8])
        self.assertEqual(dataset.metadata[0]['scenario_name'], 'full_loop_traffic_light')
        inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(inputs['route_movement'].tolist(), [1, 1])
        self.assertEqual(targets['route_permission'].tolist(), [1, -1])
        self.assertEqual(targets['route_permission_valid'].tolist(), [True, False])

    def test_only_route_movement_is_added_to_inputs_and_originals_unchanged(self):
        base = FakeAdmittedDataset([entry()])
        dataset = RoutePermissionDataset(base)
        inputs, targets = dataset[0]
        self.assertEqual(set(inputs) - set(base.samples[0][0]), {'route_movement'})
        self.assertEqual(set(targets) - set(base.samples[0][1]), {'route_permission', 'route_permission_valid'})
        self.assertNotIn('route_movement', base.samples[0][0])
        self.assertEqual(int(targets['traffic_light']), -1)
        for key in ('raw_signal_bits', 'route_permission', 'light_id', 'stopline_distance_m', 'frame_id'):
            self.assertNotIn(key, inputs)

    def test_report_metadata_cannot_mutate_training_target(self):
        dataset = RoutePermissionDataset(FakeAdmittedDataset([entry()]))
        dataset.metadata[0]['target'] = 0
        self.assertEqual(int(dataset[0][1]['route_permission']), 1)

    def test_sample_identity_mismatch_and_field_overwrite_are_rejected(self):
        base = FakeAdmittedDataset([entry()])
        dataset = RoutePermissionDataset(base)
        base.samples[0][1]['sensor_time_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            dataset[0]
        base = FakeAdmittedDataset([entry()])
        dataset = RoutePermissionDataset(base)
        base.samples[0][0]['route_movement'] = torch.tensor(0)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            dataset[0]

    def test_alignment_and_missing_scenario_are_rejected(self):
        base = FakeAdmittedDataset([entry()])
        with self.assertRaisesRegex(ValueError, 'scenario'):
            RoutePermissionDataset(base, {})
        base.row_manifest_paths = []
        with self.assertRaisesRegex(ValueError, 'alignment'):
            RoutePermissionDataset(base)


if __name__ == '__main__':
    unittest.main()
