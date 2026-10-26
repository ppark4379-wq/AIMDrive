"""Dense traffic stays unchanged; sparse uses the historical native spawns."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..main_runner import build_parser, load_configuration
from ..scenarios.full_loop_parts import highway, highway_traffic
from ..tools import run_gt_udp_full_route as udp
from ..utils.highway_density import apply_highway_density


class HighwayDensityTests(unittest.TestCase):
    def setUp(self):
        _, _, self.config = load_configuration()

    def test_default_and_dense_preserve_current_configuration(self):
        for density in (None, 'dense'):
            selected = copy.deepcopy(self.config)
            apply_highway_density(selected, density)
            self.assertEqual(selected, self.config)
            managed = selected['highway_npc']['managed_traffic']
            self.assertTrue(managed['enabled'])
            self.assertEqual(managed['downstream_lead_count'], 7)
            self.assertEqual(managed['downstream_lead_speed_kmh'], 80.)

    def test_sparse_changes_only_npcs_and_restores_native_spawn_contract(self):
        selected = copy.deepcopy(self.config)
        apply_highway_density(selected, 'sparse')
        self.assertEqual({k: v for k, v in selected.items() if k != 'highway_npc'},
                         {k: v for k, v in self.config.items() if k != 'highway_npc'})
        npc = selected['highway_npc']
        self.assertFalse(highway_traffic.enabled(SimpleNamespace(cfg=selected)))
        self.assertEqual(npc['collection_maximum_total_vehicles'], 12)
        self.assertEqual(len(npc['spawn_points']), 3)
        for point, link, speed in zip(npc['spawn_points'],
                                     ('A2256W000029', 'A2256W000406', 'A2256W000175'),
                                     (60., 40., 50.)):
            self.assertEqual(point['start_link']['id'], link)
            self.assertEqual(point['maximum_spawn_vehicle'], 4)
            self.assertEqual(point['vehicle_length_types'], [2])
            self.assertFalse(point['is_lane_change'])
            self.assertEqual(point['min_spawn_period_sec'], 5.)
            self.assertEqual(point['max_spawn_period_sec'], 5.)
            self.assertEqual(point['min_desired_velocity_custom_kmh'], speed)
            self.assertEqual(point['max_desired_velocity_custom_kmh'], speed)
        self.assertEqual(npc['spawn_points'][2]['end_link'],
                         {'id': 'A2256W000434', 'point_idx': 300})

    def test_sparse_dispatches_to_three_native_spawn_points(self):
        apply_highway_density(self.config, 'sparse')
        grpc = Mock()
        grpc.create_vehicle_spawn_point.side_effect = ['10', '6', '8']
        context = SimpleNamespace(cfg=self.config, grpc=grpc,
                                  vehicle_models=lambda: ['DEFAULT_SEDAN'],
                                  highway_spawn_points=[])
        with patch.dict(os.environ, {}, clear=True), patch.object(highway.time, 'sleep'):
            highway.spawn_highway_npcs(context)
        self.assertEqual(grpc.create_vehicle_spawn_point.call_count, 3)
        grpc.spawn_vehicle.assert_not_called()
        self.assertFalse(context.highway_spawn_points_active)

    def test_main_parser_accepts_both_choices_and_rejects_unknown(self):
        parser = build_parser()
        self.assertIsNone(parser.parse_args([]).highway_density)
        for density in ('dense', 'sparse'):
            args = parser.parse_args(['--scenario', 'full_loop_highway',
                                      '--highway-density', density])
            self.assertEqual(args.highway_density, density)
        with self.assertRaises(SystemExit):
            parser.parse_args(['--highway-density', 'unknown'])

    def test_udp_uses_same_density_without_mutating_base(self):
        baseline = copy.deepcopy(self.config)
        for density in ('dense', 'sparse'):
            args = SimpleNamespace(highway_density=density, scenario='full_loop_highway',
                                   speed_cap_kmh=60., stop_before_tunnel=False, seed=42)
            result = udp.configure_scenario(self.config, args)
            expected = copy.deepcopy(self.config)
            apply_highway_density(expected, density)
            self.assertEqual(result['highway_npc'], expected['highway_npc'])
            self.assertEqual(self.config, baseline)
        with self.assertRaises(SystemExit):
            udp.main(['--scenario', 'full_loop', '--highway-density', 'sparse'])
