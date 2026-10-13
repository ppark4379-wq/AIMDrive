"""A successful creation can still return an unusable SpawnPoint runtime ID."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from scenario_runner.scenarios.full_loop_parts import highway


def context(disable_results, create_ids):
    grpc = Mock()
    grpc.create_vehicle_spawn_point.side_effect = create_ids
    grpc.enable_vehicle_spawn_point.side_effect = disable_results
    return NS(cfg=dict(highway_npc=dict(model='car', collection_maximum_total_vehicles=4,
        spawn_points=[dict(id=6, name='competition_spawn_6', maximum_spawn_vehicle=4)],
        spawn_point_lifecycle=dict(create_attempts=3, create_settle_sec=0., recreate_delay_sec=0.))),
        vehicle_models=lambda: ['car'], highway_spawn_points=[], grpc=grpc)


class SpawnStartupTests(unittest.TestCase):
    def test_replaces_created_id_that_rejects_initial_disable(self):
        c = context([False, True], ['7', '13'])
        highway.spawn_highway_npcs(c)
        self.assertTrue(c.highway_spawn_points[0]['retired'])
        self.assertEqual(c.highway_spawn_points[1]['id'], '13')
        self.assertEqual(c.highway_npc_planned_count, 1)
        self.assertFalse(c.highway_spawn_points_active)
        self.assertTrue(all(call.args[0]['pause'] is True
                            for call in c.grpc.create_vehicle_spawn_point.call_args_list))
        self.assertTrue(all(call.args[1] is False
                            for call in c.grpc.enable_vehicle_spawn_point.call_args_list))
        c.grpc.enable_vehicle_spawn_point.reset_mock()
        c.grpc.enable_vehicle_spawn_point.side_effect = None
        c.grpc.enable_vehicle_spawn_point.return_value = True
        self.assertEqual(highway._set_highway_spawn_points_enabled(c, True), 1)
        self.assertEqual(c.grpc.enable_vehicle_spawn_point.call_args.args, ('13', True))

    def test_never_continues_after_exhausted_disable_recovery(self):
        c = context([False, False, False], ['7', '8', '9'])
        with self.assertRaisesRegex(RuntimeError, 'confirm disabled'):
            highway.spawn_highway_npcs(c)
        self.assertEqual(c.grpc.create_vehicle_spawn_point.call_count, 3)
        self.assertTrue(all(p['retired'] for p in c.highway_spawn_points))
        self.assertTrue(all(call.args[1] is False
                            for call in c.grpc.enable_vehicle_spawn_point.call_args_list))

    def test_creation_failure_and_healthy_point_use_same_bounded_budget(self):
        c = context([True], [None, '13'])
        highway.spawn_highway_npcs(c)
        self.assertEqual(c.highway_npc_planned_count, 1)
        self.assertEqual(len(c.highway_spawn_points), 1)


if __name__ == '__main__':
    unittest.main()
