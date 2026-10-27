import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from . import test_highway_traffic
from ..scenarios.full_loop_parts import highway_flow as flow, highway_traffic as traffic
from ..scenarios.full_loop_parts.route_profile import load_global_path
from ..utils.transform_utils import interpolate_on_polyline


class HighwayFlowTests(unittest.TestCase):
    context = test_highway_traffic.HighwayTrafficTests.context

    @classmethod
    def setUpClass(cls):
        test_highway_traffic.HighwayTrafficTests.setUpClass.__func__(cls)

    def flowing_context(self, start=True):
        context = self.context()
        context.cfg['highway_npc']['managed_traffic']['flowing_traffic']['enabled'] = True
        context.route_points = load_global_path(context)[1975:3639]
        traffic.prepare(context)
        if start:
            traffic.activate(context, context.highway_flow['departure_ego_s'], elapsed=10.)
        return context

    def observations(self, context, offsets, ego_offset=0., moving=6):
        f = context.highway_flow
        states = []
        for index, (item, offset) in enumerate(zip(
                [i for i in context.highway_npcs if i['merge_adjacent']], offsets)):
            x, y, z, _ = interpolate_on_polyline(f['points'], f['merge_lane_s'] + offset)
            states.append(NS(actor_info=NS(id=NS(value=item['actor_id'])),
                transform=NS(location=NS(x=x, y=y, z=z)),
                velocity=NS(x=34. if index < moving else 0., y=0., z=0.)))
        context.grpc.get_all_vehicle_states.return_value = states
        x, y, _, _ = interpolate_on_polyline(f['points'], f['merge_lane_s'] + ego_offset)
        return dict(x=x, y=y, speed_mps=10., collision_objects=[])

    def test_upstream_traffic_waits_for_approach_curve_and_starts_once(self):
        context = self.flowing_context(start=False)
        threshold = context.highway_flow['departure_ego_s']
        self.assertGreater(threshold, 60.)
        self.assertLess(threshold, 90.)
        self.assertFalse(context.highway_spawn_points_active)
        context.grpc.set_vehicle_velocity.assert_not_called()
        self.assertFalse(traffic.activate(context, threshold - .01))
        context.grpc.set_vehicle_velocity.assert_not_called()
        self.assertTrue(traffic.activate(context, threshold, elapsed=10.))
        self.assertTrue(context.highway_spawn_points_active)
        self.assertEqual(context.grpc.set_vehicle_velocity.call_count, 25)
        self.assertTrue(traffic.activate(context, threshold + 1., elapsed=11.))
        self.assertEqual(context.grpc.set_vehicle_velocity.call_count, 25)
        adjacent = [p for p in context.highway_npcs if p['merge_adjacent']]
        self.assertEqual(len({p['model'] for p in adjacent}), 6)
        self.assertEqual([p['offset_m'] for p in adjacent], [-25, -5, 15, 35, 55, 75])
        self.assertEqual(context.highway_flow['slot']['offset_m'], -25)
        self.assertEqual(context.highway_flow['slot']['route'][0], 'A2256W000415')
        self.assertIsNone(context.highway_merge_sync)

    def test_departure_point_must_be_on_current_route_and_before_merge(self):
        for offset in (-1., 10000.):
            context = self.context()
            cfg = context.cfg['highway_npc']['managed_traffic']['flowing_traffic']
            cfg.update(enabled=True, departure_distance_from_link_start_m=offset)
            context.route_points = load_global_path(context)[1975:3639]
            with self.assertRaisesRegex(ValueError, 'departure point'):
                traffic.prepare(context)
            context.grpc.spawn_vehicle.assert_not_called()

    def test_recycle_preserves_population_and_model_identity(self):
        context = self.flowing_context()
        models = {p['model'] for p in context.highway_npcs}
        speeds = {p['model']: p['speed_kmh'] for p in context.highway_npcs}
        ego = self.observations(context, [130, 170, 210, 250, 290, 330], ego_offset=-150.)
        ego.update(x=-100., y=350.)
        flow.update(context, 10., ego, 100.)
        context.grpc.destroy_vehicle_by_id.assert_called_once()
        self.assertEqual(len(context.scenario_actors), 25)
        self.assertEqual({p['model'] for p in context.highway_npcs}, models)
        self.assertEqual({p['model']: p['speed_kmh'] for p in context.highway_npcs}, speeds)
        recycled = next(p for p in context.highway_npcs if p.get('recycle_generation') == 1)
        context.grpc.set_vehicle_velocity.assert_called_with(recycled['actor'], recycled['speed_kmh'])
        context.grpc.set_vehicle_speed_limit.assert_called_with(recycled['actor'], recycled['speed_kmh'] / 3.6, enabled=True)
        indices = context.grpc.set_vehicle_route.call_args.kwargs['route_waypoint_indices']
        self.assertEqual(indices[recycled['route'][-1]],
                         len(self.loader.get_link_points(recycled['route'][-1])) - 1)
        self.assertEqual(context.highway_flow['result']['recycle_count'], 1)

    def test_no_recycling_near_ego_or_without_complete_states(self):
        context = self.flowing_context()
        ego = self.observations(context, [130, 140, 150, 160, 170, 180], ego_offset=150.)
        flow.update(context, 10., ego, 100.)
        context.grpc.destroy_vehicle_by_id.assert_not_called()
        context.grpc.get_all_vehicle_states.return_value = []
        flow.update(context, 11., ego, 100.)
        context.grpc.destroy_vehicle_by_id.assert_not_called()
        self.assertEqual(context.highway_flow['result']['status'], 'unverified_state_missing')

    def test_valid_merge_requires_moving_traffic_and_collision_data(self):
        for moving, collisions, expected in (
                (6, [], 'passed_observed_merge'),
                (3, [], 'invalid_test_insufficient_moving_traffic'),
                (6, ['other'], 'failed_collision'),
                (6, None, 'unverified_collision_data_missing')):
            context = self.flowing_context()
            f = context.highway_flow
            ego = self.observations(context, [-60, -20, 20, 60, 150, 190], moving=moving)
            if collisions is None:
                del ego['collision_objects']
            else:
                ego['collision_objects'] = collisions
            flow.update(context, 20., ego, f['merge_ego_s'])
            x, y, _, _ = interpolate_on_polyline(f['points'], f['merge_lane_s'] + 40.)
            ego.update(x=x, y=y)
            flow.update(context, 24., ego, f['merge_ego_s'] + 40.)
            self.assertEqual(f['result']['status'], expected)
            report = json.loads((Path(context.global_cfg['paths']['runtime_root']) /
                                 'highway_merge_evaluation.json').read_text())
            self.assertEqual(report['status'], expected)
