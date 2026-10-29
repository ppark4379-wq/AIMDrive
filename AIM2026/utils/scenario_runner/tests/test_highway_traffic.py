"""Population, lane topology and model identity for explicit highway traffic."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ..main_runner import load_configuration
from ..utils.map_loader import MGeoMapLoader
from ..scenarios.full_loop_parts import highway, highway_traffic as traffic


class HighwayTrafficTests(unittest.TestCase):
    def test_bus_heavy_fleet_has_exact_counts_speeds_and_no_spawn_overlap(self):
        from ..utils.highway_density import apply_highway_density, apply_bus_heavy_fleet
        from ..scenarios.full_loop_parts.tunnel_static_parts.layout import boxes_overlap
        context = self.context()
        apply_highway_density(context.cfg, 'dense')
        apply_bus_heavy_fleet(context.cfg)
        for seed in [2385576734200624830, 6986037702628052809] + list(range(20)):
            context.run_seed = seed
            plans, models = traffic.build_plan(context)
            self.assertEqual(len(plans), 22)
            self.assertEqual(models.count('2016_Hyundai_Universe'), 8)
            self.assertEqual(models.count('2013_Hyundai_Xcient(P520)'), 2)
            self.assertEqual(len(set(m for m in models if m not in
                ('2016_Hyundai_Universe', '2013_Hyundai_Xcient(P520)'))), 12)
            self.assertEqual(sum(p.get('additional_bus', False) for p in plans), 6)
            self.assertEqual(sum(p['merge_adjacent'] and p['required_model']=='2016_Hyundai_Universe' for p in plans), 2)
            moved = [p for p in plans if p.get('relocation', {}).get('reason')=='upstream_to_downstream']
            self.assertEqual(len(moved), 5)
            self.assertTrue(all(p['offset_m']>=0 and -135<=p['yaw_deg']<=-45 for p in moved))
            self.assertTrue(all(p['relocation']['original']['offset_m']<=-80 for p in moved))
            self.assertTrue(all(20 <= p['speed_kmh'] <= 100 for p in plans))
            for i, plan in enumerate(plans):
                for other in plans[:i]:
                    self.assertFalse(boxes_overlap(dict(traffic._universe_box(plan), width_m=3.146),
                        dict(traffic._universe_box(other), width_m=3.146), .25))
        traffic.prepare(context)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, 22)
        self.assertEqual(sum(p['model']=='2016_Hyundai_Universe' for p in context.highway_npcs), 8)

    def test_three_extra_buses_preserve_base_population_and_safe_slots(self):
        from ..utils.highway_density import apply_highway_density
        from ..scenarios.full_loop_parts.tunnel_static_parts.layout import boxes_overlap
        context = self.context()
        apply_highway_density(context.cfg, 'dense')
        for seed in [2385576734200624830] + list(range(10)):
            context.run_seed = seed
            traffic_cfg = context.cfg['highway_npc']['managed_traffic']
            traffic_cfg['extra_buses'] = 0
            base, base_models = traffic.build_plan(context)
            traffic_cfg['extra_buses'] = 3
            context.cfg['highway_npc']['collection_maximum_total_vehicles'] = 23
            plans, models = traffic.build_plan(context)
            self.assertEqual(plans[:20], base)
            self.assertEqual(models[:20], base_models)
            self.assertEqual(len(plans), 23)
            self.assertEqual(models[20:], ['2016_Hyundai_Universe'] * 3)
            self.assertEqual([p['lane'] for p in plans[20:]], ['merge_adjacent', 'middle', 'left'])
            for i, plan in enumerate(plans):
                for other in plans[:i]:
                    self.assertFalse(boxes_overlap(dict(traffic._universe_box(plan), width_m=3.146),
                        dict(traffic._universe_box(other), width_m=3.146), .25))
        traffic.prepare(context)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, 23)
        self.assertEqual(len(context.highway_npcs), 23)

    def test_dense_twenty_and_large_vehicle_lane_guarantees(self):
        from ..utils.highway_density import apply_highway_density
        from ..scenarios.full_loop_parts.tunnel_static_parts.layout import boxes_overlap
        context = self.context()
        apply_highway_density(context.cfg, 'dense')
        large = {'2016_Hyundai_Universe', '2013_Hyundai_Xcient(P520)'}
        for seed in range(30):
            context.run_seed = seed
            plans, models = traffic.build_plan(context)
            self.assertEqual(len(plans), 20)
            self.assertEqual(models, [p['required_model'] for p in plans])
            self.assertEqual(sum(p['merge_adjacent'] and p['required_model'] in large for p in plans), 1)
            self.assertEqual(sum(not p['merge_adjacent'] and p['required_model'] in large for p in plans), 3)
            for index, plan in enumerate(plans):
                for other in plans[:index]:
                    self.assertFalse(boxes_overlap(
                        dict(traffic._universe_box(plan), width_m=3.146),
                        dict(traffic._universe_box(other), width_m=3.146), .25))
        traffic.prepare(context)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, 20)
        self.assertEqual(len(context.highway_npcs), 20)
        self.assertFalse(context.cfg['highway_merge_static_vehicle']['enabled'])

    def test_dense_fixed_models_and_failed_required_spawn(self):
        from ..utils.highway_density import apply_highway_density
        for model in ('2016_Hyundai_Universe', '2013_Hyundai_Xcient(P520)'):
            context = self.context()
            apply_highway_density(context.cfg, 'dense')
            context.cfg['highway_npc']['managed_traffic']['fixed_model'] = model
            traffic.prepare(context)
            self.assertEqual(len(context.highway_npcs), 20)
            self.assertEqual({p['model'] for p in context.highway_npcs}, {model})
        context = self.context()
        apply_highway_density(context.cfg, 'dense')
        context.grpc.spawn_vehicle.side_effect = lambda **kw: None
        with self.assertRaises(RuntimeError):
            traffic.prepare(context)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, 1)

    def test_xcient_only_population_and_collision_free_slots(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts.layout import boxes_overlap
        context = self.context()
        original, _ = traffic.build_plan(context)
        context.cfg['highway_npc']['managed_traffic']['fixed_model'] = '2013_Hyundai_Xcient(P520)'
        plans, models = traffic.build_plan(context)
        self.assertEqual([(p['lane'],p['speed_kmh']) for p in plans],
                         [(p['lane'],p['speed_kmh']) for p in original])
        self.assertEqual(models,['2013_Hyundai_Xcient(P520)']*25)
        traffic.prepare(context)
        self.assertEqual(context.grpc.spawn_vehicle.call_count,25)
        self.assertEqual({p['model'] for p in context.highway_npcs},{'2013_Hyundai_Xcient(P520)'})
        for seed in [5488920826684496123] + list(range(30)):
            context.run_seed=seed
            plans,_=traffic.build_plan(context)
            for i,plan in enumerate(plans):
                for other in plans[:i]:
                    self.assertFalse(boxes_overlap(traffic._xcient_box(plan),traffic._xcient_box(other),.25))

    def test_fixed_universe_preserves_population_speeds_and_spawns_every_slot(self):
        context = self.context()
        original, _ = traffic.build_plan(context)
        context.cfg['highway_npc']['managed_traffic']['fixed_model'] = '2016_Hyundai_Universe'
        plans, models = traffic.build_plan(context)
        self.assertEqual([(p['lane'], p['speed_kmh']) for p in plans],
                         [(p['lane'], p['speed_kmh']) for p in original])
        self.assertEqual(models, ['2016_Hyundai_Universe'] * len(plans))
        traffic.prepare(context)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, len(plans))
        self.assertEqual({p['model'] for p in context.highway_npcs}, {'2016_Hyundai_Universe'})
        calls = context.grpc.spawn_vehicle.call_args_list
        self.assertEqual(len({c.kwargs['request_id'] for c in calls}), len(plans))
        self.assertTrue(all(c.kwargs['model_name'] == '2016_Hyundai_Universe' for c in calls))
        self.assertTrue(all(c.kwargs['lookup_before_create'] is False for c in calls))

    def test_universe_fork_collision_regression_and_seed_variations(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts.layout import boxes_overlap
        context = self.context()
        context.run_seed = 13285474119529461107
        original, _ = traffic.build_plan(context)
        self.assertTrue(boxes_overlap(traffic._universe_box(original[12]),
                                      traffic._universe_box(original[2])))
        context.cfg['highway_npc']['managed_traffic']['fixed_model'] = '2016_Hyundai_Universe'
        for seed in [13285474119529461107] + list(range(30)):
            context.run_seed = seed
            plans, _ = traffic.build_plan(context)
            self.assertEqual(len(plans), 25)
            for i, plan in enumerate(plans):
                for other in plans[:i]:
                    self.assertFalse(boxes_overlap(traffic._universe_box(plan),
                                                  traffic._universe_box(other), .25))
                    if plan['lane'] == other['lane']:
                        gap = 20 if plan['merge_adjacent'] else 30
                        self.assertGreaterEqual(plan['route_distance_m']-other['route_distance_m'], gap)
                points = context.map_loader.get_link_points(plan['route'][0])
                actual = traffic.interpolate_on_polyline(points, plan['spawn_link_s_m'])
                self.assertEqual(actual, tuple(plan[k] for k in ('x', 'y', 'z', 'yaw_deg')))
                self.assertLess(plan['waypoint'], len(points))

    def test_fixed_model_never_falls_back_to_other_vehicle(self):
        context = self.context()
        context.cfg['highway_npc']['managed_traffic']['fixed_model'] = '2016_Hyundai_Universe'
        context.grpc.spawn_vehicle.side_effect = lambda **kw: None
        with self.assertRaisesRegex(RuntimeError, 'fixed_model'):
            traffic.prepare(context)
        self.assertEqual(context.grpc.spawn_vehicle.call_count, 1)

    def test_missing_fixed_model_fails_before_spawn(self):
        context = self.context()
        context.cfg['highway_npc']['managed_traffic']['fixed_model'] = 'missing_model'
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            traffic.prepare(context)
        context.grpc.spawn_vehicle.assert_not_called()

    @classmethod
    def setUpClass(cls):
        cls.global_cfg, _, cls.config = load_configuration()
        cls.loader = MGeoMapLoader(cls.global_cfg['paths']['mgeo_root'])

    def context(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = copy.deepcopy(self.config)
        config['highway_npc']['managed_traffic']['merge_sync']['enabled'] = False
        config['highway_npc']['managed_traffic']['flowing_traffic']['enabled'] = False
        groups = config['highway_npc']['managed_traffic']['model_groups']
        models = [model for values in groups.values() for model in values]
        grpc = Mock()
        def spawn(**kwargs):
            actor = Mock()
            actor.get_id.return_value = kwargs['label']
            return actor
        grpc.spawn_vehicle.side_effect = spawn
        return SimpleNamespace(cfg=config, map_loader=self.loader, grpc=grpc,
            vehicle_models=lambda: models, run_seed=7, run_instance_id='test',
            scenario_actors=[], highway_npcs=[], highway_spawn_points=[],
            highway_spawn_points_active=False, highway_activation_s=110.,
            highway_traffic_cleared=False, spawn_highway_merge_static_vehicle=Mock(),
            global_cfg={'paths': {'runtime_root': directory.name}})

    def synchronized_context(self):
        context = self.context()
        context.cfg['highway_npc']['managed_traffic']['randomize_adjacent_count'] = False
        context.cfg['highway_npc']['managed_traffic']['merge_sync']['enabled'] = True
        context.cfg['highway_npc']['managed_traffic']['lanes'][0]['offsets_m'] = [-160, -70, 35, 75, 115, 155]
        context.cfg['highway_npc']['managed_traffic']['lanes'][1]['offsets_m'] = [-460, -400, -340, -280, -130, -40, 55, 95, 135, 175]
        context.cfg['highway_npc']['managed_traffic']['lanes'][2]['offsets_m'] = [-430, -370, -310, -100, -10, 75, 115, 155, 195]
        from ..scenarios.full_loop_parts.route_profile import load_global_path
        context.route_points = load_global_path(context)[1975:3639]
        traffic.prepare(context)
        return context

    def test_merge_sync_waits_for_measured_approach_and_releases_once(self):
        context = self.synchronized_context()
        sync = context.highway_merge_sync
        self.assertGreater(sync['merge_ego_s_m'], 250)
        self.assertLess(sync['merge_ego_s_m'], 310)
        self.assertFalse(traffic.activate(context, 10., {'speed_mps': 10.}))
        self.assertFalse(traffic.activate(context, sync['merge_ego_s_m'] - 30., {'speed_mps': 0.}))
        self.assertFalse(traffic.activate(context, sync['merge_ego_s_m'] - 30.))
        speed = 10.
        npc_speed = speed * sync['npc_speed_ratio']
        eta = sync['lead_travel_m'] / npc_speed + sync['lead_pass_before_ego_s']
        release_s = sync['merge_ego_s_m'] - speed * eta + .01
        self.assertFalse(traffic.activate(context, release_s, {'speed_mps': 3.}))
        self.assertTrue(highway.activate_highway_spawn_points(context, 20., release_s, {'speed_mps': speed}))
        self.assertEqual(context.grpc.set_vehicle_velocity.call_count, 25)
        adjacent = [p for p in context.highway_npcs if p['merge_adjacent']]
        encounters = [p for p in adjacent if abs(p['predicted_pass_relative_to_ego_s']) < 10]
        self.assertGreaterEqual(len(encounters), 4)
        self.assertEqual(len({p['model'] for p in encounters}), len(encounters))
        self.assertTrue(all(p['speed_kmh'] == 32.4 for p in adjacent))
        self.assertTrue(traffic.activate(context, release_s + 10., {'speed_mps': 0.}))
        self.assertEqual(context.grpc.set_vehicle_velocity.call_count, 25)
        report = json.loads((Path(context.global_cfg['paths']['runtime_root']) / 'highway_traffic.json').read_text())
        self.assertEqual(report['merge_sync']['phase'], 'released')

    def test_merge_sync_requires_actual_route_entry_before_spawning(self):
        context = self.context()
        context.cfg['highway_npc']['managed_traffic']['merge_sync']['enabled'] = True
        context.route_points = [(0., 0., 0.), (1., 0., 0.)]
        with self.assertRaisesRegex(ValueError, 'no identifiable'):
            traffic.prepare(context)
        context.grpc.spawn_vehicle.assert_not_called()

    def test_population_exceeds_previous_total_and_adjacent_lane_is_real(self):
        context = self.context()
        plans, models = traffic.build_plan(context)
        self.assertEqual(len(plans), 25)
        self.assertGreaterEqual(len(plans), 13 + 2)
        adjacent = [item for item in plans if item['merge_adjacent']]
        self.assertEqual(len(adjacent), 6)
        self.assertEqual(len(set(models[:25])), 25)
        actual_left = self.loader.get_link('A2256W000411')['left_lane_change_dst_link_idx']
        self.assertEqual(actual_left, 'A2256W000409')
        self.assertTrue(all(actual_left in item['route'] for item in adjacent))
        self.assertTrue(all(item['speed_kmh'] > 0 for item in plans))

    def test_upstream_approach_is_staggered_and_ego_ramp_stays_clear(self):
        context = self.context()
        plans, _ = traffic.build_plan(context)
        upstream = sorted((p for p in plans if p['offset_m'] < 0),
                          key=lambda p: p['route_distance_m'])
        self.assertEqual(len(upstream), 14)
        self.assertEqual([p['offset_m'] for p in upstream], [-460, -430, -400, -370, -340, -310, -280, -250, -220, -130, -100, -55, -25, -5])
        self.assertEqual(upstream[0]['route'][0], 'A2256W000157')
        for first, second in zip(upstream, upstream[1:]):
            gap = 20 if first['merge_adjacent'] and second['merge_adjacent'] else 30
            self.assertGreaterEqual(second['route_distance_m'] - first['route_distance_m'], gap - 1e-6)
        for lane, start in (('merge_adjacent', 15), ('middle', 25), ('left', 5)):
            mainline = [p for p in plans if p['lane'] == lane and p['offset_m'] >= 0]
            count = 3 if lane == 'middle' else 4
            gap = 20 if lane == 'merge_adjacent' else 40
            self.assertEqual([p['offset_m'] for p in mainline], [start + gap * i for i in range(count)])
        forbidden = {'A2256W000054', 'A2256W000846', 'A2256W000411'}
        self.assertTrue(all(not forbidden.intersection(p['route']) for p in plans))

    def test_random_adjacent_population_is_reproducible_and_total_stays_25(self):
        context = self.context()
        observed = set()
        for seed in range(60):
            context.run_seed = seed
            plans, models = traffic.build_plan(context)
            self.assertEqual(plans, traffic.build_plan(context)[0])
            adjacent = [p for p in plans if p['merge_adjacent']]
            observed.add(len(adjacent))
            self.assertIn(len(adjacent), (4, 5, 6))
            self.assertEqual(len(plans), 25)
            self.assertEqual(len(set(models[:25])), 25)
            self.assertEqual([p['offset_m'] for p in adjacent], [-25 + 20*i for i in range(len(adjacent))])
            self.assertEqual(min(p['offset_m'] for p in plans), -460)
        self.assertEqual(observed, {4, 5, 6})

    def test_close_positions_on_distinct_fork_links_fail_before_spawning(self):
        context = self.context()
        lanes = context.cfg['highway_npc']['managed_traffic']['lanes']
        for lane in lanes:
            lane['offsets_m'] = [35, 70, 105, 140, 175, 210]
        with self.assertRaisesRegex(ValueError, 'physical separation'):
            traffic.prepare(context)
        context.grpc.spawn_vehicle.assert_not_called()

    def test_overlapping_shared_approach_slots_are_rejected(self):
        context = self.context()
        lanes = context.cfg['highway_npc']['managed_traffic']['lanes']
        lanes[2]['offsets_m'][0] = lanes[1]['offsets_m'][0]
        with self.assertRaisesRegex(ValueError, 'shared approach'):
            traffic.prepare(context)
        context.grpc.spawn_vehicle.assert_not_called()

    def test_ego_ramp_and_out_of_bounds_upstream_slots_are_rejected(self):
        for mutation in ('ramp', 'bounds', 'origin'):
            context = self.context()
            lane = context.cfg['highway_npc']['managed_traffic']['lanes'][0]
            if mutation == 'ramp':
                lane['route'].insert(0, 'A2256W000846')
            elif mutation == 'bounds':
                lane['offsets_m'][0] = -1000
            else:
                lane['offset_origin_link'] = 'missing'
            with self.assertRaises(ValueError):
                traffic.prepare(context)
            context.grpc.spawn_vehicle.assert_not_called()

    def test_per_vehicle_random_speeds_preserve_layout_and_repeat_with_seed(self):
        context = self.context()
        all_speeds = set()
        first = None
        for seed in range(30):
            context.run_seed = seed
            plans, models = traffic.build_plan(context)
            speeds = [p['speed_kmh'] for p in plans]
            self.assertEqual(speeds, [p['speed_kmh'] for p in traffic.build_plan(context)[0]])
            self.assertEqual(len(set(speeds)), 25)
            self.assertTrue(all(20 <= speed <= 100 for speed in speeds))
            all_speeds.update(speeds)
            if first is None:
                first = speeds
            else:
                self.assertNotEqual(first, speeds)
            saved = context.cfg['highway_npc']['managed_traffic'].pop('speed_range_kmh')
            baseline, baseline_models = traffic.build_plan(context)
            context.cfg['highway_npc']['managed_traffic']['speed_range_kmh'] = saved
            self.assertEqual(models, baseline_models)
            self.assertEqual([{k:v for k,v in p.items() if k != 'speed_kmh'} for p in plans],
                             [{k:v for k,v in p.items() if k != 'speed_kmh'} for p in baseline])
        self.assertEqual(all_speeds, set(range(20, 101)))

    def test_invalid_random_speed_range_fails_before_spawn(self):
        for bounds in ([19, 60], [20, 101], [60, 20], [20], [20, float('nan')]):
            context = self.context()
            context.cfg['highway_npc']['managed_traffic']['speed_range_kmh'] = bounds
            with self.assertRaisesRegex(ValueError, 'speed range'):
                traffic.prepare(context)
            context.grpc.spawn_vehicle.assert_not_called()

    def test_exclusions_apply_even_if_added_to_config_or_live_catalog(self):
        context = self.context()
        groups = context.cfg['highway_npc']['managed_traffic']['model_groups']
        groups['extra'] = list(traffic.EXCLUDED_MODELS)
        original = context.vehicle_models()
        context.vehicle_models = lambda: original + list(traffic.EXCLUDED_MODELS)
        self.assertFalse(set(traffic.model_candidates(context)) & traffic.EXCLUDED_MODELS)

    def test_seeded_selection_is_repeatable_and_varies_between_runs(self):
        context = self.context()
        first = traffic.model_candidates(context)
        self.assertEqual(first, traffic.model_candidates(context))
        context.run_seed += 1
        self.assertNotEqual(first[:25], traffic.model_candidates(context)[:25])

    def test_prepared_population_is_complete_paused_and_activates_once(self):
        context = self.context()
        highway.spawn_highway_npcs(context)
        self.assertEqual(context.highway_npc_planned_count, 25)
        self.assertEqual(len(context.scenario_actors), 25)
        self.assertEqual(len({row['model'] for row in context.highway_npcs}), 25)
        context.grpc.create_vehicle_spawn_point.assert_not_called()
        for call in context.grpc.spawn_vehicle.call_args_list:
            self.assertTrue(call.kwargs['pause'])
            self.assertEqual(call.kwargs['velocity'], 0.)
        self.assertFalse(highway.activate_highway_spawn_points(context, 0., 9.))
        context.grpc.set_vehicle_velocity.assert_not_called()
        self.assertTrue(highway.activate_highway_spawn_points(context, 1., 10.))
        self.assertEqual(context.grpc.set_vehicle_velocity.call_count, 25)
        self.assertTrue(highway.activate_highway_spawn_points(context, 2., 11.))
        self.assertEqual(context.grpc.set_vehicle_velocity.call_count, 25)
        report = json.loads((Path(context.global_cfg['paths']['runtime_root']) / 'highway_traffic.json').read_text())
        self.assertEqual((report['count'], report['distinct_models'], report['adjacent_lane_count']), (25, 25, 6))
        self.assertEqual(report['phase'], 'active')

    def test_failed_model_retries_unused_model_without_reducing_population(self):
        context = self.context()
        spawn = context.grpc.spawn_vehicle.side_effect
        first = traffic.model_candidates(context)[0]
        context.grpc.spawn_vehicle.side_effect = lambda **kw: None if kw['model_name'] == first else spawn(**kw)
        traffic.prepare(context)
        self.assertEqual(len(context.highway_npcs), 25)
        self.assertEqual(len({row['model'] for row in context.highway_npcs}), 25)
        self.assertNotIn(first, [row['model'] for row in context.highway_npcs])

    def test_middle_lane_retires_before_taper_without_moving_spawn_slots(self):
        context = self.context()
        plans, _ = traffic.build_plan(context)
        baseline = self.context()
        for lane in baseline.cfg['highway_npc']['managed_traffic']['lanes']:
            if lane['name'] == 'middle':
                self.assertEqual(lane['route'][-1], 'A2256W000434')
                self.assertTrue(lane['retire_at_route_end'])
                self.assertEqual(lane['lane_end_margin_m'], 30.)
                lane['route'] += ['A2256W000422', 'A2256W000153', 'A2256W000451', 'A2256W000446']
            elif lane['name'] == 'left':
                lane['route'] = lane['route'][:lane['route'].index('A2256W000432') + 1]
        old, _ = traffic.build_plan(baseline)
        for before, after in zip(old, plans):
            for key in ('x', 'y', 'z', 'yaw_deg', 'waypoint', 'offset_m', 'speed_kmh'):
                self.assertEqual(before[key], after[key])
            if after['lane'] == 'middle':
                self.assertEqual(after['route'][-1], 'A2256W000434')
            elif after['lane'] == 'left':
                end = self.loader.get_link_points(after['route'][-1])[-1]
                self.assertLess(end[1], -460.)
        traffic.prepare(context)
        for item, call in zip(context.highway_npcs, context.grpc.set_vehicle_route.call_args_list):
            indices = call.kwargs['route_waypoint_indices']
            self.assertEqual(indices[item['route'][0]], item['waypoint'])
            self.assertEqual(indices[item['route'][-1]],
                             len(self.loader.get_link_points(item['route'][-1])) - 1)

    def test_ai_false_ack_does_not_abort_native_route_following(self):
        context = self.context()
        traffic.prepare(context)
        context.grpc.set_vehicle_ai.return_value = False
        self.assertTrue(traffic.activate(context, 10.))
        self.assertEqual(context.grpc.set_vehicle_velocity.call_count, 25)
        self.assertEqual(len(context.highway_npcs), 25)
        self.assertTrue(all(not row['motion_contract']['ai_enable_acknowledged']
                            for row in context.highway_npcs))
        report = json.loads((Path(context.global_cfg['paths']['runtime_root']) / 'highway_traffic.json').read_text())
        self.assertEqual(report['phase'], 'active')
        self.assertTrue(all(not row['motion_contract']['ai_enable_acknowledged']
                            for row in report['vehicles']))

    def test_ai_false_ack_does_not_hide_failed_unpause_or_velocity(self):
        for failure in ('unpause', 'velocity'):
            context = self.context()
            traffic.prepare(context)
            context.grpc.set_vehicle_ai.return_value = False
            if failure == 'unpause':
                context.highway_npcs[0]['actor'].set_pause.return_value = False
            else:
                context.grpc.set_vehicle_velocity.return_value = False
            with self.assertRaisesRegex(RuntimeError, 'Cannot start highway vehicle'):
                traffic.activate(context, 10.)
            self.assertFalse(context.highway_spawn_points_active)
            context.spawn_highway_merge_static_vehicle.assert_not_called()

    def test_not_enough_models_fails_before_any_actor_creation(self):
        context = self.context()
        models = context.vehicle_models()[:24]
        context.vehicle_models = lambda: models
        with self.assertRaisesRegex(RuntimeError, '25 distinct'):
            traffic.prepare(context)
        context.grpc.spawn_vehicle.assert_not_called()

    def test_wrong_adjacent_lane_and_close_spacing_are_rejected(self):
        for mutation in ('lane', 'spacing'):
            context = self.context()
            lanes = context.cfg['highway_npc']['managed_traffic']['lanes']
            if mutation == 'lane':
                lanes[0]['route'] = lanes[1]['route']
            else:
                lanes[0]['offsets_m'][1] = lanes[0]['offsets_m'][0] + 19
            with self.assertRaises(ValueError):
                traffic.prepare(context)
            context.grpc.spawn_vehicle.assert_not_called()

    def test_failed_setup_actor_remains_registered_for_owner_cleanup(self):
        context = self.context()
        context.grpc.set_vehicle_route.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'prepare highway'):
            traffic.prepare(context)
        self.assertEqual(len(context.scenario_actors), 1)
        self.assertFalse(context.highway_spawn_points_active)

    def test_exit_cleans_population_and_cannot_reactivate(self):
        context = self.context()
        traffic.prepare(context)
        context._set_highway_spawn_points_enabled = Mock(return_value=0)
        highway.deactivate_highway_npcs(context, 825.)
        self.assertFalse(highway.activate_highway_spawn_points(context, 50., 825.))
        self.assertEqual(context.highway_npcs, [])
        context.grpc.reset_actors.assert_called_once()


if __name__ == '__main__':
    unittest.main()
