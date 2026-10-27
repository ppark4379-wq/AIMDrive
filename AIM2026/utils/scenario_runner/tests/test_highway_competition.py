"""Competition highway traffic: independent NPC flow, lane speeds, lane-end removal, supply and no-spawn zones."""
import argparse
import bisect
import copy
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..main_runner import load_configuration
from ..scenarios.full_loop_parts import highway, highway_competition as competition, highway_traffic
from ..utils.highway_density import add_highway_density_argument, apply_highway_density
from ..utils.map_loader import MGeoMapLoader
from ..utils.transform_utils import interpolate_on_polyline, project_distance_on_polyline

EGO_START = (-25.9, 345.1)       # the highway mission starts here, ~90 m from the supply slots after the fork
EGO_FAR = (70., -300.)           # Ego far downstream: supply slots are free


class FakeSim:
    """Just enough of MORAI for the traffic manager: vehicles follow their route at the commanded speed."""

    def __init__(self, loader, ego=EGO_START):
        self.loader, self.ego, self.vehicles, self.count = loader, ego, {}, 0
        self.fail_models, self.destroyed, self.now = set(), [], 0.
        self.grpc = SimpleNamespace(
            make_transform=lambda x, y, z, yaw: (x, y, z, yaw), spawn_vehicle=self.spawn,
            set_vehicle_route=self.route, set_vehicle_speed_limit=self.limit, set_vehicle_ai=lambda a, e: True,
            set_vehicle_velocity=self.velocity, destroy_vehicle_by_id=self.destroy,
            get_all_vehicle_states=self.states)

    def spawn(self, transform, model_name, label, velocity, multi_ego, pause, request_id, recover_ambiguous):
        if model_name in self.fail_models:
            return None
        self.count += 1
        actor = Mock()
        actor.get_id.return_value = f'{label}#{self.count}'
        vehicle = dict(id=actor.get_id.return_value, x=transform[0], y=transform[1], paused=True, target=0.,
                       limit=math.inf, poly=None, s=0., model=model_name, offset_y=0.)
        self.vehicles[vehicle['id']] = vehicle
        actor.set_pause.side_effect = lambda flag: vehicle.update(paused=bool(flag)) or True
        return actor

    def route(self, actor, links, route_waypoint_indices=None, decision_range=30., label=''):
        vehicle = self.vehicles[actor.get_id()]
        vehicle['poly'] = [p for link in links for p in self.loader.get_link_points(link)]
        vehicle['s'] = project_distance_on_polyline(vehicle['poly'], vehicle['x'], vehicle['y'])
        cumulative = [0.]
        for a, b in zip(vehicle['poly'], vehicle['poly'][1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        vehicle['cum'] = cumulative
        return True

    def limit(self, actor, limit_mps, enabled=True):
        self.vehicles[actor.get_id()]['limit'] = limit_mps
        return True

    def velocity(self, actor, kmh):
        self.vehicles[actor.get_id()]['target'] = kmh
        return True

    def destroy(self, actor_id):
        self.destroyed.append(actor_id)
        return self.vehicles.pop(actor_id, None) is not None

    def clock(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        for vehicle in self.vehicles.values():
            if vehicle['paused'] or vehicle['poly'] is None:
                continue
            speed = min(vehicle['target'] / 3.6, vehicle['limit'])
            cumulative = vehicle['cum']
            vehicle['s'] = min(vehicle['s'] + speed * seconds, cumulative[-1])
            i = min(max(bisect.bisect_right(cumulative, vehicle['s']) - 1, 0), len(cumulative) - 2)
            span = cumulative[i + 1] - cumulative[i]
            t = 0. if span == 0. else (vehicle['s'] - cumulative[i]) / span
            a, b = vehicle['poly'][i], vehicle['poly'][i + 1]
            vehicle['x'], vehicle['y'] = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])

    def states(self):
        def state(identifier, x, y, kmh):
            return SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value=identifier)),
                                   transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.)),
                                   velocity=SimpleNamespace(x=kmh, y=0., z=0.))
        rows = [state(v['id'], v['x'] + v['offset_y'], v['y'], 0. if v['paused'] else min(v['target'], v['limit'] * 3.6))
                for v in self.vehicles.values()]
        return rows + [state('Ego', self.ego[0], self.ego[1], 0.)]


class CompetitionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.global_cfg, _, cls.base_config = load_configuration()
        cls.loader = MGeoMapLoader(cls.global_cfg['paths']['mgeo_root'])

    def config(self):
        config = copy.deepcopy(self.base_config)
        apply_highway_density(config, 'competition')
        return config

    def context(self, config=None, seed=7, sim=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = config or self.config()
        groups = config['highway_npc']['managed_traffic']['model_groups']
        models = [model for values in groups.values() for model in values]
        sim = sim or FakeSim(self.loader)
        context = SimpleNamespace(
            cfg=config, map_loader=self.loader, grpc=sim.grpc, vehicle_models=lambda: models, run_seed=seed,
            run_instance_id='test', scenario_actors=[], highway_npcs=[], highway_spawn_points=[],
            highway_spawn_points_active=False, highway_activation_s=110., highway_traffic_cleared=False,
            highway_exit_s=900., spawn_highway_merge_static_vehicle=Mock(),
            global_cfg={'paths': {'runtime_root': directory.name}})
        context.sim = sim
        return context

    def prepared(self, **kwargs):
        context = self.context(**kwargs)
        highway_traffic.prepare(context)
        return context

    def prewarmed(self, **kwargs):
        context = self.prepared(**kwargs)
        competition.prewarm(context, sleep=context.sim.advance, clock=context.sim.clock)
        return context

    def run_traffic(self, context, seconds, ego=EGO_FAR, step=0.1, start=0., yaw=0., progress=500.):
        sim, t = context.sim, start
        sim.ego = ego
        while t < start + seconds - 1e-9:
            sim.advance(step)
            t = round(t + step, 6)
            competition.update(context, t, {'x': ego[0], 'y': ego[1], 'yaw_deg': yaw}, progress)
        return t

    def report(self, context):
        return json.loads((Path(context.global_cfg['paths']['runtime_root']) / 'highway_competition.json').read_text())


class PlanTests(CompetitionTestCase):
    def test_the_overlay_selects_the_policy_without_touching_other_settings(self):
        config = self.config()
        self.assertTrue(competition.enabled(SimpleNamespace(cfg=config)))
        self.assertFalse(competition.enabled(SimpleNamespace(cfg=self.base_config)))
        self.assertEqual({k: v for k, v in config.items() if k != 'highway_npc'},
                         {k: v for k, v in self.base_config.items() if k != 'highway_npc'})
        managed = config['highway_npc']['managed_traffic']
        self.assertIsNone(managed['speed_range_kmh'])
        self.assertIsNone(managed['downstream_lead_speed_kmh'])
        self.assertEqual(config['highway_npc']['collection_maximum_total_vehicles'], 48)

    def test_default_traffic_is_unchanged_and_the_selector_knows_the_new_choice(self):
        for density in (None, 'dense'):
            selected = copy.deepcopy(self.base_config)
            apply_highway_density(selected, density)
            self.assertEqual(selected, self.base_config)
        self.assertEqual(self.base_config['highway_npc']['managed_traffic']['speed_range_kmh'], [20, 100])
        parser = argparse.ArgumentParser()
        add_highway_density_argument(parser)
        self.assertEqual(parser.parse_args(['--highway-density', 'competition']).highway_density, 'competition')
        with self.assertRaises(SystemExit):
            parser.parse_args(['--highway-density', 'unknown'])
        with self.assertRaises(ValueError):
            apply_highway_density(copy.deepcopy(self.base_config), 'unknown')

    def test_legacy_speed_policies_must_not_override_lane_speeds(self):
        for key, value in (('speed_range_kmh', [20, 100]), ('downstream_lead_speed_kmh', 80.)):
            config = self.config()
            config['highway_npc']['managed_traffic'][key] = value
            with self.assertRaisesRegex(ValueError, key):
                competition.build_plan(self.context(config=config))

    def test_initial_fleet_is_dense_and_spread_over_the_whole_course(self):
        context = self.prepared()
        npcs = context.highway_npcs
        self.assertEqual(len(npcs), 38)
        pool = len(context.highway_competition['pool'])
        self.assertEqual(len({n['model'] for n in npcs}), min(38, pool))       # distinct while the model pool lasts
        self.assertEqual({n['lane']: sum(1 for m in npcs if m['lane'] == n['lane']) for n in npcs},
                         {'merge_adjacent': 14, 'middle': 10, 'left': 14})
        self.assertTrue(all(n['source'] == 'initial' for n in npcs))
        for lane in ('merge_adjacent', 'middle', 'left'):
            offsets = [n['offset_m'] for n in npcs if n['lane'] == lane]
            self.assertGreater(max(offsets), 300)          # traffic well downstream, not only at the merge
        # spawned paused: nothing moves before the pre-warm
        self.assertTrue(all(v['paused'] for v in context.sim.vehicles.values()))
        self.assertFalse(context.highway_spawn_points_active)

    def test_supply_slots_are_on_each_lanes_own_link_after_the_fork(self):
        # No shared road: supply of one lane must not block (or be blocked by) the traffic of the other lanes.
        context = self.prepared()
        lanes = context.highway_competition['lanes']
        firsts = {lane['supply']['route'][0] for lane in lanes.values()}
        self.assertEqual(firsts, {'A2256W000406', 'A2256W000175', 'A2256W000029'})
        approach = {'A2256W000157', 'A2256W000868', 'A2256W000168', 'A2256W000419', 'A2256W000415', 'A2256W000404'}
        self.assertTrue(all(lane['supply']['route'][0] not in approach for lane in lanes.values()))

    def test_supply_waits_for_ego_progress_and_for_distance_then_tops_up(self):
        context = self.prewarmed()
        victim = next(n for n in context.highway_npcs if n['lane'] == 'left')
        context.sim.destroy(victim['actor_id'])
        context.highway_npcs.remove(victim)
        count = context.sim.count
        self.run_traffic(context, 5, ego=EGO_START, progress=40.)        # the start: only the initial burst exists
        self.assertEqual(context.sim.count, count)
        self.run_traffic(context, 5, ego=(66., 100.), progress=300., start=5.)   # progressed and 220 m from the slots
        self.assertGreater(context.sim.count, count)

    def test_one_target_speed_per_lane_and_different_speeds_between_lanes(self):
        context = self.prepared()
        by_lane = {}
        for npc in context.highway_npcs:
            by_lane.setdefault(npc['lane'], set()).add(npc['speed_kmh'])
            self.assertEqual(npc['speed_kmh'], npc['target_speed_kmh'])
        self.assertEqual(by_lane, {'merge_adjacent': {45.}, 'middle': {60.}, 'left': {75.}})
        limits = {v['limit'] for v in context.sim.vehicles.values()}
        self.assertEqual(limits, {45 / 3.6, 60 / 3.6, 75 / 3.6})

    def test_same_seed_reproduces_the_layout_and_models(self):
        first, second, other = self.prepared(seed=11), self.prepared(seed=11), self.prepared(seed=12)
        key = lambda c: [(n['label'], n['model'], round(n['x'], 3), round(n['y'], 3), n['speed_kmh']) for n in c.highway_npcs]
        self.assertEqual(key(first), key(second))
        self.assertNotEqual([n['model'] for n in first.highway_npcs], [n['model'] for n in other.highway_npcs])
        self.assertEqual([(n['label'], round(n['x'], 3)) for n in first.highway_npcs],
                         [(n['label'], round(n['x'], 3)) for n in other.highway_npcs])

    def test_invalid_layouts_fail_before_anything_is_spawned(self):
        cases = {
            'spacing': lambda lanes: lanes[0].update(initial_offsets_m=[-450, -440]),
            'endpoint': lambda lanes: lanes[1].update(initial_offsets_m=[-420, 480]),
            'speed entry': lambda lanes: lanes[2].update(name='unknown'),
        }
        for name, change in cases.items():
            with self.subTest(name):
                context = self.context()
                lanes = context.cfg['highway_npc']['managed_traffic']['competition']['lanes']
                change(lanes)
                with self.assertRaises(ValueError):
                    highway_traffic.prepare(context)
                self.assertEqual(context.sim.vehicles, {})

    def test_the_population_cap_is_enforced_before_spawning(self):
        context = self.context()
        context.cfg['highway_npc']['managed_traffic']['competition']['cap'] = 30
        with self.assertRaisesRegex(ValueError, 'cap'):
            highway_traffic.prepare(context)
        self.assertEqual(context.sim.vehicles, {})
        context = self.context()
        context.cfg['highway_npc']['collection_maximum_total_vehicles'] = 25
        with self.assertRaisesRegex(ValueError, 'cap'):
            highway_traffic.prepare(context)

    def test_routes_never_enter_the_ego_ramp(self):
        context = self.context()
        forbidden = set(context.cfg['highway_npc']['managed_traffic']['forbidden_route_links'])
        _, plans, lanes, _, _ = competition.build_plan(context)
        self.assertTrue(all(not forbidden & set(lane['route']) for lane in lanes.values()))
        self.assertTrue(all(not forbidden & set(plan['route']) for plan in plans))

    def test_only_the_lane_drop_lane_ends_at_a_link_before_the_taper(self):
        _, _, lanes, _, _ = competition.build_plan(self.context())
        self.assertEqual({n: (l['retire_at_route_end'], l['end_link']) for n, l in lanes.items()},
                         {'merge_adjacent': (False, 'A2256W001105'), 'middle': (True, 'A2256W000434'),
                          'left': (False, 'A2256W000444')})
        self.assertNotIn('A2256W000422', lanes['middle']['route'])      # no route into the neighbouring lane


class StartTests(CompetitionTestCase):
    def test_release_does_not_depend_on_ego_position(self):
        context = self.prepared()
        for current_s in (0., 60., 110., 500., 5000.):
            self.assertFalse(highway_traffic.activate(context, current_s, {'speed_mps': 10.}, 5.))
        self.assertTrue(all(v['paused'] for v in context.sim.vehicles.values()))
        self.assertFalse(highway.activate_highway_spawn_points(context, 5., 5000., {'speed_mps': 10.}))
        competition.prewarm(context, sleep=context.sim.advance, clock=context.sim.clock)
        self.assertTrue(highway_traffic.activate(context, 0., None, 0.))
        self.assertTrue(all(not v['paused'] for v in context.sim.vehicles.values()))

    def test_prewarm_measures_real_movement_and_records_it(self):
        context = self.prepared()
        before = {v['id']: (v['x'], v['y']) for v in context.sim.vehicles.values()}
        result = competition.prewarm(context, sleep=context.sim.advance, clock=context.sim.clock)
        self.assertEqual(result['phase'], 'verified')
        self.assertEqual((result['moved'], result['total']), (38, 38))
        self.assertGreaterEqual(result['duration_s'], 4.)
        moved = [math.hypot(v['x'] - before[v['id']][0], v['y'] - before[v['id']][1]) for v in context.sim.vehicles.values()]
        self.assertTrue(all(3. <= distance <= 100. for distance in moved))      # ~4 s at 45..75 km/h, not a jump
        self.assertTrue(all(npc.get('x_now') is None for npc in context.highway_npcs))     # no removal during the pre-warm
        context.spawn_highway_merge_static_vehicle.assert_called_once_with()
        self.assertEqual(competition.prewarm(context, sleep=context.sim.advance, clock=context.sim.clock)['phase'], 'verified')
        self.assertEqual(context.spawn_highway_merge_static_vehicle.call_count, 1)

    def test_ego_start_is_blocked_when_the_npcs_do_not_move(self):
        context = self.prepared()
        ticks = iter(range(1, 1000))
        with self.assertRaisesRegex(RuntimeError, 'movement was not verified'):
            competition.prewarm(context, sleep=lambda s: None, clock=lambda: next(ticks) * 0.5)
        self.assertEqual(context.highway_competition['prewarm']['phase'], 'failed')

    def test_expert_drive_starts_only_after_the_prewarm(self):
        from ..scenarios.full_loop_scenario import FullLoopScenario
        from ..core.full_loop_drive_base import FullLoopDriveBase
        order = []
        scenario = object.__new__(FullLoopScenario)
        scenario.cfg = self.config()
        scenario.highway_competition = {'prewarm': {'phase': 'prepared'}}
        with patch.object(competition, 'prewarm', side_effect=lambda ctx: order.append('prewarm')), \
             patch.object(FullLoopDriveBase, 'activate_expert_drive', side_effect=lambda **hooks: order.append('ego_auto_and_expert') or True):
            self.assertTrue(scenario.activate_expert_drive(before_expert_start=None))
        self.assertEqual(order, ['prewarm', 'ego_auto_and_expert'])
        other = object.__new__(FullLoopScenario)
        other.cfg = copy.deepcopy(self.base_config)
        order.clear()
        with patch.object(competition, 'prewarm', side_effect=lambda ctx: order.append('prewarm')), \
             patch.object(FullLoopDriveBase, 'activate_expert_drive', side_effect=lambda **hooks: order.append('ego_auto_and_expert') or True):
            other.activate_expert_drive()
        self.assertEqual(order, ['ego_auto_and_expert'])      # dense/sparse flow is untouched


class ManagementTests(CompetitionTestCase):
    def test_lane_drop_npcs_are_removed_before_the_taper_and_leave_the_population(self):
        context = self.prewarmed()
        self.run_traffic(context, 120)
        removed = context.highway_competition['removed']
        lane_end = [r for r in removed if r['reason'] == 'lane_end']
        self.assertGreater(len(lane_end), 0)
        self.assertEqual({r['lane'] for r in lane_end}, {'middle'})
        self.assertEqual({r['link'] for r in lane_end}, {'A2256W000434'})
        # nothing of the middle lane ever got onto the taper link: all of its vehicles end at 434's end point
        end_y = self.loader.get_link_points('A2256W000434')[-1][1]
        for npc in context.highway_npcs:
            if npc['lane'] == 'middle':
                self.assertGreaterEqual(npc.get('y_now', npc['y']), end_y - 1.0)
        self.assertTrue(all(i not in context.sim.vehicles for i in context.sim.destroyed))
        for record in lane_end:
            self.assertTrue(any(record['label'] in destroyed for destroyed in context.sim.destroyed))
        self.assertTrue(all(npc['label'] not in {r['label'] for r in removed} for npc in context.highway_npcs))

    def test_slow_updates_cannot_skip_the_lane_end(self):
        # A real run updates every few tenths of a second: at 60 km/h one step is longer than the old 5 m window.
        for step in (0.5, 1.0, 2.0):
            with self.subTest(step=step):
                context = self.prewarmed()
                context.highway_competition['cfg']['update_period_s'] = step
                self.run_traffic(context, 140, step=step)
                end = self.loader.get_link_points('A2256W000434')[-1]
                lane_end = [r for r in context.highway_competition['removed'] if r['reason'] == 'lane_end']
                self.assertGreater(len(lane_end), 5)
                for npc in context.highway_npcs:
                    if npc['lane'] == 'middle' and 'y_now' in npc:
                        self.assertGreater(npc['y_now'], end[1] + 5., 'a middle-lane NPC reached the taper')

    def test_a_vehicle_that_is_already_past_the_end_is_removed(self):
        context = self.prewarmed()
        self.run_traffic(context, 5)
        npc = next(n for n in context.highway_npcs if n['lane'] == 'middle')
        lane = context.highway_competition['lanes']['middle']
        original = context.sim.states

        def reported():
            rows = original()
            for row in rows:
                if row.actor_info.id.value == npc['actor_id']:         # reported 60 m beyond the end, on the taper
                    row.transform.location.x, row.transform.location.y = lane['xy'][-1][0] + 4., lane['xy'][-1][1] - 60.
            return rows
        context.grpc.get_all_vehicle_states = reported
        self.run_traffic(context, 1, start=5.)
        self.assertNotIn(npc, context.highway_npcs)
        record = next(r for r in context.highway_competition['removed'] if r['label'] == npc['label'])
        self.assertEqual(record['reason'], 'lane_end')

    def test_the_local_search_does_not_trust_a_vertex_at_the_window_edge(self):
        lane = dict(xy=[(float(i), 0.) for i in range(1000)])
        index, lateral = competition._local_lateral(lane, 0, 600., 0.5)       # jumped 600 vertices since the hint
        self.assertEqual(index, 600)
        self.assertAlmostEqual(lateral, 0.5)
        index, _ = competition._local_lateral(lane, 590, 600., 0.)
        self.assertEqual(index, 600)

    def test_an_npc_that_stops_on_the_road_is_removed(self):
        context = self.prewarmed()
        self.run_traffic(context, 8)
        npc = next(n for n in context.highway_npcs if n['lane'] == 'left')
        context.sim.vehicles[npc['actor_id']]['target'] = 0.            # stuck: it no longer moves
        self.run_traffic(context, 2, start=8.)
        self.assertIn(npc, context.highway_npcs)                         # not at once: brief stops are normal in traffic
        self.run_traffic(context, 6, start=10.)
        self.assertNotIn(npc, context.highway_npcs)
        record = next(r for r in context.highway_competition['removed'] if r['label'] == npc['label'])
        self.assertEqual(record['reason'], 'stalled')

    def test_other_lanes_are_not_removed_by_the_lane_end_policy(self):
        context = self.prewarmed()
        self.run_traffic(context, 150)
        reasons = {(r['lane'], r['reason']) for r in context.highway_competition['removed']}
        self.assertNotIn(('left', 'lane_end'), reasons)
        self.assertNotIn(('merge_adjacent', 'lane_end'), reasons)
        for record in context.highway_competition['removed']:
            if record['reason'] == 'terminal':
                self.assertGreaterEqual(record['ego_distance_m'], 60.)

    def test_supply_keeps_the_population_cap_and_the_lane_speeds(self):
        context = self.prewarmed()
        cap = context.highway_competition['cfg']['cap']
        t = 0.
        for _ in range(8):
            t = self.run_traffic(context, 40, start=t)
            self.assertLessEqual(len(context.highway_npcs), cap)
            pool = len(context.highway_competition['pool'])
            self.assertGreaterEqual(len({n['model'] for n in context.highway_npcs}), min(len(context.highway_npcs), pool) - 6)
        state = context.highway_competition
        self.assertGreater(state['supplied'], 0)
        speeds = {}
        for npc in context.highway_npcs:
            speeds.setdefault(npc['lane'], set()).add(npc['speed_kmh'])
        self.assertTrue(all(len(values) == 1 for values in speeds.values()))
        for npc in context.highway_npcs:
            self.assertEqual(npc['speed_kmh'], {'merge_adjacent': 45., 'middle': 60., 'left': 75.}[npc['lane']])
            self.assertEqual(context.sim.vehicles[npc['actor_id']]['limit'], npc['speed_kmh'] / 3.6)

    def test_traffic_stays_on_after_the_first_wave_has_left(self):
        context = self.prewarmed()
        self.run_traffic(context, 400)
        self.assertGreaterEqual(len(context.highway_npcs), 30)
        self.assertGreater(context.highway_competition['supplied'], 10)
        self.assertEqual(self.report(context)['phase'], 'running')

    def test_nothing_is_spawned_near_ego_and_the_shortage_is_logged(self):
        context = self.prewarmed()
        self.run_traffic(context, 60)
        victim = next(n for n in context.highway_npcs if n['lane'] == 'left')
        context.sim.destroy(victim['actor_id'])
        context.highway_npcs.remove(victim)
        supply = context.highway_competition['lanes']['left']['supply']
        near_ego = (supply['x'] + 30., supply['y'])      # 30 m from the slot
        count = context.sim.count
        self.run_traffic(context, 20, ego=near_ego, start=60.)
        self.assertEqual(context.sim.count, count)           # Ego within 120 m of the slot: no spawn at all
        shortage = context.highway_competition['shortages']['left']
        self.assertGreater(shortage['reasons']['ego_distance'], 0)
        log = self.report(context)
        self.assertTrue(any(e['reason'] == 'ego_distance' and e['lane'] == 'left' for e in log['shortage_log']))

    def test_a_vehicle_on_the_slot_blocks_the_supply(self):
        context = self.prewarmed()
        self.run_traffic(context, 50)
        victim = next(n for n in context.highway_npcs if n['lane'] == 'middle')
        context.sim.destroy(victim['actor_id'])
        context.highway_npcs.remove(victim)
        supply = context.highway_competition['lanes']['middle']['supply']
        blocker = context.sim.vehicles[next(iter(context.sim.vehicles))]
        blocker.update(x=supply['x'], y=supply['y'], paused=True)
        before = {v for v in context.sim.vehicles if v.startswith('highway_middle_s')}
        self.run_traffic(context, 3, start=50.)          # shorter than the stall removal of a stopped vehicle (4 s)
        after = {v for v in context.sim.vehicles if v.startswith('highway_middle_s')}
        self.assertEqual(after - before, set())                       # no new middle NPC (other lanes may still spawn)
        self.assertGreater(context.highway_competition['shortages']['middle']['reasons']['clearance'], 0)

    def test_incomplete_telemetry_never_deletes_or_spawns(self):
        context = self.prewarmed()
        self.run_traffic(context, 20)
        npcs, count = list(context.highway_npcs), context.sim.count
        with patch.object(context.sim, 'states', return_value=[]):
            context.grpc.get_all_vehicle_states = context.sim.states
            competition.update(context, 21., {'x': EGO_START[0], 'y': EGO_START[1]}, 10.)
        self.assertEqual(context.highway_npcs, npcs)
        self.assertEqual(context.sim.count, count)
        self.assertGreater(context.highway_competition['telemetry_gaps'], 0)

    def test_failed_model_spawns_try_the_next_model(self):
        context = self.prewarmed()
        self.run_traffic(context, 60)
        victim = next(n for n in context.highway_npcs if n['lane'] == 'left')
        context.sim.destroy(victim['actor_id'])
        context.highway_npcs.remove(victim)
        first = competition._candidates(context.highway_competition, context)[0]
        context.sim.fail_models = {first}
        self.run_traffic(context, 20, start=60.)
        added = [n for n in context.highway_npcs if n['source'] == 'supply' and n['lane'] == 'left']
        self.assertTrue(added)
        self.assertTrue(all(n['model'] != first for n in added))

    def test_cleared_highway_stops_the_manager(self):
        context = self.prewarmed()
        context.highway_traffic_cleared = True
        count = context.sim.count
        self.run_traffic(context, 30)
        self.assertEqual(context.sim.count, count)
        self.assertEqual(context.highway_competition['observations'], [])


class CoverageTests(CompetitionTestCase):
    """Ego drives the real highway mission route the way the real run did: parked while the Expert starts, then slowly."""

    def drive_mission(self, seed=7, startup_gap_s=7.0, rear=False):
        from ..scenarios.full_loop_parts.route_profile import load_global_path
        context = self.context(seed=seed)
        route = load_global_path(SimpleNamespace(global_cfg=self.global_cfg, cfg=context.cfg, map_loader=self.loader))[1975:3639]
        cumulative = [0.]
        for a, b in zip(route, route[1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        context.highway_exit_s = cumulative[-1]
        context.route_points = route                                   # known during the setup, as in a real run
        context.cfg['highway_npc']['managed_traffic']['competition']['rear_reduction']['enabled'] = rear
        highway_traffic.prepare(context)
        competition.prewarm(context, sleep=context.sim.advance, clock=context.sim.clock)
        for _ in range(int(startup_gap_s * 10)):                       # the Expert starts: NPCs run, Ego is parked
            context.sim.advance(0.1)
        t, s, last = 0., 0., 0.
        while s < cumulative[-1] - 30:
            speed = min(11., 1.1 * t)                                    # ~40 km/h cruising, slow start (real run: 38 km/h mean)
            s += speed * 0.1
            i = min(max(bisect.bisect_right(cumulative, s) - 1, 0), len(route) - 2)
            a, b = route[i], route[i + 1]
            f = (s - cumulative[i]) / max(1e-9, cumulative[i + 1] - cumulative[i])
            x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            context.sim.ego = (x, y)
            context.sim.advance(0.1)
            t = round(t + 0.1, 6)
            competition.update(context, t, {'x': x, 'y': y, 'yaw_deg': 0., 'speed_mps': speed}, s)
        competition._persist(context, 'test')
        return context, self.report(context)

    def rows(self, document, low, high):
        info = [r for r in document['observations'] if low <= r.get('ego_route_s_m', -1) <= high]
        self.assertTrue(info)
        return info

    def test_traffic_is_dense_when_ego_enters_the_lane(self):
        context, document = self.drive_mission()
        merge = context.highway_competition['route_state']['merge']['s']
        near = self.rows(document, merge - 40, merge + 40)
        self.assertGreaterEqual(min(r['around_route'] for r in near), 8)            # NPCs within 120 m around Ego
        self.assertGreaterEqual(min(r['ahead_far_route'] for r in near), 6)
        self.assertEqual(document['lane_violations'], [])
        self.assertEqual(document['spawn_failures'], 0)

    def test_traffic_stays_ahead_of_ego_after_the_merge_for_overtaking(self):
        context, document = self.drive_mission()
        merge = context.highway_competition['route_state']['merge']['s']
        total = context.highway_competition['route_state']['total']
        after = self.rows(document, merge + 40, total - 260)           # the last stretch has no road left ahead on the route
        self.assertGreaterEqual(sum(r['ahead_far_route'] for r in after) / len(after), 7)
        self.assertGreaterEqual(min(r['ahead_far_route'] for r in after), 3)
        self.assertGreaterEqual(sum(r['ahead_route'] for r in after) / len(after), 3.5)      # within 150 m

    def test_the_middle_lane_is_not_thin_and_no_lane_is_laid_out_evenly(self):
        context = self.prepared()
        by_lane = {}
        for npc in context.highway_npcs:
            by_lane.setdefault(npc['lane'], []).append(npc['offset_m'])
        self.assertGreaterEqual(len(by_lane['middle']), 0.7 * len(by_lane['merge_adjacent']))
        for lane, offsets in by_lane.items():
            offsets = sorted(offsets)
            gaps = [b - a for a, b in zip(offsets, offsets[1:])]
            self.assertGreater(max(gaps), 2.2 * min(gaps), f'{lane}: gaps {gaps} look like an even grid')
        context, document = CoverageTests.drive_mission(self)
        share = {n: lane['supplied'] for n, lane in document['summary']['lanes'].items()}
        self.assertGreaterEqual(share['middle'], 0.7 * max(share.values()))

    def test_few_rear_npcs_reduce_the_vehicles_behind_ego_only_from_the_lane_entry(self):
        context, normal = self.drive_mission()
        light_context, light = self.drive_mission(rear=True)
        merge = light_context.highway_competition['route_state']['merge']['s']
        self.assertEqual(light['initial_count'], normal['initial_count'])           # nothing differs at the start
        before = lambda d: [(r['t'], r['active'], r['around_route']) for r in d['observations'] if r.get('ego_route_s_m', 0) < merge - 5]
        self.assertEqual(before(light), before(normal))                              # ... nor before the lane entry
        started = light['rear_reduction']['started']
        self.assertIsNotNone(started)
        self.assertGreaterEqual(started['ego_route_s_m'], merge)                   # the thinning starts at the lane entry
        self.assertEqual(normal['rear_reduction']['removed'], 0)
        after = lambda d: [r for r in d['observations'] if merge + 60 <= r.get('ego_route_s_m', 0)]
        behind = lambda d: sum(r['behind_route'] for r in after(d)) / len(after(d))
        self.assertLess(behind(light), 0.7 * behind(normal))
        ahead = lambda d: sum(r['ahead_far_route'] for r in after(d)) / len(after(d))
        self.assertGreaterEqual(ahead(light), 0.75 * ahead(normal))                 # traffic ahead (overtaking) stays
        for record in light['removed']:
            if record['reason'] == 'rear_thinned':
                self.assertGreaterEqual(record['ego_distance_m'], 80.)              # never right behind Ego

    def test_the_feed_never_creates_a_vehicle_near_ego_and_replaces_the_fork_supply(self):
        context, document = self.drive_mission()
        self.assertGreater(document['fed'], 10)
        self.assertEqual(document['supplied'], 0)
        fed = [v for v in document['vehicles'] if v['source'] in ('feed_merge', 'feed_ahead')]
        self.assertTrue(fed)
        for vehicle in fed:
            minimum = 100. if vehicle['source'] == 'feed_merge' else 140.
            self.assertGreaterEqual(vehicle['spawn_ego_distance_m'], minimum - 1e-6, vehicle['label'])
        self.assertLessEqual(len(context.highway_npcs), context.highway_competition['cfg']['cap'])

    def test_the_whole_drive_is_reproducible_for_one_seed(self):
        first_context, first = self.drive_mission(seed=21)
        second_context, second = self.drive_mission(seed=21)
        key = lambda d: ([(r['label'], r['reason'], r['t']) for r in d['removed']],
                         [(v['label'], v['model'], v['target_speed_kmh']) for v in d['vehicles']], d['fed'])
        self.assertEqual(key(first), key(second))
        _, other = self.drive_mission(seed=22)
        self.assertNotEqual([v['model'] for v in first['vehicles']], [v['model'] for v in other['vehicles']])

    def test_the_feed_is_cheap_enough_for_the_scenario_tick(self):
        from ..scenarios.full_loop_parts.route_profile import load_global_path
        context = self.prewarmed()
        route = load_global_path(SimpleNamespace(global_cfg=self.global_cfg, cfg=context.cfg, map_loader=self.loader))[1975:3639]
        context.route_points = route
        competition._route_state(context, context.highway_competition)
        started = time.perf_counter()
        for k in range(300):
            context.sim.advance(0.1)
            x, y = route[min(len(route) - 1, k * 4)][:2]
            context.sim.ego = (x, y)
            competition.update(context, round(0.1 * (k + 1), 6), {'x': x, 'y': y, 'yaw_deg': 0., 'speed_mps': 10.}, k * 2.)
        per_update = (time.perf_counter() - started) / 300
        self.assertLess(per_update, 0.02, f'{per_update * 1000:.1f} ms per update with the feed')

    def test_the_manager_is_cheap_enough_for_the_scenario_tick(self):
        context = self.prewarmed()
        started = time.perf_counter()
        self.run_traffic(context, 20)                                   # 200 updates with 25 vehicles
        per_update = (time.perf_counter() - started) / 200
        self.assertLess(per_update, 0.01, f'{per_update * 1000:.1f} ms per update')


class EvidenceTests(CompetitionTestCase):
    def test_a_lane_change_is_reported_and_normal_driving_is_not(self):
        context = self.prewarmed()
        self.run_traffic(context, 30)
        self.assertEqual(context.highway_competition['lane_violations'], [])
        npc = next(n for n in context.highway_npcs if n['lane'] == 'left')
        vehicle = context.sim.vehicles[npc['actor_id']]
        vehicle['offset_y'] = 3.6                                     # reported 3.6 m to the side: one lane over
        self.run_traffic(context, 3, start=30.)
        violations = context.highway_competition['lane_violations']
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['lane'], 'left')

    def test_target_and_actual_speeds_are_recorded_separately(self):
        context = self.prewarmed()
        self.run_traffic(context, 30)
        context.sim.vehicles[context.highway_npcs[0]['actor_id']]['limit'] = 20 / 3.6      # actual differs from target
        self.run_traffic(context, 20, start=30.)
        competition._persist(context, 'test')
        document = self.report(context)
        vehicle = next(v for v in document['vehicles'] if v['label'] == context.highway_npcs[0]['label'])
        self.assertEqual(vehicle['target_speed_kmh'], context.highway_npcs[0]['speed_kmh'])
        self.assertLess(vehicle['actual_speed_kmh'], vehicle['target_speed_kmh'])
        lanes = document['summary']['lanes']
        self.assertEqual({n: l['target_speed_kmh'] for n, l in lanes.items()},
                         {'merge_adjacent': 45., 'middle': 60., 'left': 75.})
        self.assertTrue(all(len(l['speeds_distinct_targets']) <= 1 for l in lanes.values()))

    def test_observations_count_vehicles_ahead_and_behind_ego(self):
        context = self.prewarmed()
        self.run_traffic(context, 5, ego=(63.8, 300.), yaw=-90., start=0.)
        context.sim.ego = (63.8, 300.)
        rows = context.highway_competition['observations']
        self.assertTrue(rows)
        self.assertIn('ahead', rows[-1])
        self.assertIn('behind', rows[-1])
        self.assertEqual(sum(rows[-1]['per_lane'].values()), rows[-1]['active'])
        competition._persist(context, 'test')
        self.assertIn('ahead_by_progress_decile', self.report(context)['summary'])

    def test_final_evidence_is_written_before_the_highway_is_cleared(self):
        context = self.prewarmed()
        self.run_traffic(context, 10)
        context.grpc.reset_actors = Mock()
        context._set_highway_spawn_points_enabled = Mock(return_value=0)
        highway.deactivate_highway_npcs(context, 900.)
        self.assertEqual(self.report(context)['phase'], 'finished')
        self.assertTrue(context.highway_traffic_cleared)


class ZoneTests(CompetitionTestCase):
    def zone_config(self, link, **extra):
        config = self.config()
        config['highway_npc']['managed_traffic']['competition']['no_spawn_zones'] = [dict(name='test_zone', links=[link], **extra)]
        return config

    def test_zone_band_follows_the_lane_width_and_excludes_adjacent_lanes(self):
        context = self.context(config=self.zone_config('A2256W000409'))
        zone_list = competition.zones(context, competition.settings(context))
        points = self.loader.get_link_points('A2256W000409')
        x, y, _, _ = interpolate_on_polyline(points, 80.)
        self.assertEqual(competition.zone_containing(zone_list, x, y), 'test_zone')
        # the legal neighbour lane (A2256W000410, ~3.6 m away) is a normal driving lane, not part of the zone
        nx, ny, _, _ = interpolate_on_polyline(self.loader.get_link_points('A2256W000410'), 80.)
        self.assertGreater(math.hypot(nx - x, ny - y), 3.)
        self.assertIsNone(competition.zone_containing(zone_list, nx, ny))
        self.assertIsNone(competition.zone_containing(zone_list, x + 20., y + 20.))
        # a vehicle body overhanging the centre line by less than the margin is still excluded
        self.assertEqual(competition.zone_containing(zone_list, x + 2.4, y), 'test_zone')

    def test_section_range_and_margins_limit_the_zone(self):
        context = self.context(config=self.zone_config('A2256W000409', start_s_m=50., end_s_m=100., longitudinal_margin_m=0.))
        zone_list = competition.zones(context, competition.settings(context))
        points = self.loader.get_link_points('A2256W000409')
        for s, expected in ((30., None), (75., 'test_zone'), (120., None)):
            x, y, _, _ = interpolate_on_polyline(points, s)
            self.assertEqual(competition.zone_containing(zone_list, x, y), expected)

    def test_initial_and_supply_slots_inside_a_zone_are_rejected(self):
        # the right-lane supply slot is on A2256W000157; an initial slot lies on A2256W000409
        for link in ('A2256W000868', 'A2256W000168'):
            with self.subTest(link=link), self.assertRaisesRegex(ValueError, 'no-spawn zone'):
                competition.build_plan(self.context(config=self.zone_config(link)))
        for link in ('A2256W000409', 'A2256W000420'):
            context = self.context(config=self.zone_config(link))
            with self.subTest(link=link), self.assertRaisesRegex(ValueError, 'no-spawn zone'):
                competition.build_plan(context)

    def test_a_zone_elsewhere_does_not_block_spawning(self):
        context = self.prepared(config=self.zone_config('A2256W000846'))
        self.assertEqual(len(context.highway_npcs), 38)
        self.assertEqual(context.highway_competition['zones'][0]['name'], 'test_zone')

    def test_passing_through_a_zone_is_allowed(self):
        context = self.prewarmed(config=self.zone_config('A2256W000420', start_s_m=5., end_s_m=25.))
        self.run_traffic(context, 60)
        self.assertEqual(self.report(context)['no_spawn_zones'], ['test_zone'])
        self.assertTrue(any(n['lane'] == 'merge_adjacent' for n in context.highway_npcs))


if __name__ == '__main__':
    unittest.main()
