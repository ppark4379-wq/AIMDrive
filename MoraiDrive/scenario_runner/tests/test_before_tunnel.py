import copy
from types import SimpleNamespace
import unittest

from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
from scenario_runner.scenarios.full_loop_parts.before_tunnel import apply
from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.adapters.ve_signal import build_signal_targets


class BeforeTunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.global_cfg, _, cls.config = load_configuration()
        cls.loader = MGeoMapLoader(cls.global_cfg['paths']['mgeo_root'])

    def context(self, shortened):
        cfg = copy.deepcopy(self.config)
        cfg.update(route_debug_enabled=False, ve_stop_before_tunnel=shortened)
        c = FullLoopScenario(SimpleNamespace(make_transform=lambda *args: args),
                             self.loader, self.global_cfg, cfg)
        c.route_definition = c.load_route_definition()
        c.prepare_full_loop_route()
        return c

    def test_actual_route_ends_at_highway_exit_before_tunnel(self):
        full = self.context(False)
        short = self.context(True)
        self.assertEqual(full.route_points[:3530], short.route_points)
        self.assertEqual(full.route_points, short._full_route_points)
        self.assertEqual(full.route_points[3529][:3], (short.goal_x, short.goal_y, short.goal_z))
        self.assertAlmostEqual(short.route_length_m, full.highway_exit_s)
        self.assertLess(short.route_length_m, short.tunnel_entry_s)
        self.assertEqual(full.highway_activation_s, short.highway_activation_s)
        self.assertEqual(short.end_link, short.route_links[-1])
        self.assertFalse(short.cfg['expert_is_closed_path'])
        for key in ('tunnel_sudden_brake', 'tunnel_dynamic', 'tunnel_vehicle_dynamic'):
            self.assertFalse(short.cfg[key]['enabled'])
        self.assertEqual(full.cfg['highway_npc'], short.cfg['highway_npc'])
        self.assertEqual(full.cfg['static_obstacle_models'], short.cfg['static_obstacle_models'])
        self.assertEqual(4430, len(full.route_points))

    def test_endpoint_cannot_be_inside_tunnel(self):
        context = self.context(False)
        definition = copy.deepcopy(context.route_definition)
        for checkpoint in definition['checkpoints']:
            if str(checkpoint['id']) == '13':
                checkpoint['path_index'] = 3900
        context.route_definition = definition
        with self.assertRaises(ValueError):
            apply(context)

    def test_signal_binding_accepts_open_prefix_and_excludes_tunnel_signal(self):
        context = self.context(True)
        config = {'map': {'data_dir': self.global_cfg['paths']['mgeo_root'],
                         'route_signal_route_links': context.route_definition['route_links'],
                         'is_closed_path': context.cfg['expert_is_closed_path']}}
        document = build_signal_targets(context, config)
        ids = {target['signal_id'] for target in document['targets']}
        self.assertIn('C1256W000052', ids)
        self.assertIn('C1256W000077', ids)
        self.assertNotIn('LCS01', ids)
        self.assertTrue(all(t['path_index'] < len(context.route_points) for t in document['targets']))
        config['map']['is_closed_path'] = True
        with self.assertRaisesRegex(ValueError, 'does not close'):
            build_signal_targets(context, config)


if __name__ == '__main__':
    unittest.main()
