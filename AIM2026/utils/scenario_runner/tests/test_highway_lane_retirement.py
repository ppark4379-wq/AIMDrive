"""Lane-drop deletion runs independently of merge completion and never deletes on missing telemetry."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts import highway_traffic as traffic, highway_competition as competition


class LaneRetirementTest(unittest.TestCase):
    def context(self):
        actor = object()
        item = dict(actor=actor, actor_id='npc', lane='middle', label='bus')
        lane = dict(name='middle', route=['lane'], retire_at_route_end=True,
                    lane_end_margin_m=30., offsets_m=[0.])
        return SimpleNamespace(
            cfg={'highway_npc': {'managed_traffic': {'lanes': [lane]}}},
            highway_traffic_cleared=False, highway_spawn_points_active=True,
            highway_npcs=[item], scenario_actors=[actor],
            map_loader=SimpleNamespace(get_link_points=lambda _: [(float(i), 0., 0.) for i in range(101)]),
            grpc=Mock(destroy_vehicle_by_id=Mock(return_value=True)))

    def update(self, context, samples):
        with patch.object(competition, '_states', return_value=samples), patch.object(traffic, '_persist'):
            traffic.update_lane_lifecycle(context, 10.)

    def test_lane_end_deletes_actor_and_records_reason(self):
        context = self.context()
        self.update(context, {'npc': dict(x=71., y=0.)})
        context.grpc.destroy_vehicle_by_id.assert_called_once_with('npc')
        self.assertEqual(context.highway_npcs, [])
        self.assertEqual(context.scenario_actors, [])
        self.assertEqual(context.highway_lane_removals[0]['reason'], 'lane_end')

    def test_upstream_and_other_lane_continue(self):
        for x, retire in ((69., True), (90., False)):
            context = self.context()
            context.cfg['highway_npc']['managed_traffic']['lanes'][0]['retire_at_route_end'] = retire
            self.update(context, {'npc': dict(x=x, y=0.)})
            context.grpc.destroy_vehicle_by_id.assert_not_called()

    def test_skipped_end_window_still_deletes(self):
        context = self.context()
        self.update(context, {'npc': dict(x=120., y=0.)})
        self.assertEqual(context.highway_lane_removals[0]['reason'], 'lane_end')

    def test_off_route_actor_is_removed(self):
        context = self.context()
        self.update(context, {'npc': dict(x=30., y=2.)})
        self.assertEqual(context.highway_lane_removals[0]['reason'], 'lane_departure')

    def test_missing_state_does_not_delete(self):
        context = self.context()
        self.update(context, {})
        self.update(context, {'other': dict(x=90., y=0.)})
        context.grpc.destroy_vehicle_by_id.assert_not_called()

    def test_failed_delete_stops_instead_of_forgetting_actor(self):
        context = self.context()
        context.grpc.destroy_vehicle_by_id.return_value = False
        with self.assertRaises(RuntimeError):
            self.update(context, {'npc': dict(x=80., y=0.)})
        self.assertEqual(len(context.highway_npcs), 1)
        self.assertEqual(len(context.scenario_actors), 1)

    def test_competition_removes_lane_departure(self):
        context = self.context()
        context.highway_npcs[0].update(x_now=30., lane_hint=30, lane_error_m=2.)
        state = dict(cfg=dict(removal={}, lane_keeping={'max_lateral_error_m': 1.9}),
                     lanes={'middle': {'total': 100., 'cum': list(range(101))}})
        with patch.object(competition, '_remove') as remove:
            competition._remove_finished(context, state, None, 10.)
        self.assertEqual(remove.call_args.args[3], 'lane_departure')


if __name__ == '__main__':
    unittest.main()
