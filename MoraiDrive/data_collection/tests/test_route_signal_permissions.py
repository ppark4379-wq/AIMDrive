"""Protected-left GT stays route-specific without changing legacy light labels."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / 'data_collection/internal_sensor_record'),
                str(PROJECT / 'data_collection'), str(PROJECT)]
from async_capture.route_signal_binding import bind_route_progress
from tools.audit_sensor_record_route_signals import apply_route_agreement
from mgeo_navigation import MGeoMap, MGeoRouteTracker
from morai_pretraining.sensor_record_manifest import traffic_signal_metadata


class RouteSignalPermissionTests(unittest.TestCase):
    def progress(self, movement='left', **changes):
        fields = dict(route_index=0, route_link_index=0, current_link_id='approach',
                      distance_to_route_m=0., affected_light_id='signal',
                      distance_to_stopline_m=12., signal_movement=movement,
                      signal_route_link_id='movement', signal_stop_node_id='stop')
        fields.update(changes)
        return SimpleNamespace(**fields)

    def bind(self, color, movement='left', **changes):
        values = [dict(id='signal', color=color, source_record_ids=[2, 3],
                       source_bracket_ns=[900, 1100])]
        return bind_route_progress(self.progress(movement, **changes), 0., 0., values, ['signal'])

    def test_same_straight_green_stops_left_but_permits_straight(self):
        left, straight = self.bind(16), self.bind(16, 'straight')
        self.assertFalse(left['signal_permission'])
        self.assertTrue(straight['signal_permission'])
        # Existing two-class visual-head targets are not reinterpreted as stop/go.
        self.assertEqual((left['training_label'], straight['training_label']), (1, 1))

    def test_arrow_and_combined_phases_have_independent_permission_supervision(self):
        for color in (32, 33, 48):
            with self.subTest(color=color):
                left, straight = self.bind(color), self.bind(color, 'straight')
                self.assertTrue(left['permission_valid'])
                self.assertTrue(left['signal_permission'])
                self.assertEqual(straight['signal_permission'], color == 48)
                self.assertEqual(left['raw_signal_bits'], color)
                self.assertFalse(left['valid'])
                self.assertEqual(left['training_label'], -1)
                self.assertEqual(left['light_source_record_ids'], [2, 3])

    def test_unsupported_phase_unknown_movement_and_distant_light_stay_masked(self):
        for color in (0, 4, 5, 17, 20, 36, 255):
            self.assertFalse(self.bind(color)['permission_valid'])
            self.assertIsNone(self.bind(color)['signal_permission'])
        self.assertFalse(self.bind(16, 'right_unprotected')['permission_valid'])
        self.assertTrue(self.bind(16, None)['valid'])
        distant = self.bind(32, distance_to_stopline_m=46.)
        self.assertFalse(distant['permission_valid'])
        self.assertEqual(distant['reason'], 'outside_signal_lookahead')

    def test_arrow_permission_does_not_bypass_rpc_agreement_or_reverse_progress(self):
        for links, index, last, reason in (
            ({'other'}, 5, 4, 'route_and_rpc_current_link_disagree'),
            ({'approach', 'other'}, 5, 4, 'route_and_rpc_current_link_disagree'),
            ({'approach'}, 3, 4, 'route_progress_reversed'),
        ):
            row = self.bind(33)
            apply_route_agreement(row, 'approach', links, index, last)
            self.assertFalse(row['permission_valid'])
            self.assertIsNone(row['signal_permission'])
            self.assertEqual(row['reason'], reason)
        row = self.bind(33)
        apply_route_agreement(row, 'approach', {'approach'}, 5, 4)
        self.assertTrue(row['permission_valid'])

    def test_publication_requires_new_admission_and_source_time_transport(self):
        row = self.bind(33)
        admission = dict(traffic_light_gt_transport_timing_candidate=True,
                         route_signal_permission_candidate=True)
        published = traffic_signal_metadata(row, admission)
        self.assertTrue(published['permission_valid'])
        self.assertTrue(published['signal_permission'])
        for denied in ({}, dict(admission, traffic_light_gt_transport_timing_candidate=False),
                       dict(admission, route_signal_permission_candidate=False)):
            masked = traffic_signal_metadata(row, denied)
            self.assertFalse(masked['permission_valid'])
            self.assertIsNone(masked['signal_permission'])
            self.assertEqual(masked['raw_signal_bits'], 33)
        old = traffic_signal_metadata(dict(color=16, valid=True, training_label=1), admission)
        self.assertFalse(old['permission_valid'])

    def test_exact_stopline_binds_following_connected_link_not_approach(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'link_set.json'
            nodes = Path(root) / 'node_set.json'
            links = [dict(idx='approach', from_node_idx='start', to_node_idx='stop',
                          related_signal='straight', points=[[0, 0, 0], [10, 0, 0]]),
                     dict(idx='left', from_node_idx='stop', to_node_idx='end',
                          related_signal='left', points=[[10, 0, 0], [10, 10, 0]])]
            path.write_text(json.dumps(links))
            nodes.write_text(json.dumps([dict(idx='stop', on_stop_line=True, traffic_light_id='signal')]))
            world = MGeoMap(str(path), node_set_path=str(nodes))
            route = dict(nav_link_ids=['approach', 'left'])
            progress = MGeoRouteTracker(world, route).update(0, 0, 0)
            self.assertEqual(progress.signal_movement, 'left')
            self.assertEqual(progress.signal_route_link_id, 'left')
            self.assertEqual(progress.signal_stop_node_id, 'stop')
            world.links['left']['from_node_idx'] = 'unconnected'
            progress = MGeoRouteTracker(world, route).update(0, 0, 0)
            self.assertIsNone(progress.signal_movement)
            self.assertIsNone(progress.signal_route_link_id)
            world.links['left']['from_node_idx'] = 'stop'
            world.links['left']['related_signal'] = 'right_unprotected'
            self.assertIsNone(MGeoRouteTracker(world, route).update(0, 0, 0).signal_movement)


if __name__ == '__main__':
    unittest.main()
