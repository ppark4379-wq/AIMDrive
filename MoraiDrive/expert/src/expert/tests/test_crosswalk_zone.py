import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from expert.src.config.config import Config
from expert.src.localization.path_manager import PathManager
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.adaptive_cruise_control import AdaptiveCruiseControl
from expert.src.planning.traffic_light_stop import TrafficLightStopController


class CrosswalkZoneTests(unittest.TestCase):
    def setUp(self):
        config_file = Path(__file__).parents[1]/'src/config/config.json'
        self.config = json.loads(config_file.read_text(encoding='utf-8'))
        self.zone = next(z for z in self.config['planning']['velocity_profile']['speed_limit_zones']
                         if z['name'] == 'north_crosswalk_pair')
        self.path = [Point(-60., -100.+i*.25) for i in range(601)]
        self.manager = PathManager(self.path, False, 50)
        self.manager.set_velocity_profile(60., .15, 50, [self.zone])

    def acc(self, zones=True):
        return AdaptiveCruiseControl(vehicle_length=4.635,
                                     time_gap_zones=[self.zone] if zones else [],
                                     **self.config['planning']['adaptive_cruise_control'])

    def index(self, y):
        return self.manager._nearest_path_index((-60., y))

    def test_coordinate_zone_resolves_without_link_ids(self):
        config = object.__new__(Config)
        config.planning = {'velocity_profile': {'speed_limit_zones': [copy.deepcopy(self.zone)]}}
        with patch('expert.src.config.config.io.open', return_value=io.StringIO('[]')):
            config._resolve_speed_limit_zones('unused')
        self.assertEqual(self.zone, config.planning['velocity_profile']['speed_limit_zones'][0])

    def test_entire_pair_and_body_clearance_keep_twenty(self):
        for y in (-40.5, -35.85, -30., -20., -12., -8.98, -4.5):
            with self.subTest(y=y):
                self.assertLessEqual(self.manager.velocity_profile[self.index(y)], 20./3.6)
        self.assertAlmostEqual(self.manager.speed_limit_profile[self.index(-2.)], 60./3.6)
        self.assertAlmostEqual(self.manager.speed_limit_profile[self.index(-80.)], 60./3.6)

    def test_approach_is_braked_before_entry(self):
        first = self.index(-40.5)
        self.assertLess(self.manager.velocity_profile[first-10], 60./3.6)
        for i in range(first-100, first):
            a, b = self.manager.velocity_profile[i:i+2]
            ds = self.path[i].distance(self.path[i+1])
            self.assertLessEqual(a*a, b*b+2.*3.*ds+1e-8)

    def test_green_and_missing_signal_do_not_remove_zone_cap(self):
        class EmptyMapper(dict):
            def mapped_signal_ids(self):
                return []
        controller = TrafficLightStopController(self.path, EmptyMapper(), False)
        for y in (-35., -20., -10., -5.):
            idx = self.index(y)
            planned = self.manager.velocity_profile[idx]
            for signal in (None, ['test', 16], ['test', 48]):
                self.assertLessEqual(controller.get_target_velocity(signal, idx, planned), 20./3.6)

    def test_only_objects_in_zone_get_two_second_gap(self):
        acc = self.acc()
        for kind in (0, 1, 2):
            for x, y, expected in ((-60., -30., 2.), (-60., -12., 2.),
                                    (-60., -50., .8), (-60., 0., .8), (-40., -20., .8)):
                with self.subTest(kind=kind, x=x, y=y):
                    self.assertEqual(acc._time_gap_for_object(ObjectInfo(x, y, 0., kind)), expected)
        self.assertEqual(acc._time_gap_for_object(ObjectInfo(-60., -20., 0., 3)), .8)
        self.assertEqual(acc.time_gap, .8)

    def test_gap_restores_when_selected_object_changes_or_disappears(self):
        acc = self.acc()
        for y, expected in ((-15., 2.), (0., .8), (-20., 2.)):
            obj = ObjectInfo(-60., y, 0., 0)
            acc.check_object(self.path, [{'object_info': obj, 'local_position': Point(24., 0.)}], None)
            self.assertEqual(acc.object_time_gap, expected)
        acc.check_object(self.path, [], None)
        self.assertEqual(acc.object_time_gap, .8)
        self.assertEqual(acc.get_target_velocity(5., 6.), 6.)

    def test_second_crosswalk_pedestrian_reduces_target_earlier(self):
        obj = ObjectInfo(-60., -15., 0., 0)
        entry = {'object_info': obj, 'local_position': Point(24., 0.)}
        zoned, baseline = self.acc(), self.acc(False)
        for acc in (zoned, baseline):
            acc.check_object(self.path, [entry], None)
        self.assertLess(zoned.get_target_velocity(20./3.6, 20./3.6),
                        baseline.get_target_velocity(20./3.6, 20./3.6))
        entry['local_position'] = Point(8., 0.)
        zoned.check_object(self.path, [entry], None)
        self.assertEqual(zoned.get_target_velocity(20./3.6, 20./3.6), 0.)

    def test_speed_zone_buffers_wrap_on_closed_path(self):
        path = ([Point(0., y) for y in range(10)] + [Point(x, 10.) for x in range(10)]
                + [Point(10., y) for y in range(10, 0, -1)] + [Point(x, 0.) for x in range(10, 0, -1)])
        manager = PathManager(path, True, 10)
        zone = dict(start_point=[0., 1.], end_point=[0., 3.], max_velocity=20.,
                    entry_buffer_m=3., exit_buffer_m=2.)
        manager.set_velocity_profile(60., .15, 3, [zone])
        for i in (38, 39, 0, 1, 2, 3, 4, 5):
            self.assertAlmostEqual(manager.speed_limit_profile[i], 20./3.6)
        self.assertAlmostEqual(manager.speed_limit_profile[20], 60./3.6)


if __name__ == '__main__':
    unittest.main()
