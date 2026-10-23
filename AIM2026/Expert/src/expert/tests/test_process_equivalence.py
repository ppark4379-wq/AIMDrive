"""Recorded process-boundary regressions with the real persistent controller.

Section results are deterministic fixtures, not a claim of full road replay.
The expected control/state sequence was recorded from the pre-refactor driver.
"""
import contextlib
import copy
import io
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'AIM2026/Expert/src'))

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.interfaces.geometry import Point
from expert.src.interfaces.perception_data import ObjectInfo
from expert.src.interfaces.vehicle_state import VehicleState
from expert.src.planning.common.object_selection import ForwardObjectDetector
from expert.src.planning.lattice.structures import LatticePlanningResult


class Reference:
    current_waypoint = 0
    is_closed_path = False

    def __init__(self, owner, events):
        self.owner, self.events = owner, events

    def get_local_path(self, vehicle):
        self.events.append('reference')
        return ([] if self.case.get('missing_reference') else self.owner.path), self.case.get('planned', 10.)

    def get_velocity_without_curvature(self):
        self.events.append('curvature_free_speed')
        return 15.


class Detector(ForwardObjectDetector):
    def __init__(self, events):
        super().__init__([])
        self.events = events

    def detect_object(self, vehicle, forward_only=True):
        self.events.append('front_objects' if forward_only else 'all_objects')
        return super().detect_object(vehicle, forward_only)


class Section:
    max_speed = 12.

    def __init__(self, events):
        self.events = events
        self.roundabout_planner = SimpleNamespace(active=False)
        self.highway_lane_planner = SimpleNamespace(set_control_dt=lambda dt: events.append('runtime'))
        self.static_avoidance = SimpleNamespace(active=False)

    def _is_inside_static_zone(self, vehicle):
        return False

    def _roundabout_entry_state(self, vehicle):
        return {'in_approach': False}

    def roundabout_entry_should_ignore_legacy_acc(self, vehicle, obj):
        return False

    def run(self, vehicle, path, objects, *args, **kwargs):
        self.events.append('section')
        self.received = objects
        self.timing = kwargs
        return self.result


class Cruise:
    pedestrian_released = False

    def __init__(self, events):
        self.events = events

    def check_object(self, path, objects, signal, **kwargs):
        self.events.append('acc_objects')
        self.received = objects
        check = kwargs.get('pedestrian_conflict_check')
        self.person_decisions = ([check(entry['object_info']) for entry in objects] if check else [])

    def get_target_velocity(self, actual, planned):
        self.events.append('acc_speed')
        return min(planned, self.limit)


class Signal:
    def __init__(self, events):
        self.events = events

    def get_target_velocity(self, **kwargs):
        self.events.append('signal_speed')
        return min(kwargs['planned_velocity'], self.limit)


def make_driver(driver_class=AutonomousDriving, ve=False):
    driver = driver_class.__new__(driver_class)
    driver.events = []
    config = json.loads((ROOT / 'AIM2026/Expert/src/expert/src/config/config.json').read_text())
    driver._init_control(config)
    driver.perception_source = 've' if ve else 'ros'
    driver.path_manager = Reference(driver, driver.events)
    driver.forward_object_detector = Detector(driver.events)
    driver.adaptive_cruise_control = Cruise(driver.events)
    driver.traffic_light_stop_controller = Signal(driver.events)
    driver.lattice_planning_process = Section(driver.events)
    driver._merge_curvature_floor_zone = None
    driver._merge_curvature_floor_speed = None
    driver._static_reference_speed = None
    driver._static_speed_ramp = False
    original = driver._calculate_control

    def control(*args, **kwargs):
        driver.events.append('control')
        return original(*args, **kwargs)

    driver._calculate_control = control
    return driver


def cases():
    recorded = {'static_avoidance': {'path_validated': True, 'planner': 'recorded_route_graph',
                                    'preview_curvature': .03, 'preplanned_complete': True}}
    return [
        {'name': 'initial', 'speed': 8., 'planned': 12.},
        {'name': 'static_follow', 'mode': 'static_obstacle', 'active': True, 'limit': 6., 'details': recorded},
        {'name': 'static_held_reference', 'mode': 'static_obstacle', 'active': True, 'limit': 9.,
         'planned': 0., 'missing_reference': True, 'details': recorded},
        {'name': 'static_stop', 'mode': 'static_obstacle', 'active': True, 'limit': 0., 'details': recorded},
        {'name': 'static_rejoin', 'reason': 'static_recorded_rejoined', 'dt': .1},
        {'name': 'highway_escape', 'mode': 'highway_lane', 'state': 'CHANGE_LEFT', 'speed': 0., 'limit': 5.,
         'details': {'highway_static_escape': {'compact': True, 'verified': True, 'speed_limit': 5.}}},
        {'name': 'merge_latch', 'mode': 'highway_lane', 'speed': 6., 'planned': 2.,
         'details': {'merge_zone_priorities': {'FIRST_PREDICTIVE': 'EGO'}}},
        {'name': 'merge_hold', 'mode': 'highway_lane', 'speed': 2., 'planned': 1.,
         'details': {'merge_zone_priorities': {'FIRST_PREDICTIVE': 'EGO'}}},
        {'name': 'merge_signal_stop', 'mode': 'highway_lane', 'signal': 0.,
         'details': {'merge_zone_priorities': {'FIRST_PREDICTIVE': 'EGO'}}},
        {'name': 'merge_release', 'mode': 'highway_lane', 'speed': 0., 'planned': 2., 'dt': .5,
         'details': {'merge_safety': {'zone': 'SECOND_ENTRY', 'decision': 'GO', 'launch_verified': True,
                                    'launch_target': 6., 'speed_floor': 6.}}},
        {'name': 'roundabout_stop', 'mode': 'roundabout_entry', 'limit': 0., 'sync': True, 'dt': .02},
        {'name': 'roundabout_release', 'mode': 'roundabout_entry', 'reason': 'roundabout_entry_cruise',
         'speed': 0., 'limit': 5., 'sync': True, 'dt': .02,
         'details': {'roundabout_entry': {'phase': 'cruise', 'entering': True}}},
        {'name': 'second_merge_bypass', 'position': (71.5, -130.), 'yaw': -math.pi/2, 'planned': 1.},
    ]


def apply_case(driver, case, index, control_tick=None):
    driver.events.clear()
    x, y = case.get('position', (0., 0.))
    yaw = case.get('yaw', 0.)
    driver.path = [Point(x + math.cos(yaw)*n - math.sin(yaw)*.05*n,
                         y + math.sin(yaw)*n + math.cos(yaw)*.05*n) for n in range(160)]
    driver.path_manager.case = case
    driver.adaptive_cruise_control.limit = case.get('acc', float('inf'))
    driver.adaptive_cruise_control.pedestrian_released = case.get('pedestrian_released', False)
    driver.traffic_light_stop_controller.limit = case.get('signal', float('inf'))
    driver.lattice_planning_process.result = LatticePlanningResult(
        [] if case.get('no_path') else driver.path, mode=case.get('mode', 'inactive'),
        active=case.get('active', False), reason=case.get('reason', case['name']),
        highway_state=case.get('state', 'OFF'), acc_target_velocity=case.get('limit', float('inf')),
        debug_details=copy.deepcopy(case.get('details', {})))
    vehicle = VehicleState(x, y, yaw, case.get('speed', 3.))
    types = case.get('object_types', (1, 1))
    objects = [ObjectInfo(x+5., y, 0., types[0], is_static=True, unique_id=1),
               ObjectInfo(x-5., y, 3., types[1], unique_id=2)]
    timing = dict(sync_mode=case.get('sync', False), control_dt=case.get('dt', 1./30.),
                  ego_header_seq=index, ego_source_ns=index*20_000_000)
    command, path = (driver.execute(vehicle, objects, None, **timing) if control_tick is None
                     else control_tick(driver, vehicle, objects, None, **timing))
    return command, path, objects


def replay(driver_class=AutonomousDriving, control_tick=None):
    driver = make_driver(driver_class)
    result = []
    with contextlib.redirect_stdout(io.StringIO()):
        for index, case in enumerate(cases()):
            command, path, objects = apply_case(driver, case, index, control_tick)
            section = driver.lattice_planning_process
            assert path is section.result.path
            assert section.received[0]['object_info'] is objects[0]
            assert section.received[1]['object_info'] is objects[1]
            assert section.timing['ego_header_seq'] == index
            assert section.timing['ego_source_ns'] == index*20_000_000
            result.append(dict(name=case['name'], accel=float(command.accel), brake=float(command.brake),
                               steering=float(command.front_steer), target=driver.previous_target_velocity,
                               dt=driver.control_dt, pid_integral=driver.pid.integral_error,
                               ramp=driver._static_speed_ramp, merge_speed=driver._merge_curvature_floor_speed,
                               debug_counter=driver._debug_counter, events=list(driver.events)))
    return result


class ProcessBoundaryTests(unittest.TestCase):
    def test_persistent_control_and_state_match_pre_refactor_sequence(self):
        expected = json.loads((Path(__file__).parent/'fixtures/process_sequence.json').read_text())
        actual = replay()
        self.assertEqual(len(actual), len(expected))
        for got, wanted in zip(actual, expected):
            with self.subTest(tick=wanted['name']):
                for key in ('accel','brake','steering','target','dt','pid_integral'):
                    self.assertAlmostEqual(got[key], wanted[key], places=12, msg=key)
                for key in ('name','ramp','merge_speed','debug_counter','events'):
                    self.assertEqual(got[key], wanted[key], key)

    def test_main_tick_and_compatibility_execute_keep_identical_flow(self):
        spec = importlib.util.spec_from_file_location('expert_entry_for_test', ROOT/'AIM2026/Expert/main.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        self.assertEqual(replay(control_tick=entry.on_control_tick), replay())

    def test_jamming_person_identity_survives_filter_and_speed_helpers(self):
        driver = make_driver()
        seen = []

        def conflict(path, vehicle, person, speed):
            seen.append(person)
            return None if person.unique_id == 1 else {'time': .1, 'object_id': 2}

        case = dict(name='jamming_person', mode='static_obstacle', active=True,
                    object_types=(0, 0), details={'jamming': True,
                        'static_avoidance': {'path_validated': True, 'planner': 'recorded_route_graph'}})
        with patch('expert.src.planning.jamming.pedestrian_speed.pedestrian_conflict', side_effect=conflict), \
                contextlib.redirect_stdout(io.StringIO()):
            command, _, objects = apply_case(driver, case, 0)
        self.assertIs(seen[0], objects[0])
        self.assertIs(seen[1], objects[1])
        self.assertIs(driver.adaptive_cruise_control.received[0]['object_info'], objects[0])
        self.assertEqual(driver.adaptive_cruise_control.person_decisions, [False])
        self.assertTrue(driver.lattice_planning_process.result.debug_details['jamming_pedestrian']['emergency'])
        self.assertEqual(driver.previous_target_velocity, 0.)
        self.assertEqual(command.brake, 1.)

    def test_ve_force_brake_retains_steering_and_skips_path_follower(self):
        driver = make_driver(ve=True)
        cases_ve = [dict(name='ve_valid'), dict(name='ve_empty', no_path=True),
                    dict(name='ve_blocked', details={'ve_perception': {'force_brake': True}}),
                    dict(name='ve_recovered')]
        with contextlib.redirect_stdout(io.StringIO()):
            for index, case in enumerate(cases_ve):
                previous_steer = driver.previous_front_steer_rad
                command, path, _ = apply_case(driver, case, index)
                if index in (1, 2):
                    self.assertEqual(path, [])
                    self.assertEqual(command.brake, 1.)
                    self.assertEqual(command.front_steer, previous_steer)
                    self.assertEqual(driver.pid.integral_error, 0.)
                    self.assertEqual(driver.previous_target_velocity, 0.)
                    self.assertNotIn('control', driver.events)
                else:
                    self.assertIn('control', driver.events)
                self.assertEqual(driver.events[:4], ['runtime','reference','section','signal_speed'])
                self.assertFalse(hasattr(driver, '_debug_counter'))


if __name__ == '__main__':
    unittest.main()
