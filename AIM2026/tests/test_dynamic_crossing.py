"""Crossing safety before a classless object's center reaches the ACC gate."""
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.common.adaptive_cruise_control import AdaptiveCruiseControl
from AIM2026.Expert.src.planning.common.dynamic_crossing import DynamicCrossingGuard
from AIM2026.Expert.src.planning.common.reference_path import PathManager
from AIM2026.Expert.src.planning.common.target_speed import target_speed
from AIM2026.Expert.src.planning.lattice.structures import LatticePlanningResult


PATH = [Point(x, 0.) for x in range(61)]


def object_at(x=16., y=8., speed=2.5, heading=-90., kind=-1, identity=9):
    obj = ObjectInfo(x, y, speed, kind, size_x=.8, size_y=.6,
                     heading_deg=heading, unique_id=identity,
                     name='ve_tracked_unknown:moving')
    return dict(object_info=obj, local_position=Point(x, y))


def run(guard, entries, x=0., speed=5.5, now=0., path=PATH):
    return guard.evaluate(path, VehicleState(x, 0., 0., speed), entries, 60/3.6, now=now)


@pytest.mark.parametrize('kind', [-1, 0, 1, 2])
def test_crossing_class_does_not_change_early_braking(kind):
    entry = object_at(kind=kind)
    guard = DynamicCrossingGuard()
    target, details = run(guard, [entry])
    assert 0. < target < 5.5
    assert details['object_id'] == 9
    assert details['lateral_m'] > 2.5
    assert 0. < details['entry_time_s'] < 6.
    acc = AdaptiveCruiseControl(.5, 1., .8, 4.635)
    acc.check_object(PATH, [entry], None)
    if kind in (-1, 1, 2):
        assert acc.object_type is None  # The incident's original exclusion.


@pytest.mark.parametrize('entry', [
    object_at(y=5., speed=8., heading=0.),  # Adjacent parallel traffic.
    object_at(y=8., heading=90.),          # Moving away from our path.
    object_at(x=-8.),                      # Crossing behind the vehicle.
    object_at(y=20., speed=1.),            # Outside prediction horizon.
    object_at(speed=0.),                   # Stationary roadside object.
    object_at(kind=3),                     # A signal is not a moving body.
])
def test_non_conflicting_objects_do_not_stop_ego(entry):
    assert math.isinf(run(DynamicCrossingGuard(), [entry])[0])


def test_unknown_speed_bound_is_not_used_as_observed_crossing_velocity():
    entry = object_at()
    entry['object_info'].speed_unknown = True
    assert math.isinf(run(DynamicCrossingGuard(), [entry])[0])


def test_stopped_ego_waits_until_body_clears_then_can_restart():
    guard = DynamicCrossingGuard()
    assert run(guard, [object_at(x=8.5, y=4.)], speed=0.)[0] == 0.
    assert run(guard, [object_at(x=8.5, y=0., speed=0.)], speed=0., now=.5)[0] == 0.
    # Clear on the far side, but hold briefly to suppress single-frame chatter.
    assert run(guard, [object_at(x=8.5, y=-4.)], speed=0., now=.6)[0] == 0.
    assert math.isinf(run(guard, [object_at(x=8.5, y=-4.)], speed=0., now=1.2)[0])


def test_stopped_ego_can_approach_distant_crossing_with_stopping_reserve():
    guard = DynamicCrossingGuard()
    limit, details = run(guard, [object_at(x=22.)], speed=0.)
    assert limit == pytest.approx(2.)
    assert not details['clears_before_arrival']
    assert limit*.6+limit**2/(2*2.5) <= details['front_gap_m']-4.


def test_crossing_exit_before_fastest_arrival_allows_acceleration_but_keeps_stop_reserve():
    guard = DynamicCrossingGuard()
    ego = VehicleState(0., 0., 0., 1.)
    limit, details = guard.evaluate(PATH, ego, [object_at(x=40., y=2.)], 30/3.6, now=0.)
    assert details['exit_time_s']+guard.time_margin < details['earliest_arrival_s']
    assert details['clears_before_arrival']
    assert limit == pytest.approx(30/3.6)
    # This remains a speed ceiling at which we can stop if the object stops.
    assert limit*.6+limit**2/(2*2.5) <= details['front_gap_m']-4.


def test_stationary_crossing_never_uses_horizon_end_as_clearance():
    guard = DynamicCrossingGuard()
    run(guard, [object_at(x=40., y=2.)], speed=0.)
    limit, details = run(guard, [object_at(x=40., y=0., speed=0.)], speed=0., now=.1)
    assert limit == pytest.approx(2.)
    assert details['exit_time_s'] is None
    assert not details['clears_before_arrival']


def test_dropout_and_direction_noise_do_not_immediately_release_brake():
    guard = DynamicCrossingGuard()
    initial = run(guard, [object_at()])[0]
    limit, details = run(guard, [], x=1., now=.2)
    assert limit < initial
    assert details['state'] == 'clearance_hold'
    assert not math.isinf(run(guard, [object_at(heading=0.)], now=.4)[0])
    assert math.isinf(run(guard, [], now=.7)[0])


def test_turning_into_parallel_lead_hands_back_to_following_acc():
    guard = DynamicCrossingGuard()
    run(guard, [object_at()])
    assert math.isinf(run(guard, [object_at(y=0., speed=8., heading=0.)], now=1.)[0])


def test_large_body_and_reverse_side_crossing_are_included():
    a, b = object_at(y=8., heading=-90.), object_at(y=-8., heading=90.)
    assert run(DynamicCrossingGuard(), [a])[0] == pytest.approx(run(DynamicCrossingGuard(), [b])[0])
    b['object_info'].size_y = 4.
    assert run(DynamicCrossingGuard(), [b])[0] < run(DynamicCrossingGuard(), [a])[0]


def test_commanded_path_and_not_world_axis_defines_crossing():
    path = [Point(0., x) for x in range(61)]
    entry = object_at(x=-8., y=16., heading=0.)
    target, _ = DynamicCrossingGuard().evaluate(
        path, VehicleState(0., 0., math.pi/2, 5.5), [entry], 60/3.6, now=0.)
    assert target == pytest.approx(run(DynamicCrossingGuard(), [object_at()])[0])
    # A new commanded lane avoids the crossing; old-path hold must not stop it.
    guard = DynamicCrossingGuard()
    run(guard, [object_at()])
    assert math.isinf(run(guard, [], now=.1, path=[Point(x, -10.) for x in range(61)])[0])


def test_expanded_zone_holds_thirty_through_last_crossing():
    config = json.loads((Path(__file__).parents[1]/'Expert/src/config/config.json').read_text())
    config['planning']['velocity_profile']['speed_limit_zones'] = [
        z for z in config['planning']['velocity_profile']['speed_limit_zones'] if 'start_point' in z]
    manager = PathManager([Point(-60.3, y) for y in range(-80, 61)], False, 50)
    manager.set_velocity_profile(**config['planning']['velocity_profile'])
    for y in (-37., -4., 0., 6., 20., 24.):
        _, target = manager.get_local_path(VehicleState(-60.3, y, math.pi/2, 5.))
        assert target == pytest.approx(30/3.6)
    _, target = manager.get_local_path(VehicleState(-60.3, 35., math.pi/2, 5.))
    assert target > 30/3.6


def speed_owner():
    return NS(
        _merge_ego_curvature_speed_floor=lambda v, *args: v,
        _hold_merge_ego_curvature_speed=lambda v, *args: v,
        _second_merge_slowdown_band=lambda v, *args: v,
        _legacy_acc_entries=lambda ego, result, entries: entries,
        _smooth_static_target_speed=lambda v, *args: v,
        adaptive_cruise_control=AdaptiveCruiseControl(.5, 1., .8, 4.635),
        traffic_light_stop_controller=NS(get_target_velocity=lambda **kw: float('inf')),
        path_manager=NS(current_waypoint=0))


def test_crossing_ceiling_reaches_final_target_even_with_clear_legacy_acc():
    driver = speed_owner()
    result = LatticePlanningResult(PATH, reason='outside_zone')
    target, _ = target_speed(driver, VehicleState(0., 0., 0., 5.5), PATH, PATH, result,
                             60/3.6, False, float('inf'), [object_at()], [object_at()], None)
    assert driver.adaptive_cruise_control.object_type is None
    assert target < 5.5
    assert result.debug_details['dynamic_crossing']['object_id'] == 9
    assert result.debug_details['speed_limits']['final'] == target


@pytest.mark.parametrize('kind', [-1, 0, 1, 2])
def test_crossing_owns_braking_and_releases_legacy_acc_after_body_clearance(kind):
    driver = speed_owner()
    result = LatticePlanningResult(PATH, reason='outside_zone')
    ego = VehicleState(0., 0., 0., 0.)

    def step(y, now, speed=2.5):
        entries = [object_at(x=8.5, y=y, kind=kind, speed=speed)]
        with patch('AIM2026.Expert.src.planning.common.dynamic_crossing.time.monotonic', return_value=now):
            return target_speed(driver, ego, PATH, PATH, result, 30/3.6, False,
                                float('inf'), entries, entries, None)[0]

    assert step(1., 0.) == 0.
    assert driver.adaptive_cruise_control.object_type is None
    assert step(-2.1, .1) == 0.  # Body clear, still inside old 2.5 m ACC gate.
    assert step(-2.1, .7) == pytest.approx(30/3.6)
    assert step(-2.1, .8, speed=0.) == pytest.approx(30/3.6)
    # Turning back toward the road immediately restores the crossing stop.
    entries = [object_at(x=8.5, y=-2.1, kind=kind, heading=90.)]
    assert target_speed(driver, ego, PATH, PATH, result, 30/3.6, False,
                        float('inf'), entries, entries, None)[0] == 0.


def test_parallel_lead_and_stationary_unseen_obstacle_remain_with_acc():
    for entry in [object_at(x=8., y=0., heading=0., speed=1.),
                  object_at(x=8., y=0., speed=0.)]:
        driver = speed_owner()
        result = LatticePlanningResult(PATH, reason='outside_zone')
        assert target_speed(driver, VehicleState(0., 0., 0., 2.), PATH, PATH, result,
                            30/3.6, False, float('inf'), [entry], [entry], None)[0] == 0.
        assert driver.adaptive_cruise_control.object_id == 9


@pytest.mark.parametrize('flag', ['speed_unknown', 'motion_uncertain'])
def test_uncertain_observation_cannot_reuse_acc_clearance(flag):
    guard = DynamicCrossingGuard()
    entries = [object_at(x=8.5, y=-2.1)]
    run(guard, entries)
    assert not guard.following_entries(entries)
    setattr(entries[0]['object_info'], flag, True)
    run(guard, entries, now=.1)
    assert guard.following_entries(entries) == entries


def test_turn_to_parallel_lead_restores_following_acc_during_crossing_hold():
    guard = DynamicCrossingGuard()
    run(guard, [object_at(x=8., y=1.)])
    entries = [object_at(x=8., y=0., heading=0., speed=1.)]
    run(guard, entries, now=.1)
    assert guard.following_entries(entries) == entries


def test_cleared_object_does_not_release_second_crossing_or_signal_stop():
    driver = speed_owner()
    result = LatticePlanningResult(PATH, reason='outside_zone')
    entries = [object_at(x=8., y=-3.), object_at(x=8., y=1., identity=10)]
    args = (driver, VehicleState(0., 0., 0., 0.), PATH, PATH, result,
            30/3.6, False, float('inf'), entries, entries, None)
    assert target_speed(*args)[0] == 0.
    assert result.debug_details['dynamic_crossing']['object_id'] == 10
    driver.traffic_light_stop_controller.get_target_velocity = lambda **kw: 0.
    entries[:] = [object_at(x=40., y=2.)]
    driver.dynamic_crossing_guard.reset()
    assert target_speed(*args)[0] == 0.


@pytest.mark.parametrize('mode, reason', [
    ('highway', 'highway_acc_override'),
    ('highway_lane', 'highway_lane_keep'),
    ('highway_lane', 'highway_lane_match_failed'),
    ('highway_lane', 'highway_right_turn_global_path'),
])
@pytest.mark.parametrize('kind', [-1, 0, 1, 2])
def test_highway_disables_crossing_prediction_and_clears_previous_road_hold(mode, reason, kind):
    driver = speed_owner()
    guard = driver.dynamic_crossing_guard = DynamicCrossingGuard()
    ego = VehicleState(0., 0., 0., 5.5)
    entries = [object_at(kind=kind)]
    # Seed an actual crossing stop before handing control to the highway.
    assert guard.evaluate(PATH, ego, [object_at(x=8.)], 60/3.6)[0] == 0.
    result = LatticePlanningResult(PATH, mode=mode, reason=reason, acc_target_velocity=12.)
    args = (driver, ego, PATH, PATH, result, 60/3.6, False, float('inf'), entries, entries, None)
    with patch.object(guard, 'evaluate', side_effect=AssertionError('Highway must not run this guard')):
        assert target_speed(*args)[0] == 12.
        assert result.debug_details['dynamic_crossing']['state'] == 'disabled_highway'
        assert math.isinf(result.debug_details['speed_limits']['dynamic_crossing'])
        # The existing highway stop and signal stop remain authoritative.
        result.acc_target_velocity = 0.
        assert target_speed(*args)[0] == 0.
        result.acc_target_velocity = 12.
        driver.traffic_light_stop_controller.get_target_velocity = lambda **kw: 0.
        assert target_speed(*args)[0] == 0.
    assert math.isinf(guard.evaluate(PATH, ego, [], 60/3.6)[0])
    # Returning to an ordinary road enables fresh crossing decisions again.
    driver.traffic_light_stop_controller.get_target_velocity = lambda **kw: float('inf')
    result.mode = 'inactive'
    assert target_speed(*args)[0] < ego.velocity


def test_recorded_incident_frames_brake_before_legacy_two_point_five_meter_gate():
    from AIM2026.Expert.adapter import planner_inputs
    from AIM2026.Expert.src.planning.common.object_selection import ForwardObjectDetector
    fixture = json.loads((Path(__file__).parent/'fixtures/dynamic_crossing_track9_20261027.json').read_text())
    path = [Point(*p) for p in fixture['path']]
    for frame in fixture['frames']:
        ego, objects = planner_inputs(frame['packet'], frame['control_pose'])
        detector = ForwardObjectDetector([])
        detector.dynamic_object_list = objects
        entries = detector.detect_object(ego)
        acc = AdaptiveCruiseControl(.5, 1., .8, 4.635)
        acc.check_object(path, entries, None)
        assert acc.object_type is None
        assert frame['recorded_brake'] == 0.
        limit, details = DynamicCrossingGuard().evaluate(path, ego, entries, 30/3.6, now=0.)
        assert details['object_id'] == 9
        assert details['lateral_m'] > 5.
        assert limit < ego.velocity-.2
        if frame['packet']['sequence'] == 212:
            assert limit == 0.


def test_delayed_longitudinal_response_stops_short_and_resumes_after_crossing():
    # A bounded numerical plant, not a substitute for MORAI validation.
    guard = DynamicCrossingGuard()
    x, speed, minimum_gap = 0., 5.5, float('inf')
    commands = []
    for tick in range(160):
        now = tick*.05
        entry = object_at(x=16., y=8.-2.5*now)
        entry['object_info'].size_y = .8
        target, _ = run(guard, [entry], x=x, speed=speed, now=now)
        commands.append(min(5.5, target))
        delayed = commands[max(0, tick-6)]  # 0.3 s control/actuation delay.
        speed = max(0., speed+max(-2.5*.05, min(1.5*.05, delayed-speed)))
        x += speed*.05
        if abs(entry['object_info'].position.y) < 1.9:
            minimum_gap = min(minimum_gap, 16.-.4-x-4.635)
    assert minimum_gap > 3.5
    assert min(commands) == 0.  # A close crossing still requires a full stop.
    assert speed > 3.


@pytest.mark.parametrize('initial_speed', [20/3.6, 30/3.6])
def test_crossing_object_stops_on_path_ego_must_stop_with_delay(initial_speed):
    guard = DynamicCrossingGuard()
    x, speed, minimum_gap = 0., initial_speed, float('inf')
    commands = []
    for tick in range(240):
        now = tick*.05
        y = max(0., 8.-2.5*now)
        entry = object_at(x=32., y=y, speed=2.5 if y > 0. else 0.)
        target, _ = guard.evaluate(PATH, VehicleState(x, 0., 0., speed), [entry],
                                   initial_speed, now=now)
        commands.append(min(initial_speed, target))
        delayed = commands[max(0, tick-6)]
        speed = max(0., speed+max(-2.5*.05, min(1.5*.05, delayed-speed)))
        x += speed*.05
        if y < 1.9:
            minimum_gap = min(minimum_gap, 32.-.4-x-4.635)
    assert minimum_gap > 3.5
    assert commands[-1] == 0.
    assert speed == 0.
