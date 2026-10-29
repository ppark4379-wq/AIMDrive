"""Tunnel last resort: blocked and standing for FALLBACK_STALL_S, the corridor planner lowers the required clearance step
by step (the road edge condition stays) and drives on slowly; a normal path ends it; movers' own rules come first."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tunnel_corridor import ROUTE, SECTION, ego, planner_with, road  # noqa: F401  (road is a fixture)
import AIM2026.Expert.src.config.parameters.jamming as J
from AIM2026.Expert.src.planning.jamming.corridor import TunnelCorridorPlanner
from AIM2026.Expert.src.planning.static_obstacle.corridor_optimizer import RESOLVE_EVERY

ROAD_LOW, ROAD_HIGH = -6.46, .54          # road edges at forward 40 m (section frame)


class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t


def gap_cells(gap, centre=-2.):
    """Two blocks across the road leaving a gap of `gap` metres centred on `centre` at forward 40 m."""
    a, b = centre-gap/2, centre+gap/2
    return [((ROAD_LOW-.5+a)/2, 40., 2., a-(ROAD_LOW-.5)), ((b+ROAD_HIGH+.5)/2, 40., 2., (ROAD_HIGH+.5)-b)]


def tick(jam, clock, seconds, speed=0.):
    clock.t += seconds
    return jam.run(ego(-2., 12., speed), ROUTE[:5], [], global_path=ROUTE)


def info(result):
    return result.debug_details['static_avoidance']


def setup(road, gap):
    jam, m = planner_with(road, gap_cells(gap))
    clock = Clock()
    jam.corridor.clock = clock
    return jam, m, clock


def test_a_gap_too_tight_for_the_normal_clearance_stops_before_the_wait_is_over(road):
    jam, _, clock = setup(road, 2.8)
    try:
        for _ in range(5):                                       # 2.5 s standing blocked
            result = tick(jam, clock, .5)
            assert result.reason == 'jamming_corridor_blocked'
            assert 'fallback' not in info(result)
    finally:
        jam.close()


def test_after_the_wait_the_clearance_is_lowered_and_the_car_drives_on_at_the_capped_speed(road):
    jam, _, clock = setup(road, 2.8)
    try:
        for _ in range(6):
            tick(jam, clock, .5)
        result = tick(jam, clock, .5)                            # 3.5 s
        d = info(result)
        assert result.reason == 'jamming_corridor_follow' and d['path_validated']
        assert d['fallback']['required_clearance_m'] == pytest.approx(.2)
        assert 0. < result.acc_target_velocity <= J.FALLBACK_SPEED+1e-9
        assert d['min_clearance_m'] >= .15                       # the path keeps what the stage asked for
        local = np.array([SECTION.to_local((q.x, q.y)) for q in result.path])
        assert abs(float(np.interp(40., local[:, 1], local[:, 0]))-(-2.)) < .5     # through the gap
    finally:
        jam.close()


def test_creeping_towards_the_stop_point_counts_as_waiting(road):
    # Blocked, the car still closes in on the stop point and the point moves on as the map changes: it crept at up to
    # 1.3 m/s twice and a standing-only test restarted the wait (9 s instead of 3, competition tunnel 10-29).
    jam, _, clock = setup(road, 2.8)
    try:
        for speed in (1.3, .4, 1.2, .0, .8, .0):                  # 3.0 s blocked, never at a full stop for long
            result = tick(jam, clock, .5, speed=speed)
        assert 'fallback' not in info(result)
        result = tick(jam, clock, .5, speed=.2)                   # 3.5 s
        assert result.reason == 'jamming_corridor_follow' and 'fallback' in info(result)
    finally:
        jam.close()


def test_a_gap_that_needs_the_last_stages_uses_the_first_stage_that_works(road):
    jam, _, clock = setup(road, 2.4)
    try:
        for _ in range(7):
            result = tick(jam, clock, .5)
        assert result.reason == 'jamming_corridor_follow'
        assert info(result)['fallback']['required_clearance_m'] == pytest.approx(0.)
        assert 0. < result.acc_target_velocity <= J.FALLBACK_SPEED+1e-9
    finally:
        jam.close()


def test_the_contact_stage_waits_for_a_full_stop(road):
    # A 2.0 m gap passes only with contact accepted. Creeping (blocked, 0.5-1.5 m/s) starts the last resort after
    # 3 s, but the contact stage needs FALLBACK_CONTACT_STALL_S at a full stop: two offline grazes came from it
    # being used at 1-2 m/s, while the map was still settling.
    jam, _, clock = setup(road, 2.0)
    try:
        for speed in (1.5, .5, 1.0, .5, 1.0, .5, 1.0, .5):          # 4 s creeping
            result = tick(jam, clock, .5, speed=speed)
        assert jam.corridor._fallback_active and result.reason == 'jamming_corridor_blocked'
        for _ in range(6):                                          # stopped at the first of these: 0-2.5 s, waiting
            result = tick(jam, clock, .5)
            assert result.reason == 'jamming_corridor_blocked'
        result = tick(jam, clock, .5)                              # 3.0 s at a full stop
        assert result.reason == 'jamming_corridor_follow'
        assert info(result)['fallback']['required_clearance_m'] < 0.
        for speed in (.5, 1.0, 1.5):                               # moving along it does not withdraw it
            result = tick(jam, clock, .1, speed=speed)
            assert result.reason == 'jamming_corridor_follow'
    finally:
        jam.close()


def test_without_the_contact_wait_the_contact_stage_is_used_while_creeping(road, monkeypatch):
    monkeypatch.setattr(TunnelCorridorPlanner, 'FALLBACK_CONTACT_STALL_S', 0.)
    jam, _, clock = setup(road, 2.0)
    try:
        for speed in (1.5, .5, 1.0, .5, 1.0, .5, 1.0):
            result = tick(jam, clock, .5, speed=speed)
        assert result.reason == 'jamming_corridor_follow' and info(result)['fallback']['required_clearance_m'] < 0.
    finally:
        jam.close()


def test_a_moving_car_never_starts_the_last_resort(road):
    jam, _, clock = setup(road, 2.8)
    try:
        for _ in range(10):
            result = tick(jam, clock, .5, speed=J.FALLBACK_STALL_SPEED+.5)
        assert result.reason == 'jamming_corridor_blocked' and 'fallback' not in info(result)
    finally:
        jam.close()


def test_the_nominal_clearance_comes_back_and_the_last_resort_ends_after_the_release_ticks(road):
    jam, m, clock = setup(road, 2.8)
    try:
        for _ in range(8):
            tick(jam, clock, .5)
        assert jam.corridor._fallback_active
        m.occupied = np.zeros(m.shape, dtype=bool)                # the blocks were a misdetection: all clear
        m.version += 1
        fresh, ticks = 0, 0
        while jam.corridor._fallback_active:
            result = tick(jam, clock, .1, speed=2.)
            ticks += 1
            d = info(result)
            assert result.reason == 'jamming_corridor_follow'
            if 'fallback' in d:                                   # the last-resort path held by the normal solve
                assert d['fallback']['source'] == 'held' and result.acc_target_velocity <= J.FALLBACK_SPEED+1e-9
                assert fresh == 0
            else:
                fresh += 1                                        # a fresh normal solve passed
            assert ticks <= RESOLVE_EVERY+J.FALLBACK_RELEASE_TICKS+1
        assert fresh == J.FALLBACK_RELEASE_TICKS
        assert 'fallback' not in info(tick(jam, clock, .1, speed=2.))
    finally:
        jam.close()


def test_the_nominal_clearance_is_restored_after_every_solve(road):
    jam, _, clock = setup(road, 2.4)
    try:
        nominal = jam.corridor.min_clearance
        for _ in range(8):
            tick(jam, clock, .5)
            assert jam.corridor.min_clearance == nominal
    finally:
        jam.close()


def test_a_mover_in_play_keeps_its_own_rules_and_no_last_resort_starts(road, monkeypatch):
    jam, m, clock = setup(road, 2.8)
    try:
        # Traffic reserves the stretch a car being overtaken will drive through (set again on every tick).
        before = jam.traffic.before
        def reserve(local_ego):
            before(local_ego)
            m.set_virtual([(-2., 40., math.pi/2, 4., 3.)])
        monkeypatch.setattr(jam.traffic, 'before', reserve)
        for _ in range(10):
            result = tick(jam, clock, .5)
        assert result.reason == 'jamming_corridor_blocked' and 'fallback' not in info(result)
    finally:
        jam.close()


def test_switched_off_the_planner_behaves_as_before(road, monkeypatch):
    monkeypatch.setattr(TunnelCorridorPlanner, 'FALLBACK_STALL_S', 0.)
    jam, _, clock = setup(road, 2.8)
    try:
        for _ in range(10):
            result = tick(jam, clock, .5)
        assert result.reason == 'jamming_corridor_blocked'
    finally:
        jam.close()


def test_a_free_road_is_unchanged(road):
    jam, _, clock = setup(road, 20.)                              # no real gap needed: the blocks leave the road open
    try:
        result = tick(jam, clock, .5, speed=5.)
        assert result.reason == 'jamming_corridor_follow' and 'fallback' not in info(result)
    finally:
        jam.close()
