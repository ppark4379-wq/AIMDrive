"""Receding-horizon static-section planner on an occupancy map."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.vehicle_state import VehicleState
from AIM2026.Expert.src.config.parameters import lattice as p
from AIM2026.Expert.src.config.parameters.static_obstacle import CORRIDOR_MIN_CLEARANCE, MAX_CURVATURE
from AIM2026.Expert.src.planning.lattice.lattice_planner import LatticePlanner
from AIM2026.Expert.src.planning.static_obstacle.recorded_planner import RecordedStaticPlanner
from AIM2026.Expert.src.planning.static_obstacle import corridor_optimizer
from AIM2026.Expert.src.planning.static_obstacle.corridor_optimizer import StaticCorridorPlanner
from AIM2026.Expert.src.planning.section_planning import LatticePlanningProcess
from AIM2026.Expert.static_map import StaticObstacleMap

ROUTE = [Point(*row[:2]) for row in np.loadtxt(
    Path(__file__).resolve().parents[1]/'Expert/map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt')]
IN_LANE = dict(center_world=[-60.6, -142.7], length_m=1.5, width_m=1.2, heading_deg=90.)
RIGHT = [dict(center_world=[x, y], length_m=.9, width_m=.5, heading_deg=0.)
         for x, y in ((-54.4, -143.4), (-54.2, -140.9), (-54.0, -138.1))]


def occupancy(boxes):
    m = StaticObstacleMap()
    hits = np.zeros(m.shape, dtype=bool)
    for b in boxes:
        m._cover(hits, b)
    m.occupied, m.version = hits, m.version+1
    return m


def ego(x=-60.1, y=-160., heading=90., speed=4.):
    return VehicleState(x, y, math.radians(heading), speed)


def planner(boxes):
    return StaticCorridorPlanner(RecordedStaticPlanner(LatticePlanner()), occupancy(boxes))


def details(result):
    return result.debug_details['static_avoidance']


def lateral_at(result, y):
    pts = np.array([(q.x, q.y) for q in result.path])
    return float(np.interp(y, pts[:, 1], pts[:, 0]))


def test_free_section_follows_the_reference_at_full_speed():
    result = planner([]).run(ego(), [], ROUTE)
    assert details(result)['phase'] == 'follow'
    assert abs(lateral_at(result, -142.) - (-60.)) < .15
    assert result.acc_target_velocity == pytest.approx(p.STATIC_MAX_SPEED_KPH/3.6, rel=1e-6)


def test_in_lane_obstacle_is_bypassed_smoothly_with_clearance():
    plan = planner([IN_LANE]+RIGHT)
    result = plan.run(ego(), [], ROUTE)
    d = details(result)
    assert d['phase'] == 'follow' and d['path_validated']
    assert d['min_clearance_m'] >= CORRIDOR_MIN_CLEARANCE-.05
    assert d['max_curvature'] <= MAX_CURVATURE+1e-6
    assert lateral_at(result, -142.7) > -59.                     # left the lane centre beside the obstacle
    assert result.acc_target_velocity > 0.


def test_slightly_larger_obstacle_shifts_the_path_instead_of_stopping():
    plan = planner([IN_LANE]+RIGHT)
    first = plan.run(ego(y=-158.), [], ROUTE)
    grown = dict(IN_LANE, length_m=2.0, width_m=1.6)              # the map grew as the car approached
    plan.occupancy = occupancy([grown]+RIGHT)
    second = plan.run(ego(y=-157.), [], ROUTE)
    assert details(second)['phase'] == 'follow'
    assert second.acc_target_velocity > 0.
    shift = abs(lateral_at(second, -142.7)-lateral_at(first, -142.7))
    assert 0. <= shift < .6                                      # a local adjustment, not a new maneuver


def test_pose_inside_min_clearance_still_moves_away():
    plan = planner([dict(center_world=[-61.8, -147.], length_m=3., width_m=1., heading_deg=90.)])
    result = plan.run(ego(x=-59.95, y=-149.), [], ROUTE)
    assert .1 < details(result)['pose_clearance_m'] < CORRIDOR_MIN_CLEARANCE
    assert details(result)['phase'] == 'follow' and result.acc_target_velocity > 0.


def test_blocked_section_stops_before_the_blockage():
    # Cover the newly registered left corridor too; the old wall now has a valid left bypass.
    wall = [dict(center_world=[x, -140.], length_m=1., width_m=1., heading_deg=0.) for x in np.arange(-72., -47., .8)]
    result = planner(wall).run(ego(y=-160.), [], ROUTE)
    d = details(result)
    assert d['phase'] == 'blocked' and not d['path_validated']
    pts = np.array([(q.x, q.y) for q in result.path])
    assert pts[:, 1].max() < -140.-3.                             # the path ends before the wall
    close = planner(wall).run(ego(y=-147.), [], ROUTE)
    assert close.acc_target_velocity == 0.


def test_section_planner_uses_the_corridor_only_when_a_map_is_attached():
    process = LatticePlanningProcess(jamming_enabled=False)
    try:
        legacy = process.run(ego(), ROUTE[:80], [], global_path=ROUTE)
        assert not legacy.reason.startswith('static_corridor')
        process.attach_static_map(occupancy([IN_LANE]))
        result = process.run(ego(), ROUTE[:80], [], global_path=ROUTE)
        assert result.reason == 'static_corridor_follow'
        outside = process.run(ego(x=0., y=0.), ROUTE[:80], [], global_path=ROUTE)
        assert not outside.reason.startswith('static_corridor')
    finally:
        process.recorded_static.executor.shutdown(wait=True)


def test_smoothing_stays_finite_with_pinned_bounds():
    plan = planner([])
    lower = np.array([-60., -59.9, -59.9, -59.8, -59.8])
    result = plan._smooth(lower.copy(), lower, lower.copy(), -60., 0.)     # every variable pinned
    assert np.isfinite(result).all() and np.all(result >= lower-1e-9)


def test_numerical_failure_stops_the_car_instead_of_raising(monkeypatch):
    plan = planner([IN_LANE])
    monkeypatch.setattr(StaticCorridorPlanner, '_dense', lambda *a: (_ for _ in ()).throw(ValueError('nan')))
    result = plan.run(ego(), [], ROUTE)
    assert details(result)['phase'] == 'blocked' and result.acc_target_velocity == 0.
    assert details(result)['error'].startswith('planning_error')


def test_blocked_right_gap_passes_left_inside_the_own_lane():
    # This obstacle still fits a bypass inside the own lane, without using the wide left recordings.
    gap = dict(center_world=[-57.45, -143.57], length_m=3.4, width_m=1.9, heading_deg=90.)
    result = planner([gap]+RIGHT).run(ego(), [], ROUTE)
    d = details(result)
    assert d['phase'] == 'follow' and d['min_clearance_m'] >= CORRIDOR_MIN_CLEARANCE-.05
    beside = lateral_at(result, -143.6)
    assert -60.55 < beside < -60.1                                # left of the lane centre, inside the lane


def test_failed_new_solve_keeps_the_still_valid_committed_path(monkeypatch):
    plan = planner([IN_LANE]+RIGHT)
    first = plan.run(ego(y=-158.), [], ROUTE)
    assert details(first)['source'] == 'new' and plan.committed is not None
    pts = np.array([(q.x, q.y) for q in first.path])
    on_path = ego(x=float(np.interp(-156., pts[:, 1], pts[:, 0])), y=-156.)
    monkeypatch.setattr(StaticCorridorPlanner, '_choose_side', lambda *a: ([], [], 0))   # the new solve fails
    kept = plan.run(on_path, [], ROUTE)
    assert details(kept)['phase'] == 'follow' and details(kept)['source'] == 'previous'
    assert kept.acc_target_velocity > 0.
    plan.committed = None                                       # nothing valid to fall back on: stop
    assert details(plan.run(on_path, [], ROUTE))['phase'] == 'blocked'


def test_committed_path_is_dropped_when_it_becomes_unsafe(monkeypatch):
    plan = planner([IN_LANE]+RIGHT)
    first = plan.run(ego(y=-158.), [], ROUTE)
    pts = np.array([(q.x, q.y) for q in first.path])
    beside = dict(center_world=[float(np.interp(-145., pts[:, 1], pts[:, 0])), -145.], length_m=1., width_m=1., heading_deg=0.)
    plan.occupancy = occupancy([IN_LANE, beside]+RIGHT)          # a new obstacle on the committed path
    monkeypatch.setattr(StaticCorridorPlanner, '_choose_side', lambda *a: ([], [], 0))
    result = plan.run(ego(x=float(np.interp(-156., pts[:, 1], pts[:, 0])), y=-156.), [], ROUTE)
    assert details(result)['phase'] == 'blocked'


def test_approach_from_outside_the_map_is_not_blocked_by_a_distant_obstacle():
    # The planner starts at y=-180 while the map starts at the section (-162.4): points before the map
    # must get real distances, not a collapsed screening bound.
    plan = planner([dict(center_world=[-60.3, -150.1], length_m=1.6, width_m=1.3, heading_deg=90.)])
    result = plan.run(ego(x=-61.84, y=-178.46, heading=82.), [], ROUTE)
    d = details(result)
    # Seen 28 m ahead, the obstacle may still decide where to stop, but not block the road right ahead.
    assert d['feasible_stations'] > 15 and result.acc_target_velocity > 0.


MAP_DIR = Path(__file__).resolve().parents[1]/'Expert/map_data/R_KR_PG_KATRI_2025'


def test_obstacle_at_the_section_entrance_is_passed_on_the_right_turn_lane():
    # Run 20261026-192930: a car-sized box early in the section, before the recorded bypasses turn right. The gap
    # needs the right half of the own lane, the right-turn lane at the fork, and the body turned about 15 degrees.
    entrance = dict(center_world=[-60.50, -149.04], length_m=4.09, width_m=1.73, heading_deg=98.)
    plan = StaticCorridorPlanner(RecordedStaticPlanner(LatticePlanner()), occupancy([entrance]), map_data_dir=MAP_DIR)
    result = plan.run(ego(x=-59.9, y=-170.), [], ROUTE)
    d = details(result)
    assert d['phase'] == 'follow' and d['path_validated'] and result.acc_target_velocity > 0.
    assert d['min_clearance_m'] >= CORRIDOR_MIN_CLEARANCE-.05
    assert d['max_curvature'] <= MAX_CURVATURE+1e-6
    assert lateral_at(result, -151.) > -58.4                    # right of the obstacle's rear corner


def test_box_qp_matches_bounded_least_squares():
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(3)
    for _ in range(20):
        A = np.vstack([np.eye(30), rng.normal(size=(30, 30))*5.])
        b = rng.normal(size=60)*3.
        lower = rng.uniform(-1., 0., 30); upper = lower+rng.uniform(.01, 1., 30)
        solution = StaticCorridorPlanner._box_qp(A.T@A, A.T@b, lower, upper, lower.copy())
        reference = lsq_linear(A, b, bounds=(lower, upper), method='bvls').x
        assert solution is not None
        assert np.sum((A@solution-b)**2) <= np.sum((A@reference-b)**2)*(1+1e-9)+1e-9


class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t


def gap_wall(gap, centre=-60.):
    """Two 5 m blocks across the section at y=-140 leaving a gap of `gap` metres centred on `centre`."""
    a, b = centre-gap/2, centre+gap/2
    return [dict(center_world=[a-2.5, -140.], length_m=5., width_m=1., heading_deg=0.),
            dict(center_world=[b+2.5, -140.], length_m=5., width_m=1., heading_deg=0.)]


def stalled(boxes):
    plan, clock = planner(boxes), Clock()
    plan.clock = clock
    def tick(seconds=.5, speed=0.):
        clock.t += seconds
        return plan.run(ego(y=-150., speed=speed), [], ROUTE)
    return plan, tick


def test_static_last_resort_waits_then_lowers_the_clearance_and_drives_on_slowly():
    # Same last resort as the tunnel (2e0f7783): blocked and standing for FALLBACK_STALL_S, a 2.8 m gap (too tight
    # for 0.35 m) is passed with the first lower stage, at the capped speed, inside the corridor.
    plan, tick = stalled(gap_wall(2.8))
    for _ in range(5):                                            # 2.5 s standing blocked
        result = tick()
        assert details(result)['phase'] == 'blocked' and 'fallback' not in details(result)
    tick()
    result = tick()                                               # 3.5 s
    d = details(result)
    assert result.reason == 'static_corridor_follow' and d['path_validated']
    assert d['fallback']['required_clearance_m'] == pytest.approx(plan.FALLBACK_CLEARANCES[0])
    assert 0. < result.acc_target_velocity <= plan.FALLBACK_SPEED+1e-9
    assert abs(lateral_at(result, -140.)-(-60.)) < .5             # through the gap
    assert plan.min_clearance == CORRIDOR_MIN_CLEARANCE


def test_static_last_resort_contact_stage_needs_a_full_stop():
    plan, tick = stalled(gap_wall(2.4))                           # passes only at 0.1 m or less
    for _ in range(7):
        result = tick(speed=1.)                                   # creeping: lower stages, no contact yet
    assert details(result)['fallback']['required_clearance_m'] >= 0.


def test_static_moving_car_never_starts_the_last_resort():
    plan, tick = stalled(gap_wall(2.8))
    for _ in range(10):
        result = tick(speed=plan.FALLBACK_STALL_SPEED+.5)
    assert details(result)['phase'] == 'blocked' and 'fallback' not in details(result)


def test_static_last_resort_switched_off(monkeypatch):
    monkeypatch.setattr(StaticCorridorPlanner, 'FALLBACK_STALL_S', 0.)
    plan, tick = stalled(gap_wall(2.8))
    for _ in range(10):
        result = tick()
    assert details(result)['phase'] == 'blocked'


def test_static_curvature_alone_does_not_stop_the_car(monkeypatch):
    # Only the curvature limit fails (lowered here; tunnel run 20261030-044207: 0.296 > 0.24 stopped the car at its
    # start): the least-bent path keeping every clearance and corridor margin is driven slowly and never committed.
    monkeypatch.setattr(corridor_optimizer, 'MAX_CURVATURE', .005)
    plan = planner([IN_LANE]+RIGHT)
    result = plan.run(ego(), [], ROUTE)
    d = details(result)
    assert d['phase'] == 'follow' and d['path_validated'] and 'error' not in d
    assert d['violations']['curvature'] > 0 and d['violations']['clearance'] == d['violations']['corridor'] == 0
    assert d['curvature_override']['limit'] == .005
    assert d['min_clearance_m'] >= CORRIDOR_MIN_CLEARANCE-.05
    assert 0. < result.acc_target_velocity <= plan.FALLBACK_SPEED+1e-9
    assert lateral_at(result, -142.7) > -59.                     # still round the obstacle
    assert plan.committed is None


def test_static_curvature_override_switched_off(monkeypatch):
    monkeypatch.setattr(corridor_optimizer, 'MAX_CURVATURE', .005)
    monkeypatch.setattr(StaticCorridorPlanner, 'CURVATURE_OVERRIDE', False)
    d = details(planner([IN_LANE]+RIGHT).run(ego(), [], ROUTE))
    assert d['phase'] == 'blocked' and d['error'] == 'smoothed_path_violation'


# The tunnel last-resort checks (test_tunnel_fallback.py) on the static section: same logic and values.
def test_static_last_resort_counts_creeping_towards_the_stop_point_as_waiting():
    plan, tick = stalled(gap_wall(2.8))
    for speed in (1.3, .4, 1.2, 0., .8, 0.):                       # 3.0 s blocked, never at a full stop for long
        result = tick(speed=speed)
    assert 'fallback' not in details(result)
    result = tick(speed=.2)                                       # 3.5 s
    assert result.reason == 'static_corridor_follow' and 'fallback' in details(result)


def test_static_last_resort_uses_the_first_stage_that_works():
    plan, tick = stalled(gap_wall(2.4))
    for _ in range(7):
        result = tick()
    assert result.reason == 'static_corridor_follow'
    assert 0. <= details(result)['fallback']['required_clearance_m'] < plan.FALLBACK_CLEARANCES[0]
    assert 0. < result.acc_target_velocity <= plan.FALLBACK_SPEED+1e-9


def test_static_last_resort_contact_stage_after_a_full_stop_and_kept_while_moving():
    plan, tick = stalled(gap_wall(1.8))                           # passes only with contact accepted
    for speed in (1.5, .5, 1., .5, 1., .5, 1., .5):               # 4 s creeping
        result = tick(speed=speed)
    assert plan._fallback_active and details(result)['phase'] == 'blocked'
    for _ in range(6):                                            # 0-2.5 s at a full stop: still waiting
        result = tick()
        assert details(result)['phase'] == 'blocked'
    result = tick()                                               # 3.0 s at a full stop
    assert result.reason == 'static_corridor_follow'
    assert details(result)['fallback']['required_clearance_m'] < 0.
    for speed in (.5, 1., 1.5):                                   # moving along it does not withdraw it
        result = tick(.1, speed=speed)
        assert result.reason == 'static_corridor_follow'


def test_static_last_resort_ends_after_the_release_ticks_of_fresh_normal_paths():
    from AIM2026.Expert.src.planning.static_obstacle.corridor_optimizer import RESOLVE_EVERY
    plan, tick = stalled(gap_wall(2.8))
    for _ in range(8):
        tick()
    assert plan._fallback_active
    plan.occupancy.occupied = np.zeros(plan.occupancy.shape, dtype=bool)   # the blocks were a misdetection
    plan.occupancy.version += 1
    fresh = ticks = 0
    while plan._fallback_active:
        result = tick(.1, speed=2.)
        ticks += 1
        d = details(result)
        assert result.reason == 'static_corridor_follow'
        if 'fallback' in d:                                       # the last-resort path held by the normal solve
            assert d['fallback']['source'] == 'held' and result.acc_target_velocity <= plan.FALLBACK_SPEED+1e-9
            assert fresh == 0
        else:
            fresh += 1
        assert ticks <= RESOLVE_EVERY+plan.FALLBACK_RELEASE_TICKS+1
    assert fresh == plan.FALLBACK_RELEASE_TICKS
    assert 'fallback' not in details(tick(.1, speed=2.))


def test_static_last_resort_restores_the_nominal_clearance_after_every_solve():
    plan, tick = stalled(gap_wall(2.4))
    for _ in range(8):
        tick()
        assert plan.min_clearance == CORRIDOR_MIN_CLEARANCE


def test_static_last_resort_leaves_a_free_section_unchanged():
    plan, tick = stalled([])
    result = tick(speed=5.)
    assert details(result)['phase'] == 'follow' and 'fallback' not in details(result)
