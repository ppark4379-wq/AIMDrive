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
    wall = [dict(center_world=[x, -140.], length_m=1., width_m=1., heading_deg=0.) for x in np.arange(-62., -47., .8)]
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
    # All recordings bypass to the right; with that gap closed the car uses the left half of its lane.
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
    monkeypatch.setattr(StaticCorridorPlanner, '_choose_side', lambda *a: ([], 0))   # the new solve fails
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
    monkeypatch.setattr(StaticCorridorPlanner, '_choose_side', lambda *a: ([], 0))
    result = plan.run(ego(x=float(np.interp(-156., pts[:, 1], pts[:, 0])), y=-156.), [], ROUTE)
    assert details(result)['phase'] == 'blocked'
