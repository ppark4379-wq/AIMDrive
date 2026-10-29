"""Exercise the existing planners with the extended static recording library."""
import math

import numpy as np
import pytest

from AIM2026.tests.test_static_corridor import ROUTE, ego, planner, details, lateral_at, occupancy
from AIM2026.tests.test_static_map import box, frame
from AIM2026.Expert.static_map import StaticObstacleMap
from AIM2026.Expert.src.planning.section_planning import LatticePlanningProcess
from AIM2026.Expert.src.planning.static_obstacle.recorded_planner import load_routes, RecordedRouteGeometry
from AIM2026.Expert.src.config.parameters.static_obstacle import CORRIDOR_MIN_CLEARANCE, MAX_CURVATURE
from AIM2026.Expert.src.obstacle.object_info import ObjectInfo


def wall(start, end):
    return [dict(center_world=[x, -140.], length_m=1., width_m=1., heading_deg=0.)
            for x in np.arange(start, end, .8)]


def test_existing_loader_accepts_all_recordings_without_trimming_left_data():
    records, audit = load_routes()
    assert [r[0] for r in records[:6]] == [
        'right_01.csv', 'right_02.csv', 'right_03.csv', 'right_04.csv', 'right_05.csv', 'right_06_retry.csv']
    assert [r[0] for r in records[6:]] == [f'left_static_obstacle_{i:02d}.csv' for i in range(1, 6)]
    assert all(r['trimmed_prefix_points'] == 0 for r in audit[6:])
    assert all(r['end'][1] < RecordedRouteGeometry().rejoin_point[1] for r in audit)


@pytest.mark.parametrize('side,obstacles', [('left', wall(-62., -47.)), ('right', wall(-72., -59.))])
def test_same_optimizer_bypasses_the_only_open_side_and_returns(side, obstacles):
    result = planner(obstacles).run(ego(), [], ROUTE)
    d = details(result)
    assert d['phase'] == 'follow' and d['path_validated']
    assert d['min_clearance_m'] >= CORRIDOR_MIN_CLEARANCE-.05
    assert d['max_curvature'] <= MAX_CURVATURE+1e-6
    x = lateral_at(result, -140.)
    assert x < -63. if side == 'left' else x > -58.
    assert abs(lateral_at(result, -116.7)+60.) < .3


def test_left_obstacle_evidence_is_used_and_can_block_the_left_bypass():
    m = StaticObstacleMap()
    for _ in range(4):
        mapped, _ = frame(m, [box(-66., -140., length=1., width=1.)])
    assert len(mapped) == 1 and float(m.distance(-66., -140.)) == 0.
    plan = planner(wall(-62., -47.))
    assert details(plan.run(ego(), [], ROUTE))['path_validated']
    plan.occupancy = occupancy(wall(-72., -47.))
    result = plan.run(ego(), [], ROUTE)
    assert details(result)['phase'] == 'blocked'
    assert not details(result)['path_validated']


def test_actual_section_instances_keep_leftmost_recordings_inside_activation():
    process = LatticePlanningProcess(jamming_enabled=False)
    try:
        process.attach_static_map(StaticObstacleMap(), global_path=ROUTE)
        assert process.static_corridor.ready
        for _, data, _ in process.recorded_static.records[6:]:
            point = data[np.argmin(data[:, 0])]
            state = ego(x=point[0], y=point[1], heading=math.degrees(point[2]))
            assert process.recorded_static.contains(state)
            assert process.static_corridor.contains(state)
        # Shared geometry for tunnel/highway users was not widened.
        assert RecordedRouteGeometry().active_x == (-65., -44.)
        for x, y in ((0., 0.), (-77., -140.), (-60., -181.)):
            assert not process.static_corridor.contains(ego(x=x, y=y))
    finally:
        process.recorded_static.executor.shutdown(wait=True)


def test_recorded_planner_connects_left_with_existing_obb_checks():
    process = LatticePlanningProcess(jamming_enabled=False)
    try:
        plan = process.recorded_static
        # Check path feasibility independently of host load; production budgets stay unchanged.
        plan.recording_search_timeout = 30.
        plan.corridor_search_timeout = 30.
        objects = [ObjectInfo(*b['center_world'], 0., 2, name='wall', is_static=True,
                              heading_deg=0., size_x=1., size_y=1.)
                   for b in wall(-62., -47.)]
        path = plan._prepare(ego(), ROUTE, objects)
        assert path is not None
        assert np.interp(-140., path[:, 1], path[:, 0]) < -63.
        yaw = np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0]))
        assert plan.collision_free(path, yaw, objects)
    finally:
        process.recorded_static.executor.shutdown(wait=True)
