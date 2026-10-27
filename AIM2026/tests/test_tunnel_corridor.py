"""Standing obstacles in the Jamming (tunnel) section on the accumulated map; movers keep the existing planner."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.vehicle_state import VehicleState
from AIM2026.Expert.src.obstacle.object_info import ObjectInfo
from AIM2026.Expert.src.config.parameters.static_obstacle import CORRIDOR_MIN_CLEARANCE
from AIM2026.Expert.src.planning.jamming.recorded_planner import JammingPlanner, QUIET_TICKS
from AIM2026.Expert.src.planning.jamming.road_boundary import RoadBoundary
from AIM2026.Expert.src.planning.jamming.section import JammingSection
from AIM2026.Expert.tunnel_map import TunnelObstacleMap

MAP_DIR = Path(__file__).resolve().parents[1]/'Expert/map_data/R_KR_PG_KATRI_2025'
ROUTE = [Point(*row[:2]) for row in np.loadtxt(MAP_DIR/'2026_molit_comp_global_path.txt')]
SECTION = JammingSection()


@pytest.fixture(scope='module')
def road():
    return RoadBoundary.from_map_dir(MAP_DIR)


def world(lateral, forward):
    return SECTION.to_world((lateral, forward))


def world_yaw(local_yaw):
    return local_yaw-SECTION.rotation


def packet(ego_local, boxes, t, generation='g'):
    x, y = world(*ego_local)
    return dict(generation=generation, sent_ns=int(t*1e9), boxes=boxes,
                ego_pose=dict(x=x, y=y, yaw_grid_rad=world_yaw(math.pi/2), speed_mps=5.))


def box(lateral, forward, length, width, track, state='stationary', local_heading=90.):
    return dict(center_world=list(world(lateral, forward)), length_m=length, width_m=width, track_id=track,
                heading_deg=math.degrees(world_yaw(math.radians(local_heading))), motion_state=state)


def ego(lateral, forward, speed=5.):
    x, y = world(lateral, forward)
    return VehicleState(x, y, world_yaw(math.pi/2), speed)


def test_tunnel_map_is_in_the_section_frame():
    m = TunnelObstacleMap()
    for k in range(4):
        m.observe_packet(packet((-1.2, 30.), [box(-1., 50., 2., 1., track=3)], k*.1))
    assert m.distance(-1., 50.) == 0.
    assert m.distance(-1., 53.) == pytest.approx(2., abs=.15)     # 1 m half length, 3 m away along the section
    (mapped,) = m.boxes()
    assert mapped['center_world'] == pytest.approx([-1., 50.], abs=.1)


def test_standing_box_with_flickering_labels_is_mapped_and_a_mover_is_not():
    m = TunnelObstacleMap()
    for k in range(8):                   # standing: uncertain/moving labels from box jitter
        jitter = .15*(-1)**k
        m.observe_packet(packet((-1.2, 20.), [box(-1.+jitter, 40., 4.5, 2., track=5, state=('uncertain', 'moving')[k % 2]),
                                              box(-4.5, 32.+1.*k, 4.5, 2., track=9, state='moving')], k*.1))
    assert not m.motion.is_mover(5) and m.motion.is_mover(9)
    assert m.distance(-1., 40.) == 0.
    for k in range(8, 20):               # the mover drove away; its early cells are cleared as free space
        m.observe_packet(packet((-1.2, 20.), [box(-1., 40., 4.5, 2., track=5)], k*.1))
    assert m.distance(-4.5, 33.) > 1.


def planner_with(road, cells):
    jam = JammingPlanner(background=False, road_boundary=road)
    m = TunnelObstacleMap()
    hits = np.zeros(m.shape, dtype=bool)
    for lateral, forward, length, width in cells:
        m._cover(hits, dict(center_world=[lateral, forward], length_m=length, width_m=width, heading_deg=90.))
    m.occupied, m.version = hits, m.version+1
    jam.attach_map(m, ROUTE)
    assert jam.corridor.ready                      # set up before the first tick in the section
    return jam, m


def test_standing_car_in_the_own_lane_is_passed_in_the_opposite_lane(road):
    jam, _ = planner_with(road, [(-.7, 40., 4.6, 4.5)])          # a car blown up by heading noise
    try:
        result = jam.run(ego(-1.25, 5.), ROUTE[:5], [], global_path=ROUTE)
        d = result.debug_details['static_avoidance']
        assert result.reason == 'jamming_corridor_follow' and d['path_validated']
        assert d['min_clearance_m'] >= CORRIDOR_MIN_CLEARANCE-.05
        local = np.array([SECTION.to_local((q.x, q.y)) for q in result.path])
        beside = float(np.interp(40., local[:, 1], local[:, 0]))
        assert beside < -.7-2.25-1.085                            # left of the blob, in the opposite lane
        assert jam.road_boundary.mask_world([Point(*world(beside, 40.))], [world_yaw(math.pi/2)],
                                            jam.road_boundary.runtime_margin_m).all()
    finally:
        jam.close()


def test_moving_traffic_sets_the_speed_on_the_map_path(road):
    # A mover used to hand the section to the recorded-route planner, whose path starts at the lane centre: in the
    # middle of a detour the commanded path jumped by metres and the steering swung. Now the map path stays and the
    # moving-object overlay only sets the speed.
    def lateral_at_40(result):
        local = np.array([SECTION.to_local((q.x, q.y)) for q in result.path])
        return float(np.interp(40., local[:, 1], local[:, 0]))
    jam, m = planner_with(road, [(-.7, 40., 4.6, 2.)])
    try:
        free = jam.run(ego(-1.25, 5.), ROUTE[:5], [], global_path=ROUTE)
        mover = ObjectInfo(*world(-4.6, 30.), 6., -1, name='ve_tracked_unknown:moving', is_static=False,
                           size_x=4.5, size_y=2., heading_deg=math.degrees(world_yaw(-math.pi/2)), unique_id=11)
        m.motion.tracks[11] = dict(first=(0., 0.), run=0, mover=True, seen=0.)
        result = jam.run(ego(-1.25, 5.), ROUTE[:5], [dict(object_info=mover)], global_path=ROUTE)
        assert result.reason.startswith('jamming_dynamic')
        assert lateral_at_40(result) == pytest.approx(lateral_at_40(free), abs=.05)   # same detour, no jump
        assert result.acc_target_velocity < free.acc_target_velocity                  # oncoming car: slower
        assert result.debug_details['jamming_dynamic']['speed_authority']
        m.motion.tracks.clear()
        for _ in range(QUIET_TICKS):     # the band has been quiet long enough: the map planner alone again
            result = jam.run(ego(-1.25, 5.), ROUTE[:5], [], global_path=ROUTE)
        assert result.reason.startswith('jamming_corridor')
        assert result.debug_details['jamming_corridor']['standing_only']
    finally:
        jam.close()


def test_far_box_jitter_is_not_motion_but_near_motion_is():
    m = TunnelObstacleMap()
    for k in range(10):                  # 25 m ahead the box centre slides 2.5 m as more of the car is seen
        m.observe_packet(packet((-1.2, 20.), [box(-.25*k, 45., 4.5, 2., track=3, state='uncertain')], k*.1))
    assert not m.motion.is_mover(3)
    for k in range(10):                  # 12 m ahead a box crosses the lane
        m.observe_packet(packet((-1.2, 20.), [box(-6.+.4*k, 32., .8, .6, track=4, state='uncertain')], 1.+k*.1))
    assert m.motion.is_mover(4)


def test_a_mover_that_stops_is_standing_again():
    m = TunnelObstacleMap()
    t = 0.
    for k in range(8):                   # crosses ...
        m.observe_packet(packet((-1.2, 50.), [box(-6.+.9*k, 62., .8, .6, track=11, state='moving')], t)); t += .1
    assert m.motion.is_mover(11)
    for k in range(25):                  # ... and stands at the road edge for 2.5 s
        m.observe_packet(packet((-1.2, 50.), [box(.5+.05*(-1)**k, 62., .8, .6, track=11, state='stationary')], t)); t += .1
    assert not m.motion.is_mover(11)
    assert m.distance(.5, 62.) == 0.     # now it is mapped and planned around


def test_legacy_acc_leaves_ve_objects_to_the_map_planner():
    from types import SimpleNamespace
    from AIM2026.Expert.src.planning.common.object_selection import _legacy_acc_entries
    ve = ObjectInfo(0., 0., 30., -1, name='ve_tracked_unknown:uncertain', is_static=False,
                    size_x=1., size_y=1., heading_deg=0., unique_id=4)
    person = ObjectInfo(0., 0., 1., 0, name='walker', size_x=.5, size_y=.5, heading_deg=0.)
    result = SimpleNamespace(mode='static_obstacle', active=True, acc_target_velocity=5.,
                             debug_details={'static_avoidance': {'path_validated': True},
                                            'jamming_corridor': {'standing_only': True}})
    kept = _legacy_acc_entries(SimpleNamespace(), None, result, [dict(object_info=ve), dict(object_info=person)])
    assert [entry['object_info'] for entry in kept] == [person]


def test_fresh_solves_follow_the_curving_route_at_the_tunnel_exit(road):
    # The route bends right toward the exit. Slope and curvature were penalised in absolute terms, so every fresh
    # solve there pulled straight, 0.2-0.5 m off the route, and alternating solves flipped the steering.
    route = np.array([SECTION.to_local((q.x, q.y)) for q in ROUTE])
    route = route[(np.abs(route[:, 0]) < 8) & (route[:, 1] > 0) & (route[:, 1] < 150)]
    route = route[np.argsort(route[:, 1])]
    for forward in (115., 120., 125., 128.):
        jam, _ = planner_with(road, [])
        try:
            x = float(np.interp(forward, route[:, 1], route[:, 0]))
            slope = float(np.interp(forward+.5, route[:, 1], route[:, 0]))-float(np.interp(forward-.5, route[:, 1], route[:, 0]))
            heading = math.atan2(1., slope)-SECTION.rotation
            wx, wy = SECTION.to_world((x, forward))
            result = jam.run(VehicleState(wx, wy, heading, 7.5), ROUTE[:5], [], global_path=ROUTE)
            path = np.array([SECTION.to_local((q.x, q.y)) for q in result.path])
            for ahead in (3., 6., 9.):
                off = (np.interp(forward+ahead, path[:, 1], path[:, 0])
                       - np.interp(forward+ahead, route[:, 1], route[:, 0]))
                assert abs(off) < .1, (forward, ahead, off)
        finally:
            jam.close()
