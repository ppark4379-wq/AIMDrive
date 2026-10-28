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
from AIM2026.Expert.src.planning.jamming.recorded_planner import JammingPlanner
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
    for k in range(25):                  # standing: uncertain/moving labels from box jitter; a car is mapped once it
                                         # has stood PARK_OBSERVE_S near the vehicle (it might be about to drive off)
        jitter = .15*(-1)**k
        m.observe_packet(packet((-1.2, 20.), [box(-1.+jitter, 40., 4.5, 2., track=5, state=('uncertain', 'moving')[k % 2]),
                                              box(-4.5, 32.+1.*k, 4.5, 2., track=9, state='moving')], k*.1))
    assert not m.motion.is_mover(5) and m.motion.is_mover(9)
    assert m.distance(-1., 40.) == 0.
    for k in range(25, 37):              # the mover drove away; its early cells are cleared as free space
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


def test_far_box_jitter_is_not_motion_but_near_motion_is():
    m = TunnelObstacleMap()
    for k in range(10):                  # 25 m ahead the box centre slides 2.5 m as more of the car is seen
        m.observe_packet(packet((-1.2, 20.), [box(-.25*k, 45., 4.5, 2., track=3, state='uncertain')], k*.1))
    assert not m.motion.is_mover(3)
    for k in range(10):                  # 12 m ahead a box crosses the lane
        m.observe_packet(packet((-1.2, 20.), [box(-6.+.4*k, 32., .8, .6, track=4, state='uncertain')], 1.+k*.1))
    assert m.motion.is_mover(4)


def lane(forward):
    """Own lane centre (the route) in the section frame at a forward position."""
    route = np.array([SECTION.to_local((q.x, q.y)) for q in ROUTE])
    route = route[(np.abs(route[:, 0]) < 8) & (route[:, 1] > -40) & (route[:, 1] < 150)]
    route = route[np.argsort(route[:, 1])]
    return float(np.interp(forward, route[:, 1], route[:, 0]))


def drive(jam, m, frames, ego_local, start=0., dt=.1):
    """Feed one packet per frame (a list of boxes) and plan after each; returns the last result and the time."""
    t, result = start, None
    for boxes in frames:
        m.observe_packet(packet(ego_local, boxes, t))
        result = jam.run(ego(*ego_local), ROUTE[:5], [], global_path=ROUTE)
        t += dt
    return result, t


def lateral_at(result, forward):
    local = np.array([SECTION.to_local((q.x, q.y)) for q in result.path])
    return float(np.interp(forward, local[:, 1], local[:, 0]))


def test_a_moving_lead_is_followed_in_the_lane(road):
    jam, m = planner_with(road, [])
    try:
        frames = [[box(lane(24.+.6*k), 24.+.6*k, 4.6, 1.9, track=21, state='moving')] for k in range(15)]
        result, _ = drive(jam, m, frames, (lane(5.), 5.))
        details = result.debug_details['jamming_traffic']
        assert result.reason == 'jamming_traffic_lead'
        assert [d['state'] for d in details['leads']] == ['moving']
        assert details['leads'][0]['speed'] == pytest.approx(6., abs=1.)
        assert abs(lateral_at(result, 30.)-lane(30.)) < .3       # behind it in the lane, no detour
        assert not details['overtaking']
    finally:
        jam.close()


def test_a_car_that_never_moved_is_waited_behind_then_passed_as_parked(road):
    jam, m = planner_with(road, [])
    try:
        car = lambda: [box(lane(30.), 30., 4.6, 1.9, track=22, state='stationary')]
        result, t = drive(jam, m, [car() for _ in range(10)], (lane(15.), 15.))
        assert result.reason == 'jamming_traffic_lead'                    # not judged yet: may drive off
        assert result.debug_details['jamming_traffic']['leads'][0]['state'] == 'waiting'
        result, _ = drive(jam, m, [car() for _ in range(15)], (lane(15.), 15.), start=t)
        assert not result.debug_details['jamming_traffic']['leads']        # 2.5 s still: parked
        assert lateral_at(result, 30.) < lane(30.)-1.9                     # passed in the opposite lane
    finally:
        jam.close()


def test_a_slow_car_is_overtaken_and_a_pedestrian_never_is(road):
    jam, m = planner_with(road, [])
    try:
        frames = [[box(lane(22.+.12*k), 22.+.12*k, 4.6, 1.9, track=23, state='moving')] for k in range(35)]
        result, _ = drive(jam, m, frames, (lane(5.), 5.))
        details = result.debug_details['jamming_traffic']
        assert details['overtaking'] == [23] and not details['leads']
        assert lateral_at(result, 27.) < lane(27.)-1.9                     # round it in the opposite lane
    finally:
        jam.close()
    jam, m = planner_with(road, [])
    try:                                 # a pedestrian walking the lane even slower: waited for, not overtaken
        frames = [[box(lane(20.+.1*k), 20.+.1*k, .7, .6, track=24, state='moving')] for k in range(35)]
        result, _ = drive(jam, m, frames, (lane(5.), 5.))
        details = result.debug_details['jamming_traffic']
        assert not details['overtaking'] and details['yields']
        assert result.reason == 'jamming_traffic_yield'
    finally:
        jam.close()


def test_no_detour_into_an_oncoming_car(road):
    jam, m = planner_with(road, [])
    try:
        frames = [[box(lane(22.+.12*k), 22.+.12*k, 4.6, 1.9, track=23, state='moving'),
                   box(-4.8, 50.-.6*k, 4.6, 1.9, track=31, state='moving' if k > 3 else 'uncertain',
                       local_heading=-90.)] for k in range(35)]
        result, _ = drive(jam, m, frames, (lane(5.), 5.))
        details = result.debug_details['jamming_traffic']
        assert any(b['id'] == 31 for b in details['blockers'])
        entry = details['entry_station']
        # Whatever the plan, the car can stop before its body would cross into the opposite lane.
        if entry is not None:
            assert result.acc_target_velocity <= math.sqrt(2.*3.*max(0., entry-1.))+1e-6
    finally:
        jam.close()


def test_a_crossing_pedestrian_is_waited_for_and_one_on_the_far_side_is_not(road):
    jam, m = planner_with(road, [])
    try:
        frames = [[box(-6.+.25*k, 20., .7, .6, track=41, state='moving')] for k in range(10)]
        result, _ = drive(jam, m, frames, (lane(5.), 5.))
        assert result.reason == 'jamming_traffic_yield'
    finally:
        jam.close()
    jam, m = planner_with(road, [])
    try:
        frames = [[box(-5.8, 20.+.25*k, .7, .6, track=42, state='moving')] for k in range(10)]
        result, _ = drive(jam, m, frames, (lane(5.), 5.))
        assert not result.debug_details['jamming_traffic']['yields']
    finally:
        jam.close()


def test_a_pedestrian_who_stops_in_the_lane_is_waited_for():
    m = TunnelObstacleMap()
    t = 0.
    for k in range(16):                  # walks 4 m into the lane ...
        m.observe_packet(packet((-1.2, 50.), [box(-5.+.25*k, 62., .8, .6, track=11, state='moving')], t)); t += .1
    for k in range(40):                  # ... and stands there for 4 s
        m.observe_packet(packet((-1.2, 50.), [box(-1.+.05*(-1)**k, 62., .8, .6, track=11, state='stationary')], t)); t += .1
    assert m.motion.is_mover(11) and m.motion.tracks[11]['person']
    assert m.distance(-1., 62.) > 1.     # never planned around


def test_a_stopped_car_stands_again_after_the_stall_time_and_other_movers_sooner():
    m = TunnelObstacleMap()
    t = 0.
    for k in range(10):
        m.observe_packet(packet((-1.2, 30.), [box(-1.2, 40.+.5*k, 4.6, 1.9, track=12, state='moving'),
                                               box(-5., 38.+.3*k, 2.2, 1.6, track=13, state='moving')], t)); t += .1
    assert m.motion.is_mover(12) and m.motion.is_mover(13)
    still = lambda: [box(-1.2, 44.5, 4.6, 1.9, track=12, state='stationary'),
                     box(-5., 40.7, 2.2, 1.6, track=13, state='stationary')]
    for _ in range(25):                  # 2.5 s still
        m.observe_packet(packet((-1.2, 30.), still(), t)); t += .1
    assert m.motion.is_mover(12) and not m.motion.is_mover(13)
    for _ in range(8):                   # 3.3 s still
        m.observe_packet(packet((-1.2, 30.), still(), t)); t += .1
    assert not m.motion.is_mover(12) and m.motion.standing(12)


def test_a_renumbered_parked_car_keeps_its_judgement_and_a_crosswise_car_is_parked_at_once():
    m = TunnelObstacleMap()
    t = 0.
    for k in range(20):                  # a parked car seen 2 s ...
        m.observe_packet(packet((-1.2, 20.), [box(-1.2, 40., 4.6, 1.9, track=51)], t)); t += .1
    assert m.motion.standing(51)
    m.observe_packet(packet((-1.2, 20.), [box(-1.15, 40.1, 4.6, 1.9, track=52)], t))   # ... renumbered by the tracker
    assert m.motion.standing(52) and m.distance(-1.2, 40.) == 0.
    for k in range(3):                   # a car across the lane cannot be traffic about to drive off along it
        m.observe_packet(packet((-1.2, 20.), [box(-1.2, 30., 4.6, 1.9, track=53, local_heading=0.)], t)); t += .1
    assert m.motion.standing(53)


def test_a_box_merged_with_the_wall_is_not_an_oncoming_car(road):
    jam, m = planner_with(road, [])
    try:
        frames = [[box(-6.9, 40., 7.4, 2.7, track=61, state='uncertain')] for _ in range(5)]
        result, _ = drive(jam, m, frames, (lane(15.), 15.))
        assert not result.debug_details['jamming_traffic']['blockers']
        frames = [[box(-4.8, 45., 4.5, 1.9, track=62, state='uncertain')] for _ in range(5)]
        result, _ = drive(jam, m, frames, (lane(15.), 15.), start=1.)
        assert [b['id'] for b in result.debug_details['jamming_traffic']['blockers']] == [62]
    finally:
        jam.close()


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
