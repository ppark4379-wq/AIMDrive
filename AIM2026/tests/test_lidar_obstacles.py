import math
import numpy as np
import pytest

from AIM2026.Inference.vision.lidar_obstacles import LidarObstacleTracker
from AIM2026.Inference.vision.postprocess import _rectangle, decode_world_boxes
from AIM2026.UDP.Detection import Box


def side(x=4., y=10., length=10.):
    # Only one visible side: no minimum bus width or complete outline.
    return np.array([[x, a, z] for a in np.arange(y-length/2, y+length/2+.01, .2)
                     for z in (.6, 1., 1.5, 2.)])


def ve(x=4., y=10., index=1):
    b = _rectangle([x, y], [0., 1.], 2., 1.3)
    return dict(b, score=.9-index*.01, cell_index=index, partial=False)


def run(tracker, points, t, boxes=(), pose=(0., 0., 0.)):
    return tracker.apply(list(boxes), points, pose, t)


def test_ve_seeded_bus_side_is_visible_then_confirmed_without_inventing_width():
    tracker = LidarObstacleTracker()
    for i in range(3):
        boxes, counts = run(tracker, side(), .1*i, [ve()] if i == 0 else [])
        assert len(tracker.observations) == 1
        assert len(boxes) == int(i != 1)
    assert counts['lidar_backup'] == 1
    assert boxes[0]['width'] == pytest.approx(.2)
    assert boxes[0]['length'] == pytest.approx(10.)
    assert boxes[0]['observed_only']
    assert tracker.observations[0]['confirmed']


def test_ground_self_noise_removed_but_low_obstacle_kept():
    tracker = LidarObstacleTracker()
    ground = side(); ground[:, 2] = .1
    ego = side(x=0., y=1., length=2.)
    low = side(x=6., length=1.); low[:, 2] = .4
    points = np.vstack([ground, ego, low, [[8., 10., 1.], [np.nan, 1., 1.]]])
    run(tracker, points, 0.)
    assert len(tracker.observations) == 1
    assert tracker.observations[0]['length_m'] == pytest.approx(1.)


def test_longitudinal_chassis_gap_joins_but_adjacent_bus_stays_separate():
    tracker = LidarObstacleTracker()
    rear = side(x=4., y=8., length=2.)
    cab = side(x=4., y=11.6, length=2.)  # 1.6 m gap between observed ends
    bus = side(x=-4., y=10., length=12.)
    groups = tracker.components(np.vstack([rear, cab, bus]))
    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == sorted([len(rear)+len(cab), len(bus)])


def test_lateral_and_distant_longitudinal_gaps_remain_separate():
    tracker = LidarObstacleTracker()
    assert len(tracker.components(np.vstack([
        side(x=4., length=2.), side(x=5., length=2.)]))) == 2
    assert len(tracker.components(np.vstack([
        side(x=4., y=8., length=2.), side(x=4., y=13., length=2.)]))) == 2


def footprint(x, y, length=4.5, width=1.8, top=1.6):
    along = np.linspace(-length/2, length/2, 30)
    across = np.linspace(-width/2, width/2, 15)
    xy = np.vstack([np.c_[np.full(30, x-width/2), y+along],
                    np.c_[np.full(30, x+width/2), y+along],
                    np.c_[x+across, np.full(15, y-length/2)],
                    np.c_[x+across, np.full(15, y+length/2)]])
    return np.vstack([np.c_[xy, np.full(len(xy), z)] for z in (.5, .8, top)])


def test_low_thin_rails_removed_without_removing_nearby_npc():
    tracker = LidarObstacleTracker()
    rails = [footprint(-8., y, length=1.5, width=.4, top=1.) for y in (-8., 0., 8.)]
    car = footprint(-5., 8.)
    bus = footprint(5., 15., length=12., width=2.5, top=3.2)
    groups = tracker.components(np.vstack(rails+[car, bus]))
    assert len(groups) == 2
    assert sorted(map(len, groups)) == sorted([len(car), len(bus)])
    assert tracker.cluster_counts['lidar_rail_rejected'] == 3


def test_long_low_thin_structure_removed_but_tall_vehicle_side_retained():
    tracker = LidarObstacleTracker()
    rail = footprint(-7., 10., length=8., width=.4, top=1.)
    vehicle_side = side(x=5., length=10.)
    groups = tracker.components(np.vstack([rail, vehicle_side]))
    assert len(groups) == 1 and len(groups[0]) == len(vehicle_side)


def test_two_cars_with_short_bumper_gap_are_not_joined():
    tracker = LidarObstacleTracker()
    front, rear = footprint(5., 15.), footprint(5., 9.)  # 1.5 m bumper gap
    groups = tracker.components(np.vstack([front, rear]))
    assert len(groups) == 2
    assert tracker.cluster_counts['lidar_fragment_merges'] == 0


@pytest.mark.parametrize('angle_deg', [0., 17., 45., 88., 122.])
def test_lshape_fits_two_unevenly_sampled_car_faces(angle_deg):
    angle = math.radians(angle_deg)
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    center = np.array([5., 12.])
    # The long side has many more returns than the rear face; PCA rotates.
    a = np.linspace(-2.25, 2.25, 160)
    b = np.linspace(-.9, .9, 20)
    xy = np.vstack([center+a[:, None]*u-.9*v,
                    center-2.25*u+b[:, None]*v])
    measured = LidarObstacleTracker.measure(np.c_[xy, np.ones(len(xy))])
    assert abs(measured['heading']@u) >= math.cos(math.radians(1.1))
    assert measured['length'] == pytest.approx(4.5, abs=.06)
    assert measured['width'] == pytest.approx(1.8, abs=.09)
    np.testing.assert_allclose(measured['center'], center, atol=.06)
    # Entire observed point set stays inside the fitted rectangle.
    mu = measured['heading']; mv = np.array([-mu[1], mu[0]])
    delta = xy-measured['center']
    assert np.max(abs(delta@mu)) <= measured['length']/2+1e-8
    assert np.max(abs(delta@mv)) <= measured['width']/2+1e-8


def test_lshape_single_face_does_not_invent_car_width():
    measured = LidarObstacleTracker.measure(side(length=4.5))
    assert measured['width'] == pytest.approx(.2)
    assert measured['length'] == pytest.approx(4.4)


def test_observed_motion_deceleration_and_ego_compensation():
    tracker = LidarObstacleTracker()
    y = 10.
    for i in range(20):
        speed = max(0., 2.-i*.15)
        y += speed*.1
        pose = (i*.1, 0., .2)
        cloud = side(y=y)
        wx, wy = cloud[:, 1]-pose[0], -cloud[:, 0]
        cloud[:, 0] = math.sin(.2)*wx-math.cos(.2)*wy
        cloud[:, 1] = math.cos(.2)*wx+math.sin(.2)*wy
        run(tracker, cloud, i*.1, pose=pose)
        assert len(tracker.observations) == 1
        if i >= 2:
            assert tracker.observations[0]['speed_mps'] == pytest.approx(speed, abs=1e-8)


def test_missing_points_no_ghost_and_reconfirmation_required():
    tracker = LidarObstacleTracker()
    for i in range(3):
        run(tracker, side(), i*.1)
    assert run(tracker, None, .3)[0] == []
    assert tracker.observations == []
    assert run(tracker, side(), .4)[0] == []
    assert run(tracker, side(), 1.5)[0] == []
    assert tracker.observations[0]['consecutive_hits'] == 1


def test_shape_change_never_reports_fragment_centroid_shift_as_speed():
    tracker = LidarObstacleTracker()
    for i in range(3):
        run(tracker, side(), i*.1)
    run(tracker, side(y=10.2, length=9.5), .3)
    assert tracker.observations[0]['speed_mps'] is None


def test_duplicate_ve_boxes_one_component_but_distinct_boxes_preserved():
    tracker = LidarObstacleTracker()
    boxes, counts = run(tracker, side(), 0., [ve(), ve(y=10.2, index=2)])
    assert len(boxes) == 1
    assert counts['lidar_ve_duplicates'] == 1
    boxes, counts = run(tracker, side(), .1, [ve(y=7.), ve(y=13., index=2)])
    assert len(boxes) == 2
    assert counts['lidar_ve_duplicates'] == 0


def test_split_merge_ambiguous_no_confirmed_backup():
    tracker = LidarObstacleTracker()
    for i in range(3):
        run(tracker, side(length=3.), .1*i)
    split = np.vstack([side(x=3.5, length=3.), side(x=4.5, length=3.)])
    boxes, counts = run(tracker, split, .3)
    assert counts['lidar_ambiguous'] == 2
    assert boxes == []
    assert all(o['speed_mps'] is None and not o['confirmed'] for o in tracker.observations)


def test_decode_unknown_speed_observed_metadata_and_wire_roundtrip():
    head = np.zeros((2500, 8)); head[:, 0] = -20.; head[-1, 7] = np.nan
    tracker = LidarObstacleTracker()
    run(tracker, side(), -.1, [ve()])
    for i in range(3):
        boxes, _ = decode_world_boxes(head, 0., 0., 0., lidar_tracker=tracker,
            body_points_3d=side(length=10.-i*.4), observation_time=i*.1)
    assert len(boxes) == 1
    wire = boxes[0].to_wire()
    assert wire['speed_mps'] is None
    assert wire['source'] == 'lidar' and wire['extent_kind'] == 'observed'
    assert wire['observation_time'] == .2
    assert Box.from_wire(wire).to_wire() == wire


def test_reset_does_not_continue_old_identity():
    tracker = LidarObstacleTracker()
    for i in range(3):
        run(tracker, side(), 1.+i*.1)
    old_id = tracker.observations[0]['track_id']
    assert run(tracker, side(), .1)[0] == []
    assert tracker.observations[0]['track_id'] != old_id


def test_merged_extent_resets_confirmation_even_with_same_center():
    tracker = LidarObstacleTracker()
    for i in range(3):
        run(tracker, side(), i*.1)
    boxes, _ = run(tracker, side(length=14.), .3)
    assert boxes == []
    assert not tracker.observations[0]['confirmed']
    assert tracker.observations[0]['speed_mps'] is None


def test_never_ve_seen_cluster_is_confirmed_independently():
    tracker = LidarObstacleTracker()
    for i in range(10):
        boxes, counts = run(tracker, side(length=3.), i*.1)
        assert len(boxes) == int(i >= 2)
        assert counts['lidar_backup'] == int(i >= 2)
        assert len(tracker.observations) == 1


def test_ve_seed_survives_43_seconds_with_current_lidar_and_reappears_without_duplicate():
    tracker = LidarObstacleTracker()
    run(tracker, side(), 0., [ve()])
    identity = tracker.observations[0]['track_id']
    for i in range(1, 45):
        boxes, counts = run(tracker, side(y=10.+i*.1), i*.1)
        assert tracker.observations[0]['track_id'] == identity
        assert 've_seen' not in tracker.observations[0]
        assert len(boxes) == int(i >= 2)
    boxes, counts = run(tracker, side(y=14.5), 4.5, [ve(y=14.5)])
    assert len(boxes) == 1 and counts['lidar_backup'] == 0


def test_missing_lidar_reconfirms_without_ve():
    tracker = LidarObstacleTracker()
    for i in range(3):
        run(tracker, side(), i*.1)
    assert run(tracker, None, .3)[0] == []
    for i in range(4, 7):
        boxes, _ = run(tracker, side(), i*.1)
        assert len(boxes) == int(i == 6)


def test_ve_fragment_changes_do_not_reset_lidar_track():
    tracker = LidarObstacleTracker()
    run(tracker, side(), 0., [ve()])
    identity = tracker.observations[0]['track_id']
    for i in range(1, 10):
        detections = [ve(y=7.), ve(y=13., index=2)] if i % 2 else []
        boxes, counts = run(tracker, side(), i*.1, detections)
        observed = tracker.observations[0]
        assert observed['track_id'] == identity
        assert observed['consecutive_hits'] == i+1
        assert 've_seen' not in observed
        if not detections and i >= 2:
            assert counts['lidar_backup'] == 1
    assert run(tracker, side(length=14.), 1.)[0] == []
    assert not tracker.observations[0]['confirmed']
