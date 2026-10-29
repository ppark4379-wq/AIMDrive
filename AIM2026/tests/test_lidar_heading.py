"""Vehicle boxes take their heading from the LiDAR outline; centre and size stay the detector's."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AIM2026.Inference.vision import postprocess


def car_outline(center, heading_deg, length=4.8, width=2.0, seed=0):
    """Points on the two sides of a car facing the sensor at the origin (an L shape), with 3 cm noise."""
    rng = np.random.default_rng(seed)
    a = math.radians(heading_deg)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.asarray(center, dtype=float)
    corners = [c + s1 * u * length / 2 + s2 * v * width / 2 for s1 in (-1, 1) for s2 in (-1, 1)]
    nearest = min(corners, key=np.linalg.norm)
    side_u = -np.sign((nearest - c) @ u) * u
    side_v = -np.sign((nearest - c) @ v) * v
    pts = [nearest + t * side_u for t in np.linspace(0, length, 40)] + [nearest + t * side_v for t in np.linspace(0, width, 15)]
    return np.asarray(pts) + rng.normal(0, .03, (len(pts), 2))


def box(center, heading_deg, length=4.8, width=2.0):
    a = math.radians(heading_deg)
    b = postprocess._rectangle(center, [math.cos(a), math.sin(a)], length, width)
    b.update(score=.9, cell_index=1, partial=False)
    return b


def axis_error(heading, expected_deg):
    d = (math.degrees(math.atan2(heading[1], heading[0])) - expected_deg) % 180
    return min(d, 180 - d)


class LidarHeadingTest(unittest.TestCase):
    def test_a_turned_box_takes_the_outline_heading_and_keeps_centre_and_size(self):
        for true in (0., 30., 60., 90., 135.):
            with self.subTest(true=true):
                centre = (3., 12.)
                boxes = [box(centre, true + 55.)]           # head is 55 degrees off
                self.assertEqual(postprocess.refine_heading_with_lidar(boxes, car_outline(centre, true)), 1)
                self.assertLess(axis_error(boxes[0]['heading'], true), 3.)
                self.assertLess(np.linalg.norm(boxes[0]['center'] - np.asarray(centre)), .15)
                self.assertAlmostEqual(boxes[0]['length'], 4.8, delta=.15)

    def test_forward_sign_follows_the_head(self):
        boxes = [box((0., 10.), 100.)]
        postprocess.refine_heading_with_lidar(boxes, car_outline((0., 10.), 60.))
        self.assertGreater(boxes[0]['heading'] @ np.array([math.cos(math.radians(100.)), math.sin(math.radians(100.))]), 0)

    def test_only_the_rear_face_visible_keeps_the_head_heading(self):
        # A car straight ahead and aligned: the LiDAR sees a 2 m line across the lane.
        rear = np.array([[x, 10.] for x in np.linspace(-1., 1., 30)])
        boxes = [box((0., 12.4), 90.)]
        self.assertEqual(postprocess.refine_heading_with_lidar(boxes, rear), 0)
        self.assertLess(axis_error(boxes[0]['heading'], 90.), 1e-6)

    def test_a_short_box_on_a_long_barrier_is_turned_and_lengthened(self):
        # A 3.2 m barrier the head reports as a 2.4 x 0.9 m box 42 degrees off.
        barrier = np.array([[x * math.cos(math.radians(30.)), 12. + x * math.sin(math.radians(30.))]
                            for x in np.linspace(-1.6, 1.6, 40)])
        boxes = [box((0., 12.), 72., length=2.4, width=.9)]
        self.assertEqual(postprocess.refine_heading_with_lidar(boxes, barrier), 1)
        self.assertLess(axis_error(boxes[0]['heading'], 30.), 2.)
        self.assertGreater(boxes[0]['length'], 3.1)

    def test_two_barriers_touching_at_an_angle_keep_the_head_box(self):
        # Two 3.2 m barriers end to end, 0.4 m apart, the second turned 55 degrees. The head box sits near the joint,
        # so both are within the seed radius and form one cluster (outline 5.4 x 2.6 m) that still passes the
        # elongation gate; its rectangle came out 6.2 x 3.6 m (live tunnel: 5.4-6.6 x 3.6-4.2 m) into the bypass.
        a = math.radians(55.)
        first = [[0., 12. + t] for t in np.linspace(-3.2, 0., 33)]
        second = [[math.sin(a) * t, 12.4 + math.cos(a) * t] for t in np.linspace(0., 3.2, 33)]
        boxes = [box((.3, 12.3), 35., length=2.35, width=1.3)]
        self.assertEqual(postprocess.refine_heading_with_lidar(boxes, np.array(first + second)), 0)
        self.assertAlmostEqual(boxes[0]['length'], 2.35)

    def test_a_head_box_wider_than_long_keeps_its_short_side_as_the_width(self):
        # Two 1.36 m firewalls end to end across the lane (competition tunnel): the head reports 0.6 m long (along the
        # road) x 2.2 m wide; the outline turns the box across the lane, where 0.6 m must stay the depth (not 2.2 m).
        line = np.array([[x, 12.] for x in np.linspace(-1.45, 1.45, 60)])
        boxes = [box((0., 12.1), 90., length=.6, width=2.2)]
        self.assertEqual(postprocess.refine_heading_with_lidar(boxes, line), 1)
        self.assertLess(axis_error(boxes[0]['heading'], 0.), 2.)
        self.assertGreater(boxes[0]['length'], 2.8)
        self.assertLess(boxes[0]['width'], .8)

    def test_a_small_firewall_across_the_lane_takes_the_outline_heading(self):
        # A single 1.36 m firewall across the lane, the head box 1.1 x 0.7 m turned 40 degrees off.
        line = np.array([[x, 12.] for x in np.linspace(-.68, .68, 30)])
        boxes = [box((0., 12.1), 40., length=1.1, width=.7)]
        self.assertEqual(postprocess.refine_heading_with_lidar(boxes, line), 1)
        self.assertLess(axis_error(boxes[0]['heading'], 0.), 2.)

    def test_a_car_fragment_box_is_not_turned_by_the_car_rear(self):
        # The head boxed a car seen from behind as a 0.7 x 0.4 m fragment; its 1.9 m rear face must not turn it.
        rear = np.array([[x, 10.] for x in np.linspace(-.95, .95, 40)])
        boxes = [box((0., 10.3), 90., length=.7, width=.4)]
        self.assertEqual(postprocess.refine_heading_with_lidar(boxes, rear), 0)

    def test_round_piles_and_small_props_keep_the_head_box(self):
        rng = np.random.default_rng(1)
        pile = rng.normal(0, .5, (80, 2)) + np.array([0., 10.])      # irregular, not elongated
        boxes = [box((0., 10.), 20., length=3.0, width=2.5)]
        self.assertEqual(postprocess.refine_heading_with_lidar(boxes, pile), 0)
        barrel = rng.normal(0, .15, (40, 2)) + np.array([0., 10.])
        self.assertEqual(postprocess.refine_heading_with_lidar([box((0., 10.), 20., 1., .6)], barrel), 0)

    def test_sparse_points_are_left_alone(self):
        sparse = [box((0., 10.), 20.)]
        self.assertEqual(postprocess.refine_heading_with_lidar(sparse, car_outline((0., 10.), 60.)[:5]), 0)
        self.assertEqual(postprocess.refine_heading_with_lidar(sparse, None), 0)
        self.assertLess(axis_error(sparse[0]['heading'], 20.), 1e-6)


    def test_two_cells_of_one_car_become_one_box_after_the_heading_is_measured(self):
        centre = np.array([2., 14.])
        a = math.radians(60.)
        u = np.array([math.cos(a), math.sin(a)])
        points = car_outline(centre, 60.)
        # Two cells on one car: same centre region, headings 0 and 90 degrees - too little overlap for NMS before.
        values = np.full((2500, 8), -20.)
        values[:, 1:] = 0.
        for index, deg, score in ((1225, 0., 3.), (1226, 90., 2.)):
            row, col = divmod(index, 50)
            target = centre + (u * .2 if index == 1226 else 0)
            values[index] = [score, (target[0] - (col - 24.5)) / 3.5, (target[1] - (29.5 - row)) / 3.5,
                             math.cos(math.radians(deg)), -math.sin(math.radians(deg)), 4.8 / 7., 2.0 / 4., 0.]
        before, _ = postprocess.decode_boxes(values)
        self.assertEqual(len(before), 2)
        boxes, counts = postprocess.decode_world_boxes(values, 0., 0., math.pi / 2, lidar_points=points)
        self.assertEqual(counts['lidar_heading'], 2)
        self.assertEqual(len(boxes), 1)


if __name__ == '__main__':
    unittest.main()


def test_cluster_centres_ride_on_the_wire_only_with_lidar_evidence():
    from AIM2026.UDP.Detection import Box
    plain = Box(1., 2., 0., 1., 2., None)
    assert 'lidar_center' not in plain.to_wire()
    rich = Box(1., 2., 0., 1., 2., None, (1.1, 2.1), 30)
    row = rich.to_wire()
    assert row['lidar_center'] == [1.1, 2.1] and row['lidar_points'] == 30
    assert Box.from_wire(row) == rich


def test_a_short_box_grows_over_its_cluster_but_a_car_or_wall_cluster_does_not():
    import numpy as np
    from AIM2026.Inference.vision.postprocess import _cover
    box = dict(center=np.array([0., 10.]), heading=np.array([0., 1.]), length=.9, width=.8)
    barrier = np.array([[0., 10. + d] for d in np.linspace(-.9, .9, 20)])          # 1.8 m barrier, box shows half
    grown = _cover(box, barrier)
    assert abs(grown['length'] - 1.8) < 1e-6 and abs(grown['center'][1] - 10.) < 1e-6
    wall = np.array([[0., 10. + d] for d in np.linspace(-3., 3., 40)])             # 6 m: not this object
    assert _cover(box, wall) is box
    inside = np.array([[0., 10.2], [.1, 9.9]])
    assert _cover(box, inside) is box                                              # never shrunk, unchanged


def _bus(center=(3., 12.), length=12., width=2.5, height=1.8):
    """Left side and rear of a bus driving along +forward to the right of the sensor (an L shape), 3 cm noise."""
    import numpy as np
    rng = np.random.default_rng(1)
    x0, y0 = center[0] - width / 2, center[1] - length / 2
    pts = [(x0, y0 + t) for t in np.linspace(0, length, 120)] + [(x0 + t, y0) for t in np.linspace(0, width, 25)]
    pts = np.asarray(pts) + rng.normal(0, .03, (len(pts), 2))
    return pts, np.full(len(pts), height)


def _head(cells):
    """Raw (2500, 8) head with the given (right, forward, score, length, width) cells heading +forward."""
    import numpy as np
    values = np.zeros((2500, 8)); values[:, 0] = -20.
    for right, forward, score, length, width in cells:
        col, row = int(round(right + 24.5)), int(round(29.5 - forward))
        values[row * 50 + col] = [math.log(score / (1 - score)), (right - (col - 24.5)) / 3.5,
                                  (forward - (29.5 - row)) / 3.5, 0., -1., length / 7., width / 4., 1.]
    return values


def test_a_short_bus_box_grows_to_the_lidar_outline():
    pts, h = _bus()
    boxes, counts = postprocess.decode_world_boxes(_head([(3., 13.5, .8, 8.5, 2.4)]), 0., 0., math.pi / 2,
                                                   big_points=(pts, h))
    assert counts['big_grown'] == 1 and len(boxes) == 1
    assert abs(boxes[0].length_m - 12.) < .3


def test_a_missed_bus_candidate_is_rescued_but_not_without_the_outline():
    pts, h = _bus()
    boxes, counts = postprocess.decode_world_boxes(_head([(3., 12., .3, 8., 2.4)]), 0., 0., math.pi / 2,
                                                   big_points=(pts, h))
    assert counts['big_rescued'] == 1 and len(boxes) == 1 and boxes[0].length_m > 11.5
    _, counts = postprocess.decode_world_boxes(_head([(3., 12., .3, 8., 2.4)]), 0., 0., math.pi / 2)
    assert counts['decoded_boxes'] == 0                       # without the LiDAR option nothing changes


def test_guardrails_walls_and_cars_are_not_buses():
    import numpy as np
    head = _head([(3., 12., .3, 8., 2.4)])
    pts, h = _bus()
    low = postprocess.decode_world_boxes(head, 0., 0., math.pi / 2, big_points=(pts, np.full(len(h), .8)))[1]
    assert low['big_rescued'] == 0                             # a 0.8 m high barrier
    wall = np.array([[1.75, f] for f in np.linspace(-5., 30., 300)])
    assert postprocess.decode_world_boxes(head, 0., 0., math.pi / 2,
                                          big_points=(wall, np.full(len(wall), 1.8)))[1]['big_rescued'] == 0
    car, hc = _bus(length=4.6, width=1.9, height=1.5)
    assert postprocess.decode_world_boxes(head, 0., 0., math.pi / 2, big_points=(car, hc))[1]['big_rescued'] == 0
