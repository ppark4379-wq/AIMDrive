"""A box mostly inside a larger one of the same frame is a duplicate; separate or much smaller objects are kept."""
from AIM2026.Expert.box_dedup import drop_contained


def box(x, y, heading=0., length=4.5, width=2.):
    return dict(center_world=[x, y], heading_deg=heading, length_m=length, width_m=width, speed_mps=0.)


def test_a_piece_of_a_car_and_a_turned_copy_are_dropped():
    piece, car, turned = box(20.5, .2, 30., 1.9, 1.6), box(20., 0.), box(19.8, 0., 55., 4.4, 2.1)
    kept, n = drop_contained([piece, car, turned])
    assert n == 2 and len(kept) == 1 and kept[0] is not piece      # one box per car; the larger of two equals is kept


def test_neighbours_and_a_pedestrian_beside_a_car_are_kept():
    boxes = [box(20., 0.), box(20., 2.3), box(22., .9, 0., .6, .6)]   # two parked cars side by side, a pedestrian at a corner
    kept, n = drop_contained(boxes)
    assert n == 0 and kept == boxes


def test_a_thin_barrier_reported_twice_is_one_object_but_a_line_of_barriers_is_not():
    a, b = box(20., 0., 90., 2.8, .8), box(20.3, .1, 84., 2.8, .9)        # same barrier: overlap only ~40 %
    assert drop_contained([a, b])[1] == 1
    line = [box(20., 3. * k, 90., 2.8, .8) for k in range(3)]             # end to end along the lane
    assert drop_contained(line)[1] == 0
