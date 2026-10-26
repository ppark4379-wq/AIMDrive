"""A parked object's footprint is the median of its recent boxes; other labels are reported as they are."""
from AIM2026.Expert.stationary_geometry import StationaryGeometry


def box(x, heading, length=4.5, state='stationary', track=3):
    return dict(center_world=[x, 0.], heading_deg=heading, planning_heading_deg=heading, length_m=length, width_m=2.,
                motion_state=state, track_id=track, planning_speed_source='stationary_long_window')


def run(geometry, boxes, start=0.):
    return [geometry.apply(dict(generation='g', sent_ns=int((start + i * .1) * 1e9), boxes=[b]))['boxes'][0]
            for i, b in enumerate(boxes)]


def test_a_parked_box_that_jumps_and_turns_is_held_at_its_median():
    jumpy = [box(20. + (.8 if i % 4 == 0 else 0.), 0. if i % 5 else 40.) for i in range(20)]
    out = run(StationaryGeometry(), jumpy)
    assert {round(b['center_world'][0], 2) for b in out[-10:]} == {20.}
    assert all(abs(b['heading_deg']) < 10 and b['planning_heading_deg'] == b['heading_deg'] for b in out[-10:])


def test_a_released_track_is_reported_where_it_is():
    geometry = StationaryGeometry()
    run(geometry, [box(20., 0.)] * 10)
    moved = run(geometry, [box(21.5, 0., state='uncertain')], start=1.)[0]
    assert moved['center_world'] == [21.5, 0.]
    again = run(geometry, [box(22., 0.)], start=1.1)[0]                  # a new stationary history starts there
    assert again['center_world'] == [22., 0.]


def test_the_first_boxes_and_the_static_map_are_untouched():
    out = run(StationaryGeometry(), [box(20. + .3 * i, 0.) for i in range(4)])
    assert [b['center_world'][0] for b in out] == [20., 20.3, 20.6, 20.9]
    mapped = dict(box(20., 0.), planning_speed_source='static_map')
    assert run(StationaryGeometry(), [dict(mapped) for _ in range(8)])[-1]['center_world'] == [20., 0.]
