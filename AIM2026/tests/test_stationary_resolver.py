"""Parked objects with wandering box centres become stationary; movers and starting objects never do."""
import random

from AIM2026.Expert.stationary_resolver import StationaryResolver


def box(x, y, state='uncertain', track=7, speed=.2, length=4.5):
    return dict(center_world=[x, y], heading_deg=0., length_m=length, width_m=2., speed_mps=speed, track_id=track,
                motion_state=state, planning_speed_mps=speed, planning_speed_source='speed_unavailable')


def packet(t, boxes):
    return dict(generation='g', sent_ns=int(t * 1e9), boxes=boxes)


def states(resolver, centres, state='uncertain', start=0., dt=.1, **kw):
    out = []
    for i, (x, y) in enumerate(centres):
        out.append(resolver.apply(packet(start + i * dt, [box(x, y, state, **kw)]))['boxes'][0]['motion_state'])
    return out


def test_a_parked_car_with_jittering_centre_becomes_stationary_after_0_6_seconds():
    rng = random.Random(1)
    centres = [(20. + rng.uniform(-.25, .25), rng.uniform(-.25, .25)) for _ in range(30)]
    out = states(StationaryResolver(), centres)
    assert out[:6] == ['uncertain'] * 6          # less than 0.6 s of history: undecided
    assert out[-10:] == ['stationary'] * 10


def test_short_detection_gaps_do_not_restart_the_history():
    resolver, rng = StationaryResolver(), random.Random(2)
    states(resolver, [(20. + rng.uniform(-.2, .2), 0.) for _ in range(12)])
    # 0.6 s without the object, then it is back: already decided
    assert states(resolver, [(20., 0.)], start=1.8) == ['stationary']


def test_a_moving_object_and_its_moving_label_are_never_changed():
    assert set(states(StationaryResolver(), [(1.5 * i * .1, 0.) for i in range(30)])) == {'uncertain'}   # 1.5 m/s
    assert set(states(StationaryResolver(), [(20., 0.)] * 20, state='moving')) == {'moving'}


def test_a_pedestrian_that_starts_to_walk_is_released_quickly():
    resolver = StationaryResolver()
    assert states(resolver, [(20., 0.)] * 20, length=.6)[-1] == 'stationary'
    # walks off at 1.4 m/s: back to uncertain within half a second
    out = states(resolver, [(20. + 1.4 * i * .1, 0.) for i in range(1, 8)], start=2., length=.6)
    assert 'uncertain' in out[:5]


def test_a_vehicle_that_drives_off_is_released_within_a_second():
    resolver = StationaryResolver()
    assert states(resolver, [(20., 0.)] * 20)[-1] == 'stationary'
    out = states(resolver, [(20. + 1.4 * i * .1, 0.) for i in range(1, 12)], start=2.)
    assert 'uncertain' in out[:9]


def test_a_fast_detector_speed_keeps_it_uncertain():
    assert set(states(StationaryResolver(), [(20., 0.)] * 25, speed=3.)) == {'uncertain'}

