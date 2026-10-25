"""Small obstacles reported on only some frames stay published for a short time; movers and passed spots do not."""
from AIM2026.Expert.static_memory import StaticMemory


def packet(t, boxes, ego=(0., 0., 0.)):
    # yaw 0 -> forward is +x
    return dict(sent_ns=int(t * 1e9), ego_pose=dict(x=ego[0], y=ego[1], yaw_grid_rad=ego[2]), boxes=list(boxes))


def box(x, y, state='uncertain', speed=0., length=1.0, width=.5):
    return dict(center_world=[x, y], heading_deg=0., length_m=length, width_m=width, speed_mps=speed, track_id=7,
                motion_state=state, planning_speed_mps=speed, observed_speed_mps=speed, planning_heading_deg=0.)


def test_intermittent_small_object_is_published_between_hits():
    memory = StaticMemory()
    for t in (0., .3, .6):
        memory.apply(packet(t, [box(20., 0.)]))
    out = memory.apply(packet(1.5, []))
    assert len(out['boxes']) == 1
    ghost = out['boxes'][0]
    assert ghost['motion_state'] == 'stationary' and ghost['planning_speed_mps'] == 0. and ghost['track_id'] < 0
    assert abs(ghost['center_world'][0] - 20.) < .5


def test_needs_several_recent_hits_and_ends_after_hold():
    memory = StaticMemory()
    memory.apply(packet(0., [box(20., 0.)]))
    assert memory.apply(packet(1., []))['boxes'] == []          # one hit is not enough
    for t in (2., 2.3, 2.6):
        memory.apply(packet(t, [box(20., 0.)]))
    assert memory.apply(packet(5., []))['boxes']                 # remembered
    assert memory.apply(packet(12., []))['boxes'] == []          # hold ended


def test_a_mover_cancels_the_memory_and_large_objects_are_not_remembered():
    memory = StaticMemory()
    for t in (0., .3, .6):
        memory.apply(packet(t, [box(20., 0.)]))
    assert memory.apply(packet(1., [box(20.5, 0., state='moving', speed=3.)]))['boxes'][0]['motion_state'] == 'moving'
    big = StaticMemory()
    for t in (0., .3, .6):
        big.apply(packet(t, [box(30., 0., length=8., width=3.)]))
    assert big.apply(packet(1., []))['boxes'] == []


def test_position_behind_the_vehicle_is_dropped():
    memory = StaticMemory()
    for t in (0., .3, .6):
        memory.apply(packet(t, [box(20., 0.)]))
    assert memory.apply(packet(1., [], ego=(10., 0., 0.)))['boxes']           # 10 m ahead: kept
    assert memory.apply(packet(1.5, [], ego=(30., 0., 0.)))['boxes'] == []     # passed: dropped
