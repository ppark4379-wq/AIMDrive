"""Accumulated static-section occupancy map: evidence, occlusion, release and pass-through."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AIM2026.Expert.static_map import StaticObstacleMap, ID_BASE

EGO = dict(x=-60., y=-160., yaw_grid_rad=math.pi/2, speed_mps=0.)


def box(x, y, length=1.4, width=1., heading=90., state='uncertain', source='unconfirmed', track=1):
    return dict(center_world=[x, y], heading_deg=heading, planning_heading_deg=heading, length_m=length,
                width_m=width, motion_state=state, planning_speed_mps=0., planning_speed_source=source,
                track_id=track)


def frame(m, boxes, ego=EGO, generation='g'):
    out = m.apply(dict(generation=generation, ego_pose=ego, boxes=boxes))
    return [b for b in out['boxes'] if b.get('planning_speed_source') == 'static_map'], out


def test_repeated_detections_become_one_stable_box_and_survive_a_missed_frame():
    m = StaticObstacleMap()
    mapped, _ = frame(m, [box(-60.6, -142., track=1)])
    assert mapped == []                                     # one frame is not evidence enough
    for track in (2, 3, 4):                                 # the tracker keeps changing the ID
        mapped, _ = frame(m, [box(-60.6+.1*(track % 2), -142., track=track)])
    assert len(mapped) == 1 and mapped[0]['track_id'] >= ID_BASE
    first = mapped[0]
    mapped, _ = frame(m, [])                                # detector misses one frame
    assert len(mapped) == 1 and mapped[0]['track_id'] == first['track_id']
    assert mapped[0]['motion_state'] == 'stationary' and mapped[0]['planning_speed_mps'] == 0.


def test_single_frame_ghost_and_memory_boxes_do_not_create_obstacles():
    m = StaticObstacleMap()
    frame(m, [box(-55., -140., source='static_memory', track=-3)])
    mapped, out = frame(m, [box(-55., -140., source='static_memory', track=-3)])
    assert mapped == [] and out['boxes'] == []              # memory ghosts in the section are replaced by the map


def test_object_that_disappears_in_view_is_released_but_occluded_one_is_kept():
    m = StaticObstacleMap()
    near, far = box(-60.2, -150., track=1), box(-60.2, -140., track=2)   # far is hidden behind near
    for _ in range(8):
        frame(m, [near, far])
    for _ in range(12):                                     # only the near object is still reported
        mapped, _ = frame(m, [near])
    ys = sorted(round(b['center_world'][1]) for b in mapped)
    assert ys == [-150, -140]                               # the far object was occluded: not erased
    for _ in range(15):                                     # nothing reported, both visible: released
        mapped, _ = frame(m, [])
    assert mapped == []


def test_boxes_outside_the_section_pass_through_unchanged():
    m = StaticObstacleMap()
    outside = [box(-63.5, -140., state='moving', track=7), box(-60., -100., track=8), box(-60., -170., track=9)]
    _, out = frame(m, outside)
    assert out['boxes'] == outside


def test_neighbouring_objects_stay_separate():
    m = StaticObstacleMap()
    for _ in range(5):
        mapped, _ = frame(m, [box(-54.3, -143.4, track=1), box(-54.1, -140.9, track=2), box(-53.9, -138.1, track=3)])
    assert len(mapped) == 3
    assert all(b['length_m'] < 1.8 for b in mapped)


def test_new_publisher_generation_resets_the_map():
    m = StaticObstacleMap()
    for _ in range(4):
        frame(m, [box(-60.6, -142.)])
    mapped, _ = frame(m, [], generation='other')
    assert mapped == []


def test_distance_is_exact_near_obstacles_and_never_longer_than_true():
    import numpy as np
    m = StaticObstacleMap()
    for _ in range(4):
        frame(m, [box(-56.35, -146.6, length=3.2, width=1.9), box(-60.9, -140.5, length=3.8, width=1.2, heading=96., track=2),
                  box(-54.2, -138.1, length=.9, width=.5, heading=27., track=3)])
    rng = np.random.default_rng(1)
    x, y = rng.uniform(m.x0-2, m.x1+2, 5000), rng.uniform(m.y0-2, m.y1+2, 5000)
    occ = np.argwhere(m.occupied)
    left, low = m.x0+occ[:, 1]*m.cell, m.y0+occ[:, 0]*m.cell
    dx = np.maximum(np.maximum(left[None]-x[:, None], 0), x[:, None]-(left[None]+m.cell))
    dy = np.maximum(np.maximum(low[None]-y[:, None], 0), y[:, None]-(low[None]+m.cell))
    truth = np.hypot(dx, dy).min(axis=1)
    fast = m.distance(x, y)
    assert np.all(fast <= truth+1e-9)                       # never longer: no hidden loss of clearance
    near = truth < 1.5
    assert np.allclose(fast[near], truth[near], atol=1e-9)  # exact where planning decisions are made


def test_screening_bound_outside_the_map_is_still_exact():
    import numpy as np
    m = StaticObstacleMap()
    for _ in range(4):
        frame(m, [box(-60.3, -150.1, length=1.6, width=1.3)])
    x, y = np.array([-61.8, -60.0, -55.]), np.array([-178.5, -170., -120.])   # before and after the map
    assert np.allclose(m.distance(x, y, exact_below=0.), m.distance(x, y))
    assert np.all(m.distance(x, y, exact_below=0.) > 10.)
