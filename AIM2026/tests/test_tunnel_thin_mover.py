"""Tunnel map motion judge for thin objects: a steel barricade reported as pieces is no mover; cars are judged as before."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import AIM2026.Expert.tunnel_map as TM
from AIM2026.Expert.src.planning.jamming.section import JammingSection
from AIM2026.Expert.tunnel_map import TunnelObstacleMap

SECTION = JammingSection()
EGO = (-1.2, 20.)                         # section frame: lateral, forward


def world(lateral, forward):
    return SECTION.to_world((lateral, forward))


def packet(boxes, t):
    x, y = world(*EGO)
    return dict(generation='g', sent_ns=int(t*1e9), boxes=boxes,
                ego_pose=dict(x=x, y=y, yaw_grid_rad=math.pi/2-SECTION.rotation, speed_mps=0.))


def box(lateral, forward, length, width, track, local_heading=0., state='uncertain'):
    return dict(center_world=list(world(lateral, forward)), length_m=length, width_m=width, track_id=track,
                heading_deg=math.degrees(math.radians(local_heading)-SECTION.rotation), motion_state=state)


def first_mover_frame(frames, track, share):
    """Frame index at which `track` becomes a mover with THIN_LENGTH_SHARE = share (None: never)."""
    saved = TM.THIN_LENGTH_SHARE
    TM.THIN_LENGTH_SHARE = share
    try:
        m = TunnelObstacleMap()
        for k, boxes in enumerate(frames):
            m.observe_packet(packet(boxes, k*.1))
            if m.motion.is_mover(track):
                return k, m
        return None, m
    finally:
        TM.THIN_LENGTH_SHARE = saved


def test_a_steel_barricade_reported_as_sliding_pieces_stays_a_standing_obstacle():
    # A 3 m barricade across the lane, 15 m ahead: the detector reports 1-2 m pieces from its left end, then its right
    # end, so the box centre slides 2.2 m along the barricade (competition tunnel, 10-28).
    frames = [[box(-4.1+.11*k, 35., (1.2, 1.9)[k % 3 == 0], (.9, 1.)[k % 3 == 0], track=7)] for k in range(21)]
    old, _ = first_mover_frame(frames, 7, 0.)
    assert old is not None                                     # before: the slide made it a mover
    new, m = first_mover_frame(frames, 7, TM.THIN_LENGTH_SHARE)
    assert new is None
    assert m.distance(-3., 35.) == 0.                          # and it stays on the map


def test_a_slow_car_seen_partly_as_fragments_is_a_mover_as_before():
    # 15 m ahead, 1.5 m/s along the lane; every other box a fragment (1.1 x 1.7 m), the rest the car (4.4 x 2.0 m).
    frames = [[box(-1., 35.+.15*k, *((4.4, 2.) if k % 2 else (1.1, 1.7)), track=3, local_heading=90.)]
              for k in range(30)]
    old, _ = first_mover_frame(frames, 3, 0.)
    new, _ = first_mover_frame(frames, 3, TM.THIN_LENGTH_SHARE)
    assert old is not None and new == old


def test_a_thin_object_that_keeps_moving_is_still_a_mover():
    # A 1.2 x 0.8 m object crossing the lane at 2 m/s, 15 m ahead.
    frames = [[box(-6.+.2*k, 35., 1.2, .8, track=4, local_heading=0.)] for k in range(30)]
    new, _ = first_mover_frame(frames, 4, TM.THIN_LENGTH_SHARE)
    assert new is not None
