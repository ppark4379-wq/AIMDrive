import unittest

from AIM2026.Expert.track_alias import TrackAliaser


def box(track_id, x, y, length, width, state='uncertain', speed=0.):
    return dict(track_id=track_id, center_world=[x, y], heading_deg=0., length_m=length, width_m=width,
                motion_state=state, speed_mps=speed)


def packet(t, *boxes):
    return dict(generation=1, sent_ns=int(t * 1e9), boxes=list(boxes))


class TrackAliasTests(unittest.TestCase):
    def test_alternating_piece_and_whole_get_one_id(self):
        alias = TrackAliaser()
        ids = []
        for k in range(10):
            b = box(1, 0., 0., 2.3, 1.8) if k % 2 == 0 else box(2, .9, .3, .9, 1.5)
            ids.append(alias.apply(packet(k * .1, b))['boxes'][0]['track_id'])
        self.assertEqual(set(ids), {1})

    def test_neighbours_seen_together_are_never_merged(self):
        alias = TrackAliaser()
        alias.apply(packet(0., box(1, 0., 0., .6, .6), box(2, 1.2, 0., .6, .6)))
        out = alias.apply(packet(.1, box(2, 1.2, 0., .6, .6)))      # barrel 1 missed this frame
        self.assertEqual(out['boxes'][0]['track_id'], 2)

    def test_moving_track_keeps_its_id(self):
        alias = TrackAliaser()
        alias.apply(packet(0., box(1, 0., 0., 4.5, 1.9)))
        out = alias.apply(packet(.1, box(2, .8, 0., 4.5, 1.9, state='moving', speed=3.)))
        self.assertEqual(out['boxes'][0]['track_id'], 2)

    def test_far_apart_tracks_are_not_merged(self):
        alias = TrackAliaser()
        alias.apply(packet(0., box(1, 0., 0., 1., 1.)))
        out = alias.apply(packet(.1, box(2, 3., 0., 1., 1.)))
        self.assertEqual(out['boxes'][0]['track_id'], 2)

    def test_old_track_expires(self):
        alias = TrackAliaser()
        alias.apply(packet(0., box(1, 0., 0., 2., 1.)))
        out = alias.apply(packet(1.5, box(2, .5, 0., 2., 1.)))
        self.assertEqual(out['boxes'][0]['track_id'], 2)

    def test_two_boxes_in_one_frame_never_share_an_id(self):
        alias = TrackAliaser()
        alias.apply(packet(0., box(1, 0., 0., 2.3, 1.8)))
        alias.apply(packet(.1, box(2, .9, .3, .9, 1.5)))              # merged into 1
        out = alias.apply(packet(.2, box(1, 0., 0., 2.3, 1.8), box(2, .9, .3, .9, 1.5)))
        self.assertEqual(sorted(b['track_id'] for b in out['boxes']), [1, 2])


if __name__ == '__main__':
    unittest.main()
