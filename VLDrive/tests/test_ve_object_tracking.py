import unittest
from inference.ve_object_tracking import VeObjectTracker


def box(x, y=0., speed=3., heading=0., length=4., width=2.):
    return dict(center_world=[x, y], speed_mps=speed, heading_deg=heading,
                length_m=length, width_m=width)


def packet(t, boxes, generation='test'):
    return dict(generation=generation, sent_ns=round((1 + t) * 1e9), boxes=boxes)


class VeTrackingTests(unittest.TestCase):
    def test_reordered_boxes_get_distinct_stable_ids_and_motion_states(self):
        tracker = VeObjectTracker()
        first = tracker.update(packet(0., [box(0.), box(10., speed=0.)]))['boxes']
        ids = [b['track_id'] for b in first]
        for i in range(1, 8):
            rows = tracker.update(packet(i * .1, [box(10., speed=0.), box(i * .3)]))['boxes']
        self.assertEqual([ids[1], ids[0]], [b['track_id'] for b in rows])
        self.assertEqual(['stationary', 'moving'], [b['motion_state'] for b in rows])
        self.assertEqual(0., rows[0]['planning_speed_mps'])

    def test_close_tracks_cross_with_unique_assignment(self):
        tracker = VeObjectTracker()
        ids = [b['track_id'] for b in tracker.update(packet(0., [box(-2.), box(2., .5, heading=180.)]))['boxes']]
        for i in range(1, 11):
            rows = tracker.update(packet(i * .1, [box(2. - i * .3, .5, heading=180.), box(-2. + i * .3)]))['boxes']
            self.assertEqual([ids[1], ids[0]], [b['track_id'] for b in rows])

    def test_heading_flip_is_resolved_from_motion_without_rotating_obb(self):
        tracker = VeObjectTracker()
        for i in range(8):
            raw = box(-i * .4, speed=4., heading=0.)
            row = tracker.update(packet(i * .1, [raw]))['boxes'][0]
        self.assertTrue(row['motion_direction_observed'])
        self.assertAlmostEqual(abs(row['planning_heading_deg']), 180.)
        self.assertEqual(row['heading_deg'], 0.)
        self.assertEqual(row['center_world'], raw['center_world'])
        self.assertEqual(row['width_m'], raw['width_m'])

    def test_empty_detections_emit_no_held_boxes_and_expired_id_is_not_reused(self):
        tracker = VeObjectTracker()
        old = tracker.update(packet(0., [box(0.)]))['boxes'][0]['track_id']
        self.assertEqual([], tracker.update(packet(.1, []))['boxes'])
        new = tracker.update(packet(1., [box(0.)]))['boxes'][0]['track_id']
        self.assertNotEqual(old, new)

    def test_generation_change_does_not_alias_old_tracks(self):
        tracker = VeObjectTracker()
        old = tracker.update(packet(0., [box(0.)]))['boxes'][0]['track_id']
        new = tracker.update(packet(.1, [box(0.)], 'new'))['boxes'][0]['track_id']
        self.assertNotEqual(old, new)

    def test_stationary_jitter_does_not_become_motion_and_restart_releases_static(self):
        tracker = VeObjectTracker()
        for i in range(10):
            row = tracker.update(packet(i * .1, [box(.06 * (-1) ** i, speed=.1)]))['boxes'][0]
        self.assertEqual('stationary', row['motion_state'])
        identity = row['track_id']
        row = tracker.update(packet(1., [box(.3, speed=3.)]))['boxes'][0]
        self.assertEqual(identity, row['track_id'])
        self.assertNotEqual('stationary', row['motion_state'])

    def test_motion_can_be_estimated_when_scalar_speed_is_missing(self):
        tracker = VeObjectTracker()
        for i in range(8):
            row = tracker.update(packet(i * .1, [box(i * .2, speed=None)]))['boxes'][0]
        self.assertEqual('moving', row['motion_state'])
        self.assertAlmostEqual(2., row['planning_speed_mps'])


if __name__ == '__main__':
    unittest.main()
