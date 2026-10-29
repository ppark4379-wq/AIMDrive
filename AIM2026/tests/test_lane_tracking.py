import unittest
from copy import deepcopy

from AIM2026.Inference.vision.lane_tracking import LaneTracker


def detection(x, y=0., identity=1, raw=2.):
    return dict(track_id=identity, center_world=[x, y], length_m=7., width_m=2.5,
                heading_deg=0., planning_heading_deg=0., speed_mps=raw,
                planning_speed_mps=raw, motion_state='moving')


def packet(t, boxes, generation='run'):
    return dict(sensor_received_ns=round(t*1e9), object_received_ns=round(t*1e9),
                generation=generation, boxes=boxes, ego_pose=dict(x=0., y=0.))


class LaneTrackingTest(unittest.TestCase):
    def tracker(self):
        return LaneTracker({'a': [(x, 0.) for x in range(501)],
                            'b': [(x, 3.5) for x in range(501)]})

    def train(self, tracker):
        for i in range(11):
            result = tracker.apply(packet(i*.1, [detection(20+i*1.5, identity=i+1)]))
        return result['boxes'][0]

    def test_observed_motion_overrides_wrong_ve_speed_and_id_changes(self):
        tracker = self.tracker()
        box = self.train(tracker)
        self.assertTrue(box['lane_track_confirmed'])
        self.assertAlmostEqual(box['planning_speed_mps'], 15.)
        self.assertEqual(box['speed_mps'], 2.)
        self.assertEqual(len(tracker.tracks), 1)

    def test_confirmed_box_expires_at_two_seconds(self):
        tracker = self.tracker()
        original = self.train(tracker)
        a = tracker.apply(packet(2., []))['boxes'][0]
        b = tracker.apply(packet(2.9, []))['boxes'][0]
        self.assertEqual(b['track_id'], original['track_id'])
        self.assertAlmostEqual(b['center_world'][0], 63.5)
        self.assertEqual(b['longitudinal_uncertainty_m'], 2.5)
        self.assertLessEqual(a['longitudinal_uncertainty_m'], 2.5)
        self.assertAlmostEqual(b['detection_hold_age_s'], 1.9)
        self.assertEqual(tracker.apply(packet(3., []))['boxes'], [])

    def test_unknown_tracks_expire_instead_of_growing_for_minutes(self):
        tracker = self.tracker()
        tracker.apply(packet(0., [detection(20.)]))
        self.assertTrue(tracker.apply(packet(.2, []))['boxes'])
        self.assertEqual(tracker.apply(packet(.35, []))['boxes'], [])
        self.assertEqual(tracker.apply(packet(70., []))['boxes'], [])
        self.assertEqual(tracker.tracks, {})

    def test_confirmed_tracks_expire_before_association(self):
        tracker = self.tracker()
        old = self.train(tracker)
        self.assertEqual(tracker.apply(packet(5., []))['boxes'], [])
        new = tracker.apply(packet(5.1, [detection(96.5)]))['boxes'][0]
        self.assertNotEqual(old['track_id'], new['track_id'])
        self.assertFalse(new['lane_track_confirmed'])

    def test_large_uncertainty_expires_without_clipping_geometry(self):
        tracker = self.tracker()
        self.train(tracker)
        next(iter(tracker.tracks.values()))['sigma'] = 5.
        self.assertEqual(tracker.apply(packet(3., []))['boxes'], [])

    def test_reacquisition_does_not_feed_predictions_into_fit(self):
        tracker = self.tracker()
        self.train(tracker)
        tracker.apply(packet(2., []))
        history = list(next(iter(tracker.tracks.values()))['history'])
        tracker.apply(packet(2.5, []))
        self.assertEqual(list(next(iter(tracker.tracks.values()))['history']), history)
        result = tracker.apply(packet(2.6, [detection(59., identity=999)]))
        self.assertEqual(len(result['boxes']), 1)
        self.assertFalse(result['boxes'][0]['detection_held'])
        self.assertTrue(result['boxes'][0]['lane_track_confirmed'])

    def test_adjacent_lane_cannot_steal_identity(self):
        tracker = self.tracker()
        original = self.train(tracker)
        rows = tracker.apply(packet(1.1, [detection(36.5, y=3.5)]))['boxes']
        held = next(b for b in rows if b['track_id']==original['track_id'])
        self.assertEqual(held['lane_track'], 'a')
        self.assertTrue(held['detection_held'])
        self.assertEqual(len(rows), 2)

    def test_unconfirmed_speed_is_unknown_not_raw_ve(self):
        tracker = self.tracker()
        b = tracker.apply(packet(0., [detection(20.)]))['boxes'][0]
        self.assertIsNone(b['planning_speed_mps'])
        self.assertFalse(b['lane_track_confirmed'])

    def test_fit_rejects_inconsistent_motion(self):
        self.assertIsNone(LaneTracker.fit([(i*.2, v) for i,v in enumerate([0.,4.,0.,8.,1.,12.])]))

    def test_generation_reset_and_input_not_mutated(self):
        tracker = self.tracker()
        p = packet(0., [detection(20.)]); saved = deepcopy(p)
        tracker.apply(p)
        self.assertEqual(p, saved)
        self.assertEqual(tracker.apply(packet(1., [], 'new'))['boxes'], [])

    def test_prediction_follows_curve(self):
        tracker = LaneTracker({'curve': [(0,0),(10,0),(20,0),(20,10),(20,20),(20,30)]})
        for i in range(11):
            tracker.apply(packet(i*.1, [detection(5+i)]))
        b = tracker.apply(packet(2.5, []))['boxes'][0]
        self.assertAlmostEqual(b['center_world'][0],20.)
        self.assertAlmostEqual(b['center_world'][1],10.)
        self.assertAlmostEqual(b['planning_heading_deg'],90.)

    def test_external_supplement_encodes_footprint_for_unchanged_planner(self):
        from AIM2026.Expert.adapter import planner_inputs
        from AIM2026.Inference.vision.bus_truck import BusTruckSupplement
        tracker = self.tracker();self.train(tracker)
        missing = packet(2.5, [])
        missing['ego_pose']['x'] = 60.
        p = tracker.apply(missing)
        p.update(ego_position_source='GPS', ego_status_source='UDP_status_speed_IMU_heading')
        p['ego_pose'].update(yaw_grid_rad=0., speed_mps=10.)
        original = p['boxes'][0]
        wire = BusTruckSupplement.box(original).to_wire()
        p['boxes'] = [dict(original, **wire)]
        _, objects = planner_inputs(p)
        obj = objects[0]
        self.assertEqual(obj.size_x, original['length_m']+5.)
        self.assertLessEqual(obj.size_y, original['width_m']+5.)
        self.assertFalse(hasattr(obj, 'speed_uncertainty_mps'))
        self.assertNotIn('unresolved_lane_vehicles', wire)


if __name__ == '__main__':
    unittest.main()
