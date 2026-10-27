import math
import unittest
from inference.ve_object_tracking import VeObjectTracker


def box(x, y=0., speed=3., heading=0., length=4., width=2.):
    return dict(center_world=[x, y], speed_mps=speed, heading_deg=heading,
                length_m=length, width_m=width)


def packet(t, boxes, generation='test'):
    return dict(generation=generation, sent_ns=round((1 + t) * 1e9), boxes=boxes)


class VeTrackingTests(unittest.TestCase):
    def test_overlapping_duplicate_keeps_geometry_and_supported_motion(self):
        tracker = VeObjectTracker(moving_carry_s=.75)
        for i in range(8):
            tracker.update(packet(i*.1,[box(i, speed=10., length=5., width=2.2)]))
        raw = box(9.2,.9,speed=10.,length=5.,width=2.2)
        rows = tracker.update(packet(.8,[box(8.,speed=10.,length=5.,width=2.2),raw]))['boxes']
        self.assertEqual(2,len(rows))
        duplicate = rows[1]
        self.assertEqual('overlap_supported',duplicate['planning_speed_source'])
        self.assertEqual('uncertain',duplicate['motion_state'])
        self.assertFalse(duplicate['motion_direction_observed'])
        self.assertEqual(rows[0]['track_id'],duplicate['motion_support_track_id'])
        for key in raw:self.assertEqual(raw[key],duplicate[key])
        # New measured displacement then supplies its own motion evidence.
        rows = tracker.update(packet(1.2,[box(12.,speed=10.,length=5.,width=2.2),
                                          box(13.2,.9,speed=10.,length=5.,width=2.2)]))['boxes']
        self.assertTrue(rows[1]['motion_direction_observed'])

    def test_distinct_or_stopped_boxes_do_not_borrow_motion(self):
        for offset,speed in ((3.5,10.),(.9,0.),(.9,3.)):
            with self.subTest(offset=offset,speed=speed):
                tracker=VeObjectTracker(moving_carry_s=.75)
                for i in range(8):
                    tracker.update(packet(i*.1,[box(i,speed=10.,length=5.,width=2.2)]))
                rows=tracker.update(packet(.8,[box(8.,speed=10.,length=5.,width=2.2),
                    box(9.2,offset,speed=speed,length=5.,width=2.2)]))['boxes']
                self.assertEqual('ve_provisional',rows[1]['planning_speed_source'])

    def test_small_box_size_jitter_preserves_identity_and_raw_geometry(self):
        tracker = VeObjectTracker()
        identity = None
        for i, length in enumerate((.67, .28, .65, .31, .68, .29)):
            raw = box(.03 * (-1)**i, speed=0., length=length, width=.8)
            result = tracker.update(packet(i*.1, [raw]))['boxes'][0]
            if identity is None:
                identity = result['track_id']
            self.assertEqual(identity, result['track_id'])
            for key in raw:
                self.assertEqual(raw[key], result[key])

    def test_adjacent_small_objects_stay_distinct_when_sizes_change(self):
        tracker = VeObjectTracker()
        initial = tracker.update(packet(0., [box(0., speed=0., length=.28, width=.3),
                                             box(.65, speed=0., length=.67, width=.3)]))['boxes']
        ids = [b['track_id'] for b in initial]
        for i in range(1, 8):
            rows = [box(.65, speed=0., length=.28 if i%2 else .67, width=.3),
                    box(0., speed=0., length=.67 if i%2 else .28, width=.3)]
            observed = tracker.update(packet(i*.1, rows))['boxes']
            self.assertEqual([ids[1], ids[0]], [b['track_id'] for b in observed])
            self.assertEqual(2, len(observed))

    def test_short_occlusion_keeps_id_but_incompatible_size_does_not(self):
        tracker = VeObjectTracker()
        small = box(0., speed=0., length=.3, width=.3)
        identity = tracker.update(packet(0., [small]))['boxes'][0]['track_id']
        tracker.update(packet(.1, []))
        result = tracker.update(packet(.4, [small]))['boxes'][0]
        self.assertEqual(identity, result['track_id'])
        large = tracker.update(packet(.5, [box(0., length=5., width=2.)]))['boxes'][0]
        self.assertNotEqual(identity, large['track_id'])

    def test_stale_duplicate_does_not_steal_continuously_observed_track(self):
        tracker = VeObjectTracker()
        def small(x):
            return box(x, speed=0., length=.4, width=.4)
        initial = tracker.update(packet(0., [small(0.), small(.1)]))['boxes']
        identity = initial[0]['track_id']
        for i in range(1, 5):
            result = tracker.update(packet(i*.1, [small(0.)]))['boxes'][0]
            self.assertEqual(identity, result['track_id'])
        # Slightly closer to the old duplicate, but continuous evidence wins.
        result = tracker.update(packet(.5, [small(.06)]))['boxes'][0]
        self.assertEqual(identity, result['track_id'])

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

    def test_high_model_speed_cannot_move_a_stationary_track(self):
        tracker = VeObjectTracker()
        rows = [tracker.update(packet(i*.1, [box(20., speed=14.)]))['boxes'][0]
                for i in range(10)]
        self.assertEqual({1}, {r['track_id'] for r in rows})
        self.assertEqual('uncertain', rows[0]['motion_state'])
        self.assertEqual('stationary', rows[-1]['motion_state'])
        self.assertEqual(14., rows[0]['planning_speed_mps'])
        self.assertEqual('ve_provisional', rows[0]['planning_speed_source'])
        self.assertEqual(0., rows[-1]['planning_speed_mps'])
        self.assertTrue(all(r['speed_mps'] == 14. for r in rows))

    def test_large_oscillation_is_uncertain_not_a_fast_moving_vehicle(self):
        tracker = VeObjectTracker()
        # Alternating metre-scale bbox error: stable identity, no fabricated
        # motion speed, and no falsely precise stationary declaration.
        for i in range(14):
            raw = box(20. + .7*(-1)**i, speed=13.)
            row = tracker.update(packet(i*.12, [raw]))['boxes'][0]
            self.assertEqual(1, row['track_id'])
            self.assertEqual('uncertain', row['motion_state'])
            self.assertEqual(13., row['planning_speed_mps'])
            self.assertFalse(row['motion_direction_observed'])
            self.assertEqual(raw['center_world'], row['center_world'])

    def test_actual_fast_motion_is_recovered_despite_zero_or_missing_raw_speed(self):
        for raw_speed in (0., None):
            with self.subTest(raw_speed=raw_speed):
                tracker = VeObjectTracker()
                for i in range(10):
                    row = tracker.update(packet(i*.1, [box(i*1.5, speed=raw_speed)]))['boxes'][0]
                self.assertEqual(1, row['track_id'])
                self.assertEqual('moving', row['motion_state'])
                self.assertAlmostEqual(15., row['planning_speed_mps'])
                self.assertEqual('position_history', row['planning_speed_source'])

    def test_inconsistent_high_speed_uses_observed_motion(self):
        tracker = VeObjectTracker()
        for i in range(10):
            row = tracker.update(packet(i*.1, [box(i*.3, speed=15.)]))['boxes'][0]
        self.assertEqual('moving', row['motion_state'])
        self.assertAlmostEqual(3., row['planning_speed_mps'])
        self.assertEqual(15., row['speed_mps'])

    def test_gap_discards_motion_confidence_but_retains_current_model_speed(self):
        tracker = VeObjectTracker()
        for i in range(8):
            row = tracker.update(packet(i*.1, [box(i*.3)]))['boxes'][0]
        self.assertEqual('moving', row['motion_state'])
        row = tracker.update(packet(1.1, [box(3.3)]))['boxes'][0]
        self.assertEqual('uncertain', row['motion_state'])
        self.assertEqual(3., row['planning_speed_mps'])
        self.assertEqual('ve_provisional', row['planning_speed_source'])
        self.assertEqual(0., row['track_span_s'])

    def test_small_jitter_does_not_interrupt_supported_moving_track(self):
        tracker = VeObjectTracker()
        for i in range(20):
            # Establish motion before adding lateral noise with a path/net
            # ratio in the hysteresis band.
            y = 0. if i < 8 else .16*(-1)**i
            row = tracker.update(packet(i*.1, [box(i*.3, y=y)]))['boxes'][0]
            if i >= 4:
                self.assertEqual('moving', row['motion_state'])
                self.assertAlmostEqual(3., row['planning_speed_mps'])


class MovingFragmentTests(unittest.TestCase):
    """A partial detection must not erase a measured moving vehicle's history."""

    def moving_track(self):
        tracker = VeObjectTracker(moving_carry_s=.75)
        for i in range(9):
            row = tracker.update(packet(i*.1, [box(i, speed=10., length=4.5, width=2.4)]))['boxes'][0]
        return tracker, row['track_id']

    def test_partial_body_and_full_body_keep_measured_speed_and_geometry(self):
        tracker, identity = self.moving_track()
        for t, raw in ((.9, box(9., speed=.5, length=1.4, width=1.2)),
                       (1., box(10., speed=3., length=4.5, width=2.4))):
            row = tracker.update(packet(t, [raw]))['boxes'][0]
            self.assertEqual(identity, row['track_id'])
            self.assertEqual('moving', row['motion_state'])
            self.assertAlmostEqual(10., row['planning_speed_mps'])
            self.assertEqual('position_history', row['planning_speed_source'])
            for key in raw:
                self.assertEqual(raw[key], row[key])

    def test_full_body_return_is_not_stolen_by_a_young_duplicate(self):
        tracker = VeObjectTracker(moving_carry_s=.75)
        for i in range(8):
            tracker.update(packet(i*.1, [box(i, speed=10., length=4.5, width=2.6)]))
        rows = tracker.update(packet(.8, [box(8., speed=6., length=4.5, width=2.6),
                                           box(12.15, speed=5.4, length=4.4, width=2.1)]))['boxes']
        identity, duplicate = [b['track_id'] for b in rows]
        partial = tracker.update(packet(.94, [box(9.066, speed=2.1, length=1.84, width=1.19)]))['boxes'][0]
        full = tracker.update(packet(1.065, [box(10.812, speed=3.76, length=4.42, width=2.77)]))['boxes'][0]
        self.assertEqual(identity, partial['track_id'])
        self.assertEqual(identity, full['track_id'])
        self.assertNotEqual(duplicate, full['track_id'])
        self.assertGreater(full['planning_speed_mps'], 8.)

    def test_relaxation_excludes_stale_sideways_and_noncontained_boxes(self):
        cases = ((.9, box(9., 1.4, length=1.4, width=1.2)),
                 (.9, box(9., heading=90., length=1.4, width=1.2)),
                 (.9, box(9., length=.4, width=.4)),
                 (1.1, box(11., length=1.4, width=1.2)))
        for t, raw in cases:
            with self.subTest(time=t, raw=raw):
                tracker, identity = self.moving_track()
                row = tracker.update(packet(t, [raw]))['boxes'][0]
                self.assertNotEqual(identity, row['track_id'])
                self.assertEqual('ve_provisional', row['planning_speed_source'])

    def test_unconfirmed_or_stationary_history_cannot_support_fragment(self):
        for count in (1, 9):
            tracker = VeObjectTracker(moving_carry_s=.75)
            for i in range(count):
                row = tracker.update(packet(i*.1, [box(0., speed=10., length=4.5, width=2.4)]))['boxes'][0]
            identity = row['track_id']
            row = tracker.update(packet(count*.1, [box(0., speed=.2, length=1.4, width=1.2)]))['boxes'][0]
            self.assertNotEqual(identity, row['track_id'])
            self.assertEqual(.2, row['planning_speed_mps'])

    def test_actual_stop_is_still_recognized_after_partial_detection(self):
        tracker, identity = self.moving_track()
        row = tracker.update(packet(.9, [box(9., speed=0., length=1.4, width=1.2)]))['boxes'][0]
        self.assertEqual(identity, row['track_id'])
        for i in range(10, 25):
            row = tracker.update(packet(i*.1, [box(9., speed=0., length=1.4, width=1.2)]))['boxes'][0]
        self.assertEqual('stationary', row['motion_state'])
        self.assertEqual(0., row['planning_speed_mps'])


class MovingCarryTests(unittest.TestCase):
    """A slow vehicle that briefly loses its motion evidence must not turn into a phantom stop."""

    def test_confirmed_motion_survives_repeated_slow_observation_intervals(self):
        tracker = VeObjectTracker(moving_carry_s=.75)
        for i in range(10):
            tracker.update(packet(i*.1, [box(i, speed=10.)]))
        for t in (1.3, 1.7, 2.1, 2.5):
            row = tracker.update(packet(t, [box(t*10., speed=10.)]))['boxes'][0]
            self.assertEqual(row['motion_state'], 'moving')
            self.assertAlmostEqual(row['planning_speed_mps'], 10.)

    def test_raw_speed_collapse_disables_carry(self):
        tracker = VeObjectTracker(moving_carry_s=.75)
        self.feed(tracker, [.16*i for i in range(14)])
        row = tracker.update(packet(1.4, [box(2.5, speed=0., length=4.5, width=1.9)]))['boxes'][0]
        self.assertNotEqual(row['planning_speed_source'], 'carry_over')

    @staticmethod
    def feed(tracker, positions, speed=1.6):
        out = None
        for i, x in enumerate(positions):
            out = tracker.update(packet(i * .1, [box(x, speed=speed, length=4.5, width=1.9)]))['boxes'][0]
        return out

    def test_off_by_default(self):
        tracker = VeObjectTracker()
        xs = [.16 * i for i in range(12)]
        self.feed(tracker, xs)
        # box centre jumps back and forth: evidence breaks down
        jitter = [xs[-1] + d for d in (.9, -.5, .8, -.6)]
        row = self.feed(tracker, jitter)
        self.assertEqual(row['planning_speed_mps'], 1.6)
        self.assertEqual(row['planning_speed_source'], 've_provisional')

    def test_carry_keeps_last_speed_then_expires(self):
        tracker = VeObjectTracker(moving_carry_s=1.0)
        xs = [.16 * i for i in range(14)]
        moving = self.feed(tracker, xs)
        self.assertEqual(moving['motion_state'], 'moving')
        speed = moving['planning_speed_mps']
        t = len(xs) * .1
        # evidence lost: the box centre scatters by decimetres around the old position
        scatter = []
        for k, d in enumerate((.9, -.5, .8, -.6, .7)):
            scatter.append(tracker.update(packet(t + k * .1, [box(xs[-1] + d, speed=speed, length=4.5, width=1.9)]))['boxes'][0])
        carried = [b for b in scatter if b['planning_speed_source'] == 'carry_over']
        self.assertTrue(carried)
        self.assertAlmostEqual(carried[0]['planning_speed_mps'], speed, places=6)
        self.assertEqual(carried[0]['motion_state'], 'uncertain')       # label stays honest
        # After the carry window use the current model estimate, never a fabricated stop.
        later = None
        for k in range(14):
            later = tracker.update(packet(t + .5 + k * .1, [box(xs[-1] + (.9, -.5)[k % 2], speed=speed, length=4.5, width=1.9)]))['boxes'][0]
        self.assertEqual(later['planning_speed_mps'], speed)
        self.assertEqual(later['planning_speed_source'], 've_provisional')

    def test_a_stopped_vehicle_is_not_carried(self):
        tracker = VeObjectTracker(moving_carry_s=1.0)
        self.feed(tracker, [.16 * i for i in range(14)])
        final = None
        for k in range(10):                                            # stops dead: clean stationary evidence
            final = tracker.update(packet(1.4 + k * .1, [box(2.1, speed=0., length=4.5, width=1.9)]))['boxes'][0]
        self.assertEqual(final['motion_state'], 'stationary')
        self.assertEqual(final['planning_speed_mps'], 0.)

    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            VeObjectTracker(moving_carry_s=3.)


class StaticGeometryFusionTests(unittest.TestCase):
    @staticmethod
    def moving_ego_packets(boxes_at, steps=30, start=-30., speed=5.):
        for i in range(steps):
            t = i * .1
            ego_y = start + speed * t
            pkt = packet(t, boxes_at(i))
            pkt['ego_pose'] = dict(x=0., y=ego_y)
            yield pkt

    def test_disabled_by_default_and_fusion_reduces_static_jitter(self):
        import random
        rng = random.Random(3)
        jitter = [(rng.gauss(0, .08), rng.gauss(0, .08)) for _ in range(30)]
        def boxes_at(i):
            return [box(3. + jitter[i][0], 10. + jitter[i][1], speed=0., heading=30., length=1.8, width=.8)]
        plain, fused = VeObjectTracker(), VeObjectTracker(static_geometry_fusion=True)
        raw_err, fused_err = [], []
        for pkt in self.moving_ego_packets(boxes_at):
            a = plain.update(dict(pkt))['boxes'][0]
            b = fused.update(dict(pkt))['boxes'][0]
            self.assertNotIn('raw_center_world', a)
            if b['motion_state'] == 'stationary' and 'raw_center_world' in b:
                self.assertEqual(a['center_world'], b['raw_center_world'])
                raw_err.append(math.dist(a['center_world'], (3., 10.)))
                fused_err.append(math.dist(b['center_world'], (3., 10.)))
                self.assertEqual(a['track_id'], b['track_id'])
        self.assertGreater(len(fused_err), 5)
        self.assertLess(sum(fused_err) / len(fused_err), .6 * sum(raw_err) / len(raw_err))

    def test_canonical_footprint_and_stopped_ego_adds_no_samples(self):
        tracker = VeObjectTracker(static_geometry_fusion=True)
        result = None
        for i in range(12):
            pkt = packet(i * .1, [box(0., 20., speed=0., heading=10. if i % 2 else 100.,
                                      length=.8 if i % 2 else 1.8, width=1.8 if i % 2 else .8)])
            pkt['ego_pose'] = dict(x=0., y=min(i, 5) * 1.)   # moves 5 m, then stops
            result = tracker.update(pkt)['boxes'][0]
        self.assertEqual('stationary', result['motion_state'])
        self.assertEqual(6, result['geometry_samples'])
        self.assertAlmostEqual(1.8, result['length_m'])
        self.assertAlmostEqual(.8, result['width_m'])
        self.assertAlmostEqual(100., result['heading_deg'] % 180., places=5)

    def test_near_range_freezes_samples(self):
        tracker = VeObjectTracker(static_geometry_fusion=True)
        for i in range(20):
            pkt = packet(i * .1, [box(0., 15. if i < 10 else 15.3, speed=0., length=1., width=1.)])
            pkt['ego_pose'] = dict(x=0., y=i * 1.)   # within 8 m from i = 8
            result = tracker.update(pkt)['boxes'][0]
        self.assertTrue(all(s[0][1] == 15. for s in tracker.tracks[result['track_id']]['static_geometry'].samples))

    def test_moving_track_keeps_raw_geometry(self):
        tracker = VeObjectTracker(static_geometry_fusion=True)
        for i in range(12):
            pkt = packet(i * .1, [box(i * .5, speed=5.)])
            pkt['ego_pose'] = dict(x=0., y=-20. + i)
            result = tracker.update(pkt)['boxes'][0]
        self.assertEqual('moving', result['motion_state'])
        self.assertNotIn('raw_center_world', result)
        self.assertEqual([5.5, 0.], result['center_world'])


class StationaryHoldTests(unittest.TestCase):
    def settle(self, tracker, seconds=4.):
        """A parked box seen long enough to be a confirmed stationary track."""
        t = 0.
        while t <= seconds:
            result = tracker.update(packet(t, [box(10., speed=0., length=1.4, width=.6)]))['boxes'][0]
            t += .1
        self.assertEqual('stationary', result['motion_state'])
        return t, result['track_id']

    def test_off_by_default_so_a_jittery_frame_is_uncertain(self):
        tracker = VeObjectTracker()
        t, _ = self.settle(tracker)
        states = {tracker.update(packet(t + i*.1, [box(10.3 + .5*(-1)**i, speed=0., length=1.4, width=.6)]))['boxes'][0]['motion_state']
                  for i in range(6)}
        self.assertIn('uncertain', states)

    def test_established_stationary_track_survives_box_jitter(self):
        tracker = VeObjectTracker(stationary_hold_s=3.)
        t, identity = self.settle(tracker)
        for i in range(12):
            result = tracker.update(packet(t + i*.1, [box(10. + .35*(-1)**i, speed=0., length=1.4, width=.6)]))['boxes'][0]
            self.assertEqual('stationary', result['motion_state'])
            self.assertEqual(identity, result['track_id'])
            self.assertEqual(0., result['planning_speed_mps'])

    def test_short_lived_stationary_evidence_is_not_held(self):
        tracker = VeObjectTracker(stationary_hold_s=3.)
        t = 0.
        while t <= 1.2:
            result = tracker.update(packet(t, [box(10., speed=0., length=1.4, width=.6)]))['boxes'][0]
            t += .1
        states = {tracker.update(packet(t + i*.1, [box(10. + .35*(-1)**i, speed=0., length=1.4, width=.6)]))['boxes'][0]['motion_state']
                  for i in range(6)}
        self.assertNotEqual({'stationary'}, states)

    def test_real_motion_ends_the_hold_and_is_reported_as_moving(self):
        tracker = VeObjectTracker(stationary_hold_s=3.)
        t, _ = self.settle(tracker)
        state = None
        for i in range(25):
            result = tracker.update(packet(t + i*.1, [box(10. + .3*i, speed=3., length=1.4, width=.6)]))['boxes'][0]
            state = result['motion_state']
        self.assertEqual('moving', state)
        self.assertGreater(result['planning_speed_mps'], 1.)

    def test_hold_never_extends_past_its_radius(self):
        tracker = VeObjectTracker(stationary_hold_s=3.)
        t, _ = self.settle(tracker)
        result = tracker.update(packet(t, [box(11.5, speed=0., length=1.4, width=.6)]))['boxes'][0]
        self.assertNotEqual('stationary_hold', result['planning_speed_source'])

    def test_recreated_track_at_the_same_place_keeps_identity_and_state(self):
        tracker = VeObjectTracker(stationary_hold_s=3.)
        t, identity = self.settle(tracker)
        tracker.update(packet(t, []))
        result = tracker.update(packet(t + 1.2, [box(10.1, speed=0., length=1.4, width=.6)]))['boxes'][0]
        self.assertEqual(identity, result['track_id'])
        self.assertEqual('stationary', result['motion_state'])

    def test_stationary_hold_is_bounded(self):
        with self.assertRaises(ValueError):
            VeObjectTracker(stationary_hold_s=7.)


class FragmentGateTests(unittest.TestCase):
    def ids(self, gate):
        tracker = VeObjectTracker(fragment_gate=gate)
        out = []
        sizes = [(4.5, 2.2)] * 4 + [(1.9, 1.6)] * 2 + [(4.5, 2.2)] * 3     # the car is reported as a piece, then whole again
        for i, (length, width) in enumerate(sizes):
            out.append(tracker.update(packet(i * .1, [box(30., 4., speed=0., length=length, width=width)]))['boxes'][0]['track_id'])
        return out

    def test_a_reported_piece_does_not_start_a_new_identity_when_the_gate_is_on(self):
        self.assertEqual(1, len(set(self.ids(True))))
        self.assertGreater(len(set(self.ids(False))), 1)      # the default behaviour is unchanged

    def test_two_separate_nearby_objects_are_not_merged(self):
        tracker = VeObjectTracker(fragment_gate=True)
        for i in range(4):
            rows = tracker.update(packet(i * .1, [box(30., 4., speed=0., length=4.5, width=2.2),
                                                  box(36., 4., speed=0., length=1.0, width=1.0)]))['boxes']
        self.assertEqual(2, len({b['track_id'] for b in rows}))


class HeadingGateTests(unittest.TestCase):
    def run_car(self, tracker, speed=17., dt=.25, frames=8, y=0., ve_speed=3.):
        # VE reads 3 m/s for a car at 17 m/s; frames 0.25 s apart move it 4.25 m, beyond the normal gate
        return [tracker.update(packet(i * dt, [box(20. + speed * dt * i, y, speed=ve_speed, length=4.6, width=2.)]))['boxes'][0]
                for i in range(frames)]

    def test_a_fast_car_keeps_its_id_and_is_confirmed_moving(self):
        out = self.run_car(VeObjectTracker(heading_gate=True))
        self.assertEqual(1, len({b['track_id'] for b in out}))
        self.assertEqual('moving', out[-1]['motion_state'])
        self.assertGreater(len({b['track_id'] for b in self.run_car(VeObjectTracker())}), 1)   # default unchanged

    def test_a_car_in_the_next_lane_is_not_taken_over(self):
        tracker = VeObjectTracker(heading_gate=True)
        tracker.update(packet(0., [box(20., 0., speed=3., length=4.6, width=2.)]))
        out = tracker.update(packet(.25, [box(24., 3.5, speed=3., length=4.6, width=2.)]))['boxes'][0]   # 3.5 m sideways
        self.assertNotEqual(1, out['track_id'])

    def test_a_pedestrian_box_is_not_extended(self):
        tracker = VeObjectTracker(heading_gate=True)
        tracker.update(packet(0., [box(20., 0., speed=1., length=.6, width=.6)]))
        out = tracker.update(packet(.25, [box(23., 0., speed=1., length=.6, width=.6)]))['boxes'][0]
        self.assertNotEqual(1, out['track_id'])


class HistoryGapTests(unittest.TestCase):
    def states(self, gap_s):
        tracker = VeObjectTracker(history_gap_s=gap_s)
        times = [0., .1, .2, .3, .4, .8, .9]                     # one missed frame at 4-7 Hz: a 0.4 s gap
        return [tracker.update(packet(t, [box(20. + 5. * t, speed=5.)]))['boxes'][0]['motion_state'] for t in times]

    def test_a_confirmed_mover_stays_moving_through_one_missed_frame(self):
        self.assertEqual('moving', self.states(.45)[-2])
        self.assertEqual('uncertain', self.states(.3)[-2])                  # default 0.3 s restarts the history


class FittedSpeedTests(unittest.TestCase):
    def speed(self, threshold, ve_speed, true_speed):
        tracker = VeObjectTracker(fitted_speed_above_mps=threshold)
        for i in range(8):
            out = tracker.update(packet(i * .1, [box(20. + true_speed * .1 * i, speed=ve_speed)]))['boxes'][0]
        return out['planning_speed_mps'], out['planning_speed_source']

    def test_a_fast_car_reports_the_fitted_speed_and_a_slow_one_keeps_ve(self):
        self.assertAlmostEqual(self.speed(3., 9., 10.)[0], 10., places=3)        # VE agrees within tolerance but reads low
        self.assertEqual(self.speed(3., 9., 10.)[1], 've_supported_by_position')
        self.assertEqual(self.speed(0., 9., 10.)[0], 9.)                         # default: VE
        self.assertEqual(self.speed(3., 2.5, 2.4)[0], 2.5)                       # pedestrian speed: VE kept


if __name__ == '__main__':
    unittest.main()
