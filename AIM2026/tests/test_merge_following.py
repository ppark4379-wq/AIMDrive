"""Stable adjacent leader, bounded dropout, continuous following, and scope."""
import unittest

from AIM2026.Expert.src.config.parameters import highway as p


class MergeFollowingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from AIM2026.tests.test_entry_merge import EntryMergeTests
        EntryMergeTests.setUpClass()
        cls.fixture_type = EntryMergeTests

    def setUp(self):
        self.case = self.fixture_type()
        self.case.setUp()
        self.planner = self.case.planner
        self.follow = self.planner.entry_merge.following
        self.planner._requested_speed = 16.
        self.ego = self.case.ego(speed=10.)

    def update(self, cars, committed=False):
        return self.follow.update(self.ego, self.case.target,
            self.planner._match_objects(cars), 80., committed=committed)

    def test_parallel_car_is_selected_and_gap_is_opened_in_source_lane(self):
        result = self.case.step(self.ego, [self.case.car(250., 10., identity=5)])
        self.assertEqual(result.target_link_id, p.ENTRY_MERGE_SOURCE)
        self.assertEqual(result.debug_details['entry_merge']['lead_id'], 5)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertLess(result.acc_target_velocity, 10.)

    def test_nearer_new_car_does_not_replace_selected_leader(self):
        self.update([self.case.car(249., 10., identity=5)])
        for _ in range(10):
            self.update([self.case.car(235., 10., identity=5), self.case.car(249., 10., identity=6)])
            self.assertEqual(self.follow.details['lead_id'], 5)
        # Both cars still reach the planner's collision checks.
        result = self.case.step(self.ego, [self.case.car(235., 10., identity=5), self.case.car(249., 10., identity=6)])
        self.assertIsNone(self.planner.active_candidate)

    def test_new_id_reassociates_only_an_unambiguous_physical_match(self):
        self.update([self.case.car(235., 10., identity=5)])
        self.update([self.case.car(234.67, 10., identity=50)])
        self.assertEqual(self.follow.details['lead_id'], 50)
        self.assertEqual(self.follow.details['target_event'], 'reassociated')
        self.follow.reset()
        self.update([self.case.car(235., 10., identity=5)])
        self.update([self.case.car(234.67, 10., identity=50), self.case.car(234.8, 10., identity=51)])
        self.assertEqual(self.follow.details['lead_id'], 5)

    def test_dropout_is_bounded_and_not_an_empty_lane(self):
        start = self.update([self.case.car(235., 10., identity=5)])
        for _ in range(5):
            speed = self.update([], committed=True)
            self.assertGreater(speed, 0.)
            self.assertEqual(self.follow.details['lead_id'], 5)
            self.assertEqual(self.follow.details['target_event'], 'predicted_short_dropout')
        for _ in range(20):
            self.update([], committed=True)
        self.assertEqual(self.follow.details['target_event'], 'target_lost')
        self.assertLessEqual(self.follow.command, 10.)

    def test_following_responds_continuously_to_leader_after_commit(self):
        self.ego.velocity = 6.
        previous = self.update([self.case.car(238., 8., identity=5)])
        increased = False
        for _ in range(12):
            speed = self.update([self.case.car(225., 11., identity=5)], committed=True)
            self.assertLessEqual(abs(speed-previous), p.ENTRY_MERGE_FOLLOW_ACCEL*self.planner.control_dt+1e-9)
            increased |= speed > previous
            previous = speed
        self.assertTrue(increased)

    def test_missing_leader_cleared_ahead_releases_slot_instead_of_deadlock(self):
        self.update([self.case.car(220.5, 10., identity=5)])
        for _ in range(30):
            self.update([])
        self.assertIsNone(self.follow.details['lead_id'])
        self.assertEqual(self.follow.details['target_event'], 'empty')

    def test_search_range_is_on_target_lane_and_not_ego_heading(self):
        self.ego = self.case.ego(y=230., speed=10.)
        self.update([])
        polygon = self.follow.details['search_polygon']
        self.assertEqual(polygon[0], polygon[-1])
        self.assertGreater(min(x for x,y in polygon), self.ego.position.x)
        self.assertAlmostEqual(max(y for x,y in polygon)-min(y for x,y in polygon),
                               p.ENTRY_MERGE_SEARCH_FRONT+p.ENTRY_MERGE_SEARCH_REAR, delta=.2)

    def test_no_following_logic_on_other_lane_or_after_handoff(self):
        from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
        from AIM2026.Expert.src.interfaces.geometry import Point
        lane = self.case.lanes.links['A2256W000410']
        match = self.case.lanes.project(Point(70., 200.), lane)
        ego = VehicleState(match.x, match.y, match.yaw, 10.)
        self.assertIsNone(self.planner.entry_merge.plan(ego, match, [], [], True))
        self.planner.entry_merge.completed = True
        self.planner.entry_merge.handoff_remaining = 0.
        match = self.case.lanes.project(Point(66., 200.), self.case.target)
        self.assertIsNone(self.planner.entry_merge.plan(ego, match, [], [], True))

    def test_reduced_front_reserve_accepts_earlier_gap_but_keeps_rear_guard(self):
        cars = [self.case.car(235.7, 10., identity=5)]
        result = self.case.step(self.ego, cars)
        self.assertEqual(result.reason, 'entry_merge_start')
        self.assertLess(result.debug_details['entry_merge']['lead_gap'], 15.)
        self.planner.reset()
        result = self.case.step(self.ego, cars+[self.case.car(254., 13., identity=6)])
        self.assertIsNone(self.planner.active_candidate)

    def test_distant_provisional_moving_box_does_not_create_stopping_envelope(self):
        from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
        import math
        self.case.step(self.ego)
        candidate = self.planner.active_candidate
        ego = self.case.pose_on_candidate(candidate, 10, 8.)
        point = candidate.points[70]
        obj = ObjectInfo(point.x, point.y, 9., -1, unique_id=89,
                         size_x=4.5, size_y=2., heading_deg=math.degrees(ego.yaw))
        obj.motion_uncertain = True
        result = self.case.step(ego, [{'object_info':obj}])
        self.assertEqual(result.reason, 'entry_merge_in_progress')
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertNotIn('stopping_speed', result.debug_details['entry_merge'])
        self.assertEqual(obj.velocity, 9.)


if __name__ == '__main__':
    unittest.main()
