"""Speed-matched admission and the recorded 16:18 one-tick brake regression."""
import math
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from AIM2026.Expert.src.config.parameters import highway as p
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.highway.planner import HighwayLanePlanner
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap
from AIM2026.Expert.src.control.path_controller import PathController


class EntryFollowResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.lanes = HighwayLaneMap(str(root/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lanes)
        self.source = self.lanes.links[p.ENTRY_MERGE_SOURCE]
        self.target = self.lanes.links[p.ENTRY_MERGE_TARGET]

    def step(self, ego, entries):
        match = self.lanes.project(ego.position,self.source)
        reference = [row[0] for row in self.lanes.route_points(match,150.,.5)]
        return self.planner.plan(ego,reference,entries,desired_velocity=16.6666667)

    def test_matches_leader_before_turn_and_admits_matched_trajectory(self):
        ego = VehicleState(62.14,249.58,math.radians(-89.),11.88)
        lead = ObjectInfo(66.,232.8,10.,-1,unique_id=8,size_x=4.55,size_y=1.98,heading_deg=-89.)
        result = self.step(ego,[{'object_info':lead}])
        self.assertEqual(result.reason,'entry_merge_approach')
        self.assertLessEqual(result.acc_target_velocity,10.)
        self.assertTrue(any(a.get('speed_policy')=='match_lead'
                            for a in result.debug_details['entry_merge']['attempts']))
        ego.velocity = result.acc_target_velocity
        result = self.step(ego,[{'object_info':lead}])
        self.assertEqual(result.reason,'entry_merge_start')
        self.assertAlmostEqual(result.acc_target_velocity,10.)
        self.assertEqual(result.debug_details['entry_merge']['speed_policy'],'match_lead')
        # The rejected close-gap acceleration must not produce a stop next tick.
        result = self.step(ego,[{'object_info':lead}])
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertNotEqual(result.reason,'entry_merge_front_hazard')

    def test_old_observation_cannot_start_but_fresh_frame_can(self):
        ego = VehicleState(62.14, 249.58, math.radians(-89.), 10.)
        lead = ObjectInfo(66., 232.8, 10., -1, unique_id=8, size_x=4.55,
                          size_y=1.98, heading_deg=-89.)
        entries = [{'object_info': lead}]
        ego.object_observation_age_s = .453  # latest recorded launch used a frame this old
        result = self.step(ego, entries)
        self.assertEqual(result.reason, 'entry_merge_approach')
        self.assertEqual(result.debug_details['entry_merge']['conflict']['kind'],
                         'await_fresh_merge_observation')
        self.assertIsNone(self.planner.active_candidate)
        ego.object_observation_age_s = .15
        result = self.step(ego, entries)
        self.assertEqual(result.reason, 'entry_merge_start')
        candidate = self.planner.active_candidate
        ego.object_observation_age_s = .45
        result = self.step(ego, entries)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertIs(self.planner.active_candidate, candidate)
        self.assertNotEqual(result.reason, 'entry_merge_approach')

    def recorded_maneuver(self):
        planner = self.planner
        start = VehicleState(62.74897801480256,168.70136533631012,-1.554447211514818,.0004572778)
        candidate = planner._build_candidate(start,self.target,
            self.lanes.project(start.position,self.target),'left',self.source,22.098498,
            prediction_horizon=p.MERGE_SAFETY_HORIZON)
        candidate.entry_merge = candidate.compact_static = True
        candidate.safety_speed = candidate.entry_speed_ceiling = 6.97584141373164
        candidate.speed_limit = candidate.maneuver_speed_limit = candidate.safety_speed
        planner.active_candidate = candidate
        planner.state = planner.CHANGE_LEFT
        planner.source_link_id,planner.target_link_id = self.source.id,self.target.id
        planner.entry_merge.following.command = candidate.safety_speed
        ego = VehicleState(64.43182894145139,158.4848008370027,-1.272772533832692,5.219495561387804)
        planner.set_control_dt(.0348059)
        data = [
            (1,66.58954382,128.61986234,11.07470214,4.91983515,2.03224659,-87.59178653,False),
            (24,69.97114977,148.57618718,10.79909706,4.90355933,2.06586075,-89.85853407,False),
            (30,74.39369987,149.89584214,13.83104513,4.73364514,2.22389865,-77.96600512,False),
            (29,66.23585002,172.20907731,10.82802105,4.78466642,1.97014940,-84.93590786,False),
            (31,73.42761306,178.89818346,11.97213412,4.40924519,1.98173928,-87.11313509,False),
            (33,79.76107744,142.97400488,4.83649397,3.78422296,2.17156339,-168.71129525,True),
        ]
        entries=[]
        for identity,x,y,v,length,width,yaw,uncertain in data:
            obj=ObjectInfo(x,y,v,-1,unique_id=identity,size_x=length,size_y=width,heading_deg=yaw)
            obj.motion_uncertain=uncertain
            entries.append({'object_info':obj})
        return ego,entries

    def test_recorded_distant_heading_outlier_does_not_issue_zero(self):
        ego,entries=self.recorded_maneuver()
        result=self.step(ego,entries)
        info=result.debug_details['entry_merge']
        self.assertEqual(result.reason,'entry_merge_confirm_crossing')
        self.assertAlmostEqual(result.acc_target_velocity,ego.velocity)
        self.assertEqual(info['conflict']['object_id'],29)
        self.assertEqual(info['front_conflict']['object_id'],33)
        self.assertEqual(info['crossing_confirmation']['object_id'],33)
        self.assertFalse(info['verified'])
        # The next frame corrects the crossing heading: resume the SAME path.
        path=self.planner.active_candidate
        entries[-1]['object_info'].heading_deg=-117.5506864
        entries[-1]['object_info'].position=Point(79.77,140.91)
        entries[-1]['object_info'].velocity=7.3864
        result=self.step(ego,entries)
        self.assertNotEqual(result.reason,'entry_merge_front_hazard')
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertIs(self.planner.active_candidate,path)

    def test_confirmation_cannot_hide_a_persistent_crossing(self):
        ego,entries=self.recorded_maneuver()
        for _ in range(8):
            result=self.step(ego,entries)
        self.assertEqual(result.reason,'entry_merge_front_hazard')
        self.assertEqual(result.acc_target_velocity,0.)

    def test_confirmed_crossing_and_close_body_are_not_delayed(self):
        ego,entries=self.recorded_maneuver()
        entries[-1]['object_info'].motion_uncertain=False
        result=self.step(ego,entries)
        self.assertEqual(result.reason,'entry_merge_front_hazard')
        self.planner.reset()
        ego,entries=self.recorded_maneuver()
        obj=entries[-1]['object_info']
        obj.position=Point(ego.position.x+1.,ego.position.y-3.)
        obj.velocity=0.
        result=self.step(ego,[{'object_info':obj}])
        self.assertEqual(result.acc_target_velocity,0.)

    def test_boost_remains_active_above_old_3mps_gate_but_not_for_stops(self):
        ego,entries=self.recorded_maneuver()
        result=self.step(ego,[])
        self.assertGreater(ego.velocity,3.)
        self.assertGreater(result.acc_target_velocity,ego.velocity)
        self.assertEqual(PathController._entry_merge_launch_acceleration_gain(
            result.acc_target_velocity,ego,result),p.ENTRY_MERGE_LAUNCH_P_GAIN)
        self.assertIsNone(PathController._entry_merge_launch_acceleration_gain(0.,ego,result))
        result.debug_details['entry_merge']['phase']='COMPLETE'
        self.assertIsNone(PathController._entry_merge_launch_acceleration_gain(8.,ego,result))

    def test_recorded_parallel_bus_does_not_apply_zero_gap_acc_during_crossing(self):
        # 2026-10-28 17:41, seed 208256029199771517, control cycle 943:
        # the newly visible bus is alongside, not yet on our swept path.
        ego = VehicleState(62.00748143659439, 250.66841629426926,
                           -1.540708823877215, 15.361817677815754)
        start = self.step(ego, [])
        self.assertEqual(start.reason, 'entry_merge_start')
        candidate = self.planner.active_candidate
        bus = ObjectInfo(65.83681277693157, 248.3953975559801,
                         9.928157806396484, -1, unique_id=14,
                         size_x=7.438310329767903, size_y=2.922016143798828,
                         heading_deg=-90.27611306672931)
        bus.motion_uncertain = True
        self.planner.set_control_dt(.0324227)
        result = self.step(ego, [{'object_info': bus}])
        self.assertIs(self.planner.active_candidate, candidate)
        self.assertEqual(result.reason, 'entry_merge_in_progress')
        self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
        self.assertIsNone(result.debug_details['entry_merge']['front_follow_limit'])
        self.assertAlmostEqual(result.acc_target_velocity-start.acc_target_velocity,
                               p.ENTRY_MERGE_PROGRESS_ACCEL*.0324227)

    def test_progress_increase_is_bounded_and_does_not_depend_on_follower_slew(self):
        ego, _ = self.recorded_maneuver()
        candidate = self.planner.active_candidate
        candidate.entry_speed_ceiling = candidate.safety_speed + 3.
        self.planner.entry_merge.following.command = 0.
        result = self.step(ego, [])
        self.assertAlmostEqual(result.acc_target_velocity,
            candidate.safety_speed+p.ENTRY_MERGE_PROGRESS_ACCEL*self.planner.control_dt)
        previous = result.acc_target_velocity
        result = self.step(ego, [])
        self.assertAlmostEqual(result.acc_target_velocity-previous,
                               p.ENTRY_MERGE_PROGRESS_ACCEL*self.planner.control_dt)

    def test_completed_merge_returns_to_real_front_following(self):
        ego = VehicleState(62.14, 249.58, math.radians(-89.), 10.)
        self.assertEqual(self.step(ego, []).reason, 'entry_merge_start')
        target_pose = self.lanes.project(Point(66.4, 220.), self.target)
        ego = VehicleState(target_pose.x, target_pose.y,
                           target_pose.yaw, 10.)
        lead = ObjectInfo(ego.position.x, ego.position.y-12., 6., -1,
                          unique_id=20, size_x=4.8, size_y=2., heading_deg=-90.)
        result = self.step(ego, [{'object_info': lead}])
        self.assertEqual(result.reason, 'entry_merge_complete')
        self.assertLess(result.acc_target_velocity, ego.velocity)
        self.assertIsNone(self.planner.active_candidate)

    def test_handoff_expands_spacing_gradually_and_retains_close_obstacle_braking(self):
        merge = self.planner.entry_merge
        ego = VehicleState(66.4, 200., -math.pi/2, 10.)
        merge.handoff_remaining = p.ENTRY_MERGE_HANDOFF_TIME
        initial = merge._handoff_front_limit(ego, 3., float('inf'), 10.)
        merge.handoff_remaining -= self.planner.control_dt
        next_value = merge._handoff_front_limit(ego, 3., float('inf'), 10.)
        self.assertLess(abs(next_value-initial), .1)
        merge.handoff_remaining = 0.
        self.assertAlmostEqual(merge._handoff_front_limit(ego, 3., float('inf'), 10.),
                               self.planner._acc_speed_limit(10., 3., float('inf'), 10.))
        # Interpolation never delays a real near stationary body.
        merge.handoff_remaining = p.ENTRY_MERGE_HANDOFF_TIME
        self.assertEqual(merge._handoff_front_limit(ego, .5, .05, 0.), 0.)

    def test_verified_moving_launch_releases_previous_gap_making_brake(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'Expert/src/config/config.json').read_text())
        controller = PathController(config)
        ego = VehicleState(62.14, 249.58, math.radians(-89.), 9.4)
        result = self.step(ego, [])
        controller.previous_target_velocity = 5.6
        controller.previous_acc_cmd = -.7
        command, _, _ = controller._calculate_control(ego, result.path,
                                                       result.acc_target_velocity, result)
        self.assertGreaterEqual(command, 0.)
        # A lower emergency target still overrides the launch release.
        command, _, _ = controller._calculate_control(ego, result.path, 0., result)
        self.assertEqual(command, -1.)

    def test_recorded_speed_dip_does_not_choose_braking_to_let_rear_overtake(self):
        start = VehicleState(62.0764058243,249.070189637,-1.533369526,9.404489729)
        c = self.planner._build_candidate(start, self.target,
            self.lanes.project(start.position,self.target), 'left', self.source,
            56.953134, prediction_horizon=p.MERGE_SAFETY_HORIZON)
        c.entry_merge = True
        c.compact_static = False
        c.safety_speed = 9.656252048
        c.entry_last_verified_speed = 9.975305393
        c.entry_speed_ceiling = 20.
        self.planner.active_candidate = c
        self.planner.state = self.planner.CHANGE_LEFT
        self.planner.source_link_id, self.planner.target_link_id = self.source.id,self.target.id
        ego = VehicleState(62.1075010166,248.10666163,-1.534628225,8.332477146)
        data = [(2,66.1142129453,260.932391511,11.0059108734,4.852992,2.072742,-89.203877),
                (14,66.205535665,231.130159941,8.4609012604,7.123889,2.962166,-87.927720)]
        entries = [{'object_info':ObjectInfo(x,y,v,-1,unique_id=i,size_x=ln,size_y=wd,heading_deg=yaw)}
                   for i,x,y,v,ln,wd,yaw in data]
        result = self.step(ego, entries)
        self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
        self.assertNotEqual(result.reason, 'entry_merge_front_hazard')

    def test_recorded_front_and_fast_rear_slot_does_not_default_to_zero(self):
        # Same 17:41 recording, cycle 1160: a prior brake has already allowed
        # a faster rear car to close. Full braking is NOT a safe escape.
        self.step(VehicleState(62.00748, 250.6684, -1.5402, 15.3618), [])
        ego = VehicleState(63.6032690, 219.06029598, -1.4964631, 9.5634863)
        data = [(15, 66.5092814111, 204.669633109, 10.1539783478, 7.04536235, 2.83070326, -89.71014),
                (16, 66.1957664075, 232.998577171, 13.2277231216, 5.05505, 2.11059, -91.71014)]
        entries = [{'object_info': ObjectInfo(x,y,v,-1,unique_id=i,size_x=ln,size_y=wd,heading_deg=yaw)}
                   for i,x,y,v,ln,wd,yaw in data]
        result = self.step(ego, entries)
        self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
        self.assertEqual(result.reason, 'entry_merge_rear_risk_progress')
        info = result.debug_details['entry_merge']
        self.assertTrue(info['rear_gap_unresolved']['rear'])
        self.assertFalse(info['verified'])
        matches = self.planner._match_objects(entries)
        self.assertIsNone(self.planner.entry_merge._front_conflict(
            self.planner.active_candidate, ego, entries, matches, result.acc_target_velocity))

    def test_matched_lead_does_not_override_a_fast_rear_vehicle(self):
        ego=VehicleState(62.14,249.58,math.radians(-89.),10.)
        lead=ObjectInfo(66.,232.8,10.,-1,unique_id=8,size_x=4.55,size_y=1.98,heading_deg=-89.)
        rear=ObjectInfo(66.,253.,15.,-1,unique_id=9,size_x=4.55,size_y=1.98,heading_deg=-89.)
        result=self.step(ego,[{'object_info':lead},{'object_info':rear}])
        self.assertNotEqual(result.reason,'entry_merge_start')
        self.assertIsNone(self.planner.active_candidate)

    def seed_recorded_lost_target(self, age=16.7):
        # 22:39 replay: ID 22 vanished near y=252; Ego later stopped at y=169.
        # Its last gap was only ~3 m, so the old forward-ROI-only release never ran.
        following = self.planner.entry_merge.following
        origin = self.lanes.project(Point(66.6, 252.), self.target)
        following.lead = dict(identity=22, s=origin.s, gap=3.2, speed=5.788,
                              length=6.35, x=66.6, y=252., yaw=-math.pi/2, age=0.)
        following.age = age
        following.command = 0.
        self.planner.set_control_dt(.1)
        return VehicleState(62.75, 168.89, math.radians(-89.), 0.)

    def test_recorded_lost_target_cannot_lock_empty_merge_from_standstill(self):
        ego = self.seed_recorded_lost_target()
        for _ in range(15):
            result = self.step(ego, [])
            self.assertNotEqual(result.debug_details['entry_merge']['target_event'], 'target_lost')
            if result.reason == 'entry_merge_start':
                break
        else:
            self.fail('Expired ID still prevented a clear stopped-state merge')
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertEqual(result.target_link_id, p.ENTRY_MERGE_TARGET)

    def test_expired_target_reselects_current_bus_and_does_not_force_merge(self):
        ego = self.seed_recorded_lost_target()
        bus = ObjectInfo(66.4, 169., 0., -1, unique_id=55, size_x=11.,
                         size_y=2.6, heading_deg=-89., is_static=True)
        result = self.step(ego, [{'object_info':bus}])
        details = result.debug_details['entry_merge']
        self.assertEqual(details['expired_lead_id'], 22)
        self.assertEqual(details['lead_id'], 55)
        self.assertEqual(details['target_event'], 'reselected_after_dropout')
        self.assertNotEqual(result.reason, 'entry_merge_start')
        self.assertIsNone(self.planner.active_candidate)

    def test_short_dropout_still_holds_and_same_id_reconnects(self):
        ego = self.seed_recorded_lost_target(age=0.)
        following = self.planner.entry_merge.following
        self.planner._requested_speed = 10.
        following.update(ego, self.target, [], 30.)
        self.assertEqual(following.details['target_event'], 'predicted_short_dropout')
        self.assertEqual(following.details['lead_id'], 22)
        obj = ObjectInfo(66.6, 251., 5.8, -1, unique_id=22, size_x=6.35,
                         size_y=2.6, heading_deg=-90.)
        entries = [{'object_info':obj}]
        following.update(ego, self.target, self.planner._match_objects(entries), 30.)
        self.assertEqual(following.details['lead_id'], 22)
        self.assertEqual(following.details['lead_age'], 0.)
        self.assertIsNone(following.details['expired_lead_id'])

    def test_expired_target_still_rejects_a_fast_approaching_rear_car(self):
        ego = self.seed_recorded_lost_target()
        rear = ObjectInfo(66.4, 181., 25., -1, unique_id=56, size_x=4.8,
                          size_y=2., heading_deg=-90.)
        for _ in range(8):
            result = self.step(ego, [{'object_info':rear}])
            self.assertNotEqual(result.reason, 'entry_merge_start')
        self.assertIsNone(self.planner.active_candidate)

    def test_target_expiry_keeps_an_already_committed_path(self):
        ego, _ = self.recorded_maneuver()
        candidate = self.planner.active_candidate
        following = self.planner.entry_merge.following
        following.lead = dict(identity=22, s=1., gap=3.2, speed=5.8,
                              length=6.35, x=66.6, y=252., yaw=-math.pi/2, age=0.)
        following.age = 20.
        result = self.step(ego, [])
        self.assertIs(self.planner.active_candidate, candidate)
        self.assertEqual(result.target_link_id, p.ENTRY_MERGE_TARGET)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertIsNone(result.debug_details['entry_merge']['lead_id'])

    def rear_follow_fixture(self):
        ego, _ = self.recorded_maneuver()
        rear = ObjectInfo(66.4, ego.position.y+10., 3., -1, unique_id=2,
                          size_x=4.8, size_y=2., heading_deg=-89.)
        entries = [{'object_info': rear}]
        following = self.planner.entry_merge.following
        self.planner._requested_speed = 10.
        following.update(ego, self.target, self.planner._match_objects(entries), 100.)
        self.assertEqual(following.lead['identity'], 2)
        following.command = 2.  # Already lowered by the erroneous rear follower.
        return ego, entries

    def test_existing_rear_target_cannot_decelerate_committed_merge(self):
        ego, entries = self.rear_follow_fixture()
        candidate = self.planner.active_candidate
        result = self.step(ego, entries)
        self.assertEqual(result.debug_details['entry_merge']['released_rear_id'], 2)
        for _ in range(20):
            self.assertIs(self.planner.active_candidate, candidate)
            self.assertIsNone(result.debug_details['entry_merge']['lead_id'])
            self.assertGreaterEqual(result.acc_target_velocity, ego.velocity)
            result = self.step(ego, entries)

    def test_rear_release_also_applies_to_completed_merge_handoff(self):
        ego, entries = self.rear_follow_fixture()
        following = self.planner.entry_merge.following
        for _ in range(20):
            # EntryMerge's completion handoff calls this same committed path.
            speed = following.update(ego, self.target,
                self.planner._match_objects(entries), 100., committed=True, speed_ceiling=10.)
            self.assertIsNone(following.lead)
            self.assertGreaterEqual(speed, ego.velocity)

    def test_rear_release_keeps_actual_front_emergency_braking(self):
        ego, entries = self.rear_follow_fixture()
        front = ObjectInfo(ego.position.x, ego.position.y-3., 0., -1,
            unique_id=9, size_x=4.8, size_y=2., heading_deg=-89., is_static=True)
        result = self.step(ego, entries+[{'object_info':front}])
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_precommit_rear_yield_target_is_unchanged(self):
        ego, entries = self.rear_follow_fixture()
        following = self.planner.entry_merge.following
        following.update(ego, self.target, self.planner._match_objects(entries), 100.)
        self.assertEqual(following.lead['identity'], 2)
        self.assertIsNone(following.details['released_rear_id'])


if __name__=='__main__':
    unittest.main()
