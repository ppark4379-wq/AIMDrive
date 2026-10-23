import json
import math
from pathlib import Path
import time
import unittest

import numpy as np

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState
from expert.src.planning.lattice.lattice_planner import LatticePlanner
from expert.src.planning.static_obstacle.recorded_planner import RecordedStaticPlanner, connector, GLOBAL_REJOIN_POINT


class RecordedStaticTests(unittest.TestCase):
    def setUp(self):
        self.planner = RecordedStaticPlanner(LatticePlanner())
        self.ego = VehicleState(-60.1, -162., math.pi/2, 0.)
        self.route = [Point(-59.85, -165.+.2*i) for i in range(350)]

    def start(self, obstacles=()):
        result = self.planner.plan(self.ego, self.route[:80], obstacles, self.route)
        self.assertTrue(result.active, result.reason)
        self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
        return result

    def test_six_selected_files_and_reverse_trim(self):
        audit = {r['name']: r for r in self.planner.audit}
        self.assertEqual(len(audit), 6)
        self.assertNotIn('right_4.csv', audit)
        self.assertNotIn('right_06.csv', audit)
        self.assertEqual(audit['right_04.csv']['trimmed_prefix_points'], 5)
        self.assertEqual(audit['right_05.csv']['trimmed_prefix_points'], 0)
        self.assertEqual(audit['right_06_retry.csv']['trimmed_prefix_points'], 1)

    def stopped_scene(self):
        scene = json.loads((Path(__file__).parent/'static_scene_20260915.json').read_text())
        self.route = [Point(*row) for row in scene['route']]
        objects = [ObjectInfo(**row) for row in scene['objects']]
        self.start(objects)
        return VehicleState(*scene['stopped_ego']), objects

    def test_live_stop_pose_is_clear_without_expanding_recorded_corridor(self):
        ego, objects = self.stopped_scene()
        result = self.planner.plan(ego, self.route, objects)
        self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
        self.assertLessEqual(result.acc_target_velocity, 2.)
        outside = VehicleState(ego.position.x+3., ego.position.y, ego.yaw, 0.)
        self.assertFalse(self.planner._envelope_clear(np.array([outside.position]), np.array([outside.yaw])))
        obj = ObjectInfo(ego.position.x, ego.position.y+2., 0., 2,
                         is_static=True, size_x=1., size_y=1.)
        result = self.planner.plan(ego, self.route, objects+[obj])
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_cutoff_does_not_override_collision_check_during_handoff(self):
        self.start()
        ego = VehicleState(-59.85, GLOBAL_REJOIN_POINT[1]+.1, math.pi/2, 2.)
        obj = ObjectInfo(-59.85, ego.position.y+2., 0., 2,
                         is_static=True, size_x=1., size_y=1.)
        result = self.planner.plan(ego, self.route, [obj])
        self.assertTrue(result.active)
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_live_stop_recovery_with_steering_lag_and_global_handoff(self):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.control.pure_pursuit import PurePursuit
        import contextlib
        import io
        ego, objects = self.stopped_scene()
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.control_dt = dt = 1./30.
        controller.previous_target_velocity = 0.
        controller.pure_pursuit = PurePursuit(.65, 3., 2., 30.)
        steering = math.radians(25.86)
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(1200):
                reference = [v for v in self.route if v.y >= ego.position.y-.25]
                result = self.planner.plan(ego, reference, objects)
                self.assertGreater(result.acc_target_velocity, 0., (step, ego.position, result.debug_details))
                target = controller._smooth_static_target_speed(min(16.67, result.acc_target_velocity), ego, result)
                controller.previous_target_velocity = target
                pp = controller.pure_pursuit
                pp.path, pp.vehicle_state = result.path, ego
                command = float(np.clip(pp.calculate_steering_angle(controller._recorded_lookahead_limits(ego, result)),
                                        -math.radians(40.), math.radians(40.)))
                steering += float(np.clip((command-steering)*dt/.2, -math.radians(60.)*dt, math.radians(60.)*dt))
                speed = max(0., ego.velocity+float(np.clip(target-ego.velocity, -3.*dt, 1.2*dt)))
                ego = VehicleState(ego.position.x+speed*dt*math.cos(ego.yaw),
                                   ego.position.y+speed*dt*math.sin(ego.yaw),
                                   ego.yaw+speed*dt*math.tan(steering)/3., speed)
                if self.planner.done and ego.position.y > self.planner.handoff_y+5.:
                    break
            else:
                self.fail('Failed to recover from the captured stop pose')
        self.assertTrue(self.planner.done)

    def test_complete_return_and_hold_without_global_path(self):
        self.start()
        original = np.array(self.planner.path)
        self.assertGreater(original[-1, 1], -110.)
        self.assertAlmostEqual(original[-1, 0], -59.85, places=2)
        point = self.planner.path[40]
        ego = VehicleState(point.x, point.y, LatticePlanner._path_heading(self.planner.path,40), 1.)
        result = self.planner.plan(ego, [], [])
        self.assertGreater(result.acc_target_velocity, 0.)
        np.testing.assert_array_equal(original, self.planner.path)
        self.assertEqual(self.planner.plan_count, 1)

    def test_return_geometry_joins_global_at_configured_coordinate(self):
        self.planner._setup(self.route, self.ego)
        self.assertEqual(self.planner.rejoin_y, -116.70)
        for curve in self.planner.curves:
            for y in (-116.70, -116., -110.):
                self.assertAlmostEqual(float(curve(y)), self.planner.global_x(y), places=8)

    def test_handoff_at_coordinate_without_static_speed_limit(self):
        self.start()
        x, y, _ = GLOBAL_REJOIN_POINT
        ego = VehicleState(x, y, math.pi/2, 2.)
        self.assertFalse(self.planner.contains(ego))
        result = self.planner.plan(ego, self.route, [])
        self.assertIs(result.path, self.route)
        self.assertFalse(result.active)
        self.assertFalse(result.acc_override)
        self.assertEqual(result.acc_target_velocity, float('inf'))
        self.assertTrue(self.planner.done)
        self.assertFalse(self.planner.active)

    def test_before_handoff_keeps_recorded_path(self):
        self.start()
        ego = VehicleState(-59.85, self.planner.handoff_y-.01, math.pi/2, 1.)
        result = self.planner.plan(ego, self.route, [])
        self.assertTrue(result.active)
        self.assertFalse(self.planner.done)

    def test_early_return_is_smooth_and_does_not_wait_for_old_cutoff(self):
        self.start()
        self.assertLess(self.planner.handoff_y, GLOBAL_REJOIN_POINT[1]-1.)
        xy = np.array(self.planner.path)
        returned = xy[xy[:,1] >= self.planner.handoff_y]
        self.assertTrue(np.all(np.abs(returned[:,0]+59.85) < 1e-6))
        start = self.planner.return_start_y
        tail = [point for point in self.planner.path if point.y >= start]
        self.assertLess(LatticePlanner._max_curvature(tail), .125)
        ego = VehicleState(-59.85, self.planner.handoff_y+.1, math.pi/2, 2.)
        result = self.planner.plan(ego, self.route, [])
        self.assertEqual(result.reason, 'static_recorded_rejoined')
        self.assertFalse(result.active)
        self.assertTrue(self.planner.past_rejoin(ego))

    def test_early_return_does_not_ignore_new_obstacle(self):
        self.start()
        ego = VehicleState(-59.85, self.planner.handoff_y+.1, math.pi/2, 2.)
        obstacle = ObjectInfo(-59.85, ego.position.y+7., 0., 2, is_static=True, size_x=1., size_y=1.)
        result = self.planner.plan(ego, self.route, [obstacle])
        self.assertTrue(result.active)
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_early_return_waits_for_actual_alignment(self):
        self.start()
        ego = VehicleState(-59.85, self.planner.handoff_y+.1, math.pi/2+.3, 2.)
        self.assertFalse(self.planner._early_handoff_ready(ego, self.route, []))
        self.assertFalse(self.planner.done)

    def test_missing_local_path_at_handoff_keeps_cached_global_tail(self):
        self.start()
        ego = VehicleState(-59.85, GLOBAL_REJOIN_POINT[1]+.2, math.pi/2, 1.)
        result = self.planner.plan(ego, [], [])
        self.assertTrue(result.active)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertTrue(all(abs(point.x+59.85) < .01 for point in result.path))

    def test_handoff_discards_pending_plan(self):
        from concurrent.futures import Future
        future = Future()
        self.planner.pending = future
        ego = VehicleState(-59.97, -116.70, math.pi/2, 1.)
        result = self.planner.plan(ego, self.route, [])
        self.assertIs(result.path, self.route)
        self.assertIsNone(self.planner.pending)
        self.assertTrue(future.cancelled())

    def test_global_handoff_does_not_reactivate_legacy_static_planner(self):
        from unittest.mock import patch
        from expert.src.planning.section_planning import LatticePlanningProcess
        process = LatticePlanningProcess(highway_zone_points=())
        ego = VehicleState(-59.97, -116.70, math.pi/2, 1.)
        obj = ObjectInfo(-59.97, -120., 0., 2, is_static=True, size_x=1., size_y=1.)
        try:
            with patch.object(process.static_avoidance, 'plan') as legacy:
                for y in (-116.70, -115., -110.):
                    ego.position = Point(-59.97, y)
                    result = process.run(ego, self.route, [{'object_info': obj}], prediction_path=self.route)
                    self.assertIs(result.path, self.route)
                    self.assertFalse(result.acc_override)
                legacy.assert_not_called()
        finally:
            process.recorded_static.executor.shutdown(wait=True)

    def test_new_obstacle_stops_without_switching_path(self):
        self.start()
        original = np.array(self.planner.path)
        point = self.planner.path[80]
        obj = ObjectInfo(point.x, point.y, 0., 2, is_static=True, size_x=2., size_y=2.)
        result = self.planner.plan(self.ego, [], [obj])
        self.assertEqual(result.acc_target_velocity, 0.)
        np.testing.assert_array_equal(original, self.planner.path)

    def _block_committed_path(self, velocity):
        self.start()
        original = np.array(self.planner.path)
        point = self.planner.path[160]      # 16 m ahead: other recorded routes can pass it
        obj = ObjectInfo(point.x, point.y, 0., 2, is_static=True, size_x=1., size_y=1.)
        ego = VehicleState(self.ego.position.x, self.ego.position.y, self.ego.yaw, velocity)
        return original, obj, ego

    def test_stopped_and_still_blocked_replans_from_measured_pose(self):
        from unittest.mock import patch
        import expert.src.planning.static_obstacle.recorded_planner as module
        original, obj, ego = self._block_committed_path(0.)
        clock = [1000.]
        with patch.object(module.time, 'monotonic', lambda: clock[0]):
            result = self.planner.plan(ego, self.route[:80], [obj], self.route)
            self.assertEqual(result.acc_target_velocity, 0.)                 # immediate stop, same path
            np.testing.assert_array_equal(original, self.planner.path)
            clock[0] += module.REPLAN_AFTER_BLOCKED_S + .1
            result = self.planner.plan(ego, self.route[:80], [obj], self.route)
            self.assertEqual(result.debug_details['static_avoidance']['error'], 'replanning_after_block')
            self.assertFalse(self.planner.active)
            clock[0] += 1.1
            result = self.planner.plan(ego, self.route[:80], [obj], self.route)
        self.assertTrue(self.planner.active, result.reason)
        xy = np.array(self.planner.path)
        yaw = np.array([self.planner.planner._path_heading(self.planner.path, i) for i in range(len(xy))])
        self.assertTrue(self.planner.collision_free(xy, yaw, [obj]))
        self.assertFalse(np.array_equal(original, xy))

    def test_moving_vehicle_does_not_replan_a_blocked_path(self):
        from unittest.mock import patch
        import expert.src.planning.static_obstacle.recorded_planner as module
        original, obj, ego = self._block_committed_path(2.)
        clock = [1000.]
        with patch.object(module.time, 'monotonic', lambda: clock[0]):
            for _ in range(4):
                result = self.planner.plan(ego, self.route[:80], [obj], self.route)
                clock[0] += 1.
        self.assertEqual(result.acc_target_velocity, 0.)
        np.testing.assert_array_equal(original, self.planner.path)

    def test_runtime_envelope_check_tolerates_small_tracking_error(self):
        import expert.src.planning.static_obstacle.recorded_planner as module
        self.start()
        point = self.planner.path[20]
        base = np.array([[point.x, point.y]]); yaw = np.array([self.ego.yaw])
        tol = module.RUNTIME_ENVELOPE_TOLERANCE
        clear = lambda d, margin=0.: self.planner._envelope_clear(base + [[d, 0.]], yaw, body_margin=margin)
        # Footprint-limited side: the first zero-margin failure is still inside the tolerance.
        shift = next(d for step in (-.005, .005) for d in np.arange(step, 1.5*np.sign(step), step)
                     if not clear(d) and clear(d, -tol))
        self.assertFalse(clear(shift))
        ego = VehicleState(point.x + shift, point.y, self.ego.yaw, 0.)
        result = self.planner.plan(ego, self.route[:80], [], self.route)
        self.assertNotEqual(result.debug_details['static_avoidance']['error'], 'recorded_envelope')
        self.assertFalse(clear(shift + np.sign(shift)*(tol + .3), -tol))     # a real departure is still blocked

    def test_adopted_plan_is_logged_with_route_and_points(self):
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.start()
        line = [l for l in out.getvalue().splitlines() if l.startswith('[STATIC_PATH]')]
        self.assertEqual(1, len(line))
        self.assertIn('routes=', line[0]); self.assertIn('points=[(', line[0])

    def test_editor_preview_replans_after_obstacle_or_ego_move(self):
        self.planner.set_live_preview(True)
        self.assertFalse(self.planner.refresh_live_preview(self.ego, []))
        result = self.start()
        self.assertEqual(result.reason, 'static_recorded_preview')
        self.assertTrue(self.planner.active)

        obstacle = ObjectInfo(-59.85, -145., 0., 2, is_static=True,
                              size_x=1., size_y=1.)
        self.assertTrue(self.planner.refresh_live_preview(self.ego, [obstacle]))
        self.assertFalse(self.planner.active)

        # Establish the new scene, then reject tiny pose noise but reset for
        # an actual operator teleport.
        self.assertFalse(self.planner.refresh_live_preview(self.ego, [obstacle]))
        self.planner.path = [Point(-59.85, -162.), Point(-59.85, -161.)]
        jitter = VehicleState(self.ego.position.x+.05, self.ego.position.y,
                              self.ego.yaw, 0.)
        self.assertFalse(self.planner.refresh_live_preview(jitter, [obstacle]))
        moved = VehicleState(self.ego.position.x+1., self.ego.position.y,
                             self.ego.yaw, 0.)
        self.assertTrue(self.planner.refresh_live_preview(moved, [obstacle]))
        self.assertFalse(self.planner.active)

    def test_editor_drive_commits_preview_and_uses_production_tracking(self):
        self.planner.set_live_preview(True)
        self.assertFalse(self.planner.refresh_live_preview(self.ego, []))
        preview = self.start()
        self.assertEqual(preview.reason, 'static_recorded_preview')

        original = np.array(self.planner.path)
        self.planner.set_live_drive(True)
        moved = VehicleState(self.ego.position.x, self.ego.position.y+.5,
                             self.ego.yaw, 0.)
        self.assertFalse(self.planner.refresh_live_preview(moved, []))
        np.testing.assert_array_equal(original, self.planner.path)

        tracked = self.planner.plan(self.ego, self.route[:80], [], self.route)
        self.assertEqual(tracked.reason, 'static_recorded_follow')
        self.assertTrue(tracked.debug_details['static_avoidance']['live_drive_committed'])

        # Browser stop is a command pause. It must not reactivate preview
        # invalidation while Ego is already away from the editor start pose.
        self.planner.set_live_drive(False)
        self.assertFalse(self.planner.interactive_preview)

    def test_editor_drive_uses_global_route_until_avoidance_entry(self):
        self.planner.set_live_preview(True)
        self.assertFalse(self.planner.refresh_live_preview(self.ego, []))
        self.start()
        avoidance = list(self.planner.path)
        self.planner.set_live_drive(True)

        far_ego = VehicleState(-59.85, -175., math.pi/2, 0.)
        approach_route = [Point(-59.85, -180.+.2*i) for i in range(425)]
        result = self.planner.plan(
            far_ego, approach_route[:100], [], approach_route
        )
        self.assertEqual(result.reason, 'static_recorded_drive_approach')
        self.assertEqual(result.path, approach_route[:100])
        self.assertEqual(result.candidates[0].points, avoidance)
        self.assertGreater(result.acc_target_velocity, 0.)

    def test_production_mode_ignores_editor_refresh_hook(self):
        self.start()
        original = np.array(self.planner.path)
        obstacle = ObjectInfo(-59.85, -145., 0., 2, is_static=True,
                              size_x=1., size_y=1.)
        self.assertFalse(self.planner.refresh_live_preview(self.ego, [obstacle]))
        np.testing.assert_array_equal(original, self.planner.path)

    def test_no_global_return_fails_closed(self):
        result = self.planner.plan(self.ego, [], [], [])
        self.assertFalse(result.active)
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_vehicle_width_and_gt_heading(self):
        obj = ObjectInfo(-58.9, -159., 0., 2, is_static=True, size_x=.3, size_y=.3)
        self.assertFalse(self.planner.collision_free(np.array([self.ego.position]), np.array([self.ego.yaw]), [obj]))
        obj.position = Point(-50., -159.)
        self.assertTrue(self.planner.collision_free(np.array([self.ego.position]), np.array([self.ego.yaw]), [obj]))
        obj.position = Point(-58., -159.)
        obj.size_x, obj.size_y, obj.heading_deg = 6., .2, 0.
        self.assertFalse(self.planner.collision_free(np.array([self.ego.position]), np.array([self.ego.yaw]), [obj]))

    def test_connector_matches_endpoint_derivatives(self):
        a, b = (0., 0., .1, .02), (10., 2., -.1, -.02)
        curve = connector(a,b)
        for row in (a,b):
            for d in range(3):
                self.assertAlmostEqual(float(curve(row[0],d)), row[d+1], places=9)

    def test_saved_scene_needs_safe_combination(self):
        scene = json.loads((Path(__file__).parent/'static_scene_20260914.json').read_text())
        result = self.start([ObjectInfo(**row) for row in scene['objects']])
        self.assertTrue(result.debug_details['static_avoidance']['preplanned_complete'])
        self.assertLessEqual(LatticePlanner._max_curvature(result.path), .24)
        self.assertGreaterEqual(len(self.planner.selected_names), 1)

    def test_asynchronous_search_does_not_hold_control_loop(self):
        planner = RecordedStaticPlanner(LatticePlanner(), background=True)
        try:
            begin = time.monotonic()
            result = planner.plan(self.ego, self.route, [], self.route)
            self.assertGreater(result.acc_target_velocity, 0.)
            self.assertLess(time.monotonic()-begin, .2)
            planner.pending.result(timeout=10.)
            result = planner.plan(self.ego, self.route, [], self.route)
            self.assertGreater(result.acc_target_velocity, 0.)
        finally:
            planner.executor.shutdown(wait=True)

    def test_gt_preplanning_keeps_approach_and_moving_entry_continuous(self):
        from expert.src.planning.section_planning import LatticePlanningProcess
        process = LatticePlanningProcess(highway_zone_points=())
        route = [Point(-59.85, -220.+.2*i) for i in range(625)]
        obstacle = ObjectInfo(-59.85, -145., 0., 2, is_static=True, size_x=1., size_y=1.)
        entries = [{'object_info': obstacle}]
        approach = VehicleState(-59.85, -205., math.pi/2, 8.)
        planner = process.recorded_static
        try:
            result = process.run(approach, route[:100], entries, prediction_path=route[:100], global_path=route)
            self.assertIsNotNone(planner.pending)
            self.assertFalse(result.active)
            self.assertGreaterEqual(result.acc_target_velocity, approach.velocity)
            future = planner.pending
            future.result(timeout=12.)
            result = process.run(approach, route[:100], entries, global_path=route)
            self.assertIs(planner.pending, future)
            self.assertEqual(result.reason, 'static_recorded_approach')
            self.assertGreater(result.acc_target_velocity, approach.velocity)
            entry = VehicleState(-59.85, -163.9, math.pi/2, 4.)
            result = process.run(entry, [v for v in route if v.y >= -164.], entries, global_path=route)
            self.assertTrue(result.active, result.debug_details)
            self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
            self.assertEqual(planner.plan_count, 1)
        finally:
            planner.executor.shutdown(wait=True)

    def test_pending_search_keeps_safe_motion_but_stops_for_close_obstacle(self):
        from concurrent.futures import Future
        self.planner.pending = Future()
        ego = VehicleState(-59.85, -162., math.pi/2, 5.)
        result = self.planner.plan(ego, self.route, [])
        self.assertGreater(result.acc_target_velocity, 5.)
        self.assertFalse(result.debug_details['static_avoidance']['path_validated'])
        obstacle = ObjectInfo(-59.85, -158., 0., 2, is_static=True, size_x=1., size_y=1.)
        self.assertEqual(self.planner.plan(ego, self.route, [obstacle]).acc_target_velocity, 0.)

    def test_worker_finishing_after_reset_cannot_mutate_live_state(self):
        from concurrent.futures import Future
        from unittest.mock import Mock
        self.planner.executor = Mock()
        future = Future()
        future.set_running_or_notify_cancel()
        self.planner.executor.submit.return_value = future
        self.planner._submit(self.ego, self.route, [])
        function, *args = self.planner.executor.submit.call_args.args
        self.planner.reset()
        worker = function(*args)
        self.assertTrue(worker.path)
        self.assertEqual(self.planner.path, [])
        self.assertIsNone(self.planner.goal_y)
        self.assertEqual(self.planner.plan_count, 0)

    def test_failed_worker_keeps_search_evidence_without_adopting_its_state(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        worker = SimpleNamespace(path=[], search_details={
            'method': 'corridor_astar', 'status': 'timeout', 'expanded': 7},
            rejections={'curvature': 2, 'collision': 3, 'envelope': 4},
            search_progress=6, plan_count=1, route=[Point(999., 999.)])
        future = Future()
        future.set_result(worker)
        self.planner.pending = future
        result = self.planner.plan(self.ego, self.route, [])
        self.assertEqual(result.reason, 'static_recorded_no_complete_route')
        self.assertFalse(result.active)
        self.assertEqual(result.acc_target_velocity, 0.)
        details = result.debug_details['static_avoidance']
        self.assertEqual(details['corridor_search']['status'], 'timeout')
        self.assertEqual(details['search_rejections']['collision'], 3)
        self.assertEqual(details['search_progress'], 6)
        self.assertEqual(details['plan_count'], 1)
        self.assertEqual(self.planner.path, [])
        self.assertEqual(self.planner.route, [])
        self.planner.pending = Future()
        waiting = self.planner.plan(self.ego, self.route, [])
        self.assertEqual(waiting.debug_details['static_avoidance']['corridor_search'],
                         details['corridor_search'])

    def test_preplanned_route_is_rechecked_against_latest_gt(self):
        planner = RecordedStaticPlanner(LatticePlanner(), background=True)
        try:
            planner._submit(self.ego, self.route, [])
            worker = planner.pending.result(timeout=12.)
            point = worker.path[80]
            obstacle = ObjectInfo(point.x, point.y, 0., 2, is_static=True, size_x=2., size_y=2.)
            result = planner.plan(self.ego, self.route, [obstacle])
            self.assertEqual(result.acc_target_velocity, 0.)
            self.assertEqual(result.debug_details['static_avoidance']['error'], 'remaining_path_collision')
        finally:
            planner.executor.shutdown(wait=True)

    def test_moving_approach_tracks_preplanned_route_without_zero_target(self):
        import contextlib
        import io
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.control.pure_pursuit import PurePursuit
        from expert.src.planning.section_planning import LatticePlanningProcess
        process = LatticePlanningProcess(highway_zone_points=())
        route = [Point(-59.85, -220.+.2*i) for i in range(625)]
        obstacle = ObjectInfo(-59.85, -145., 0., 2, is_static=True, size_x=1., size_y=1.)
        entries = [{'object_info': obstacle}]
        ego = VehicleState(-59.85, -195., math.pi/2, 20./3.6)
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.control_dt = dt = .1
        controller.previous_target_velocity = ego.velocity
        controller.pure_pursuit = PurePursuit(.65, 3., 2., 30.)
        planner = process.recorded_static
        entered = False
        try:
            planner.preplan(ego, route, [obstacle])
            planner.pending.result(timeout=12.)
            with contextlib.redirect_stdout(io.StringIO()):
                for step in range(1200):
                    reference = [v for v in route if ego.position.y-.25 <= v.y <= ego.position.y+30.]
                    result = process.run(ego, reference, entries, prediction_path=route, global_path=route)
                    target = controller._smooth_static_target_speed(min(20./3.6, result.acc_target_velocity), ego, result)
                    self.assertGreater(target, 0., (step, ego.position, result.debug_details))
                    if result.active and not entered:
                        self.assertGreater(ego.velocity, .3)
                        entered = True
                    controller.previous_target_velocity = target
                    pp = controller.pure_pursuit
                    pp.path, pp.vehicle_state = result.path, ego
                    steering = float(np.clip(pp.calculate_steering_angle(controller._recorded_lookahead_limits(ego, result)),
                                             -math.radians(40.), math.radians(40.)))
                    speed = max(0., ego.velocity+float(np.clip(target-ego.velocity, -2.*dt, 1.2*dt)))
                    ego = VehicleState(ego.position.x+speed*dt*math.cos(ego.yaw),
                                       ego.position.y+speed*dt*math.sin(ego.yaw),
                                       ego.yaw+speed*dt*math.tan(steering)/3., speed)
                    self.assertTrue(planner.collision_free(np.array([ego.position]), np.array([ego.yaw]), [obstacle]))
                    if planner.done:
                        break
                else:
                    self.fail('Moving approach did not finish')
            self.assertTrue(entered)
            self.assertEqual(planner.plan_count, 1)
        finally:
            planner.executor.shutdown(wait=True)

    def test_bicycle_tracking_through_scene_without_local_path(self):
        from expert.src.control.pure_pursuit import PurePursuit
        import contextlib
        import io
        scene = json.loads((Path(__file__).parent/'static_scene_20260914.json').read_text())
        obstacles = [ObjectInfo(**row) for row in scene['objects']]
        self.start(obstacles)
        ego = self.ego
        pursuit = PurePursuit(.65, 3., 1., 1.5)
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(500):
                result = self.planner.plan(ego, [], obstacles)
                self.assertGreater(result.acc_target_velocity, 0., (step, ego.position, result.reason))
                pursuit.path, pursuit.vehicle_state = result.path, ego
                steer = float(np.clip(pursuit.calculate_steering_angle((1.,1.5)), -math.pi/6, math.pi/6))
                speed = min(result.acc_target_velocity, 2.)
                ego = VehicleState(ego.position.x+.1*speed*math.cos(ego.yaw),
                    ego.position.y+.1*speed*math.sin(ego.yaw), ego.yaw+.1*speed*math.tan(steer)/3., speed)
                if ego.position.y >= self.planner.rejoin_y:
                    break
            else:
                self.fail('Did not return to global route')

    def test_bicycle_handoff_has_no_zero_target_or_acceleration_jump(self):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.control.pure_pursuit import PurePursuit
        from expert.src.planning.lattice import parameters
        import contextlib
        import io
        scene = json.loads((Path(__file__).parent/'static_scene_20260914.json').read_text())
        obstacles = [ObjectInfo(**row) for row in scene['objects']]
        self.start(obstacles)
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.control_dt = .1
        controller.previous_target_velocity = 2.
        controller.pure_pursuit = PurePursuit(.65, 3., 2., 30.)
        ego = VehicleState(self.ego.position.x, self.ego.position.y, self.ego.yaw, 2.)
        handed_off, previous_steer = False, 0.
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(500):
                reference = [point for point in self.route if point.y >= ego.position.y-.25]
                result = self.planner.plan(ego, reference, obstacles)
                target = controller._smooth_static_target_speed(min(16.67, result.acc_target_velocity), ego, result)
                self.assertGreater(target, 0., (ego.position, result.debug_details))
                self.assertLessEqual(target-controller.previous_target_velocity,
                                     parameters.STATIC_TARGET_ACCEL*.1+1e-8)
                controller.previous_target_velocity = target
                controller.pure_pursuit.path = result.path
                controller.pure_pursuit.vehicle_state = ego
                limits = controller._recorded_lookahead_limits(ego, result)
                steer = float(np.clip(controller.pure_pursuit.calculate_steering_angle(limits), -math.pi/6, math.pi/6))
                if result.reason == 'static_recorded_rejoined' and not handed_off:
                    self.assertLess(abs(steer-previous_steer), math.radians(5.))
                    handed_off = True
                previous_steer = steer
                speed = max(0., ego.velocity+float(np.clip(target-ego.velocity, -.2, .12)))
                ego = VehicleState(ego.position.x+.1*speed*math.cos(ego.yaw),
                    ego.position.y+.1*speed*math.sin(ego.yaw), ego.yaw+.1*speed*math.tan(steer)/3., speed)
                if handed_off and ego.position.y > self.planner.handoff_y+5.:
                    break
            else:
                self.fail('Continuous handoff did not finish')
        self.assertTrue(handed_off)


if __name__ == '__main__':
    unittest.main()
