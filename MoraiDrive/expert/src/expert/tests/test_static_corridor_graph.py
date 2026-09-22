import importlib
import json
import math
from pathlib import Path
import unittest
import contextlib
import io
import time

import numpy as np

from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.lattice_planning.lattice_planner import LatticePlanner


module = importlib.import_module('expert.src.planning.\uC815\uC801\uC7A5\uC560\uBB3C.recorded_planner')


def captured_scene():
    root = Path(__file__).resolve().parents[3]
    data = json.loads((Path(__file__).parent/'static_scene_20260916_zigzag.json').read_text())
    route = [Point(*map(float, row.split()[:2])) for row in
             (root/'map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt').read_text().splitlines()
             if row.strip()]
    x, y, heading = data['ego']
    return VehicleState(x, y, math.radians(heading), 0.), route, [ObjectInfo(**v) for v in data['objects']]


class CorridorGraphTests(unittest.TestCase):
    def test_sparse_box_has_twenty_kph_avoidance_and_return_geometry(self):
        ego, route, _ = captured_scene()
        objects = [ObjectInfo(-59.9, -143., 0., 2, is_static=True, size_x=1.2, size_y=1.2)]
        planner = module.RecordedStaticPlanner(LatticePlanner())
        result = planner.plan(ego, route, objects, route)
        self.assertTrue(result.active, result.debug_details)
        xy = np.array(planner.path)
        yaw = np.array([planner.planner._path_heading(planner.path, i) for i in range(len(xy))])
        self.assertTrue(planner._envelope_clear(xy, yaw))
        self.assertTrue(planner.collision_free(xy, yaw, objects, .12))
        self.assertGreater(planner.handoff_y-planner.return_start_y, 10.)
        # This recorded-map scene previously selected right_03 and a 4 m return.
        # Its slew limit dropped below 8 km/h even with perfect tracking.
        for y in np.arange(-153., planner.handoff_y, .5):
            index = int(np.argmin(abs(xy[:, 1]-y)))
            speed, _ = planner._curve_speed_limit(planner.path[index:])
            self.assertGreaterEqual(speed*3.6, 20., (y, planner.curve_speed_limits))
        # A newly observed obstruction must still stop the smoothed maneuver.
        point = planner.path[100]
        blocked = ObjectInfo(point.x, point.y, 0., 2, is_static=True, size_x=1., size_y=1.)
        self.assertEqual(planner.plan(ego, route, objects+[blocked]).acc_target_velocity, 0.)

    def test_sparse_box_tracks_twenty_without_a_second_return_slowdown(self):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.control.pure_pursuit import PurePursuit
        for lag, gain in ((.2, 1.), (.45, .75)):
            with self.subTest(lag=lag, gain=gain):
                ego, route, _ = captured_scene()
                ego.velocity = 20./3.6
                objects = [ObjectInfo(-59.9, -143., 0., 2, is_static=True, size_x=1.2, size_y=1.2)]
                planner = module.RecordedStaticPlanner(LatticePlanner())
                controller = AutonomousDriving.__new__(AutonomousDriving)
                controller.control_dt = dt = 1./30.
                controller.previous_target_velocity = ego.velocity
                controller.pure_pursuit = PurePursuit(.65, 3., 2., 30.)
                steering, avoidance_speeds, return_speeds = 0., [], []
                with contextlib.redirect_stdout(io.StringIO()):
                    for step in range(1500):
                        reference = [v for v in route if ego.position.y-.25 <= v.y < -90. and -65. < v.x < -54.]
                        result = planner.plan(ego, reference, objects, route)
                        self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
                        target = controller._smooth_static_target_speed(min(20./3.6, result.acc_target_velocity), ego, result)
                        controller.previous_target_velocity = target
                        pp = controller.pure_pursuit
                        pp.path, pp.vehicle_state = result.path, ego
                        command = float(np.clip(pp.calculate_steering_angle(controller._recorded_lookahead_limits(ego, result)),
                                                -math.radians(40.), math.radians(40.)))
                        steering += float(np.clip((gain*command-steering)*dt/lag,
                                                  -math.radians(45.)*dt, math.radians(45.)*dt))
                        speed = max(0., ego.velocity+float(np.clip(target-ego.velocity, -3.*dt, 1.2*dt)))
                        if -150. <= ego.position.y < planner.return_start_y:
                            avoidance_speeds.append(speed*3.6)
                        if planner.return_start_y <= ego.position.y < planner.handoff_y+3.:
                            return_speeds.append(speed*3.6)
                        ego = VehicleState(ego.position.x+speed*dt*math.cos(ego.yaw),
                                           ego.position.y+speed*dt*math.sin(ego.yaw),
                                           ego.yaw+speed*dt*math.tan(steering)/3., speed)
                        if result.active:
                            self.assertTrue(planner._envelope_clear(np.array([ego.position]), np.array([ego.yaw])),
                                            (step, ego.position, result.debug_details))
                        self.assertTrue(planner.collision_free(np.array([ego.position]), np.array([ego.yaw]), objects))
                        if planner.done and ego.position.y > planner.handoff_y+3.:
                            break
                    else:
                        self.fail('Sparse avoidance did not finish')
                self.assertTrue(avoidance_speeds)
                self.assertTrue(return_speeds)
                if gain == 1.:
                    self.assertGreaterEqual(min(avoidance_speeds), 19.)
                    self.assertGreaterEqual(min(return_speeds), 19.)
                else:
                    # Sluggish/weak steering must retain the alignment slowdown.
                    self.assertGreater(min(avoidance_speeds), 10.)
                    self.assertGreater(min(return_speeds), 10.)
                self.assertEqual(planner.plan_count, 1)

    def test_gentle_tracking_preserves_route_cruise(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        for bend in (0., .001):
            with self.subTest(bend=bend):
                points = [Point(-60.+bend*s*s, -160.+s) for s in np.linspace(0., 160., 1601)]
                planner.path = points
                self.assertLess(planner.planner._max_curvature(points), .08)
                result = planner._result(points, True, 'follow')
                self.assertAlmostEqual(result.acc_target_velocity*3.6, 60.)

    def test_bend_preview_and_recovery_keep_slow_target(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        points = [Point(-60.+.12*s*s, -160.+s) for s in np.linspace(0., 10., 101)]
        planner.path = points
        self.assertGreater(planner.planner._max_curvature(points), .08)
        self.assertFalse(planner.alignment_slow)
        speed = planner._result(points, True, 'follow').acc_target_velocity
        self.assertGreater(speed*3.6, 10.)
        self.assertLess(speed*3.6, 15.)
        straight = [Point(-60., -160.+s) for s in np.linspace(0., 10., 101)]
        planner.path = straight
        planner.recovery_until_y = -155.
        self.assertAlmostEqual(planner._result(straight, True, 'recover_alignment').acc_target_velocity*3.6, 3.6)
        planner.recovery_until_y = None
        self.assertGreater(planner._result(straight, True, 'follow').acc_target_velocity*3.6, 20.)

    def test_alignment_target_preserves_stop_limits(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        points = [Point(-60., -160.+.1*i) for i in range(61)]
        planner.path = points
        planner.alignment_slow = True
        for heading, lateral in ((module.ALIGNMENT_FULL_HEADING_ERROR, 0.),
                                 (0., module.ALIGNMENT_FULL_CROSS_TRACK_ERROR)):
            planner.heading_error, planner.cross_track_error = heading, lateral
            result = planner._result(points, True, 'follow')
            self.assertAlmostEqual(result.acc_target_velocity*3.6, 3.6)
        self.assertEqual(planner._result(points, False, 'blocked').acc_target_velocity, 0.)
        self.assertEqual(planner._result(points[-3:], True, 'follow').acc_target_velocity, 0.)

    def test_small_heading_error_has_progressive_slowdown(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        planner.path = [Point(-60., -160.+.1*i) for i in range(101)]
        planner.alignment_slow = True
        speeds = []
        for error in (1.99, 2.01, 3., 4., 5.):
            planner.heading_error = math.radians(error)
            speeds.append(planner._result(planner.path, True, 'follow').acc_target_velocity*3.6)
        self.assertGreater(speeds[0], 20.)
        self.assertLess(speeds[0]-speeds[1], .2)
        self.assertGreater(speeds[2], 3.6)
        self.assertLess(speeds[2], speeds[0])
        self.assertAlmostEqual(speeds[3], 3.6)
        self.assertAlmostEqual(speeds[4], 3.6)

    def test_moderate_curvature_is_not_a_binary_crawl_limit(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        speeds = []
        for curvature in (.0799, .0801, .12, .18):
            radius = 1./curvature
            angles = np.linspace(0., 10./radius, 201)
            points = [Point(-60.+radius*(1-math.cos(a)), -160.+radius*math.sin(a)) for a in angles]
            planner.path = points
            speeds.append(planner._result(points, True, 'follow').acc_target_velocity*3.6)
        self.assertAlmostEqual(speeds[0], 3.6*math.sqrt(1.5/.0799)+5., places=5)
        self.assertLess(speeds[0]-speeds[1], .02)
        self.assertGreater(speeds[2], 3.6)
        self.assertLess(speeds[2], speeds[0])
        self.assertAlmostEqual(speeds[3], 3.6*math.sqrt(1.5/.18)+5., places=5)

    def test_curvature_offset_adds_five_kph_across_turn_radii(self):
        from unittest.mock import patch
        # Analytic circular arcs isolate curvature from slew and corridor
        # limits; check both signs and ensure the final stop still wins.
        for curvature in (.03, .05, .10, .15, .24):
            for sign in (-1., 1.):
                radius = 1./curvature
                angles = np.linspace(0., 20./radius, 401)
                points = [Point(sign*radius*(1-math.cos(a)), radius*math.sin(a)) for a in angles]
                speeds = []
                for offset in (0., 5.):
                    with patch.object(module.p, 'STATIC_CURVATURE_SPEED_OFFSET_KPH', offset):
                        planner = module.RecordedStaticPlanner(LatticePlanner())
                        planner.path = points
                        result = planner._result(points, True, 'follow')
                        speeds.append(result.acc_target_velocity*3.6)
                        self.assertAlmostEqual(speeds[-1], 3.6*math.sqrt(1.5/curvature)+offset, places=5)
                        self.assertEqual(planner._result(points, False, 'blocked').acc_target_velocity, 0.)
                        self.assertEqual(planner._result(points[-3:], True, 'follow').acc_target_velocity, 0.)
                self.assertAlmostEqual(speeds[1]-speeds[0], 5., places=5)

    def test_upcoming_tight_bend_has_a_braking_profile(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        points = [Point(-60., -160.+s) for s in np.linspace(0., 10., 101)]
        points += [Point(-60.+5*(1-math.cos(a)), -150.+5*math.sin(a))
                   for a in np.linspace(0., 1.4, 141)[1:]]
        planner.path = points
        targets = [planner._result(points[i:], True, 'follow').acc_target_velocity for i in range(101)]
        self.assertGreater(targets[0]*3.6, 10.)
        self.assertGreater(min(targets), 0.)
        self.assertLess(min(targets), math.sqrt(1.5/.2))
        minimum = int(np.argmin(targets))
        for before, after in zip(targets[:minimum], targets[1:minimum+1]):
            self.assertLessEqual(after, before+1e-8)
            self.assertLessEqual(before**2-after**2, 2*module.p.STATIC_END_DECELERATION*.1+1e-8)

    def test_waypoint_spacing_does_not_latch_alignment_slowdown(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        points = [Point(-59.85, -162.+.5*i) for i in range(120)]
        ego = VehicleState(-59.85, -160.25, math.pi/2, 1.)
        planner._setup(points, ego)
        planner.path = points
        planner.alignment_slow = True
        result = planner.plan(ego, [], [])
        details = result.debug_details['static_avoidance']
        self.assertAlmostEqual(details['tracking_error_m'], .25)
        self.assertAlmostEqual(details['cross_track_error_m'], 0.)
        self.assertFalse(details['alignment_slow'])
        cruise = result.acc_target_velocity*3.6
        self.assertGreater(cruise, 20.)
        # Ordinary 11 cm tracking error no longer reduces an otherwise clear
        # straight-path target; larger error still slows and eventually crawls.
        result = planner.plan(VehicleState(-59.74, -160.25, math.pi/2, 1.), [], [])
        self.assertFalse(result.debug_details['static_avoidance']['alignment_slow'])
        self.assertAlmostEqual(result.acc_target_velocity*3.6, cruise)
        offset = VehicleState(-59.70, -160.25, math.pi/2, 1.)
        result = planner.plan(offset, [], [])
        self.assertTrue(result.debug_details['static_avoidance']['alignment_slow'])
        self.assertGreater(result.acc_target_velocity*3.6, 3.6)
        self.assertLess(result.acc_target_velocity*3.6, cruise)
        result = planner.plan(VehicleState(-59.59, -160.25, math.pi/2, 1.), [], [])
        # This 26 cm deviation leaves the recorded body envelope. Relaxing the
        # speed policy must not lift the independent boundary stop.
        self.assertEqual(result.acc_target_velocity, 0.)
        self.assertEqual(result.debug_details['static_avoidance']['error'], 'recorded_envelope')

    def test_tangent_is_interpolated_at_the_measured_projection(self):
        planner = module.RecordedStaticPlanner(LatticePlanner())
        ys = np.linspace(-160., -150., 51)
        planner.path = [Point(-60.+.04*(y+160.)**2, y) for y in ys]
        y = -157.51
        ego = VehicleState(-60.+.04*(y+160.)**2, y, math.atan2(1., .08*(y+160.)), 1.)
        nearest = min(range(len(ys)), key=lambda i: ego.position.distance(planner.path[i]))
        cross, heading = planner._tracking_alignment(ego, nearest)
        self.assertLess(cross, .001)
        self.assertLess(math.degrees(heading), .01)

    def test_recorded_04_log_poses_use_distinct_speed_limits(self):
        # Speed-policy replay from the supplied log; GT collision regression uses
        # the captured scene tests, not an assumed obstacle layout for this trace.
        ego, route, _ = captured_scene()
        planner = module.RecordedStaticPlanner(LatticePlanner())
        planner._setup(route, ego)
        curve = planner.curves[3]
        start, end = -134.2, -121.7
        tail = module.connector(planner._state(curve, start), planner._state(planner.global_curve, end))
        ys = np.arange(-155., planner.goal_y, .1)
        xs = np.where(ys < start, curve(ys), np.where(ys < end, tail(ys), planner.global_curve(ys)))
        planner.path = [Point(x,y) for x,y in zip(xs,ys)]
        planner.handoff_y = end
        cases = [(-59.31, -154.47, 79.84), (-58.14, -150.98, 66.04),
                 (-56.99, -148.31, 68.23), (-56.18, -145.99, 73.57),
                 (-55.14, -141.43, 81.72), (-55.25, -136.19, 101.63),
                 (-55.73, -134.63, 109.88)]
        for x, y, heading in cases:
            with self.subTest(y=y):
                result = planner.plan(VehicleState(x,y,math.radians(heading),1.), [], [])
                details = result.debug_details['static_avoidance']
                self.assertIsNone(details['error'])
                self.assertIn(details['speed_policy_reason'], ('curvature', 'alignment', 'path_end', 'tracking'))
                self.assertGreater(result.acc_target_velocity, 0.)
                self.assertLessEqual(result.acc_target_velocity, module.TRACKING_SPEED)

    def test_speed_ramp_does_not_raise_or_delay_stop_requests(self):
        from expert.src.autonomous_driving import AutonomousDriving
        planner = module.RecordedStaticPlanner(LatticePlanner())
        planner.path = [Point(-60., -160.+.1*i) for i in range(101)]
        result = planner._result(planner.path, True, 'follow')
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.control_dt = 1./30.
        controller.previous_target_velocity = 1.
        ego = VehicleState(-60., -160., math.pi/2, 1.)
        target = controller._smooth_static_target_speed(result.acc_target_velocity, ego, result)
        self.assertAlmostEqual(target, 1.+module.p.STATIC_TARGET_ACCEL/30.)
        # A downstream signal/ACC stop takes precedence over the gentle-path cap.
        self.assertEqual(controller._smooth_static_target_speed(0., ego, result), 0.)

    def test_batched_connector_matches_scalar(self):
        a = (0., 2., .1, .01)
        bs = np.array([(8., 3., -.2, 0.), (8., 4., .3, -.01)])
        batch = module.connector(a, (8., bs[:, 1], bs[:, 2], bs[:, 3]))
        ys = np.linspace(0., 8., 101)
        for d in range(3):
            expected = np.array([module.connector(a, tuple(b))(ys, d) for b in bs])
            np.testing.assert_allclose(batch(ys, d), expected, atol=1e-10)

    def test_captured_staggered_obstacles(self):
        ego, route, objects = captured_scene()
        planner = module.RecordedStaticPlanner(LatticePlanner())
        result = planner.plan(ego, route, objects, route)
        self.assertGreater(result.acc_target_velocity, 0., (result.reason, planner.search_details))
        self.assertEqual(planner.search_details['status'], 'complete')
        xy = np.array(result.path)
        yaw = np.array([planner.planner._path_heading(result.path, i) for i in range(len(xy))])
        self.assertTrue(planner._envelope_clear(xy, yaw))
        self.assertTrue(planner.collision_free(xy, yaw, objects, .12))
        self.assertLessEqual(planner.planner._max_curvature(result.path), module.MAX_CURVATURE)
        self.assertTrue(np.all(np.diff(xy[:, 1]) > 0.))
        self.assertAlmostEqual(xy[-1, 0], float(planner.global_curve(xy[-1, 1])), places=6)
        # Return left after passing the middle obstacle on its right.
        self.assertGreater(np.interp(-146., xy[:, 1], xy[:, 0]), np.interp(-153., xy[:, 1], xy[:, 0])+.5)
        return_section = xy[(xy[:, 1] > -145.) & (xy[:, 1] < -133.), 0]
        self.assertLess(return_section.min(), np.interp(-146., xy[:, 1], xy[:, 0])-.25)

    def test_zigzag_tracking_with_steering_lag(self):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.control.pure_pursuit import PurePursuit
        ego, route, objects = captured_scene()
        planner = module.RecordedStaticPlanner(LatticePlanner())
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.control_dt = dt = 1./30.
        controller.previous_target_velocity = 0.
        controller.pure_pursuit = PurePursuit(.65, 3., 2., 30.)
        steering = 0.
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(2000):
                reference = [v for v in route if ego.position.y-.25 <= v.y < -90. and -65. < v.x < -54.]
                result = planner.plan(ego, reference, objects, route)
                self.assertGreater(result.acc_target_velocity, 0., (step, ego.position, ego.velocity, result.debug_details))
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
                self.assertTrue(planner.collision_free(np.array([ego.position]), np.array([ego.yaw]), objects))
                if planner.done and ego.position.y > planner.handoff_y+5.:
                    break
            else:
                self.fail('Zigzag tracking did not reach the global route')
        self.assertEqual(planner.plan_count, 1)

    def test_blocked_corridor_does_not_relax_clearance(self):
        ego, route, _ = captured_scene()
        obstacle = ObjectInfo(-56., -148., 0., 2, is_static=True, size_x=20., size_y=2.)
        planner = module.RecordedStaticPlanner(LatticePlanner())
        result = planner.plan(ego, route, [obstacle], route)
        self.assertEqual(result.acc_target_velocity, 0.)
        self.assertFalse(result.active)

    def test_captured_stop_reconnects_without_removing_obstacle_clearance(self):
        ego, route, objects = captured_scene()
        planner = module.RecordedStaticPlanner(LatticePlanner())
        planner.plan(ego, route, objects, route)
        stopped = VehicleState(-59.63, -151.02, math.radians(81.72), 0.)
        xy, yaw = np.array([stopped.position]), np.array([stopped.yaw])
        self.assertFalse(planner.collision_free(xy, yaw, objects))
        self.assertTrue(planner.collision_free(xy, yaw, objects, tracking_reserve=False))
        original = np.array(planner.path)
        result = planner.plan(stopped, [], objects)
        self.assertEqual(result.reason, 'static_recorded_recover_alignment', result.debug_details)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertLessEqual(result.acc_target_velocity, module.ALIGNMENT_SPEED)
        self.assertEqual(planner.recovery_count, 1)
        self.assertEqual(planner.planner.collision_margin, .5)
        tail = original[original[:, 1] > planner.recovery_until_y]
        np.testing.assert_array_equal(np.array(planner.path)[-len(tail):], tail)
        obstacle = ObjectInfo(stopped.position.x, stopped.position.y+2., 0., 2,
                              is_static=True, size_x=.75, size_y=.75)
        blocked = planner.plan(stopped, [], objects+[obstacle])
        self.assertEqual(blocked.acc_target_velocity, 0.)
        self.assertEqual(blocked.debug_details['static_avoidance']['error'], 'ego_collision')

    def test_captured_stop_tracking_with_slower_steering_response(self):
        self._drive_with_slower_response(start_at_stop=True)

    def test_initial_run_with_slower_steering_response(self):
        self._drive_with_slower_response(start_at_stop=False)

    def _drive_with_slower_response(self, start_at_stop):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.control.pure_pursuit import PurePursuit
        ego, route, objects = captured_scene()
        planner = module.RecordedStaticPlanner(LatticePlanner())
        planner.plan(ego, route, objects, route)
        if start_at_stop:
            ego = VehicleState(-59.63, -151.02, math.radians(81.72), 0.)
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.control_dt = dt = 1./30.
        controller.previous_target_velocity = 0.
        controller.pure_pursuit = PurePursuit(.65, 3., 2., 30.)
        steering = -math.radians(18.5) if start_at_stop else 0.
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(2200):
                reference = [v for v in route if ego.position.y-.25 <= v.y < -90. and -65. < v.x < -54.]
                result = planner.plan(ego, reference, objects, route)
                self.assertGreater(result.acc_target_velocity, 0., (step, ego.position, result.debug_details))
                target = controller._smooth_static_target_speed(min(16.67, result.acc_target_velocity), ego, result)
                controller.previous_target_velocity = target
                pp = controller.pure_pursuit
                pp.path, pp.vehicle_state = result.path, ego
                command = float(np.clip(pp.calculate_steering_angle(controller._recorded_lookahead_limits(ego, result)),
                                        -math.radians(40.), math.radians(40.)))
                steering += float(np.clip((.75*command-steering)*dt/.45,
                                          -math.radians(45.)*dt, math.radians(45.)*dt))
                speed = max(0., ego.velocity+float(np.clip(target-ego.velocity, -3.*dt, 1.2*dt)))
                ego = VehicleState(ego.position.x+speed*dt*math.cos(ego.yaw),
                                   ego.position.y+speed*dt*math.sin(ego.yaw),
                                   ego.yaw+speed*dt*math.tan(steering)/3., speed)
                self.assertTrue(planner.collision_free(np.array([ego.position]), np.array([ego.yaw]),
                                                       objects, tracking_reserve=False))
                if planner.done and ego.position.y > planner.handoff_y+5.:
                    break
            else:
                self.fail('Captured stop did not recover with actuator lag')

    def test_async_zigzag_preplans_once_and_holds_without_local_path(self):
        ego, route, objects = captured_scene()
        planner = module.RecordedStaticPlanner(LatticePlanner(), background=True)
        try:
            begin = time.monotonic()
            result = planner.plan(ego, route, objects, route)
            self.assertLess(time.monotonic()-begin, .2)
            self.assertGreater(result.acc_target_velocity, 0.)
            planner.pending.result(timeout=12.)
            result = planner.plan(ego, [], objects)
            self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
            original = np.array(planner.path)
            for index in (20, 90, 140, 200):
                point = planner.path[index]
                pose = VehicleState(point.x, point.y, planner.planner._path_heading(planner.path, index), 1.)
                result = planner.plan(pose, [], objects)
                self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
                np.testing.assert_array_equal(original, planner.path)
            self.assertEqual(planner.plan_count, 1)
        finally:
            planner.executor.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
