import math
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.control.pid import Pid
from expert.src.control.pure_pursuit import PurePursuit
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.adaptive_cruise_control import AdaptiveCruiseControl
from expert.src.planning.lattice.lattice_planner import LatticePlanner
from expert.src.planning.lattice import parameters
from expert.src.planning.section_planning import LatticePlanningProcess
from expert.src.vehicle_state import VehicleState


class StaticAvoidanceTest(unittest.TestCase):
    def setUp(self):
        self.path = [Point(i * 0.5, 0.0) for i in range(161)]
        self.ego = VehicleState(0.0, 0.0, 0.0, 0.0)
        self.obstacle = ObjectInfo(15.0, 0.0, 0.0, 2, is_static=True, size_x=2.0, size_y=1.0)
        self.entries = [{"object_info": self.obstacle, "local_position": Point(15.0, 0.0)}]
        self.process = LatticePlanningProcess(
            zone_bounds=(-5.0, 60.0, -1.0, 1.0), highway_zone_points=(),
        )

    def start(self):
        result = self.process.run(self.ego, self.path[:50], self.entries, prediction_path=self.path)
        self.assertTrue(result.active, result.reason)
        self.assertGreater(result.acc_target_velocity, 0.0)
        self.assertTrue(result.debug_details["static_avoidance"]["path_validated"])
        return result

    def at(self, path, index):
        point = path[index]
        return VehicleState(point.x, point.y, LatticePlanner._path_heading(path, index), 2.0)

    def test_curvature_offset_also_applies_to_lattice_fallback(self):
        maneuver = self.process.static_avoidance
        radius = 10.
        points = [Point(radius*math.sin(i*.005), radius*(1-math.cos(i*.005))) for i in range(401)]
        maneuver.path = points
        speeds = []
        for offset in (0., 5.):
            with patch.object(parameters, 'STATIC_CURVATURE_SPEED_OFFSET_KPH', offset):
                speeds.append(maneuver._result(points, True, 'avoid').acc_target_velocity*3.6)
                self.assertEqual(maneuver._result(points, False, 'blocked').acc_target_velocity, 0.)
        self.assertAlmostEqual(speeds[1]-speeds[0], 5., places=5)
        self.assertAlmostEqual(speeds[1], 3.6*math.sqrt(1.5/.1)+5., places=5)

    def finish_leg(self, maneuver):
        return maneuver.plan(self.at(maneuver.path, len(maneuver.path) - 8), [], [])

    def test_zone_exit_and_missing_reference_keep_world_path(self):
        self.start()
        held = list(self.process.static_avoidance.path)
        index = next(i for i, point in enumerate(held) if point.y < -1.1)
        ego = self.at(held, index)
        self.assertFalse(self.process._is_inside_static_zone(ego))
        result = self.process.run(ego, [], [])
        self.assertTrue(result.active)
        self.assertEqual("static_obstacle", result.mode)
        self.assertGreater(result.acc_target_velocity, 0.0)
        maneuver = self.process.static_avoidance
        self.assertEqual(self.path, maneuver.route)
        self.assertFalse(maneuver._recovery_ready)
        self.assertEqual(held, maneuver.path)
        result = self.finish_leg(maneuver)
        self.assertTrue(maneuver._recovery_ready)
        self.assertEqual(held, maneuver.path[:len(held)])
        self.assertAlmostEqual(0.0, result.path[-1].y)
        self.assertEqual('lattice_endpoint', result.debug_details['static_avoidance']['recovery_trigger'])
        self.assertEqual(len(held) - 1, maneuver.return_index)

    def test_return_starts_only_at_unchanged_lattice_endpoint_and_rejoins_route(self):
        self.start()
        maneuver = self.process.static_avoidance
        self.finish_leg(maneuver)
        held = list(maneuver.path)
        self.assertTrue(maneuver._clear(held[maneuver.return_index:], [self.obstacle], self.ego))
        self.assertAlmostEqual(0.0, held[maneuver.rejoin_index].y, places=5)
        saw_return = False
        for index in range(maneuver.progress + 1, len(held)):
            ego = self.at(held, index)
            local = self.path[min(150, int(ego.position.x * 2)):]
            result = self.process.run(ego, local, [])
            if result.reason == "static_rejoined":
                self.assertTrue(saw_return)
                self.assertFalse(maneuver.active)
                break
            self.assertTrue(result.active)
            self.assertGreater(result.acc_target_velocity, 0.0, result.reason)
            self.assertFalse(self.process.planner._obb_collision(ego.position, ego.yaw, self.obstacle))
            saw_return |= result.reason == "static_return"
        else:
            self.fail("The bypass never rejoined the reference route")

    def test_lattice_grid_is_dense_inside_reduced_width(self):
        with patch.object(self.process.planner, '_generate_candidates',
                          wraps=self.process.planner._generate_candidates) as generate:
            result = self.start()
        self.assertTrue(generate.called)
        self.assertTrue(all(len(call.kwargs['offsets']) == 17 for call in generate.call_args_list))
        self.assertFalse(self.process.static_avoidance._recovery_ready)
        self.assertEqual(34, len(result.candidates))
        self.assertEqual([-3.0 + i * .375 for i in range(17)], [c.offset for c in result.candidates[:17]])
        self.assertTrue(all(not c.valid for c in result.candidates if c.offset > 0))
        self.assertEqual(1, sum(c.points is result.path for c in result.candidates))
        self.assertLess(result.selected_offset, 0.0)
        self.assertGreaterEqual(result.selected_offset, -parameters.STATIC_MAX_RIGHT_OFFSET)
        self.assertTrue(all(point.y <= 0.1 for point in result.path))
        self.assertGreater(len(result.path), 50)

    @patch.multiple(parameters, STATIC_MAX_RIGHT_OFFSET=8.0, STATIC_RIGHT_BOUNDARY=-10.0)
    def test_explicit_legacy_width_recording_holds_endpoint_return(self):
        data = json.loads(Path(__file__).with_name('static_scene_20260914.json').read_text())
        ego = VehicleState(*data['ego'])
        route = [Point(*xy) for xy in data['route']]
        objects = [ObjectInfo(**item) for item in data['objects']]
        maneuver = self.process.static_avoidance
        maneuver.reset()
        result = maneuver.plan(ego, route[:50], objects, route)
        self.assertTrue(result.active, result.debug_details)
        self.assertIn(result.selected_offset, (-7.0, -8.0))
        self.assertEqual(29, len(objects))
        self.assertFalse(maneuver._recovery_ready)
        entry = list(maneuver.path)
        result = self.finish_leg(maneuver)
        self.assertTrue(maneuver._recovery_ready, result.debug_details)
        details = result.debug_details['static_avoidance']
        self.assertEqual(entry, maneuver.path[:len(entry)])
        self.assertEqual([entry[-1].x, entry[-1].y], details['return_start'])
        self.assertLessEqual(details['rejoin_distance'], parameters.STATIC_RETURN_MAX_DISTANCE)
        self.assertLess(details['rejoin_point'][1], -108.0)
        self.assertIs(route[details['rejoin_route_index']], maneuver.path[maneuver.rejoin_index])
        held = list(maneuver.path)
        for index in range(maneuver.progress + 1, maneuver.rejoin_index, 15):
            result = maneuver.plan(self.at(held, index), [], [])
            self.assertTrue(result.debug_details['static_avoidance']['path_validated'], result.debug_details)
            self.assertEqual(34, len(result.candidates))
            self.assertEqual(held, maneuver.path)
        result = maneuver.plan(self.at(held, maneuver.rejoin_index), route, objects)
        self.assertEqual('static_rejoined', result.reason)

    @patch.multiple(parameters, STATIC_MAX_RIGHT_OFFSET=8.0, STATIC_RIGHT_BOUNDARY=-10.0)
    def test_explicit_legacy_width_recording_with_actual_steering_limit(self):
        data = json.loads(Path(__file__).with_name('static_scene_20260914.json').read_text())
        ego = VehicleState(*data['ego'])
        route = [Point(*xy) for xy in data['route']]
        objects = [ObjectInfo(**item) for item in data['objects']]
        maneuver = self.process.static_avoidance
        maneuver.reset()
        pursuit = PurePursuit(.65, 3.0, 2.0, 30.0)
        saw_return = False
        for _ in range(1600):
            result = maneuver.plan(ego, route, objects, route)
            if result.reason == 'static_rejoined':
                self.assertTrue(saw_return)
                self.assertLess(ego.position.y, -108.0)
                break
            self.assertTrue(result.active, result.debug_details)
            self.assertGreater(result.acc_target_velocity, 0, result.debug_details)
            saw_return |= result.reason == 'static_return'
            pursuit.path, pursuit.vehicle_state = result.path, ego
            steer = pursuit.calculate_steering_angle(lookahead_limits=parameters.STATIC_TRACKING_LOOKAHEAD)
            steer = max(-math.pi / 6, min(math.pi / 6, steer))
            speed = max(0, ego.velocity + max(-3, min(1.5, (result.acc_target_velocity - ego.velocity) * 2)) * .05)
            yaw = ego.yaw + speed / 3 * math.tan(steer) * .05
            ego = VehicleState(ego.position.x + speed * math.cos(ego.yaw) * .05,
                               ego.position.y + speed * math.sin(ego.yaw) * .05, yaw, speed)
            for obj in objects:
                self.assertFalse(maneuver.planner._obb_collision(ego.position, ego.yaw, obj))
        else:
            self.fail('Recorded-scene short return did not finish')

    def test_repeated_ticks_keep_one_identical_world_path(self):
        result = self.start()
        held = list(result.path)
        for _ in range(20):
            result = self.process.run(self.ego, self.path, self.entries)
            self.assertEqual(34, len(result.candidates))
            self.assertEqual(held, result.path)

    def test_obstacle_already_behind_does_not_start_another_bypass(self):
        obstacle = ObjectInfo(-8, 0, 0, 2, is_static=True, size_x=2, size_y=2)
        result = self.process.run(self.ego, self.path, [{"object_info": obstacle}])
        self.assertFalse(result.active)
        self.assertFalse(result.acc_override)

    def test_blocked_plan_does_not_flicker_clear_on_missing_object_packet(self):
        obstacle = ObjectInfo(15, 0, 0, 2, is_static=True, size_x=2, size_y=20)
        first = self.process.run(self.ego, self.path, [{"object_info": obstacle}])
        second = self.process.run(self.ego, self.path, [])
        self.assertEqual(0.0, first.acc_target_velocity)
        self.assertEqual(0.0, second.acc_target_velocity)
        self.assertEqual([], second.candidates)

    def test_left_path_rejected_even_when_no_obstacle_blocks_it(self):
        self.start()
        left_path = [Point(i * 0.2, 1.0) for i in range(60)]
        self.assertFalse(self.process.static_avoidance._clear(left_path, [], self.ego))

    def test_body_cannot_swing_outside_corridor_with_center_inside(self):
        self.start()
        maneuver = self.process.static_avoidance
        points = [Point(0.0, -2.8), Point(1.0, -3.0)]
        self.assertTrue(maneuver._right_only(points))
        self.assertFalse(maneuver._within_corridor(points, self.ego, parameters.STATIC_TRACKING_MARGIN))

    def test_oversize_blockage_stops_instead_of_widening_bypass(self):
        blocker = ObjectInfo(15.0, 0.0, 0.0, 2, is_static=True, size_x=3.0, size_y=20.0)
        result = self.process.run(self.ego, self.path[:50], [{"object_info": blocker}], prediction_path=self.path)
        self.assertFalse(result.active)
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertEqual([], result.candidates)

    def test_right_is_relative_to_route_not_world_y(self):
        route = [Point(0.0, i * 0.5) for i in range(161)]
        ego = VehicleState(0.0, 0.0, math.pi / 2, 0.0)
        obstacle = ObjectInfo(0.0, 15.0, 0.0, 2, is_static=True, size_x=1.0, size_y=2.0)
        result = self.process.run(ego, route[:50], [{"object_info": obstacle}], prediction_path=route)
        self.assertTrue(result.active, result.reason)
        self.assertLess(result.selected_offset, 0)
        self.assertTrue(all(point.x >= -0.1 for point in result.path))

    def test_vehicle_width_and_margin_reject_too_narrow_gap(self):
        checker = self.process.planner
        def walls(width):
            return [ObjectInfo(10.0, side * (width * 0.5 + 0.5), 0.0, 2,
                               is_static=True, size_x=10.0, size_y=1.0) for side in (-1, 1)]
        points = [Point(i * 0.2, 0.0) for i in range(120)]
        self.process.static_avoidance._set_route(self.path, self.ego)
        self.assertTrue(self.process.static_avoidance._clear(points, walls(3.2), self.ego))
        self.assertFalse(self.process.static_avoidance._clear(points, walls(2.8), self.ego))
        self.assertEqual(2.0, checker.vehicle_half_width * 2)
        self.assertEqual(0.5, checker.collision_margin)

    def test_gap_between_restored_grid_samples_stops_instead_of_forcing_path(self):
        obstacles = [self.obstacle,
                     ObjectInfo(24.0, 4.0, 0.0, 2, is_static=True, size_x=10.0, size_y=9.3),
                     ObjectInfo(24.0, -8.0, 0.0, 2, is_static=True, size_x=10.0, size_y=8.1)]
        entries = [{"object_info": obj} for obj in obstacles]
        fine = self.process.run(self.ego, self.path[:50], entries, prediction_path=self.path)
        self.assertFalse(fine.active)
        self.assertEqual(0.0, fine.acc_target_velocity)

    def test_new_obstacle_stops_without_dropping_maneuver(self):
        self.start()
        held = list(self.process.static_avoidance.path)
        ego = self.at(held, 4)
        point = held[12]
        blocker = ObjectInfo(point.x, point.y, 0.0, 2, is_static=True, size_x=2.0, size_y=2.0)
        result = self.process.run(ego, self.path, [{"object_info": blocker}])
        self.assertTrue(result.active)
        self.assertEqual("static_blocked", result.reason)
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertFalse(result.debug_details["static_avoidance"]["path_validated"])
        self.assertEqual(held, self.process.static_avoidance.path)

    def test_actual_pose_collision_blocks_otherwise_clear_path(self):
        self.start()
        held = self.process.static_avoidance.path
        ego = self.at(held, 8)
        blocker = ObjectInfo(ego.position.x, ego.position.y, 0.0, 2, is_static=True)
        result = self.process.run(ego, self.path, [{"object_info": blocker}])
        self.assertEqual(0.0, result.acc_target_velocity)

    def test_insufficient_return_route_holds_lattice_then_stops_at_its_end(self):
        result = self.process.run(self.ego, self.path[:38], self.entries)
        self.assertEqual("static_avoid", result.reason)
        self.assertTrue(result.active)
        maneuver = self.process.static_avoidance
        maneuver.plan(self.ego, [], [])
        self.assertFalse(maneuver._recovery_ready)
        held = list(maneuver.path)
        previous = float('inf')
        for i in range(len(held) - 12, len(held)):
            result = maneuver.plan(self.at(held, i), [], [])
            self.assertLessEqual(result.acc_target_velocity, previous)
            previous = result.acc_target_velocity
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertEqual(held, maneuver.path)

    def test_teleport_clears_old_commitment(self):
        self.start()
        result = self.process.run(VehicleState(200.0, 200.0, 0.0, 0.0), self.path, [])
        self.assertFalse(self.process.static_avoidance.active)
        self.assertFalse(result.active)

    def test_missing_reference_at_path_end_stops_instead_of_driving_blind(self):
        self.start()
        self.finish_leg(self.process.static_avoidance)
        held = self.process.static_avoidance.path
        result = self.process.run(self.at(held, len(held) - 1), [], [])
        self.assertEqual(0.0, result.acc_target_velocity)
        self.assertTrue(self.process.static_avoidance.active)

    def test_execute_releases_brake_but_red_signal_still_stops(self):
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller._configure_runtime = lambda **kwargs: None
        controller.path = self.path
        controller.path_manager = SimpleNamespace(
            get_local_path=lambda ego: (self.path[:50], 16.67), current_waypoint=0, is_closed_path=False)
        controller.forward_object_detector = SimpleNamespace(detect_object=lambda *args, **kwargs: self.entries)
        controller.lattice_planning_process = self.process
        controller.adaptive_cruise_control = AdaptiveCruiseControl(
            vehicle_length=4.635, velocity_gain=0.5, distance_gain=1.0, time_gap=0.8)
        signal_limit = [16.67]
        controller.traffic_light_stop_controller = SimpleNamespace(get_target_velocity=lambda **kwargs: signal_limit[0])
        controller.pure_pursuit = PurePursuit(0.65, 3.0, 2.0, 30.0)
        controller.pid = Pid(0.075, 0.0001, 0.025, 1 / 30)
        controller.control_dt = 1 / 30
        controller.previous_target_velocity = 0.0
        controller.previous_acc_cmd = -1.0
        controller.steering_sign = 1.0
        controller.max_steering = math.radians(30)
        for _ in range(10):
            command, path = controller.execute(self.ego, [], [])
        self.assertGreater(command.accel, 0.0)
        self.assertEqual(0.0, command.brake)
        self.assertIs(path, self.process.last_result.path)
        controller.path_manager.get_local_path = lambda ego: (self.path[:50], 0.0)
        command, _ = controller.execute(self.ego, [], [])
        self.assertEqual(1.0, command.brake)
        controller.path_manager.get_local_path = lambda ego: (self.path[:50], 16.67)
        controller.execute(self.ego, [], [])
        controller.path_manager.get_local_path = lambda ego: ([], 0.0)
        for _ in range(10):
            command, path = controller.execute(self.ego, [], [])
        self.assertGreater(command.accel, 0.0)
        self.assertEqual(0.0, command.brake)
        self.assertGreater(len(path), 2)
        signal_limit[0] = 0.0
        command, _ = controller.execute(self.ego, [], [])
        self.assertEqual(1.0, command.brake)
        self.assertEqual("signal", self.process.last_result.debug_details["static_avoidance"]["speed_source"])

    def test_valid_static_bypass_preserves_dynamic_and_signal_acc(self):
        result = self.start()
        controller = AutonomousDriving.__new__(AutonomousDriving)
        dynamic = {"object_info": ObjectInfo(12, -5, 0, 1)}
        person = {"object_info": ObjectInfo(12, -5, 0, 0, is_static=True)}
        signal = {"object_info": ObjectInfo(12, -5, 0, 3)}
        moving_static = {"object_info": ObjectInfo(12, -5, 2, 2, is_static=True)}
        others = [dynamic, person, signal, moving_static]
        kept = controller._legacy_acc_entries(self.ego, result, self.entries + others)
        self.assertEqual(others, kept)

    def test_blocked_static_path_keeps_legacy_acc(self):
        controller = AutonomousDriving.__new__(AutonomousDriving)
        controller.lattice_planning_process = self.process
        result = self.start()
        result.debug_details["static_avoidance"]["path_validated"] = False
        self.assertEqual(self.entries, controller._legacy_acc_entries(self.ego, result, self.entries))

    def test_false_static_acc_stop_releases_only_on_validated_path(self):
        result = self.start()
        checker = self.process.planner
        # Find a center inside the legacy 2.5 m corridor, but outside the swept OBB.
        candidate = next((ObjectInfo(7.0, y * 0.1, 0.0, 2, is_static=True, size_x=1.0, size_y=1.0)
                          for y in range(-100, 101)
                          if 2.1 < checker._distance_to_path(Point(7.0, y * 0.1), result.path) < 2.5
                          and not any(checker._obb_collision(point, checker._path_heading(result.path, i),
                              ObjectInfo(7.0, y * 0.1, 0, 2, size_x=1.0, size_y=1.0))
                              for i, point in enumerate(result.path))), None)
        self.assertIsNotNone(candidate)
        entry = {"object_info": candidate, "local_position": Point(7.0, candidate.position.y)}
        acc = AdaptiveCruiseControl(vehicle_length=4.635, velocity_gain=0.5, distance_gain=1.0, time_gap=0.8)
        acc.check_object(result.path, [entry], [])
        self.assertEqual(0.0, acc.get_target_velocity(0.0, 5.0))
        result = self.process.run(self.ego, self.path, [entry])
        self.assertTrue(result.debug_details["static_avoidance"]["path_validated"])
        controller = AutonomousDriving.__new__(AutonomousDriving)
        entries = controller._legacy_acc_entries(self.ego, result, [entry])
        acc.check_object(result.path, entries, [])
        target = min(acc.get_target_velocity(0, 5), result.acc_target_velocity)
        self.assertGreater(target, 0.5)
        self.assertGreater(Pid(0.075, 0.0001, 0.025, 1 / 30).get_output(target, 0), 0)

    def test_bicycle_drive_from_stop_avoids_and_returns(self):
        self._drive_and_rejoin()

    def test_long_leg_preferred_and_layer_preserved_while_following(self):
        self.entries = [{'object_info': ObjectInfo(30, 0, 0, 2, is_static=True, size_x=2, size_y=1)}]
        result = self.start()
        details = result.debug_details['static_avoidance']
        self.assertGreater(details['valid_short_candidates'], 0)
        self.assertGreater(details['valid_long_candidates'], 0)
        self.assertEqual('long', details['selected_layer'])
        self.assertAlmostEqual(35, result.path[-1].x)
        valid_long = [c for c in result.candidates if c.valid and c.layer == 'long']
        self.assertEqual(min(abs(c.offset) for c in valid_long), abs(result.selected_offset))
        held = list(result.path)
        result = self.process.run(self.at(held, 8), [], self.entries)
        selected = [c for c in result.candidates if c.points is result.path]
        self.assertEqual(1, len(selected))
        self.assertEqual('long', selected[0].layer)
        self.assertEqual(held, self.process.static_avoidance.path)
        result = self.finish_leg(self.process.static_avoidance)
        self.assertTrue(self.process.static_avoidance._recovery_ready)
        self.assertEqual(held, self.process.static_avoidance.path[:len(held)])

    def test_close_obstacle_uses_short_only_when_long_is_blocked(self):
        result = self.start()
        details = result.debug_details['static_avoidance']
        self.assertEqual(0, details['valid_long_candidates'])
        self.assertGreater(details['valid_short_candidates'], 0)
        self.assertEqual('short_fallback', details['selection_reason'])
        self.assertAlmostEqual(15, result.path[-1].x)
        self.assertTrue(self.process.static_avoidance._within_corridor(
            result.path, self.ego, parameters.STATIC_TRACKING_MARGIN))

    def test_recorded_obstacle_cluster_stops_with_reduced_width(self):
        data = json.loads(Path(__file__).with_name('static_scene_20260914.json').read_text())
        route = [Point(*xy) for xy in data['route']]
        objects = [ObjectInfo(**item) for item in data['objects']]
        result = self.process.static_avoidance.plan(VehicleState(*data['ego']), route, objects, route)
        self.assertFalse(result.active)
        self.assertEqual(0, result.acc_target_velocity)
        self.assertEqual(3, result.debug_details['static_avoidance']['allowed_right_offset'])
        self.assertEqual([], result.candidates)

    def test_measured_stop_line_gates_only_new_lateral_planning(self):
        process = LatticePlanningProcess(highway_zone_points=())
        x, y = parameters.STATIC_PLANNING_STOP_POINT
        route = [Point(x, y + i * .5) for i in range(-20, 120)]
        before = VehicleState(x, y - process.planner.vehicle_length - .2, math.pi / 2, 0)
        touching = VehicleState(x, y - process.planner.vehicle_length, math.pi / 2, 0)
        with patch.object(process.static_avoidance, 'plan', return_value='planned') as plan:
            result = process.run(before, route, self.entries, prediction_path=route)
            self.assertEqual('static_before_stop_line', result.reason)
            self.assertIs(route, result.path)
            self.assertFalse(result.acc_override)
            plan.assert_not_called()
            controller = AutonomousDriving.__new__(AutonomousDriving)
            controller.lattice_planning_process = process
            self.assertEqual(self.entries, controller._legacy_acc_entries(before, result, self.entries))
            self.assertEqual('front_bumper', result.debug_details['static_avoidance']['stop_line_reference'])
            self.assertEqual('planned', process._run_static_zone(touching, route, self.entries, route))
            plan.assert_called_once()
            plan.reset_mock()
            process.static_avoidance.path = [Point(x, y - 3), Point(x, y - 1)]
            self.assertEqual('planned', process._run_static_zone(before, route, self.entries, route))
            plan.assert_called_once()

    def test_bumper_contact_without_obstacles_keeps_reference_path(self):
        process = LatticePlanningProcess(highway_zone_points=())
        x, y = parameters.STATIC_PLANNING_STOP_POINT
        ego = VehicleState(x, y - process.planner.vehicle_length, math.pi / 2, 0)
        route = [Point(x, ego.position.y + i * .5) for i in range(100)]
        result = process.run(ego, route, [], prediction_path=route)
        self.assertEqual('no_static_obstacle', result.reason)
        self.assertIs(route, result.path)
        self.assertFalse(result.active)

    def test_after_stop_line_wrong_direction_does_not_start_avoidance(self):
        process = LatticePlanningProcess(highway_zone_points=())
        x, y = parameters.STATIC_PLANNING_STOP_POINT
        with patch.object(process.static_avoidance, 'plan') as plan:
            result = process.run(VehicleState(x, y + 1, -math.pi / 2, 0), self.path, self.entries)
            self.assertEqual('static_wrong_direction', result.reason)
            plan.assert_not_called()

    def test_bicycle_drive_through_grid_aligned_gap(self):
        self.entries.extend([
            {"object_info": ObjectInfo(24.0, 4.0, 0.0, 2, is_static=True, size_x=10.0, size_y=9.0)},
            {"object_info": ObjectInfo(24.0, -8.0, 0.0, 2, is_static=True, size_x=10.0, size_y=5.0)},
        ])
        self._drive_and_rejoin()

    def _drive_and_rejoin(self):
        self.start()
        ego = self.ego
        pursuit = PurePursuit(0.65, 3.0, 2.0, 30.0)
        saw_outside = False
        for _ in range(1500):
            local = self.path[min(150, int(max(0, ego.position.x) * 2)):]
            result = self.process.run(ego, local, self.entries)
            if result.reason == "static_rejoined":
                self.assertTrue(saw_outside)
                self.assertLess(abs(ego.position.y), 0.6)
                break
            self.assertGreater(result.acc_target_velocity, 0.0, result.reason)
            pursuit.path, pursuit.vehicle_state = result.path, ego
            steer = pursuit.calculate_steering_angle(lookahead_limits=parameters.STATIC_TRACKING_LOOKAHEAD)
            speed = max(0.0, ego.velocity + max(-3.0, min(1.5, (result.acc_target_velocity - ego.velocity) * 2.0)) * 0.05)
            yaw = ego.yaw + speed / 3.0 * math.tan(steer) * 0.05
            ego = VehicleState(ego.position.x + speed * math.cos(ego.yaw) * 0.05,
                               ego.position.y + speed * math.sin(ego.yaw) * 0.05, yaw, speed)
            for entry in self.entries:
                self.assertFalse(self.process.planner._obb_collision(ego.position, ego.yaw, entry["object_info"]),
                                 "ego={} object={}".format(ego, entry["object_info"]))
            saw_outside |= not self.process._is_inside_static_zone(ego)
        else:
            self.fail("Closed-loop avoidance did not finish")


if __name__ == "__main__":
    unittest.main()
