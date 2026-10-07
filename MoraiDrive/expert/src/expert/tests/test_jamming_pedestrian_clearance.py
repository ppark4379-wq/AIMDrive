"""A departing tunnel pedestrian must release on fresh body clearance."""
import math
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.vehicle_state import VehicleState
from expert.src.planning.adaptive_cruise_control import AdaptiveCruiseControl
from expert.src.planning.Jamming.pedestrian_safety import pedestrian_conflict


class TunnelPedestrianClearanceTests(unittest.TestCase):
    def setUp(self):
        self.path = [Point(float(x), 0.) for x in range(81)]
        self.ego = VehicleState(0., 0., 0., 0.)
        self.acc = AdaptiveCruiseControl(.5, 1., .8, 4.635)

    @staticmethod
    def person(x, y, speed=2., heading=90., uid=1):
        return ObjectInfo(x, y, speed, 0, size_x=.8, size_y=.6,
                          heading_deg=heading, unique_id=uid)

    def check(self, *people, tunnel=True, signal=None):
        entries = [dict(object_info=person,
                        local_position=Point(*person.position)) for person in people]
        kwargs = {}
        if tunnel:
            kwargs['pedestrian_conflict_check'] = lambda person: pedestrian_conflict(
                self.path, self.ego, person, 5.) is not None
        self.acc.check_object(self.path, entries, signal, **kwargs)
        return self.acc.get_target_velocity(self.ego.velocity, 5.)

    def test_body_overlap_then_departing_pedestrian_releases_same_tick(self):
        person = self.person(3., 0.)
        self.assertEqual(self.check(person), 0.)
        person.position = Point(3., 2.)
        self.assertIsNone(pedestrian_conflict(self.path, self.ego, person, 5.))
        self.assertEqual(self.check(person), 5.)
        self.assertTrue(self.acc.pedestrian_released)
        for _ in range(20):
            self.assertEqual(self.check(person), 5.)

    def test_approaching_and_recrossing_pedestrians_still_conflict(self):
        person = self.person(4., 4., heading=-90.)
        collision = pedestrian_conflict(self.path, self.ego, person, 5.)
        self.assertIsNotNone(collision)
        self.assertGreater(collision['time'], 0.)
        self.assertEqual(self.check(person), 0.)
        person.heading_deg = 90.
        self.assertEqual(self.check(person), 5.)
        person.heading_deg = -90.
        self.assertEqual(self.check(person), 0.)

    def test_stationary_person_on_path_blocks_but_person_beside_path_does_not(self):
        person = self.person(5., 0., speed=0.)
        self.assertIsNotNone(pedestrian_conflict(self.path, self.ego, person, 5.))
        self.assertEqual(self.check(person), 0.)
        person.position = Point(5., 2.5)
        self.assertEqual(self.check(person), 5.)

    def test_current_overlap_and_second_person_prevent_false_release(self):
        departed = self.person(3., 3.)
        for person in (self.person(3., 0., speed=0., uid=2),
                       self.person(4., -4., uid=2)):
            self.assertEqual(self.check(departed, person), 0.)
        self.assertEqual(self.check(departed), 5.)

    def test_collision_check_includes_people_behind_current_forward_gate(self):
        self.ego.velocity = 0.
        person = self.person(-1., 0., speed=3., heading=0.)
        collision = pedestrian_conflict(self.path, self.ego, person, 5.)
        self.assertIsNotNone(collision)
        self.assertLess(collision['time'], 1.)

    def test_red_signal_and_default_crosswalk_policy_are_not_bypassed(self):
        departed = self.person(3., 3.)
        self.assertEqual(self.check(departed, tunnel=False), 0.)
        self.assertEqual(self.check(departed), 5.)
        signal = ObjectInfo(3., 0., 0., 3, name='red')
        self.assertEqual(self.check(departed, signal, signal=('red', 1)), 0.)

    def test_missing_or_invalid_prediction_never_claims_clearance(self):
        person = self.person(3., 3.)
        for path in ([], [Point(0., 0.)], [Point(0., 0.), Point(float('nan'), 0.)]):
            self.assertIsNotNone(pedestrian_conflict(path, self.ego, person, 5.))
        person.velocity = float('nan')
        self.assertIsNotNone(pedestrian_conflict(self.path, self.ego, person, 5.))

    def test_world_rotation_keeps_same_clearance(self):
        person = self.person(3., 2.)
        yaw = math.radians(-145.)
        rotate = lambda p: Point(31.+p.x*math.cos(yaw)-p.y*math.sin(yaw),
                                 -486.+p.x*math.sin(yaw)+p.y*math.cos(yaw))
        self.path = [rotate(p) for p in self.path]
        self.ego = VehicleState(31., -486., yaw, 0.)
        person.position = rotate(person.position)
        person.heading_deg += math.degrees(yaw)
        self.assertIsNone(pedestrian_conflict(self.path, self.ego, person, 5.))


class TunnelPedestrianControllerTests(unittest.TestCase):
    def setUp(self):
        from expert.src.autonomous_driving import AutonomousDriving
        from expert.src.planning.Jamming.recorded_planner import JammingPlanner
        root = Path(__file__).resolve().parents[3]
        self.route = [Point(*row[:2]) for row in np.loadtxt(
            root/'map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt')]
        self.planner = JammingPlanner(background=False)
        self.addCleanup(self.planner.close)
        coordinates = np.asarray(self.planner._path(self.route))
        index = int(np.argmin(np.linalg.norm(coordinates-np.array([-.6, 8.]), axis=1)))
        start, end = self.route[index:index+2]
        self.ego = VehicleState(*start, math.atan2(end.y-start.y, end.x-start.x), 0.)
        self.reference = self.route[index:index+80]
        self.local = self.planner._ego(self.ego)
        with contextlib.redirect_stdout(io.StringIO()):
            self.controller = AutonomousDriving()
        self.controller.lattice_planning_process.recorded_jamming.close()
        self.controller.lattice_planning_process.recorded_jamming = self.planner
        self.addCleanup(self.controller.lattice_planning_process.recorded_static.executor.shutdown, True)
        self.controller.path = self.route
        self.controller.path_manager = SimpleNamespace(
            get_local_path=lambda ego: (self.reference, 8.), current_waypoint=index,
            is_closed_path=False)
        # Begin with the real corridor's validated baseline geometry. This is
        # the held path during the user's repeated crossing/clearance scene.
        self.planner.local.plan(self.local, self.planner._path(self.reference), [],
                                self.planner._path(self.route))

    def person(self, forward=3., lateral=0., speed=2.):
        section = self.planner.section
        return ObjectInfo(*section.to_world((self.local.position.x+lateral,
                                             self.local.position.y+forward)),
                          speed, 0, size_x=.8, size_y=.6,
                          heading_deg=math.degrees(-section.rotation), unique_id=101)

    def tick(self, people):
        with contextlib.redirect_stdout(io.StringIO()):
            command, _ = self.controller.execute(self.ego, people, None, control_dt=1./30.)
        result = self.controller.lattice_planning_process.last_result
        return command, result

    def test_repeated_pedestrian_stop_releases_without_negative_speed_debt(self):
        self.controller.previous_target_velocity = -2.58
        for cycle in range(15):
            with self.subTest(cycle=cycle):
                person = self.person()
                command, result = self.tick([person])
                self.assertEqual(result.debug_details['static_avoidance']['final_target_velocity'], 0.)
                self.assertEqual(command.brake, 1.)
                self.assertTrue(result.debug_details['jamming_pedestrian']['emergency'])
                # Reproduce the negative historical ACC target from the log
                # before the first release; subsequent cycles use live state.
                if cycle == 0:
                    self.controller.previous_target_velocity = -2.58
                person.position = self.person(lateral=4.).position
                for tick in range(10):
                    command, result = self.tick([person])
                    if tick < 3:
                        self.assertGreater(result.debug_details['static_avoidance']['final_target_velocity'], 0.)
                    self.assertEqual(result.debug_details['jamming_pedestrian']['conflicts'], 0)
                self.assertEqual(command.brake, 0.)
                self.assertGreater(command.accel, 0.)
                self.assertNotEqual(self.controller.adaptive_cruise_control.object_type, 0)

    def test_stationary_person_behind_forward_gate_is_still_checked(self):
        person = self.person(forward=-.2, speed=0.)
        command, result = self.tick([person])
        forward = self.controller.forward_object_detector.detect_object(self.ego)
        self.assertEqual(forward, [])
        self.assertTrue(result.debug_details['jamming_pedestrian']['emergency'])
        self.assertEqual(result.debug_details['static_avoidance']['final_target_velocity'], 0.)
        self.assertEqual(command.brake, 1.)
        person.position = self.person(forward=-.2, lateral=4., speed=0.).position
        for _ in range(10):
            command, result = self.tick([person])
        self.assertEqual(result.debug_details['jamming_pedestrian']['conflicts'], 0)
        self.assertGreater(result.debug_details['static_avoidance']['final_target_velocity'], 0.)
        self.assertGreater(command.accel, 0.)
        self.assertEqual(command.brake, 0.)

    def test_departed_head_on_person_releases_old_recorded_tracking_stop(self):
        owner = self.planner.local
        owner.reset()
        entry = VehicleState(40.17, -479.75, math.radians(-143.1), 0.)
        # Load the exact recording named in the old log as a held maneuver,
        # rather than allowing today's corrected lane selection to replace it.
        owner._setup(self.planner._path(self.route), self.planner._ego(entry))
        start = float(owner.records[0][1][0, 1])
        stations = np.linspace(start, owner.goal_y, int((owner.goal_y-start)/.1)+1)
        recorded = np.column_stack((owner.curves[0](stations), stations))
        held = owner._smooth_return(owner._smooth_recorded_path(recorded, []), [])
        owner.path = [Point(*point) for point in held]
        owner.selected_names = ['jamming_01.csv']
        self.assertEqual(owner.selected_names, ['jamming_01.csv'])
        self.ego = VehicleState(24.10, -491.25, math.radians(-148.), 0.)
        self.local = self.planner._ego(self.ego)
        index = min(range(len(self.route)), key=lambda i: self.route[i].distance(self.ego.position))
        self.reference = self.route[index:index+80]
        blocked = owner.plan(self.local, self.planner._path(self.reference), [])
        details = blocked.debug_details['static_avoidance']
        self.assertEqual(details['error'], 'tracking_error')
        self.assertAlmostEqual(details['tracking_error_m'], 1.037, delta=.08)
        self.assertEqual(blocked.acc_target_velocity, 0.)
        # The attachment begins after the actor disappeared. Preserve the
        # preceding moving-traffic episode, without inventing stale GT.
        self.planner.dynamic.dynamic_seen = True
        self.planner._last_position = self.local.position.copy()
        self.controller.previous_target_velocity = 0.
        self.controller.previous_acc_cmd = -1.
        for _ in range(10):
            command, result = self.tick([])
        repaired = result.debug_details['static_avoidance']
        self.assertTrue(repaired['path_validated'], repaired)
        self.assertIsNone(repaired['error'])
        self.assertGreater(repaired['final_target_velocity'], 0.)
        self.assertEqual(result.debug_details['jamming_pedestrian']['checked'], 0)
        self.assertIsNone(result.debug_details['jamming_dynamic']['collision'])
        self.assertGreater(command.accel, 0.)
        self.assertEqual(command.brake, 0.)


if __name__ == '__main__':
    unittest.main()
